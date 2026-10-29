from __future__ import annotations

import cv2
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock

from app import ingestion_pipeline
from app.db import get_db
from app.main import app
from app.models import Base, Document, Page
from app.table_reconstruction import reconstruct_tables
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def _token(text: str, box: list[int], line: int, word: int) -> dict[str, object]:
    return {
        "token": text,
        "absolute_bbox": box,
        "bbox": [round(value * 1000 / 400) for value in box],
        "confidence": 91.0,
        "block_num": 1,
        "paragraph_num": 1,
        "line_num": line,
        "word_num": word,
    }


def test_reconstructs_ruled_table_cells_and_column_widths(tmp_path) -> None:
    image = np.full((300, 400, 3), 255, dtype=np.uint8)
    for x in (50, 200, 350):
        cv2.line(image, (x, 50), (x, 260), (0, 0, 0), 2)
    for y in (50, 120, 190, 260):
        cv2.line(image, (50, y), (350, y), (0, 0, 0), 2)
    image_path = tmp_path / "ruled-table.png"
    assert cv2.imwrite(str(image_path), image)
    tokens = [
        _token("Table 1", [170, 22, 230, 42], 0, 1),
        _token("Course", [70, 70, 140, 90], 1, 1),
        _token("Units", [220, 70, 280, 90], 1, 2),
        _token("Math", [70, 140, 130, 160], 2, 1),
        _token("3", [220, 140, 230, 160], 2, 2),
        _token("Science", [70, 210, 140, 230], 3, 1),
        _token("4", [220, 210, 230, 230], 3, 2),
    ]

    result = reconstruct_tables(image_path, tokens, page_id=7, page_number=2)

    assert (result["page_width"], result["page_height"]) == (400, 300)
    assert len(result["tables"]) == 1
    table = result["tables"][0]
    assert table["style"]["layout_source"] == "detected_rules"
    assert [len(row["cells"]) for row in table["rows"]] == [2, 2, 2]
    assert [cell["text"] for cell in table["rows"][0]["cells"]] == ["Course", "Units"]
    assert table["rows"][0]["cells"][0]["original_text"] == "Course"
    assert table["rows"][0]["cells"][0]["bbox"][0] < table["rows"][0]["cells"][1]["bbox"][0]
    assert len(table["columns"]) == 2
    assert table["rows"][0]["cells"][0]["is_header"] is True
    assert table["caption"] == "Table 1"


def test_ignores_page_sized_ruled_frame(tmp_path) -> None:
    image = np.full((800, 600, 3), 255, dtype=np.uint8)
    for x in (25, 300, 575):
        cv2.line(image, (x, 25), (x, 775), (0, 0, 0), 2)
    for y in (25, 275, 525, 775):
        cv2.line(image, (25, y), (575, y), (0, 0, 0), 2)
    image_path = tmp_path / "page-frame.png"
    assert cv2.imwrite(str(image_path), image)
    tokens = [
        _token("Section", [50, 50, 125, 72], 1, 1),
        _token("Prose", [325, 50, 390, 72], 1, 2),
        _token("Details", [50, 325, 120, 347], 2, 1),
        _token("More", [325, 325, 380, 347], 2, 2),
    ]

    result = reconstruct_tables(image_path, tokens, page_id=4, page_number=1)

    assert result["tables"] == []


def test_reconstructs_borderless_aligned_columns_as_uncertain(tmp_path) -> None:
    image_path = tmp_path / "plain-page.png"
    assert cv2.imwrite(str(image_path), np.full((300, 400, 3), 255, dtype=np.uint8))
    tokens = [
        _token("Name", [30, 80, 75, 100], 1, 1),
        _token("Score", [200, 80, 250, 100], 1, 2),
        _token("Ari", [30, 125, 55, 145], 2, 1),
        _token("95", [200, 125, 225, 145], 2, 2),
        _token("Bo", [30, 170, 50, 190], 3, 1),
        _token("88", [200, 170, 225, 190], 3, 2),
    ]

    result = reconstruct_tables(image_path, tokens, page_id=3, page_number=1)

    assert len(result["tables"]) == 1
    table = result["tables"][0]
    assert table["style"]["layout_source"] == "ocr_alignment"
    assert table["rows"][1]["cells"][0]["text"] == "Ari"
    assert table["rows"][1]["cells"][1]["text"] == "95"
    assert table["rows"][1]["cells"][0]["uncertain"] is True
    assert "merged_cells" in table["uncertain_fields"]


def test_page_ocr_data_is_cached_for_unchanged_image(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "cached-page.png"
    image_path.write_bytes(b"page image placeholder")
    page = SimpleNamespace(id=987654, page_number=9)
    tokens = [{"token": "cached"}]
    layout = {"page_width": 400, "page_height": 300, "tables": []}
    extract = Mock(return_value=tokens)
    reconstruct = Mock(return_value=layout)
    monkeypatch.setattr(ingestion_pipeline, "extract_tokens_with_bboxes", extract)
    monkeypatch.setattr(ingestion_pipeline, "reconstruct_tables", reconstruct)

    first = ingestion_pipeline._get_page_ocr_data(page, image_path)
    second = ingestion_pipeline._get_page_ocr_data(page, image_path)

    assert first == second == (tokens, layout)
    extract.assert_called_once_with(image_path)
    reconstruct.assert_called_once_with(image_path, tokens, page.id, page.page_number)


def test_document_ocr_endpoint_returns_table_layout(tmp_path, monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    image_path = tmp_path / "ocr-page.png"
    assert cv2.imwrite(str(image_path), np.full((300, 400, 3), 255, dtype=np.uint8))
    document = Document(name="table.pdf", format="pdf", owner_id="local-development-user", page_count=1, status="PROCESSED")
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, image_path=str(image_path))
    session.add(page)
    session.commit()
    expected_table = {"id": "page-1-table-1", "rows": [], "columns": [], "style": {"layout_source": "detected_rules"}}

    def override_get_db():
        yield session

    monkeypatch.setattr(ingestion_pipeline, "extract_tokens_with_bboxes", lambda _: [])
    monkeypatch.setattr(
        ingestion_pipeline,
        "reconstruct_tables",
        lambda *_: {"page_width": 400, "page_height": 300, "tables": [expected_table]},
    )
    app.dependency_overrides[get_db] = override_get_db
    try:
        response = TestClient(app).get(f"/api/v1/documents/{document.id}/ocr")
        assert response.status_code == 200
        page_payload = response.json()["pages"][0]
        assert (page_payload["page_width"], page_payload["page_height"]) == (400, 300)
        assert page_payload["tables"] == [expected_table]
    finally:
        app.dependency_overrides.clear()
        session.close()
        engine.dispose()