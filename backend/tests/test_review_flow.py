from __future__ import annotations

from collections.abc import Callable, Generator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import ingestion_pipeline
from app.db import get_db
from app.claude_extractor import (
    DocumentExtraction,
    ExtractedAttribute,
    ExtractedTable,
    ExtractionCanceled,
    ClaudeExtractionError,
)
from app.main import app
from app.models import Attribute, Base, Correction, Document, Page


def _build_session_factory() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def _override_get_db(session: Session) -> Callable[[], Generator[Session, None, None]]:
    def _dependency() -> Generator[Session, None, None]:
        try:
            yield session
        finally:
            pass

    return _dependency


def _use_fake_claude(
    monkeypatch: pytest.MonkeyPatch,
    extraction: DocumentExtraction | None = None,
    error: BaseException | None = None,
    calls: list[Any] | None = None,
) -> None:
    """Replace the Claude service so tests never touch the network or need real page images."""

    class FakeClaudeService:
        def extract_document(self, pages: list[tuple[int, Path]], should_cancel: Callable[[], bool] | None = None) -> DocumentExtraction:
            if calls is not None:
                calls.append(list(pages))
            if error is not None:
                raise error
            return extraction if extraction is not None else DocumentExtraction()

    monkeypatch.setattr(ingestion_pipeline, "ClaudeExtractionService", FakeClaudeService)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _seed_document(factory: sessionmaker[Session], **fields: Any) -> int:
    session = factory()
    values: dict[str, Any] = {"name": "doc.pdf", "format": "pdf", "owner_id": "local-development-user", "page_count": 0, "status": "QUEUED"}
    values.update(fields)
    document = Document(**values)
    session.add(document)
    session.commit()
    document_id = document.id
    session.close()
    return document_id


def test_document_status_endpoint_validates_transitions_and_prerequisites() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(name="status.pdf", format="pdf", owner_id="local-development-user", page_count=0, status="QUEUED")
    session.add(document)
    session.commit()
    document_id = document.id

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)
    try:
        assert client.patch(f"/api/v1/documents/{document_id}/status?status=unknown").status_code == 400
        assert client.patch(f"/api/v1/documents/{document_id}/status?status=PROCESSED").status_code == 400
        assert client.patch(f"/api/v1/documents/{document_id}/status?status=PROCESSING").status_code == 400
        unchanged_document = session.get(Document, document_id)
        assert unchanged_document is not None
        assert unchanged_document.status == "QUEUED"
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_document_status_endpoint_persists_canceled_state() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(name="cancel.pdf", format="pdf", owner_id="local-development-user", status="QUEUED")
    session.add(document)
    session.commit()
    document_id = document.id

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)
    try:
        response = client.patch(f"/api/v1/documents/{document_id}/status?status=CANCELED")
        assert response.status_code == 200
        assert response.json()["status"] == "CANCELED"
        canceled_document = session.get(Document, document_id)
        assert canceled_document is not None
        assert canceled_document.status == "CANCELED"
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_processing_stages_attributes_without_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _build_session_factory()
    seed_session = factory()
    document = Document(name="sample.pdf", format="pdf", owner_id="local-development-user", page_count=0, status="QUEUED")
    seed_session.add(document)
    seed_session.commit()
    document_id = document.id
    seed_session.close()

    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\n%%EOF")
    rasterized_page = tmp_path / "page-1.png"
    rasterized_page.write_bytes(b"fake-png")

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)

    def fake_rasterize(path: Path, doc_id: str) -> list[Path]:
        return [rasterized_page]

    claude_calls: list[Any] = []
    _use_fake_claude(
        monkeypatch,
        DocumentExtraction(
            document_type="Sales Invoice",
            page_text={1: "Total Amount 123.45"},
            attributes=[ExtractedAttribute("Total Amount", "123.45", 0.91, 1, box=[10, 20, 30, 40])],
            tables=[ExtractedTable.from_rows(1, [["Item", "Price"], ["Pen", "123.45"]], title="Items", box=[0, 100, 500, 300])],
        ),
        calls=claude_calls,
    )
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", fake_rasterize)

    result = ingestion_pipeline.process_document_job(document_id, str(source_path), "sample.pdf", "local-development-user")
    assert result["status"] == "processed"

    verify_session = factory()
    try:
        saved_attributes = verify_session.query(Attribute).filter(Attribute.document_id == document_id).all()
        assert saved_attributes == []

        refreshed_document = verify_session.get(Document, document_id)
        assert refreshed_document is not None
        assert refreshed_document.status == "PROCESSED"
        assert refreshed_document.processing_started_at is not None
        assert refreshed_document.processing_finished_at is not None
        assert refreshed_document.processing_duration_seconds is not None
        assert refreshed_document.processing_duration_seconds >= 0
        assert refreshed_document.document_type == "sales_invoice"
        assert refreshed_document.processing_error is None
        assert refreshed_document.extracted_attributes
        staged = refreshed_document.extracted_attributes[0]
        assert staged["entity_type_label"] == "Total Amount"
        assert staged["extracted_value"] == "123.45"
        assert staged["bounding_boxes"] == [[10, 20, 30, 40]]
        assert staged["page_number"] == 1
        assert staged["validation_status"] == "APPROVED"
        assert staged["saved"] is False
        assert staged["continues_from_previous_page"] is False

        page = verify_session.query(Page).filter(Page.document_id == document_id).one()
        assert page.ocr_text == "Total Amount 123.45"
        assert page.ocr_layout[0]["title"] == "Items"
        assert page.ocr_layout[0]["page_id"] == page.id
        assert staged["page_id"] == page.id
        assert claude_calls == [[(1, rasterized_page)]]
    finally:
        verify_session.close()


def test_processing_worker_downloads_source_storage_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _build_session_factory()
    seed_session = factory()
    document = Document(name="remote.pdf", format="pdf", owner_id="local-development-user", page_count=0, status="QUEUED")
    seed_session.add(document)
    seed_session.commit()
    document_id = document.id
    seed_session.close()

    rasterized_page = tmp_path / "page-001.png"
    rasterized_page.write_bytes(b"fake-png")
    downloaded_keys: list[str] = []

    def fake_download(key: str) -> bytes:
        downloaded_keys.append(key)
        return b"downloaded-pdf"

    def fake_rasterize(path: Path, doc_id: str) -> list[Path]:
        assert path.read_bytes() == b"downloaded-pdf"
        return [rasterized_page]

    def storage_is_enabled() -> bool:
        return True

    def ignore_upload(path: Path, object_key: str) -> None:
        return None

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)
    monkeypatch.setattr(ingestion_pipeline, "storage_enabled", storage_is_enabled)
    monkeypatch.setattr(ingestion_pipeline, "download_file", fake_download)
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", fake_rasterize)
    monkeypatch.setattr(ingestion_pipeline, "upload_file", ignore_upload)
    _use_fake_claude(monkeypatch)

    result = ingestion_pipeline.process_document_job(
        document_id,
        "owner/document/source.pdf",
        "remote.pdf",
        "local-development-user",
    )

    assert result["status"] == "processed"
    assert downloaded_keys == ["owner/document/source.pdf"]


def test_reprocessing_failure_preserves_previous_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _build_session_factory()
    old_page_path = tmp_path / "old-page.png"
    old_page_path.write_bytes(b"old-page")
    new_page_path = tmp_path / "new-page.png"
    new_page_path.write_bytes(b"new-page")
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"source")

    seed_session = factory()
    document = Document(
        name="reprocess.pdf",
        format="pdf",
        owner_id="local-development-user",
        page_count=1,
        status="PROCESSED",
        processing_finished_at=datetime.now(timezone.utc),
        processing_duration_seconds=1.0,
    )
    seed_session.add(document)
    seed_session.flush()
    old_page = Page(document_id=document.id, page_number=1, image_path=str(old_page_path))
    seed_session.add(old_page)
    seed_session.commit()
    document_id = document.id
    old_page_id = old_page.id
    seed_session.close()

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)
    def fake_rasterize(path: Path, artifact_id: str) -> list[Path]:
        return [new_page_path]

    def ignore_upload(path: Path, object_key: str) -> None:
        return None

    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", fake_rasterize)
    monkeypatch.setattr(ingestion_pipeline, "upload_file", ignore_upload)
    _use_fake_claude(monkeypatch, error=RuntimeError("model failed"))

    with pytest.raises(RuntimeError, match="model failed"):
        ingestion_pipeline.process_document_job(document_id, str(source_path), "reprocess.pdf", "local-development-user")

    verify_session = factory()
    try:
        refreshed = verify_session.get(Document, document_id)
        assert refreshed is not None
        assert refreshed.status == "PROCESSED"
        assert refreshed.page_count == 1
        assert refreshed.processing_error is None
        assert verify_session.get(Page, old_page_id) is not None
        assert verify_session.query(Page).filter(Page.document_id == document_id).count() == 1
        assert old_page_path.read_bytes() == b"old-page"
        assert not new_page_path.exists()
    finally:
        verify_session.close()


def test_docx_processing_uses_page_aware_pdf_rasterization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "multi-page.docx"
    source_path.write_bytes(b"not a real docx")
    rasterized_pages = [tmp_path / "page-001.png", tmp_path / "page-002.png"]
    convert_calls: list[tuple[Path, Path]] = []

    def fake_convert(path: Path, output_dir: Path) -> None:
        convert_calls.append((path, output_dir))
        (output_dir / "multi-page.pdf").write_bytes(b"fake-pdf")

    monkeypatch.setattr(ingestion_pipeline, "_convert_with_libreoffice", fake_convert)
    def fake_rasterize(path: Path, document_id: str) -> list[Path]:
        return rasterized_pages

    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", fake_rasterize)

    result = ingestion_pipeline.rasterize_office_document(source_path, "document-1")

    assert result == rasterized_pages
    assert len(convert_calls) == 1
    assert convert_calls[0][0] == source_path


def test_docx_conversion_failure_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "unrenderable.docx"
    source_path.write_bytes(b"not a real docx")

    def fail_conversion(path: Path, output_dir: Path) -> None:
        raise RuntimeError("LibreOffice unavailable")

    monkeypatch.setattr(ingestion_pipeline, "_convert_with_libreoffice", fail_conversion)

    with pytest.raises(RuntimeError, match="Failed to rasterize office document"):
        ingestion_pipeline.rasterize_office_document(source_path, "document-1")


def test_processing_missing_source_marks_document_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _build_session_factory()
    seed_session = factory()
    document = Document(name="missing.pdf", format="pdf", owner_id="local-development-user", page_count=0, status="QUEUED")
    seed_session.add(document)
    seed_session.commit()
    document_id = document.id
    seed_session.close()

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)

    with pytest.raises(FileNotFoundError, match="Source file missing"):
        ingestion_pipeline.process_document_job(
            document_id,
            str(tmp_path / "missing.pdf"),
            "missing.pdf",
            "local-development-user",
        )

    verify_session = factory()
    try:
        refreshed_document = verify_session.get(Document, document_id)
        assert refreshed_document is not None
        assert refreshed_document.status == "FAILED"
        assert refreshed_document.processing_started_at is not None
        assert refreshed_document.processing_finished_at is not None
        assert refreshed_document.processing_duration_seconds is not None
        assert refreshed_document.processing_duration_seconds >= 0
    finally:
        verify_session.close()


def test_save_selected_persists_only_requested_attributes_and_manual_entries() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(
        name="review.pdf",
        owner_id="local-development-user",
        format="pdf",
        page_count=1,
        status="PROCESSED",
        extracted_attributes=[
            {
                "temp_id": "model-1",
                "page_id": 1,
                "page_number": 1,
                "document_id": 1,
                "entity_type_label": "TOTAL",
                "extracted_value": "123.45",
                "original_prediction": "123.45",
                "bounding_boxes": [[10, 10, 20, 20]],
                "confidence_score": 0.93,
                "validation_status": "APPROVED",
                "source": "model",
                "saved": False,
            }
        ],
    )
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(page)
    session.flush()
    document.extracted_attributes = [
        {
            **document.extracted_attributes[0],
            "page_id": page.id,
        }
    ]
    session.commit()

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)

    try:
        response = client.post(
            f"/api/v1/documents/{document.id}/attributes/save-selected",
            json={
                "document_id": document.id,
                "attributes": [
                    {
                        "temp_id": "model-1",
                        "source": "model",
                        "entity_type_label": "TOTAL",
                        "extracted_value": "150.00",
                        "page_number": 1,
                        "page_id": page.id,
                        "bounding_boxes": [[10, 10, 20, 20]],
                        "confidence_score": 0.93,
                        "validation_status": "APPROVED",
                    },
                    {
                        "temp_id": "manual-1",
                        "source": "manual",
                        "entity_type_label": "NOTES",
                        "extracted_value": "Customer requested paper copy",
                        "page_number": 1,
                        "bounding_boxes": [],
                        "confidence_score": 0.0,
                        "validation_status": "PENDING",
                    },
                ],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["saved_count"] == 2

        saved_attributes = session.query(Attribute).order_by(Attribute.id).all()
        assert len(saved_attributes) == 2
        assert {attribute.source for attribute in saved_attributes} == {"model", "manual"}
        saved_document = session.get(Document, document.id)
        assert saved_document is not None
        assert saved_document.extracted_attributes == []
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_corrections_still_apply_to_saved_attributes() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(name="corrected.pdf", format="pdf", owner_id="local-development-user", page_count=1, status="PROCESSED")
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(page)
    session.flush()

    attribute = Attribute(
        page_id=page.id,
        document_id=document.id,
        entity_type_label="TOTAL",
        extracted_value="123.45",
        bounding_boxes=[[0, 0, 10, 10]],
        confidence_score=0.88,
        validation_status="APPROVED",
        source="manual",
    )
    session.add(attribute)
    session.commit()

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/attributes/correct",
            json={
                "document_id": document.id,
                "corrections": [
                    {
                        "attribute_id": attribute.id,
                        "original_prediction": "123.45",
                        "corrected_value": "150.00",
                        "validation_status": "APPROVED",
                        "operator_user": "tester",
                        "corrected_label": "TOTAL",
                    }
                ],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["correction_count"] == 1

        refreshed_attribute = session.get(Attribute, attribute.id)
        assert refreshed_attribute is not None
        assert refreshed_attribute.extracted_value == "150.00"
        assert refreshed_attribute.source == "manual"
        corrections = session.query(Correction).all()
        assert len(corrections) == 1
        assert corrections[0].operator_user == "local-development-user"
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_saved_attributes_are_searchable_but_staged_ones_are_not() -> None:
    factory = _build_session_factory()
    session = factory()
    staged_document = Document(
        owner_id="local-development-user",
        name="staged.pdf",
        format="pdf",
        document_type="invoice",
        page_count=1,
        status="PROCESSED",
        extracted_attributes=[
            {
                "temp_id": "model-1",
                "page_id": 1,
                "page_number": 1,
                "document_id": 1,
                "entity_type_label": "UNSAVED_TOTAL",
                "extracted_value": "999.99",
                "original_prediction": "999.99",
                "bounding_boxes": [[1, 1, 2, 2]],
                "confidence_score": 0.91,
                "validation_status": "APPROVED",
                "source": "model",
                "saved": False,
            }
        ],
    )
    session.add(staged_document)
    session.flush()
    staged_page = Page(document_id=staged_document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(staged_page)
    session.flush()
    session.add(
        Attribute(
            page_id=staged_page.id,
            document_id=staged_document.id,
            entity_type_label="UNSAVED_TOTAL",
            extracted_value="999.99",
            bounding_boxes=[[1, 1, 2, 2]],
            confidence_score=0.91,
            validation_status="PENDING",
            source="model",
        )
    )
    staged_document.extracted_attributes = [
        {
            **staged_document.extracted_attributes[0],
            "page_id": staged_page.id,
        }
    ]

    saved_document = Document(name="saved.pdf", format="pdf", owner_id="local-development-user", document_type="invoice", page_count=1, status="PROCESSED")
    session.add(saved_document)
    session.flush()
    saved_page = Page(document_id=saved_document.id, page_number=1, image_path="datasets/raw-pages/2/page-001.png")
    session.add(saved_page)
    session.flush()
    session.add(
        Attribute(
            page_id=saved_page.id,
            document_id=saved_document.id,
            entity_type_label="SAVED_TOTAL",
            extracted_value="111.11",
            bounding_boxes=[[0, 0, 10, 10]],
            confidence_score=0.95,
            validation_status="APPROVED",
            source="model",
        )
    )
    session.commit()

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)

    try:
        staged_response = client.get("/api/v1/search/attributes", params={"q": "UNSAVED_TOTAL"})
        assert staged_response.status_code == 200
        assert staged_response.json()["total"] == 0

        staged_document_response = client.get("/api/v1/search/documents", params={"q": "UNSAVED_TOTAL"})
        assert staged_document_response.status_code == 200
        assert staged_document_response.json()["total"] == 1

        saved_response = client.get("/api/v1/search/attributes", params={"q": "SAVED_TOTAL"})
        assert saved_response.status_code == 200
        assert saved_response.json()["total"] >= 1

        saved_document_response = client.get("/api/v1/search/documents", params={"q": "111.11"})
        assert saved_document_response.status_code == 200
        assert saved_document_response.json()["total"] == 1
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_partial_save_keeps_unselected_staged_attributes() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(
        owner_id="local-development-user",
        name="partial.pdf",
        format="pdf",
        page_count=1,
        status="PROCESSED",
        extracted_attributes=[
            {
                "temp_id": "model-a",
                "page_id": 1,
                "page_number": 1,
                "document_id": 1,
                "entity_type_label": "A",
                "extracted_value": "one",
                "original_prediction": "one",
                "bounding_boxes": [],
                "confidence_score": 0.8,
                "validation_status": "PENDING",
                "source": "model",
                "saved": False,
            },
            {
                "temp_id": "model-b",
                "page_id": 1,
                "page_number": 1,
                "document_id": 1,
                "entity_type_label": "B",
                "extracted_value": "two",
                "original_prediction": "two",
                "bounding_boxes": [],
                "confidence_score": 0.8,
                "validation_status": "PENDING",
                "source": "model",
                "saved": False,
            },
        ],
    )
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(page)
    session.flush()
    document.extracted_attributes = [
        {**document.extracted_attributes[0], "page_id": page.id},
        {**document.extracted_attributes[1], "page_id": page.id},
    ]
    session.commit()

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)
    try:
        response = client.post(
            f"/api/v1/documents/{document.id}/attributes/save-selected",
            json={
                "document_id": document.id,
                "attributes": [
                    {
                        "temp_id": "model-a",
                        "source": "model",
                        "entity_type_label": "A",
                        "extracted_value": "one-edited",
                        "page_number": 1,
                        "page_id": page.id,
                        "bounding_boxes": [],
                        "confidence_score": 0.8,
                        "validation_status": "APPROVED",
                    }
                ],
            },
        )
        assert response.status_code == 200

        refreshed_document = session.get(Document, document.id)
        assert refreshed_document is not None
        assert len(refreshed_document.extracted_attributes) == 1
        assert refreshed_document.extracted_attributes[0]["temp_id"] == "model-b"
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_save_selected_returns_attribute_details() -> None:
    factory = _build_session_factory()
    session = factory()
    document = Document(
        owner_id="local-development-user",
        name="details.pdf",
        format="pdf",
        page_count=1,
        status="PROCESSED",
        extracted_attributes=[
            {
                "temp_id": "model-x",
                "page_id": 1,
                "page_number": 1,
                "document_id": 1,
                "entity_type_label": "REF",
                "extracted_value": "ABC",
                "original_prediction": "ABC",
                "bounding_boxes": [[5, 5, 15, 15]],
                "confidence_score": 0.85,
                "validation_status": "PENDING",
                "source": "model",
                "saved": False,
            }
        ],
    )
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(page)
    session.flush()
    document.extracted_attributes = [
        {**document.extracted_attributes[0], "page_id": page.id},
    ]
    session.commit()

    app.dependency_overrides[get_db] = _override_get_db(session)
    client = TestClient(app)

    try:
        response = client.post(
            f"/api/v1/documents/{document.id}/attributes/save-selected",
            json={
                "document_id": document.id,
                "attributes": [
                    {
                        "temp_id": "model-x",
                        "source": "model",
                        "entity_type_label": "REF",
                        "extracted_value": "XYZ",
                        "page_number": 1,
                        "page_id": page.id,
                        "bounding_boxes": [[5, 5, 15, 15]],
                        "confidence_score": 0.85,
                        "validation_status": "APPROVED",
                    }
                ],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["saved_count"] == 1
        saved_ids: list[int] = payload.get("saved_attribute_ids") or []
        assert len(saved_ids) == 1

        # Fetch attribute detail and validate values persisted
        attr_id = saved_ids[0]
        detail_resp = client.get(f"/api/v1/attributes/{attr_id}")
        assert detail_resp.status_code == 200
        attr_payload = detail_resp.json()
        assert attr_payload["attribute"]["extracted_value"] == "XYZ"
        assert attr_payload["attribute"]["entity_type_label"] == "REF"

    finally:
        app.dependency_overrides.clear()
        session.close()


def test_claude_failure_marks_document_failed_with_a_readable_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _build_session_factory()
    document_id = _seed_document(factory)
    source_path = tmp_path / "doc.pdf"
    source_path.write_bytes(b"%PDF-1.4\n%%EOF")
    page = tmp_path / "page-1.png"
    page.write_bytes(b"fake-png")

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", lambda path, artifact_id: [page])
    _use_fake_claude(monkeypatch, error=ClaudeExtractionError("Claude API returned HTTP 401 (check ANTHROPIC_API_KEY)"))

    with pytest.raises(ClaudeExtractionError):
        ingestion_pipeline.process_document_job(document_id, str(source_path), "doc.pdf", "local-development-user")

    verify_session = factory()
    try:
        document = verify_session.get(Document, document_id)
        assert document is not None
        assert document.status == "FAILED"
        assert document.processing_error == "Claude API returned HTTP 401 (check ANTHROPIC_API_KEY)"
        assert document.processing_finished_at is not None
        assert verify_session.query(Page).filter(Page.document_id == document_id).count() == 0
        assert not page.exists()
    finally:
        verify_session.close()

    app.dependency_overrides[get_db] = _override_get_db(factory())
    try:
        body = TestClient(app).get(f"/api/v1/documents/{document_id}").json()
        assert body["status"] == "FAILED"
        assert body["processing_error"] == "Claude API returned HTTP 401 (check ANTHROPIC_API_KEY)"
    finally:
        app.dependency_overrides.clear()


def test_unexpected_errors_get_a_generic_reason_without_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _build_session_factory()
    document_id = _seed_document(factory)
    source_path = tmp_path / "doc.pdf"
    source_path.write_bytes(b"%PDF-1.4\n%%EOF")
    page = tmp_path / "page-1.png"
    page.write_bytes(b"fake-png")

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", lambda path, artifact_id: [page])
    _use_fake_claude(monkeypatch, error=ValueError("secret request detail"))

    with pytest.raises(ValueError):
        ingestion_pipeline.process_document_job(document_id, str(source_path), "doc.pdf", "local-development-user")

    verify_session = factory()
    try:
        document = verify_session.get(Document, document_id)
        assert document is not None and document.status == "FAILED"
        assert document.processing_error == "Processing failed (ValueError)"
    finally:
        verify_session.close()


def test_missing_api_key_fails_the_document_before_rasterizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _build_session_factory()
    document_id = _seed_document(factory)
    source_path = tmp_path / "doc.pdf"
    source_path.write_bytes(b"%PDF-1.4\n%%EOF")
    rasterize_calls: list[Path] = []

    def rasterize(path: Path, artifact_id: str) -> list[Path]:
        rasterize_calls.append(path)
        return []

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", rasterize)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ClaudeExtractionError, match="ANTHROPIC_API_KEY is not set"):
        ingestion_pipeline.process_document_job(document_id, str(source_path), "doc.pdf", "local-development-user")

    verify_session = factory()
    try:
        document = verify_session.get(Document, document_id)
        assert document is not None
        assert document.status == "FAILED"
        assert document.processing_error == "ANTHROPIC_API_KEY is not set"
        assert rasterize_calls == []
    finally:
        verify_session.close()


def test_a_successful_retry_clears_the_previous_failure_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _build_session_factory()
    document_id = _seed_document(factory, processing_error="Claude API returned HTTP 429 (rate limit or spend cap reached)")
    source_path = tmp_path / "doc.pdf"
    source_path.write_bytes(b"%PDF-1.4\n%%EOF")
    page = tmp_path / "page-1.png"
    page.write_bytes(b"fake-png")

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", lambda path, artifact_id: [page])
    _use_fake_claude(monkeypatch, DocumentExtraction(page_text={1: "blank"}))

    ingestion_pipeline.process_document_job(document_id, str(source_path), "doc.pdf", "local-development-user")

    verify_session = factory()
    try:
        document = verify_session.get(Document, document_id)
        assert document is not None
        assert document.status == "PROCESSED"
        assert document.processing_error is None
        assert document.document_type == "unknown"
        assert document.extracted_attributes == []  # a page with nothing to extract is not an error
    finally:
        verify_session.close()


def test_cancel_during_extraction_stops_without_writing_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _build_session_factory()
    document_id = _seed_document(factory)
    source_path = tmp_path / "doc.pdf"
    source_path.write_bytes(b"%PDF-1.4\n%%EOF")
    page = tmp_path / "page-1.png"
    page.write_bytes(b"fake-png")

    class CancelingService:
        def extract_document(self, pages: list[tuple[int, Path]], should_cancel: Callable[[], bool] | None = None) -> DocumentExtraction:
            other_session = factory()
            other = other_session.get(Document, document_id)
            assert other is not None
            other.status = "CANCELED"
            other_session.commit()
            other_session.close()
            assert should_cancel is not None and should_cancel() is True
            raise ExtractionCanceled()

    monkeypatch.setattr(ingestion_pipeline, "SessionLocal", factory)
    monkeypatch.setattr(ingestion_pipeline, "_rasterize_pdf", lambda path, artifact_id: [page])
    monkeypatch.setattr(ingestion_pipeline, "ClaudeExtractionService", CancelingService)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    result = ingestion_pipeline.process_document_job(document_id, str(source_path), "doc.pdf", "local-development-user")

    assert result["status"] == "canceled"
    verify_session = factory()
    try:
        document = verify_session.get(Document, document_id)
        assert document is not None and document.status == "CANCELED"
        assert verify_session.query(Page).filter(Page.document_id == document_id).count() == 0
        assert not page.exists()
    finally:
        verify_session.close()


def test_ocr_endpoint_returns_stored_data_without_running_any_ocr(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "page-1.png"
    Image.new("RGB", (300, 200), "white").save(image_path)
    table = {"id": "page-1-table-1", "page_number": 1, "title": "Items", "rows": [], "columns": []}

    factory = _build_session_factory()
    session = factory()
    document = Document(
        name="stored.pdf",
        format="pdf",
        owner_id="local-development-user",
        page_count=2,
        status="PROCESSED",
        document_type="sales_invoice",
    )
    session.add(document)
    session.flush()
    session.add(Page(document_id=document.id, page_number=1, image_path=str(image_path), ocr_text="Invoice\n\nTotal 5", ocr_layout=[table]))
    # A page stored before migration 0013 has no transcription at all.
    session.add(Page(document_id=document.id, page_number=2, image_path=str(tmp_path / "missing.png")))
    session.commit()
    document_id = document.id

    app.dependency_overrides[get_db] = _override_get_db(session)
    try:
        response = TestClient(app).get(f"/api/v1/documents/{document_id}/ocr")
    finally:
        app.dependency_overrides.clear()
        session.close()

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id and body["document_type"] == "sales_invoice"
    first, second = body["pages"]
    assert first["page_number"] == 1
    assert first["text"] == "Invoice\n\nTotal 5"
    assert [line["text"] for line in first["lines"]] == ["Invoice", "Total 5"]
    assert all(line["tokens"] == [] for line in first["lines"])
    assert first["tokens"] == []
    assert (first["page_width"], first["page_height"]) == (300, 200)
    assert first["tables"] == [table]
    assert first["image_url"] == f"/api/v1/documents/{document_id}/pages/1/image"
    assert {"page_id", "lines", "tokens", "page_width", "page_height", "tables"} <= set(first)
    assert second["text"] == "" and second["lines"] == [] and second["tables"] == []
    assert second["page_width"] is None
    assert not hasattr(ingestion_pipeline, "extract_tokens_with_bboxes")  # nothing here can run OCR


def _staged(label: str, value: str, page: int, **extra: Any) -> dict[str, Any]:
    return {
        "entity_type_label": label,
        "extracted_value": value,
        "original_prediction": value,
        "page_number": page,
        "page_id": page,
        "bounding_boxes": [[0, 0, 10, 10]],
        "page_spans": [{"page_id": page, "page_number": page, "text": value, "bounding_boxes": [[0, 0, 10, 10]]}],
        "confidence_score": 0.9,
        "validation_status": "APPROVED",
        **extra,
    }


def test_same_label_on_adjacent_pages_is_not_merged_without_the_continuation_flag() -> None:
    merged = ingestion_pipeline._merge_staged_attributes(
        [_staged("Total", "10", 1), _staged("Total", "20", 2), _staged("Total", "30", 3, continues_from_previous_page=False)]
    )

    assert [item["extracted_value"] for item in merged] == ["10", "20", "30"]


def test_flagged_continuations_merge_across_several_pages() -> None:
    merged = ingestion_pipeline._merge_staged_attributes(
        [
            _staged("Notes", "first part", 1),
            _staged("Total", "99", 1),
            _staged("Notes", "second part", 2, continues_from_previous_page=True, confidence_score=0.6),
            _staged("Notes", "third part", 3, continues_from_previous_page=True),
        ]
    )

    assert [item["entity_type_label"] for item in merged] == ["Notes", "Total"]
    notes = merged[0]
    assert notes["extracted_value"] == "first part\n\nsecond part\n\nthird part"
    assert notes["original_prediction"] == notes["extracted_value"]
    assert [span["page_number"] for span in notes["page_spans"]] == [1, 2, 3]
    assert len(notes["bounding_boxes"]) == 3
    assert notes["confidence_score"] == 0.6 and notes["validation_status"] == "HITL"


def test_a_continuation_flag_without_a_matching_previous_page_value_stays_separate() -> None:
    merged = ingestion_pipeline._merge_staged_attributes(
        [_staged("Notes", "page one", 1), _staged("Notes", "page three", 3, continues_from_previous_page=True)]
    )

    assert len(merged) == 2
