from __future__ import annotations

import json

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.main import app
from app.models import Attribute, Base, Document, Page


def _build_test_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)()


def test_export_validated_attributes_json_and_csv() -> None:
    session = _build_test_session()
    document = Document(name="test.pdf", format="pdf", page_count=1, status="PROCESSED")
    session.add(document)
    session.flush()

    page = Page(document_id=document.id, page_number=1, image_path="datasets/raw-pages/1/page-001.png")
    session.add(page)
    session.flush()

    session.add_all(
        [
            Attribute(
                page_id=page.id,
                document_id=document.id,
                entity_type_label="TOTAL",
                extracted_value="42.00",
                bounding_boxes=[[0, 0, 10, 10]],
                confidence_score=0.97,
                validation_status="APPROVED",
            ),
            Attribute(
                page_id=page.id,
                document_id=document.id,
                entity_type_label="IGNORED",
                extracted_value="pending",
                bounding_boxes=[[1, 1, 2, 2]],
                confidence_score=0.12,
                validation_status="PENDING",
            ),
        ]
    )
    session.commit()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        json_response = client.get(f"/api/v1/attributes/export/{document.id}/json")
        assert json_response.status_code == 200
        payload = json_response.json()
        assert payload["exported_count"] == 1
        assert payload["rows"][0]["entity_type_label"] == "TOTAL"

        csv_response = client.get(f"/api/v1/attributes/export/{document.id}/csv")
        assert csv_response.status_code == 200
        assert csv_response.headers["content-type"].startswith("text/csv")
        assert "TOTAL" in csv_response.text
        assert "PENDING" not in csv_response.text
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_selected_table_round_trips_original_and_corrected_cells() -> None:
    session = _build_test_session()
    document = Document(name="table.pdf", format="pdf", owner_id="local-development-user", page_count=1, status="PROCESSED")
    session.add(document)
    session.flush()
    page = Page(document_id=document.id, page_number=1, image_path="table-page.png")
    session.add(page)
    session.flush()
    session.commit()

    table = {
        "id": "page-1-table-1",
        "title": "Course schedule",
        "page_number": 1,
        "bbox": [100, 120, 900, 760],
        "columns": [360, 440],
        "rows": [
            {"index": 0, "height": 120, "cells": [
                {"id": "r0-c0", "row_index": 0, "column_index": 0, "text": "Course", "original_text": "Course", "bbox": [100, 120, 460, 240], "rowspan": 1, "colspan": 1, "is_header": True},
                {"id": "r0-c1", "row_index": 0, "column_index": 1, "text": "Units", "original_text": "Unit5", "bbox": [460, 120, 900, 240], "rowspan": 1, "colspan": 1, "is_header": True},
            ]},
        ],
        "style": {"bordered": True, "layout_source": "detected_rules"},
        "uncertain_fields": ["font_size"],
    }
    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        save_response = client.post(
            f"/api/v1/documents/{document.id}/attributes/save-selected",
            json={
                "document_id": document.id,
                "attributes": [{
                    "temp_id": "table-page-1-table-1",
                    "kind": "table",
                    "metadata": table,
                    "source": "model",
                    "entity_type_label": table["title"],
                    "extracted_value": "Course\tUnits",
                    "page_number": 1,
                    "page_id": page.id,
                    "bounding_boxes": [table["bbox"]],
                    "page_spans": [{"page_number": 1, "bounding_boxes": [table["bbox"]]}],
                    "confidence_score": 0.9,
                }],
            },
        )
        assert save_response.status_code == 200
        saved_id = save_response.json()["saved_attribute_ids"][0]

        detail_response = client.get(f"/api/v1/attributes/{saved_id}")
        assert detail_response.status_code == 200
        persisted = detail_response.json()["attribute"]["table_structure"]
        assert persisted["style"]["layout_source"] == "detected_rules"
        assert persisted["rows"][0]["cells"][1]["original_text"] == "Unit5"
        assert persisted["rows"][0]["cells"][1]["text"] == "Units"

        reopened_response = client.get(f"/api/v1/documents/{document.id}/attributes")
        reopened_attribute = reopened_response.json()["attributes"][0]
        assert reopened_response.status_code == 200
        assert reopened_attribute["extracted_value"]
        reopened_table = json.loads(reopened_attribute["extracted_value"])
        assert reopened_table["table"]["rows"][0]["cells"][1]["original_text"] == "Unit5"
        assert reopened_table["table"]["rows"][0]["cells"][1]["text"] == "Units"

        json_response = client.get(f"/api/v1/attributes/export/{document.id}/json")
        exported_table = json_response.json()["rows"][0]["table_structure"]
        assert exported_table["bbox"] == table["bbox"]
        assert exported_table["rows"][0]["cells"][1]["original_text"] == "Unit5"

        csv_response = client.get(f"/api/v1/attributes/export/{document.id}/csv")
        assert "table_structure" in csv_response.text.splitlines()[0]
        assert "Unit5" in csv_response.text
    finally:
        app.dependency_overrides.clear()
        session.close()


