from __future__ import annotations

import csv
import io
import logging
import json
from typing import Any, List, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import CurrentUserId, auth_required, owner_scope
from app.models import Attribute, Correction, Document, Page

logger = logging.getLogger(__name__)
router = APIRouter()
VALIDATED_STATUSES = {"APPROVED", "VALIDATED", "CORRECTED"}


def _table_structure(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or parsed.get("kind") != "table":
        return None
    table = parsed.get("table")
    return table if isinstance(table, dict) else parsed


class CorrectionItem(BaseModel):
    attribute_id: int = Field(..., ge=1)
    original_prediction: str
    corrected_value: str
    validation_status: str | None = None


class CorrectionBatchItem(CorrectionItem):
    corrected_label: str | None = None


class CorrectionRequest(BaseModel):
    document_id: int | str
    corrections: List[CorrectionBatchItem]


class SelectedAttributeItem(BaseModel):
    temp_id: str = Field(..., min_length=1)
    source: Literal["model", "manual"] = "model"
    entity_type_label: str = Field(..., min_length=1)
    extracted_value: str = Field(..., min_length=1)
    page_number: int = Field(1, ge=1)
    page_id: int | None = None
    bounding_boxes: list[list[float | int]] = Field(default_factory=list)
    page_spans: list[dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 0.0
    validation_status: str = "PENDING"
    kind: str | None = None
    metadata: dict[str, Any] | None = None


class SaveSelectedAttributesRequest(BaseModel):
    document_id: int | str
    attributes: List[SelectedAttributeItem]


def _load_export_rows(document_id: int, db: Session) -> tuple[Document, list[dict[str, Any]]]:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    statement = (
        select(Attribute, Page)
        .join(Page, Attribute.page_id == Page.id)
        .where(
            Attribute.document_id == document_id,
            Attribute.validation_status.in_(VALIDATED_STATUSES),
        )
        .order_by(Page.page_number, Attribute.id)
    )
    rows_result = db.execute(statement).all()

    rows: list[dict[str, Any]] = []
    for attribute, page in rows_result:
        rows.append(
            {
                "document_id": document.id,
                "document_name": document.name,
                "document_format": document.format,
                "page_id": page.id,
                "page_number": page.page_number,
                "attribute_id": attribute.id,
                "entity_type_label": attribute.entity_type_label,
                "extracted_value": attribute.extracted_value,
                "confidence_score": attribute.confidence_score,
                "validation_status": attribute.validation_status,
                "bounding_boxes": attribute.bounding_boxes,
                "page_spans": attribute.page_spans,
                "kind": "table" if _table_structure(attribute.extracted_value) else None,
                "table_structure": _table_structure(attribute.extracted_value),
            }
        )

    return document, rows


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    fieldnames = [
        "document_id",
        "document_name",
        "document_format",
        "page_id",
        "page_number",
        "attribute_id",
        "entity_type_label",
        "extracted_value",
        "confidence_score",
        "validation_status",
        "bounding_boxes",
        "page_spans",
        "kind",
        "table_structure",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        csv_row = dict(row)
        csv_row["bounding_boxes"] = json.dumps(row["bounding_boxes"], ensure_ascii=False)
        csv_row["page_spans"] = json.dumps(row["page_spans"], ensure_ascii=False)
        csv_row["table_structure"] = json.dumps(row["table_structure"], ensure_ascii=False) if row["table_structure"] else ""
        writer.writerow(csv_row)
    return buffer.getvalue()


@router.post("/attributes/correct")
def correct_attributes(payload: CorrectionRequest, user_id: CurrentUserId, db: Session = Depends(get_db)) -> dict[str, Any]:
    applied_corrections: List[int] = []
    logger.info(
        "Received %s attribute correction(s) for document %s",
        len(payload.corrections),
        payload.document_id,
    )

    try:
        for item in payload.corrections:
            attribute = db.query(Attribute).join(Document).filter(Attribute.id == item.attribute_id, owner_scope(Document.owner_id, user_id)).first()
            if attribute is None:
                raise HTTPException(status_code=404, detail=f"Attribute {item.attribute_id} not found")
            if str(payload.document_id) != str(attribute.document_id):
                raise HTTPException(status_code=403, detail="Attribute does not belong to document")

            attribute.extracted_value = item.corrected_value
            if item.corrected_label:
                attribute.entity_type_label = item.corrected_label
            if item.validation_status:
                attribute.validation_status = item.validation_status

            correction = Correction(
                attribute_id=item.attribute_id,
                original_prediction=item.original_prediction,
                corrected_value=item.corrected_value,
                operator_user=user_id,
            )
            db.add(correction)
            db.add(attribute)
            db.flush()
            applied_corrections.append(correction.id)

        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("Failed to persist corrections for document %s", payload.document_id)
        raise HTTPException(status_code=500, detail="Failed to persist corrections")

    return {
        "status": "accepted",
        "document_id": payload.document_id,
        "correction_count": len(payload.corrections),
        "applied_correction_ids": applied_corrections,
    }


@router.post("/documents/{document_id}/attributes/save-selected")
def save_selected_attributes(document_id: int, payload: SaveSelectedAttributesRequest, user_id: CurrentUserId, db: Session = Depends(get_db)) -> dict[str, Any]:
    if int(payload.document_id) != document_id:
        raise HTTPException(status_code=400, detail="Payload document_id does not match URL")
    document = db.query(Document).filter(Document.id == document_id, owner_scope(Document.owner_id, user_id)).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    staged_rows = document.extracted_attributes or []
    staged_lookup = {str(row.get("temp_id")): row for row in staged_rows if row.get("temp_id")}
    saved_attribute_ids: list[int] = []
    selected_temp_ids: set[str] = {item.temp_id for item in payload.attributes}

    pages_by_number = {page.page_number: page for page in document.pages}

    try:
        for item in payload.attributes:
            staged_row = staged_lookup.get(item.temp_id)

            page_id = item.page_id or (staged_row.get("page_id") if staged_row else None)
            page_number = item.page_number or (staged_row.get("page_number") if staged_row else 1)
            if page_id is None:
                if not isinstance(page_number, int):
                    raise HTTPException(status_code=400, detail=f"Invalid page number for attribute {item.temp_id}")
                page = pages_by_number.get(page_number)
                if page is None:
                    raise HTTPException(status_code=400, detail=f"Page {page_number} not found for attribute {item.temp_id}")
                page_id = page.id

            # Force saved attributes to be visible in search by marking them APPROVED.
            # Ignore incoming validation_status from the client to avoid saved
            # rows remaining in HITL/PENDING states.
            validation_status_value = "APPROVED"

            if page_id not in {page.id for page in document.pages}:
                raise HTTPException(status_code=403, detail="Page does not belong to document")

            normalized_label = item.entity_type_label.strip()
            normalized_value = item.extracted_value.strip()
            if item.kind == "table" and item.metadata:
                table_structure = {**item.metadata, "kind": "table", "version": item.metadata.get("version", 1)}
                normalized_label = str(table_structure.get("title") or item.entity_type_label).strip() or normalized_label
                normalized_value = json.dumps(
                    {"kind": "table", "version": table_structure["version"], "table": table_structure},
                    ensure_ascii=False,
                )
            elif item.kind == "text_field" and item.metadata:
                normalized_label = str(item.metadata.get("label") or item.entity_type_label).strip() or normalized_label
                normalized_value = str(item.metadata.get("value") or item.extracted_value).strip()

            saved_attribute = Attribute(
                page_id=page_id,
                document_id=document.id,
                entity_type_label=normalized_label,
                extracted_value=normalized_value,
                bounding_boxes=item.bounding_boxes or (staged_row.get("bounding_boxes") if staged_row else []),
                page_spans=item.page_spans or (staged_row.get("page_spans") if staged_row else []),
                confidence_score=float(item.confidence_score if item.source == "manual" else (staged_row.get("confidence_score", item.confidence_score) if staged_row else item.confidence_score)),
                validation_status=validation_status_value,
                source=item.source,
            )
            db.add(saved_attribute)
            db.flush()
            saved_attribute_ids.append(saved_attribute.id)

        document.extracted_attributes = [
            row for row in staged_rows if str(row.get("temp_id")) not in selected_temp_ids
        ]
        db.add(document)
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save selected attributes")

    return {
        "status": "accepted",
        "document_id": document.id,
        "saved_attribute_ids": saved_attribute_ids,
        "saved_count": len(saved_attribute_ids),
    }


@router.get("/attributes/{attribute_id}")
def get_attribute_detail(attribute_id: int, user_id: CurrentUserId, db: Session = Depends(get_db)) -> dict[str, Any]:
    statement = (
        select(Attribute, Document, Page)
        .join(Document, Attribute.document_id == Document.id)
        .join(Page, Attribute.page_id == Page.id)
        .where(Attribute.id == attribute_id, owner_scope(Document.owner_id, user_id))
    )
    row = db.execute(statement).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Attribute not found")

    attribute, document, page = row
    return {
        "attribute": {
            "id": attribute.id,
            "page_id": attribute.page_id,
            "document_id": attribute.document_id,
            "document_name": document.name,
            "document_status": document.status,
            "page_number": page.page_number,
            "entity_type_label": attribute.entity_type_label,
            "extracted_value": attribute.extracted_value,
            "confidence_score": attribute.confidence_score,
            "validation_status": attribute.validation_status,
            "bounding_boxes": attribute.bounding_boxes,
            "page_spans": attribute.page_spans,
            "kind": "table" if _table_structure(attribute.extracted_value) else None,
            "table_structure": _table_structure(attribute.extracted_value),
        }
    }


@router.delete("/attributes/{attribute_id}")
def delete_attribute(attribute_id: int, user_id: CurrentUserId, db: Session = Depends(get_db)) -> dict[str, Any]:
    attribute = db.query(Attribute).join(Document).filter(Attribute.id == attribute_id, owner_scope(Document.owner_id, user_id)).first()
    if attribute is None:
        raise HTTPException(status_code=404, detail="Attribute not found")

    document_id = attribute.document_id
    db.delete(attribute)
    db.commit()
    return {"status": "deleted", "attribute_id": attribute_id, "document_id": document_id}


@router.get("/attributes/export/{document_id}/json")
def export_validated_attributes_json(document_id: int, user_id: CurrentUserId, db: Session = Depends(get_db)) -> dict[str, Any]:
    document, rows = _load_export_rows(document_id, db)
    if auth_required() and document.owner_id != user_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": document.id,
        "document_name": document.name,
        "document_format": document.format,
        "exported_count": len(rows),
        "rows": rows,
    }


@router.get("/attributes/export/{document_id}/csv")
def export_validated_attributes_csv(document_id: int, user_id: CurrentUserId, db: Session = Depends(get_db)) -> Response:
    document, rows = _load_export_rows(document_id, db)
    if auth_required() and document.owner_id != user_id:
        raise HTTPException(status_code=404, detail="Document not found")
    csv_content = _rows_to_csv(rows)
    filename = f"document-{document.id}-validated-attributes.csv"
    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


