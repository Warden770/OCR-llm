from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

from app.document_schemas import get_document_schema


MAX_UNSTRUCTURED_ATTRIBUTE_LENGTH = 600


def _build_unstructured_ocr_payloads(
    page_id: int,
    document_id: int,
    tokens: Sequence[Dict[str, Any]],
    document_type: str | None,
) -> List[Dict[str, Any]]:
    usable_tokens = [
        item for item in tokens
        if str(item.get("token", "")).strip() and len(item.get("bbox", [])) == 4
    ]
    if not usable_tokens:
        return []

    paragraphs: dict[tuple[int, int], list[Dict[str, Any]]] = defaultdict(list)
    for token in usable_tokens:
        key = (
            int(token.get("block_num", 0) or 0),
            int(token.get("paragraph_num", 0) or 0),
        )
        paragraphs[key].append(token)

    payloads: list[dict[str, Any]] = []
    for paragraph in paragraphs.values():
        chunks: list[dict[str, Any]] = []
        current_tokens: list[Dict[str, Any]] = []
        current_length = 0
        for token in paragraph:
            token_text = str(token["token"]).strip()
            proposed_length = current_length + len(token_text) + (1 if current_tokens else 0)
            if current_tokens and proposed_length > MAX_UNSTRUCTURED_ATTRIBUTE_LENGTH:
                chunks.append({"tokens": current_tokens})
                current_tokens = []
                current_length = 0
            current_tokens.append(token)
            current_length += len(token_text) + (1 if current_length else 0)
        if current_tokens:
            chunks.append({"tokens": current_tokens})

        for chunk in chunks:
            chunk_tokens = chunk["tokens"]
            confidence = sum(float(item.get("confidence", 0.0)) / 100.0 for item in chunk_tokens) / len(chunk_tokens)
            payloads.append(
                {
                    "page_id": page_id,
                    "document_id": document_id,
                    "entity_type_label": f"ocr_text_{len(payloads) + 1}",
                    "extracted_value": " ".join(str(item["token"]).strip() for item in chunk_tokens),
                    "bounding_boxes": [list(item["bbox"]) for item in chunk_tokens],
                    "confidence_score": confidence,
                    "validation_status": "APPROVED" if confidence >= 0.75 else "HITL",
                    "document_type": document_type or "unknown",
                }
            )
    if len(payloads) == 1:
        payloads[0]["entity_type_label"] = "ocr_text"
    return payloads


def build_ocr_baseline_payloads(
    page_id: int,
    document_id: int,
    tokens: Sequence[Dict[str, Any]],
    document_type: str | None = None,
) -> List[Dict[str, Any]]:
    """Create useful field candidates from Tesseract layout without a trained model.

    This intentionally favors explicit labels and long text preservation over
    guessing arbitrary fields from raw OCR.
    """
    schema = get_document_schema(document_type)
    alias_map = {key.lower(): value for key, value in schema.label_aliases.items()}
    alias_map.update({field.lower(): field for field in schema.expected_fields})
    alias_map.update({field.lower().replace("_", " "): field for field in schema.expected_fields})
    if not alias_map:
        return _build_unstructured_ocr_payloads(page_id, document_id, tokens, document_type)

    lines: dict[tuple[int, int, int], list[Dict[str, Any]]] = {}
    for token in tokens:
        key = (
            int(token.get("block_num", 0) or 0),
            int(token.get("paragraph_num", 0) or 0),
            int(token.get("line_num", 0) or 0),
        )
        lines.setdefault(key, []).append(token)

    ordered_lines = [line for _, line in sorted(lines.items())]
    candidates: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    body_candidate: dict[str, Any] | None = None
    for line in ordered_lines:
        text = " ".join(str(item.get("token", "")).strip() for item in line).strip()
        if not text:
            continue
        label, separator, value = text.partition(":")
        normalized_label = alias_map.get(label.strip().lower())
        if normalized_label is None:
            first_word, _, remainder = text.partition(" ")
            normalized_label = alias_map.get(first_word.strip().lower())
            if normalized_label is not None:
                value = remainder.strip()
                separator = " " if value else ""

        if normalized_label is not None and (separator or value.strip()):
            if current:
                candidates.append(current)
                current = None
            if body_candidate:
                candidates.append(body_candidate)
                body_candidate = None
            value_tokens = line[len(line) - len(value.split()):] if value else []
            current = {
                "entity_type_label": normalized_label,
                "text_parts": [value.strip()] if value.strip() else [],
                "boxes": [item["bbox"] for item in (value_tokens or line)],
                "confidence": [float(item.get("confidence", 0.0)) / 100.0 for item in (value_tokens or line)],
                "page_id": page_id,
                "document_id": document_id,
                "block_num": int(line[0].get("block_num", 0) or 0),
                "paragraph_num": int(line[0].get("paragraph_num", 0) or 0),
                "line_num": int(line[0].get("line_num", 0) or 0),
            }
        elif current:
            if document_type != "letter":
                current["text_parts"].append(text)
                current["boxes"].extend(item["bbox"] for item in line)
                current["confidence"].extend(float(item.get("confidence", 0.0)) / 100.0 for item in line)
            else:
                candidates.append(current)
                current = None
        if document_type == "letter" and normalized_label is None:
            paragraph_key = (
                int(line[0].get("block_num", 0) or 0),
                int(line[0].get("paragraph_num", 0) or 0),
            )
            if body_candidate and body_candidate["paragraph_key"] == paragraph_key:
                body_candidate["text_parts"].append(text)
                body_candidate["boxes"].extend(item["bbox"] for item in line)
                body_candidate["confidence"].extend(float(item.get("confidence", 0.0)) / 100.0 for item in line)
            else:
                if body_candidate:
                    candidates.append(body_candidate)
                body_candidate = {
                    "entity_type_label": f"body_{len(candidates) + 1}",
                    "text_parts": [text],
                    "boxes": [item["bbox"] for item in line],
                    "confidence": [float(item.get("confidence", 0.0)) / 100.0 for item in line],
                    "page_id": page_id,
                    "document_id": document_id,
                    "paragraph_key": paragraph_key,
                }

    if current:
        candidates.append(current)
    if body_candidate:
        candidates.append(body_candidate)

    payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        value = "\n".join(part for part in candidate["text_parts"] if part).strip()
        if not value:
            continue
        confidence = sum(candidate["confidence"]) / max(len(candidate["confidence"]), 1)
        payloads.append(
            {
                "page_id": candidate["page_id"],
                "document_id": candidate["document_id"],
                "entity_type_label": candidate["entity_type_label"],
                "extracted_value": value,
                "bounding_boxes": candidate["boxes"],
                "confidence_score": confidence,
                "validation_status": "APPROVED" if confidence >= 0.75 else "HITL",
                "document_type": document_type or "unknown",
            }
        )
    if payloads:
        return payloads

    return _build_unstructured_ocr_payloads(page_id, document_id, tokens, document_type)
