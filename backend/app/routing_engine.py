from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from app.document_schemas import get_document_schema, normalize_prediction_label


SAME_LINE_HORIZONTAL_GAP_MULTIPLIER = 1.5
SAME_BLOCK_VERTICAL_GAP_MULTIPLIER = 1.75
SAME_BLOCK_LEFT_ALIGNMENT_MULTIPLIER = 2.0
SAME_LINE_VERTICAL_OVERLAP_RATIO = 0.25
MAX_UNSTRUCTURED_ATTRIBUTE_LENGTH = 600


@dataclass
class RoutedAttribute:
    entity_type_label: str
    extracted_value: str
    bounding_boxes: List[List[int]]
    confidence_score: float
    validation_status: str


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


def _bbox_metrics(bbox: Sequence[int]) -> dict[str, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    return {
        "left": x0,
        "top": y0,
        "right": x1,
        "bottom": y1,
        "width": width,
        "height": height,
        "center_x": x0 + width / 2.0,
        "center_y": y0 + height / 2.0,
    }


def _reading_order_key(prediction: Dict[str, Any]) -> tuple[int, int, int, float, float]:
    block_num = int(prediction.get("block_num", 0) or 0)
    paragraph_num = int(prediction.get("paragraph_num", 0) or 0)
    line_num = int(prediction.get("line_num", 0) or 0)
    metrics = _bbox_metrics(prediction.get("bbox", [0, 0, 0, 0]))
    return (block_num, paragraph_num, line_num, metrics["top"], metrics["left"])


def _should_merge_spatially(current_cluster: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    """Merge only when the next token is plausibly part of the same field span.

    The current heuristic is intentionally conservative:
    - same-line spans require matching block/paragraph metadata and a horizontal gap
      no larger than 1.5x the wider token box
    - same-block fallback merges require a vertical gap no larger than 1.75x the
      taller box and a left-edge delta no larger than 2.0x the wider box
    - when line metadata is missing, same-line detection falls back to a 25% vertical
      overlap ratio between the two boxes
    """
    current_metrics = current_cluster["metrics"]
    candidate_metrics = candidate["metrics"]

    same_block = current_cluster["block_num"] == candidate["block_num"] if current_cluster["block_num"] and candidate["block_num"] else True
    same_paragraph = (
        current_cluster["paragraph_num"] == candidate["paragraph_num"]
        if current_cluster["paragraph_num"] and candidate["paragraph_num"]
        else True
    )

    vertical_overlap = min(current_metrics["bottom"], candidate_metrics["bottom"]) - max(
        current_metrics["top"], candidate_metrics["top"]
    )
    horizontal_gap = candidate_metrics["left"] - current_metrics["right"]
    vertical_gap = candidate_metrics["top"] - current_metrics["bottom"]

    same_line = False
    if current_cluster["line_num"] and candidate["line_num"]:
        same_line = current_cluster["line_num"] == candidate["line_num"]
    elif vertical_overlap > 0:
        same_line = vertical_overlap >= SAME_LINE_VERTICAL_OVERLAP_RATIO * max(
            1.0,
            min(current_metrics["height"], candidate_metrics["height"]),
        )

    if same_line and same_block and same_paragraph:
        return horizontal_gap <= max(current_metrics["width"], candidate_metrics["width"]) * SAME_LINE_HORIZONTAL_GAP_MULTIPLIER

    if same_block and vertical_gap >= 0:
        return vertical_gap <= max(current_metrics["height"], candidate_metrics["height"]) * SAME_BLOCK_VERTICAL_GAP_MULTIPLIER and abs(
            candidate_metrics["left"] - current_metrics["left"]
        ) <= max(current_metrics["width"], candidate_metrics["width"]) * SAME_BLOCK_LEFT_ALIGNMENT_MULTIPLIER

    return False


def cluster_entity_tokens(token_predictions: Sequence[Dict[str, Any]], threshold: float = 0.75) -> List[RoutedAttribute]:
    labels_to_predictions: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for prediction in token_predictions:
        label = str(prediction.get("label", "")).strip()
        token = str(prediction.get("token", "")).strip()
        bbox = list(prediction.get("bbox", []))
        if not label or not token or len(bbox) != 4:
            continue

        metrics = _bbox_metrics(bbox)
        labels_to_predictions[label].append(
            {
                "label": label,
                "token": token,
                "confidence": float(prediction.get("confidence", 0.0)),
                "bbox": bbox,
                "metrics": metrics,
                "sort_key": _reading_order_key(prediction),
                "block_num": int(prediction.get("block_num", 0) or 0),
                "paragraph_num": int(prediction.get("paragraph_num", 0) or 0),
                "line_num": int(prediction.get("line_num", 0) or 0),
            }
        )

    clusters: List[Dict[str, Any]] = []
    for label, label_predictions in labels_to_predictions.items():
        ordered_predictions = sorted(label_predictions, key=lambda item: item["sort_key"])
        current: Dict[str, Any] | None = None

        for prediction in ordered_predictions:
            if current is None:
                current = {
                    "label": label,
                    "tokens": [prediction["token"]],
                    "boxes": [prediction["bbox"]],
                    "confidences": [prediction["confidence"]],
                    "paragraphs": [prediction["paragraph_num"]],
                    "lines": [prediction["line_num"]],
                    "sort_key": prediction["sort_key"],
                    "metrics": prediction["metrics"],
                    "block_num": prediction["block_num"],
                    "paragraph_num": prediction["paragraph_num"],
                    "line_num": prediction["line_num"],
                }
                continue

            if _should_merge_spatially(current, prediction):
                current["tokens"].append(prediction["token"])
                current["boxes"].append(prediction["bbox"])
                current["confidences"].append(prediction["confidence"])
                current["paragraphs"].append(prediction["paragraph_num"])
                current["lines"].append(prediction["line_num"])
                current["metrics"] = prediction["metrics"]
                current["sort_key"] = min(current["sort_key"], prediction["sort_key"])
            else:
                clusters.append(current)
                current = {
                    "label": label,
                    "tokens": [prediction["token"]],
                    "boxes": [prediction["bbox"]],
                    "confidences": [prediction["confidence"]],
                    "paragraphs": [prediction["paragraph_num"]],
                    "lines": [prediction["line_num"]],
                    "sort_key": prediction["sort_key"],
                    "metrics": prediction["metrics"],
                    "block_num": prediction["block_num"],
                    "paragraph_num": prediction["paragraph_num"],
                    "line_num": prediction["line_num"],
                }

        if current:
            clusters.append(current)

    clusters.sort(key=lambda cluster: cluster["sort_key"])

    routed: List[RoutedAttribute] = []
    for cluster in clusters:
        mean_confidence = sum(cluster["confidences"]) / max(len(cluster["confidences"]), 1)
        validation_status = "APPROVED" if mean_confidence >= threshold else "HITL"
        value_parts: list[str] = []
        previous_paragraph: int | None = None
        previous_line: int | None = None
        for index, token in enumerate(cluster["tokens"]):
            paragraph_num = cluster["paragraphs"][index]
            line_num = cluster["lines"][index]
            if index:
                if paragraph_num != previous_paragraph:
                    value_parts.append("\n\n")
                else:
                    value_parts.append(" ")
            value_parts.append(token)
            previous_paragraph = paragraph_num
            previous_line = line_num
        routed.append(
            RoutedAttribute(
                entity_type_label=str(cluster["label"]),
                extracted_value="".join(value_parts).strip(),
                bounding_boxes=[list(box) for box in cluster["boxes"]],
                confidence_score=mean_confidence,
                validation_status=validation_status,
            )
        )
    return routed


def build_attribute_payloads(
    page_id: int,
    document_id: int,
    token_predictions: Sequence[Dict[str, Any]],
    document_type: str | None = None,
    threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    normalized_predictions: list[Dict[str, Any]] = []
    for prediction in token_predictions:
        normalized_prediction = dict(prediction)
        normalized_prediction["label"] = normalize_prediction_label(document_type, str(prediction.get("label", "")))
        normalized_predictions.append(normalized_prediction)

    routed_attributes = cluster_entity_tokens(normalized_predictions, threshold=threshold)
    payloads: List[Dict[str, Any]] = []
    for attribute in routed_attributes:
        payloads.append(
            {
                "page_id": page_id,
                "document_id": document_id,
                "entity_type_label": attribute.entity_type_label,
                "extracted_value": attribute.extracted_value,
                "bounding_boxes": attribute.bounding_boxes,
                "confidence_score": attribute.confidence_score,
                "validation_status": attribute.validation_status,
                "document_type": document_type or "unknown",
            }
        )
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
