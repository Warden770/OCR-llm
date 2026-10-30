from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence, cast
from xml.etree import ElementTree as ET

from PIL import Image

from app.llm_extractor import create_extraction_service, label_tokens
from app.ocr_processor import extract_tokens_with_bboxes

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
PAGE_NAME_PATTERN = re.compile(r"^page-(\d+)$", re.IGNORECASE)


def _dataset_identity(image_name: str) -> tuple[str, int]:
    relative_path = PurePosixPath(image_name.replace("\\", "/"))
    page_match = PAGE_NAME_PATTERN.fullmatch(relative_path.stem)
    if page_match and str(relative_path.parent) != ".":
        return relative_path.parent.as_posix(), int(page_match.group(1))
    return relative_path.with_suffix("").as_posix(), 1


def _record_key(record: dict[str, Any], fallback: str) -> str:
    if record.get("document_id") is not None and record.get("page_id") is not None:
        return f"{record['document_id']}:{record['page_id']}"
    if record.get("document_id") is not None and record.get("page_number") is not None:
        return f"{record['document_id']}:{record['page_number']}"
    return str(record.get("id") or fallback)


def _as_json_dict_list(payload: Any, source: str) -> list[dict[str, Any]]:
    values: list[Any] = cast(list[Any], payload if isinstance(payload, list) else [payload])
    normalized: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{source} must contain JSON objects")
        normalized.append(cast(dict[str, Any], value))
    return normalized


def _as_string_list(value: Any, source: str, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{source} requires {field_name} to be a list")
    items: list[Any] = cast(list[Any], value)
    return [str(item) for item in items]


def _as_bbox_list(value: Any, source: str, field_name: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{source} requires {field_name} to be a list")
    boxes: list[Any] = cast(list[Any], value)
    result: list[list[int]] = []
    for box in boxes:
        sequence = cast(Sequence[Any], box)
        if len(sequence) != 4:
            raise ValueError(f"{source} has malformed bbox entries in {field_name}")
        result.append([int(round(float(item))) for item in sequence])
    return result


def _load_json_records(folder: str | Path) -> list[dict[str, Any]]:
    root = Path(folder)
    if root.is_file():
        if root.suffix.lower() == ".jsonl":
            return _load_jsonl(root)
        payload = json.loads(root.read_text(encoding="utf-8"))
        return _as_json_dict_list(payload, str(root))
    if not root.is_dir():
        raise FileNotFoundError(f"Annotation folder or JSON file not found: {root}")

    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(_as_json_dict_list(payload, str(path)))
    if not records:
        raise ValueError(f"No JSON annotation files found in {root}")
    return records


def _validate_token_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    tokens = record.get("tokens")
    boxes = record.get("bboxes")
    labels = record.get("labels")
    if not isinstance(tokens, list) or not isinstance(boxes, list) or not isinstance(labels, list):
        raise ValueError(f"{source} requires tokens, bboxes, and labels arrays")
    if not len(cast(list[Any], tokens)) == len(cast(list[Any], boxes)) == len(cast(list[Any], labels)):
        raise ValueError(f"{source} has mismatched token, bbox, and label counts")
    normalized: dict[str, Any] = dict(record)
    normalized["tokens"] = _as_string_list(tokens, source, "tokens")
    normalized["bboxes"] = _as_bbox_list(boxes, source, "bboxes")
    normalized["labels"] = _as_string_list(labels, source, "labels")
    return normalized


def _validate_evaluation_alignment(
    truth: dict[str, Any], prediction: dict[str, Any], key: str
) -> None:
    if len(truth["tokens"]) != len(prediction["tokens"]):
        raise ValueError(
            f"Evaluation token count mismatch for {key}: "
            f"ground_truth={len(truth['tokens'])} predictions={len(prediction['tokens'])}"
        )

    for index, (truth_token, prediction_token) in enumerate(zip(truth["tokens"], prediction["tokens"])):
        if truth_token != prediction_token:
            raise ValueError(
                f"Evaluation token mismatch for {key} at index {index}: "
                f"ground_truth={truth_token!r} prediction={prediction_token!r}"
            )


def _write_jsonl(records: Iterable[dict[str, Any]], output: str | Path) -> int:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def generate_predictions(raw_folder: str | Path, output: str | Path, model: str | None = None) -> int:
    root = Path(raw_folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw image folder not found: {root}")
    extractor = create_extraction_service(model=model)
    records: list[dict[str, Any]] = []
    for image_path in sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        relative_id = image_path.relative_to(root).with_suffix("").as_posix()
        document_id, page_id = _dataset_identity(relative_id)
        tokens = extract_tokens_with_bboxes(image_path)
        extraction = extractor.extract_page(image_path, tokens)
        prediction_tokens = label_tokens(extraction, tokens)
        records.append(
            {
                "id": relative_id,
                "document_id": document_id,
                "page_id": page_id,
                "image": str(image_path),
                "tokens": [item["token"] for item in prediction_tokens],
                "bboxes": [item["bbox"] for item in prediction_tokens],
                "labels": [item["label"] for item in prediction_tokens],
                "confidences": [item["confidence"] for item in prediction_tokens],
                "document_type": extraction.document_type,
            }
        )
    if not records:
        raise ValueError(f"No supported images found in {root}")
    return _write_jsonl(records, output)


def _intersection_over_union(first: list[int], second: list[int]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    area_first = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_second = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def evaluate_predictions(predictions: str | Path, ground_truth: str | Path, iou_threshold: float = 0.5) -> dict[str, Any]:
    prediction_records = _load_jsonl(predictions)
    truth_records = _load_json_records(ground_truth)
    predictions_by_key = {_record_key(record, str(index)): record for index, record in enumerate(prediction_records)}
    truth_by_key = {_record_key(record, str(index)): record for index, record in enumerate(truth_records)}

    total = correct = bbox_matches = 0
    label_counts: dict[str, dict[str, int]] = {}
    missing_predictions = sorted(set(truth_by_key) - set(predictions_by_key))
    for key, truth in truth_by_key.items():
        prediction = predictions_by_key.get(key)
        if prediction is None:
            continue
        truth = _validate_token_record(truth, key)
        prediction = _validate_token_record(prediction, key)
        _validate_evaluation_alignment(truth, prediction, key)
        for index, truth_label in enumerate(truth["labels"]):
            total += 1
            predicted_label = prediction["labels"][index] if index < len(prediction["labels"]) else "O"
            stats = label_counts.setdefault(truth_label, {"tp": 0, "fp": 0, "fn": 0})
            if predicted_label == truth_label:
                correct += 1
                stats["tp"] += 1
            else:
                stats["fn"] += 1
                label_counts.setdefault(predicted_label, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
            if index < len(prediction["bboxes"]) and _intersection_over_union(truth["bboxes"][index], prediction["bboxes"][index]) >= iou_threshold:
                bbox_matches += 1

    per_label: dict[str, dict[str, float]] = {}
    for label, stats in label_counts.items():
        precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
        recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }
    exact_fields = 0
    comparable_fields = 0
    for key, truth in truth_by_key.items():
        if key in predictions_by_key and isinstance(truth.get("fields"), dict):
            predicted_fields = predictions_by_key[key].get("fields", {})
            for field, value in truth["fields"].items():
                comparable_fields += 1
                if str(predicted_fields.get(field, "")).strip() == str(value).strip():
                    exact_fields += 1

    return {
        "documents": len(truth_by_key),
        "evaluated_documents": len(set(truth_by_key) & set(predictions_by_key)),
        "missing_predictions": missing_predictions,
        "token_count": total,
        "token_accuracy": correct / max(total, 1),
        "bbox_iou_match_rate": bbox_matches / max(total, 1),
        "field_exact_match_rate": exact_fields / max(comparable_fields, 1) if comparable_fields else None,
        "per_label": per_label,
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Prediction line {line_number} must be an object")
                records.append(cast(dict[str, Any], value))
    return records


def convert_cvat_xml(xml_path: str | Path, image_root: str | Path, output: str | Path) -> int:
    root = Path(image_root)
    xml_root = ET.parse(xml_path).getroot()
    records: list[dict[str, Any]] = []
    for image_node in xml_root.findall(".//image"):
        image_name = image_node.attrib.get("name", "")
        document_id, page_id = _dataset_identity(image_name)
        image_path = root / image_name
        if not image_path.exists():
            image_path = root / Path(image_name).name
        if not image_path.exists():
            raise FileNotFoundError(f"CVAT image is missing: {image_name}")
        with Image.open(image_path) as image:
            width, height = image.size
        boxes: list[tuple[str, list[int]]] = []
        for box_node in image_node.findall("box"):
            box = [
                int(round(float(box_node.attrib[key])))
                for key in ("xtl", "ytl", "xbr", "ybr")
            ]
            boxes.append((str(box_node.attrib.get("label", "O")), box))
        ocr_tokens = extract_tokens_with_bboxes(image_path)
        tokens: list[str] = []
        bboxes: list[list[int]] = []
        labels: list[str] = []
        for token in ocr_tokens:
            absolute = token["absolute_bbox"]
            normalized = [
                int(round(1000 * absolute[0] / width)),
                int(round(1000 * absolute[1] / height)),
                int(round(1000 * absolute[2] / width)),
                int(round(1000 * absolute[3] / height)),
            ]
            best_label = "O"
            best_iou = 0.0
            for label, box in boxes:
                iou = _intersection_over_union(absolute, box)
                if iou > best_iou:
                    best_label, best_iou = label, iou
            tokens.append(str(token["token"]))
            bboxes.append(normalized)
            labels.append(best_label if best_iou >= 0.1 else "O")
        records.append({
            "id": image_name.replace("\\", "/").rsplit(".", 1)[0],
            "document_id": document_id,
            "page_id": page_id,
            "image": image_name,
            "tokens": tokens,
            "bboxes": bboxes,
            "labels": labels,
        })
    if not records:
        raise ValueError("No CVAT image annotations found")
    return _write_jsonl(records, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ground-truth, prediction, evaluation, and CVAT dataset tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Run OCR and LLM extraction over a raw image folder")
    predict_parser.add_argument("--raw", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--model")

    evaluate_parser = subparsers.add_parser("evaluate", help="Compare predictions with independent ground truth")
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--ground-truth", required=True)
    evaluate_parser.add_argument("--iou-threshold", type=float, default=0.5)

    cvat_parser = subparsers.add_parser("cvat-to-jsonl", help="Convert CVAT image XML rectangles to token-level ground-truth JSONL")
    cvat_parser.add_argument("--xml", required=True)
    cvat_parser.add_argument("--image-root", required=True)
    cvat_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "predict":
        count = generate_predictions(args.raw, args.output, args.model)
        result: Any = {"records_written": count, "output": args.output}
    elif args.command == "evaluate":
        result = evaluate_predictions(args.predictions, args.ground_truth, args.iou_threshold)
    else:
        count = convert_cvat_xml(args.xml, args.image_root, args.output)
        result = {"records_written": count, "output": args.output}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
