from __future__ import annotations

import json

import pytest

from app import dataset_tools
from app.dataset_tools import evaluate_predictions
from app.llm_extractor import ExtractedField, PageExtraction


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_predictions_reports_label_and_bbox_metrics(tmp_path) -> None:
    ground_truth = tmp_path / "ground_truth"
    _write_json(
        ground_truth / "invoice-001.json",
        {
            "document_id": "invoice-001",
            "page_id": 1,
            "tokens": ["Total", "42.00"],
            "bboxes": [[10, 10, 100, 40], [110, 10, 180, 40]],
            "labels": ["O", "TOTAL"],
        },
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "document_id": "invoice-001",
                "page_id": 1,
                "tokens": ["Total", "42.00"],
                "bboxes": [[10, 10, 100, 40], [110, 10, 180, 40]],
                "labels": ["O", "TOTAL"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_predictions(predictions, ground_truth)
    assert result["evaluated_documents"] == 1
    assert result["token_accuracy"] == 1.0
    assert result["bbox_iou_match_rate"] == 1.0
    assert result["per_label"]["TOTAL"]["f1"] == 1.0


def test_generated_prediction_keys_match_page_based_ground_truth(tmp_path, monkeypatch) -> None:
    raw_folder = tmp_path / "raw"
    image_path = raw_folder / "invoice-001" / "page-002.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        dataset_tools,
        "extract_tokens_with_bboxes",
        lambda path: [{"token": "Total", "bbox": [10, 10, 100, 40]}],
    )

    class FakeExtractor:
        def __init__(self, model=None):
            pass

        def extract_page(self, image_path, tokens, document_type=None):
            return PageExtraction(
                document_type="invoice",
                fields=[ExtractedField(label="TOTAL", value="Total", token_indices=[0], confidence=1.0)],
            )

    monkeypatch.setattr(dataset_tools, "create_extraction_service", lambda model=None: FakeExtractor(model))
    predictions_path = tmp_path / "predictions.jsonl"
    assert dataset_tools.generate_predictions(raw_folder, predictions_path) == 1

    prediction = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert prediction["document_id"] == "invoice-001"
    assert prediction["page_id"] == 2

    ground_truth_path = tmp_path / "ground-truth.jsonl"
    ground_truth_path.write_text(json.dumps({
        "document_id": "invoice-001",
        "page_id": 2,
        "tokens": ["Total"],
        "bboxes": [[10, 10, 100, 40]],
        "labels": ["TOTAL"],
    }) + "\n", encoding="utf-8")

    result = evaluate_predictions(predictions_path, ground_truth_path)
    assert result["evaluated_documents"] == 1
    assert result["missing_predictions"] == []


def test_evaluation_rejects_shifted_or_truncated_predictions(tmp_path) -> None:
    ground_truth = tmp_path / "ground-truth.jsonl"
    ground_truth.write_text(json.dumps({
        "document_id": "doc-1",
        "page_id": 1,
        "tokens": ["Total", "42.00"],
        "bboxes": [[10, 10, 100, 40], [110, 10, 180, 40]],
        "labels": ["O", "TOTAL"],
    }) + "\n", encoding="utf-8")

    shifted_predictions = tmp_path / "shifted.jsonl"
    shifted_predictions.write_text(json.dumps({
        "document_id": "doc-1",
        "page_id": 1,
        "tokens": ["42.00", "Total"],
        "bboxes": [[10, 10, 100, 40], [110, 10, 180, 40]],
        "labels": ["O", "TOTAL"],
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Evaluation token mismatch"):
        evaluate_predictions(shifted_predictions, ground_truth)

    truncated_predictions = tmp_path / "truncated.jsonl"
    truncated_predictions.write_text(json.dumps({
        "document_id": "doc-1",
        "page_id": 1,
        "tokens": ["Total"],
        "bboxes": [[10, 10, 100, 40]],
        "labels": ["O"],
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Evaluation token count mismatch"):
        evaluate_predictions(truncated_predictions, ground_truth)
