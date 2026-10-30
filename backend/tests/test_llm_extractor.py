from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from app.llm_extractor import (
    LLMExtractionError,
    LLMExtractionService,
    build_llm_payloads,
    get_extraction_backend,
    label_tokens,
)

TOKENS = [
    {"token": "Invoice", "bbox": [0, 0, 100, 20], "confidence": 95.0, "block_num": 1, "paragraph_num": 1, "line_num": 1},
    {"token": "No:", "bbox": [110, 0, 150, 20], "confidence": 95.0, "block_num": 1, "paragraph_num": 1, "line_num": 1},
    {"token": "A-1O2", "bbox": [160, 0, 230, 20], "confidence": 60.0, "block_num": 1, "paragraph_num": 1, "line_num": 1},
    {"token": "Total", "bbox": [0, 40, 60, 60], "confidence": 95.0, "block_num": 1, "paragraph_num": 2, "line_num": 1},
    {"token": "42.00", "bbox": [70, 40, 130, 60], "confidence": 95.0, "block_num": 1, "paragraph_num": 2, "line_num": 1},
]


class FakeClient:
    def __init__(self, fields=None, content=None, error=None):
        self.calls: list[dict] = []
        self._content = content if content is not None else [
            SimpleNamespace(type="tool_use", name="record_fields", input={"fields": fields or []})
        ]
        self._error = error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(content=self._content)


@pytest.fixture
def page_image(tmp_path):
    path = tmp_path / "page-001.png"
    Image.new("RGB", (200, 100), "white").save(path)
    return path


def _extract(client, page_image, document_type="invoice"):
    return LLMExtractionService(client=client, model="test-model").extract_page(page_image, TOKENS, document_type)


def test_request_includes_image_numbered_tokens_and_schema_enum(page_image) -> None:
    client = FakeClient(fields=[])
    _extract(client, page_image)

    call = client.calls[0]
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert "[0]Invoice [1]No: [2]A-1O2" in content[1]["text"]
    assert call["tool_choice"] == {"type": "tool", "name": "record_fields"}
    label_schema = call["tools"][0]["input_schema"]["properties"]["fields"]["items"]["properties"]["label"]
    assert "invoice_number" in label_schema["enum"]


def test_grounded_fields_get_bboxes_and_review_status(page_image) -> None:
    client = FakeClient(
        fields=[
            {"label": "total_amount", "value": "42.00", "token_indices": [4], "confidence": 0.95},
            {"label": "invoice_number", "value": "A-102", "token_indices": [2], "confidence": 0.9},
        ]
    )
    extraction = _extract(client, page_image)
    payloads = build_llm_payloads(extraction, TOKENS, page_id=3, document_id=7)

    total = next(p for p in payloads if p["entity_type_label"] == "total_amount")
    assert total["bounding_boxes"] == [[70, 40, 130, 60]]
    assert total["validation_status"] == "APPROVED"
    assert total["page_id"] == 3 and total["document_id"] == 7

    # The model corrected an OCR error (A-1O2 -> A-102): confidence drops, so a person checks it.
    invoice = next(p for p in payloads if p["entity_type_label"] == "invoice_number")
    assert invoice["confidence_score"] == pytest.approx(0.72)
    assert invoice["validation_status"] == "HITL"


def test_ungrounded_and_invalid_fields_are_handled(page_image) -> None:
    client = FakeClient(
        fields=[
            {"label": "vendor_name", "value": "Acme", "token_indices": [], "confidence": 0.99},
            {"label": "bill_to", "value": "   ", "token_indices": [0], "confidence": 0.9},
            {"label": "total_amount", "value": "42.00", "token_indices": [99, -1, True], "confidence": 7},
            "not-a-dict",
        ]
    )
    payloads = build_llm_payloads(_extract(client, page_image), TOKENS, page_id=1, document_id=1)

    assert [p["entity_type_label"] for p in payloads] == ["vendor_name", "total_amount"]
    assert payloads[0]["confidence_score"] == 0.5 and payloads[0]["validation_status"] == "HITL"
    assert payloads[0]["bounding_boxes"] == []
    assert payloads[1]["confidence_score"] == 0.5  # clamped to 1.0, then capped because no valid token cited


def test_labels_are_normalized_through_schema_aliases(page_image) -> None:
    client = FakeClient(fields=[{"label": "Total", "value": "42.00", "token_indices": [4], "confidence": 0.9}])
    assert _extract(client, page_image).fields[0].label == "total_amount"


def test_missing_tool_output_raises(page_image) -> None:
    with pytest.raises(LLMExtractionError):
        _extract(FakeClient(content=[SimpleNamespace(type="text", text="sorry")]), page_image)


def test_api_errors_are_wrapped(page_image) -> None:
    with pytest.raises(LLMExtractionError, match="boom"):
        _extract(FakeClient(error=RuntimeError("boom")), page_image)


def test_label_tokens_marks_unlabeled_tokens_as_outside(page_image) -> None:
    client = FakeClient(fields=[{"label": "total_amount", "value": "42.00", "token_indices": [3, 4], "confidence": 0.9}])
    labels = [item["label"] for item in label_tokens(_extract(client, page_image), TOKENS)]
    assert labels == ["O", "O", "O", "total_amount", "total_amount"]


def test_backend_selection(monkeypatch) -> None:
    for name in ("EXTRACTION_BACKEND", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert get_extraction_backend() == "ocr"
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    assert get_extraction_backend() == "gemini"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    assert get_extraction_backend() == "llm"  # Claude wins when both are set
    monkeypatch.setenv("EXTRACTION_BACKEND", "gemini")
    assert get_extraction_backend() == "gemini"
    monkeypatch.setenv("EXTRACTION_BACKEND", "ocr")
    assert get_extraction_backend() == "ocr"
