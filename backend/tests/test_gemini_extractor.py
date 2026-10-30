from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from app.gemini_extractor import GeminiExtractionService
from app.llm_extractor import LLMExtractionError, build_llm_payloads

TOKENS = [
    {"token": "Total", "bbox": [0, 40, 60, 60], "confidence": 95.0, "block_num": 1, "paragraph_num": 1, "line_num": 1},
    {"token": "42.00", "bbox": [70, 40, 130, 60], "confidence": 95.0, "block_num": 1, "paragraph_num": 1, "line_num": 1},
]


def _response(fields=None, status=200, raw_body=None):
    body = raw_body if raw_body is not None else {
        "candidates": [{"content": {"parts": [{"text": json.dumps({"fields": fields or []})}]}}]
    }
    return SimpleNamespace(status_code=status, json=lambda: body)


@pytest.fixture
def page_image(tmp_path):
    path = tmp_path / "page-001.png"
    Image.new("RGB", (200, 100), "white").save(path)
    return path


def _service(response, calls=None):
    def post(url, headers, payload, timeout):
        if calls is not None:
            calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        if isinstance(response, Exception):
            raise response
        return response

    return GeminiExtractionService(model="test-model", api_key="test-key", http_post=post)


def test_request_shape_and_key_in_header_only(page_image) -> None:
    calls: list[dict] = []
    _service(_response(), calls).extract_page(page_image, TOKENS, "invoice")

    call = calls[0]
    assert call["url"].endswith("/models/test-model:generateContent")
    assert "test-key" not in call["url"]
    assert call["headers"]["x-goog-api-key"] == "test-key"
    parts = call["payload"]["contents"][0]["parts"]
    assert "inlineData" in parts[0]
    assert "[0]Total [1]42.00" in parts[1]["text"]
    config = call["payload"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    label_schema = config["responseSchema"]["properties"]["fields"]["items"]["properties"]["label"]
    assert "total_amount" in label_schema["enum"]


def test_fields_become_grounded_payloads(page_image) -> None:
    extraction = _service(
        _response([{"label": "total_amount", "value": "42.00", "token_indices": [1], "confidence": 0.95}])
    ).extract_page(page_image, TOKENS, "invoice")
    payload = build_llm_payloads(extraction, TOKENS, page_id=1, document_id=1)[0]
    assert payload["bounding_boxes"] == [[70, 40, 130, 60]]
    assert payload["validation_status"] == "APPROVED"


@pytest.mark.parametrize(
    "response",
    [
        _response(status=429, raw_body={"error": {"message": "secret detail"}}),
        _response(raw_body={"candidates": []}),
        _response(raw_body={"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}),
        _response(raw_body={"candidates": [{"content": {"parts": [{"text": json.dumps({"fields": "x"})}]}}]}),
        RuntimeError("secret detail"),
    ],
)
def test_failures_raise_without_leaking_details(page_image, response) -> None:
    with pytest.raises(LLMExtractionError) as error:
        _service(response).extract_page(page_image, TOKENS, "invoice")
    assert "secret detail" not in str(error.value)


def test_missing_key_raises(page_image, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMExtractionError, match="GEMINI_API_KEY"):
        GeminiExtractionService(http_post=lambda *a, **k: None).extract_page(page_image, TOKENS)
