from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from app.gemini_extractor import (
    DEFAULT_GEMINI_MODEL,
    DocumentExtraction,
    ExtractedAttribute,
    ExtractedCell,
    ExtractedTable,
    ExtractionCanceled,
    GeminiExtractionError,
    GeminiServiceError,
    GeminiExtractionService,
    build_attribute_payloads,
    build_table_layouts,
    extraction_from_json,
    extraction_to_json,
    normalize_document_type,
)

API_KEY = "test-key-SECRET-123"
RESPONSE_BODY_MARKER = "BODY-MARKER-do-not-leak"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _attribute(label="Student Name", value="John Doe", page=1, confidence=0.98, **extra) -> dict[str, Any]:
    return {"attribute": label, "value": value, "confidence": confidence, "page": page, **extra}


def _payload(document_type="Student Registration Form", pages=None, attributes=None, tables=None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "document_type": document_type,
        "pages": pages if pages is not None else [{"page": 1, "text": "Student Name: John Doe"}],
        "attributes": attributes if attributes is not None else [_attribute()],
    }
    if tables is not None:
        body["tables"] = tables
    return body


def _response(payload=None, status=200, headers=None, finish_reason=None, raw_text=None, body=None):
    if body is None:
        text = raw_text if raw_text is not None else json.dumps(payload if payload is not None else _payload())
        candidate: dict[str, Any] = {"content": {"parts": [{"text": text}]}}
        if finish_reason:
            candidate["finishReason"] = finish_reason
        body = {"candidates": [candidate]}
    return SimpleNamespace(status_code=status, json=lambda: body, headers=headers or {})


class FakeTransport:
    """Replays queued responses (or raises queued exceptions) and records every call."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        item = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def _service(transport, **kwargs) -> tuple[GeminiExtractionService, list[float]]:
    sleeps: list[float] = []
    kwargs.setdefault("pages_per_request", 6)
    service = GeminiExtractionService(
        model="test-model",
        api_key=API_KEY,
        http_post=transport,
        sleep=sleeps.append,
        timeout_seconds=30,
        request_delay_seconds=kwargs.pop("request_delay_seconds", 0.0),
        **kwargs,
    )
    return service, sleeps


@pytest.fixture
def pages(tmp_path):
    def make(count: int, size=(200, 100)) -> list[tuple[int, Any]]:
        result = []
        for number in range(1, count + 1):
            path = tmp_path / f"page-{number:03d}.png"
            Image.new("RGB", size, "white").save(path)
            result.append((number, path))
        return result

    return make


def _page_labels(call) -> list[str]:
    parts = call["payload"]["contents"][0]["parts"]
    return [part["text"] for part in parts if "text" in part and part["text"].startswith("Page ")]


# ---------------------------------------------------------------------------
# request shape
# ---------------------------------------------------------------------------

def test_request_shape_key_only_in_header_and_no_attribute_enum(pages) -> None:
    transport = FakeTransport(_response(_payload(pages=[{"page": 1, "text": "a"}, {"page": 2, "text": "b"}], attributes=[])))
    service, _ = _service(transport)

    service.extract_document(pages(2))

    call = transport.calls[0]
    assert call["url"].endswith("/models/test-model:generateContent")
    assert API_KEY not in call["url"]
    assert call["headers"]["x-goog-api-key"] == API_KEY
    assert API_KEY not in json.dumps(call["payload"])
    assert call["timeout"] == 30

    parts = call["payload"]["contents"][0]["parts"]
    assert [list(part)[0] for part in parts[:4]] == ["text", "inlineData", "text", "inlineData"]
    assert parts[0]["text"] == "Page 1" and parts[2]["text"] == "Page 2"
    assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"
    assert "1, 2" in parts[-1]["text"]

    config = call["payload"]["generationConfig"]
    assert config["temperature"] == 0
    assert config["responseMimeType"] == "application/json"
    schema = config["responseSchema"]
    assert '"enum"' not in json.dumps(schema["properties"]["attributes"])  # attribute names are free-form
    table_props = schema["properties"]["tables"]["items"]["properties"]
    assert table_props["table_type"]["enum"] == ["data", "form"]
    assert {"cells", "row_count", "column_count", "column_widths", "row_heights", "bordered"} <= set(table_props)
    assert {"row", "column", "row_span", "column_span", "text", "is_header"} == set(table_props["cells"]["items"]["properties"])
    assert schema["properties"]["attributes"]["items"]["properties"]["attribute"] == {"type": "STRING"}
    assert call["payload"]["systemInstruction"]["parts"][0]["text"]


def test_default_model_comes_from_environment_then_constant(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_EXTRACTION_MODEL", raising=False)
    assert GeminiExtractionService(api_key="k").model == DEFAULT_GEMINI_MODEL
    monkeypatch.setenv("GEMINI_EXTRACTION_MODEL", "custom-model")
    assert GeminiExtractionService(api_key="k").model == "custom-model"


def test_settings_read_from_environment_and_fall_back_when_invalid(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_PAGES_PER_REQUEST", "3")
    monkeypatch.setenv("GEMINI_IMAGE_MAX_EDGE", "1024")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("GEMINI_REQUEST_DELAY_SECONDS", "1.5")
    service = GeminiExtractionService(api_key="k")
    assert (service.pages_per_request, service.image_max_edge, service.timeout_seconds, service.request_delay_seconds) == (3, 1024, 45.0, 1.5)

    monkeypatch.setenv("GEMINI_PAGES_PER_REQUEST", "not-a-number")
    monkeypatch.setenv("GEMINI_IMAGE_MAX_EDGE", "-5")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "")
    service = GeminiExtractionService(api_key="k")
    assert (service.pages_per_request, service.image_max_edge, service.timeout_seconds) == (6, 2048, 120.0)


def test_large_pages_are_downscaled_to_the_configured_edge(pages) -> None:
    transport = FakeTransport(_response(_payload(attributes=[])))
    service, _ = _service(transport, image_max_edge=512)

    service.extract_document(pages(1, size=(2000, 1000)))

    data = transport.calls[0]["payload"]["contents"][0]["parts"][1]["inlineData"]["data"]
    with Image.open(io.BytesIO(base64.b64decode(data))) as image:
        assert max(image.size) == 512


def test_missing_api_key_fails_clearly(pages, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    transport = FakeTransport(_response())
    service = GeminiExtractionService(model="m", http_post=transport)

    with pytest.raises(GeminiExtractionError, match="GEMINI_API_KEY is not set"):
        service.extract_document(pages(1))
    assert transport.calls == []


def test_unreadable_page_image_fails_the_document(tmp_path) -> None:
    broken = tmp_path / "page-001.png"
    broken.write_bytes(b"fake-png")
    service, _ = _service(FakeTransport(_response()))

    with pytest.raises(GeminiExtractionError, match="Page 1 image could not be read"):
        service.extract_document([(1, broken)])


# ---------------------------------------------------------------------------
# multi-page handling
# ---------------------------------------------------------------------------

def test_long_documents_are_chunked_and_merged_in_order(pages) -> None:
    transport = FakeTransport(
        _response(_payload("Invoice", [{"page": n, "text": f"t{n}"} for n in (1, 2, 3)], [_attribute("Total", "1", page=2)])),
        _response(_payload("Something Else", [{"page": n, "text": f"t{n}"} for n in (4, 5, 6)], [_attribute("Tax", "2", page=5)])),
        _response(_payload("Third", [{"page": 7, "text": "t7"}], [_attribute("Due Date", "3", page=7)])),
    )
    service, sleeps = _service(transport, pages_per_request=3, request_delay_seconds=0.5)

    result = service.extract_document(pages(7))

    assert [_page_labels(call) for call in transport.calls] == [
        ["Page 1", "Page 2", "Page 3"],
        ["Page 4", "Page 5", "Page 6"],
        ["Page 7"],
    ]
    assert result.document_type == "Invoice"  # taken from the first chunk
    assert sorted(result.page_text) == [1, 2, 3, 4, 5, 6, 7]
    assert [(a.label, a.page) for a in result.attributes] == [("Total", 2), ("Tax", 5), ("Due Date", 7)]
    assert sleeps == [0.5, 0.5]  # a delay between requests, none before the first
    assert '"Invoice"' in transport.calls[1]["payload"]["contents"][0]["parts"][-1]["text"]
    assert "Invoice" not in transport.calls[0]["payload"]["contents"][0]["parts"][-1]["text"]


def test_request_size_limit_lowers_the_chunk_size(pages) -> None:
    transport = FakeTransport(_response(_payload(attributes=[])))
    service, _ = _service(transport, pages_per_request=10)
    service.max_request_bytes = 1  # every page is "too big" to share a request

    service.extract_document(pages(3))

    assert [_page_labels(call) for call in transport.calls] == [["Page 1"], ["Page 2"], ["Page 3"]]


def test_max_tokens_splits_the_chunk_and_retries(pages) -> None:
    transport = FakeTransport(
        _response(finish_reason="MAX_TOKENS", payload={}),
        _response(_payload("Invoice", [{"page": 1, "text": "a"}, {"page": 2, "text": "b"}], [_attribute("A", "1", page=1)])),
        _response(_payload("Other", [{"page": 3, "text": "c"}, {"page": 4, "text": "d"}], [_attribute("B", "2", page=4)])),
    )
    service, _ = _service(transport)

    result = service.extract_document(pages(4))

    assert [_page_labels(call) for call in transport.calls] == [
        ["Page 1", "Page 2", "Page 3", "Page 4"],
        ["Page 1", "Page 2"],
        ["Page 3", "Page 4"],
    ]
    assert result.document_type == "Invoice"
    assert [a.label for a in result.attributes] == ["A", "B"]
    assert sorted(result.page_text) == [1, 2, 3, 4]


def test_max_tokens_on_a_single_page_fails(pages) -> None:
    transport = FakeTransport(_response(finish_reason="MAX_TOKENS", payload={}))
    service, _ = _service(transport)

    with pytest.raises(GeminiExtractionError, match="page 1"):
        service.extract_document(pages(1))
    assert len(transport.calls) == 1


def _system_prompt(call) -> str:
    return call["payload"]["systemInstruction"]["parts"][0]["text"]


def test_recitation_splits_the_chunk_so_other_pages_keep_their_transcription(pages) -> None:
    transport = FakeTransport(
        _response(finish_reason="RECITATION", payload={}),  # pages 1-4 together
        _response(_payload("Syllabus", [{"page": 1, "text": "a"}, {"page": 2, "text": "b"}], [_attribute("Course Code", "CSC 106", page=1)])),
        _response(finish_reason="RECITATION", payload={}),  # pages 3-4 together
        _response(_payload("Syllabus", [{"page": 3, "text": "c"}], [_attribute("Units", "3", page=3)])),
        _response(_payload("Syllabus", [{"page": 4, "text": "d"}], [_attribute("Term", "1st Semester", page=4)])),
    )
    service, _ = _service(transport)

    result = service.extract_document(pages(4))

    assert [_page_labels(call) for call in transport.calls] == [
        ["Page 1", "Page 2", "Page 3", "Page 4"],
        ["Page 1", "Page 2"],
        ["Page 3", "Page 4"],
        ["Page 3"],
        ["Page 4"],
    ]
    assert all("do NOT transcribe" not in _system_prompt(call) for call in transport.calls)
    assert [a.label for a in result.attributes] == ["Course Code", "Units", "Term"]
    assert sorted(result.page_text) == [1, 2, 3, 4]
    assert result.untranscribed_pages == []


def test_recitation_on_a_single_page_retries_without_a_transcription(pages) -> None:
    transport = FakeTransport(
        _response(finish_reason="RECITATION", payload={}),
        _response(_payload("Syllabus", [{"page": 1, "text": ""}], [_attribute("University Vision", "A globally engaged University", page=1)])),
    )
    service, _ = _service(transport, request_delay_seconds=0.25)

    result = service.extract_document(pages(1))

    assert len(transport.calls) == 2
    assert "do NOT transcribe" not in _system_prompt(transport.calls[0])
    assert "do NOT transcribe" in _system_prompt(transport.calls[1])
    assert _page_labels(transport.calls[1]) == ["Page 1"]
    assert [a.label for a in result.attributes] == ["University Vision"]
    assert result.untranscribed_pages == [1]


def test_the_first_recitation_retry_keeps_the_tables(pages) -> None:
    table = {"page": 1, "title": "Course Schedule", "rows": [["Week", "Topic"], ["1-3", "Introduction"]]}
    transport = FakeTransport(
        _response(finish_reason="RECITATION", payload={}),
        _response(_payload("Syllabus", [{"page": 1, "text": ""}], [_attribute("Course Code", "CSC 106")], tables=[table])),
    )
    service, _ = _service(transport)

    result = service.extract_document(pages(1))

    assert len(transport.calls) == 2
    retry_prompt = _system_prompt(transport.calls[1])
    assert "do NOT transcribe" in retry_prompt and "any tables" in retry_prompt
    assert "do not return any tables" not in retry_prompt
    assert [t.title for t in result.tables] == ["Course Schedule"]
    assert result.untranscribed_pages == [1]


def test_a_second_recitation_stop_falls_back_to_attributes_only(pages) -> None:
    transport = FakeTransport(
        _response(finish_reason="RECITATION", payload={}),
        _response(finish_reason="RECITATION", payload={}),
        _response(_payload("Syllabus", [{"page": 1, "text": ""}], [_attribute("Course Code", "CSC 106")])),
    )
    service, _ = _service(transport)

    result = service.extract_document(pages(1))

    assert len(transport.calls) == 3
    assert "do not return any tables" in _system_prompt(transport.calls[2])
    assert [a.label for a in result.attributes] == ["Course Code"]
    assert result.untranscribed_pages == [1]


def test_prompt_asks_for_the_body_of_letters_and_both_kinds_of_tables(pages) -> None:
    transport = FakeTransport(_response(_payload(attributes=[])))
    service, _ = _service(transport)

    service.extract_document(pages(1))

    call = transport.calls[0]
    prompt = call["payload"]["systemInstruction"]["parts"][0]["text"]
    for expected in (
        "letters, memos", "main body", "one item per line", "never summarized or paraphrased",
        "visible borders and lines", "merged cells", "conventional table", "form-like table",
        "never split them into independent attributes", "Do not decide from the number of rows or columns",
        "is not a table", "row_span and column_span", "column_widths and row_heights", "continues on the next page",
    ):
        assert expected in prompt
    assert "every table on these pages in the tables list" in call["payload"]["contents"][0]["parts"][-1]["text"]


def test_untranscribed_pages_send_their_attributes_to_review() -> None:
    extraction = DocumentExtraction(
        document_type="Syllabus",
        page_text={1: "", 2: "Units 3"},
        attributes=[ExtractedAttribute("Course Code", "CSC 106", 0.99, 1), ExtractedAttribute("Units", "3", 0.99, 2)],
        untranscribed_pages=[1],
    )

    payloads = build_attribute_payloads(extraction, {1: 11, 2: 12}, 1)

    assert [(p["confidence_score"], p["validation_status"]) for p in payloads] == [(0.5, "HITL"), (0.99, "APPROVED")]


def test_recitation_that_persists_without_a_transcription_fails_with_a_clear_reason(pages) -> None:
    transport = FakeTransport(_response(finish_reason="RECITATION", payload={}))
    service, _ = _service(transport)

    with pytest.raises(GeminiExtractionError, match=r"would not read page 1 \(RECITATION\)"):
        service.extract_document(pages(1))

    # normal request, then without a transcription (tables kept), then attributes only
    assert len(transport.calls) == 3


def test_recitation_retry_respects_cancellation(pages) -> None:
    transport = FakeTransport(_response(finish_reason="RECITATION", payload={}))
    service, _ = _service(transport)
    answers = iter([False, True])

    with pytest.raises(ExtractionCanceled):
        service.extract_document(pages(1), should_cancel=lambda: next(answers))

    assert len(transport.calls) == 1


def test_a_failing_chunk_fails_the_whole_document(pages) -> None:
    transport = FakeTransport(_response(_payload(attributes=[])), _response(status=403))
    service, _ = _service(transport, pages_per_request=1)

    with pytest.raises(GeminiExtractionError, match="HTTP 403"):
        service.extract_document(pages(2))


def test_cancellation_is_checked_before_every_request(pages) -> None:
    transport = FakeTransport(_response(_payload(attributes=[])))
    service, _ = _service(transport, pages_per_request=1)

    with pytest.raises(ExtractionCanceled):
        service.extract_document(pages(2), should_cancel=lambda: True)
    assert transport.calls == []

    answers = iter([False, True])
    transport = FakeTransport(_response(_payload(attributes=[])))
    service, _ = _service(transport, pages_per_request=1)
    with pytest.raises(ExtractionCanceled):
        service.extract_document(pages(2), should_cancel=lambda: next(answers))
    assert len(transport.calls) == 1


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_fail_immediately_without_retry(pages, status) -> None:
    transport = FakeTransport(_response(status=status, body={"error": RESPONSE_BODY_MARKER}))
    service, sleeps = _service(transport)

    with pytest.raises(GeminiExtractionError, match=f"HTTP {status}") as excinfo:
        service.extract_document(pages(1))

    assert len(transport.calls) == 1 and sleeps == []
    assert RESPONSE_BODY_MARKER not in str(excinfo.value)
    assert API_KEY not in str(excinfo.value)


def test_rate_limit_is_retried_honouring_retry_after(pages) -> None:
    transport = FakeTransport(_response(status=429, headers={"Retry-After": "2"}), _response(_payload(attributes=[])))
    service, sleeps = _service(transport)

    result = service.extract_document(pages(1))

    assert len(transport.calls) == 2
    assert sleeps == [2.0]
    assert result.attributes == []


def test_retry_after_is_capped_and_backoff_is_exponential(pages) -> None:
    transport = FakeTransport(
        _response(status=429, headers={"Retry-After": "9999"}),
        _response(status=503),
        _response(status=500),
        _response(_payload(attributes=[])),
    )
    service, sleeps = _service(transport)

    service.extract_document(pages(1))

    assert sleeps == [60.0, 4.0, 8.0]


def test_server_errors_give_up_after_five_retries(pages) -> None:
    transport = FakeTransport(_response(status=503, body={"error": RESPONSE_BODY_MARKER}))
    service, sleeps = _service(transport)

    with pytest.raises(GeminiExtractionError, match="HTTP 503") as excinfo:
        service.extract_document(pages(1))

    assert len(transport.calls) == 6  # first attempt + 5 retries
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 30.0]
    assert RESPONSE_BODY_MARKER not in str(excinfo.value)


def test_the_number_of_retries_is_configurable(pages, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")
    transport = FakeTransport(_response(status=503))
    service, sleeps = _service(transport)
    with pytest.raises(GeminiExtractionError, match="HTTP 503"):
        service.extract_document(pages(1))
    assert len(transport.calls) == 2 and sleeps == [2.0]

    monkeypatch.setenv("GEMINI_MAX_RETRIES", "0")
    transport = FakeTransport(_response(status=503))
    service, sleeps = _service(transport)
    with pytest.raises(GeminiExtractionError):
        service.extract_document(pages(1))
    assert len(transport.calls) == 1 and sleeps == []


def test_a_persistent_server_error_on_a_multi_page_request_splits_it(pages) -> None:
    ok_first = _response(_payload("Syllabus", [{"page": 1, "text": "a"}, {"page": 2, "text": "b"}], [_attribute("A", "1", page=1)]))
    ok_second = _response(_payload("Syllabus", [{"page": 3, "text": "c"}, {"page": 4, "text": "d"}], [_attribute("B", "2", page=4)]))
    transport = FakeTransport(*([_response(status=503)] * 6), ok_first, ok_second)
    service, sleeps = _service(transport)

    result = service.extract_document(pages(4))

    assert [_page_labels(call) for call in transport.calls] == (
        [["Page 1", "Page 2", "Page 3", "Page 4"]] * 6 + [["Page 1", "Page 2"], ["Page 3", "Page 4"]]
    )
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 30.0]
    assert [a.label for a in result.attributes] == ["A", "B"]
    assert sorted(result.page_text) == [1, 2, 3, 4]


def test_splitting_stops_at_a_single_page_and_the_error_names_the_status(pages) -> None:
    transport = FakeTransport(_response(status=503))
    service, _ = _service(transport)

    with pytest.raises(GeminiServiceError, match="HTTP 503"):
        service.extract_document(pages(2))

    assert [_page_labels(call) for call in transport.calls] == [["Page 1", "Page 2"]] * 6 + [["Page 1"]] * 6


def test_a_rate_limit_is_not_split_because_smaller_requests_do_not_help(pages) -> None:
    transport = FakeTransport(_response(status=429))
    service, _ = _service(transport)

    with pytest.raises(GeminiExtractionError, match="HTTP 429") as excinfo:
        service.extract_document(pages(2))

    assert not isinstance(excinfo.value, GeminiServiceError)
    assert [_page_labels(call) for call in transport.calls] == [["Page 1", "Page 2"]] * 6


def test_rate_limit_that_never_clears_reports_it(pages) -> None:
    transport = FakeTransport(_response(status=429))
    service, _ = _service(transport)

    with pytest.raises(GeminiExtractionError, match=r"HTTP 429 \(rate limit"):
        service.extract_document(pages(1))


def _rate_limited(retry_delay="34s", quota_id="GenerateRequestsPerMinutePerProjectPerModel-FreeTier", message="Quota exceeded"):
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]}]
    if retry_delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    return _response(status=429, body={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": message, "details": details}})


def test_the_suggested_retry_delay_is_honoured_and_later_requests_are_spaced_out(pages) -> None:
    transport = FakeTransport(_rate_limited("34s"), _response(_payload(attributes=[])))
    service, sleeps = _service(transport, pages_per_request=1)

    service.extract_document(pages(2))

    assert len(transport.calls) == 3  # page 1 (limited), page 1 again, page 2
    assert sleeps == [34.0, 10.0]  # the API's own wait, then a pause before the next request


def test_a_daily_quota_fails_at_once_instead_of_burning_retries(pages) -> None:
    transport = FakeTransport(_rate_limited("5s", quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    service, sleeps = _service(transport)

    with pytest.raises(GeminiExtractionError, match=r"HTTP 429 \(daily quota used up") as excinfo:
        service.extract_document(pages(1))

    assert len(transport.calls) == 1 and sleeps == []
    assert not isinstance(excinfo.value, GeminiServiceError)


def test_a_per_minute_token_limit_is_retried(pages) -> None:
    transport = FakeTransport(
        _rate_limited("12s", quota_id="GenerateContentInputTokensPerModelPerMinute-FreeTier"),
        _response(_payload(attributes=[])),
    )
    service, sleeps = _service(transport)

    service.extract_document(pages(1))

    assert sleeps == [12.0]


def test_the_retry_delay_is_capped_and_never_shorter_than_the_backoff(pages) -> None:
    transport = FakeTransport(_rate_limited("9999s"), _rate_limited("1s"), _rate_limited("1s"), _response(_payload(attributes=[])))
    service, sleeps = _service(transport)

    service.extract_document(pages(1))

    assert sleeps == [120.0, 4.0, 8.0]  # the cap; then the backoff, which is longer than the 1s the API asked for


@pytest.mark.parametrize("retry_delay", ["soon", "-5s", "5", "", None, 7])
def test_an_unreadable_retry_delay_falls_back_to_the_backoff(pages, retry_delay) -> None:
    response = _rate_limited(retry_delay) if isinstance(retry_delay, (str, type(None))) else _response(
        status=429, body={"error": {"details": [{"retryDelay": retry_delay}]}}
    )
    transport = FakeTransport(response, _response(_payload(attributes=[])))
    service, sleeps = _service(transport)

    service.extract_document(pages(1))

    assert sleeps == [2.0]


def test_a_429_whose_body_is_not_json_still_retries(pages) -> None:
    def broken():
        raise ValueError("html error page")

    transport = FakeTransport(SimpleNamespace(status_code=429, json=broken, headers={}), _response(_payload(attributes=[])))
    service, sleeps = _service(transport)

    service.extract_document(pages(1))

    assert sleeps == [2.0]


def test_a_rate_limit_message_is_never_echoed(pages) -> None:
    transport = FakeTransport(_rate_limited("1s", message=f"{RESPONSE_BODY_MARKER} {API_KEY}"))
    service, _ = _service(transport)

    with pytest.raises(GeminiExtractionError, match=r"HTTP 429") as excinfo:
        service.extract_document(pages(1))

    assert RESPONSE_BODY_MARKER not in str(excinfo.value) and API_KEY not in str(excinfo.value)


def test_transport_error_is_retried_once_then_reports_type_only(pages) -> None:
    class ReadTimeout(Exception):
        pass

    transport = FakeTransport(ReadTimeout(f"timed out talking to {API_KEY}"), _response(_payload(attributes=[])))
    service, _ = _service(transport)
    assert service.extract_document(pages(1)).attributes == []
    assert len(transport.calls) == 2

    transport = FakeTransport(ReadTimeout(f"timed out talking to {API_KEY}"))
    service, _ = _service(transport)
    with pytest.raises(GeminiExtractionError, match="Gemini request failed: ReadTimeout") as excinfo:
        service.extract_document(pages(1))
    assert len(transport.calls) == 2
    assert API_KEY not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


def test_blocked_prompt_fails(pages) -> None:
    body = {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}
    service, _ = _service(FakeTransport(_response(body=body)))
    with pytest.raises(GeminiExtractionError, match=r"blocked the request \(SAFETY\)"):
        service.extract_document(pages(1))


def test_empty_candidates_fail(pages) -> None:
    service, _ = _service(FakeTransport(_response(body={"candidates": []})))
    with pytest.raises(GeminiExtractionError, match="no candidates"):
        service.extract_document(pages(1))


@pytest.mark.parametrize("reason", ["SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"])
def test_unfinished_candidates_fail(pages, reason) -> None:
    service, _ = _service(FakeTransport(_response(finish_reason=reason, payload=_payload())))
    with pytest.raises(GeminiExtractionError, match=reason):
        service.extract_document(pages(1))


def test_json_wrapped_in_a_markdown_fence_or_stray_text_is_still_read(pages) -> None:
    body = json.dumps(_payload())
    for wrapped in (f"```json\n{body}\n```", f"Here is the result:\n{body}\nHope that helps.", f"  {body}  "):
        service, _ = _service(FakeTransport(_response(raw_text=wrapped)))
        assert service.extract_document(pages(1)).attributes[0].label == "Student Name"


def test_a_cut_off_answer_on_a_multi_page_request_splits_it(pages) -> None:
    truncated = json.dumps(_payload(pages=[{"page": 1, "text": "a"}]))[:-9]  # ends in the middle of the object
    transport = FakeTransport(
        _response(raw_text=truncated),
        _response(_payload("Syllabus", [{"page": 1, "text": "a"}], [_attribute("A", "1", page=1)])),
        _response(_payload("Syllabus", [{"page": 2, "text": "b"}, {"page": 3, "text": "c"}], [_attribute("B", "2", page=3)])),
    )
    service, _ = _service(transport)

    result = service.extract_document(pages(3))

    assert [_page_labels(call) for call in transport.calls] == [["Page 1", "Page 2", "Page 3"], ["Page 1"], ["Page 2", "Page 3"]]
    assert all("do NOT transcribe" not in _system_prompt(call) for call in transport.calls)  # halves keep the full answer
    assert [a.label for a in result.attributes] == ["A", "B"]
    assert sorted(result.page_text) == [1, 2, 3]


def test_an_unreadable_answer_on_one_page_retries_with_a_shorter_answer(pages) -> None:
    transport = FakeTransport(
        _response(raw_text="{\"document_type\": \"x\", \"pages\": [{\"page\": 1, \"text\": \"cut"),
        _response(raw_text="{\"document_type\": "),
        _response(_payload("Syllabus", [{"page": 1, "text": ""}], [_attribute("Course Code", "CSC 106")])),
    )
    service, _ = _service(transport)

    result = service.extract_document(pages(1))

    prompts = [_system_prompt(call) for call in transport.calls]
    assert "do NOT transcribe" not in prompts[0]
    assert "do NOT transcribe" in prompts[1] and "any tables" in prompts[1] and "do not return any tables" not in prompts[1]
    assert "do not return any tables" in prompts[2]
    assert [a.label for a in result.attributes] == ["Course Code"]
    assert result.untranscribed_pages == [1]


def test_a_page_that_never_yields_readable_json_fails_naming_the_page(pages, caplog) -> None:
    transport = FakeTransport(_response(raw_text=f"{RESPONSE_BODY_MARKER} {{ not json"))
    service, _ = _service(transport)

    with caplog.at_level("WARNING"):
        with pytest.raises(GeminiExtractionError, match=r"not valid JSON \(page 1") as excinfo:
            service.extract_document(pages(1))

    assert len(transport.calls) == 3  # full answer, no transcription, attributes only
    assert RESPONSE_BODY_MARKER not in str(excinfo.value)
    assert RESPONSE_BODY_MARKER not in caplog.text  # only the parser's position is logged
    assert "not valid JSON (finish reason" in caplog.text


def test_an_http_body_that_is_not_json_is_handled_the_same_way(pages) -> None:
    def broken():
        raise ValueError("no json here")

    response = SimpleNamespace(status_code=200, json=broken, headers={})
    transport = FakeTransport(response)
    service, _ = _service(transport)

    with pytest.raises(GeminiExtractionError, match="not valid JSON"):
        service.extract_document(pages(1))

    assert len(transport.calls) == 3


def test_malformed_json_fails_without_echoing_the_text(pages) -> None:
    service, _ = _service(FakeTransport(_response(raw_text=f"{RESPONSE_BODY_MARKER} {{not json")))
    with pytest.raises(GeminiExtractionError, match="not valid JSON") as excinfo:
        service.extract_document(pages(1))
    assert RESPONSE_BODY_MARKER not in str(excinfo.value)


@pytest.mark.parametrize(
    "payload",
    [
        [],  # not an object
        {"pages": [], "attributes": []},  # no document_type
        {"document_type": "x", "pages": [], "attributes": "nope"},
        {"document_type": "x", "pages": [{"page": "one", "text": "t"}], "attributes": []},
        {"document_type": "x", "pages": [], "attributes": [{"attribute": "A", "value": "v", "confidence": 0.9}]},  # no page
        {"document_type": "x", "pages": [], "attributes": [{"attribute": "A", "value": "v", "confidence": "high", "page": 1}]},
        {"document_type": "x", "pages": [], "attributes": [{"attribute": 5, "value": "v", "confidence": 0.9, "page": 1}]},
        {"document_type": "x", "pages": [], "attributes": [_attribute(continues_from_previous_page="yes")]},
    ],
)
def test_responses_that_do_not_match_the_schema_fail(pages, payload) -> None:
    service, _ = _service(FakeTransport(_response(payload)))
    with pytest.raises(GeminiExtractionError, match="expected structure"):
        service.extract_document(pages(1))


def test_empty_content_fails(pages) -> None:
    service, _ = _service(FakeTransport(_response(raw_text="  ")))
    with pytest.raises(GeminiExtractionError, match="no content"):
        service.extract_document(pages(1))


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_box_2d_is_converted_to_x0_y0_x1_y1_and_clamped(pages) -> None:
    attributes = [
        _attribute("A", "1", box_2d=[100, 200, 300, 400]),  # [ymin, xmin, ymax, xmax]
        _attribute("B", "2", box_2d=[-5, -10, 1200, 1500]),
        _attribute("C", "3", box_2d=[10, 10, 10, 50]),  # zero height
        _attribute("D", "4", box_2d=[1, 2, 3]),  # wrong length
        _attribute("E", "5", box_2d=["a", 1, 2, 3]),
        _attribute("F", "6"),
        _attribute("G", "7", box_2d=[300, 400, 100, 200]),  # reversed corners
    ]
    service, _ = _service(FakeTransport(_response(_payload(attributes=attributes))))

    result = service.extract_document(pages(1))

    assert [a.box for a in result.attributes] == [[200, 100, 400, 300], [0, 0, 1000, 1000], None, None, None, None, [200, 100, 400, 300]]


def test_thought_parts_are_ignored_and_optional_fields_default(pages) -> None:
    body = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": "thinking...", "thought": True}, {"text": json.dumps(_payload(attributes=[_attribute()]))}]},
            }
        ]
    }
    service, _ = _service(FakeTransport(_response(body=body)))

    result = service.extract_document(pages(1))

    assert result.attributes[0].continues_from_previous_page is False
    assert result.tables == []




# ---------------------------------------------------------------------------
# conversion to staged payloads
# ---------------------------------------------------------------------------

PAGE_IDS = {1: 101, 2: 102, 3: 103}


def _extraction(attributes, page_text=None, document_type="Student Registration Form") -> DocumentExtraction:
    return DocumentExtraction(
        document_type=document_type,
        page_text=page_text if page_text is not None else {1: "Student Name: John Doe\nTotal 1,200.00", 2: "Total 5"},
        attributes=attributes,
    )


def test_payload_shape_status_threshold_and_box() -> None:
    extraction = _extraction(
        [
            ExtractedAttribute("Student Name", "John Doe", 0.98, 1, box=[10, 20, 30, 40]),
            ExtractedAttribute("Total Amount", "1,200.00", 0.75, 1),
            ExtractedAttribute("Maybe", "John", 0.74, 1),
        ]
    )

    payloads = build_attribute_payloads(extraction, PAGE_IDS, document_id=9)

    assert payloads[0] == {
        "page_id": 101,
        "page_number": 1,
        "document_id": 9,
        "entity_type_label": "Student Name",
        "extracted_value": "John Doe",
        "bounding_boxes": [[10, 20, 30, 40]],
        "confidence_score": 0.98,
        "validation_status": "APPROVED",
        "document_type": "student_registration_form",
        "continues_from_previous_page": False,
    }
    assert payloads[1]["validation_status"] == "APPROVED" and payloads[1]["bounding_boxes"] == []
    assert payloads[2]["validation_status"] == "HITL"


def test_value_missing_from_the_page_transcription_is_capped_at_half() -> None:
    extraction = _extraction(
        [
            ExtractedAttribute("Student Name", "John Doe", 0.99, 1),
            ExtractedAttribute("Course", "Computer Science", 0.99, 1),  # not on the page
            ExtractedAttribute("Total", "5", 0.99, 1),  # on page 2 only
            ExtractedAttribute("Low", "Jane", 0.3, 1),  # already below the cap
        ]
    )

    by_label = {p["entity_type_label"]: p for p in build_attribute_payloads(extraction, PAGE_IDS, 1)}

    assert by_label["Student Name"]["confidence_score"] == 0.99
    assert by_label["Course"]["confidence_score"] == 0.5 and by_label["Course"]["validation_status"] == "HITL"
    assert by_label["Total"]["confidence_score"] == 0.5
    assert by_label["Low"]["confidence_score"] == 0.3


def test_grounding_ignores_case_spacing_and_punctuation() -> None:
    extraction = _extraction([ExtractedAttribute("Amount", "1200.00", 0.9, 1), ExtractedAttribute("Name", "JOHN  DOE", 0.9, 1)])

    payloads = build_attribute_payloads(extraction, PAGE_IDS, 1)

    assert [p["confidence_score"] for p in payloads] == [0.9, 0.9]


def test_page_without_a_transcription_sends_everything_to_review() -> None:
    extraction = _extraction([ExtractedAttribute("Name", "John Doe", 0.99, 1)], page_text={})

    assert build_attribute_payloads(extraction, PAGE_IDS, 1)[0]["confidence_score"] == 0.5


def test_empty_and_out_of_range_attributes_are_dropped_and_values_trimmed() -> None:
    extraction = _extraction(
        [
            ExtractedAttribute("Name", "   ", 0.9, 1),
            ExtractedAttribute("   ", "John Doe", 0.9, 1),
            ExtractedAttribute("Name", "John Doe", 0.9, 0),
            ExtractedAttribute("Name", "John Doe", 0.9, 4),
            ExtractedAttribute("  Student   Name ", "  John Doe  ", 0.9, 1),
        ]
    )

    payloads = build_attribute_payloads(extraction, PAGE_IDS, 1)

    assert len(payloads) == 1
    assert payloads[0]["entity_type_label"] == "Student Name"
    assert payloads[0]["extracted_value"] == "John Doe"


def test_a_page_with_no_text_and_no_attributes_is_not_an_error() -> None:
    assert build_attribute_payloads(_extraction([], page_text={1: ""}), PAGE_IDS, 1) == []


def test_confidence_is_clamped_and_labels_truncated() -> None:
    extraction = _extraction(
        [
            ExtractedAttribute("Over", "John", 1.7, 1),
            ExtractedAttribute("Under", "Doe", -2, 1),
            ExtractedAttribute("L" * 300, "John Doe", 0.9, 1),
        ]
    )

    payloads = build_attribute_payloads(extraction, PAGE_IDS, 1)

    assert payloads[0]["confidence_score"] == 1.0
    assert payloads[1]["confidence_score"] == 0.0
    assert len(payloads[2]["entity_type_label"]) == 128


def test_duplicate_label_and_value_on_a_page_keeps_the_higher_confidence() -> None:
    extraction = _extraction(
        [
            ExtractedAttribute("Student Name", "John Doe", 0.6, 1),
            ExtractedAttribute("student name", "john  doe", 0.95, 1),
            ExtractedAttribute("Student Name", "John Doe", 0.8, 1, box=[1, 1, 5, 5]),
        ]
    )

    payloads = build_attribute_payloads(extraction, PAGE_IDS, 1)

    assert len(payloads) == 1
    assert payloads[0]["confidence_score"] == 0.95
    # the same label and value on a different page is not a duplicate
    extraction = _extraction(
        [ExtractedAttribute("Total", "5", 0.9, 2), ExtractedAttribute("Total", "5", 0.9, 1)],
        page_text={1: "Total 5", 2: "Total 5"},
    )
    assert len(build_attribute_payloads(extraction, PAGE_IDS, 1)) == 2


def test_continuation_flag_is_carried_but_never_set_on_the_first_page() -> None:
    extraction = _extraction(
        [
            ExtractedAttribute("Notes", "John", 0.9, 1, continues_from_previous_page=True),
            ExtractedAttribute("Notes", "Doe", 0.9, 2, continues_from_previous_page=True),
        ],
        page_text={1: "John", 2: "Doe"},
    )

    payloads = build_attribute_payloads(extraction, PAGE_IDS, 1)

    assert [p["continues_from_previous_page"] for p in payloads] == [False, True]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Student Registration Form", "student_registration_form"),
        ("  Invoice / Receipt (2024)  ", "invoice_receipt_2024"),
        ("", "unknown"),
        (None, "unknown"),
        ("!!!", "unknown"),
        ("x" * 100, "x" * 64),
        ("a" * 63 + " b", "a" * 63),
    ],
)
def test_document_type_is_normalized(raw, expected) -> None:
    assert normalize_document_type(raw) == expected


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# tables: conventional tables and form-like layouts with merged cells
# ---------------------------------------------------------------------------

def _table_item(**overrides) -> dict[str, Any]:
    item: dict[str, Any] = {
        "page": 1,
        "title": "Subjects",
        "table_type": "data",
        "row_count": 2,
        "column_count": 2,
        "cells": [
            {"row": 0, "column": 0, "text": "Code"},
            {"row": 0, "column": 1, "text": "Title"},
            {"row": 1, "column": 0, "text": "IT101"},
            {"row": 1, "column": 1, "text": "Intro"},
        ],
    }
    item.update(overrides)
    return item


def _read_tables(pages, service_factory, *items):
    transport = FakeTransport(_response(_payload(attributes=[], tables=list(items))))
    service, _ = service_factory(transport)
    return service.extract_document(pages(1)).tables


FORM_CELLS = [
    {"row": 0, "column": 0, "column_span": 2, "text": "University Vision and Mission", "is_header": True},
    {"row": 1, "column": 0, "text": "Vision", "is_header": True},
    {"row": 1, "column": 1, "text": "A globally engaged university."},
    {"row": 2, "column": 0, "row_span": 2, "text": "Goals\nResearch. Instruction. Production."},
    {"row": 2, "column": 1, "text": "Objectives"},
    {"row": 3, "column": 1, "text": "1. To train students.\n2. To provide skills."},
]


def test_a_form_like_table_keeps_merged_cells_headings_and_long_text_in_one_table(pages) -> None:
    item = _table_item(title="Vision, Mission and Goals", table_type="form", row_count=4, column_count=2, cells=FORM_CELLS)

    (table,) = _read_tables(pages, _service, item)

    assert (table.kind, table.row_count, table.column_count, table.bordered) == ("form", 4, 2, True)
    banner = table.cells[0]
    assert (banner.row, banner.column, banner.column_span, banner.row_span, banner.is_header) == (0, 0, 2, 1, True)
    goals = next(cell for cell in table.cells if cell.text.startswith("Goals"))
    assert (goals.row_span, goals.column_span) == (2, 1)
    assert "\n" in table.cells[-1].text  # line breaks inside a cell are kept


def test_merged_cells_become_spans_and_covered_slots_are_not_repeated() -> None:
    table = ExtractedTable(page=1, cells=[_cell(c) for c in FORM_CELLS], row_count=4, column_count=2, kind="form")

    built = build_table_layouts(DocumentExtraction(tables=[table]), {1: 11})[1][0]

    assert built["table_type"] == "form"
    rows = built["rows"]
    assert [len(row["cells"]) for row in rows] == [1, 2, 2, 1]  # banner spans both columns; goals spans two rows
    banner = rows[0]["cells"][0]
    assert (banner["rowspan"], banner["colspan"], banner["is_header"], banner["bold"]) == (1, 2, True, True)
    goals = rows[2]["cells"][0]
    assert (goals["rowspan"], goals["colspan"]) == (2, 1)
    assert rows[3]["cells"][0]["column_index"] == 1  # row 3 only has the slot the merged cell does not cover
    assert built["style"] == {"bordered": True, "border_width": 1, "layout_source": "gemini"}
    assert built["source"] == "gemini"


def _cell(raw: dict[str, Any]) -> ExtractedCell:
    return ExtractedCell(
        row=raw["row"],
        column=raw["column"],
        text=raw["text"],
        row_span=raw.get("row_span", 1),
        column_span=raw.get("column_span", 1),
        is_header=raw.get("is_header", False),
    )


def test_missing_slots_are_filled_with_blank_cells_so_every_row_is_complete() -> None:
    table = ExtractedTable(page=1, cells=[ExtractedCell(0, 0, "a"), ExtractedCell(1, 1, "d")], row_count=2, column_count=3)

    built = build_table_layouts(DocumentExtraction(tables=[table]), {1: 1})[1][0]

    assert [[cell["text"] for cell in row["cells"]] for row in built["rows"]] == [["a", "", ""], ["", "d", ""]]
    assert built["columns"] == [333.33, 333.33, 333.33]


def test_an_overlapping_cell_is_dropped_and_the_first_one_wins() -> None:
    table = ExtractedTable(
        page=1,
        cells=[ExtractedCell(0, 0, "wide", column_span=2), ExtractedCell(0, 1, "overlap"), ExtractedCell(1, 0, "x"), ExtractedCell(1, 1, "y")],
        row_count=2,
        column_count=2,
    )

    built = build_table_layouts(DocumentExtraction(tables=[table]), {1: 1})[1][0]

    assert [[cell["text"] for cell in row["cells"]] for row in built["rows"]] == [["wide"], ["x", "y"]]


def test_the_grid_grows_to_fit_the_cells_and_spans_are_clipped_by_the_declared_size(pages) -> None:
    # declared 1x1 but cells reach row 2 / column 1: the grid is at least as large as the cells need
    item = _table_item(row_count=1, column_count=1, cells=[{"row": 2, "column": 1, "text": "far", "column_span": 2}, {"row": 0, "column": 0, "text": "near"}])

    (table,) = _read_tables(pages, _service, item)

    assert (table.row_count, table.column_count) == (3, 3)
    built = build_table_layouts(DocumentExtraction(tables=[table]), {1: 1})[1][0]
    assert [len(row["cells"]) for row in built["rows"]] == [3, 3, 2]


def test_relative_widths_and_heights_set_the_columns_and_cell_boxes() -> None:
    table = ExtractedTable(
        page=1,
        cells=[ExtractedCell(0, 0, "label"), ExtractedCell(0, 1, "paragraph"), ExtractedCell(1, 0, "x", column_span=2)],
        row_count=2,
        column_count=2,
        box=[0, 100, 1000, 300],
        column_widths=[1, 3],
        row_heights=[3, 1],
    )

    built = build_table_layouts(DocumentExtraction(tables=[table]), {1: 1})[1][0]

    assert built["columns"] == [250.0, 750.0]
    assert [row["height"] for row in built["rows"]] == [150.0, 50.0]
    assert built["rows"][0]["cells"][1]["bbox"] == [250, 100, 1000, 250]
    assert built["rows"][1]["cells"][0]["bbox"] == [0, 250, 1000, 300]  # a merged cell covers both columns


@pytest.mark.parametrize("widths", [[1], [1, 2, 3], [1, 0], [1, -2], ["a", 1], "wide", None])
def test_unusable_widths_fall_back_to_even_columns(pages, widths) -> None:
    (table,) = _read_tables(pages, _service, _table_item(column_widths=widths))

    assert table.column_widths is None
    built = build_table_layouts(DocumentExtraction(tables=[table]), {1: 1})[1][0]
    assert built["columns"] == [500.0, 500.0]


def test_headers_default_to_the_first_row_only_for_conventional_tables_without_flags(pages) -> None:
    data, form, flagged = _read_tables(
        pages,
        _service,
        _table_item(),
        _table_item(table_type="form"),
        _table_item(cells=[{"row": 0, "column": 0, "text": "a"}, {"row": 1, "column": 0, "text": "b", "is_header": True}]),
    )

    assert [cell.is_header for cell in data.cells] == [True, True, False, False]
    assert [cell.is_header for cell in form.cells] == [False] * 4
    assert [cell.is_header for cell in flagged.cells] == [False, True]  # the model's own flags win


def test_table_kind_and_borders_are_read_with_safe_defaults(pages) -> None:
    unknown, upper, borderless = _read_tables(
        pages, _service, _table_item(table_type="spreadsheet"), _table_item(table_type=" FORM "), _table_item(bordered=False)
    )

    assert (unknown.kind, upper.kind, unknown.bordered) == ("data", "form", True)
    assert borderless.bordered is False
    built = build_table_layouts(DocumentExtraction(tables=[borderless]), {1: 1})[1][0]
    assert built["style"]["bordered"] is False and built["style"]["border_width"] == 0


def test_the_older_rows_of_strings_shape_is_still_accepted(pages) -> None:
    legacy = {"page": 1, "title": "Ok", "rows": [["a", None], ["b", "c"]]}

    (table,) = _read_tables(pages, _service, legacy)

    assert [(c.row, c.column, c.text, c.is_header) for c in table.cells] == [(0, 0, "a", True), (0, 1, "", True), (1, 0, "b", False), (1, 1, "c", False)]
    assert (table.row_count, table.column_count) == (2, 2)


def test_malformed_tables_are_skipped_without_failing_the_page(pages) -> None:
    bad = [
        {"page": 1, "rows": "nope"},
        {"rows": [["a"]]},  # no page
        _table_item(cells=[]),
        _table_item(cells=["not a cell"]),
        _table_item(cells=[{"row": -1, "column": 0, "text": "x"}]),
        _table_item(cells=[{"row": "one", "column": 0, "text": "x"}]),
        _table_item(cells=[{"row": 0, "column": 0, "text": "x", "row_span": 100000}]),
        "text",
    ]

    tables = _read_tables(pages, _service, *bad, _table_item(title="Kept"))

    assert [t.title for t in tables] == ["Kept"]


def test_tables_with_no_text_or_on_unknown_pages_produce_no_card() -> None:
    blank = ExtractedTable.from_rows(1, [["", " "], ["", ""]])
    elsewhere = ExtractedTable.from_rows(9, [["a", "b"]])

    assert build_table_layouts(DocumentExtraction(tables=[blank, elsewhere]), {1: 1}) == {}


def test_each_page_numbers_its_own_tables_and_titles_default_sensibly() -> None:
    extraction = DocumentExtraction(
        tables=[
            ExtractedTable.from_rows(1, [["a", "b"]], title="Named"),
            ExtractedTable.from_rows(1, [["c", "d"]]),
            ExtractedTable.from_rows(2, [["e", "f"]]),
        ]
    )

    layouts = build_table_layouts(extraction, {1: 11, 2: 12})

    assert [(t["id"], t["title"], t["page_id"]) for t in layouts[1]] == [("page-1-table-1", "Named", 11), ("page-1-table-2", "Table 2", 11)]
    assert layouts[2][0]["id"] == "page-2-table-1"


# ---------------------------------------------------------------------------
# table contents are not repeated as independent text fields
# ---------------------------------------------------------------------------

def _grounded(*attributes, tables):
    texts = " ".join(a.value for a in attributes)
    return DocumentExtraction(document_type="Syllabus", page_text={1: texts, 2: texts}, attributes=list(attributes), tables=tables)


def test_an_attribute_that_sits_inside_a_table_cell_stays_in_the_table() -> None:
    table = ExtractedTable.from_rows(1, [["Course Code: CSC 106", "SOFTWARE ENGINEERING 1"]], kind="form")
    extraction = _grounded(
        ExtractedAttribute("Course Code", "CSC 106", 0.99, 1),
        ExtractedAttribute("Descriptive Title", "Software Engineering 1", 0.99, 1),
        ExtractedAttribute("Prepared By", "Department Chair", 0.99, 1),
        tables=[table],
    )

    labels = [p["entity_type_label"] for p in build_attribute_payloads(extraction, {1: 1, 2: 2}, 1)]

    assert labels == ["Prepared By"]


def test_short_values_other_pages_and_regions_outside_the_table_are_kept() -> None:
    table = ExtractedTable.from_rows(1, [["Total units: 3", "Course Code: CSC 106"]], kind="form", box=[100, 100, 900, 300])
    extraction = _grounded(
        ExtractedAttribute("Units", "3", 0.99, 1),  # too short to match safely
        ExtractedAttribute("Course Code", "CSC 106", 0.99, 2),  # same text, but a different page
        ExtractedAttribute("Reference", "CSC 106", 0.99, 1, box=[100, 600, 300, 700]),  # located below the table
        ExtractedAttribute("Inside", "CSC 106", 0.99, 1, box=[150, 150, 400, 250]),  # located inside the table
        tables=[table],
    )

    labels = [p["entity_type_label"] for p in build_attribute_payloads(extraction, {1: 1, 2: 2}, 1)]

    assert labels == ["Units", "Course Code", "Reference"]


def test_without_tables_attributes_are_unaffected() -> None:
    extraction = _grounded(ExtractedAttribute("Course Code", "CSC 106", 0.99, 1), tables=[])

    assert len(build_attribute_payloads(extraction, {1: 1, 2: 2}, 1)) == 1


def test_table_boxes_use_gemini_box_2d_order(pages) -> None:
    (table,) = _read_tables(pages, _service, _table_item(box_2d=[100, 200, 300, 400]))

    assert table.box == [200, 100, 400, 300]


# ---------------------------------------------------------------------------
# long documents: resume, progress, adaptive request size, page numbers, context
# ---------------------------------------------------------------------------

def _truncated():
    return _response(finish_reason="MAX_TOKENS", payload={})


def _refused():
    return _response(finish_reason="RECITATION", payload={})


def _task(call) -> str:
    return call["payload"]["contents"][0]["parts"][-1]["text"]


def _pages_reply(numbers, label_prefix="Attr", doc_type="Syllabus", text_suffix=""):
    return _response(
        _payload(
            doc_type,
            [{"page": n, "text": f"text of page {n}{text_suffix}"} for n in numbers],
            [_attribute(f"{label_prefix} {n}", f"value {n}", page=n) for n in numbers],
        )
    )


def _saved(*numbers, doc_type="Syllabus") -> DocumentExtraction:
    return DocumentExtraction(
        document_type=doc_type,
        page_text={n: f"text of page {n}" for n in numbers},
        attributes=[ExtractedAttribute(f"Saved {n}", f"saved {n}", 0.9, n) for n in numbers],
        read_pages=list(numbers),
    )


def test_a_resumed_run_only_sends_the_pages_that_were_not_read(pages) -> None:
    transport = FakeTransport(_pages_reply([3, 4, 5, 6]))
    service, _ = _service(transport)

    result = service.extract_document(pages(6), resume=_saved(1, 2))

    assert [_page_labels(call) for call in transport.calls] == [["Page 3", "Page 4", "Page 5", "Page 6"]]
    assert 'identified as "Syllabus"' in _task(transport.calls[0])  # the earlier reading's document type carries on
    assert result.read_pages == [1, 2, 3, 4, 5, 6]
    assert [a.page for a in result.attributes] == [1, 2, 3, 4, 5, 6]  # in page order
    assert result.document_type == "Syllabus"
    assert sorted(result.page_text) == [1, 2, 3, 4, 5, 6]


def test_a_resumed_run_with_everything_read_sends_nothing(pages) -> None:
    transport = FakeTransport(_pages_reply([1]))
    service, _ = _service(transport)

    result = service.extract_document(pages(2), resume=_saved(1, 2, doc_type="Invoice"))

    assert transport.calls == []
    assert (result.document_type, result.read_pages) == ("Invoice", [1, 2])


def test_saved_pages_that_are_not_part_of_this_document_are_ignored(pages) -> None:
    transport = FakeTransport(_pages_reply([2]))
    service, _ = _service(transport)

    result = service.extract_document(pages(2), resume=_saved(1, 9))

    assert [_page_labels(call) for call in transport.calls] == [["Page 2"]]
    assert result.read_pages == [1, 2]
    assert 9 not in result.page_text and all(a.page in (1, 2) for a in result.attributes)


def test_progress_is_reported_after_every_request_and_survives_a_later_failure(pages) -> None:
    snapshots: list[list[int]] = []
    transport = FakeTransport(_pages_reply([1, 2]), _pages_reply([3, 4]), _response(status=403))
    service, _ = _service(transport, pages_per_request=2)

    with pytest.raises(GeminiExtractionError, match="HTTP 403"):
        service.extract_document(pages(6), on_progress=lambda partial: snapshots.append(list(partial.read_pages)))

    assert snapshots == [[1, 2], [1, 2, 3, 4]]  # what was read before the failure was reported, so a retry can resume


def test_a_progress_callback_that_fails_never_fails_the_document(pages) -> None:
    def broken(partial):
        raise OSError("disk full")

    service, _ = _service(FakeTransport(_pages_reply([1, 2])))

    assert service.extract_document(pages(2), on_progress=broken).read_pages == [1, 2]


def test_a_split_shrinks_later_requests_instead_of_repeating_the_failure(pages) -> None:
    transport = FakeTransport(
        _truncated(),  # pages 1-6 do not fit in one answer
        _pages_reply([1, 2, 3]),
        _pages_reply([4, 5, 6]),
        _pages_reply([7, 8, 9]),
        _pages_reply([10, 11, 12]),
    )
    service, _ = _service(transport)

    result = service.extract_document(pages(12))

    assert [_page_labels(call) for call in transport.calls] == [
        ["Page 1", "Page 2", "Page 3", "Page 4", "Page 5", "Page 6"],
        ["Page 1", "Page 2", "Page 3"],
        ["Page 4", "Page 5", "Page 6"],
        ["Page 7", "Page 8", "Page 9"],  # not 6 pages again
        ["Page 10", "Page 11", "Page 12"],
    ]
    assert result.read_pages == list(range(1, 13))


def test_a_refusal_split_does_not_shrink_later_requests(pages) -> None:
    transport = FakeTransport(
        _refused(),
        _pages_reply([1, 2, 3]),
        _pages_reply([4, 5, 6]),
        _pages_reply([7, 8, 9, 10, 11, 12]),
    )
    service, _ = _service(transport)

    service.extract_document(pages(12))

    # one bad page is a problem with that page, not with the size of the requests
    assert [len(_page_labels(call)) for call in transport.calls] == [6, 3, 3, 6]


def test_a_timeout_on_a_multi_page_request_splits_it_instead_of_retrying_at_full_size(pages) -> None:
    class ReadTimeout(Exception):
        pass

    transport = FakeTransport(ReadTimeout("slow"), _pages_reply([1, 2]), _pages_reply([3, 4]))
    service, sleeps = _service(transport)

    result = service.extract_document(pages(4))

    assert [_page_labels(call) for call in transport.calls] == [["Page 1", "Page 2", "Page 3", "Page 4"], ["Page 1", "Page 2"], ["Page 3", "Page 4"]]
    assert sleeps == []  # no wasted wait before the split
    assert result.read_pages == [1, 2, 3, 4]


def test_a_reply_that_numbers_pages_by_position_is_mapped_back_to_the_real_pages(pages) -> None:
    transport = FakeTransport(_pages_reply([1]), _response(_payload("Syllabus", [{"page": 1, "text": "second"}], [_attribute("Second", "x", page=1)])))
    service, _ = _service(transport, pages_per_request=1)

    result = service.extract_document(pages(2))

    assert [(a.label, a.page) for a in result.attributes] == [("Attr 1", 1), ("Second", 2)]
    assert result.page_text == {1: "text of page 1", 2: "second"}


def test_results_for_pages_that_were_not_part_of_the_request_are_dropped(pages) -> None:
    reply = _response(
        _payload(
            "Syllabus",
            [{"page": 1, "text": "one"}, {"page": 7, "text": "elsewhere"}],
            [_attribute("Kept", "a", page=1), _attribute("Stray", "b", page=7)],
            tables=[_table_item(page=7)],
        )
    )
    service, _ = _service(FakeTransport(reply))

    result = service.extract_document(pages(2))

    assert [a.label for a in result.attributes] == ["Kept"]
    assert result.tables == [] and 7 not in result.page_text


def test_the_next_request_is_told_how_the_previous_page_ended(pages) -> None:
    long_text = "START-OF-PAGE " + ("x" * 800) + " ...and then the sentence continues"
    transport = FakeTransport(
        _response(_payload("Syllabus", [{"page": 1, "text": long_text}], [])),
        _response(_payload("Syllabus", [{"page": 2, "text": "tail"}], [])),
    )
    service, _ = _service(transport, pages_per_request=1)

    service.extract_document(pages(2))

    first, second = (_task(call) for call in transport.calls)
    assert "<previous_page_end>" not in first
    assert "page 1 ended with" in second and "<previous_page_end>" in second
    assert second.rstrip().endswith("and then the sentence continues\n</previous_page_end>")
    assert "START-OF-PAGE" not in second  # only the last few hundred characters are passed on


def test_a_saved_reading_survives_a_round_trip_through_json(pages) -> None:
    table = ExtractedTable(
        page=2,
        cells=[ExtractedCell(0, 0, "Label", is_header=True), ExtractedCell(0, 1, "x", column_span=2, row_span=2)],
        row_count=2,
        column_count=3,
        title="Form",
        kind="form",
        box=[1, 2, 3, 4],
        column_widths=[1.0, 2.0, 3.0],
        row_heights=[1.0, 1.0],
        bordered=False,
    )
    original = DocumentExtraction(
        document_type="Syllabus",
        page_text={1: "one", 2: "two\nlines"},
        attributes=[ExtractedAttribute("A", "1", 0.9, 1, box=[1, 2, 3, 4], continues_from_previous_page=True)],
        tables=[table],
        read_pages=[2, 1, 1],
    )

    restored = extraction_from_json(json.loads(json.dumps(extraction_to_json(original))))

    original.read_pages = [1, 2]  # saved sorted and de-duplicated
    assert restored == original


def _drop(data, key):
    data.pop(key)
    return data


def _set(data, **changes):
    data.update(changes)
    return data


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: None,
        lambda d: [],
        lambda d: _drop(d, "document_type"),
        lambda d: _drop(d, "read_pages"),
        lambda d: _set(d, read_pages=["x"]),
        lambda d: _set(d, page_text=[1]),
        lambda d: _set(d, attributes=[{"label": "A"}]),
        lambda d: _set(d, attributes=[{"label": "A", "value": "v", "confidence": "high", "page": 1}]),
        lambda d: _set(d, tables=[{"page": 1}]),
        lambda d: _set(d, tables="nope"),
    ],
)
def test_a_damaged_saved_reading_is_ignored(mutate) -> None:
    good = extraction_to_json(
        DocumentExtraction(
            document_type="Syllabus",
            page_text={1: "a"},
            attributes=[ExtractedAttribute("A", "1", 0.9, 1)],
            tables=[ExtractedTable.from_rows(1, [["a", "b"]])],
            read_pages=[1],
        )
    )
    assert extraction_from_json(good) is not None  # the undamaged version is accepted

    assert extraction_from_json(mutate(json.loads(json.dumps(good)))) is None


def test_the_settings_fingerprint_tracks_what_changes_a_reading() -> None:
    base = GeminiExtractionService(model="m1", api_key="k").settings_fingerprint()

    assert base == GeminiExtractionService(model="m1", api_key="another-key").settings_fingerprint()  # the key is not part of it
    assert base != GeminiExtractionService(model="m2", api_key="k").settings_fingerprint()
    assert base != GeminiExtractionService(model="m1", api_key="k", image_max_edge=1024).settings_fingerprint()


def test_an_eleven_page_document_that_fails_midway_is_finished_by_a_retry_without_rereading(pages, tmp_path) -> None:
    from app.extraction_checkpoint import CheckpointStore

    store = CheckpointStore(tmp_path / "42" / "extraction-checkpoint.json")
    document_pages = pages(11)

    # First attempt: pages 1-6 are read, then the service keeps failing for the rest.
    first_transport = FakeTransport(_pages_reply(range(1, 7)), _response(status=503))
    first_service, _ = _service(first_transport)
    fingerprint = f"source-hash-{first_service.settings_fingerprint()}"

    def save(partial):
        store.save(fingerprint, 11, extraction_to_json(partial))

    with pytest.raises(GeminiExtractionError, match="HTTP 503"):
        first_service.extract_document(document_pages, on_progress=save)

    assert store.progress() == {"pages_read": 6, "total_pages": 11}  # what the processing screen shows after the failure

    # Retry: a fresh service loads the saved reading and sends only pages 7-11.
    resume = extraction_from_json(store.load(fingerprint, 11))
    assert resume is not None and resume.read_pages == [1, 2, 3, 4, 5, 6]
    second_transport = FakeTransport(_pages_reply(range(7, 12)))
    second_service, _ = _service(second_transport)

    result = second_service.extract_document(document_pages, resume=resume, on_progress=save)

    assert [_page_labels(call) for call in second_transport.calls] == [[f"Page {n}" for n in range(7, 12)]]
    assert result.read_pages == list(range(1, 12))
    assert [a.page for a in result.attributes] == list(range(1, 12))
    assert sorted(result.page_text) == list(range(1, 12))
    assert store.progress() == {"pages_read": 11, "total_pages": 11}

    # A change of model (or prompt, or image size) invalidates the saved reading.
    other = GeminiExtractionService(model="a-different-model", api_key="k")
    assert store.load(f"source-hash-{other.settings_fingerprint()}", 11) is None


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Gemini API returned HTTP 429 (daily quota used up; it resets daily, or use a key or plan with a higher quota)", "right now"),
        ("Gemini API returned HTTP 429 (rate limit or quota exceeded)", "right now"),
        ("Gemini API returned HTTP 503 (Gemini service error)", "right now"),
        ("Gemini request failed: ReadTimeout", "right now"),
        ("Gemini response was not valid JSON (page 3)", "right now"),
        ("Gemini API returned HTTP 403 (check GEMINI_API_KEY and model access)", "right now"),
        ("Gemini API returned HTTP 401 (check GEMINI_API_KEY)", "right now"),
        ("GEMINI_API_KEY is not set", "right now"),
        ("Gemini stopped before finishing (RECITATION)", "right now"),
        ("Gemini declined to read page 2 (refusal)", "right now"),
        ("The document has no pages to read", "no pages"),
        ("Page 4 image could not be read (OSError)", "Page 4"),
    ],
)
def test_the_reason_shown_to_the_user_never_names_the_provider_or_the_quota(message: str, expected: str) -> None:
    from app.gemini_extractor import user_facing_reason

    reason = user_facing_reason(GeminiExtractionError(message))

    assert expected in reason
    lowered = reason.lower()
    for word in ("gemini", "claude", "anthropic", "api", "quota", "credit", "http", "key", "429", "403"):
        assert word not in lowered

