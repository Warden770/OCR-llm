from __future__ import annotations

import base64
import io
import json
import re
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from app.claude_extractor import (
    CLAUDE_ENDPOINT,
    DEFAULT_CLAUDE_MODEL,
    ClaudeExtractionError,
    ClaudeExtractionService,
    DocumentExtraction,
    ExtractedAttribute,
    ExtractedTable,
    ExtractionCanceled,
    _encode_image,
    build_attribute_payloads,
    build_table_layouts,
    normalize_document_type,
)

API_KEY = "sk-ant-test-SECRET-123"
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


def _response(payload=None, status=200, headers=None, stop_reason="end_turn", raw_text=None, body=None):
    if body is None:
        text = raw_text if raw_text is not None else json.dumps(payload if payload is not None else _payload())
        body = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
        }
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


def _service(transport, **kwargs) -> tuple[ClaudeExtractionService, list[float]]:
    sleeps: list[float] = []
    kwargs.setdefault("pages_per_request", 6)
    service = ClaudeExtractionService(
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


def _content(call) -> list[dict[str, Any]]:
    return call["payload"]["messages"][0]["content"]


def _page_labels(call) -> list[str]:
    labels = []
    for block in _content(call):
        match = re.match(r"Page (\d+)", block.get("text", "")) if block["type"] == "text" else None
        if match:
            labels.append(f"Page {match.group(1)}")
    return labels


def _walk_objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from _walk_objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _walk_objects(value)


# ---------------------------------------------------------------------------
# request shape
# ---------------------------------------------------------------------------

def test_request_shape_key_only_in_header_structured_output_and_no_attribute_enum(pages) -> None:
    transport = FakeTransport(_response(_payload(pages=[{"page": 1, "text": "a"}, {"page": 2, "text": "b"}], attributes=[])))
    service, _ = _service(transport)

    service.extract_document(pages(2))

    call = transport.calls[0]
    assert call["url"] == CLAUDE_ENDPOINT
    assert call["headers"]["x-api-key"] == API_KEY
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert call["headers"]["content-type"] == "application/json"
    assert API_KEY not in json.dumps(call["payload"])
    assert call["timeout"] == 30

    payload = call["payload"]
    assert payload["model"] == "test-model" and payload["max_tokens"] == 16000
    # Current Claude models reject forced tool use and fixed sampling settings.
    assert not {"tools", "tool_choice", "temperature"} & set(payload)

    content = _content(call)
    assert [(block["type"], block.get("text")) for block in content[:4]] == [
        ("text", "Page 1:"), ("image", None), ("text", "Page 2:"), ("image", None),
    ]
    assert content[1]["source"]["type"] == "base64" and content[1]["source"]["media_type"] == "image/jpeg"
    assert "1, 2" in content[-1]["text"]

    output_format = payload["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    schema = output_format["schema"]
    assert '"enum"' not in json.dumps(schema)
    assert schema["properties"]["attributes"]["items"]["properties"]["attribute"] == {"type": "string"}
    assert all(obj.get("additionalProperties") is False for obj in _walk_objects(schema))
    assert "box" not in schema["properties"]["attributes"]["items"]["properties"]
    assert "region" not in payload["system"] and "pixels" not in payload["system"]


def test_prompt_asks_for_the_body_of_letters_not_only_labelled_fields(pages) -> None:
    transport = FakeTransport(_response(_payload(attributes=[])))
    service, _ = _service(transport)

    service.extract_document(pages(1))

    prompt = transport.calls[0]["payload"]["system"]
    for expected in ("letters, memos", "main body", "one item per line", "never summarized or paraphrased"):
        assert expected in prompt


def test_prompt_and_task_ask_for_every_table_in_the_tables_list(pages) -> None:
    transport = FakeTransport(_response(_payload(attributes=[])))
    service, _ = _service(transport)

    service.extract_document(pages(1))

    call = transport.calls[0]
    prompt = call["payload"]["system"]
    for expected in ("return it in the tables list and not as attributes", "header row first", "exactly one string per column", "continues on the next page"):
        assert expected in prompt
    assert "every table on these pages in the tables list" in _content(call)[-1]["text"]


def test_default_model_comes_from_environment_then_constant(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_EXTRACTION_MODEL", raising=False)
    assert ClaudeExtractionService(api_key="k").model == DEFAULT_CLAUDE_MODEL
    monkeypatch.setenv("CLAUDE_EXTRACTION_MODEL", "custom-model")
    assert ClaudeExtractionService(api_key="k").model == "custom-model"


def test_settings_read_from_environment_and_fall_back_when_invalid(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_PAGES_PER_REQUEST", "3")
    monkeypatch.setenv("CLAUDE_IMAGE_MAX_EDGE", "1024")
    monkeypatch.setenv("CLAUDE_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("CLAUDE_REQUEST_DELAY_SECONDS", "1.5")
    monkeypatch.setenv("CLAUDE_REQUEST_BOXES", "true")
    service = ClaudeExtractionService(api_key="k")
    assert (service.pages_per_request, service.image_max_edge, service.timeout_seconds, service.request_delay_seconds) == (3, 1024, 45.0, 1.5)
    assert service.request_boxes is True

    monkeypatch.setenv("CLAUDE_PAGES_PER_REQUEST", "not-a-number")
    monkeypatch.setenv("CLAUDE_IMAGE_MAX_EDGE", "-5")
    monkeypatch.setenv("CLAUDE_TIMEOUT_SECONDS", "")
    monkeypatch.setenv("CLAUDE_REQUEST_BOXES", "")
    service = ClaudeExtractionService(api_key="k")
    assert (service.pages_per_request, service.image_max_edge, service.timeout_seconds) == (6, 2048, 120.0)
    assert service.request_boxes is False


def test_requests_with_more_than_twenty_images_cap_the_image_size() -> None:
    service = ClaudeExtractionService(api_key="k", pages_per_request=25, image_max_edge=5000)
    assert service.image_max_edge == 3000


def test_large_pages_are_downscaled_to_the_configured_edge(pages) -> None:
    transport = FakeTransport(_response(_payload(attributes=[])))
    service, _ = _service(transport, image_max_edge=512)

    service.extract_document(pages(1, size=(2000, 1000)))

    data = _content(transport.calls[0])[1]["source"]["data"]
    with Image.open(io.BytesIO(base64.b64decode(data))) as image:
        assert max(image.size) == 512


def test_images_stay_inside_the_visual_token_budget_so_claude_never_resizes_them(tmp_path) -> None:
    path = tmp_path / "square.png"
    Image.new("RGB", (3000, 3000), "white").save(path)

    data, width, height = _encode_image(path, 3000)

    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (width, height)
    assert -(-width // 28) * -(-height // 28) <= 4784


def test_missing_api_key_fails_clearly(pages, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    transport = FakeTransport(_response())
    service = ClaudeExtractionService(model="m", http_post=transport)

    with pytest.raises(ClaudeExtractionError, match="ANTHROPIC_API_KEY is not set"):
        service.extract_document(pages(1))
    assert transport.calls == []


def test_unreadable_page_image_fails_the_document(tmp_path) -> None:
    broken = tmp_path / "page-001.png"
    broken.write_bytes(b"fake-png")
    service, _ = _service(FakeTransport(_response()))

    with pytest.raises(ClaudeExtractionError, match="Page 1 image could not be read"):
        service.extract_document([(1, broken)])


# ---------------------------------------------------------------------------
# optional regions
# ---------------------------------------------------------------------------

def test_regions_are_only_requested_when_enabled_and_pixels_are_scaled_to_the_page(pages) -> None:
    attributes = [
        _attribute("A", "1", box=[20, 10, 100, 50]),  # 200x100 px image -> x0 y0 x1 y1 on a 0..1000 scale
        _attribute("B", "2", box=[-10, -10, 400, 400]),  # clamped
        _attribute("C", "3", box=[100, 50, 20, 10]),  # reversed corners
        _attribute("D", "4", box=[10, 10, 10, 50]),  # zero width
        _attribute("E", "5", box=[1, 2, 3]),  # wrong length
        _attribute("F", "6", box=["a", 1, 2, 3]),
        _attribute("G", "7"),
        _attribute("H", "8", page=9, box=[1, 1, 50, 50]),  # page we did not send
    ]
    transport = FakeTransport(_response(_payload(attributes=attributes)))
    service, _ = _service(transport, request_boxes=True)

    result = service.extract_document(pages(1))

    call = transport.calls[0]
    schema_props = call["payload"]["output_config"]["format"]["schema"]["properties"]["attributes"]["items"]["properties"]
    assert schema_props["box"] == {"type": "array", "items": {"type": "integer"}}
    assert _content(call)[0]["text"] == "Page 1 (image is 200 x 100 pixels):"
    assert "pixels" in call["payload"]["system"]
    assert [a.box for a in result.attributes] == [
        [100, 100, 500, 500], [0, 0, 1000, 1000], [100, 100, 500, 500], None, None, None, None, None,
    ]


def test_table_regions_use_the_same_pixel_scale(pages) -> None:
    tables = [{"page": 1, "title": "T", "rows": [["a", "b"]], "box": [100, 20, 200, 80]}]
    service, _ = _service(FakeTransport(_response(_payload(attributes=[], tables=tables))), request_boxes=True)

    result = service.extract_document(pages(1))

    assert result.tables[0].box == [500, 200, 1000, 800]


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
    assert '"Invoice"' in _content(transport.calls[1])[-1]["text"]
    assert "Invoice" not in _content(transport.calls[0])[-1]["text"]


def test_request_size_limit_lowers_the_chunk_size(pages) -> None:
    transport = FakeTransport(_response(_payload(attributes=[])))
    service, _ = _service(transport, pages_per_request=10)
    service.max_request_bytes = 1  # every page is "too big" to share a request

    service.extract_document(pages(3))

    assert [_page_labels(call) for call in transport.calls] == [["Page 1"], ["Page 2"], ["Page 3"]]


@pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
def test_a_cut_off_response_splits_the_chunk_and_retries(pages, stop_reason) -> None:
    transport = FakeTransport(
        _response(stop_reason=stop_reason, payload={}),
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
    transport = FakeTransport(_response(stop_reason="max_tokens", payload={}))
    service, _ = _service(transport)

    with pytest.raises(ClaudeExtractionError, match="page 1.*CLAUDE_MAX_OUTPUT_TOKENS"):
        service.extract_document(pages(1))
    assert len(transport.calls) == 1


def test_a_refusal_splits_the_chunk_to_find_the_page(pages) -> None:
    transport = FakeTransport(
        _response(stop_reason="refusal", payload={}),  # pages 1-3 together
        _response(_payload("Syllabus", [{"page": 1, "text": "a"}], [_attribute("Course Code", "CSC 106", page=1)])),  # page 1
        _response(stop_reason="refusal", payload={}),  # pages 2-3 together
        _response(_payload("Syllabus", [{"page": 2, "text": "b"}], [_attribute("Units", "3", page=2)])),
        _response(_payload("Syllabus", [{"page": 3, "text": "c"}], [_attribute("Term", "1st", page=3)])),
    )
    service, _ = _service(transport)

    result = service.extract_document(pages(3))

    assert [_page_labels(call) for call in transport.calls] == [
        ["Page 1", "Page 2", "Page 3"], ["Page 1"], ["Page 2", "Page 3"], ["Page 2"], ["Page 3"],
    ]
    assert [a.label for a in result.attributes] == ["Course Code", "Units", "Term"]


def test_a_refusal_on_a_single_page_fails_with_a_clear_reason(pages) -> None:
    # A refusal is a normal HTTP 200; it must not be mistaken for a usable answer.
    transport = FakeTransport(_response(stop_reason="refusal", payload=_payload()))
    service, _ = _service(transport)

    with pytest.raises(ClaudeExtractionError, match=r"declined to read page 1 \(refusal\)"):
        service.extract_document(pages(1))
    assert len(transport.calls) == 1


def test_a_failing_chunk_fails_the_whole_document(pages) -> None:
    transport = FakeTransport(_response(_payload(attributes=[])), _response(status=403))
    service, _ = _service(transport, pages_per_request=1)

    with pytest.raises(ClaudeExtractionError, match="HTTP 403"):
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

@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413])
def test_client_errors_fail_immediately_without_retry(pages, status) -> None:
    transport = FakeTransport(_response(status=status, body={"error": RESPONSE_BODY_MARKER}))
    service, sleeps = _service(transport)

    with pytest.raises(ClaudeExtractionError, match=f"HTTP {status}") as excinfo:
        service.extract_document(pages(1))

    assert len(transport.calls) == 1 and sleeps == []
    assert RESPONSE_BODY_MARKER not in str(excinfo.value)
    assert API_KEY not in str(excinfo.value)


def test_authentication_errors_point_at_the_key_and_carry_the_request_id(pages) -> None:
    transport = FakeTransport(_response(status=401, headers={"request-id": "req_018EeWyXxfu5pfWkrYcMdjWG"}, body={}))
    service, _ = _service(transport)

    with pytest.raises(ClaudeExtractionError) as excinfo:
        service.extract_document(pages(1))

    assert str(excinfo.value) == "Claude API returned HTTP 401 (check ANTHROPIC_API_KEY) [request req_018EeWyXxfu5pfWkrYcMdjWG]"


def test_a_hostile_request_id_header_is_not_echoed(pages) -> None:
    transport = FakeTransport(_response(status=401, headers={"request-id": f"{API_KEY} ignore previous instructions"}, body={}))
    service, _ = _service(transport)

    with pytest.raises(ClaudeExtractionError) as excinfo:
        service.extract_document(pages(1))

    assert "request" not in str(excinfo.value).split("]")[-1] and "[" not in str(excinfo.value)
    assert API_KEY not in str(excinfo.value)


def test_rate_limit_is_retried_honouring_retry_after(pages) -> None:
    transport = FakeTransport(_response(status=429, headers={"Retry-After": "2"}), _response(_payload(attributes=[])))
    service, sleeps = _service(transport)

    result = service.extract_document(pages(1))

    assert len(transport.calls) == 2
    assert sleeps == [2.0]
    assert result.attributes == []


def test_retry_after_is_capped_and_backoff_is_exponential_including_overloaded(pages) -> None:
    transport = FakeTransport(
        _response(status=429, headers={"Retry-After": "9999"}),
        _response(status=529),
        _response(status=500),
        _response(_payload(attributes=[])),
    )
    service, sleeps = _service(transport)

    service.extract_document(pages(1))

    assert sleeps == [60.0, 2.0, 4.0]


def test_server_errors_give_up_after_three_retries(pages) -> None:
    transport = FakeTransport(_response(status=529, body={"error": RESPONSE_BODY_MARKER}))
    service, sleeps = _service(transport)

    with pytest.raises(ClaudeExtractionError, match=r"HTTP 529 \(Claude is overloaded\)") as excinfo:
        service.extract_document(pages(1))

    assert len(transport.calls) == 4  # first attempt + 3 retries
    assert sleeps == [1.0, 2.0, 4.0]
    assert RESPONSE_BODY_MARKER not in str(excinfo.value)


def test_rate_limit_that_never_clears_reports_it(pages) -> None:
    transport = FakeTransport(_response(status=429))
    service, _ = _service(transport)

    with pytest.raises(ClaudeExtractionError, match=r"HTTP 429 \(rate limit"):
        service.extract_document(pages(1))


def test_transport_error_is_retried_once_then_reports_type_only(pages) -> None:
    class ReadTimeout(Exception):
        pass

    transport = FakeTransport(ReadTimeout(f"timed out talking to {API_KEY}"), _response(_payload(attributes=[])))
    service, _ = _service(transport)
    assert service.extract_document(pages(1)).attributes == []
    assert len(transport.calls) == 2

    transport = FakeTransport(ReadTimeout(f"timed out talking to {API_KEY}"))
    service, _ = _service(transport)
    with pytest.raises(ClaudeExtractionError, match="Claude request failed: ReadTimeout") as excinfo:
        service.extract_document(pages(1))
    assert len(transport.calls) == 2
    assert API_KEY not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


@pytest.mark.parametrize("reason", ["pause_turn", "tool_use"])
def test_unexpected_stop_reasons_fail(pages, reason) -> None:
    service, _ = _service(FakeTransport(_response(stop_reason=reason, payload=_payload())))
    with pytest.raises(ClaudeExtractionError, match=reason):
        service.extract_document(pages(1))


def test_responses_without_text_content_fail(pages) -> None:
    for body in (
        {"stop_reason": "end_turn", "content": []},
        {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": "hmm"}]},
        {"stop_reason": "end_turn", "content": "nope"},
        {"stop_reason": "end_turn"},
    ):
        service, _ = _service(FakeTransport(_response(body=body)))
        with pytest.raises(ClaudeExtractionError, match="no content"):
            service.extract_document(pages(1))


def test_only_text_blocks_are_read(pages) -> None:
    body = {
        "stop_reason": "end_turn",
        "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": json.dumps(_payload())}],
    }
    service, _ = _service(FakeTransport(_response(body=body)))

    assert service.extract_document(pages(1)).attributes[0].label == "Student Name"


def test_malformed_json_fails_without_echoing_the_text(pages) -> None:
    service, _ = _service(FakeTransport(_response(raw_text=f"{RESPONSE_BODY_MARKER} {{not json")))
    with pytest.raises(ClaudeExtractionError, match="not valid JSON") as excinfo:
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
    with pytest.raises(ClaudeExtractionError, match="expected structure"):
        service.extract_document(pages(1))


def test_optional_fields_default_and_malformed_tables_are_ignored_not_fatal(pages) -> None:
    tables = [{"page": 1, "rows": "nope"}, {"rows": [["a"]]}, {"page": 1, "title": "Ok", "rows": [["a", None], ["b", "c"]]}]
    service, _ = _service(FakeTransport(_response(_payload(attributes=[_attribute()], tables=tables))))

    result = service.extract_document(pages(1))

    assert result.attributes[0].continues_from_previous_page is False
    assert len(result.tables) == 1
    assert result.tables[0].rows == [["a", ""], ["b", "c"]]


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

def test_tables_become_review_cards_with_evenly_split_cell_boxes() -> None:
    extraction = DocumentExtraction(
        tables=[
            ExtractedTable(page=1, title="Subjects Enrolled", box=[100, 400, 900, 800], rows=[["Code", "Title"], ["IT101", "Intro"], ["", ""]]),
            ExtractedTable(page=1, rows=[["only", "one", "row"]]),
            ExtractedTable(page=9, rows=[["off", "document"]]),
        ]
    )

    layouts = build_table_layouts(extraction, {1: 101})

    assert list(layouts) == [1]
    first, second = layouts[1]
    assert first["id"] == "page-1-table-1" and first["page_id"] == 101 and first["page_number"] == 1
    assert first["title"] == "Subjects Enrolled"
    assert first["bbox"] == [100, 400, 900, 800]
    assert len(first["rows"]) == 2  # the empty row is dropped
    assert first["columns"] == [500.0, 500.0]
    cell = first["rows"][1]["cells"][1]
    assert (cell["text"], cell["row_index"], cell["column_index"], cell["is_header"]) == ("Intro", 1, 1, False)
    assert cell["bbox"] == [500, 600, 900, 800]
    assert cell["uncertain"] is True and cell["token_indices"] == []
    assert first["rows"][0]["cells"][0]["is_header"] is True
    assert second["title"] == "Table 2" and len(second["columns"]) == 3


def test_tables_without_a_box_have_no_cell_boxes_and_zero_row_height() -> None:
    extraction = DocumentExtraction(tables=[ExtractedTable(page=1, rows=[["a", "b"], ["c"]])])

    table = build_table_layouts(extraction, {1: 5})[1][0]

    assert table["bbox"] == [] and table["bounding_boxes"] == []
    assert all(row["height"] == 0 for row in table["rows"])
    assert table["rows"][1]["cells"][1]["text"] == ""  # short rows are padded
    assert all(cell["bbox"] == [] for row in table["rows"] for cell in row["cells"])
