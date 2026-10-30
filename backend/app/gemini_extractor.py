"""Gemini backend for LLM-based attribute extraction.

Same contract as ``LLMExtractionService`` in ``llm_extractor.py``: it receives the
page image plus numbered OCR tokens and returns fields that cite OCR token
indices. It calls the Gemini REST API directly, using structured JSON output,
so no extra SDK is required. The API key is read from ``GEMINI_API_KEY`` and is
sent in a request header, never in the URL.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from app.document_schemas import get_document_schema
from app.llm_extractor import (
    SYSTEM_PROMPT,
    LLMExtractionError,
    LLMExtractionService,
    PageExtraction,
    _encode_image,
    _format_tokens,
)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def gemini_is_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def _build_response_schema(expected_fields: Sequence[str]) -> Dict[str, Any]:
    label_schema: Dict[str, Any] = {"type": "STRING"}
    if expected_fields:
        label_schema["enum"] = list(expected_fields)
    return {
        "type": "OBJECT",
        "properties": {
            "fields": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "label": label_schema,
                        "value": {"type": "STRING"},
                        "token_indices": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                        "confidence": {"type": "NUMBER"},
                    },
                    "required": ["label", "value", "token_indices", "confidence"],
                },
            }
        },
        "required": ["fields"],
    }


def _default_post(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise LLMExtractionError("The 'httpx' package is not installed") from exc
    return httpx.post(url, headers=headers, json=payload, timeout=timeout)


class GeminiExtractionService:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        http_post: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_EXTRACTION_MODEL", DEFAULT_GEMINI_MODEL)
        self._api_key = api_key
        self._post = http_post or _default_post

    def extract_page(
        self,
        image_path: str | Path,
        tokens: Sequence[Dict[str, Any]],
        document_type: str | None = None,
    ) -> PageExtraction:
        api_key = self._api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise LLMExtractionError("GEMINI_API_KEY is not set")

        schema = get_document_schema(document_type)
        media_type, image_data = _encode_image(image_path)
        if schema.expected_fields:
            task = (
                f"Document type: {schema.display_name}. Extract these fields when present: "
                f"{', '.join(schema.expected_fields)}."
            )
        else:
            task = (
                "Document type is unknown. Extract the key facts you can identify (names, dates, reference "
                "numbers, amounts, addresses) using short snake_case labels."
            )

        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": media_type, "data": image_data}},
                        {
                            "text": f"{task}\n\nOCR tokens, one line of text per row:\n<ocr_tokens>\n{_format_tokens(tokens)}\n</ocr_tokens>"
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": _build_response_schema(schema.expected_fields),
            },
        }

        try:
            response = self._post(
                GEMINI_ENDPOINT.format(model=self.model),
                {"x-goog-api-key": api_key, "Content-Type": "application/json"},
                payload,
                float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60")),
            )
        except LLMExtractionError:
            raise
        except Exception as exc:
            raise LLMExtractionError(f"Gemini request failed: {type(exc).__name__}") from exc

        raw_fields = self._read_fields(response)
        fields = [
            parsed
            for parsed in (LLMExtractionService._parse_field(raw, document_type, len(tokens)) for raw in raw_fields)
            if parsed
        ]
        return PageExtraction(document_type=document_type or schema.document_type, fields=fields)

    @staticmethod
    def _read_fields(response: Any) -> List[Any]:
        status = getattr(response, "status_code", 200)
        if status != 200:
            # Status only: response bodies can echo request details, so they are not logged or raised.
            raise LLMExtractionError(f"Gemini API returned HTTP {status}")
        try:
            body = response.json()
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            fields = json.loads(text)["fields"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMExtractionError("Gemini response did not contain a structured field list") from exc
        if not isinstance(fields, list):
            raise LLMExtractionError("Gemini response did not contain a structured field list")
        return fields
