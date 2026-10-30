from app.routing_engine import build_ocr_baseline_payloads


def test_ocr_baseline_extracts_labeled_multiline_value() -> None:
    payloads = build_ocr_baseline_payloads(
        page_id=1,
        document_id=2,
        document_type="invoice",
        tokens=[
            {"token": "Bill", "bbox": [10, 10, 30, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "to:", "bbox": [32, 10, 50, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Acme", "bbox": [55, 10, 80, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Corporation", "bbox": [10, 30, 70, 40], "confidence": 90, "block_num": 1, "paragraph_num": 2, "line_num": 1},
        ],
    )

    assert len(payloads) == 1
    assert payloads[0]["entity_type_label"] == "bill_to"
    assert payloads[0]["extracted_value"] == "Acme\nCorporation"


def test_ocr_baseline_keeps_text_for_unknown_document_type() -> None:
    payloads = build_ocr_baseline_payloads(
        page_id=1,
        document_id=2,
        document_type="unknown",
        tokens=[
            {"token": "Unclassified", "bbox": [10, 10, 70, 20], "confidence": 95},
            {"token": "document", "bbox": [75, 10, 125, 20], "confidence": 90},
        ],
    )

    assert len(payloads) == 1
    assert payloads[0]["entity_type_label"] == "ocr_text"
    assert payloads[0]["extracted_value"] == "Unclassified document"


def test_ocr_baseline_keeps_text_when_known_schema_has_no_matching_labels() -> None:
    payloads = build_ocr_baseline_payloads(
        page_id=1,
        document_id=2,
        document_type="invoice",
        tokens=[
            {"token": "Letter", "bbox": [10, 10, 50, 20], "confidence": 95},
            {"token": "content", "bbox": [55, 10, 100, 20], "confidence": 90},
        ],
    )

    assert len(payloads) == 1
    assert payloads[0]["entity_type_label"] == "ocr_text"
    assert payloads[0]["extracted_value"] == "Letter content"


def test_ocr_baseline_splits_long_unstructured_text() -> None:
    tokens = [
        {"token": f"word{index}", "bbox": [index, 10, index + 5, 20], "confidence": 95, "block_num": 1, "paragraph_num": index // 20 + 1, "line_num": index}
        for index in range(160)
    ]

    payloads = build_ocr_baseline_payloads(1, 2, tokens, document_type="unknown")

    assert len(payloads) > 1
    assert all(len(item["extracted_value"]) <= 600 for item in payloads)
    assert payloads[0]["entity_type_label"] == "ocr_text_1"


def test_ocr_baseline_extracts_letter_fields() -> None:
    payloads = build_ocr_baseline_payloads(
        page_id=1,
        document_id=2,
        document_type="letter",
        tokens=[
            {"token": "To:", "bbox": [10, 10, 25, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Dr.", "bbox": [30, 10, 45, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Riah", "bbox": [50, 10, 80, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Subject:", "bbox": [10, 30, 55, 40], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 2},
            {"token": "Permission", "bbox": [60, 30, 120, 40], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 2},
        ],
    )

    assert [(item["entity_type_label"], item["extracted_value"]) for item in payloads] == [
        ("recipient", "Dr. Riah"),
        ("subject", "Permission"),
    ]


def test_ocr_baseline_keeps_letter_body_paragraphs() -> None:
    payloads = build_ocr_baseline_payloads(
        page_id=1,
        document_id=2,
        document_type="letter",
        tokens=[
            {"token": "To:", "bbox": [10, 10, 25, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Recipient", "bbox": [30, 10, 80, 20], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 1},
            {"token": "Subject:", "bbox": [10, 30, 55, 40], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 2},
            {"token": "Request", "bbox": [60, 30, 110, 40], "confidence": 95, "block_num": 1, "paragraph_num": 1, "line_num": 2},
            {"token": "This", "bbox": [10, 60, 35, 70], "confidence": 95, "block_num": 2, "paragraph_num": 1, "line_num": 1},
            {"token": "is", "bbox": [40, 60, 50, 70], "confidence": 95, "block_num": 2, "paragraph_num": 1, "line_num": 1},
            {"token": "the", "bbox": [55, 60, 75, 70], "confidence": 95, "block_num": 2, "paragraph_num": 1, "line_num": 1},
            {"token": "body.", "bbox": [80, 60, 115, 70], "confidence": 95, "block_num": 2, "paragraph_num": 1, "line_num": 1},
            {"token": "More", "bbox": [10, 80, 35, 90], "confidence": 95, "block_num": 2, "paragraph_num": 2, "line_num": 1},
            {"token": "body.", "bbox": [40, 80, 75, 90], "confidence": 95, "block_num": 2, "paragraph_num": 2, "line_num": 1},
        ],
    )

    assert [(item["entity_type_label"], item["extracted_value"]) for item in payloads] == [
        ("recipient", "Recipient"),
        ("subject", "Request"),
        ("body_3", "This is the body."),
        ("body_4", "More body."),
    ]
