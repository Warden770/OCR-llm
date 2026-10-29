from __future__ import annotations

import logging
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike

logger = logging.getLogger(__name__)


def _normalized_box(box: list[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, round(x0 * 1000 / width))),
        max(0, min(1000, round(y0 * 1000 / height))),
        max(0, min(1000, round(x1 * 1000 / width))),
        max(0, min(1000, round(y1 * 1000 / height))),
    ]


def _token_box(token: dict[str, Any]) -> list[float] | None:
    box = token.get("absolute_bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        return [float(value) for value in box]
    except (TypeError, ValueError):
        return None


def _group_ocr_lines(tokens: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    lines: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for index, token in enumerate(tokens):
        key = (
            int(token.get("block_num", 0)),
            int(token.get("paragraph_num", 0)),
            int(token.get("line_num", 0)),
        )
        lines.setdefault(key, []).append({**token, "_index": index})
    return sorted(
        (sorted(line, key=lambda item: (_token_box(item) or [0, 0])[0]) for line in lines.values()),
        key=lambda line: min((_token_box(item) or [0, 0, 0, 0])[1] for item in line),
    )


def _group_visual_lines(tokens: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    positioned = [
        ({**token, "_index": index}, _token_box(token))
        for index, token in enumerate(tokens)
        if _token_box(token) is not None
    ]
    positioned.sort(key=lambda item: ((item[1][1] + item[1][3]) / 2, item[1][0]))
    line_groups: list[list[tuple[dict[str, Any], list[float]]]] = []
    for token, box in positioned:
        center_y = (box[1] + box[3]) / 2
        matching_group = None
        for group in reversed(line_groups[-4:]):
            group_top = min(item_box[1] for _, item_box in group)
            group_bottom = max(item_box[3] for _, item_box in group)
            overlap = max(0.0, min(group_bottom, box[3]) - max(group_top, box[1]))
            if overlap >= min(group_bottom - group_top, box[3] - box[1]) * 0.35:
                matching_group = group
                break
            group_center = median((item_box[1] + item_box[3]) / 2 for _, item_box in group)
            typical_height = median(item_box[3] - item_box[1] for _, item_box in group)
            if abs(center_y - group_center) <= max(2.0, typical_height * 0.35):
                matching_group = group
                break
        if matching_group is None:
            line_groups.append([(token, box)])
        else:
            matching_group.append((token, box))
    return [
        [token for token, _ in sorted(group, key=lambda item: item[1][0])]
        for group in sorted(line_groups, key=lambda group: min(item_box[1] for _, item_box in group))
    ]


def _union_box(tokens: list[dict[str, Any]]) -> list[float]:
    boxes = [_token_box(token) for token in tokens]
    valid = [box for box in boxes if box is not None]
    if not valid:
        return [0.0, 0.0, 0.0, 0.0]
    return [min(box[0] for box in valid), min(box[1] for box in valid), max(box[2] for box in valid), max(box[3] for box in valid)]


def _cell_text(tokens: list[dict[str, Any]]) -> str:
    text_lines = []
    for line in _group_visual_lines(tokens):
        words = []
        for token in line:
            value = str(token.get("token", "")).strip()
            if not value:
                continue
            value = value.replace("|*", "*").replace("|•", "•").replace("|", "")
            if value:
                words.append(value)
        if words:
            text_lines.append(" ".join(words))
    return "\n".join(text_lines)


def _nearby_caption(table: dict[str, Any], tokens: list[dict[str, Any]], width: int, height: int) -> tuple[str, list[int] | None]:
    table_top = float(table["bbox"][1]) * height / 1000
    token_heights = [max(1.0, (_token_box(token) or [0, 0, 0, 0])[3] - (_token_box(token) or [0, 0, 0, 0])[1]) for token in tokens]
    max_gap = max(18.0, (median(token_heights) if token_heights else 10.0) * 2.5)
    candidates = []
    for line in _group_ocr_lines(tokens):
        box = _union_box(line)
        gap = table_top - box[3]
        if gap < 0 or gap > max_gap:
            continue
        text = _cell_text(line).replace("\n", " ").strip()
        normalized = text.casefold()
        looks_labeled = normalized.startswith(("table ", "tab. ", "schedule ", "figure ", "grading ", "assessment "))
        looks_heading = len(text) <= 90 and text.isupper() and len(text.split()) >= 2
        if text and (looks_labeled or looks_heading):
            candidates.append((gap, text, _normalized_box(box, width, height)))
    if not candidates:
        return "", None
    _, caption, caption_box = min(candidates, key=lambda candidate: candidate[0])
    return caption, caption_box


def _projection_centers(mask: MatLike, axis: int, threshold: float) -> list[int]:
    if axis == 0:
        values = np.count_nonzero(mask, axis=1)
        span = mask.shape[1]
    else:
        values = np.count_nonzero(mask, axis=0)
        span = mask.shape[0]
    active = values >= max(2, int(span * threshold))
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    groups = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)
    return [int(round(float(group.mean()))) for group in groups if len(group)]


def _line_support(mask: MatLike, axis: int, coordinate: int, start: int, end: int) -> float:
    if end <= start:
        return 0.0
    radius = 2
    if axis == 0:
        area = mask[max(0, coordinate - radius):coordinate + radius + 1, start:end]
    else:
        area = mask[start:end, max(0, coordinate - radius):coordinate + radius + 1]
    return cv2.countNonZero(area) / max(1, area.size)


def _tokens_in_box(tokens: list[dict[str, Any]], box: list[float]) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = box
    return [
        token for token in tokens
        if (token_box := _token_box(token)) is not None
        and x0 <= (token_box[0] + token_box[2]) / 2 <= x1
        and y0 <= (token_box[1] + token_box[3]) / 2 <= y1
    ]


def _make_cell(
    tokens: list[dict[str, Any]],
    box: list[float],
    width: int,
    height: int,
    row_index: int,
    column_index: int,
    *,
    rowspan: int = 1,
    colspan: int = 1,
    ruled: bool,
) -> dict[str, Any]:
    selected_tokens = _tokens_in_box(tokens, box)
    text = _cell_text(selected_tokens)
    confidence_values = [float(token.get("confidence", 0)) for token in selected_tokens]
    text_box = _union_box(selected_tokens) if selected_tokens else box
    horizontal_width = max(1.0, box[2] - box[0])
    left_padding = text_box[0] - box[0]
    right_padding = box[2] - text_box[2]
    if abs(left_padding - right_padding) <= horizontal_width * 0.12:
        alignment = "center"
    elif right_padding < left_padding * 0.35:
        alignment = "right"
    else:
        alignment = "left"
    page_text_heights = [
        max(1.0, (_token_box(token) or [0, 0, 0, 0])[3] - (_token_box(token) or [0, 0, 0, 0])[1])
        for token in tokens
    ]
    cell_text_heights = [
        max(1.0, (_token_box(token) or [0, 0, 0, 0])[3] - (_token_box(token) or [0, 0, 0, 0])[1])
        for token in selected_tokens
    ]
    font_scale = median(cell_text_heights) / median(page_text_heights) if cell_text_heights and page_text_heights else 1.0
    return {
        "id": f"r{row_index}-c{column_index}",
        "row_index": row_index,
        "column_index": column_index,
        "text": text,
        "original_text": text,
        "bbox": _normalized_box(box, width, height),
        "rowspan": rowspan,
        "colspan": colspan,
        "is_header": row_index == 0,
        "bold": row_index == 0,
        "alignment": alignment,
        "font_scale": round(max(0.75, min(1.5, font_scale)), 2),
        "confidence_score": round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else 0.0,
        "token_indices": [token.get("_index") for token in selected_tokens if token.get("_index") is not None],
        "uncertain": not ruled,
    }


def _ruled_tables(
    image: MatLike,
    tokens: list[dict[str, Any]],
    page_id: int,
    page_number: int,
) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width // 100), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, height // 100)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    combined = cv2.bitwise_or(horizontal, vertical)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables: list[dict[str, Any]] = []

    for contour in contours:
        x, y, region_width, region_height = cv2.boundingRect(contour)
        if region_width < width * 0.08 or region_height < height * 0.015:
            continue
        x1, y1 = x + region_width, y + region_height
        region_area_ratio = (region_width * region_height) / max(1, width * height)
        if region_area_ratio >= 0.55 or (
            x <= width * 0.08
            and y <= height * 0.08
            and x1 >= width * 0.92
            and y1 >= height * 0.92
        ):
            continue
        local_horizontal = horizontal[y:y1, x:x1]
        local_vertical = vertical[y:y1, x:x1]
        row_lines = _projection_centers(local_horizontal, axis=0, threshold=0.2)
        column_lines = _projection_centers(local_vertical, axis=1, threshold=0.12)
        if len(row_lines) < 2 or len(column_lines) < 2:
            continue

        absolute_rows = [y + value for value in row_lines]
        absolute_columns = [x + value for value in column_lines]
        if len(absolute_rows) * len(absolute_columns) < 6:
            continue

        cells: list[list[dict[str, Any] | None]] = [
            [None for _ in range(len(absolute_columns) - 1)]
            for _ in range(len(absolute_rows) - 1)
        ]
        occupied: set[tuple[int, int]] = set()
        for row_index in range(len(absolute_rows) - 1):
            for column_index in range(len(absolute_columns) - 1):
                if (row_index, column_index) in occupied:
                    continue
                cell_x0 = absolute_columns[column_index]
                cell_y0 = absolute_rows[row_index]
                colspan = 1
                while column_index + colspan < len(absolute_columns) - 1:
                    separator = absolute_columns[column_index + colspan] - x
                    local_support = _line_support(local_vertical, 1, separator, cell_y0 - y, absolute_rows[row_index + 1] - y)
                    if local_support >= 0.12:
                        break
                    colspan += 1
                rowspan = 1
                while row_index + rowspan < len(absolute_rows) - 1:
                    separator = absolute_rows[row_index + rowspan] - y
                    local_support = _line_support(local_horizontal, 0, separator, cell_x0 - x, absolute_columns[column_index + colspan] - x)
                    if local_support >= 0.12:
                        break
                    rowspan += 1
                for covered_row in range(row_index, row_index + rowspan):
                    for covered_column in range(column_index, column_index + colspan):
                        occupied.add((covered_row, covered_column))
                cell_x1 = absolute_columns[column_index + colspan]
                cell_y1 = absolute_rows[row_index + rowspan]
                cells[row_index][column_index] = _make_cell(
                    tokens,
                    [cell_x0, cell_y0, cell_x1, cell_y1],
                    width,
                    height,
                    row_index,
                    column_index,
                    rowspan=rowspan,
                    colspan=colspan,
                    ruled=True,
                )

        rows = [
            {
                "index": index,
                "cells": [cell for cell in row if cell is not None],
                "height": round((absolute_rows[index + 1] - absolute_rows[index]) * 1000 / height, 2),
            }
            for index, row in enumerate(cells)
        ]
        populated_cells = [cell for row in rows for cell in row["cells"] if cell["text"]]
        if not populated_cells:
            continue
        if len(absolute_columns) == 2 and max(len(cell["text"]) for cell in populated_cells) > 280:
            continue
        table_box = [absolute_columns[0], absolute_rows[0], absolute_columns[-1], absolute_rows[-1]]
        tables.append({
            "id": f"page-{page_number}-table-{len(tables) + 1}",
            "page_id": page_id,
            "page_number": page_number,
            "title": f"Table {len(tables) + 1}",
            "caption": "",
            "bbox": _normalized_box(table_box, width, height),
            "columns": [round((right - left) * 1000 / width, 2) for left, right in zip(absolute_columns, absolute_columns[1:])],
            "rows": rows,
            "style": {"bordered": True, "border_width": 1, "layout_source": "detected_rules"},
            "confidence_score": 0.9,
            "uncertain_fields": ["font_family", "font_size", "font_emphasis", "cell_background_color"],
            "source": "ocr_layout",
            "bounding_boxes": [_normalized_box(table_box, width, height)],
        })
    return tables


def _borderless_tables(
    tokens: list[dict[str, Any]],
    page_id: int,
    page_number: int,
    width: int,
    height: int,
    ruled_tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lines = _group_visual_lines(tokens)
    heights = [max(1.0, (_token_box(token) or [0, 0, 0, 0])[3] - (_token_box(token) or [0, 0, 0, 0])[1]) for token in tokens]
    typical_height = median(heights) if heights else 10
    candidates: list[dict[str, Any]] = []
    for line in lines:
        valid = [token for token in line if _token_box(token)]
        if len(valid) < 2:
            continue
        gaps = []
        for previous, current in zip(valid, valid[1:]):
            previous_box, current_box = _token_box(previous), _token_box(current)
            gaps.append(current_box[0] - previous_box[2])
        split_threshold = max(typical_height * 2.2, width * 0.006)
        chunks: list[list[dict[str, Any]]] = [[]]
        for token_index, token in enumerate(valid):
            if token_index and gaps[token_index - 1] > split_threshold:
                chunks.append([])
            chunks[-1].append(token)
        chunks = [chunk for chunk in chunks if chunk]
        if len(chunks) >= 2:
            candidates.append({"chunks": chunks, "box": _union_box(valid), "tokens": valid})

    row_groups: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        box = candidate["box"]
        inside_ruled_table = any(
            box[0] >= table["bbox"][0] * width / 1000
            and box[2] <= table["bbox"][2] * width / 1000
            and box[1] >= table["bbox"][1] * height / 1000
            and box[3] <= table["bbox"][3] * height / 1000
            for table in ruled_tables
        )
        if inside_ruled_table:
            continue
        if not row_groups:
            row_groups.append([candidate])
            continue
        previous = row_groups[-1][-1]
        previous_box = previous["box"]
        close_vertically = box[1] - previous_box[3] <= typical_height * 2.8
        same_columns = len(candidate["chunks"]) == len(previous["chunks"])
        aligned = same_columns and all(
            abs(_union_box(candidate["chunks"][index])[0] - _union_box(previous["chunks"][index])[0]) <= typical_height * 2.5
            for index in range(len(candidate["chunks"]))
        )
        if close_vertically and aligned:
            row_groups[-1].append(candidate)
        else:
            row_groups.append([candidate])

    tables = []
    for row_group in row_groups:
        if len(row_group) < 2 or len(row_group[0]["chunks"]) < 2:
            continue
        column_count = min(16, max(len(row["chunks"]) for row in row_group))
        row_bounds = [_union_box(row["tokens"]) for row in row_group]
        table_box = [
            min(box[0] for box in row_bounds),
            min(box[1] for box in row_bounds),
            max(box[2] for box in row_bounds),
            max(box[3] for box in row_bounds),
        ]
        column_starts = []
        column_ends = []
        for column_index in range(column_count):
            column_boxes = [_union_box(row["chunks"][column_index]) for row in row_group if column_index < len(row["chunks"])]
            column_starts.append(median(box[0] for box in column_boxes))
            column_ends.append(median(box[2] for box in column_boxes))
        boundaries = [table_box[0]] + [
            (column_ends[index] + column_starts[index + 1]) / 2
            for index in range(column_count - 1)
        ] + [table_box[2]]
        row_bands = [table_box[1]] + [
            (row_bounds[index][3] + row_bounds[index + 1][1]) / 2
            for index in range(len(row_bounds) - 1)
        ] + [table_box[3]]
        rows = []
        for row_index, row in enumerate(row_group):
            cells = []
            for column_index in range(column_count):
                cell_tokens = row["chunks"][column_index] if column_index < len(row["chunks"]) else []
                cell_box = _union_box(cell_tokens) if cell_tokens else [boundaries[column_index], row_bounds[row_index][1], boundaries[column_index + 1], row_bounds[row_index][3]]
                cell = _make_cell(cell_tokens, cell_box, width, height, row_index, column_index, ruled=False)
                cell["bbox"] = _normalized_box(
                    [boundaries[column_index], row_bands[row_index], boundaries[column_index + 1], row_bands[row_index + 1]],
                    width,
                    height,
                )
                cells.append(cell)
            rows.append({"index": row_index, "cells": cells, "height": round((row_bands[row_index + 1] - row_bands[row_index]) * 1000 / height, 2)})
        table_index = len(ruled_tables) + len(tables) + 1
        tables.append({
            "id": f"page-{page_number}-table-{table_index}",
            "page_id": page_id,
            "page_number": page_number,
            "title": f"Table {table_index}",
            "caption": "",
            "bbox": _normalized_box(table_box, width, height),
            "columns": [round((right - left) * 1000 / width, 2) for left, right in zip(boundaries, boundaries[1:])],
            "rows": rows,
            "style": {"bordered": False, "border_width": 0, "layout_source": "ocr_alignment"},
            "confidence_score": 0.55,
            "uncertain_fields": ["cell_boundaries", "merged_cells", "font_emphasis"],
            "source": "ocr_layout",
            "bounding_boxes": [_normalized_box(table_box, width, height)],
        })
    return tables


def reconstruct_tables(
    image_path: str | Path,
    tokens: list[dict[str, Any]],
    page_id: int,
    page_number: int,
) -> dict[str, Any]:
    """Reconstruct ruled and conservative borderless tables using OCR and page geometry."""
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Cannot reconstruct tables without page image: %s", image_path)
        return {"page_width": None, "page_height": None, "tables": []}
    height, width = image.shape[:2]
    indexed_tokens = [{**token, "_index": index} for index, token in enumerate(tokens)]
    ruled = _ruled_tables(image, indexed_tokens, page_id, page_number)
    borderless = _borderless_tables(indexed_tokens, page_id, page_number, width, height, ruled)
    tables = ruled + borderless
    for table in tables:
        caption, caption_box = _nearby_caption(table, indexed_tokens, width, height)
        table["caption"] = caption
        table["caption_bbox"] = caption_box
        if not caption:
            table["uncertain_fields"] = list(dict.fromkeys([*table["uncertain_fields"], "caption"]))
    return {"page_width": width, "page_height": height, "tables": tables}