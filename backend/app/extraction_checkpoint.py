"""Remembers what has been read of a document so that Retry resumes instead of starting over.

After every request the extractor reports everything read so far; it is written next to the uploaded file.
A later run for the same document loads it and only sends the pages that are still missing, which matters
on rate-limited keys where re-reading a long document can use up the quota before it finishes.

A saved reading is only trusted when the source file, the page count and the extraction settings (model,
prompt, schema, image size) are all unchanged. Anything damaged or out of date is ignored, never repaired.
The file holds the document's text, so it lives beside the upload and is removed once the document succeeds.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # A tiny sidecar so the status endpoint does not have to parse the whole saved reading on every poll.
        self.progress_path = self.path.with_name(f"{self.path.stem}-progress.json")

    def load(self, fingerprint: str, total_pages: int) -> Optional[Dict[str, Any]]:
        """The saved extraction (as plain JSON) if it still applies, otherwise ``None``. Never raises."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("version") != CHECKPOINT_VERSION:
            return None
        if data.get("fingerprint") != fingerprint or data.get("total_pages") != total_pages:
            return None
        extraction = data.get("extraction")
        return extraction if isinstance(extraction, dict) else None

    def save(self, fingerprint: str, total_pages: int, extraction: Dict[str, Any]) -> None:
        read_pages = extraction.get("read_pages")
        self._write(
            self.path,
            {"version": CHECKPOINT_VERSION, "fingerprint": fingerprint, "total_pages": total_pages, "extraction": extraction},
        )
        self._write(
            self.progress_path,
            {"pages_read": len(read_pages) if isinstance(read_pages, list) else 0, "total_pages": total_pages},
        )

    def progress(self) -> Optional[Dict[str, int]]:
        """``{"pages_read": n, "total_pages": N}`` for the status screen, or ``None`` when nothing is saved."""
        try:
            data = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        pages_read, total_pages = (data.get("pages_read"), data.get("total_pages")) if isinstance(data, dict) else (None, None)
        if isinstance(pages_read, int) and isinstance(total_pages, int) and 0 <= pages_read <= total_pages and total_pages > 0:
            return {"pages_read": pages_read, "total_pages": total_pages}
        return None

    def clear(self) -> None:
        for path in (self.path, self.progress_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove %s", path.name)

    @staticmethod
    def _write(path: Path, data: Dict[str, Any]) -> None:
        """Write atomically so a crash mid-write never leaves a half-written file that looks valid."""
        path.parent.mkdir(parents=True, exist_ok=True)
        handle_fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
