from __future__ import annotations

import json

from app.extraction_checkpoint import CHECKPOINT_VERSION, CheckpointStore, file_sha256

FINGERPRINT = "source123-settings456"
EXTRACTION = {"document_type": "Syllabus", "read_pages": [1, 2], "page_text": {"1": "a", "2": "b"}, "attributes": [], "tables": []}


def _store(tmp_path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "42" / "extraction-checkpoint.json")


def test_a_saved_reading_is_loaded_when_everything_still_matches(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(FINGERPRINT, 5, EXTRACTION)

    assert store.load(FINGERPRINT, 5) == EXTRACTION


def test_nothing_is_loaded_when_the_source_settings_or_page_count_changed(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(FINGERPRINT, 5, EXTRACTION)

    assert store.load("another-source-or-prompt", 5) is None
    assert store.load(FINGERPRINT, 6) is None


def test_missing_damaged_and_foreign_files_are_ignored_without_raising(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.load(FINGERPRINT, 5) is None  # nothing saved, folder does not exist

    store.path.parent.mkdir(parents=True)
    for content in ("", "{not json", "[]", "null", json.dumps({"version": 999}), json.dumps({"version": CHECKPOINT_VERSION}),
                    json.dumps({"version": CHECKPOINT_VERSION, "fingerprint": FINGERPRINT, "total_pages": 5, "extraction": "text"})):
        store.path.write_text(content, encoding="utf-8")
        assert store.load(FINGERPRINT, 5) is None


def test_progress_reports_pages_read_and_survives_for_the_status_screen(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.progress() is None

    store.save(FINGERPRINT, 11, {"read_pages": [1, 2, 3, 4, 5, 6]})

    assert store.progress() == {"pages_read": 6, "total_pages": 11}


def test_nonsense_progress_is_ignored(tmp_path) -> None:
    store = _store(tmp_path)
    store.progress_path.parent.mkdir(parents=True)
    for content in ('{"pages_read": 9, "total_pages": 3}', '{"pages_read": -1, "total_pages": 3}', '{"pages_read": 0, "total_pages": 0}',
                    '{"pages_read": "2", "total_pages": 3}', "[]", "oops"):
        store.progress_path.write_text(content, encoding="utf-8")
        assert store.progress() is None


def test_clear_removes_both_files_and_is_safe_to_repeat(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(FINGERPRINT, 5, EXTRACTION)

    store.clear()
    store.clear()

    assert store.load(FINGERPRINT, 5) is None and store.progress() is None
    assert not store.path.exists() and not store.progress_path.exists()


def test_saving_again_replaces_the_previous_reading_and_leaves_no_temp_files(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(FINGERPRINT, 5, {"read_pages": [1]})
    store.save(FINGERPRINT, 5, EXTRACTION)

    assert store.load(FINGERPRINT, 5) == EXTRACTION
    assert sorted(p.name for p in store.path.parent.iterdir()) == ["extraction-checkpoint-progress.json", "extraction-checkpoint.json"]


def test_a_failed_write_keeps_the_previous_reading_intact(tmp_path) -> None:
    store = _store(tmp_path)
    store.save(FINGERPRINT, 5, EXTRACTION)

    class Unserializable:
        pass

    try:
        store.save(FINGERPRINT, 5, {"read_pages": [1], "bad": Unserializable()})
    except TypeError:
        pass

    assert store.load(FINGERPRINT, 5) == EXTRACTION
    assert [p.name for p in store.path.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_file_sha256_changes_with_the_content(tmp_path) -> None:
    first = tmp_path / "a.pdf"
    first.write_bytes(b"one")
    second = tmp_path / "b.pdf"
    second.write_bytes(b"two")

    assert file_sha256(first) == file_sha256(first)
    assert file_sha256(first) != file_sha256(second)
    assert len(file_sha256(first)) == 64
