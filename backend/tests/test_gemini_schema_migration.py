from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

MIGRATION_PATH = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "0013_gemini_page_text_and_error.py"

# Schema as it stands after migration 0012, reduced to the tables 0013 touches.
PRE_0013_SCHEMA = [
    """
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(255) NOT NULL,
        format VARCHAR(32) NOT NULL,
        owner_id VARCHAR(128),
        document_type VARCHAR(64) NOT NULL DEFAULT 'unknown',
        page_count INTEGER NOT NULL DEFAULT 0,
        status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
        extracted_attributes JSON NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
        page_number INTEGER NOT NULL,
        image_path VARCHAR(512) NOT NULL,
        storage_key VARCHAR(512),
        CONSTRAINT uq_pages_document_id_id UNIQUE (document_id, id)
    )
    """,
    """
    CREATE TABLE attributes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        page_id INTEGER NOT NULL REFERENCES pages (id) ON DELETE CASCADE,
        document_id INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
        entity_type_label VARCHAR(128) NOT NULL,
        extracted_value TEXT NOT NULL,
        FOREIGN KEY (document_id, page_id) REFERENCES pages (document_id, id) ON DELETE CASCADE
    )
    """,
    "INSERT INTO documents (id, name, format, owner_id, status) VALUES (1, 'old.pdf', 'pdf', 'user-1', 'PROCESSED')",
    "INSERT INTO pages (id, document_id, page_number, image_path) VALUES (1, 1, 1, 'uploads/1/page-001.png')",
    "INSERT INTO attributes (id, page_id, document_id, entity_type_label, extracted_value) VALUES (1, 1, 1, 'TOTAL', '42.00')",
]


def _engine_with_existing_rows():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with engine.begin() as connection:
        for statement in PRE_0013_SCHEMA:
            connection.execute(text(statement))
    return engine


def _load_migration():
    spec = importlib.util.spec_from_file_location("migration_0013", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns(engine, table: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(table)}


def test_revision_chain_is_linear() -> None:
    module = _load_migration()
    assert module.revision == "0013_gemini_page_text_and_error"
    assert module.down_revision == "0012_add_user_profiles"


def test_0013_upgrades_a_sqlite_database_with_existing_rows_and_downgrades_again() -> None:
    migration = _load_migration()
    engine = _engine_with_existing_rows()

    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()

    assert {"ocr_text", "ocr_layout"} <= _columns(engine, "pages")
    assert "processing_error" in _columns(engine, "documents")
    with engine.connect() as connection:
        document = connection.execute(text("SELECT name, status, processing_error FROM documents")).one()
        page = connection.execute(text("SELECT image_path, ocr_text, ocr_layout FROM pages")).one()
        attribute = connection.execute(text("SELECT entity_type_label, extracted_value FROM attributes")).one()
        connection.execute(text("INSERT INTO pages (document_id, page_number, image_path) VALUES (1, 2, 'p2.png')"))
        connection.execute(text("UPDATE pages SET ocr_text = 'hello', ocr_layout = '[{\"id\": 1}]' WHERE page_number = 2"))
        stored_layout = connection.execute(text("SELECT ocr_layout FROM pages WHERE page_number = 2")).scalar_one()
    assert tuple(document) == ("old.pdf", "PROCESSED", None)
    assert page.image_path == "uploads/1/page-001.png" and page.ocr_text is None
    assert page.ocr_layout in (None, "[]") or json.loads(page.ocr_layout) == []
    assert tuple(attribute) == ("TOTAL", "42.00")
    assert json.loads(stored_layout) == [{"id": 1}]
    # the unique constraint other tables reference must survive the table rebuild
    constraints = {constraint["name"] for constraint in inspect(engine).get_unique_constraints("pages")}
    assert "uq_pages_document_id_id" in constraints

    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()

    assert not ({"ocr_text", "ocr_layout"} & _columns(engine, "pages"))
    assert "processing_error" not in _columns(engine, "documents")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM documents")).scalar_one() == 1
        assert connection.execute(text("SELECT COUNT(*) FROM attributes")).scalar_one() == 1


def test_sqlite_compatibility_path_adds_the_new_columns_to_an_older_file(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import db

    engine = _engine_with_existing_rows()
    monkeypatch.setattr(db, "engine", engine)

    db._ensure_review_columns()

    assert {"ocr_text", "ocr_layout"} <= _columns(engine, "pages")
    assert "processing_error" in _columns(engine, "documents")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT name FROM documents")).scalar_one() == "old.pdf"
        assert connection.execute(text("SELECT ocr_text FROM pages")).scalar_one() is None

    db._ensure_review_columns()  # running it again is a no-op
