from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "sqlite:///./ocr_system.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def _create_engine(database_url: str):
    connect_args: dict[str, bool | int] = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    if database_url.startswith("postgres") or database_url.startswith("postgresql"):
        connect_args["connect_timeout"] = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
    return create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)


engine = _create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

from .models import Base
from .auth import CurrentUserId

if DATABASE_URL.startswith("postgres") or DATABASE_URL.startswith("postgresql"):
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

if DATABASE_URL.startswith("sqlite") and os.getenv("RUN_MIGRATIONS", "false").strip().lower() not in {"1", "true", "yes", "on"}:
    Base.metadata.create_all(bind=engine)


def _ensure_document_type_column() -> None:
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return

    document_columns = {column["name"] for column in inspector.get_columns("documents")}
    if "document_type" in document_columns:
        return

    logger.warning("Applying compatibility migration: adding documents.document_type column")
    with engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE documents ADD COLUMN document_type VARCHAR(64) NOT NULL DEFAULT 'unknown'")
        )


def _ensure_document_owner_column() -> None:
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return

    document_columns = {column["name"] for column in inspector.get_columns("documents")}
    if "owner_id" in document_columns:
        return

    logger.warning("Applying compatibility migration: adding documents.owner_id column")
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE documents ADD COLUMN owner_id VARCHAR(128)"))


def _ensure_review_columns() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    if "documents" in table_names:
        document_columns = {column["name"] for column in inspector.get_columns("documents")}
        if "extracted_attributes" not in document_columns:
            logger.warning("Applying compatibility migration: adding documents.extracted_attributes column")
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE documents ADD COLUMN extracted_attributes JSON NOT NULL DEFAULT '[]'")
                )

        document_compatibility_columns = {
            "processing_started_at": "DATETIME",
            "processing_finished_at": "DATETIME",
            "processing_duration_seconds": "FLOAT",
        }
        for column_name, column_type in document_compatibility_columns.items():
            if column_name not in document_columns:
                logger.warning("Applying compatibility migration: adding documents.%s column", column_name)
                with engine.begin() as connection:
                    connection.execute(text(f"ALTER TABLE documents ADD COLUMN {column_name} {column_type}"))

    if "pages" in table_names:
        page_columns = {column["name"] for column in inspector.get_columns("pages")}
        if "storage_key" not in page_columns:
            logger.warning("Applying compatibility migration: adding pages.storage_key column")
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE pages ADD COLUMN storage_key VARCHAR(512)"))

    if "attributes" in table_names:
        attribute_columns = {column["name"] for column in inspector.get_columns("attributes")}
        if "source" not in attribute_columns:
            logger.warning("Applying compatibility migration: adding attributes.source column")
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE attributes ADD COLUMN source VARCHAR(32) NOT NULL DEFAULT 'model'")
                )


_ensure_document_type_column()
_ensure_document_owner_column()
_ensure_review_columns()


def _ensure_search_indexes() -> None:
    """Create helpful indexes for search queries if they do not exist.

    Creates simple btree indexes that work on both Postgres and SQLite
    for columns frequently used in WHERE clauses.
    """
    try:
        with engine.begin() as connection:
            # attributes.entity_type_label
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS idx_attributes_entity_type_label ON attributes (entity_type_label)")
            )
            # attributes.extracted_value (simple index; full-text is optional)
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS idx_attributes_extracted_value ON attributes (extracted_value)")
            )
            # documents.document_type
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS idx_documents_document_type ON documents (document_type)")
            )
            # documents.status
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status)"))
            # attributes.confidence_score
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS idx_attributes_confidence_score ON attributes (confidence_score)")
            )
    except Exception:
        logger.exception("Failed to create search indexes; continuing without failing startup")


if os.getenv("RUN_MIGRATIONS", "false").strip().lower() not in {"1", "true", "yes", "on"}:
    _ensure_search_indexes()


def set_rls_user(session: Session, user_id: str | None) -> None:
    if user_id and session.bind is not None and session.bind.dialect.name == "postgresql":
        session.execute(
            text("SELECT set_config('app.user_id', :user_id, true)"),
            {"user_id": user_id},
        )


def get_db(user_id: CurrentUserId) -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        set_rls_user(db, user_id)
        yield db
    finally:
        db.close()