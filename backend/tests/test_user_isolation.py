from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.main import app
from app.models import Attribute, Base, Document, Page
from app import ingestion_pipeline


def test_attribute_cannot_reference_page_from_another_document() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, future=True)
    session = session_factory()
    first = Document(name="first.pdf", format="pdf", status="PROCESSED")
    second = Document(name="second.pdf", format="pdf", status="PROCESSED")
    session.add_all([first, second])
    session.flush()
    page = Page(document_id=first.id, page_number=1, image_path="page-001.png")
    session.add(page)
    session.flush()

    session.add(
        Attribute(
            page_id=page.id,
            document_id=second.id,
            entity_type_label="TOTAL",
            extracted_value="10.00",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.close()


def test_users_cannot_access_each_others_documents(monkeypatch, supabase_jwt_signer) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, future=True)
    session = session_factory()
    first_user_id = str(uuid4())
    second_user_id = str(uuid4())
    first = Document(name="first.pdf", format="pdf", owner_id=first_user_id, status="PROCESSED")
    second = Document(name="second.pdf", format="pdf", owner_id=second_user_id, status="PROCESSED")
    session.add_all([first, second])
    session.commit()
    session.refresh(first)
    session.refresh(second)

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        first_headers = {"Authorization": f"Bearer {supabase_jwt_signer(first_user_id)}"}
        second_headers = {"Authorization": f"Bearer {supabase_jwt_signer(second_user_id)}"}
        limited_response = client.get("/api/v1/documents?limit=1", headers=first_headers)
        assert limited_response.status_code == 200
        assert len(limited_response.json()["documents"]) == 1
        assert client.get("/api/v1/documents", headers=first_headers).json()["documents"][0]["name"] == "first.pdf"
        assert client.get("/api/v1/documents", headers=second_headers).json()["documents"][0]["name"] == "second.pdf"
        assert client.get(f"/api/v1/documents/{second.id}", headers=first_headers).status_code == 404
        assert client.get(f"/api/v1/documents/{first.id}", headers=second_headers).status_code == 404
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_deleted_account_cannot_access_existing_document(monkeypatch, supabase_jwt_signer, supabase_auth_api) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    supabase_auth_api.set_response(404, {"message": "User not found"})
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, future=True)
    account_id = str(uuid4())
    with session_factory() as session:
        document = Document(name="account-a.pdf", format="pdf", owner_id=account_id, status="PROCESSED")
        session.add(document)
        session.commit()
        document_id = document.id

    def override_get_db():
        nonlocal database_accessed
        database_accessed = True
        with session_factory() as db:
            yield db

    database_accessed = False
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        response = client.get(
            f"/api/v1/documents/{document_id}",
            headers={"Authorization": f"Bearer {supabase_jwt_signer(account_id)}"},
        )

        assert response.status_code == 401
        assert database_accessed is False
    finally:
        app.dependency_overrides.clear()


def test_uploaded_document_owner_comes_from_verified_sub(monkeypatch, supabase_jwt_signer, tmp_path) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, future=True)
    authenticated_user_id = str(uuid4())

    def override_get_db():
        with session_factory() as db:
            yield db

    def queue_job(*args):
        return SimpleNamespace(id="local-test-job")

    monkeypatch.setattr(ingestion_pipeline, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(ingestion_pipeline.celery_app.conf, "task_always_eager", False)
    monkeypatch.setattr(ingestion_pipeline.process_document_job, "delay", queue_job)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        response = client.post(
            "/api/v1/ingest",
            headers={"Authorization": f"Bearer {supabase_jwt_signer(authenticated_user_id)}"},
            data={
                "user_id": str(uuid4()),
                "owner_id": str(uuid4()),
                "device_id": "untrusted-browser-id",
            },
            files={"file": ("test.png", b"not-rasterized-by-the-mocked-worker", "image/png")},
        )

        assert response.status_code == 200
        document_id = response.json()["document_id"]
        with session_factory() as db:
            document = db.get(Document, document_id)
            assert document is not None
            assert document.owner_id == authenticated_user_id
    finally:
        app.dependency_overrides.clear()


def test_raw_page_directory_is_not_public() -> None:
    client = TestClient(app)
    response = client.get("/pages/1/page-001.png")
    assert response.status_code == 404
