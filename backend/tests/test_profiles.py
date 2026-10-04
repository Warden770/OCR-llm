from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.main import app
from app.models import Base, Document, UserProfile


def _session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, future=True)


def test_profile_completion_trims_name_and_uses_authenticated_uuid(monkeypatch, supabase_jwt_signer) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    session_factory = _session_factory()
    user_id = str(uuid4())

    def override_get_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        headers = {"Authorization": f"Bearer {supabase_jwt_signer(user_id)}"}
        missing_profile = client.get("/api/v1/profile", headers=headers)
        assert missing_profile.status_code == 200
        assert missing_profile.json() == {"profile": None}

        response = client.put("/api/v1/profile", headers=headers, json={"name": "  Juan Dela Cruz  "})
        assert response.status_code == 200
        profile = response.json()["profile"]
        assert profile["user_id"] == user_id
        assert profile["name"] == "Juan Dela Cruz"
        assert profile["created_at"]
        assert profile["updated_at"]

        with session_factory() as db:
            assert db.get(UserProfile, user_id).name == "Juan Dela Cruz"
    finally:
        app.dependency_overrides.clear()


def test_duplicate_names_are_allowed_and_name_edit_preserves_ocr_owner(monkeypatch, supabase_jwt_signer) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    session_factory = _session_factory()
    account_a = str(uuid4())
    account_b = str(uuid4())

    def override_get_db():
        with session_factory() as db:
            yield db

    with session_factory() as db:
        document = Document(name="account-a.pdf", format="pdf", owner_id=account_a, status="PROCESSED")
        db.add(document)
        db.commit()
        document_id = document.id

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        account_a_headers = {"Authorization": f"Bearer {supabase_jwt_signer(account_a)}"}
        account_b_headers = {"Authorization": f"Bearer {supabase_jwt_signer(account_b)}"}
        account_a_profile = client.put(
            "/api/v1/profile",
            headers=account_a_headers,
            json={"name": "Elape"},
        )
        account_b_profile = client.put(
            "/api/v1/profile",
            headers=account_b_headers,
            json={"name": "Elape"},
        )

        assert account_a_profile.status_code == 200
        assert account_b_profile.status_code == 200
        assert account_a_profile.json()["profile"]["user_id"] == account_a
        assert account_b_profile.json()["profile"]["user_id"] == account_b

        renamed = client.put(
            "/api/v1/profile",
            headers=account_a_headers,
            json={"name": "Elape Updated"},
        )
        assert renamed.status_code == 200
        assert renamed.json()["profile"]["user_id"] == account_a
        assert renamed.json()["profile"]["name"] == "Elape Updated"
        assert client.get("/api/v1/documents", headers=account_a_headers).json()["documents"][0]["id"] == document_id
        assert client.get("/api/v1/documents", headers=account_b_headers).json()["documents"] == []
        assert client.get("/api/v1/profile", headers=account_b_headers).json()["profile"]["name"] == "Elape"

        with session_factory() as db:
            assert db.get(Document, document_id).owner_id == account_a
            assert db.get(UserProfile, account_a).name == "Elape Updated"
            assert db.get(UserProfile, account_b).name == "Elape"
    finally:
        app.dependency_overrides.clear()


def test_profile_name_validation_rejects_blank_short_and_long_values(monkeypatch, supabase_jwt_signer) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    session_factory = _session_factory()
    user_id = str(uuid4())

    def override_get_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        headers = {"Authorization": f"Bearer {supabase_jwt_signer(user_id)}"}
        assert client.put("/api/v1/profile", headers=headers, json={"name": "  "}).status_code == 422
        assert client.put("/api/v1/profile", headers=headers, json={"name": "A"}).status_code == 422
        assert client.put("/api/v1/profile", headers=headers, json={"name": "N" * 101}).status_code == 422
    finally:
        app.dependency_overrides.clear()
