from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import get_current_user_id
from app.auth import (
    auth_required,
    normalize_email,
    owner_scope,
    validate_registration,
)
from app.main import app
from app.models import Document


def test_authentication_and_owner_scope_default_to_no_login(monkeypatch) -> None:
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    assert auth_required() is False
    assert "IS NULL" not in str(owner_scope(Document.owner_id, "user-1"))


def test_protected_routes_reject_missing_and_invalid_tokens(monkeypatch, supabase_jwt_signer, supabase_auth_api) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    client = TestClient(app)
    missing = client.get("/api/v1/documents")
    invalid = client.get("/api/v1/documents", headers={"Authorization": "Bearer invalid"})
    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert supabase_auth_api.requests == []


def test_valid_jwks_access_token_returns_verified_uuid_subject(
    monkeypatch,
    supabase_jwt_signer,
    supabase_auth_api,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    user_id = str(uuid4())
    token = supabase_jwt_signer(user_id)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    assert get_current_user_id(credentials) == user_id
    assert supabase_auth_api.requests == [
        {
            "url": "https://jwks-test.supabase.co/auth/v1/user",
            "publishable_key_present": True,
            "bearer_token_present": True,
            "timeout": 5.0,
        }
    ]


def test_deleted_supabase_user_is_rejected(supabase_jwt_signer, supabase_auth_api) -> None:
    supabase_auth_api.set_response(404, {"message": "User not found"})
    token = supabase_jwt_signer(str(uuid4()))

    with pytest.raises(HTTPException) as error:
        get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    assert error.value.status_code == 401


def test_revoked_supabase_session_is_rejected(supabase_jwt_signer, supabase_auth_api) -> None:
    supabase_auth_api.set_response(401, {"message": "Invalid token"})
    token = supabase_jwt_signer(str(uuid4()))

    with pytest.raises(HTTPException) as error:
        get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    assert error.value.status_code == 401


def test_supabase_user_must_match_verified_subject(supabase_jwt_signer, supabase_auth_api) -> None:
    token = supabase_jwt_signer(str(uuid4()))
    supabase_auth_api.set_response(200, {"id": str(uuid4())})

    with pytest.raises(HTTPException) as error:
        get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    assert error.value.status_code == 401


def test_es256_jwks_access_token_uses_advertised_algorithm(supabase_jwt_signer) -> None:
    user_id = str(uuid4())
    token = supabase_jwt_signer(user_id, algorithm="ES256")

    assert get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)) == user_id


def test_expired_access_token_is_rejected(supabase_jwt_signer) -> None:
    token = supabase_jwt_signer(
        str(uuid4()),
        {"exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
    )

    with pytest.raises(HTTPException) as error:
        get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    assert error.value.status_code == 401
    assert error.value.detail == "Invalid authentication token"


def test_invalid_signature_is_rejected(supabase_jwt_signer) -> None:
    unrelated_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = supabase_jwt_signer(str(uuid4()), signing_key=unrelated_key)

    with pytest.raises(HTTPException) as error:
        get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    assert error.value.status_code == 401


def test_wrong_issuer_is_rejected(supabase_jwt_signer) -> None:
    token = supabase_jwt_signer(str(uuid4()), {"iss": "https://other-project.example/auth/v1"})

    with pytest.raises(HTTPException) as error:
        get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    assert error.value.status_code == 401


def test_wrong_audience_is_rejected(supabase_jwt_signer) -> None:
    token = supabase_jwt_signer(str(uuid4()), {"aud": "other-audience"})

    with pytest.raises(HTTPException) as error:
        get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    assert error.value.status_code == 401


def test_missing_subject_is_rejected(supabase_jwt_signer) -> None:
    token = supabase_jwt_signer(None)

    with pytest.raises(HTTPException) as error:
        get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    assert error.value.status_code == 401


def test_non_uuid_subject_is_rejected(supabase_jwt_signer) -> None:
    token = supabase_jwt_signer("not-a-user-uuid")

    with pytest.raises(HTTPException) as error:
        get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    assert error.value.status_code == 401


def test_unknown_jwks_key_id_is_rejected(supabase_jwt_signer) -> None:
    token = supabase_jwt_signer(str(uuid4()), kid="unknown-key-id")

    with pytest.raises(HTTPException) as error:
        get_current_user_id(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    assert error.value.status_code == 401


def test_registration_inputs_are_normalized_and_validated() -> None:
    assert normalize_email("  STUDENT@EXAMPLE.COM  ") == "student@example.com"
    validate_registration("student@example.com", "StrongPass123!", "StrongPass123!")

    try:
        validate_registration("", "StrongPass123!", "StrongPass123!")
        raise AssertionError("empty email should have been rejected")
    except ValueError:
        pass

    try:
        validate_registration("student@example.com", "short", "short")
        raise AssertionError("weak password should have been rejected")
    except ValueError:
        pass

    try:
        validate_registration("student@example.com", "StrongPass123!", "DifferentPass123!")
        raise AssertionError("password mismatch should have been rejected")
    except ValueError:
        pass