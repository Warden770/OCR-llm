from __future__ import annotations

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from app import auth


os.environ.setdefault("AUTH_REQUIRED", "false")


@pytest.fixture
def supabase_auth_api(monkeypatch):
    supabase_url = "https://jwks-test.supabase.co"
    response_state: dict[str, Any] = {"status_code": 200, "payload": None}
    requests: list[dict[str, Any]] = []

    def mock_get(url: str, *, headers: dict[str, str], timeout: float):
        token = headers["Authorization"].removeprefix("Bearer ")
        claims = jwt.decode(token, options={"verify_signature": False})
        requests.append(
            {
                "url": url,
                "publishable_key_present": bool(headers.get("apikey")),
                "bearer_token_present": bool(token),
                "timeout": timeout,
            }
        )
        payload = response_state["payload"]
        if payload is None:
            payload = {"id": claims.get("sub")}
        return SimpleNamespace(status_code=response_state["status_code"], json=lambda: payload)

    monkeypatch.setattr(auth.httpx, "get", mock_get)
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test_key")
    return SimpleNamespace(
        requests=requests,
        set_response=lambda status_code, payload=None: response_state.update(
            status_code=status_code,
            payload=payload,
        ),
    )


@pytest.fixture
def supabase_jwt_signer(monkeypatch, supabase_auth_api):
    supabase_url = "https://jwks-test.supabase.co"
    issuer = f"{supabase_url}/auth/v1"
    jwks_url = f"{issuer}/.well-known/jwks.json"
    private_keys = {
        "RS256": rsa.generate_private_key(public_exponent=65537, key_size=2048),
        "ES256": ec.generate_private_key(ec.SECP256R1()),
    }
    key_ids = {algorithm: f"local-generated-{algorithm.lower()}" for algorithm in private_keys}
    jwk_algorithms = {"RS256": jwt.algorithms.RSAAlgorithm, "ES256": jwt.algorithms.ECAlgorithm}
    public_jwks = []
    for algorithm, key in private_keys.items():
        public_jwk = json.loads(jwk_algorithms[algorithm].to_jwk(key.public_key()))
        public_jwk.update(
            {"kid": key_ids[algorithm], "alg": algorithm, "use": "sig", "key_ops": ["verify"]}
        )
        public_jwks.append(public_jwk)
    jwks = {"keys": public_jwks}

    client = jwt.PyJWKClient(jwks_url, timeout=1)
    monkeypatch.setattr(client, "fetch_data", lambda: jwks)

    def resolve_client(requested_url: str) -> jwt.PyJWKClient:
        assert requested_url == jwks_url
        return client

    monkeypatch.setattr(auth, "_get_jwks_client", resolve_client)
    monkeypatch.setenv("SUPABASE_URL", supabase_url)

    def issue_token(
        user_id: str | None,
        claims: dict[str, Any] | None = None,
        *,
        algorithm: str = "RS256",
        kid: str | None = None,
        signing_key: Any | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "aud": "authenticated",
            "iss": issuer,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        if user_id is not None:
            payload["sub"] = user_id
        payload.update(claims or {})
        return jwt.encode(
            payload,
            signing_key or private_keys[algorithm],
            algorithm=algorithm,
            headers={"kid": kid or key_ids[algorithm]},
        )

    return issue_token