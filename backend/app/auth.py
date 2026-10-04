from __future__ import annotations

import os
import re
import httpx
from functools import lru_cache
from typing import Annotated, Any
from urllib.parse import urlsplit
from uuid import UUID

try:
    from email_validator import EmailNotValidError, validate_email
except ImportError:  # pragma: no cover - optional dependency, fallback to basic checks
    EmailNotValidError = ValueError

    def validate_email(value: str) -> dict[str, str]:
        normalized = value.strip()
        if "@" not in normalized or normalized.count("@") != 1:
            raise ValueError("Invalid email address")
        local, domain = normalized.split("@", 1)
        if not local or not domain or "." not in domain:
            raise ValueError("Invalid email address")
        return {"email": normalized.lower()}

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import ColumnElement

try:
    import jwt
except ImportError:  # pragma: no cover - dependency is required in deployment
    jwt = None

bearer_scheme = HTTPBearer(auto_error=False)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
JWT_AUDIENCE = "authenticated"
ASYMMETRIC_JWK_TYPES = {"RSA", "EC", "OKP"}


@lru_cache(maxsize=8)
def _get_jwks_client(jwks_url: str) -> Any:
    if jwt is None or not hasattr(jwt, "PyJWKClient"):
        raise RuntimeError("JWT JWKS support is unavailable")
    return jwt.PyJWKClient(jwks_url, timeout=5)


def normalize_email(email: str) -> str:
    value = (email or "").strip().lower()
    if not value:
        return ""
    try:
        return validate_email(value)["email"]
    except (TypeError, ValueError):
        return value


def validate_registration(email: str, password: str, confirm_password: str) -> None:
    normalized = normalize_email(email)
    if not normalized:
        raise ValueError("Email is required.")
    if not EMAIL_PATTERN.fullmatch(normalized):
        raise ValueError("Enter a valid email address.")
    if not password or not confirm_password:
        raise ValueError("Password and confirmation are required.")
    if password != confirm_password:
        raise ValueError("Passwords do not match.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if not any(character.isupper() for character in password):
        raise ValueError("Password must include at least one uppercase letter.")
    if not any(character.islower() for character in password):
        raise ValueError("Password must include at least one lowercase letter.")
    if not any(character.isdigit() for character in password):
        raise ValueError("Password must include at least one number.")
    if not any(not character.isalnum() for character in password):
        raise ValueError("Password must include at least one symbol.")


def auth_required() -> bool:
    return os.getenv("AUTH_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}


def owner_scope(column: Any, user_id: str) -> ColumnElement[bool]:
    return column == user_id


def _verify_active_supabase_user(supabase_url: str, publishable_key: str, token: str, user_id: str) -> None:
    try:
        response = httpx.get(
            f"{supabase_url}/auth/v1/user",
            headers={"apikey": publishable_key, "Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable",
        ) from exc

    if response.status_code != 200:
        if 400 <= response.status_code < 500:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable",
        )

    try:
        user = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token") from exc

    if not isinstance(user, dict) or user.get("id") != user_id or user.get("deleted_at"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")


def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str:
    if credentials is None:
        if auth_required():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
        return os.getenv("DEV_USER_ID", "local-development-user")

    if jwt is None:
        raise HTTPException(status_code=500, detail="JWT authentication dependency is unavailable")

    token = credentials.credentials
    if not hasattr(jwt, "PyJWKClient"):
        raise HTTPException(status_code=500, detail="JWT JWKS support is unavailable")

    supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    if not supabase_url:
        raise HTTPException(status_code=500, detail="Supabase authentication is not configured")
    publishable_key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
    if not publishable_key:
        raise HTTPException(status_code=500, detail="Supabase authentication is not configured")
    try:
        parsed_url = urlsplit(supabase_url)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Supabase authentication is not configured") from exc
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc or parsed_url.query or parsed_url.fragment:
        raise HTTPException(status_code=500, detail="Supabase authentication is not configured")

    issuer = f"{supabase_url}/auth/v1"
    jwks_url = f"{issuer}/.well-known/jwks.json"
    try:
        header = jwt.get_unverified_header(token)
        if not isinstance(header.get("kid"), str) or not header["kid"]:
            raise jwt.InvalidTokenError("JWT signing key id is missing")

        signing_key = _get_jwks_client(jwks_url).get_signing_key_from_jwt(token)
        algorithm = signing_key.algorithm_name
        if (
            signing_key.key_type not in ASYMMETRIC_JWK_TYPES
            or not algorithm
            or header.get("alg") != algorithm
        ):
            raise jwt.InvalidAlgorithmError("JWT algorithm does not match its signing key")

        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=[algorithm],
            audience=JWT_AUDIENCE,
            issuer=issuer,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
    except (jwt.PyJWTError, jwt.PyJWKClientError, OSError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token") from exc

    user_id = claims.get("sub")
    if not isinstance(user_id, str):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
    try:
        verified_user_id = str(UUID(user_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token") from exc

    _verify_active_supabase_user(supabase_url, publishable_key, token, verified_user_id)
    return verified_user_id


CurrentUserId = Annotated[str, Depends(get_current_user_id)]
