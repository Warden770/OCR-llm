from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.auth import CurrentUserId
from app.db import get_db
from app.models import UserProfile

router = APIRouter()


class ProfileUpdate(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        name = value.strip()
        if len(name) < 2:
            raise ValueError("Name must be at least 2 characters long.")
        if len(name) > 100:
            raise ValueError("Name must be 100 characters or fewer.")
        if any(ord(character) < 32 for character in name):
            raise ValueError("Name contains unsupported control characters.")
        return name


def _serialize_profile(profile: UserProfile | None) -> dict[str, object] | None:
    if profile is None:
        return None
    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


@router.get("/profile")
def get_profile(user_id: CurrentUserId, db: Session = Depends(get_db)) -> dict[str, object]:
    profile = db.get(UserProfile, user_id)
    return {"profile": _serialize_profile(profile)}


@router.put("/profile")
def save_profile(
    payload: ProfileUpdate,
    user_id: CurrentUserId,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    profile = db.get(UserProfile, user_id)
    if profile is None:
        profile = UserProfile(user_id=user_id, name=payload.name)
        db.add(profile)
    else:
        profile.name = payload.name
        profile.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(profile)
    return {"profile": _serialize_profile(profile)}
