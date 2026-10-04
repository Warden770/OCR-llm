from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.attributes_api import router as attributes_router
from app.ingestion_pipeline import router as ingestion_router
from app.debug_api import router as debug_router
from app.search_api import router as search_router
from app.profiles_api import router as profiles_router

app = FastAPI(
    title="OCR Attribute Extraction API",
    version="1.0.0",
)

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://ocr-elape.vercel.app",
]
configured_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", ",".join(default_origins)).split(",")
    if origin.strip()
]
merged_local_origins = list(dict.fromkeys(default_origins + configured_origins))

# Only allow permissive CORS in explicitly opted-in development environments.
# A default restricted allowlist is the safe production behavior.
raw_dev_flag = os.getenv("DEV_ALLOW_ALL_CORS")
env_name = os.getenv("ENV", "").strip().lower()
if raw_dev_flag is not None:
    allow_all_cors = raw_dev_flag.lower() in ("1", "true", "yes", "on")
elif env_name in {"development", "dev"}:
    allow_all_cors = True
else:
    allow_all_cors = False

if allow_all_cors:
    cors_mode = "CORS: permissive (all origins) — DEV ONLY"
    allowed_origins = ["*"]
    allow_credentials = False
else:
    cors_mode = f"CORS: restricted to [{', '.join(merged_local_origins)}]"
    allowed_origins = merged_local_origins
    allow_credentials = True

print(cors_mode)

# Star origins and credentials do not work together in browsers. Only use
# wildcard mode for explicitly opted-in local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router, prefix="/api/v1", tags=["ingestion"])
app.include_router(attributes_router, prefix="/api/v1", tags=["attributes"])
app.include_router(profiles_router, prefix="/api/v1", tags=["profiles"])
debug_enabled = os.getenv("ENABLE_DEBUG_ENDPOINTS", "false").strip().lower() in {"1", "true", "yes", "on"}
debug_environment = os.getenv("ENV", "").strip().lower() in {"development", "dev", "test"}
if debug_enabled and debug_environment:
    app.include_router(debug_router, prefix="/api/v1", tags=["debug"])
app.include_router(search_router, prefix="/api/v1", tags=["search"])

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
