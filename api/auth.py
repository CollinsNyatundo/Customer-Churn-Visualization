"""
api/auth.py
------------
API key authentication via request header.
API keys are stored as a comma-separated env var: API_KEYS=key1,key2

Usage
-----
Protected route:
    @app.post("/predict")
    async def predict(customer: CustomerFeatures, _: str = Depends(verify_api_key)):
        ...

Generating a key for local dev:
    python -c "import secrets; print(secrets.token_urlsafe(32))"
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_ENV_VAR = "API_KEYS"


@lru_cache(maxsize=1)
def _load_valid_keys() -> frozenset[str]:
    """Load API keys from environment. Cached after first call."""
    raw = os.getenv(_ENV_VAR, "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not keys:
        # Dev fallback: accept any non-empty key when no keys configured
        return frozenset()
    return frozenset(keys)


async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """
    FastAPI dependency — raises 401 if the key is missing or invalid.
    Set API_KEYS env var to enable enforcement. If unset, auth is bypassed
    (development mode).
    """
    valid_keys = _load_valid_keys()

    # Dev mode: no keys configured → allow all requests
    if not valid_keys:
        return api_key or "dev-bypass"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    if not any(secrets.compare_digest(api_key, k) for k in valid_keys):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
