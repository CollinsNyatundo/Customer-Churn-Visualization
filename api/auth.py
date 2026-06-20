"""
api/auth.py
------------
API key authentication.

Modes
-----
- AUTH ENFORCED: API_KEYS env var is set → all protected routes require
  a valid X-API-Key header (401 if missing, 403 if invalid).
- DEV BYPASS: API_KEYS is unset → requests pass through, but a WARNING
  is printed at startup and on every bypassed request.
- PRODUCTION GUARD: ENV=production + API_KEYS unset → startup raises,
  preventing accidental open deployments.
"""

from __future__ import annotations

import logging
import os
import secrets
from functools import lru_cache

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config import settings

log = logging.getLogger(__name__)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


@lru_cache(maxsize=1)
def _load_valid_keys() -> frozenset[str]:
    raw = os.getenv("API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not keys:
        if settings.env == "production":
            raise RuntimeError(
                "API_KEYS must be set in production mode (ENV=production). "
                'Generate a key with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        log.warning(
            "⚠️  API authentication BYPASSED — API_KEYS is not set. "
            "This is acceptable for local development only. "
            "Set API_KEYS and ENV=production before deploying."
        )
    return frozenset(keys)


async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    valid_keys = _load_valid_keys()

    # Dev bypass
    if not valid_keys:
        log.debug("API auth bypassed (dev mode).")
        return api_key or "dev-bypass"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not any(secrets.compare_digest(api_key, k) for k in valid_keys):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
