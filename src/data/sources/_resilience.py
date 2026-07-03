"""
src/data/sources/_resilience.py
---------------------------------
Circuit breaker + exponential-backoff retry for all external API sources.

One CircuitBreaker instance per source_name. Opens after fail_max failures,
resets after reset_timeout_seconds. All API sources import resilient_get().
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any
from urllib.parse import urlencode

log = logging.getLogger(__name__)


@dataclass
class _Breaker:
    fail_max: int = 3
    reset_timeout: int = 60
    _count: int = 0
    _opened_at: datetime | None = None
    _lock: Lock = field(default_factory=Lock)

    def trip(self) -> None:
        with self._lock:
            self._count += 1
            if self._count >= self.fail_max:
                self._opened_at = datetime.utcnow()
                log.error("Circuit breaker OPEN (%d failures). Reset in %ds.", self._count, self.reset_timeout)

    def success(self) -> None:
        with self._lock:
            self._count = 0
            self._opened_at = None

    @property
    def open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if datetime.utcnow() - self._opened_at > timedelta(seconds=self.reset_timeout):
                self._opened_at = None  # half-open: allow one test
                return False
            return True


_breakers: dict[str, _Breaker] = {}


def resilient_get(
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    source_name: str = "api",
    max_retries: int = 3,
    timeout: int = 30,
    backoff: float = 2.0,
) -> dict | list:
    """HTTP GET with circuit breaker + exponential backoff."""
    if source_name not in _breakers:
        _breakers[source_name] = _Breaker()
    breaker = _breakers[source_name]

    if breaker.open:
        raise RuntimeError(
            f"Circuit breaker OPEN for '{source_name}' — API is unhealthy. "
            "Use synthetic fallback (USE_CRM_API=false) or wait for reset."
        )

    full_url = (url + "?" + urlencode(params)) if params else url
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(full_url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            breaker.success()
            return data
        except Exception as exc:
            last_exc = exc
            breaker.trip()
            wait = backoff * (2 ** (attempt - 1))
            log.warning("%s attempt %d/%d failed: %s. Retry in %.1fs", source_name, attempt, max_retries, exc, wait)
            if attempt < max_retries:
                time.sleep(wait)

    raise RuntimeError(f"{source_name}: {max_retries} retries exhausted. Last: {last_exc}") from last_exc
