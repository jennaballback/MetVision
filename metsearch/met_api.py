"""Client for the Met Collection API with rate limiting and retries.

Endpoints used:
  GET /departments
  GET /objects?departmentIds=N[&metadataDate=YYYY-MM-DD]   -> all IDs (optionally updated since)
  GET /objects/{id}                                         -> one object's metadata
"""
from __future__ import annotations

import logging
import random
import threading
import time
from datetime import date

import requests

log = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket shared by all threads that use one client."""

    def __init__(self, rate_per_second: float, burst: int | None = None):
        self.rate = rate_per_second
        self.capacity = burst or max(1, int(rate_per_second))
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)


class MetApiError(RuntimeError):
    pass


class MetClient:
    RETRY_STATUSES = {403, 429, 500, 502, 503, 504}  # the Met's CDN answers bursts with 403

    def __init__(self, base_url: str, rate_per_second: float = 15, user_agent: str = "metvision",
                 max_retries: int = 5, timeout: float = 30, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.limiter = RateLimiter(rate_per_second)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=16)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, path: str, params: dict | None = None, allow_404: bool = False) -> tuple[int, dict | None]:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                status, err = None, exc
            else:
                status, err = resp.status_code, None
                if status == 200:
                    return 200, resp.json()
                if status == 404 and allow_404:
                    return 404, None
                if status not in self.RETRY_STATUSES:
                    raise MetApiError(f"GET {url} -> {status}: {resp.text[:1000]}")
            if attempt == self.max_retries:
                raise MetApiError(f"GET {url} failed after {attempt + 1} attempts (last: {status or err})")
            # exponential backoff with jitter, honouring Retry-After when given
            delay = min(60.0, 2 ** attempt) + random.random()
            if status == 429 and resp.headers.get("Retry-After", "").isdigit():
                delay = max(delay, float(resp.headers["Retry-After"]))
            log.warning("GET %s -> %s; retrying in %.1fs", path, status or err, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")

    def departments(self) -> list[dict]:
        return self._get("/departments")[1]["departments"]

    def object_ids(self, department_id: int, updated_since: date | None = None) -> list[int]:
        params: dict = {"departmentIds": department_id}
        if updated_since:
            params["metadataDate"] = updated_since.isoformat()
        data = self._get("/objects", params)[1]
        return data.get("objectIDs") or []

    def get_object(self, object_id: int) -> tuple[int, dict | None]:
        """Returns (http_status, payload). 404 means the object was withdrawn."""
        return self._get(f"/objects/{object_id}", allow_404=True)
