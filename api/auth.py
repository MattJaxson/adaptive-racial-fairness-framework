"""
API key authentication middleware for the fairness audit service.
"""

import logging
import os

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

SKIP_AUTH_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


def _load_valid_keys() -> set[str]:
    raw = os.environ.get("API_KEYS", "dev-key-12345")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    return keys


# Loaded once at module import time; refreshed on process restart.
_VALID_KEYS: set[str] = _load_valid_keys()


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces X-API-Key header authentication."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in SKIP_AUTH_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in _VALID_KEYS:
            logger.warning(
                "Unauthorized request to %s — missing or invalid API key",
                request.url.path,
            )
            return Response(
                content='{"detail": "Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)
