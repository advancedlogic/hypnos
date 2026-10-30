"""Exceptions raised by the client, one per error ``type`` of docs/api.md section 4."""

from __future__ import annotations

from typing import Any, Dict, Optional


class HypnosError(Exception):
    """Base class of everything the client raises."""


class APIConnectionError(HypnosError):
    """The server could not be reached, or the connection broke before a response arrived."""


class APIError(HypnosError):
    """The server answered with an error status (or failed a stream it had started)."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        type: Optional[str] = None,
        code: Optional[str] = None,
        request_id: Optional[str] = None,
        retry_after: Optional[float] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.type = type
        self.code = code
        self.request_id = request_id
        self.retry_after = retry_after
        self.body = body

    def __str__(self) -> str:
        kind = f" {self.type}" if self.type else ""
        rid = f" (request {self.request_id})" if self.request_id else ""
        return f"HTTP {self.status}{kind}: {self.message}{rid}"


class BadRequestError(APIError):
    """400 ``invalid_request_error``, or 413 (body above ``max_body_bytes``)."""


class AuthenticationError(APIError):
    """401: missing or unknown API key."""


class NotFoundError(APIError):
    """404 ``model_not_found``, ``model_disabled`` or ``not_found``."""


class ModelBusyError(APIError):
    """429 ``model_busy``: the model's queue is full, or a placement change met a leased model."""


class RateLimitError(APIError):
    """429 ``rate_limit_error``: a per-key limit; ``code`` says which, ``retry_after`` when."""


class ServiceUnavailableError(APIError):
    """503 ``model_not_available``: still staging the startup models, or a configuration problem."""


class InsufficientMemoryError(APIError):
    """507: the model does not fit next to the leased ones, or the engine ran out of memory."""


class APITimeoutError(APIError):
    """408 (no response headers within ``request_timeout_secs``) or 504 ``timeout``."""


class ServerError(APIError):
    """500 ``server_error`` (an engine failure) or any other 5xx."""


class StreamError(APIError):
    """A stream that had already started (HTTP 200) ended with an error event."""


def error_class(status: int, type: Optional[str]) -> type[APIError]:
    if status in (400, 413):
        return BadRequestError
    if status == 401:
        return AuthenticationError
    if status == 404:
        return NotFoundError
    if status == 429:
        return RateLimitError if type == "rate_limit_error" else ModelBusyError
    if status == 503:
        return ServiceUnavailableError
    if status == 507:
        return InsufficientMemoryError
    if status in (408, 504):
        return APITimeoutError
    if status >= 500:
        return ServerError
    return APIError
