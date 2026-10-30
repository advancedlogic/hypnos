"""The retry policy of docs/api.md section 17."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .errors import (
    APIConnectionError,
    APITimeoutError,
    HypnosError,
    InsufficientMemoryError,
    ModelBusyError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
)


@dataclass(frozen=True)
class RetryPolicy:
    """When and how long to wait before sending a request again.

    Only failures that arrive before any response body are retried: a stream that produced
    output is never replayed. ``max_retries=0`` turns retrying off.
    """

    max_retries: int = 4
    #: First wait after 429 ``model_busy`` (no ``Retry-After``); doubles each attempt.
    busy_backoff: float = 0.25
    #: 503 without ``Retry-After``.
    unavailable_wait: float = 5.0
    #: First wait after 507 (another model holds the memory); doubles each attempt.
    memory_backoff: float = 2.0
    #: First wait after 504 or a connection error; doubles each attempt.
    backoff: float = 1.0
    #: Upper bound of any single wait.
    max_wait: float = 60.0

    def delay(self, error: HypnosError, attempt: int) -> Optional[float]:
        """Seconds to wait before attempt ``attempt + 1`` (``attempt`` counts from 0), or None."""
        if attempt >= self.max_retries:
            return None
        retry_after = getattr(error, "retry_after", None)
        if isinstance(error, ModelBusyError):
            wait = self.busy_backoff * 2**attempt
        elif isinstance(error, (RateLimitError, ServiceUnavailableError)):
            wait = retry_after if retry_after is not None else self.unavailable_wait
        elif isinstance(error, InsufficientMemoryError):
            wait = max(retry_after or 0.0, self.memory_backoff * 2**attempt)
        elif isinstance(error, APITimeoutError) and error.status == 504:
            wait = self.backoff * 2**attempt
        elif isinstance(error, APIConnectionError):
            wait = self.backoff * 2**attempt
        elif isinstance(error, ServerError) and error.status == 500 and attempt == 0:
            wait = self.backoff
        else:
            return None
        # Up to 10 % jitter so that many clients do not come back in lockstep.
        return min(wait * (1 + random.random() * 0.1), self.max_wait)


NO_RETRY = RetryPolicy(max_retries=0)
