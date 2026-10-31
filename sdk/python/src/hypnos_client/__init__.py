"""Python client for the Hypnos inference server (https://github.com/advancedlogic/hypnos).

    from hypnos_client import Client

    with Client() as hypnos:                      # http://127.0.0.1:9999, $HYPNOS_API_KEY
        reply = hypnos.chat("gemma4:26b-q4", [{"role": "user", "content": "Hello"}])
        print(reply["choices"][0]["message"]["content"])

``AsyncClient`` has the same methods as coroutines. The API itself is described in
docs/api.md of the Hypnos repository.
"""

from . import types
from ._async_client import AsyncClient
from ._base import DEFAULT_BASE_URL, Speech, SpeechTelemetry, __version__
from ._client import Client
from .errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    HypnosError,
    InsufficientMemoryError,
    ModelBusyError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    StreamError,
)
from .retry import NO_RETRY, RetryPolicy
from .tools import assistant_message, function_tool, parse_arguments, tool_message
from .streams import (
    AsyncChatStream,
    AsyncDocumentStream,
    AsyncSpeechStream,
    ChatStream,
    DocumentStream,
    SpeechStream,
)

__all__ = [
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "AsyncChatStream",
    "AsyncClient",
    "AsyncDocumentStream",
    "AsyncSpeechStream",
    "AuthenticationError",
    "BadRequestError",
    "ChatStream",
    "Client",
    "DEFAULT_BASE_URL",
    "DocumentStream",
    "HypnosError",
    "InsufficientMemoryError",
    "ModelBusyError",
    "NO_RETRY",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "RetryPolicy",
    "ServerError",
    "ServiceUnavailableError",
    "Speech",
    "SpeechStream",
    "SpeechTelemetry",
    "StreamError",
    "__version__",
    "assistant_message",
    "function_tool",
    "parse_arguments",
    "tool_message",
    "types",
]
