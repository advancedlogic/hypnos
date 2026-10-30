"""Generates the synchronous client (`_client.py`) from the asynchronous one.

    python scripts/unasync.py           # rewrite _client.py
    python scripts/unasync.py --check   # exit 1 when _client.py is out of date (tests run this)
"""

from __future__ import annotations

import pathlib
import sys

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "src" / "hypnos_client"
SOURCE = PACKAGE / "_async_client.py"
TARGET = PACKAGE / "_client.py"

HEADER = '"""The synchronous client.\n\nGENERATED from ``_async_client.py`` by ``scripts/unasync.py``; do not edit by hand."""\n'

# Order matters: longer and more specific tokens first.
REPLACEMENTS = [
    ("import asyncio\n", ""),
    ("AsyncChatStream", "ChatStream"),
    ("AsyncDocumentStream", "DocumentStream"),
    ("AsyncSpeechStream", "SpeechStream"),
    ("httpx.AsyncClient", "httpx.Client"),
    ("AsyncClient", "Client"),
    ("asyncio.Semaphore", "threading.BoundedSemaphore"),
    ("await asyncio.sleep(", "time.sleep("),
    ("async def ", "def "),
    ("async with ", "with "),
    ("await ", ""),
    (".aread()", ".read()"),
    (".aclose()", ".close()"),
    ("def aclose(", "def close("),
    ("__aenter__", "__enter__"),
    ("__aexit__", "__exit__"),
]


def generate() -> str:
    text = SOURCE.read_text()
    # Replace the module docstring.
    end = text.index('"""', 3) + 3
    text = HEADER + text[end:]
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    left = [n for n, line in enumerate(text[len(HEADER):].splitlines(), 1) if "async" in line or "await" in line]
    if left:
        raise SystemExit(f"unasync: async tokens left on lines {left} of the generated client")
    return text


def main() -> int:
    generated = generate()
    if "--check" in sys.argv:
        if not TARGET.exists() or TARGET.read_text() != generated:
            print(f"{TARGET.name} is out of date: run python scripts/unasync.py", file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(generated)
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
