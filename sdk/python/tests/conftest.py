"""Starts the mock Hypnos server (the real router with mock engines) for the tests.

The binary is `$HYPNOS_MOCK_SERVER` when set, else `backend/target/debug/examples/mock_server`,
built with cargo when it is missing.
"""

from __future__ import annotations

import os
import pathlib
import platform
import subprocess
import sys
from typing import Iterator, List

import httpx
import pytest

from hypnos_client import Client, RetryPolicy

REPO = pathlib.Path(__file__).resolve().parents[3]
BACKEND = REPO / "backend"
TARGET = BACKEND / "target" / "debug"
EXE = ".exe" if sys.platform == "win32" else ""

#: Retries without the real waits, so the retry tests take milliseconds.
FAST_RETRY = RetryPolicy(busy_backoff=0.01, memory_backoff=0.01, backoff=0.01, unavailable_wait=0.01)


def mock_binary() -> pathlib.Path:
    configured = os.environ.get("HYPNOS_MOCK_SERVER")
    if configured:
        return pathlib.Path(configured)
    binary = TARGET / "examples" / f"mock_server{EXE}"
    if not binary.exists():
        features = ["--no-default-features"]
        if sys.platform == "darwin" and platform.machine() == "arm64":
            features += ["--features", "metal"]
        subprocess.run(
            ["cargo", "build", "-p", "hypnos-server", "--example", "mock_server", *features],
            cwd=BACKEND,
            check=True,
        )
    return binary


class MockServer:
    def __init__(self, args: List[str]) -> None:
        binary = mock_binary()
        env = dict(os.environ)
        libs = os.pathsep.join([str(binary.parent.parent), str(binary.parent.parent / "deps")])
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"):
            env[var] = libs + os.pathsep + env.get(var, "")
        self.process = subprocess.Popen(
            [str(binary), "--port", "0", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        assert self.process.stdout is not None
        line = self.process.stdout.readline().strip()
        if not line.startswith("listening on "):
            self.process.kill()
            raise RuntimeError(f"mock server did not start: {line!r}")
        self.url = line.removeprefix("listening on ")

    def control(self, path: str, body: dict) -> None:
        httpx.post(f"{self.url}/mock/{path}", json=body).raise_for_status()

    def fail(self, model: str, error: str, times: int = 1) -> None:
        self.control("fail", {"model": model, "error": error, "times": times})

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


@pytest.fixture(scope="session")
def server() -> Iterator[MockServer]:
    s = MockServer([])
    yield s
    s.stop()


@pytest.fixture(scope="session")
def keyed_server() -> Iterator[MockServer]:
    # `limited` (1 request a minute) is for the rate limit test, `member` is any non-admin key.
    s = MockServer(
        ["--api-key", "secret", "--limited-key", "limited:1", "--limited-key", "member:1000"]
    )
    yield s
    s.stop()


@pytest.fixture(autouse=True)
def reset(request: pytest.FixtureRequest) -> None:
    if "server" in request.fixturenames:
        request.getfixturevalue("server").control("reset", {})


@pytest.fixture
def client(server: MockServer) -> Iterator[Client]:
    with Client(server.url, retry=FAST_RETRY) as c:
        yield c


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
