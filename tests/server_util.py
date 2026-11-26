"""Helpers to run the FastAPI app in a background uvicorn thread for tests."""
from __future__ import annotations

import socket
import threading
import time
from dataclasses import replace
from typing import Any

import httpx
import uvicorn

from app.config import Settings, get_settings
from app.main import create_app


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ServerHandle:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.port = settings.port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ServerHandle":
        app = create_app(self.settings)
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            lifespan="on",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                r = httpx.get(self.base_url + "/health", timeout=1.0)
                if r.status_code == 200:
                    return self
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("test server did not become healthy")

    def __exit__(self, *exc: Any) -> None:
        assert self._server is not None and self._thread is not None
        self._server.should_exit = True
        self._thread.join(timeout=10)


def make_settings(**overrides: Any) -> Settings:
    base = get_settings()
    # Tests always bind an ephemeral port unless explicitly given.
    overrides.setdefault("port", free_port())
    overrides.setdefault("host", "127.0.0.1")
    return replace(base, **overrides)
