from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

# app/ and preprocessing/ are not packages -- their modules import each other
# flatly (`from settings import ...`), which only works because app/ is on
# sys.path at runtime. preprocessing/ goes first so app/ ends up ahead of it.
ASSISTANT_DIR = Path(__file__).resolve().parent.parent
for _dir in (ASSISTANT_DIR / "preprocessing", ASSISTANT_DIR / "app"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import generate  # noqa: E402
import retrieve  # noqa: E402


@pytest.fixture(autouse=True)
def reset_module_caches():
    def clear():
        retrieve._client = None
        retrieve._collections = None
        retrieve._bm25_cache = None
        retrieve._reranker = None
        retrieve._symbol_vocab = None
        retrieve._embed_query.cache_clear()
        generate._ollama_client = None

    clear()
    yield
    clear()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("test attempted a real connection; stub the client")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
