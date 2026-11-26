"""End-to-end HTTP semantics against a real uvicorn server in a thread."""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("httpx")
pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")
pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")

import httpx  # noqa: E402

from server_util import ServerHandle, make_settings  # noqa: E402


@pytest.fixture(scope="module")
def server():
    settings = make_settings(batch_window_ms=25.0, max_elements=4096)
    with ServerHandle(settings) as h:
        yield h


def test_health(server: ServerHandle):
    r = httpx.get(server.base_url + "/health", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/score", {"values": [0.1, -0.2, 0.3], "timeout_ms": 3000}),
        ("/predict", {"samples": [1, 2, 3, 4], "timeoutMs": 3000}),
        ("/anomaly", {"data": [0.5, 0.5], "timeout": 3000}),
        ("/api/v1/anomaly/score", {"signal": [0.1] * 8, "timeout_ms": 3000}),
        ("/some/unknown/deep/path", {"values": [1, 2], "timeout_ms": 3000}),
    ],
)
def test_score_aliases(server, path, payload):
    r = httpx.post(server.base_url + path, json=payload, timeout=5)
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["score"], float)
    assert body["status"] == "ok"
    assert body["success"] is True


def test_get_query_score(server):
    r = httpx.get(
        server.base_url + "/score",
        params={"values": "0.1,0.2,0.3,0.4", "timeout_ms": 3000},
        timeout=5,
    )
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["score"], float)


def test_bare_json_array(server):
    r = httpx.post(
        server.base_url + "/score?timeout_ms=3000",
        content=b"[0.2,0.4,0.6]",
        headers={"content-type": "application/json"},
        timeout=5,
    )
    assert r.status_code == 200, r.text
    assert r.json()["sequence_length"] == 3


def test_invalid_body(server):
    r = httpx.post(server.base_url + "/score", json={"foo": [1]}, timeout=5)
    assert r.status_code >= 400
    body = r.json()
    assert body["success"] is False


def test_score_consistency_solo_vs_batch(server):
    import random

    rng = random.Random(1234)
    sequences = [
        [round(rng.uniform(-4, 4), 5) for _ in range(n)]
        for n in (1, 2, 5, 9, 17, 32, 64, 100, 41, 7)
    ]
    solo = []
    with httpx.Client(timeout=10) as c:
        for seq in sequences:
            r = c.post(server.base_url + "/score", json={"values": seq, "timeout_ms": 5000})
            assert r.status_code == 200, r.text
            solo.append(r.json()["score"])

        results: list = [None] * len(sequences)

        def do(i, seq):
            rr = c.post(server.base_url + "/score", json={"values": seq, "timeout_ms": 5000})
            results[i] = rr

        threads = [threading.Thread(target=do, args=(i, s)) for i, s in enumerate(sequences)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    for i, r in enumerate(results):
        assert r.status_code == 200, getattr(r, "text", None)
        got = r.json()["score"]
        assert got == solo[i] or abs(got - solo[i]) <= 1e-7


def test_oversize_single_rejected_immediately(server):
    too_big = server.settings.max_elements + 1
    t0 = time.perf_counter()
    r = httpx.post(
        server.base_url + "/score",
        json={"values": [0.01] * too_big, "timeout_ms": 5000},
        timeout=10,
    )
    elapsed = time.perf_counter() - t0
    assert r.status_code == server.settings.too_large_status_code
    assert elapsed < 2.0  # rejected without waiting for the batch window
    body = r.json()
    assert body["status"] == "rejected"
    assert body["success"] is False


def test_exact_budget_accepted(server):
    exact = server.settings.max_elements
    r = httpx.post(
        server.base_url + "/score",
        json={"values": [0.0] * exact, "timeout_ms": 10000},
        timeout=15,
    )
    assert r.status_code == 200, r.text


def test_multi_instances_envelope(server):
    r = httpx.post(
        server.base_url + "/score",
        json={"instances": [[0.1, 0.2], [3.0, 2.0, 1.0]], "timeout_ms": 5000},
        timeout=10,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["scores"]) == 2
    assert all(isinstance(s, float) for s in body["scores"])
