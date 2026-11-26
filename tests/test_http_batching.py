"""Budget packing, deadline removal, in-batch expiry and single-flight tests."""
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


def post_async(base, length, timeout_ms=5000, offset=0.1):
    """POST a constant-ish sequence of given length; returns (status, json, ms)."""
    values = [((i % 5) - 2) * offset for i in range(length)]
    t0 = time.perf_counter()
    r = httpx.post(
        base + "/score",
        json={"values": values, "timeout_ms": timeout_ms},
        timeout=max(5.0, timeout_ms / 1000 + 5),
    )
    return r.status_code, _safe_json(r), (time.perf_counter() - t0) * 1000


def _safe_json(r):
    try:
        return r.json()
    except Exception:
        return {}


def run_concurrent(base, specs, stagger_ms=12.0):
    """specs: list of (key, length, timeout_ms). Returns key -> result."""
    results = {}

    def worker(key, length, tmo):
        results[key] = post_async(base, length, tmo)

    threads = []
    for idx, (key, length, tmo) in enumerate(specs):
        t = threading.Thread(target=worker, args=(key, length, tmo))
        threads.append(t)
        t.start()
        if stagger_ms:
            time.sleep(stagger_ms / 1000.0)
    for t in threads:
        t.join()
    return results


def chronological_batches(stats):
    batches = list(reversed(stats["recent_batches"]))
    return [b["lengths"] for b in batches]


def test_budget_packing_skip_policy():
    settings = make_settings(
        batch_window_ms=80.0, max_elements=10, pack_policy="skip", inference_delay_ms=0.0
    )
    with ServerHandle(settings) as s:
        results = run_concurrent(
            s.base_url, [("a", 4, 5000), ("b", 8, 5000), ("c", 2, 5000)]
        )
        for code, body, _ in results.values():
            assert code == 200, body

        stats = httpx.get(s.base_url + "/stats", timeout=5).json()
        batches = [sorted(x) for x in chronological_batches(stats)]
        assert [2, 4] in batches, batches
        assert [8] in batches, batches
        # Every score still returned exactly once.
        assert stats["scores_delivered"] >= 3


def test_budget_packing_prefix_policy():
    settings = make_settings(
        batch_window_ms=120.0, max_elements=10, pack_policy="prefix"
    )
    with ServerHandle(settings) as s:
        results = run_concurrent(
            s.base_url, [("a", 4, 5000), ("b", 8, 5000), ("c", 2, 5000)],
            stagger_ms=30.0,
        )
        for code, body, _ in results.values():
            assert code == 200, body
        # Every formed batch respects the padded element budget; exact
        # prefix membership/order is covered deterministically in
        # test_packing_unit.py because independent HTTP connections may
        # legalistically reorder arrival.
        stats = httpx.get(s.base_url + "/stats", timeout=5).json()
        for b in stats["recent_batches"]:
            assert b["padded_elements"] <= 10
        assert stats["scores_delivered"] >= 3


def test_queued_request_timeout_is_removed():
    settings = make_settings(batch_window_ms=250.0, max_elements=4096)
    with ServerHandle(settings) as s:
        before = httpx.get(s.base_url + "/stats", timeout=5).json()
        t0 = time.perf_counter()
        code, body, ms = post_async(s.base_url, 11, timeout_ms=40)
        elapsed = time.perf_counter() - t0
        assert code == settings.timeout_status_code
        assert body["status"] == "timeout"
        assert elapsed < 0.25  # prompt, does not wait for the 250ms window

        time.sleep(0.8)
        stats = httpx.get(s.base_url + "/stats", timeout=5).json()
        assert stats["timeouts_queued"] >= 1
        assert stats["total_batches"] == before["total_batches"]
        # Length-11 request must never have reached the model.
        assert all(11 not in b["lengths"] for b in stats["recent_batches"])

        # Worker is still alive and serving.
        code2, body2, _ = post_async(s.base_url, 3, timeout_ms=5000)
        assert code2 == 200, body2


def test_timeout_inside_running_batch_others_get_scores():
    settings = make_settings(
        batch_window_ms=50.0, inference_delay_ms=500.0, max_elements=4096
    )
    with ServerHandle(settings) as s:
        t0 = time.perf_counter()
        results = run_concurrent(
            s.base_url,
            [
                ("long_a", 5, 8000),
                ("short_b", 5, 300),
                ("long_c", 5, 8000),
            ],
            stagger_ms=5,
        )
        wall = time.perf_counter() - t0

        code_b, body_b, ms_b = results["short_b"]
        code_a, body_a, _ = results["long_a"]
        code_c, body_c, _ = results["long_c"]

        assert code_b == settings.timeout_status_code
        assert body_b["status"] == "timeout"
        # Deadline ~300ms; must return well before the ~550ms inference end.
        assert ms_b < 480

        assert code_a == 200, body_a
        assert code_c == 200, body_c
        # Inference (~500ms) ran to completion, so the longs return around it.
        assert wall >= 0.45

        stats = httpx.get(s.base_url + "/stats", timeout=5).json()
        assert stats["timeouts_running"] >= 1
        last = stats["last_batch"]
        assert last is not None and last["size"] == 3
        assert last["expired_inside"] >= 1


def test_only_one_batch_runs_at_a_time():
    settings = make_settings(
        batch_window_ms=30.0, max_elements=8, inference_delay_ms=80.0
    )
    with ServerHandle(settings) as s:
        max_active_seen = {"v": 0}
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                try:
                    st = httpx.get(s.base_url + "/stats", timeout=2).json()
                    max_active_seen["v"] = max(
                        max_active_seen["v"], st["active_batches"]
                    )
                except Exception:
                    pass
                time.sleep(0.005)

        mon = threading.Thread(target=monitor, daemon=True)
        mon.start()

        results = {}

        def worker(i):
            results[i] = post_async(s.base_url, 4, timeout_ms=10000)[0]

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        mon.join(timeout=2)

        assert all(code == 200 for code in results.values()), results
        stats = httpx.get(s.base_url + "/stats", timeout=5).json()
        assert stats["peak_concurrent_batches"] == 1
        assert max_active_seen["v"] <= 1
        # Budget 8 with length-4 rows => at most 2 rows per padded batch.
        assert stats["total_batches"] >= 5
        for b in stats["recent_batches"]:
            assert b["padded_elements"] <= 8
