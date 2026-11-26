#!/usr/bin/env python3
"""Dependency-free end-to-end smoke test against a running service.

Usage:
    python scripts/smoke.py [BASE_URL]

Checks: health, single score, solo-vs-batched consistency, over-budget
rejection and deadline timeout.  Exits non-zero if a hard assertion fails.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def request(method: str, path: str, payload=None, timeout: float = 10.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode())
        except Exception:
            return exc.code, {}


def wait_health(deadline_s: float = 30.0) -> None:
    end = time.time() + deadline_s
    last = None
    while time.time() < end:
        try:
            status, body = request("GET", "/health", timeout=2)
            if status == 200 and body.get("model_loaded", body.get("healthy")):
                print(f"[ok] health at {BASE}")
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.5)
    raise SystemExit(f"service not healthy at {BASE}: {last}")


def main() -> int:
    wait_health()
    failures: list[str] = []

    status, body = request("POST", "/score", {"values": [0.1, -0.2, 0.3, 0.05], "timeout_ms": 3000})
    assert status == 200, (status, body)
    assert isinstance(body.get("score"), float)
    print(f"[ok] single score={body['score']:.6f} batch_size={body.get('batch_size')}")

    sequences = [
        [((i * 7 + j) % 11) / 11.0 - 0.5 for j in range(n)]
        for i, n in enumerate([3, 8, 15, 32, 64, 11, 1], start=1)
    ]
    solo = []
    for seq in sequences:
        st, bd = request("POST", "/score", {"values": seq, "timeout_ms": 5000})
        if st != 200:
            failures.append(f"solo status {st}")
        solo.append(bd.get("score"))

    with ThreadPoolExecutor(max_workers=len(sequences)) as pool:
        batched = list(
            pool.map(
                lambda s: request("POST", "/score", {"values": s, "timeout_ms": 5000})[1].get("score"),
                sequences,
            )
        )
    for a, b in zip(solo, batched):
        if a is None or b is None or abs(a - b) > 1e-7:
            failures.append(f"inconsistent scores: solo={a} batch={b}")
    if not failures:
        print("[ok] batched scores identical to solo inference")

    st, bd = request("POST", "/score", {"values": [0.0] * 1_000_000, "timeout_ms": 5000})
    if not (400 <= st < 500 and bd.get("success") is False):
        failures.append(f"oversize should be rejected, got {st}")
    else:
        print(f"[ok] over-budget rejected with HTTP {st} ({bd.get('error_code')})")

    t0 = time.time()
    st, bd = request("POST", "/score", {"values": [1, 2, 3], "timeout_ms": 5}, timeout=5)
    elapsed = time.time() - t0
    if st != 408 or bd.get("status") != "timeout":
        failures.append(f"expected 408 timeout, got {st} {bd.get('status')}")
    elif elapsed > 1.0:
        failures.append(f"timeout not prompt: {elapsed:.3f}s")
    else:
        print(f"[ok] deadline timeout after {elapsed*1000:.0f}ms")

    try:
        st, stats = request("GET", "/stats")
        if st == 200:
            assert stats.get("peak_concurrent_batches", 1) <= 1
            print(f"[ok] stats: batches={stats.get('total_batches')} "
                  f"peak_concurrent={stats.get('peak_concurrent_batches')}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"stats failed: {exc}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
