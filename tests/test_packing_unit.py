"""Deterministic unit tests for the packing algorithm."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from app.batcher import Batcher, Entry  # noqa: E402
from app.config import get_settings  # noqa: E402


def make_entry(batcher: Batcher, length: int, *, deadline_s: float = 10.0) -> Entry:
    loop = asyncio.get_event_loop()
    return Entry(
        id=f"e{length}",
        values=np.zeros(length, dtype=np.float32),
        length=length,
        timeout_ms=5000,
        deadline=loop.time() + deadline_s,
        future=loop.create_future(),
        created_at=loop.time(),
    )


def pack_once(policy: str, lengths: list[int], budget: int = 10):
    async def scenario():
        settings = replace(get_settings(), pack_policy=policy, max_elements=budget)
        b = Batcher(settings, engine=None)  # type: ignore[arg-type]
        for n in lengths:
            b._queue.append(make_entry(b, n))
        batch = b._pack()
        return [e.length for e in batch], [e.length for e in b._queue]

    return asyncio.run(scenario())


def test_skip_policy_keeps_trying():
    packed, left = pack_once("skip", [4, 8, 2])
    assert packed == [4, 2]
    assert left == [8]


def test_prefix_policy_stops_at_first_nonfit():
    packed, left = pack_once("prefix", [4, 8, 2])
    assert packed == [4]
    assert left == [8, 2]


def test_skip_all_fit():
    packed, left = pack_once("skip", [1, 2, 3], budget=20)
    assert packed == [1, 2, 3]
    assert left == []


def test_padded_budget_boundary():
    # 2 * 5 == 10 fits exactly; third would be 15 > 10.
    packed, left = pack_once("skip", [5, 5, 1], budget=10)
    assert packed == [5, 5]
    assert left == [1]


def test_expired_entries_are_dropped():
    async def scenario():
        settings = replace(get_settings(), pack_policy="skip", max_elements=100)
        b = Batcher(settings, engine=None)  # type: ignore[arg-type]
        good = make_entry(b, 3, deadline_s=10)
        stale = make_entry(b, 4, deadline_s=-1)
        b._queue.extend([good, stale])
        batch = b._pack()
        return [e.length for e in batch], [e.length for e in b._queue], b.timeouts_queued

    packed, left, timeouts = asyncio.run(scenario())
    assert packed == [3]
    assert left == []
    assert timeouts == 1


def test_sum_budget_mode():
    async def scenario():
        settings = replace(
            get_settings(), budget_mode="sum", max_elements=10, pack_policy="prefix"
        )
        b = Batcher(settings, engine=None)  # type: ignore[arg-type]
        for n in (6, 3, 2):  # 6+3=9 fits, +2=11 no
            b._queue.append(make_entry(b, n))
        batch = b._pack()
        return [e.length for e in batch], [e.length for e in b._queue]

    packed, left = asyncio.run(scenario())
    assert packed == [6, 3]
    assert left == [2]
