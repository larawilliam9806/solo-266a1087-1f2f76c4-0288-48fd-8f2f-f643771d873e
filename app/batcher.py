"""Deadline-aware dynamic batching engine.

Rules implemented (see README for the full contract):

* A single asyncio worker runs **at most one ONNX batch at a time**.
* When the engine is idle and at least one request is queued it waits a
  configurable ``batch_window_ms`` to collect more arrivals.
* Queued requests are tried in arrival order.  A request is admitted while
  the padded tensor ``(new_count * new_max_length)`` (or, optionally, the sum
  of true lengths) stays within the element budget and batch-size cap.
* Requests that do not fit stay queued for a later batch.  With the default
  ``skip`` policy later shorter requests may still be admitted; ``prefix``
  stops at the first request that does not fit.
* A single request whose own length exceeds the budget is rejected before
  reaching this module.
* Client deadlines are enforced by the HTTP layer via :meth:`Batcher.wait`.
  If a deadline wins while the request is still queued it is removed and never
  sent to the model.  If the request is already part of a running batch the
  ONNX call runs to completion; the other requests in the batch still receive
  their scores.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Settings
from .engine import OnnxAnomalyEngine

logger = logging.getLogger("anomaly_batcher.core")


@dataclass
class Entry:
    id: str
    values: np.ndarray
    length: int
    timeout_ms: float
    deadline: float  # asyncio loop time
    future: asyncio.Future
    created_at: float
    state: str = "queued"  # queued | running | done | timeout | failed
    client_alive: bool = True
    deadline_reported: bool = False
    batch_id: str | None = None
    batch_size: int | None = None
    packed_length_max: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    timed_out_while_running: bool = False
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def remaining(self) -> float:
        try:
            loop = asyncio.get_running_loop()
            return max(0.0, self.deadline - loop.time())
        except RuntimeError:
            return max(0.0, self.deadline - time.monotonic())


@dataclass
class BatchRecord:
    batch_id: str
    size: int
    lengths: list[int]
    max_length: int
    padded_elements: int
    actual_elements: int
    budget: int
    duration_ms: float
    request_ids: list[str] = field(default_factory=list)
    expired_inside: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "size": self.size,
            "lengths": list(self.lengths),
            "max_length": self.max_length,
            "padded_elements": self.padded_elements,
            "actual_elements": self.actual_elements,
            "budget": self.budget,
            "duration_ms": round(self.duration_ms, 3),
            "request_ids": list(self.request_ids),
            "expired_inside": self.expired_inside,
        }


class Batcher:
    def __init__(self, settings: Settings, engine: OnnxAnomalyEngine) -> None:
        self.settings = settings
        self.engine = engine
        self._queue: deque[Entry] = deque()
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._stopping = False
        self._batch_seq = itertools.count(1)

        self.active_batches = 0
        self.peak_concurrent_batches = 0
        self.total_batches = 0
        self.total_items_inferred = 0
        self.timeouts_queued = 0
        self.timeouts_running = 0
        self.rejected_oversize = 0
        self.submitted = 0
        self.completed = 0
        self.scores_delivered = 0
        self.failed = 0
        self.recent_batches: deque[BatchRecord] = deque(maxlen=100)
        self.last_batch: BatchRecord | None = None
        self.started_wall = time.time()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._stopping = False
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = None
            self._worker = asyncio.create_task(self._run(), name="dynamic-batcher")
            logger.info(
                "batcher started window=%.1fms budget=%s mode=%s policy=%s max_batch_size=%s",
                self.settings.batch_window_ms,
                self.settings.max_elements,
                self.settings.budget_mode,
                self.settings.pack_policy,
                self.settings.max_batch_size or "inf",
            )

    async def aclose(self) -> None:
        self._stopping = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
        # Release anything still waiting so shutdown does not hang.
        while self._queue:
            entry = self._queue.popleft()
            if not entry.future.done():
                entry.future.cancel()
        self._wake.set()

    # ------------------------------------------------------------------ submission
    def is_single_too_large(self, length: int) -> bool:
        return length > self.settings.max_elements

    def submit(self, values: np.ndarray, timeout_ms: float, request_id: str | None = None) -> Entry:
        loop = asyncio.get_running_loop()
        now = loop.time()
        future = loop.create_future()

        def _retrieve_exception(fut: asyncio.Future) -> None:
            if fut.cancelled():
                return
            try:
                fut.exception()  # mark any exception as observed
            except Exception:
                pass

        future.add_done_callback(_retrieve_exception)
        entry = Entry(
            id=str(request_id) if request_id else uuid.uuid4().hex[:12],
            values=values,
            length=int(values.shape[0]),
            timeout_ms=float(timeout_ms),
            deadline=now + max(0.0, float(timeout_ms)) / 1000.0,
            future=future,
            created_at=now,
        )
        self._queue.append(entry)
        self.submitted += 1
        self._wake.set()
        return entry

    async def wait(self, entry: Entry) -> dict[str, Any]:
        """Wait for the score, enforcing the caller-supplied deadline."""
        timeout_s = max(0.0, entry.deadline - asyncio.get_running_loop().time())
        try:
            return await asyncio.wait_for(asyncio.shield(entry.future), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._handle_client_timeout(entry)
            raise

    def _handle_client_timeout(self, entry: Entry) -> None:
        entry.client_alive = False
        if entry.state == "queued":
            if entry.deadline_reported:
                return
            entry.deadline_reported = True
            entry.state = "timeout"
            try:
                self._queue.remove(entry)
            except ValueError:
                pass
            self.timeouts_queued += 1
            logger.info("request %s expired while queued (len=%s)", entry.id, entry.length)
        elif entry.state == "running":
            if not entry.timed_out_while_running:
                entry.timed_out_while_running = True
                self.timeouts_running += 1
                logger.info(
                    "request %s expired during inference; batch continues", entry.id
                )
        # state == done: result landed before/at cancellation; nothing to do.

    # ------------------------------------------------------------------ packing
    def _fits(self, count: int, current_max: int, current_sum: int, length: int) -> bool:
        s = self.settings
        if s.max_batch_size and count + 1 > s.max_batch_size:
            return False
        if s.budget_mode == "sum":
            return current_sum + length <= s.max_elements
        new_max = max(current_max, length)
        return (count + 1) * new_max <= s.max_elements

    def _expire_before_model(self, entry: Entry) -> None:
        """Drop a queued entry whose deadline has already passed."""
        if entry.deadline_reported:
            return
        entry.deadline_reported = True
        entry.state = "timeout"
        entry.client_alive = False
        self.timeouts_queued += 1

    def _pack(self) -> list[Entry]:
        """Build the next batch from the queue in arrival order."""
        now = asyncio.get_running_loop().time()
        batch: list[Entry] = []
        leftovers: deque[Entry] = deque()
        current_max = 0
        current_sum = 0
        blocked = False

        for entry in self._queue:
            if entry.state == "timeout" or not entry.client_alive or entry.deadline <= now:
                if entry.state != "timeout":
                    self._expire_before_model(entry)
                # It is excluded from leftovers -> removed without inference.
                continue

            if blocked:
                leftovers.append(entry)
                continue

            if self._fits(len(batch), current_max, current_sum, entry.length):
                batch.append(entry)
                current_max = max(current_max, entry.length)
                current_sum += entry.length
            else:
                leftovers.append(entry)
                if self.settings.pack_policy == "prefix":
                    blocked = True
                # skip policy: keep trying later arrivals

        self._queue = leftovers
        return batch

    # ------------------------------------------------------------------ worker
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stopping:
            try:
                # Block until there is something queued.
                if not self._queue:
                    self._wake.clear()
                    if not self._queue:
                        await self._wake.wait()
                    if self._stopping:
                        return

                # Idle with work available -> configurable collection window.
                await asyncio.sleep(self.settings.batch_window_ms / 1000.0)
                if self._stopping:
                    return

                batch = self._pack()
                if not batch:
                    # Everything expired; loop back and wait for new work.
                    continue

                batch_id = f"b{next(self._batch_seq)}"
                padded_elements = len(batch) * max(e.length for e in batch)
                actual_elements = sum(e.length for e in batch)
                for e in batch:
                    e.state = "running"
                    e.batch_id = batch_id
                    e.started_at = loop.time()

                self.active_batches += 1
                self.peak_concurrent_batches = max(
                    self.peak_concurrent_batches, self.active_batches
                )
                t0 = loop.time()
                try:
                    rows = [e.values for e in batch]
                    results = await loop.run_in_executor(None, self.engine.infer, rows)
                except Exception as exc:  # inference failure
                    logger.exception("batch %s inference failed", batch_id)
                    self.failed += len(batch)
                    for entry in batch:
                        entry.state = "failed"
                        entry.error = str(exc)
                        if not entry.future.done():
                            entry.future.set_exception(exc)
                    continue
                finally:
                    self.active_batches -= 1

                duration_ms = (loop.time() - t0) * 1000.0
                expired_inside = 0
                for entry, result in zip(batch, results):
                    entry.finished_at = loop.time()
                    entry.result = result
                    entry.state = "done"
                    entry.batch_size = len(batch)
                    self.completed += 1
                    if entry.timed_out_while_running or not entry.client_alive:
                        expired_inside += 1
                    else:
                        self.scores_delivered += 1
                    if not entry.future.done():
                        entry.future.set_result(result)

                record = BatchRecord(
                    batch_id=batch_id,
                    size=len(batch),
                    lengths=[e.length for e in batch],
                    max_length=max(e.length for e in batch),
                    padded_elements=padded_elements,
                    actual_elements=actual_elements,
                    budget=self.settings.max_elements,
                    duration_ms=duration_ms,
                    request_ids=[e.id for e in batch],
                    expired_inside=expired_inside,
                )
                self.total_batches += 1
                self.total_items_inferred += len(batch)
                self.last_batch = record
                self.recent_batches.append(record)
                logger.info(
                    "batch %s size=%s lengths=%s padded_elements=%s %.2fms",
                    batch_id,
                    len(batch),
                    record.lengths,
                    padded_elements,
                    duration_ms,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("unexpected batcher loop error")
                await asyncio.sleep(0.05)

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "queue_depth": len(self._queue),
            "waiting": len(self._queue),
            "active_batches": self.active_batches,
            "inference_active": self.active_batches > 0,
            "busy": self.active_batches > 0,
            "peak_concurrent_batches": self.peak_concurrent_batches,
            "one_batch_at_a_time": self.peak_concurrent_batches <= 1,
            "submitted": self.submitted,
            "accepted": self.submitted,
            "completed": self.completed,
            "scores_delivered": self.scores_delivered,
            "succeeded": self.scores_delivered,
            "total_batches": self.total_batches,
            "batches_completed": self.total_batches,
            "total_items_inferred": self.total_items_inferred,
            "timeouts_queued": self.timeouts_queued,
            "timeouts_waiting": self.timeouts_queued,
            "timeouts_running": self.timeouts_running,
            "timeouts_in_batch": self.timeouts_running,
            "rejected_oversize": self.rejected_oversize,
            "oversize_rejected": self.rejected_oversize,
            "failed": self.failed,
            "uptime_s": round(time.time() - self.started_wall, 3),
            "window_ms": self.settings.batch_window_ms,
            "budget": self.settings.max_elements,
            "max_elements": self.settings.max_elements,
            "last_batch": self.last_batch.as_dict() if self.last_batch else None,
            "recent_batches": [b.as_dict() for b in reversed(self.recent_batches)],
            "config": self.settings.public(),
        }
