"""
app/queue_monitor.py

Manages local inference concurrency using a semaphore-backed slot system.

Design decisions:
  - Owns its own asyncio.Lock (separate from BudgetManager._lock).
    The two locks are NEVER acquired simultaneously, so deadlock is impossible.
  - The semaphore tracks actual slot availability.
  - `active_count` is a plain int updated under the lock — fast to read.
  - acquire() returns immediately (no blocking). If at capacity, returns False.
  - The slot() context manager guarantees release() is called even on exception,
    including asyncio.CancelledError.

Weakness addressed (§3.1): the queue lock is independent of the budget lock.
  Each operates on its own resource domain.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from app.config import LOCAL_QUEUE_CAPACITY


class QueueMonitor:
    """Tracks and enforces local execution concurrency limits."""

    def __init__(self, max_concurrency: int = LOCAL_QUEUE_CAPACITY):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be strictly positive")

        self._max_concurrency = max_concurrency
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_count: int = 0

    # ── Read-only properties ──────────────────────────────────────────────────

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def active_count(self) -> int:
        # int read in CPython is effectively atomic for advisory use.
        return self._active_count

    @property
    def is_at_capacity(self) -> bool:
        return self._active_count >= self._max_concurrency

    # ── Acquire / Release ─────────────────────────────────────────────────────

    async def acquire(self) -> bool:
        """
        Non-blocking attempt to acquire a local execution slot.
        Returns True on success, False if at capacity.
        """
        async with self._lock:
            if self._active_count >= self._max_concurrency:
                return False
            # Verified under lock — semaphore MUST have capacity; won't block.
            await self._semaphore.acquire()
            self._active_count += 1
            return True

    async def release(self) -> None:
        """Release a previously acquired slot."""
        async with self._lock:
            if self._active_count > 0:
                self._active_count -= 1
                self._semaphore.release()

    # ── Context Manager ───────────────────────────────────────────────────────

    @asynccontextmanager
    async def slot(self):
        """
        Async context manager that acquires a slot on entry and guarantees
        release on exit, even if an exception (including CancelledError) occurs.

        Raises RuntimeError if at capacity.
        """
        acquired = await self.acquire()
        if not acquired:
            raise RuntimeError("Local queue is at capacity — cannot acquire slot")
        try:
            yield
        finally:
            await self.release()
