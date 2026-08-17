"""
app/latency_tracker.py

Per-backend latency estimation using a hybrid EMA + sliding window approach.

Weakness addressed (§3.4): Latency Estimation Accuracy
  Plain EMA smooths over spikes — if one call took 2000ms, the EMA barely
  moves. The next routing decision might still send to the same degraded backend.

  Solution: get_estimate() returns max(EMA, recent_window_max).
  After a spike, the estimate is immediately pessimistic. It relaxes naturally
  as the window slides past the slow observation, which takes at most
  LATENCY_WINDOW_SIZE calls.

  Tradeoff: get_estimate() is a lock-free read of Python primitives.
  CPython's GIL makes individual reads of float/list elements effectively
  atomic. We accept the tiny risk of reading a mid-update window in exchange
  for zero contention on the hot read path. Writes always acquire the lock.

Design:
  - Owns its own asyncio.Lock (Weakness §3.1: separate per-resource locks).
  - EMA seeded with realistic baseline values from config.
  - Sliding window is a collections.deque with maxlen = LATENCY_WINDOW_SIZE.
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Dict

from app.config import AVG_LOCAL_LATENCY_MS, AVG_REMOTE_LATENCY_MS, EMA_ALPHA, LATENCY_WINDOW_SIZE

BACKEND_LOCAL = "local"
BACKEND_REMOTE = "remote"


class LatencyTracker:
    """
    Tracks per-backend latency estimates for routing decisions.
    Uses EMA for long-run smoothing and a recent-max window for spike detection.
    """

    def __init__(
        self,
        alpha: float = EMA_ALPHA,
        window_size: int = LATENCY_WINDOW_SIZE,
    ):
        self._alpha = alpha
        self._lock = asyncio.Lock()

        # EMA per backend — seeded from config to avoid cold-start bias
        self._ema: Dict[str, float] = {
            BACKEND_LOCAL: AVG_LOCAL_LATENCY_MS,
            BACKEND_REMOTE: AVG_REMOTE_LATENCY_MS,
        }

        # Sliding window of recent raw observations per backend
        self._windows: Dict[str, deque] = {
            BACKEND_LOCAL: deque(maxlen=window_size),
            BACKEND_REMOTE: deque(maxlen=window_size),
        }

    async def record(self, backend: str, latency_ms: float) -> None:
        """
        Record an observed latency. Updates both EMA and the sliding window.
        Always acquires the write lock.
        """
        async with self._lock:
            if backend not in self._ema:
                self._ema[backend] = latency_ms
                self._windows[backend] = deque(maxlen=len(self._windows[BACKEND_LOCAL]))

            # Update EMA
            self._ema[backend] = (
                self._alpha * latency_ms
                + (1.0 - self._alpha) * self._ema[backend]
            )

            # Append to window
            self._windows[backend].append(latency_ms)

    def get_estimate(self, backend: str) -> float:
        """
        Returns the conservative latency estimate for routing decisions.
        = max(EMA, recent_window_max)

        Lock-free read: safe for CPython advisory use. A write mid-read would at
        worst cause us to return a slightly stale value — acceptable for routing.
        """
        ema = self._ema.get(backend, 500.0)
        window = self._windows.get(backend)
        if window:
            recent_max = max(window, default=0.0)
            return max(ema, recent_max)
        return ema

    def get_ema(self, backend: str) -> float:
        """Returns just the EMA value (useful for observability/logging)."""
        return self._ema.get(backend, 500.0)

    async def reset(self) -> None:
        """Reset to seed values (useful for test isolation)."""
        async with self._lock:
            self._ema = {
                BACKEND_LOCAL: AVG_LOCAL_LATENCY_MS,
                BACKEND_REMOTE: AVG_REMOTE_LATENCY_MS,
            }
            for w in self._windows.values():
                w.clear()
