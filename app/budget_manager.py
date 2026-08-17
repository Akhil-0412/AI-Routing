"""
app/budget_manager.py

Two-phase atomic reservation protocol using asyncio.Lock.

Design decisions:
  - ONE lock (budget_lock) guards ALL budget mutations.
  - No lock is held during network I/O; the lock is released between phases.
  - Every reservation carries a TTL; a background reaper task rolls back stale
    reservations to prevent budget being permanently locked by crashed requests.
  - Settlement is idempotent-safe: committing or rolling back a non-pending
    reservation raises InvalidReservationStateError.

Weakness addressed (§3.2): Reservation Expiry
  TTL on each Reservation + background reaper guarantees budget is always
  recovered, even if the FastAPI handler crashes or is cancelled.

Weakness addressed (§3.1): Single Lock
  This module owns ONLY the budget lock. Queue state is owned by QueueMonitor,
  latency estimates by LatencyTracker. No method here ever acquires two locks,
  so deadlock is structurally impossible.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from app.config import (
    MAX_CUMULATIVE_LATENCY_MS,
    MAX_REMOTE_BUDGET,
    REAPER_INTERVAL_SECONDS,
    RESERVATION_TTL_SECONDS,
)
from app.models import BudgetSnapshot
import app.observability as _obs


# ─── Custom Exceptions ────────────────────────────────────────────────────────

class InvalidReservationStateError(Exception):
    """Raised when commit/rollback is called on an already-settled reservation."""


class ActualCostExceededReservationError(Exception):
    """Raised when actual cost exceeds the reserved amount at commit time."""


class ActualLatencyExceededReservationError(Exception):
    """Raised when committing would breach the global cumulative latency cap."""


# ─── Reservation ──────────────────────────────────────────────────────────────

@dataclass
class Reservation:
    """
    Holds a budget + latency slot while a request executes.

    Created by BudgetManager.try_reserve().
    Must be settled by calling either commit() or rollback() exactly once.
    After TTL seconds the background reaper will automatically rollback if
    neither has been called.
    """
    reserved_cost: float
    reserved_latency_ms: float
    ttl_seconds: float
    _manager: "BudgetManager"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.monotonic)
    _state: Literal["pending", "committed", "rolled_back"] = field(
        default="pending", init=False
    )

    @property
    def state(self) -> Literal["pending", "committed", "rolled_back"]:
        return self._state

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds

    async def commit(self, actual_cost: float, actual_latency_ms: float) -> None:
        """
        Phase 2: Settle the reservation with actual spend.
        Releases reserved amounts and records actuals.
        Raises if actual_cost exceeds reservation or latency cap would be breached.
        """
        await self._manager._commit(self.id, actual_cost, actual_latency_ms)

    async def rollback(self) -> None:
        """
        Phase 2 (error path): Cancel the reservation and return funds to pool.
        Safe to call after an exception — funds are always returned.
        """
        await self._manager._rollback(self.id)


# ─── BudgetManager ────────────────────────────────────────────────────────────

class BudgetManager:
    """
    Manages global remote budget using a two-phase atomic reservation protocol.

    Phase 1 (reserve): Acquire lock → check liquid budget → deduct reservation →
                        release lock → return Reservation object.
    Phase 2 (settle):  Acquire lock → pop reservation → record actual spend →
                        release lock.

    The lock is NEVER held during backend I/O. This is the core guarantee that
    prevents budget double-spend while not serializing execution latency.
    """

    def __init__(
        self,
        max_remote_budget: float = MAX_REMOTE_BUDGET,
        max_cumulative_latency_ms: float = MAX_CUMULATIVE_LATENCY_MS,
        reservation_ttl_seconds: float = RESERVATION_TTL_SECONDS,
    ):
        if max_remote_budget <= 0.0:
            raise ValueError("max_remote_budget must be strictly positive")

        self._lock = asyncio.Lock()

        # Configuration (immutable after init)
        self._max_remote_budget = max_remote_budget
        self._max_cumulative_latency_ms = max_cumulative_latency_ms
        self._reservation_ttl_seconds = reservation_ttl_seconds

        # Financial state (mutated only inside _lock)
        self._cumulative_remote_cost: float = 0.0
        self._reserved_for_execution: float = 0.0
        self._cumulative_actual_latency_ms: float = 0.0
        self._reserved_latency_ms: float = 0.0

        # Active reservations indexed by id
        self._active_reservations: Dict[str, Reservation] = {}

        # Reaper task handle (set during lifespan)
        self._reaper_task: Optional[asyncio.Task] = None

    # ── Properties (computed, lock-free for observability) ───────────────────

    @property
    def remaining_budget(self) -> float:
        """Absolute remaining budget (ignores pending reservations)."""
        return max(0.0, self._max_remote_budget - self._cumulative_remote_cost)

    @property
    def liquid_budget(self) -> float:
        """Budget available for new reservations = remaining minus already-reserved."""
        return max(
            0.0,
            self._max_remote_budget
            - self._cumulative_remote_cost
            - self._reserved_for_execution,
        )

    @property
    def active_reservation_count(self) -> int:
        return len(self._active_reservations)

    # ── Phase 1: Reserve ─────────────────────────────────────────────────────

    async def try_reserve(
        self, request_id: str, cost: float, projected_latency_ms: float
    ) -> Optional[Reservation]:
        """
        Atomically checks and reserves budget + latency capacity.

        Returns a Reservation if both checks pass, or None if either limit
        would be breached. Callers MUST call reservation.commit() or
        reservation.rollback() exactly once.
        """
        if cost < 0 or projected_latency_ms < 0:
            raise ValueError("cost and projected_latency_ms must be non-negative")
        if cost == 0.0 and projected_latency_ms == 0.0:
            raise ValueError("At least one of cost or projected_latency_ms must be > 0")

        _obs.lock_event(request_id, "reserve", "acquire")
        async with self._lock:
            liquid = (
                self._max_remote_budget
                - self._cumulative_remote_cost
                - self._reserved_for_execution
            )
            latency_remaining = (
                self._max_cumulative_latency_ms
                - self._cumulative_actual_latency_ms
                - self._reserved_latency_ms
            )

            if liquid < cost - 1e-12 or latency_remaining < projected_latency_ms - 1e-12:
                _obs.lock_event(request_id, "reserve", "release")
                return None

            # Commit the reservation atomically
            self._reserved_for_execution += cost
            self._reserved_latency_ms += projected_latency_ms

            reservation = Reservation(
                reserved_cost=cost,
                reserved_latency_ms=projected_latency_ms,
                ttl_seconds=self._reservation_ttl_seconds,
                _manager=self,
            )
            self._active_reservations[reservation.id] = reservation
            _obs.lock_event(request_id, "reserve", "release")
            return reservation

    # ── Phase 2a: Commit ─────────────────────────────────────────────────────

    async def _commit(
        self, reservation_id: str, actual_cost: float, actual_latency_ms: float
    ) -> None:
        """Internal: settle a reservation with actual spend. Called by Reservation.commit()."""
        _obs.lock_event(reservation_id, "settle", "acquire")
        async with self._lock:
            if reservation_id not in self._active_reservations:
                _obs.lock_event(reservation_id, "settle", "release")
                raise InvalidReservationStateError(
                    f"Reservation {reservation_id!r} is not active. "
                    "It may have already been settled or expired."
                )

            reservation = self._active_reservations[reservation_id]

            # Guard: actual cost must not exceed what was reserved
            if actual_cost > reservation.reserved_cost + 1e-9:
                # Roll back — no spend committed. Caller should handle this.
                reservation._state = "rolled_back"
                self._active_reservations.pop(reservation_id)
                self._reserved_for_execution -= reservation.reserved_cost
                self._reserved_latency_ms -= reservation.reserved_latency_ms
                _obs.lock_event(reservation_id, "settle", "release")
                raise ActualCostExceededReservationError(
                    f"actual_cost ({actual_cost:.6f}) > reserved ({reservation.reserved_cost:.6f})"
                )

            # Guard: committing latency must not breach global cap
            if (
                self._cumulative_actual_latency_ms + actual_latency_ms
                > self._max_cumulative_latency_ms + 1e-9
            ):
                reservation._state = "rolled_back"
                self._active_reservations.pop(reservation_id)
                self._reserved_for_execution -= reservation.reserved_cost
                self._reserved_latency_ms -= reservation.reserved_latency_ms
                _obs.lock_event(reservation_id, "settle", "release")
                raise ActualLatencyExceededReservationError(
                    f"Committing {actual_latency_ms:.1f}ms would breach global "
                    f"cap of {self._max_cumulative_latency_ms:.1f}ms"
                )

            # All checks passed — settle
            reservation._state = "committed"
            self._active_reservations.pop(reservation_id)
            self._reserved_for_execution = max(
                0.0, self._reserved_for_execution - reservation.reserved_cost
            )
            self._reserved_latency_ms = max(
                0.0, self._reserved_latency_ms - reservation.reserved_latency_ms
            )
            self._cumulative_remote_cost += actual_cost
            self._cumulative_actual_latency_ms += actual_latency_ms
            _obs.lock_event(reservation_id, "settle", "release")

    # ── Phase 2b: Rollback ───────────────────────────────────────────────────

    async def _rollback(self, reservation_id: str) -> None:
        """Internal: cancel a reservation and return funds. Called by Reservation.rollback()."""
        _obs.lock_event(reservation_id, "settle", "acquire")
        async with self._lock:
            if reservation_id not in self._active_reservations:
                _obs.lock_event(reservation_id, "settle", "release")
                raise InvalidReservationStateError(
                    f"Reservation {reservation_id!r} is not active. "
                    "It may have already been settled or expired."
                )

            reservation = self._active_reservations.pop(reservation_id)
            reservation._state = "rolled_back"
            self._reserved_for_execution = max(
                0.0, self._reserved_for_execution - reservation.reserved_cost
            )
            self._reserved_latency_ms = max(
                0.0, self._reserved_latency_ms - reservation.reserved_latency_ms
            )
            _obs.lock_event(reservation_id, "settle", "release")

    # ── Snapshot (lock-safe read) ─────────────────────────────────────────────

    async def budget_snapshot(self) -> BudgetSnapshot:
        """Returns a consistent point-in-time view of all budget state."""
        async with self._lock:
            remaining = max(
                0.0,
                self._max_remote_budget
                - self._cumulative_remote_cost
                - self._reserved_for_execution,
            )
            return BudgetSnapshot(
                cumulative_cost=self._cumulative_remote_cost,
                reserved=self._reserved_for_execution,
                remaining=remaining,
                cumulative_latency_ms=self._cumulative_actual_latency_ms,
                reserved_latency_ms=self._reserved_latency_ms,
            )

    # ── Background Reaper (Weakness §3.2 mitigation) ─────────────────────────

    async def _reaper_loop(self, interval: float) -> None:
        """
        Background task that periodically scans for expired reservations and
        rolls them back. This is the safety net for requests that crash,
        are cancelled, or throw BaseException before settling.
        """
        import logging
        log = logging.getLogger("lec_ai_router")

        while True:
            await asyncio.sleep(interval)
            await self._expire_stale_reservations(log)

    async def _expire_stale_reservations(self, log=None) -> list[str]:
        """Rolls back all expired reservations. Returns list of expired IDs."""
        import logging
        log = log or logging.getLogger("lec_ai_router")

        expired_ids = []
        async with self._lock:
            for res_id, reservation in list(self._active_reservations.items()):
                if reservation.is_expired:
                    reservation._state = "rolled_back"
                    self._active_reservations.pop(res_id)
                    self._reserved_for_execution = max(
                        0.0, self._reserved_for_execution - reservation.reserved_cost
                    )
                    self._reserved_latency_ms = max(
                        0.0,
                        self._reserved_latency_ms - reservation.reserved_latency_ms,
                    )
                    expired_ids.append(res_id)
                    log.warning(
                        "reservation_expired",
                        extra={
                            "event": "reservation_expired",
                            "reservation_id": res_id,
                            "age_seconds": round(
                                time.monotonic() - reservation.created_at, 2
                            ),
                            "recovered_cost": reservation.reserved_cost,
                        },
                    )
        return expired_ids

    def start_reaper(self, interval: float = REAPER_INTERVAL_SECONDS) -> None:
        """Start the background reaper task. Call from FastAPI lifespan."""
        self._reaper_task = asyncio.create_task(
            self._reaper_loop(interval), name="budget_reaper"
        )

    def stop_reaper(self) -> None:
        """Cancel the reaper task. Call from FastAPI lifespan shutdown."""
        if self._reaper_task and not self._reaper_task.done():
            self._reaper_task.cancel()
