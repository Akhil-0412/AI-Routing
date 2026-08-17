"""
app/decision_agent.py

Stateless routing decision logic. Consumes read-only snapshots and advisory
reads from shared state, then calls try_reserve() for the one atomic mutation.

Decision tree — evaluated differently depending on ComplexityTier:

  FAST tier (simple queries — prefer local):
    1. LOCAL   — queue has capacity AND estimated local latency ≤ SLA
    2. REMOTE  — local infeasible AND budget available (last resort only)
    3. QUEUE_LOCAL — budget exhausted but local has a slot (degraded fallback)
    4. FAIL_FAST — all paths exhausted

  QUALITY tier (reasoning/generation — prefer remote):
    1. REMOTE  — budget available AND estimated remote latency ≤ SLA (tried first)
    2. LOCAL   — remote budget exhausted, local has capacity AND within SLA
    3. QUEUE_LOCAL — all preferred routes exhausted but local slot exists
    4. FAIL_FAST — all paths exhausted

The tier changes the *ordering of paths*, never the hard constraints.
Budget caps and latency limits apply identically for both tiers.

The agent reads queue depth and latency estimates WITHOUT acquiring any lock.
These are advisory signals — the actual constraint enforcement happens inside
BudgetManager.try_reserve() and QueueMonitor.acquire(), which own their own locks.

If try_reserve() returns None (concurrent request beat us to the budget), we
fall through to the next path. This means the decision tree handles the race
gracefully without any retry loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.budget_manager import BudgetManager, Reservation
from app.complexity_classifier import ComplexityTier
from app.config import COST_PER_REMOTE_REQUEST
from app.latency_tracker import BACKEND_LOCAL, BACKEND_REMOTE, LatencyTracker
from app.models import InferenceRequest, RouteDecision
from app.queue_monitor import QueueMonitor


@dataclass
class RoutingDecision:
    """The output of a routing evaluation — includes the reservation if budget was claimed."""
    route: RouteDecision
    reservation: Optional[Reservation]
    estimated_latency_ms: float
    reason: str


class DecisionAgent:
    """
    Stateless routing agent. All shared state lives in the injected managers.
    An instance can be reused across requests — it holds no per-request state.
    """

    def __init__(
        self,
        budget_manager: BudgetManager,
        queue_monitor: QueueMonitor,
        latency_tracker: LatencyTracker,
    ):
        self._bm = budget_manager
        self._qm = queue_monitor
        self._lt = latency_tracker

    async def route(self, request: InferenceRequest) -> RoutingDecision:
        """
        Evaluate routing paths in priority order and return the best decision.

        FAST tier:    local → remote → queue_local → fail_fast
        QUALITY tier: remote → local → queue_local → fail_fast

        This method is safe to call concurrently — all state mutations are
        delegated to the respective managers' atomic methods.
        """
        tier = request.complexity_tier  # guaranteed non-None by main.py

        # ── Advisory reads (no locks held) ───────────────────────────────────
        queue_depth = self._qm.active_count
        at_capacity = self._qm.is_at_capacity

        # Little's Law: estimated wait = (queue_depth + 1) * latency_per_slot
        est_local_ms  = self._lt.get_estimate(BACKEND_LOCAL) * (queue_depth + 1)
        est_remote_ms = self._lt.get_estimate(BACKEND_REMOTE)

        local_within_sla  = est_local_ms  <= request.latency_budget_ms
        remote_within_sla = est_remote_ms <= request.latency_budget_ms

        # ── Route based on complexity tier ────────────────────────────────────
        if tier == ComplexityTier.QUALITY:
            return await self._route_quality(
                request, at_capacity,
                est_local_ms, est_remote_ms,
                local_within_sla, remote_within_sla,
                queue_depth,
            )
        else:
            return await self._route_fast(
                request, at_capacity,
                est_local_ms, est_remote_ms,
                local_within_sla, remote_within_sla,
                queue_depth,
            )

    # ── FAST tier: local → remote → queue_local → fail_fast ──────────────────

    async def _route_fast(
        self,
        request: InferenceRequest,
        at_capacity: bool,
        est_local_ms: float,
        est_remote_ms: float,
        local_within_sla: bool,
        remote_within_sla: bool,
        queue_depth: int,
    ) -> RoutingDecision:
        """FAST tier: prefers local. Remote is only a last resort."""

        # PATH 1: Local is optimal
        if not at_capacity and local_within_sla:
            reservation = await self._bm.try_reserve(request.request_id, 0.0, est_local_ms)
            if reservation is not None:
                return RoutingDecision(
                    route=RouteDecision.LOCAL,
                    reservation=reservation,
                    estimated_latency_ms=est_local_ms,
                    reason=(
                        f"[FAST] Local queue healthy ({queue_depth}/{self._qm.max_concurrency}), "
                        f"est latency {est_local_ms:.0f}ms ≤ SLA {request.latency_budget_ms:.0f}ms"
                    ),
                )
            # Global latency cap hit — fail fast
            return RoutingDecision(
                route=RouteDecision.FAIL_FAST,
                reservation=None,
                estimated_latency_ms=est_local_ms,
                reason="[FAST] Global cumulative latency cap would be breached by local route",
            )

        # PATH 2: Local infeasible — escalate to remote (last resort for FAST tier)
        if remote_within_sla:
            reservation = await self._bm.try_reserve(
                request.request_id, COST_PER_REMOTE_REQUEST, est_remote_ms
            )
            if reservation is not None:
                return RoutingDecision(
                    route=RouteDecision.REMOTE,
                    reservation=reservation,
                    estimated_latency_ms=est_remote_ms,
                    reason=(
                        f"[FAST] Local infeasible (capacity={at_capacity}, "
                        f"est_local={est_local_ms:.0f}ms); "
                        f"escalating to remote as last resort"
                    ),
                )

        # PATH 3: Degraded local fallback (budget exhausted or remote over SLA)
        if not at_capacity:
            degraded_est = self._lt.get_estimate(BACKEND_LOCAL) * (queue_depth + 1)
            reservation = await self._bm.try_reserve(request.request_id, 0.0, degraded_est)
            if reservation is not None:
                return RoutingDecision(
                    route=RouteDecision.QUEUE_LOCAL,
                    reservation=reservation,
                    estimated_latency_ms=degraded_est,
                    reason=(
                        f"[FAST] Budget/SLA exhausted; degrading to local queue "
                        f"(SLA will be missed: est {degraded_est:.0f}ms)"
                    ),
                )
            return RoutingDecision(
                route=RouteDecision.FAIL_FAST,
                reservation=None,
                estimated_latency_ms=0.0,
                reason="[FAST] Budget exhausted and global latency cap would be breached",
            )

        # PATH 4: Hard fail-fast circuit breaker
        return RoutingDecision(
            route=RouteDecision.FAIL_FAST,
            reservation=None,
            estimated_latency_ms=0.0,
            reason=(
                "[FAST] All routes exhausted: local at capacity, "
                f"remote {'over SLA' if not remote_within_sla else 'budget exhausted'}"
            ),
        )

    # ── QUALITY tier: remote → local → queue_local → fail_fast ───────────────

    async def _route_quality(
        self,
        request: InferenceRequest,
        at_capacity: bool,
        est_local_ms: float,
        est_remote_ms: float,
        local_within_sla: bool,
        remote_within_sla: bool,
        queue_depth: int,
    ) -> RoutingDecision:
        """QUALITY tier: prefers remote for best model capability."""

        # PATH 1: Remote is optimal (quality preference)
        if remote_within_sla:
            reservation = await self._bm.try_reserve(
                request.request_id, COST_PER_REMOTE_REQUEST, est_remote_ms
            )
            if reservation is not None:
                return RoutingDecision(
                    route=RouteDecision.REMOTE,
                    reservation=reservation,
                    estimated_latency_ms=est_remote_ms,
                    reason=(
                        f"[QUALITY] Remote preferred for high-complexity request; "
                        f"budget available, est {est_remote_ms:.0f}ms ≤ SLA {request.latency_budget_ms:.0f}ms"
                    ),
                )
            # Budget exhausted — fall through to local

        # PATH 2: Remote budget exhausted — fall back to local
        if not at_capacity and local_within_sla:
            reservation = await self._bm.try_reserve(request.request_id, 0.0, est_local_ms)
            if reservation is not None:
                return RoutingDecision(
                    route=RouteDecision.LOCAL,
                    reservation=reservation,
                    estimated_latency_ms=est_local_ms,
                    reason=(
                        f"[QUALITY] Remote budget exhausted or over SLA; "
                        f"falling back to local (est {est_local_ms:.0f}ms)"
                    ),
                )
            return RoutingDecision(
                route=RouteDecision.FAIL_FAST,
                reservation=None,
                estimated_latency_ms=est_local_ms,
                reason="[QUALITY] Global cumulative latency cap would be breached by local fallback",
            )

        # PATH 3: Degraded local fallback (SLA will be missed but we prefer degraded over rejection)
        if not at_capacity:
            degraded_est = self._lt.get_estimate(BACKEND_LOCAL) * (queue_depth + 1)
            reservation = await self._bm.try_reserve(request.request_id, 0.0, degraded_est)
            if reservation is not None:
                return RoutingDecision(
                    route=RouteDecision.QUEUE_LOCAL,
                    reservation=reservation,
                    estimated_latency_ms=degraded_est,
                    reason=(
                        f"[QUALITY] All preferred routes exhausted; degrading to local queue "
                        f"(SLA will be missed: est {degraded_est:.0f}ms)"
                    ),
                )
            return RoutingDecision(
                route=RouteDecision.FAIL_FAST,
                reservation=None,
                estimated_latency_ms=0.0,
                reason="[QUALITY] Budget exhausted and global latency cap would be breached",
            )

        # PATH 4: Hard fail-fast circuit breaker
        return RoutingDecision(
            route=RouteDecision.FAIL_FAST,
            reservation=None,
            estimated_latency_ms=0.0,
            reason=(
                "[QUALITY] All routes exhausted: remote budget exhausted, local at capacity"
            ),
        )
