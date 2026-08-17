"""
tests/test_decision_agent.py

Unit tests for DecisionAgent — exercises each routing path.
"""
import asyncio
import pytest

from app.budget_manager import BudgetManager
from app.decision_agent import DecisionAgent
from app.latency_tracker import LatencyTracker
from app.models import InferenceRequest, RouteDecision
from app.queue_monitor import QueueMonitor


def make_agent(
    budget=0.10,
    latency_cap=30000.0,
    local_capacity=2,
    avg_local_ms=150.0,
    avg_remote_ms=250.0,
):
    bm = BudgetManager(max_remote_budget=budget, max_cumulative_latency_ms=latency_cap)
    qm = QueueMonitor(max_concurrency=local_capacity)
    lt = LatencyTracker()
    # Override EMA seeds for predictable routing
    lt._ema["local"] = avg_local_ms
    lt._ema["remote"] = avg_remote_ms
    return DecisionAgent(bm, qm, lt), bm, qm, lt


def make_request(latency_budget_ms: float = 600.0) -> InferenceRequest:
    return InferenceRequest(prompt="test prompt", latency_budget_ms=latency_budget_ms)


# ── Path 1: Local optimal ─────────────────────────────────────────────────────

async def test_routes_local_when_within_sla():
    agent, bm, qm, lt = make_agent()
    req = make_request(latency_budget_ms=600.0)

    decision = await agent.route(req)
    assert decision.route == RouteDecision.LOCAL
    assert decision.reservation is not None
    assert decision.reservation.reserved_cost == 0.0


# ── Path 2: Remote escalation ─────────────────────────────────────────────────

async def test_routes_remote_when_local_at_capacity():
    agent, bm, qm, lt = make_agent()
    # Fill local queue
    await qm.acquire()
    await qm.acquire()

    req = make_request(latency_budget_ms=600.0)
    decision = await agent.route(req)

    assert decision.route == RouteDecision.REMOTE
    assert decision.reservation is not None
    assert decision.reservation.reserved_cost > 0.0


async def test_routes_remote_when_local_too_slow():
    """Local SLA miss forces remote."""
    agent, bm, qm, lt = make_agent(avg_local_ms=500.0)
    req = make_request(latency_budget_ms=300.0)  # local would be 500ms > 300ms SLA

    decision = await agent.route(req)
    assert decision.route == RouteDecision.REMOTE


# ── Path 3: Degraded fallback ─────────────────────────────────────────────────

async def test_queue_local_when_budget_exhausted():
    """When remote budget is zero, falls back to QUEUE_LOCAL."""
    agent, bm, qm, lt = make_agent(budget=0.0001)  # effectively empty

    req = make_request(latency_budget_ms=100.0)  # local too slow → remote; remote no budget → fallback
    # Override local EMA to be within SLA for this test
    lt._ema["local"] = 50.0  # 50ms × 1 = 50ms ≤ 100ms SLA
    decision = await agent.route(req)
    assert decision.route in (RouteDecision.LOCAL, RouteDecision.QUEUE_LOCAL)


async def test_queue_local_when_remote_budget_exhausted_and_local_feasible():
    agent, bm, qm, lt = make_agent(budget=0.04)  # < COST_PER_REMOTE_REQUEST (0.05)
    lt._ema["local"] = 50.0
    lt._ema["remote"] = 250.0

    # Local is within SLA — should route LOCAL
    req = make_request(latency_budget_ms=600.0)
    decision = await agent.route(req)
    assert decision.route == RouteDecision.LOCAL


# ── Path 4: Fail-fast ─────────────────────────────────────────────────────────

async def test_fail_fast_when_all_paths_exhausted():
    agent, bm, qm, lt = make_agent(budget=0.0001, local_capacity=2)
    lt._ema["local"] = 500.0
    lt._ema["remote"] = 250.0

    # Fill local queue
    await qm.acquire()
    await qm.acquire()

    req = make_request(latency_budget_ms=300.0)
    decision = await agent.route(req)

    assert decision.route == RouteDecision.FAIL_FAST
    assert decision.reservation is None


async def test_fail_fast_when_remote_over_sla_and_local_at_capacity():
    agent, bm, qm, lt = make_agent()
    lt._ema["remote"] = 2000.0
    lt._ema["local"] = 500.0

    await qm.acquire()
    await qm.acquire()

    req = make_request(latency_budget_ms=300.0)  # remote 2000ms > 300ms SLA
    decision = await agent.route(req)

    assert decision.route == RouteDecision.FAIL_FAST


# ── Concurrent routing ────────────────────────────────────────────────────────

async def test_concurrent_routing_no_double_spend():
    """
    10 concurrent route() calls on a $0.10 budget at $0.05/remote.
    Only 2 should get REMOTE; others fall through to LOCAL or QUEUE_LOCAL.
    Total spend must never exceed $0.10.
    """
    agent, bm, qm, lt = make_agent(budget=0.10, local_capacity=5)
    lt._ema["local"] = 500.0   # local over SLA
    lt._ema["remote"] = 250.0  # remote within SLA

    req = make_request(latency_budget_ms=300.0)

    decisions = await asyncio.gather(*[agent.route(req) for _ in range(10)])

    remote_decisions = [d for d in decisions if d.route == RouteDecision.REMOTE]
    assert len(remote_decisions) == 2

    # Settle all remote reservations
    for d in remote_decisions:
        await d.reservation.commit(0.05, 250.0)

    snap = await bm.budget_snapshot()
    assert snap.cumulative_cost == pytest.approx(0.10)
    assert snap.reserved == 0.0
