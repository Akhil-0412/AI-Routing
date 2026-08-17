"""
tests/test_concurrency.py

End-to-end concurrency proof tests via the FastAPI HTTP layer.

These are the primary "proof of correctness" tests. Each scenario corresponds
to one of the five demo scenarios in the implementation plan.

  Scenario A: Budget double-spend prevention (primary)
  Scenario B: Graceful degradation under exhaustion
  Scenario C: Remote failure + rollback — budget is reusable after failure
  Scenario D: Reservation expiry via TTL reaper
"""
import asyncio
import math

import pytest
from httpx import AsyncClient

from app.main import app


async def fetch_metrics(client: AsyncClient):
    resp = await client.get("/metrics")
    return resp.json()


async def fetch_status(client: AsyncClient):
    resp = await client.get("/status")
    return resp.json()


def assert_no_leaks(metrics: dict, max_budget: float):
    """Core invariant: no overspend and no orphaned reservations."""
    assert metrics["cumulative_remote_cost"] <= max_budget + 1e-9, (
        f"BUDGET OVERSPEND: {metrics['cumulative_remote_cost']:.6f} > {max_budget:.6f}"
    )
    assert math.isclose(metrics["reserved_remote_budget"], 0.0, abs_tol=1e-9), (
        f"RESERVATION LEAK: reserved={metrics['reserved_remote_budget']}"
    )
    assert metrics["active_remote_reservations"] == 0, (
        f"RESERVATION LEAK: {metrics['active_remote_reservations']} active"
    )
    assert metrics["active_local_inferences"] == 0, (
        f"QUEUE LEAK: {metrics['active_local_inferences']} active local"
    )
    assert math.isclose(metrics["reserved_latency_ms"], 0.0, abs_tol=1e-9), (
        f"LATENCY RESERVATION LEAK: {metrics['reserved_latency_ms']}"
    )


# ── Scenario A: Double-spend prevention ──────────────────────────────────────

async def test_scenario_a_no_budget_double_spend(client: AsyncClient):
    """
    5 concurrent requests when budget only allows 2 remote calls.
    Exactly $0.10 may be spent; never more.
    """
    requests = [
        {"prompt": f"concurrent-{i}", "latency_budget_ms": 100.0}
        for i in range(5)
    ]

    async def post(payload):
        resp = await client.post("/inference", json=payload)
        return resp.status_code, resp.json()

    results = await asyncio.gather(*[post(r) for r in requests])

    # Collect outcomes
    status_codes = [r[0] for r in results]
    remote_count = sum(
        1 for _, data in results
        if isinstance(data, dict) and data.get("route") == "REMOTE"
    )

    # At most 2 remote calls ($0.05 each, budget = $0.10)
    assert remote_count <= 2, f"Expected ≤2 remote, got {remote_count}"

    # No 500s (500 = unexpected internal error, 503 = correct fail-fast)
    assert 500 not in status_codes, "Internal error should never occur"

    metrics = await fetch_metrics(client)
    assert_no_leaks(metrics, max_budget=0.10)


# ── Scenario B: Graceful degradation ─────────────────────────────────────────

async def test_scenario_b_graceful_degradation(client: AsyncClient):
    """
    Requests with tight SLA force REMOTE; once budget is spent, system
    degrades to QUEUE_LOCAL or FAIL_FAST. Never 500.
    """
    requests = [
        {"prompt": f"degradation-{i}", "latency_budget_ms": 100.0}
        for i in range(8)
    ]

    async def post(payload):
        return (await client.post("/inference", json=payload)).status_code

    status_codes = await asyncio.gather(*[post(r) for r in requests])

    assert 500 not in status_codes, "No internal errors expected"
    assert any(s == 200 for s in status_codes), "At least some should succeed"

    metrics = await fetch_metrics(client)
    assert_no_leaks(metrics, max_budget=0.10)


# ── Scenario C: Remote failure + rollback ────────────────────────────────────

async def test_scenario_c_remote_failure_rollback(client: AsyncClient):
    """
    Three concurrent requests. The first contains 'fail_remote' in the prompt.
    After rollback, the budget should be fully available for subsequent requests.
    The system should not 500 on the failure — it should 503 or fall back.
    """
    requests = [
        {"prompt": "fail_remote trigger", "latency_budget_ms": 100.0},
        {"prompt": "normal request alpha", "latency_budget_ms": 600.0},
        {"prompt": "normal request beta", "latency_budget_ms": 600.0},
    ]

    async def post(payload):
        resp = await client.post("/inference", json=payload)
        return resp.status_code, resp.json()

    results = await asyncio.gather(*[post(r) for r in requests])
    status_codes = [r[0] for r in results]

    # The failure should produce 503, not 500
    assert 500 not in status_codes, f"Got unexpected 500: {results}"

    metrics = await fetch_metrics(client)
    assert_no_leaks(metrics, max_budget=0.10)


# ── Scenario D: Reservation TTL expiry ───────────────────────────────────────

async def test_scenario_d_ttl_reaper_recovers_budget():
    """
    Reserve budget, never settle it, wait past TTL.
    The reaper should recover the budget automatically.
    """
    from app.budget_manager import BudgetManager

    bm = BudgetManager(
        max_remote_budget=0.10,
        max_cumulative_latency_ms=99999.0,
        reservation_ttl_seconds=0.05,  # 50ms for test speed
    )

    # Reserve everything
    res = await bm.try_reserve('test-req', 0.10, 10.0)
    assert res is not None

    snap = await bm.budget_snapshot()
    assert snap.remaining == 0.0  # fully locked

    # Wait past TTL
    await asyncio.sleep(0.1)

    # Trigger reaper manually
    expired = await bm._expire_stale_reservations()
    assert len(expired) == 1, "Reaper should have expired exactly 1 reservation"

    snap = await bm.budget_snapshot()
    assert snap.remaining == pytest.approx(0.10), "Budget should be fully restored"
    assert snap.reserved == 0.0


# ── Stress test ───────────────────────────────────────────────────────────────

async def test_stress_50_concurrent_no_corruption(client: AsyncClient):
    """
    50 concurrent requests. Budget allows ≤2 remote calls.
    System must handle all without corruption or 500s.
    """
    requests = [
        {"prompt": f"stress-{i}", "latency_budget_ms": 600.0}
        for i in range(50)
    ]

    async def post(payload):
        return (await client.post("/inference", json=payload)).status_code

    status_codes = await asyncio.gather(*[post(r) for r in requests])

    assert 500 not in status_codes, "No internal errors in stress test"
    assert any(s == 200 for s in status_codes)

    metrics = await fetch_metrics(client)
    assert_no_leaks(metrics, max_budget=0.10)
