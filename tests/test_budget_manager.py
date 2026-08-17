"""
tests/test_budget_manager.py

Unit tests for BudgetManager — the core concurrency primitive.

Tests cover:
  - Happy path reserve → commit
  - Reserve → rollback (funds returned)
  - Double-spend prevention under concurrent load
  - Actual cost exceeding reservation → rolled back
  - Latency cap enforcement
  - Idempotent settlement (double-commit raises)
  - TTL expiry and reaper recovery
  - Float precision (no sub-cent rounding errors)
"""
import asyncio
import time

import pytest

from app.budget_manager import (
    ActualCostExceededReservationError,
    ActualLatencyExceededReservationError,
    BudgetManager,
    InvalidReservationStateError,
)


@pytest.fixture
def bm():
    return BudgetManager(
        max_remote_budget=0.10,
        max_cumulative_latency_ms=5000.0,
        reservation_ttl_seconds=2.0,
    )


# ── Basic reserve / commit ────────────────────────────────────────────────────

async def test_reserve_and_commit(bm):
    res = await bm.try_reserve('test-req', 0.05, 200.0)
    assert res is not None
    assert res.state == "pending"

    await res.commit(0.05, 200.0)
    assert res.state == "committed"

    snap = await bm.budget_snapshot()
    assert snap.cumulative_cost == pytest.approx(0.05)
    assert snap.reserved == 0.0
    assert snap.remaining == pytest.approx(0.05)


async def test_reserve_and_rollback(bm):
    res = await bm.try_reserve('test-req', 0.05, 200.0)
    assert res is not None

    await res.rollback()
    assert res.state == "rolled_back"

    snap = await bm.budget_snapshot()
    assert snap.cumulative_cost == 0.0
    assert snap.reserved == 0.0
    assert snap.remaining == pytest.approx(0.10)


# ── Budget enforcement ────────────────────────────────────────────────────────

async def test_insufficient_budget_returns_none(bm):
    """Requesting more than available returns None immediately."""
    res = await bm.try_reserve('test-req', 0.15, 100.0)
    assert res is None


async def test_no_double_spend_under_concurrency():
    """
    Core concurrency test: 10 coroutines all try to reserve $0.05 from a $0.10 budget.
    Exactly 2 should succeed; the other 8 must get None.
    No overspend is allowed.
    """
    bm = BudgetManager(max_remote_budget=0.10, max_cumulative_latency_ms=100000.0)

    async def attempt():
        return await bm.try_reserve('test-req', 0.05, 10.0)

    reservations = await asyncio.gather(*[attempt() for _ in range(10)])
    granted = [r for r in reservations if r is not None]
    assert len(granted) == 2

    snap = await bm.budget_snapshot()
    assert snap.reserved == pytest.approx(0.10)
    assert snap.remaining == 0.0

    # Settle all
    for r in granted:
        await r.commit(0.05, 10.0)

    snap = await bm.budget_snapshot()
    assert snap.cumulative_cost == pytest.approx(0.10)
    assert snap.reserved == 0.0


async def test_actual_cost_exceeds_reservation_raises(bm):
    res = await bm.try_reserve('test-req', 0.05, 100.0)
    assert res is not None

    with pytest.raises(ActualCostExceededReservationError):
        await res.commit(0.06, 100.0)  # 0.06 > 0.05

    # Reservation should have been rolled back
    snap = await bm.budget_snapshot()
    assert snap.reserved == 0.0


# ── Latency cap ───────────────────────────────────────────────────────────────

async def test_latency_cap_enforcement():
    bm = BudgetManager(max_remote_budget=10.0, max_cumulative_latency_ms=500.0)

    res = await bm.try_reserve('test-req', 0.01, 400.0)
    assert res is not None

    # Committing 400ms brings total to 400ms — fine
    await res.commit(0.01, 400.0)

    # Next reservation: only 100ms left
    res2 = await bm.try_reserve('test-req', 0.01, 200.0)  # 200ms > 100ms remaining
    assert res2 is None


async def test_actual_latency_exceeds_cap_raises():
    bm = BudgetManager(max_remote_budget=10.0, max_cumulative_latency_ms=300.0)

    res = await bm.try_reserve('test-req', 0.01, 300.0)
    assert res is not None

    with pytest.raises(ActualLatencyExceededReservationError):
        # Reserve was fine (300ms == cap), but commit actual of 301ms would breach
        await res.commit(0.01, 301.0)

    snap = await bm.budget_snapshot()
    assert snap.reserved == 0.0


# ── Idempotency guard ─────────────────────────────────────────────────────────

async def test_double_commit_raises(bm):
    res = await bm.try_reserve('test-req', 0.05, 100.0)
    await res.commit(0.05, 100.0)

    with pytest.raises(InvalidReservationStateError):
        await res.commit(0.05, 100.0)


async def test_double_rollback_raises(bm):
    res = await bm.try_reserve('test-req', 0.05, 100.0)
    await res.rollback()

    with pytest.raises(InvalidReservationStateError):
        await res.rollback()


async def test_commit_after_rollback_raises(bm):
    res = await bm.try_reserve('test-req', 0.05, 100.0)
    await res.rollback()

    with pytest.raises(InvalidReservationStateError):
        await res.commit(0.05, 100.0)


# ── TTL Expiry (Weakness §3.2) ────────────────────────────────────────────────

async def test_reservation_is_expired_after_ttl():
    """is_expired flag is set after TTL seconds."""
    bm = BudgetManager(
        max_remote_budget=1.0,
        max_cumulative_latency_ms=99999.0,
        reservation_ttl_seconds=0.05,  # 50ms for fast tests
    )
    res = await bm.try_reserve('test-req', 0.10, 10.0)
    assert res is not None
    assert not res.is_expired

    await asyncio.sleep(0.1)  # wait past TTL
    assert res.is_expired


async def test_reaper_recovers_stale_reservation():
    """
    A reservation that is never settled should be automatically rolled back
    by _expire_stale_reservations() and the budget returned to the pool.
    """
    bm = BudgetManager(
        max_remote_budget=0.10,
        max_cumulative_latency_ms=99999.0,
        reservation_ttl_seconds=0.05,
    )
    res = await bm.try_reserve('test-req', 0.10, 10.0)
    assert res is not None

    snap = await bm.budget_snapshot()
    assert snap.remaining == 0.0  # locked out

    await asyncio.sleep(0.1)  # wait past TTL

    # Manually trigger reaper (don't need background task for unit test)
    expired = await bm._expire_stale_reservations()
    assert len(expired) == 1

    snap = await bm.budget_snapshot()
    assert snap.reserved == 0.0
    assert snap.remaining == pytest.approx(0.10)  # budget recovered


# ── Float precision ───────────────────────────────────────────────────────────

async def test_many_small_commits_do_not_overspend():
    """
    20 x $0.005 commits should total exactly $0.10 — no float bleed.
    """
    bm = BudgetManager(max_remote_budget=0.10, max_cumulative_latency_ms=99999.0)

    for _ in range(20):
        res = await bm.try_reserve('test-req', 0.005, 1.0)
        assert res is not None, "Should have budget for 20 x $0.005 = $0.10"
        await res.commit(0.005, 1.0)

    snap = await bm.budget_snapshot()
    assert snap.cumulative_cost == pytest.approx(0.10, abs=1e-9)
    assert snap.remaining == pytest.approx(0.0, abs=1e-9)

    # 21st should fail
    res = await bm.try_reserve('test-req', 0.005, 1.0)
    assert res is None
