"""
tests/test_queue_monitor.py

Unit tests for QueueMonitor.
"""
import asyncio
import pytest
from app.queue_monitor import QueueMonitor


@pytest.fixture
def qm():
    return QueueMonitor(max_concurrency=2)


async def test_acquire_within_capacity(qm):
    result = await qm.acquire()
    assert result is True
    assert qm.active_count == 1


async def test_acquire_at_capacity_returns_false(qm):
    await qm.acquire()
    await qm.acquire()
    assert qm.is_at_capacity

    result = await qm.acquire()
    assert result is False
    assert qm.active_count == 2


async def test_release_frees_slot(qm):
    await qm.acquire()
    await qm.acquire()
    await qm.release()
    assert qm.active_count == 1
    assert not qm.is_at_capacity


async def test_slot_context_manager_guarantees_release(qm):
    async with qm.slot():
        assert qm.active_count == 1
    assert qm.active_count == 0


async def test_slot_releases_on_exception(qm):
    try:
        async with qm.slot():
            assert qm.active_count == 1
            raise ValueError("injected error")
    except ValueError:
        pass
    assert qm.active_count == 0


async def test_slot_raises_when_at_capacity(qm):
    await qm.acquire()
    await qm.acquire()
    with pytest.raises(RuntimeError, match="capacity"):
        async with qm.slot():
            pass


async def test_concurrent_acquire_respects_limit():
    """50 concurrent coroutines — only max_concurrency may hold a slot."""
    qm = QueueMonitor(max_concurrency=3)
    held_at_peak = []

    async def worker():
        acquired = await qm.acquire()
        if acquired:
            held_at_peak.append(qm.active_count)
            await asyncio.sleep(0.02)
            await qm.release()

    await asyncio.gather(*[worker() for _ in range(50)])
    assert max(held_at_peak) <= 3
    assert qm.active_count == 0


async def test_release_below_zero_is_safe(qm):
    """Calling release without a prior acquire should not go negative."""
    await qm.release()
    assert qm.active_count == 0
