"""
tests/test_latency_tracker.py

Unit tests for LatencyTracker.
"""
import asyncio
import pytest
from app.latency_tracker import LatencyTracker, BACKEND_LOCAL, BACKEND_REMOTE
from app.config import AVG_LOCAL_LATENCY_MS, AVG_REMOTE_LATENCY_MS


@pytest.fixture
def lt():
    return LatencyTracker(alpha=0.5, window_size=3)


async def test_seed_values_are_config_defaults(lt):
    assert lt.get_estimate(BACKEND_LOCAL) == AVG_LOCAL_LATENCY_MS
    assert lt.get_estimate(BACKEND_REMOTE) == AVG_REMOTE_LATENCY_MS


async def test_recording_updates_estimate(lt):
    await lt.record(BACKEND_LOCAL, 200.0)
    estimate = lt.get_estimate(BACKEND_LOCAL)
    # With alpha=0.5 and seed=AVG_LOCAL_LATENCY_MS: ema = 0.5*200 + 0.5*seed
    # Estimate = max(ema, 200) so should be at least 200
    assert estimate >= 200.0


async def test_spike_detection(lt):
    """
    After a spike, get_estimate() should return ≥ spike value immediately.
    This is the core of the hybrid EMA + window approach.
    """
    # Prime the EMA with low values
    for _ in range(5):
        await lt.record(BACKEND_REMOTE, 100.0)

    ema_before = lt.get_ema(BACKEND_REMOTE)
    assert ema_before < 150.0  # Should be converging toward 100

    # Single spike
    await lt.record(BACKEND_REMOTE, 2000.0)
    estimate_after = lt.get_estimate(BACKEND_REMOTE)

    # get_estimate returns max(EMA, recent_max), so must be ≥ 2000
    assert estimate_after >= 2000.0


async def test_estimate_relaxes_after_spike_slides_out(lt):
    """After WINDOW_SIZE normal readings, the spike should no longer dominate."""
    await lt.record(BACKEND_REMOTE, 2000.0)  # spike
    # window_size = 3, so 3 more normal readings push the spike out
    for _ in range(3):
        await lt.record(BACKEND_REMOTE, 200.0)

    estimate = lt.get_estimate(BACKEND_REMOTE)
    # Window is now [2000, 200, 200, 200] with maxlen=3 → [200, 200, 200]
    # recent_max = 200; estimate = max(EMA, 200)
    assert estimate < 500.0


async def test_reset_restores_defaults(lt):
    await lt.record(BACKEND_LOCAL, 5000.0)
    await lt.reset()
    assert lt.get_estimate(BACKEND_LOCAL) == AVG_LOCAL_LATENCY_MS


async def test_concurrent_recording_does_not_crash():
    """Concurrent record() calls should not cause data corruption."""
    lt = LatencyTracker(alpha=0.25, window_size=10)

    async def record_many():
        for _ in range(20):
            await lt.record(BACKEND_LOCAL, 100.0)
            await asyncio.sleep(0)

    await asyncio.gather(*[record_many() for _ in range(5)])
    # Should finish without error; estimate should be close to 100ms
    assert lt.get_estimate(BACKEND_LOCAL) < 300.0
