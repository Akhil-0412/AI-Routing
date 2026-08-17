import pytest
from app.budget_manager import BudgetManager
import app.observability as obs

@pytest.mark.asyncio
async def test_reserve_lock_event_has_correct_request_id():
    """
    Ensure that the lock event logged during the 'reserve' phase uses the actual
    request_id passed into try_reserve(), rather than a hardcoded string like "reserve".
    """
    bm = BudgetManager(max_remote_budget=1.0, max_cumulative_latency_ms=1000.0)
    
    # Clear any previous traces
    obs.clear_lock_trace()
    
    test_id = "test-req-abc-123"
    await bm.try_reserve(test_id, 0.1, 10.0)
    
    trace = obs.get_lock_trace()
    
    # There should be an acquire and release event for the reserve phase
    reserve_events = [e for e in trace if e[2] == "reserve"]
    assert len(reserve_events) == 2
    
    for event in reserve_events:
        assert event[1] == test_id
        assert event[1] != "reserve"
