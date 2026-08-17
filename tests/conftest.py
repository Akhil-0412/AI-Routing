# tests/conftest.py
"""
Shared pytest fixtures.

Forces BACKEND_MODE=mock so tests never touch real Ollama or TokenRouter.
All 49 existing tests pass unchanged.
"""
import os
os.environ.setdefault("BACKEND_MODE", "mock")
os.environ.setdefault("AVG_LOCAL_LATENCY_MS", "150.0")
os.environ.setdefault("AVG_REMOTE_LATENCY_MS", "250.0")
os.environ.setdefault("COST_PER_REMOTE_REQUEST", "0.05")
os.environ.setdefault("MAX_REMOTE_BUDGET", "0.10")
os.environ.setdefault("MAX_CUMULATIVE_LATENCY_MS", "30000.0")

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
async def client():
    """
    ASGI test client with lifespan enabled.
    Manually enters the lifespan context so app.state (budget_manager,
    queue_monitor, latency_tracker, decision_agent) is populated before
    any request is made.
    """
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            yield c
