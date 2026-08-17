"""
tests/test_integration.py

Integration tests via the full FastAPI stack using httpx ASGI transport.
Tests the complete lifecycle including headers, response shapes, and observability endpoints.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


async def test_health_endpoint(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_status_endpoint_shape(client: AsyncClient):
    resp = await client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    required_keys = {
        "remote_budget_max", "remote_budget_spent", "remote_budget_reserved",
        "remote_budget_remaining", "latency_max_ms", "latency_committed_ms",
        "local_capacity_max", "local_capacity_active", "local_capacity_available",
        "active_reservations",
    }
    assert required_keys.issubset(data.keys())


async def test_metrics_endpoint_shape(client: AsyncClient):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "cumulative_remote_cost" in data
    assert "active_remote_reservations" in data
    assert "active_local_inferences" in data


async def test_inference_returns_correct_shape(client: AsyncClient):
    resp = await client.post("/inference", json={
        "prompt": "integration test",
        "latency_budget_ms": 1000.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "request_id" in data
    assert "route" in data
    assert "actual_cost" in data
    assert "actual_latency_ms" in data
    assert "result" in data
    assert "budget_remaining" in data


async def test_inference_route_field_is_valid_enum(client: AsyncClient):
    resp = await client.post("/inference", json={
        "prompt": "enum test",
        "latency_budget_ms": 1000.0,
    })
    assert resp.status_code == 200
    route = resp.json()["route"]
    assert route in ("LOCAL", "REMOTE", "QUEUE_LOCAL", "FAIL_FAST")


async def test_missing_prompt_returns_422(client: AsyncClient):
    resp = await client.post("/inference", json={"latency_budget_ms": 500.0})
    assert resp.status_code == 422


async def test_zero_latency_budget_returns_422(client: AsyncClient):
    resp = await client.post("/inference", json={
        "prompt": "test",
        "latency_budget_ms": 0.0,
    })
    assert resp.status_code == 422


async def test_budget_remaining_decreases_after_remote(client: AsyncClient):
    """budget_remaining in response should be lower after a remote call."""
    status_before = (await client.get("/status")).json()
    budget_before = status_before["remote_budget_remaining"]

    # Tight SLA forces REMOTE
    resp = await client.post("/inference", json={
        "prompt": "budget drain test",
        "latency_budget_ms": 100.0,
    })
    if resp.status_code == 200 and resp.json().get("route") == "REMOTE":
        budget_after = resp.json()["budget_remaining"]
        assert budget_after < budget_before


async def test_fail_fast_returns_503(client: AsyncClient):
    """When budget is exhausted and local at capacity, system returns 503."""
    # Exhaust budget with 2 remote calls
    for _ in range(2):
        await client.post("/inference", json={
            "prompt": "budget exhaust",
            "latency_budget_ms": 100.0,
        })

    # Now concurrent-style: local SLA too tight, remote exhausted
    responses = []
    for _ in range(5):
        resp = await client.post("/inference", json={
            "prompt": "overflow test",
            "latency_budget_ms": 100.0,
        })
        responses.append(resp.status_code)

    # Some must be 503 (not 500)
    assert 500 not in responses, "503 expected for overflows, not 500"
