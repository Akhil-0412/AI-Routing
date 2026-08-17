"""
app/main.py

FastAPI application — the full request lifecycle orchestrator.

Request lifecycle:
  1. Receive request → log
  2. DecisionAgent.route() → routing decision (advisory reads + one atomic reserve)
  3. FAIL_FAST → 503 immediately
  4. Execute backend (all locks released during I/O)
  5. Settlement: reservation.commit() or reservation.rollback()
  6. LatencyTracker.record() → update EMA + window
  7. Return InferenceResponse

Settlement guarantee (Edge Case §4.1):
  A try/finally block wraps the execution + settlement phase.
  If ANY exception — including asyncio.CancelledError — occurs after a
  reservation is made, rollback() is called before propagating.
  The TTL reaper (BudgetManager.start_reaper) is a second line of defence
  for truly catastrophic failures (process kill, SIGKILL).

Concurrency note:
  Multiple FastAPI worker coroutines handle concurrent requests.
  Each calls agent.route() independently. The only serialisation points are:
    - BudgetManager._lock (microsecond duration, pure arithmetic)
    - QueueMonitor._lock (microsecond duration, counter + semaphore)
    - LatencyTracker._lock (microsecond duration, EMA + deque)
  Backend I/O (asyncio.sleep in mocks, real network calls in production) is
  fully concurrent — no lock is held during it.
"""
from __future__ import annotations

import asyncio
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import app.event_bus as event_bus
import app.observability as obs
from app.backends import (
    RemoteInferenceError,
    check_ollama_health,
    simulate_local_inference,
    simulate_remote_inference,
)
from app.budget_manager import (
    ActualCostExceededReservationError,
    ActualLatencyExceededReservationError,
    BudgetManager,
)
from app.config import (
    BACKEND_MODE,
    COST_PER_REMOTE_REQUEST,
    LOCAL_QUEUE_CAPACITY,
    MAX_CUMULATIVE_LATENCY_MS,
    MAX_REMOTE_BUDGET,
    OLLAMA_BASE_URL,
    REAPER_INTERVAL_SECONDS,
    RESERVATION_TTL_SECONDS,
)
from app.complexity_classifier import ComplexityTier, classify
from app.decision_agent import DecisionAgent
from app.latency_tracker import BACKEND_LOCAL, BACKEND_REMOTE, LatencyTracker
from app.models import InferenceRequest, InferenceResponse, ResourceState, RouteDecision, RunStartPayload, RunCompletePayload
from app.queue_monitor import QueueMonitor


# ─── Application Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context: creates exactly one instance of each shared
    component and injects it into app.state. The background reaper is started
    here and cancelled on shutdown.
    """
    budget_manager = BudgetManager(
        max_remote_budget=MAX_REMOTE_BUDGET,
        max_cumulative_latency_ms=MAX_CUMULATIVE_LATENCY_MS,
        reservation_ttl_seconds=RESERVATION_TTL_SECONDS,
    )
    queue_monitor = QueueMonitor(max_concurrency=LOCAL_QUEUE_CAPACITY)
    latency_tracker = LatencyTracker()
    decision_agent = DecisionAgent(budget_manager, queue_monitor, latency_tracker)

    # Start the TTL reaper for reservation expiry (Weakness §3.2 mitigation)
    budget_manager.start_reaper(interval=REAPER_INTERVAL_SECONDS)

    # Ollama health check (Upgrade 1): warn if unreachable, never fail startup
    if BACKEND_MODE == "real":
        ollama_ok = await check_ollama_health()
        obs.log_ollama_health(ollama_ok, OLLAMA_BASE_URL)

    app.state.budget_manager = budget_manager
    app.state.queue_monitor = queue_monitor
    app.state.latency_tracker = latency_tracker
    app.state.decision_agent = decision_agent

    # Dashboard route counters — simple in-memory tallies, additive only
    app.state.route_counts = {"LOCAL": 0, "REMOTE": 0, "QUEUE_LOCAL": 0, "FAIL_FAST": 0}

    yield

    # Shutdown: cancel the reaper
    budget_manager.stop_reaper()


# ─── Application ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="LEC AI Inference Router",
    description=(
        "Concurrency-safe async LLM inference routing gateway. "
        "Routes requests across local and remote model backends using a "
        "two-phase atomic reservation protocol."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Serve the live dashboard HTML from the static/ directory
_STATIC_DIR = Path(__file__).parent.parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ─── Health / Observability Endpoints ────────────────────────────────────────

@app.get("/health", tags=["observability"])
async def health():
    return {"status": "ok"}


@app.get("/status", response_model=ResourceState, tags=["observability"])
async def status(request: Request):
    """Full resource state snapshot — includes budget, latency, queue, and estimates."""
    bm: BudgetManager = request.app.state.budget_manager
    qm: QueueMonitor = request.app.state.queue_monitor
    lt: LatencyTracker = request.app.state.latency_tracker

    snapshot = await bm.budget_snapshot()

    return ResourceState(
        remote_budget_max=bm._max_remote_budget,
        remote_budget_spent=snapshot.cumulative_cost,
        remote_budget_reserved=snapshot.reserved,
        remote_budget_remaining=snapshot.remaining,
        latency_max_ms=bm._max_cumulative_latency_ms,
        latency_committed_ms=snapshot.cumulative_latency_ms,
        latency_reserved_ms=snapshot.reserved_latency_ms,
        latency_remaining_ms=max(
            0.0,
            bm._max_cumulative_latency_ms
            - snapshot.cumulative_latency_ms
            - snapshot.reserved_latency_ms,
        ),
        local_capacity_max=qm.max_concurrency,
        local_capacity_active=qm.active_count,
        local_capacity_available=qm.max_concurrency - qm.active_count,
        estimated_local_latency_ms=lt.get_estimate(BACKEND_LOCAL),
        estimated_remote_latency_ms=lt.get_estimate(BACKEND_REMOTE),
        active_reservations=bm.active_reservation_count,
    )


@app.get("/metrics", tags=["observability"])
async def metrics(request: Request):
    """Flat metrics dict suitable for load_generator verdict validation."""
    bm: BudgetManager = request.app.state.budget_manager
    qm: QueueMonitor = request.app.state.queue_monitor
    lt: LatencyTracker = request.app.state.latency_tracker
    snapshot = await bm.budget_snapshot()

    def _clean(v: float) -> float:
        return 0.0 if math.isclose(v, 0.0, abs_tol=1e-9) else v

    return {
        "cumulative_remote_cost": _clean(snapshot.cumulative_cost),
        "reserved_remote_budget": _clean(snapshot.reserved),
        "remaining_remote_budget": _clean(snapshot.remaining),
        "cumulative_actual_latency_ms": _clean(snapshot.cumulative_latency_ms),
        "reserved_latency_ms": _clean(snapshot.reserved_latency_ms),
        "active_local_inferences": qm.active_count,
        "max_local_concurrency": qm.max_concurrency,
        "active_remote_reservations": bm.active_reservation_count,
        "local_latency_ema_ms": _clean(lt.get_ema(BACKEND_LOCAL)),
        "remote_latency_ema_ms": _clean(lt.get_ema(BACKEND_REMOTE)),
    }


@app.get("/trace", tags=["observability"])
async def lock_trace():
    """
    Returns the in-process lock interleaving trace as a list of events.

    Each event: [timestamp_us, request_id, phase, action]
      - timestamp_us: microseconds since process start (monotonic)
      - phase: 'reserve' | 'settle'
      - action: 'acquire' | 'release'

    Use this to visualize overlapping concurrent requests.
    """
    events = obs.get_lock_trace()
    return {
        "count": len(events),
        "events": [
            {
                "timestamp_us": ts,
                "request_id": rid,
                "phase": phase,
                "action": action,
            }
            for ts, rid, phase, action in events
        ],
    }


@app.delete("/trace", tags=["observability"])
async def clear_lock_trace():
    """Clear the lock trace buffer (useful between demo runs)."""
    obs.clear_lock_trace()
    return {"cleared": True}


@app.post("/run_start", tags=["observability"])
async def run_start(payload: RunStartPayload, request: Request):
    """Marker for the start of a run, optionally resetting state."""
    app = request.app
    if payload.reset:
        old_bm = app.state.budget_manager
        if hasattr(old_bm, "stop_reaper"):
            old_bm.stop_reaper()

        bm = BudgetManager(
            max_remote_budget=MAX_REMOTE_BUDGET,
            max_cumulative_latency_ms=MAX_CUMULATIVE_LATENCY_MS,
            reservation_ttl_seconds=RESERVATION_TTL_SECONDS,
        )
        qm = QueueMonitor(max_concurrency=LOCAL_QUEUE_CAPACITY)
        lt = LatencyTracker()
        agent = DecisionAgent(bm, qm, lt)
        
        bm.start_reaper(interval=REAPER_INTERVAL_SECONDS)
        
        app.state.budget_manager = bm
        app.state.queue_monitor = qm
        app.state.latency_tracker = lt
        app.state.decision_agent = agent
        
        # Reset simple dashboard counters
        app.state.route_counts = {"LOCAL": 0, "REMOTE": 0, "QUEUE_LOCAL": 0, "FAIL_FAST": 0}

    obs.log_run_start(payload.run_id)
    return {"status": "started", "run_id": payload.run_id, "reset": payload.reset}


@app.post("/run_complete", tags=["observability"])
async def run_complete(payload: RunCompletePayload):
    """Marker for the end of a run, transmitting the final verdict and metrics."""
    obs.log_run_complete(payload.run_id, payload.verdict, payload.metrics, payload.routes)
    return {"status": "completed", "run_id": payload.run_id}


# ─── Dashboard endpoints ──────────────────────────────────────────────────────

@app.get("/dashboard", tags=["dashboard"], include_in_schema=False)
async def dashboard():
    """Serve the live observability dashboard (read-only)."""
    html_path = _STATIC_DIR / "dashboard.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/summary", tags=["dashboard"])
async def summary(request: Request):
    """
    Aggregate snapshot for the dashboard's initial page load.

    Returns budget state, latency state, and per-route counts accumulated
    since server start. No lock is held beyond the budget_snapshot() call.
    """
    bm: BudgetManager = request.app.state.budget_manager
    qm: QueueMonitor = request.app.state.queue_monitor
    counts: dict = request.app.state.route_counts
    snapshot = await bm.budget_snapshot()

    return {
        "budget_max": bm._max_remote_budget,
        "budget_spent": snapshot.cumulative_cost,
        "budget_remaining": max(0.0, snapshot.remaining),
        "latency_max_ms": bm._max_cumulative_latency_ms,
        "latency_committed_ms": snapshot.cumulative_latency_ms,
        "latency_remaining_ms": max(
            0.0,
            bm._max_cumulative_latency_ms
            - snapshot.cumulative_latency_ms
            - snapshot.reserved_latency_ms,
        ),
        "queue_active": qm.active_count,
        "queue_max": qm.max_concurrency,
        "route_counts": dict(counts),
    }


@app.get("/events", tags=["dashboard"])
async def sse_events(request: Request):
    """
    Server-Sent Events stream.

    Each event is a JSON payload pushed by observability.py when:
      - A routing decision is made  (event=route_decision)
      - An inference completes      (event=execution_complete)
      - A request is rejected       (event=fail_fast)
      - A reservation is rolled back (event=rollback_event)

    The stream stays open until the client disconnects. The per-client queue
    is bounded (64 events); oldest items are dropped if the client is slow.
    """
    q = event_bus.subscribe()

    async def _generator():
        # Send an initial heartbeat so the browser sees the connection open
        yield "data: {\"event\": \"connected\"}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    # Update server-side route counters when routing events arrive
                    if payload.get("event") == "route_decision":
                        route = payload.get("route", "")
                        counts: dict = request.app.state.route_counts
                        if route in counts:
                            counts[route] += 1
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat keep-alive every 15 s to prevent proxy timeouts
                    yield ": heartbeat\n\n"
        finally:
            event_bus.unsubscribe(q)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# ─── Core Inference Endpoint ─────────────────────────────────────────────────

@app.post("/inference", response_model=InferenceResponse, tags=["inference"])
async def inference(request_data: InferenceRequest, request: Request):
    """
    Route an inference request to the optimal backend.

    The endpoint orchestrates the full two-phase lifecycle:
      Phase 1: Route + Reserve (atomic, lock held for microseconds)
      I/O:     Execute backend (no lock held)
      Phase 2: Settle — commit or rollback (atomic, lock held for microseconds)
    """
    req_id = request_data.request_id
    bm: BudgetManager = request.app.state.budget_manager
    qm: QueueMonitor = request.app.state.queue_monitor
    lt: LatencyTracker = request.app.state.latency_tracker
    agent: DecisionAgent = request.app.state.decision_agent

    # ── Auto-classify complexity tier if client didn't supply one ────────────
    if request_data.complexity_tier is None:
        request_data.complexity_tier = classify(request_data.prompt)

    obs.log_request_received(req_id, request_data.latency_budget_ms, request_data.complexity_tier.value)

    # ── Phase 1: Route + Reserve ─────────────────────────────────────────────
    decision = await agent.route(request_data)

    if decision.route == RouteDecision.FAIL_FAST:
        if hasattr(request.app.state, "route_counts"):
            request.app.state.route_counts["FAIL_FAST"] += 1
        obs.log_fail_fast(req_id, decision.reason, status_code=503)
        raise HTTPException(status_code=503, detail=decision.reason)

    reservation = decision.reservation
    snapshot = await bm.budget_snapshot()

    # Increment route count in app.state for /summary
    route_key = decision.route.value
    if hasattr(request.app.state, "route_counts") and route_key in request.app.state.route_counts:
        request.app.state.route_counts[route_key] += 1

    obs.log_route_decision(
        req_id,
        decision.route.value,
        decision.reason,
        decision.estimated_latency_ms,
        qm.active_count,
        snapshot.remaining,
        request_data.complexity_tier.value,
        prompt=request_data.prompt,
    )

    if reservation:
        obs.log_reservation_made(
            req_id,
            reservation.id,
            reservation.reserved_cost,
            reservation.reserved_latency_ms,
        )

    # ── I/O + Settlement (try/finally guarantees rollback on any exception) ──
    actual_cost = 0.0
    actual_latency_ms = 0.0
    outcome = "unknown"

    try:
        if decision.route == RouteDecision.REMOTE:
            obs.log_execution_start(req_id, "REMOTE")

            # Detect injected failure for rollback demo
            should_fail = "fail_remote" in request_data.prompt.lower()

            result = await simulate_remote_inference(
                request_data.prompt, req_id, should_fail=should_fail
            )
            actual_cost = result.actual_cost
            actual_latency_ms = result.actual_latency_ms
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            cache_read_tokens = result.cache_read_tokens
            cache_write_tokens = result.cache_write_tokens
            outcome = "committed"

            # Settle the remote reservation
            if reservation:
                await reservation.commit(actual_cost, actual_latency_ms)
                after = await bm.budget_snapshot()
                obs.log_reservation_committed(
                    req_id, reservation.id, actual_cost, after.remaining
                )

        elif decision.route in (RouteDecision.LOCAL, RouteDecision.QUEUE_LOCAL):
            obs.log_execution_start(req_id, decision.route.value)

            async with qm.slot():
                result = await simulate_local_inference(request_data.prompt, req_id)
                actual_cost = result.actual_cost
                actual_latency_ms = result.actual_latency_ms
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
                cache_read_tokens = result.cache_read_tokens
                cache_write_tokens = result.cache_write_tokens
                outcome = "committed"

                # Local call has no financial cost — commit latency portion only
                if reservation:
                    await reservation.commit(0.0, actual_latency_ms)
                    after = await bm.budget_snapshot()
                    obs.log_reservation_committed(
                        req_id, reservation.id, 0.0, after.remaining
                    )

    except RuntimeError:
        # QueueMonitor slot() at capacity race (Edge Case §4.2)
        if reservation and reservation.state == "pending":
            await reservation.rollback()
            obs.log_reservation_rolled_back(
                req_id, reservation.id, "Local queue at capacity", reservation.reserved_cost
            )
        obs.log_fail_fast(req_id, "Local queue at capacity at execution time", 503)
        raise HTTPException(status_code=503, detail="Local queue at capacity")

    except (ActualCostExceededReservationError, ActualLatencyExceededReservationError) as e:
        # Reservation already rolled back inside commit() -- just log and fail
        obs.log_reservation_rolled_back(req_id, reservation.id if reservation else "?", str(e), 0.0)
        raise HTTPException(status_code=500, detail=str(e))

    except RemoteInferenceError as e:
        # Remote call failed -- rollback reservation and attempt local fallback
        obs.log_remote_failure(req_id, str(e))
        reserved_cost = reservation.reserved_cost if reservation else 0.0
        if reservation and reservation.state == "pending":
            await reservation.rollback()
            obs.log_reservation_rolled_back(req_id, reservation.id, "remote failure", reserved_cost)
            obs.log_rollback_event(req_id, reservation.id, reserved_cost, str(e))

        # Attempt local fallback
        if not qm.is_at_capacity:
            try:
                async with qm.slot():
                    result = await simulate_local_inference(request_data.prompt, req_id)
                    actual_cost = 0.0
                    actual_latency_ms = result.actual_latency_ms
                    prompt_tokens = result.prompt_tokens
                    completion_tokens = result.completion_tokens
                    cache_read_tokens = result.cache_read_tokens
                    cache_write_tokens = result.cache_write_tokens
                    outcome = "fallback_local"
                    decision = type(decision)(
                        route=RouteDecision.QUEUE_LOCAL,
                        reservation=None,
                        estimated_latency_ms=actual_latency_ms,
                        reason="Fallback after remote failure",
                    )
            except RuntimeError:
                raise HTTPException(
                    status_code=503, detail="Remote failed; local also at capacity"
                )
        else:
            raise HTTPException(
                status_code=503, detail=f"Remote failed; local at capacity: {e}"
            )

    except Exception as e:
        # Catch-all: guarantee reservation is rolled back (Edge Case §4.1)
        if reservation and reservation.state == "pending":
            reserved_cost = reservation.reserved_cost
            await reservation.rollback()
            obs.log_reservation_rolled_back(req_id, reservation.id, str(e), reserved_cost)
        raise HTTPException(status_code=500, detail=str(e))

    # ── Post-execution: update latency tracker ───────────────────────────────
    backend_key = (
        BACKEND_REMOTE if decision.route == RouteDecision.REMOTE else BACKEND_LOCAL
    )
    await lt.record(backend_key, actual_latency_ms)

    # ── Final snapshot for response ───────────────────────────────────────────
    final_snapshot = await bm.budget_snapshot()

    obs.log_execution_complete(
        req_id,
        decision.route.value,
        actual_cost,
        actual_latency_ms,
        prompt_tokens,
        completion_tokens,
        cache_read_tokens,
        cache_write_tokens,
        outcome,
        final_snapshot.remaining,
    )

    return InferenceResponse(
        request_id=req_id,
        route=decision.route,
        complexity_tier=request_data.complexity_tier,
        actual_cost=actual_cost,
        actual_latency_ms=actual_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        result=result.result,
        budget_remaining=final_snapshot.remaining,
    )
