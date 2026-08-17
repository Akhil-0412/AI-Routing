# AI Inference Router

> **Assessment submission** — LEC AI Engineering Intern, August 2026

A **concurrency-safe, asynchronous LLM inference routing gateway** built with FastAPI and Python `asyncio`. Routes requests across local and remote model backends using a two-phase atomic reservation protocol that prevents budget double-spending under concurrent load.

---

## Architecture

```
Client Request
     │
     ▼
FastAPI /inference
     │
     ▼
DecisionAgent.route()          ← advisory reads (no locks)
   ├── QueueMonitor.active_count
   ├── LatencyTracker.get_estimate()
   └── BudgetManager.liquid_budget
     │
     ▼
BudgetManager.try_reserve()    ← 🔒 budget_lock (microseconds, pure arithmetic)
     │                         ← 🔓 released
     ▼
Backend I/O                    ← fully async, NO locks held
(asyncio.sleep mock / real API)
     │
     ▼
reservation.commit/rollback()  ← 🔒 budget_lock (microseconds)
     │                         ← 🔓 released
     ▼
LatencyTracker.record()        ← 🔒 latency_lock (microseconds)
     │
     ▼
InferenceResponse → Client
```

### Routing Decision Tree

| Priority | Route | Condition |
|----------|-------|-----------|
| 1 | `LOCAL` | Local queue has capacity AND estimated latency ≤ SLA |
| 2 | `REMOTE` | Local infeasible AND remote budget available AND remote within SLA |
| 3 | `QUEUE_LOCAL` | Budget exhausted; local has a slot (degraded — SLA missed) |
| 4 | `FAIL_FAST` | All paths exhausted → 503 |

---

## The Four Weaknesses — Design Decisions

### 1. Single Global Lock → Split Per-Resource Locks

Three independent locks, one per resource domain:
- `BudgetManager._lock` — guards financial state only
- `QueueMonitor._lock` — guards slot counters only  
- `LatencyTracker._lock` — guards EMA + sliding window only

No method ever holds two locks simultaneously → **deadlock is structurally impossible**.

Advisory reads (queue depth, latency estimates) happen lock-free. The only write-under-lock is `try_reserve()`, which takes microseconds (pure arithmetic).

### 2. Reservation Expiry → TTL + Background Reaper

Every `Reservation` carries a `created_at` timestamp and `ttl_seconds`. A background `asyncio.Task` (started in FastAPI lifespan) wakes every `REAPER_INTERVAL_SECONDS`, scans for expired reservations, and rolls them back — returning budget to the liquid pool.

This is the safety net for `BaseException`, `SIGKILL`, or any failure that bypasses the `try/finally` settlement path.

### 3. State Persistence → Documented (Not Implemented)

**Explicit tradeoff**: In-process `asyncio.Lock` state is correct for a single-process demo and is the right choice for this assessment. 

A production version would need:
- **Redis `MULTI`/`EXEC`** or Lua scripts for atomic check-and-reserve
- Redis key TTL for reservation expiry (native to Redis — reaper not needed)
- `INCR`/`DECR` for distributed queue depth
- Redis time-series for latency EMA across instances

The `BudgetManager` interface (`try_reserve`, `commit`, `rollback`, `budget_snapshot`) is designed to be swappable: a `RedisBudgetManager` implementing the same methods requires no changes to `DecisionAgent` or `main.py`.

### 4. Latency Estimation → Hybrid EMA + Sliding Window

`LatencyTracker.get_estimate()` returns `max(EMA, recent_window_max)`.

- After a spike: estimate immediately reflects worst recent observation
- After the window slides past: estimate relaxes to EMA naturally
- `get_estimate()` is lock-free (CPython GIL makes float/deque reads advisory-safe)
- `record()` always acquires the write lock

---

## Quick Start

### Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

```bash
# Install uv if needed
pip install uv
```

### Run the server

```bash
cd Lec-AI-Routers

# Install dependencies
uv sync

# Start the server
uv run uvicorn app.main:app --reload
```

The server starts at `http://127.0.0.1:8000`.

### Run the concurrent demo

In a second terminal:

```bash
# Standard burst (5 concurrent requests)
uv run python load_generator.py

# Tight SLA — forces REMOTE routing to demonstrate budget enforcement
uv run python load_generator.py --tight-sla

# Failure demo — request #1 fails remotely, budget is rolled back
uv run python load_generator.py --failure-demo

# Larger burst
uv run python load_generator.py --requests 10 --tight-sla
```

### Live Observability Dashboard

With the server running, open your browser and go to:

```
http://127.0.0.1:8000/dashboard
```

The dashboard streams live events via Server-Sent Events (SSE) and updates in real time as requests flow through the router. No page refreshes needed.

**What you'll see:**
| Component | Description |
|-----------|-------------|
| **Budget Remaining** | Real-time $-value with a drain bar |
| **Latency Used** | Cumulative committed latency vs. cap |
| **Routed Local** | Count of LOCAL + QUEUE_LOCAL decisions |
| **Rejected** | Count of FAIL_FAST / rollback events |
| **Live log table** | Per-request row: tier, colour-coded route badge, cost, latency, status |

**Route badge colours:**
- 🔵 `REMOTE` — teal
- ⬜ `LOCAL` — neutral grey
- 🟡 `QUEUE_LOCAL` — amber
- 🔴 `FAIL_FAST` — red

Run the load generator while the dashboard is open to watch rows appear live:

```bash
# Open http://127.0.0.1:8000/dashboard in your browser first, then:
uv run python load_generator.py
```

The dashboard is **read-only** — it exposes no inputs, no credentials, and no controls. It is a pure observability layer that taps into the existing structured JSON logging via an in-memory SSE broadcast bus.

#### Expected output (`--tight-sla`, 5 requests)

```
==========================================
 LEC AI CONCURRENT ROUTER — DEMO
==========================================

Configuration
  Remote budget:       $0.10
  Cost/remote request: $0.05
  Concurrent requests: 5
  Latency budget/req:  100ms
  Mode:                TIGHT SLA (forces REMOTE routing)

Launching 5 concurrent requests via asyncio.gather()...

Request        Route          Cost       Latency      HTTP
──────────────────────────────────────────────────────────────
demo-001       REMOTE         $0.05      271ms        200
demo-002       REMOTE         $0.05      268ms        200
demo-003       QUEUE_LOCAL    $0.00      163ms        200
demo-004       QUEUE_LOCAL    $0.00      168ms        200
demo-005       FAIL_FAST      $0.00      4ms          503

──────────────────────────────────────────────────────────────
FINAL SYSTEM STATE
──────────────────────────────────────────────────────────────
  Remote cost spent:         $0.10000
  Remaining remote budget:   $0.00000
  Reserved (must be 0):      $0.000000
  Active local inferences:   0
  Active remote reservations:0

VERDICT: ✅ PASS
  ✓ No budget overspend
  ✓ No reservation leak (reserved == $0.000000)
  ✓ No queue depth leak (active_local == 0)
  ✓ No latency reservation leak (reserved_latency_ms == 0)
==========================================
```

### Run the test suite

```bash
# Full suite
uv run pytest -v

# Concurrency proof tests only
uv run pytest tests/test_concurrency.py -v

# Budget manager unit tests
uv run pytest tests/test_budget_manager.py -v
```

---

## Observability Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check |
| `GET /status` | Full `ResourceState` — budget, latency, queue, estimates |
| `GET /metrics` | Flat metrics dict for load_generator verdict validation |
| `POST /inference` | Route an inference request |

---

## Configuration

All tunables are environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_REMOTE_BUDGET` | `0.10` | Hard cap on remote spend ($) |
| `COST_PER_REMOTE_REQUEST` | `0.05` | Cost per remote call ($) |
| `MAX_CUMULATIVE_LATENCY_MS` | `30000` | Global latency circuit breaker (ms) |
| `LOCAL_QUEUE_CAPACITY` | `2` | Max concurrent local inferences |
| `RESERVATION_TTL_SECONDS` | `30` | Stale reservation auto-expiry |
| `REAPER_INTERVAL_SECONDS` | `5` | Reaper wake interval |

---

## What I Would Do With More Time

1. **Real Ollama integration**: The `simulate_local_inference()` and `simulate_remote_inference()` functions in `app/backends.py` are clean swap-in points. Replacing them with `httpx` calls to `http://localhost:11434/api/generate` and a real OpenAI/Anthropic client respectively requires no changes to any other module.

2. **Redis-backed state**: Implement `RedisBudgetManager` matching the same `try_reserve` / `commit` / `rollback` / `budget_snapshot` interface. Redis key TTL replaces the in-process reaper. `INCR`/`DECR` handles distributed queue depth. The `DecisionAgent` and `main.py` are unchanged.

3. **Circuit breaker for remote**: Track consecutive remote failures. After N failures, short-circuit to `QUEUE_LOCAL` for a backoff window without incurring network timeouts. Libraries like `tenacity` or a custom `asyncio.Event`-based breaker would work.

4. **Token-aware cost estimation**: Replace the flat `COST_PER_REMOTE_REQUEST` with per-request token counting (`prompt_tokens + max_tokens × price_per_token`). The reservation amount becomes the true worst-case cost, reducing average over-reservation.

5. **Prometheus metrics**: Expose EMA values, reservation counts, and routing distribution as Prometheus counters/gauges for production observability dashboards.
