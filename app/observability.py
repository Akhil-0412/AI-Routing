"""
app/observability.py

Structured JSON logging for every request lifecycle event.

Each log line is a complete, self-contained JSON object with:
  - event: what happened
  - request_id: for correlation
  - relevant numeric fields (cost, latency, queue depth, budget)

Upgrade 4: Lock interleaving trace
  lock_event() emits microsecond-precision timestamps for every lock acquire/release.
  load_generator.py reads these and prints a text-based interleaving timeline after
  each run, visually proving concurrent lock-free I/O.
"""
from __future__ import annotations

import logging
import time
from typing import List, Tuple

from pythonjsonlogger.json import JsonFormatter

import app.event_bus as _bus  # SSE side-channel — purely additive


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("lec_ai_router")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter(
            "%(asctime)s %(levelname)s %(message)s",
            rename_fields={"levelname": "level", "asctime": "timestamp"},
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _setup_logger()


def _clip(value: float) -> float:
    """Suppress floating-point noise below 1e-9."""
    return 0.0 if abs(value) < 1e-9 else float(value)


# ---------------------------------------------------------------------------
# Lock interleaving trace (Upgrade 4)
# ---------------------------------------------------------------------------

# In-process ring buffer for lock events (thread-safe via GIL; asyncio is
# single-threaded so no additional locking needed).
# Each entry: (timestamp_us, request_id, phase, action)
#   phase:  "reserve" | "settle"
#   action: "acquire" | "release"
_lock_trace: List[Tuple[float, str, str, str]] = []
_MAX_TRACE_ENTRIES = 4096


def lock_event(request_id: str, phase: str, action: str) -> None:
    """
    Record a microsecond-precision lock acquire/release event.

    Called by BudgetManager at the start and end of every critical section.
    The in-process buffer is read by load_generator.py via /trace endpoint.

    Args:
        request_id: correlates to the request that acquired/released the lock
        phase: "reserve" or "settle"
        action: "acquire" or "release"
    """
    ts_us = time.perf_counter() * 1_000_000  # microseconds since process start
    if len(_lock_trace) >= _MAX_TRACE_ENTRIES:
        _lock_trace.pop(0)  # drop oldest; keep ring semantics without deque import
    _lock_trace.append((ts_us, request_id, phase, action))

    logger.debug("lock_event", extra={
        "event": "lock_event",
        "request_id": request_id,
        "phase": phase,
        "action": action,
        "timestamp_us": ts_us,
    })


def get_lock_trace() -> List[Tuple[float, str, str, str]]:
    """Return a snapshot of recorded lock events, ordered by timestamp."""
    return list(sorted(_lock_trace, key=lambda e: e[0]))


def clear_lock_trace() -> None:
    """Clear the in-process lock event buffer."""
    _lock_trace.clear()


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

def log_request_received(request_id: str, latency_budget_ms: float, complexity_tier: str = "UNKNOWN") -> None:
    logger.info("request_received", extra={
        "event": "request_received",
        "request_id": request_id,
        "latency_budget_ms": latency_budget_ms,
        "complexity_tier": complexity_tier,
    })


def log_route_decision(
    request_id: str,
    route: str,
    reason: str,
    estimated_latency_ms: float,
    queue_depth: int,
    liquid_budget: float,
    complexity_tier: str = "UNKNOWN",
    prompt: str = "",
) -> None:
    payload = {
        "event": "route_decision",
        "request_id": request_id,
        "route": route,
        "complexity_tier": complexity_tier,
        "reason": reason,
        "estimated_latency_ms": _clip(estimated_latency_ms),
        "queue_depth": queue_depth,
        "liquid_budget": _clip(liquid_budget),
        "prompt": prompt,
    }
    logger.info("route_decision", extra=payload)
    _bus.publish(payload)  # SSE side-effect: non-blocking, dropped if no client


def log_reservation_made(
    request_id: str, reservation_id: str, reserved_cost: float, reserved_latency_ms: float
) -> None:
    logger.info("reservation_made", extra={
        "event": "reservation_made",
        "request_id": request_id,
        "reservation_id": reservation_id,
        "reserved_cost": _clip(reserved_cost),
        "reserved_latency_ms": _clip(reserved_latency_ms),
    })


def log_execution_start(request_id: str, backend: str) -> None:
    logger.info("execution_start", extra={
        "event": "execution_start",
        "request_id": request_id,
        "backend": backend,
    })


def log_execution_complete(
    request_id: str,
    backend: str,
    actual_cost: float,
    actual_latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    outcome: str,
    budget_remaining: float,
) -> None:
    payload = {
        "event": "execution_complete",
        "request_id": request_id,
        "backend": backend,
        "actual_cost": _clip(actual_cost),
        "actual_latency_ms": _clip(actual_latency_ms),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "outcome": outcome,
        "budget_remaining": _clip(budget_remaining),
    }
    logger.info("execution_complete", extra=payload)
    _bus.publish(payload)  # SSE side-effect


def log_reservation_committed(
    request_id: str, reservation_id: str, actual_cost: float, budget_remaining: float
) -> None:
    logger.info("reservation_committed", extra={
        "event": "reservation_committed",
        "request_id": request_id,
        "reservation_id": reservation_id,
        "actual_cost": _clip(actual_cost),
        "budget_remaining": _clip(budget_remaining),
    })


def log_reservation_rolled_back(
    request_id: str,
    reservation_id: str,
    reason: str,
    reserved_cost: float = 0.0,
) -> None:
    logger.warning("reservation_rolled_back", extra={
        "event": "reservation_rolled_back",
        "request_id": request_id,
        "reservation_id": reservation_id,
        "reason": reason,
        "reserved_cost_returned": _clip(reserved_cost),
    })


def log_fail_fast(request_id: str, reason: str, status_code: int = 503) -> None:
    payload = {
        "event": "fail_fast",
        "request_id": request_id,
        "reason": reason,
        "status_code": status_code,
    }
    logger.warning("fail_fast", extra=payload)
    _bus.publish(payload)  # SSE side-effect


def log_remote_failure(request_id: str, error: str) -> None:
    logger.error("remote_failure", extra={
        "event": "remote_failure",
        "request_id": request_id,
        "error": error,
    })


def log_ollama_health(reachable: bool, url: str) -> None:
    if reachable:
        logger.info("ollama_health_ok", extra={
            "event": "ollama_health_ok",
            "url": url,
        })
    else:
        logger.warning("ollama_health_warn", extra={
            "event": "ollama_health_warn",
            "url": url,
            "detail": "Ollama unreachable at startup — local backend will fail until it is available",
        })


def log_rollback_event(
    request_id: str,
    reservation_id: str,
    reserved_cost: float,
    reason: str,
) -> None:
    """Explicit rollback marker for load_generator.py CLI output (Upgrade 5)."""
    payload = {
        "event": "rollback_event",
        "request_id": request_id,
        "reservation_id": reservation_id,
        "reserved_cost_returned": _clip(reserved_cost),
        "reason": reason,
    }
    logger.warning("rollback_event", extra=payload)
    _bus.publish(payload)  # SSE side-effect


def log_run_start(run_id: str) -> None:
    """Marker for the start of a test run/batch."""
    payload = {
        "event": "run_start",
        "run_id": run_id,
    }
    logger.info("run_start", extra=payload)
    _bus.publish(payload)


def log_run_complete(run_id: str, verdict: str, metrics: dict, routes: dict) -> None:
    """Marker for the end of a test run/batch."""
    payload = {
        "event": "run_complete",
        "run_id": run_id,
        "verdict": verdict,
        "metrics": metrics,
        "routes": routes,
    }
    logger.info("run_complete", extra=payload)
    _bus.publish(payload)
