"""
app/models.py

Pydantic schemas, enums, and dataclasses shared across the application.
No business logic here — pure data contracts.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.complexity_classifier import ComplexityTier


# ─── Routing Decision ─────────────────────────────────────────────────────────

class RouteDecision(str, Enum):
    LOCAL = "LOCAL"            # Route to local model, within capacity + SLA
    REMOTE = "REMOTE"          # Escalate to remote API, budget reserved
    QUEUE_LOCAL = "QUEUE_LOCAL"  # Budget exhausted; degraded local fallback
    FAIL_FAST = "FAIL_FAST"    # All paths exhausted; return 503


# ─── HTTP Schemas ─────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    """Incoming inference request from a client."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = Field(..., min_length=1)
    latency_budget_ms: float = Field(..., gt=0.0,
        description="Per-request latency SLA in milliseconds")
    # Auto-classified from prompt if not provided by the client
    complexity_tier: Optional[ComplexityTier] = Field(
        default=None,
        description="Routing preference: FAST (prefer local) or QUALITY (prefer remote). Auto-classified if omitted."
    )


class InferenceResponse(BaseModel):
    """Response returned to the client after routing and execution."""
    request_id: str
    route: RouteDecision
    complexity_tier: ComplexityTier   # The tier assigned (auto or client-supplied)
    actual_cost: float
    actual_latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    result: str
    budget_remaining: float


# ─── Internal Result ──────────────────────────────────────────────────────────

class BackendResult(BaseModel):
    """Result produced by a backend (local or remote) after inference."""
    request_id: str
    actual_cost: float
    actual_latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    result: str


# ─── State Snapshots ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BudgetSnapshot:
    """Immutable point-in-time view of budget state. Returned outside the lock."""
    cumulative_cost: float
    reserved: float
    remaining: float       # liquid: usable for new reservations
    cumulative_latency_ms: float
    reserved_latency_ms: float


class ResourceState(BaseModel):
    """Full resource state for observability endpoints."""
    # Financial
    remote_budget_max: float
    remote_budget_spent: float
    remote_budget_reserved: float
    remote_budget_remaining: float

    # Latency
    latency_max_ms: float
    latency_committed_ms: float
    latency_reserved_ms: float
    latency_remaining_ms: float

    # Local queue
    local_capacity_max: int
    local_capacity_active: int
    local_capacity_available: int

    # Latency estimates (from LatencyTracker)
    estimated_local_latency_ms: Optional[float] = None
    estimated_remote_latency_ms: Optional[float] = None

    # Active reservations (for leak detection)
    active_reservations: int = 0


# ─── Run Tracking (Dashboard Catalogue) ───────────────────────────────────────

class RunStartPayload(BaseModel):
    run_id: str
    reset: bool = True

class RunCompletePayload(BaseModel):
    run_id: str
    verdict: str
    metrics: dict
    routes: dict
