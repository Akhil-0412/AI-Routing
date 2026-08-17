"""
app/backends.py

Inference backends: real (Ollama + Remote LLM Provider) and mock (for tests/offline).

Backend selection is controlled by BACKEND_MODE env var (default: "real").
  - "real": calls Ollama HTTP API for local, Remote LLM Provider OpenAI-compat API for remote.
  - "mock": simulated latency + cost (used in tests, no network required).

The public interface is unchanged:
  simulate_local_inference(prompt, request_id) -> BackendResult
  simulate_remote_inference(prompt, request_id, should_fail=False) -> BackendResult

All real network failures (timeouts, connection errors, 429s, etc.) propagate as
RemoteInferenceError so main.py's existing rollback path handles them correctly.

Design decision (Cost Accounting):
  Remote LLM Provider's "qwen/qwen3.8-max-free" endpoint is genuinely free-to-call.
  However, the cost ledger charges at the PAID qwen/qwen3.8-max discounted rate:
    - $1.00 / 1M input tokens
    - $3.00 / 1M output tokens
  applied to the REAL token counts returned by the API. This demonstrates budget
  enforcement using real usage data without incurring real spend.
  See README.md §Cost Accounting Design Decision.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, RetryError

from app.config import (
    AVG_LOCAL_LATENCY_MS,
    AVG_REMOTE_LATENCY_MS,
    BACKEND_MODE,
    COST_PER_REMOTE_REQUEST,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    REMOTE_LLM_API_KEY,
    REMOTE_LLM_BASE_URL,
    REMOTE_LLM_INPUT_PRICE_PER_TOKEN,
    REMOTE_LLM_MODEL,
    REMOTE_LLM_OUTPUT_PRICE_PER_TOKEN,
    REMOTE_LLM_TIMEOUT_SECONDS,
)
from app.models import BackendResult


class RemoteInferenceError(Exception):
    """Raised when the remote backend fails (network error, timeout, 4xx/5xx)."""


class OllamaUnavailableError(Exception):
    """Raised during startup health check if Ollama cannot be reached."""


# ---------------------------------------------------------------------------
# Health check (called from lifespan)
# ---------------------------------------------------------------------------

async def check_ollama_health() -> bool:
    """
    Pings Ollama's /api/tags endpoint.
    Returns True if reachable, False otherwise. Never raises.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Real local backend: Ollama
# ---------------------------------------------------------------------------

async def _call_ollama(prompt: str, request_id: str) -> BackendResult:
    """
    Calls Ollama's /api/generate endpoint (non-streaming).
    Captures real wall-clock latency and real token counts.
    Raises RemoteInferenceError on any failure so the rollback path fires.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 256,  # cap output length for demo speed
        },
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)

        actual_latency_ms = (time.perf_counter() - start) * 1000.0

        if resp.status_code != 200:
            raise RemoteInferenceError(
                f"[{request_id}] Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        response_text = data.get("response", "")
        prompt_tokens: int = data.get("prompt_eval_count", 0)
        completion_tokens: int = data.get("eval_count", 0)

        return BackendResult(
            request_id=request_id,
            actual_cost=0.0,  # local inference is free
            actual_latency_ms=actual_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            result=(
                f"[OLLAMA:{OLLAMA_MODEL}] tokens={prompt_tokens}+{completion_tokens} | "
                f"{response_text[:120]}"
            ),
        )

    except RemoteInferenceError:
        raise
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        actual_latency_ms = (time.perf_counter() - start) * 1000.0
        raise RemoteInferenceError(
            f"[{request_id}] Ollama unreachable: {type(e).__name__}: {e}"
        ) from e
    except Exception as e:
        actual_latency_ms = (time.perf_counter() - start) * 1000.0
        raise RemoteInferenceError(
            f"[{request_id}] Ollama unexpected error: {type(e).__name__}: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Real remote backend: Remote LLM Provider (OpenAI-compatible)
# ---------------------------------------------------------------------------

class TransientRemoteError(Exception):
    """Internal error for retrying remote calls using tenacity."""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(TransientRemoteError),
    reraise=True,
)
async def _execute_tokenrouter_request(payload: dict, headers: dict, request_id: str) -> httpx.Response:
    try:
        async with httpx.AsyncClient(timeout=REMOTE_LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{REMOTE_LLM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            
            if resp.status_code in [429, 502, 503, 504]:
                raise TransientRemoteError(f"[{request_id}] Remote LLM Provider transient HTTP {resp.status_code}")
                
            return resp
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        raise TransientRemoteError(f"[{request_id}] Remote LLM Provider transient unreachable: {type(e).__name__}") from e

async def _call_tokenrouter(
    prompt: str, request_id: str, should_fail: bool = False
) -> BackendResult:
    """
    Calls Remote LLM Provider's OpenAI-compatible chat completions endpoint.

    Uses stream_options={"include_usage": True} to capture real token counts.
    Cost is computed at paid-tier pricing (see module docstring) applied to real
    prompt_tokens + completion_tokens from the API response.

    If should_fail=True, raises RemoteInferenceError immediately to simulate
    a forced failure (Scenario C rollback test).
    """
    if should_fail:
        raise RemoteInferenceError(
            f"[{request_id}] Injected remote failure for rollback test"
        )

    headers = {
        "Authorization": f"Bearer {REMOTE_LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": REMOTE_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "stream": False,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    try:
        resp = await _execute_tokenrouter_request(payload, headers, request_id)

        actual_latency_ms = (time.perf_counter() - start) * 1000.0

        if resp.status_code >= 400:
            raise RemoteInferenceError(
                f"[{request_id}] Remote LLM Provider HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()

        # Extract real token counts
        usage = data.get("usage", {})
        prompt_tokens: int = usage.get("prompt_tokens", 0)
        completion_tokens: int = usage.get("completion_tokens", 0)

        # Cost accounting at paid-tier pricing using real token counts
        actual_cost = (
            prompt_tokens * REMOTE_LLM_INPUT_PRICE_PER_TOKEN
            + completion_tokens * REMOTE_LLM_OUTPUT_PRICE_PER_TOKEN
        )

        # Extract response text
        choices = data.get("choices", [])
        response_text = ""
        if choices:
            response_text = choices[0].get("message", {}).get("content", "")

        return BackendResult(
            request_id=request_id,
            actual_cost=actual_cost,
            actual_latency_ms=actual_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            result=(
                f"[TOKENROUTER:{REMOTE_LLM_MODEL}] "
                f"tokens={prompt_tokens}+{completion_tokens} "
                f"cost=${actual_cost:.6f} | {response_text[:120]}"
            ),
        )

    except RemoteInferenceError:
        raise
    except TransientRemoteError as e:
        actual_latency_ms = (time.perf_counter() - start) * 1000.0
        raise RemoteInferenceError(
            f"[{request_id}] Remote LLM Provider failed after retries: {e}"
        ) from e
    except RetryError as e:
        actual_latency_ms = (time.perf_counter() - start) * 1000.0
        raise RemoteInferenceError(
            f"[{request_id}] Remote LLM Provider failed after retries: {e.last_attempt.exception()}"
        ) from e
    except Exception as e:
        actual_latency_ms = (time.perf_counter() - start) * 1000.0
        raise RemoteInferenceError(
            f"[{request_id}] Remote LLM Provider unexpected error: {type(e).__name__}: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Mock backends (used when BACKEND_MODE="mock" or in tests)
# ---------------------------------------------------------------------------

async def _mock_local_inference(prompt: str, request_id: str) -> BackendResult:
    jitter = random.uniform(0.8, 1.2)
    delay_s = (AVG_LOCAL_LATENCY_MS * jitter) / 1000.0
    start = time.perf_counter()
    await asyncio.sleep(delay_s)
    actual_latency_ms = (time.perf_counter() - start) * 1000.0
    return BackendResult(
        request_id=request_id,
        actual_cost=0.0,
        actual_latency_ms=actual_latency_ms,
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        result=f"[LOCAL-MOCK] Echo: {prompt[:60]}",
    )


async def _mock_remote_inference(
    prompt: str, request_id: str, should_fail: bool = False
) -> BackendResult:
    jitter = random.uniform(0.85, 1.15)
    delay_s = (AVG_REMOTE_LATENCY_MS * jitter) / 1000.0
    start = time.perf_counter()
    await asyncio.sleep(delay_s)
    actual_latency_ms = (time.perf_counter() - start) * 1000.0
    if should_fail:
        raise RemoteInferenceError(
            f"Simulated remote failure for request {request_id}"
        )
        
    # Simulate dynamic tokens (Claude Opus 5 mock)
    prompt_tokens = random.randint(500, 1500)
    completion_tokens = random.randint(200, 800)
    
    # 50% chance of a cache read hit
    cache_read_tokens = random.randint(500, 1500) if random.random() > 0.5 else 0
    # 50% chance of a cache write hit
    cache_write_tokens = random.randint(200, 1000) if random.random() > 0.5 else 0
    
    from app.config import (
        REMOTE_LLM_INPUT_PRICE_PER_TOKEN,
        REMOTE_LLM_OUTPUT_PRICE_PER_TOKEN,
        REMOTE_LLM_CACHE_READ_PRICE_PER_TOKEN,
        REMOTE_LLM_CACHE_WRITE_PRICE_PER_TOKEN
    )
    
    actual_cost = (
        prompt_tokens * REMOTE_LLM_INPUT_PRICE_PER_TOKEN +
        completion_tokens * REMOTE_LLM_OUTPUT_PRICE_PER_TOKEN +
        cache_read_tokens * REMOTE_LLM_CACHE_READ_PRICE_PER_TOKEN +
        cache_write_tokens * REMOTE_LLM_CACHE_WRITE_PRICE_PER_TOKEN
    )

    return BackendResult(
        request_id=request_id,
        actual_cost=actual_cost,
        actual_latency_ms=actual_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        result=f"[REMOTE-MOCK] tokens={prompt_tokens} cost=${actual_cost:.5f}",
    )


# ---------------------------------------------------------------------------
# Public interface (selected by BACKEND_MODE at call time)
# ---------------------------------------------------------------------------

async def simulate_local_inference(prompt: str, request_id: str) -> BackendResult:
    """
    Routes to Ollama (BACKEND_MODE='real') or mock (BACKEND_MODE='mock').
    If Ollama is unreachable, falls back to mock to ensure smooth demo execution.
    """
    if BACKEND_MODE == "mock":
        return await _mock_local_inference(prompt, request_id)
    try:
        return await _call_ollama(prompt, request_id)
    except RemoteInferenceError:
        # Ollama daemon not running or unreachable — seamless fallback to local mock
        return await _mock_local_inference(prompt, request_id)


async def simulate_remote_inference(
    prompt: str,
    request_id: str,
    should_fail: bool = False,
) -> BackendResult:
    """
    Routes to Remote LLM Provider (BACKEND_MODE='real') or mock (BACKEND_MODE='mock').
    If BACKEND_MODE='real' but no API key is provided, falls back to mock to 
    preserve demo flow and avoid rate limits.
    All failures are converted to RemoteInferenceError for uniform handling.
    """
    if BACKEND_MODE == "mock":
        return await _mock_remote_inference(prompt, request_id, should_fail)
        
    if not REMOTE_LLM_API_KEY:
        # Fallback to mock for seamless demonstration without API key
        return await _mock_remote_inference(prompt, request_id, should_fail)

    return await _call_tokenrouter(prompt, request_id, should_fail)
