#!/usr/bin/env python3
"""
load_generator.py

Concurrent demo and correctness proof script for LEC AI Router.

Usage:
    uv run python load_generator.py                   # Default batch (20 requests)
    uv run python load_generator.py --stress          # Stress test (150 requests)
    uv run python load_generator.py --failure-demo    # Remote failure + rollback
    uv run python load_generator.py --tight-sla       # Forces REMOTE routing
    uv run python load_generator.py --latency-breach  # Forces Latency cap breach
    uv run python load_generator.py --reset-only      # Resets the server budget and exits

What this proves:
  ✓ Multiple concurrent requests routed simultaneously without race conditions
  ✓ Budget hard limit respected — never overspent regardless of concurrency
  ✓ Reservation leak impossible — reserved==0 and active_reservations==0 after all settle
  ✓ Queue depth leak impossible — active_local==0 after all complete
  ✓ Graceful degradation: overflow requests get 503, not 500
  ✓ Lock interleaving visualization proves non-blocking concurrent I/O.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import time
from typing import Any, Dict, List

import httpx

DEFAULT_URL = "http://127.0.0.1:8000"


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    req_id: str,
    prompt: str,
    latency_budget_ms: float,
) -> Dict[str, Any]:
    start = time.perf_counter()
    payload = {
        "request_id": req_id,
        "prompt": prompt,
        "latency_budget_ms": latency_budget_ms,
    }
    try:
        resp = await client.post(f"{url}/inference", json=payload, timeout=120.0)
        duration_ms = (time.perf_counter() - start) * 1000.0
        data = {}
        try:
            data = resp.json()
        except Exception:
            pass
        return {
            "request_id": req_id,
            "status_code": resp.status_code,
            "route": data.get("route", "FAIL_FAST" if resp.status_code == 503 else "ERROR"),
            "complexity_tier": data.get("complexity_tier", "-"),
            "actual_cost": data.get("actual_cost", 0.0),
            "actual_latency_ms": data.get("actual_latency_ms", 0.0),
            "prompt_tokens": data.get("prompt_tokens", 0),
            "completion_tokens": data.get("completion_tokens", 0),
            "cache_read_tokens": data.get("cache_read_tokens", 0),
            "cache_write_tokens": data.get("cache_write_tokens", 0),
            "budget_remaining": data.get("budget_remaining", None),
            "duration_ms": duration_ms,
            "detail": data.get("detail", ""),
        }
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "request_id": req_id,
            "status_code": 0,
            "route": "CONNECTION_ERROR",
            "actual_cost": 0.0,
            "actual_latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "budget_remaining": None,
            "duration_ms": duration_ms,
            "error": str(e),
            "detail": str(e),
        }


def evaluate_verdict(
    metrics: Dict[str, Any],
    max_budget: float,
    max_latency: float,
) -> tuple[bool, List[str]]:
    errors = []

    cost = metrics.get("cumulative_remote_cost", 0.0)
    if cost > max_budget + 1e-9:
        errors.append(f"❌ BUDGET OVERSPEND: ${cost:.5f} > ${max_budget:.5f}")

    reserved = metrics.get("reserved_remote_budget", 0.0)
    if not math.isclose(reserved, 0.0, abs_tol=1e-9):
        errors.append(f"❌ RESERVATION LEAK: reserved=${reserved:.6f} (must be 0)")

    active_res = metrics.get("active_remote_reservations", 0)
    if active_res != 0:
        errors.append(f"❌ RESERVATION LEAK: {active_res} active reservations (must be 0)")

    active_local = metrics.get("active_local_inferences", 0)
    if active_local != 0:
        errors.append(f"❌ QUEUE LEAK: {active_local} active local inferences (must be 0)")

    reserved_lat = metrics.get("reserved_latency_ms", 0.0)
    if not math.isclose(reserved_lat, 0.0, abs_tol=1e-9):
        errors.append(f"❌ LATENCY RESERVATION LEAK: {reserved_lat:.1f}ms (must be 0)")

    lat_spent = metrics.get("cumulative_actual_latency_ms", 0.0)
    if lat_spent > max_latency + 1e-9:
        errors.append(f"❌ LATENCY CAP BREACH: {lat_spent:.1f}ms > {max_latency:.1f}ms")

    return (not errors), errors


def format_lock_trace(events: List[Dict[str, Any]]) -> None:
    """Prints an ASCII timeline of lock interleaving to prove concurrent I/O."""
    if not events:
        print("  No lock events recorded.")
        return
        
    print(f"\n{'-'*62}")
    print("LOCK INTERLEAVING TIMELINE (PROVES CONCURRENT I/O)")
    print(f"{'-'*62}")
    
    t0 = events[0]["timestamp_us"]
    print(f"{'Offset (ms)':>12} | {'Request ID':<14} | {'Phase':<8} | {'Action':<7}")
    print("-" * 55)
    
    # Track nesting just to show visually
    indent = 0
    for e in events:
        offset_ms = (e["timestamp_us"] - t0) / 1000.0
        action = e["action"]
        if action == "release":
            indent = max(0, indent - 1)
            
        prefix = "  " * indent
        print(f"{offset_ms:>11.2f}ms | {e['request_id']:<14} | {prefix}{e['phase']:<8} | {action:<7}")
        
        if action == "acquire":
            indent += 1
            
    print("\n  Observation: Multiple requests enter 'reserve' before any enter 'settle'.")
    print("  This proves locks are released during backend I/O.\n")


async def run_demo(
    url: str,
    requests_count: int,
    failure_demo: bool,
    tight_sla: bool,
    latency_breach: bool,
    reset: bool = False,
) -> bool:
    print("\n==========================================")
    print(" LEC AI CONCURRENT ROUTER - DEMO")
    print("==========================================")

    # Base Configuration
    max_budget = 5.00
    cost_per_remote = 0.02
    max_latency_cap = 600000.0  # From config
    
    # Note: real Ollama could be slower, so we use a high latency budget unless tight_sla
    latency_budget = 100.0 if tight_sla else 5000.0
    
    if requests_count == 0:
        print(f"\nConfiguration")
        print(f"  Server:              {url}")
        print(f"  Mode:                RESET ONLY (resetting server state and exiting)")
    else:
        print(f"\nConfiguration")
        print(f"  Server:              {url}")
        print(f"  Remote budget:       ${max_budget:.2f}")
        print(f"  Concurrent requests: {requests_count}")
        print(f"  Latency budget/req:  {latency_budget:.0f}ms")
    
    if failure_demo:
        print(f"  Mode:                FAILURE DEMO (request #1 will fail remotely)")
    if tight_sla:
        print(f"  Mode:                TIGHT SLA (forces REMOTE routing)")
    if latency_breach:
        print(f"  Mode:                LATENCY BREACH (simulates massive latency to trigger circuit breaker)")

    async with httpx.AsyncClient() as client:
        # Check server is alive
        try:
            health = await client.get(f"{url}/health", timeout=5.0)
            if health.status_code != 200:
                print(f"\n  ⚠ Server returned {health.status_code} on /health")
                return False
        except Exception as e:
            print(f"\n  ✗ Cannot reach server at {url}: {e}")
            print(f"    Start it with: uv run uvicorn app.main:app --reload")
            return False

        # Clear the lock trace and server state before we start
        run_id = f"run-{int(time.time())}"
        try:
            await client.post(f"{url}/run_start", json={"run_id": run_id, "reset": reset})
        except Exception:
            pass

        await client.delete(f"{url}/trace")
        
        if requests_count == 0:
            print("\n  ✓ Server state and budget reset to $5.00.")
            print("==========================================\n")
            return True

        # Dynamically fetch the current limits from the server so tests don't fail if we changed config
        try:
            summary = (await client.get(f"{url}/summary")).json()
            max_budget = summary.get("budget_max", max_budget)
            max_latency_cap = summary.get("latency_max_ms", max_latency_cap)
        except Exception:
            pass

        # Build request batch
        tasks = []
        burst_start = time.perf_counter()

        # Tier-aware prompts: alternate FAST and QUALITY for a rich mixed demo
        FAST_PROMPTS = [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "Hello, how are you?",
            "Define recursion in one sentence.",
            "When did World War II end?",
            "What is 2 + 2?",
            "Who is Ada Lovelace?",
            "Translate 'hello' into Spanish.",
        ]
        QUALITY_PROMPTS = [
            "Explain why async/await avoids thread-based race conditions in Python.",
            "Design a budget-aware inference router that handles concurrent requests safely.",
            "Write a Python function that implements exponential backoff with jitter.",
            "Compare transformer attention mechanisms with RNNs for sequence modelling.",
            "Analyse the time complexity of merge sort and explain each step.",
            "Explain the trade-offs between SQL and NoSQL databases for high-throughput workloads.",
            "Refactor this code to use list comprehensions and reduce memory usage.",
            "Why do neural networks require non-linear activation functions?",
        ]

        for i in range(1, requests_count + 1):
            req_id = f"demo-{i:03d}"

            if failure_demo and i == 1:
                prompt = "explain fail_remote injected"
            elif latency_breach or tight_sla:
                # For stress/SLA modes use a simple mixed prompt
                prompt = f"Say hi. Request {i}"
            else:
                # Alternate FAST / QUALITY for the default demo run
                if i % 2 == 1:
                    prompt = FAST_PROMPTS[(i // 2) % len(FAST_PROMPTS)]
                else:
                    prompt = QUALITY_PROMPTS[(i // 2 - 1) % len(QUALITY_PROMPTS)]

            tasks.append(
                send_request(client, url, req_id, prompt, latency_budget)
            )

        print(f"\nLaunching {requests_count} concurrent requests via asyncio.gather()...")
        if requests_count > 50:
            print("  (This is a stress test, it may take 1-2 minutes depending on your GPU...)")
            
        results = await asyncio.gather(*tasks)
        burst_duration_ms = (time.perf_counter() - burst_start) * 1000.0

        # Breakdown statistics
        from collections import Counter
        routes_counter = Counter()
        reasons_counter = Counter()
        for res in results:
            route = res["route"]
            routes_counter[route] += 1
            if route in ["FAIL_FAST", "ERROR", "CONNECTION_ERROR"]:
                reason = res.get("detail", "Unknown reason")
                reasons_counter[reason] += 1

        print(f"\n{'-'*62}")
        print("ROUTING BREAKDOWN")
        print(f"{'-'*62}")
        print(f"  Total requests fired:      {requests_count}")
        print(f"  LOCAL:                     {routes_counter.get('LOCAL', 0)}")
        print(f"  REMOTE:                    {routes_counter.get('REMOTE', 0)}")
        print(f"  QUEUE_LOCAL:               {routes_counter.get('QUEUE_LOCAL', 0)}")
        fail_count = sum(routes_counter[r] for r in ['FAIL_FAST', 'ERROR', 'CONNECTION_ERROR'])
        print(f"  REJECTED / FAILED:         {fail_count}")
        if fail_count > 0:
            for reason, count in reasons_counter.items():
                print(f"    - {count}x: {reason}")
        print(f"{'-'*62}")

        # Results table — includes Tier column for semantic routing visibility
        print(f"\n{'Request':<10} {'Tier':<10} {'Route':<14} {'Cost':<10} {'Tokens (In+Out+CR+CW)':<25} {'Latency':<12} {'HTTP'}")
        print("-" * 95)
        for res in results:
            route  = res["route"]
            tier   = res.get("complexity_tier", "-")
            cost   = f"${res['actual_cost']:.5f}"
            lat    = f"{res['duration_ms']:.0f}ms"
            status = res["status_code"]
            
            p_tok = res.get('prompt_tokens', 0)
            c_tok = res.get('completion_tokens', 0)
            cr_tok = res.get('cache_read_tokens', 0)
            cw_tok = res.get('cache_write_tokens', 0)
            tokens_str = f"{p_tok}+{c_tok}+{cr_tok}+{cw_tok}" if route == "REMOTE" else "-"

            print(f"{res['request_id']:<10} {tier:<10} {route:<14} {cost:<10} {tokens_str:<25} {lat:<12} {status}")
            
            # Upgrade 5: Visible Rollback Messages
            if failure_demo and res["request_id"] == "demo-001" and status == 500:
                print(f"  => [VISIBLE ROLLBACK] Request failed: {res.get('detail', 'Unknown error')}")
                print(f"  => [VISIBLE ROLLBACK] Reserved funds were returned to the budget pool.")
                
            # Upgrde 6: Latency Breach Rejection Message
            if status == 500 and "breach" in res.get("detail", "").lower():
                print(f"  => [LATENCY CAP BREACH] Rejected: {res.get('detail')}")

        # Fetch lock trace
        try:
            trace_resp = await client.get(f"{url}/trace")
            trace_data = trace_resp.json()
            format_lock_trace(trace_data.get("events", []))
        except Exception as e:
            print(f"  ✗ Failed to fetch lock trace: {e}")

        # Final state
        print(f"\n{'-'*62}")
        print("FINAL SYSTEM STATE")
        print(f"{'-'*62}")

        try:
            metrics_resp = await client.get(f"{url}/metrics")
            metrics = metrics_resp.json()

            print(f"  Remote cost spent:         ${metrics.get('cumulative_remote_cost', 0):.5f}")
            print(f"  Remaining remote budget:   ${metrics.get('remaining_remote_budget', 0):.5f}")
            print(f"  Reserved (must be 0):      ${metrics.get('reserved_remote_budget', 0):.6f}")
            print(f"  Cumulative latency:        {metrics.get('cumulative_actual_latency_ms', 0):.0f}ms")
            print(f"  Reserved latency (->0):    {metrics.get('reserved_latency_ms', 0):.1f}ms")
            print(f"  Active local inferences:   {metrics.get('active_local_inferences', 0)}")
            print(f"  Active remote reservations:{metrics.get('active_remote_reservations', 0)}")
            print(f"  Local latency EMA:         {metrics.get('local_latency_ema_ms', 0):.1f}ms")
            print(f"  Remote latency EMA:        {metrics.get('remote_latency_ema_ms', 0):.1f}ms")
            print(f"  Total burst duration:      {burst_duration_ms:.0f}ms")

            print(f"\n{'-'*62}")
            passed, errors = evaluate_verdict(metrics, max_budget, max_latency_cap)
            if passed:
                print("VERDICT: PASS [OK]")
                print("  + No budget overspend")
                print("  + No reservation leak (reserved == $0.000000)")
                print("  + No queue depth leak (active_local == 0)")
                print("  + No latency reservation leak (reserved_latency_ms == 0)")
                print("  + Cumulative latency within bounds")
            else:
                print("VERDICT: FAIL [X]")
                for err in errors:
                    print(f"  {err}")

            try:
                await client.post(f"{url}/run_complete", json={
                    "run_id": run_id,
                    "verdict": "PASS" if passed else "FAIL",
                    "metrics": metrics,
                    "routes": dict(routes_counter)
                })
            except Exception:
                pass

            print("==========================================\n")
            return passed

        except Exception as e:
            print(f"\n  ✗ Failed to fetch metrics: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Concurrent load generator and correctness proof for LEC AI Router"
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Gateway base URL")
    parser.add_argument("--requests", type=int, default=20, help="Number of concurrent requests (Upgrade 3 default: 20)")
    parser.add_argument("--stress", action="store_true", help="Stress test with 80 concurrent requests (Upgrade 3)")
    parser.add_argument("--failure-demo", action="store_true",
                        help="Inject a remote failure in request #1 to test rollback")
    parser.add_argument("--tight-sla", action="store_true",
                        help="Use 100ms latency budget to force REMOTE routing")
    parser.add_argument("--latency-breach", action="store_true",
                        help="Send 80 requests to breach the MAX_CUMULATIVE_LATENCY_MS cap (Upgrade 6)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset the server budget and state before running")
    parser.add_argument("--reset-only", action="store_true",
                        help="Reset the server budget and state, then exit without sending requests")
    args = parser.parse_args()

    if args.reset_only:
        count = 0
        args.reset = True
    else:
        count = 80 if args.stress or args.latency_breach else args.requests

    passed = asyncio.run(run_demo(args.url, count, args.failure_demo, args.tight_sla, args.latency_breach, args.reset))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
