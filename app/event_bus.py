"""
app/event_bus.py

Minimal broadcast bus for Server-Sent Events (SSE).

Design:
  - A global module-level list holds one asyncio.Queue per connected client.
  - publish(payload) puts the payload onto every queue (fire-and-forget).
  - If a queue is full (client not consuming), the oldest item is dropped to
    prevent unbounded memory growth. Queue capacity = 64 events per client.
  - subscribe() / unsubscribe() are called by the /events SSE handler in main.py.

This module has NO dependency on any routing, budget, or latency logic.
It is a pure side-channel for observability events.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

_MAX_QUEUE_SIZE = 64  # per-client; oldest dropped if client is slow

# Module-level fan-out list — one queue per live SSE connection.
# asyncio is single-threaded so plain list mutation is safe here.
_subscribers: List[asyncio.Queue] = []


def subscribe() -> asyncio.Queue:
    """
    Create and register a new per-client event queue.
    Call this when a client connects to /events.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """
    Remove the queue when the client disconnects.
    Safe to call even if q is no longer in the list.
    """
    try:
        _subscribers.remove(q)
    except ValueError:
        pass


def publish(payload: Dict[str, Any]) -> None:
    """
    Broadcast a payload dict to every connected SSE client.

    Called from observability.py as a pure logging side-effect.
    Non-blocking: if a client's queue is full the oldest item is silently
    dropped and the new one is enqueued (ring-buffer behaviour).

    Safe to call from any coroutine — does NOT use await.
    """
    for q in _subscribers:
        if q.full():
            try:
                q.get_nowait()  # drop oldest to make room
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # extremely unlikely after the drop above; ignore
