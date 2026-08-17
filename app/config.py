"""
app/config.py

All tunables, read from environment variables with production-safe defaults.
Never import from here inside a lock -- reads are lock-free, values are immutable
after startup.
"""
import os


# --- Remote Budget --------------------------------------------------------

# Hard cap on cumulative remote spend across the lifetime of the process.
# Production default: $5.00 per server session — gives ~1,000 typical remote calls
# before the circuit breaker engages. Tune per your billing limits.
MAX_REMOTE_BUDGET: float = float(os.getenv("MAX_REMOTE_BUDGET", "5.00"))

# Worst-case cost reserved per remote call (admission-time accounting).
# Assumes up to ~5000 input + 2000 output tokens at Claude Opus 5 pricing.
# $10.00/1M input + $50.00/1M output → worst-case ≈ $0.15.
# Real cost is settled from exact dynamic token counts.
COST_PER_REMOTE_REQUEST: float = float(os.getenv("COST_PER_REMOTE_REQUEST", "0.15"))


# --- Latency Limits -------------------------------------------------------

# Hard cap on the sum of all admitted request latencies.
# Production default: 600,000ms = 10 minutes of cumulative in-flight work.
# This prevents unbounded queueing under sustained overload. A real system
# would use a sliding-window variant; this lifetime total is intentionally
# large so the circuit breaker only fires under genuine sustained overload,
# not after a short burst.
MAX_CUMULATIVE_LATENCY_MS: float = float(os.getenv("MAX_CUMULATIVE_LATENCY_MS", "600000.0"))

# EMA seed latencies — realistic priors before any observations are recorded.
# LOCAL: quantized 7B model on a mid-range GPU (RTX 3090 / A10G) — ~250ms TTFT.
# REMOTE: GPT-4o / Claude Sonnet measured median first-token + generation ≈ 800ms.
AVG_LOCAL_LATENCY_MS: float = float(os.getenv("AVG_LOCAL_LATENCY_MS", "250.0"))
AVG_REMOTE_LATENCY_MS: float = float(os.getenv("AVG_REMOTE_LATENCY_MS", "800.0"))


# --- Queue ----------------------------------------------------------------

# Maximum number of local inference requests allowed to execute simultaneously.
# Production default: 8 concurrent slots — appropriate for a single GPU node
# or a 16-core CPU serving a quantized model. Scale with your hardware.
LOCAL_QUEUE_CAPACITY: int = int(os.getenv("LOCAL_QUEUE_CAPACITY", "8"))


# --- EMA + Latency Tracker ------------------------------------------------

# Smoothing factor for the Exponential Moving Average (0 < alpha <= 1).
EMA_ALPHA: float = float(os.getenv("EMA_ALPHA", "0.25"))

# Number of recent latency observations kept per backend for spike detection.
LATENCY_WINDOW_SIZE: int = int(os.getenv("LATENCY_WINDOW_SIZE", "5"))


# --- Reservation TTL ------------------------------------------------------

# Maximum age (seconds) of a pending reservation before the reaper rolls it back.
# Production default: 45s — covers a slow remote API under retry (3 attempts × 10s)
# without holding budget hostage for too long.
RESERVATION_TTL_SECONDS: float = float(os.getenv("RESERVATION_TTL_SECONDS", "45.0"))

# How often the background reaper task wakes up to scan for expired reservations.
REAPER_INTERVAL_SECONDS: float = float(os.getenv("REAPER_INTERVAL_SECONDS", "5.0"))


# --- Ollama (Local Backend) -----------------------------------------------

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "my-nemotron:latest")
OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120.0"))


# --- Remote LLM Provider (Remote Backend) -----------------------------------------

REMOTE_LLM_API_KEY: str = os.getenv("REMOTE_LLM_API_KEY", "")
REMOTE_LLM_BASE_URL: str = os.getenv("REMOTE_LLM_BASE_URL", "https://api.tokenrouter.com/v1")
REMOTE_LLM_MODEL: str = os.getenv("REMOTE_LLM_MODEL", "qwen/qwen3.8-max-free")
REMOTE_LLM_TIMEOUT_SECONDS: float = float(os.getenv("REMOTE_LLM_TIMEOUT_SECONDS", "30.0"))

# Cost accounting: realistic pricing simulating Claude Opus 5.
# The exact token breakdown is dynamically settled post-execution.
REMOTE_LLM_INPUT_PRICE_PER_TOKEN: float = float(
    os.getenv("REMOTE_LLM_INPUT_PRICE_PER_TOKEN", str(10.00 / 1_000_000))
)  # $10.00 / 1M input tokens
REMOTE_LLM_OUTPUT_PRICE_PER_TOKEN: float = float(
    os.getenv("REMOTE_LLM_OUTPUT_PRICE_PER_TOKEN", str(50.00 / 1_000_000))
)  # $50.00 / 1M output tokens
REMOTE_LLM_CACHE_READ_PRICE_PER_TOKEN: float = float(
    os.getenv("REMOTE_LLM_CACHE_READ_PRICE_PER_TOKEN", str(1.00 / 1_000_000))
)  # $1.00 / 1M cache read tokens
REMOTE_LLM_CACHE_WRITE_PRICE_PER_TOKEN: float = float(
    os.getenv("REMOTE_LLM_CACHE_WRITE_PRICE_PER_TOKEN", str(12.50 / 1_000_000))
)  # $12.50 / 1M cache write tokens


# --- Backend Selection ----------------------------------------------------

# "mock" uses simulated backends (for tests / offline dev).
# "real" uses Ollama (local) + Remote LLM Provider (remote).
BACKEND_MODE: str = os.getenv("BACKEND_MODE", "real")
