"""
app/complexity_classifier.py

Heuristic prompt complexity classifier.

Bins any incoming prompt into one of two tiers:
  - FAST    → simple factual lookups, greetings, short queries
              → prefer local Ollama (fast, free, good enough)
  - QUALITY → reasoning, code generation, multi-step analysis, writing
              → prefer remote cloud LLM (higher capability)

Design decisions:
  - Pure Python (regex + word count). Zero I/O, zero latency overhead.
  - No ML model required — avoids "use a model to route a model" absurdity.
  - Priority order: quality signals → fast signals → token-count fallback.
  - Entirely unit-testable with deterministic inputs.
  - The tier is a *preference signal* fed to DecisionAgent, not a hard override.
    Budget caps and latency limits still apply regardless of tier.
"""
from __future__ import annotations

import re
from enum import Enum


class ComplexityTier(str, Enum):
    FAST    = "FAST"     # Lightweight — local model is fine
    QUALITY = "QUALITY"  # Heavyweight — route to best available model


# ---------------------------------------------------------------------------
# Signal sets
# ---------------------------------------------------------------------------

# Patterns that strongly suggest the request needs deep reasoning or generation.
# Checked first — they take priority over FAST signals.
_QUALITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bexplain\b",
        r"\bwhy\b",
        r"\banalys[ei]s?\b",         # analyse / analyze / analysis
        r"\banalyze?\b",
        r"\bcompare\b",
        r"\bcontrast\b",
        r"\bdesign\b",
        r"\barchitect\b",
        r"\bimplement\b",
        r"\bcode\b",
        r"\bwrite\b.{0,40}\b(essay|report|letter|function|class|module|script)\b",
        r"\bprove\b",
        r"\bderive\b",
        r"\boptimi[sz]e?\b",
        r"\brefactor\b",
        r"\bdebug\b",
        r"\bstep.by.step\b",
        r"\bbreakdown\b",
        r"\bsummarise\b",
        r"\bsummarize\b",
        r"\bhow does\b",
        r"\bhow do\b",
        r"\bwhat are the\b",
        r"\badvantages?\b",
        r"\bdisadvantages?\b",
        r"\btrade.?offs?\b",
        r"\brecommend\b",
        r"\bsugggest\b",
        r"\bevaluate\b",
        r"\bcritique\b",
    ]
]

# Patterns that indicate simple, lookup-style requests.
_FAST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^what is\b",
        r"^who is\b",
        r"^who was\b",
        r"^when (is|was|did)\b",
        r"^where (is|was)\b",
        r"^(hi|hello|hey|good morning|good evening)\b",
        r"^thanks?\b",
        r"^(yes|no|ok|okay|sure)\b",
        r"\bdefine\b",
        r"\bspell\b",
        r"\btranslat[ei]\b",
        r"^list\b.{0,20}\b(of\b)?",
        r"^\d[\d\s\+\-\*\/\^]+\=?$",  # pure arithmetic expression
    ]
]

# Prompts longer than this word count default to QUALITY when no signal matches.
_TOKEN_THRESHOLD = 60


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(prompt: str) -> ComplexityTier:
    """
    Classify a prompt into FAST or QUALITY.

    Priority:
      1. Explicit QUALITY signals (regex match) → QUALITY
      2. Explicit FAST signals (regex match)    → FAST
      3. Word count >= threshold                → QUALITY
      4. Default                               → FAST
    """
    stripped = prompt.strip()

    for pattern in _QUALITY_PATTERNS:
        if pattern.search(stripped):
            return ComplexityTier.QUALITY

    for pattern in _FAST_PATTERNS:
        if pattern.search(stripped):
            return ComplexityTier.FAST

    word_count = len(stripped.split())
    if word_count >= _TOKEN_THRESHOLD:
        return ComplexityTier.QUALITY

    return ComplexityTier.FAST
