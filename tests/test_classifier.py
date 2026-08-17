"""
tests/test_classifier.py

Unit tests for app/complexity_classifier.py.

All tests are synchronous — the classifier is pure Python with no I/O.
"""
from __future__ import annotations

import pytest

from app.complexity_classifier import ComplexityTier, classify


# ---------------------------------------------------------------------------
# FAST tier — explicit signal matches
# ---------------------------------------------------------------------------

class TestFastSignals:
    def test_greeting_hi(self):
        assert classify("Hi there") == ComplexityTier.FAST

    def test_greeting_hello(self):
        assert classify("Hello, how are you?") == ComplexityTier.FAST

    def test_what_is_capital(self):
        assert classify("What is the capital of France?") == ComplexityTier.FAST

    def test_who_is(self):
        assert classify("Who is Ada Lovelace?") == ComplexityTier.FAST

    def test_who_was(self):
        assert classify("Who was Alan Turing?") == ComplexityTier.FAST

    def test_when_did(self):
        assert classify("When did World War II end?") == ComplexityTier.FAST

    def test_define(self):
        assert classify("Define recursion.") == ComplexityTier.FAST

    def test_translate(self):
        assert classify("Translate 'hello' to Spanish.") == ComplexityTier.FAST

    def test_arithmetic(self):
        assert classify("2 + 2") == ComplexityTier.FAST

    def test_thanks(self):
        assert classify("Thanks!") == ComplexityTier.FAST

    def test_yes(self):
        assert classify("yes") == ComplexityTier.FAST


# ---------------------------------------------------------------------------
# QUALITY tier — explicit signal matches
# ---------------------------------------------------------------------------

class TestQualitySignals:
    def test_explain(self):
        assert classify("Explain why async/await avoids race conditions.") == ComplexityTier.QUALITY

    def test_why(self):
        assert classify("Why do neural networks need non-linear activations?") == ComplexityTier.QUALITY

    def test_compare(self):
        assert classify("Compare transformers and RNNs for NLP.") == ComplexityTier.QUALITY

    def test_design(self):
        assert classify("Design a rate-limiting middleware for a REST API.") == ComplexityTier.QUALITY

    def test_implement(self):
        assert classify("Implement a binary search tree in Python.") == ComplexityTier.QUALITY

    def test_code(self):
        assert classify("Code a function to reverse a linked list.") == ComplexityTier.QUALITY

    def test_write_function(self):
        assert classify("Write a function that computes Fibonacci numbers.") == ComplexityTier.QUALITY

    def test_write_essay(self):
        assert classify("Write an essay on the ethics of AI.") == ComplexityTier.QUALITY

    def test_debug(self):
        assert classify("Debug this Python traceback: AttributeError...") == ComplexityTier.QUALITY

    def test_refactor(self):
        assert classify("Refactor this function to use list comprehensions.") == ComplexityTier.QUALITY

    def test_step_by_step(self):
        assert classify("Solve this step-by-step: integrate x^2 from 0 to 3.") == ComplexityTier.QUALITY

    def test_summarise(self):
        assert classify("Summarise the key contributions of the Attention Is All You Need paper.") == ComplexityTier.QUALITY

    def test_how_does(self):
        assert classify("How does gradient descent work?") == ComplexityTier.QUALITY

    def test_trade_offs(self):
        assert classify("What are the trade-offs between SQL and NoSQL databases?") == ComplexityTier.QUALITY

    def test_optimise(self):
        assert classify("Optimise this SQL query for large datasets.") == ComplexityTier.QUALITY

    def test_analyse(self):
        assert classify("Analyse the time complexity of merge sort.") == ComplexityTier.QUALITY

    def test_evaluate(self):
        assert classify("Evaluate the pros and cons of microservices architecture.") == ComplexityTier.QUALITY

    def test_critique(self):
        assert classify("Critique this essay draft for logical coherence.") == ComplexityTier.QUALITY


# ---------------------------------------------------------------------------
# Token-count fallback
# ---------------------------------------------------------------------------

class TestTokenFallback:
    def test_short_unknown_prompt_is_fast(self):
        # No signal keywords, short → FAST
        assert classify("Tell me something") == ComplexityTier.FAST

    def test_long_unknown_prompt_is_quality(self):
        # No signal keywords, but >= 60 words → QUALITY
        long_prompt = " ".join(["word"] * 65)
        assert classify(long_prompt) == ComplexityTier.QUALITY

    def test_exactly_threshold_is_quality(self):
        threshold_prompt = " ".join(["word"] * 60)
        assert classify(threshold_prompt) == ComplexityTier.QUALITY

    def test_one_under_threshold_is_fast(self):
        short_prompt = " ".join(["word"] * 59)
        assert classify(short_prompt) == ComplexityTier.FAST


# ---------------------------------------------------------------------------
# Priority ordering — QUALITY wins over FAST signals in same prompt
# ---------------------------------------------------------------------------

class TestPriority:
    def test_quality_beats_fast_when_both_present(self):
        # "what is" is a FAST signal, "explain" is QUALITY — QUALITY should win
        assert classify("What is the best way to explain recursion?") == ComplexityTier.QUALITY

    def test_quality_beats_short_length(self):
        # Short prompt but has explicit quality signal
        assert classify("Why?") == ComplexityTier.QUALITY


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_ish_prompt(self):
        # Very short prompt with no signals — FAST by default
        assert classify("ok") == ComplexityTier.FAST

    def test_case_insensitive_quality(self):
        assert classify("EXPLAIN the Fourier transform.") == ComplexityTier.QUALITY

    def test_case_insensitive_fast(self):
        assert classify("DEFINE entropy.") == ComplexityTier.FAST

    def test_multiline_quality(self):
        prompt = """
        I have this Python function and it's slow.
        Can you debug and refactor it to improve performance?
        """
        assert classify(prompt) == ComplexityTier.QUALITY

    def test_whitespace_only_is_fast(self):
        # Stripped empty string → no signals, no words → FAST
        assert classify("   ") == ComplexityTier.FAST
