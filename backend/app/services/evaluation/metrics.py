"""Pure retrieval-evaluation metrics that do not require a database or LLM."""

from collections.abc import Mapping, Sequence
from math import log2
from typing import Hashable, TypeVar

T = TypeVar("T", bound=Hashable)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")


def recall_at_k(retrieved: Sequence[T], relevant: set[T], k: int) -> float:
    """Fraction of all relevant items found in the first k retrieved items."""
    _validate_k(k)
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[T], relevant: set[T], k: int) -> float:
    """Fraction of the first k result slots that contain relevant items."""
    _validate_k(k)
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: Sequence[T], relevant: set[T]) -> float:
    """Reciprocal rank of the first relevant result; zero when none is found."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[T], relevance: Mapping[T, float], k: int) -> float:
    """Normalized discounted cumulative gain for graded relevance labels."""
    _validate_k(k)
    if any(score < 0 for score in relevance.values()):
        raise ValueError("Relevance scores must be non-negative.")

    def gain(score: float) -> float:
        return (2**score - 1) / log2(2 + 0)  # denominator replaced by rank discount below

    actual = sum(
        (2**relevance.get(item, 0) - 1) / log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
    )
    ideal_scores = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(
        (2**score - 1) / log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )
    return actual / ideal if ideal else 0.0
