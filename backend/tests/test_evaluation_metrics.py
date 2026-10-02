import pytest

from app.services.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_relevant_items_found():
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 2) == pytest.approx(1 / 3)


def test_precision_at_k_uses_k_as_denominator():
    assert precision_at_k(["a", "x"], {"a", "b"}, 3) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == pytest.approx(0.5)


def test_reciprocal_rank_is_zero_when_no_relevant_result():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_is_one_for_ideal_order():
    assert ndcg_at_k(["a", "b", "c"], {"a": 3, "b": 2, "c": 1}, 3) == pytest.approx(1.0)


def test_ndcg_penalizes_relevant_item_lower_in_ranking():
    ideal = ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2)
    swapped = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2)
    assert swapped < ideal


@pytest.mark.parametrize("k", [0, -1, True])
def test_metrics_reject_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(["a"], {"a"}, k)


def test_recall_returns_zero_for_empty_relevance_set():
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_ndcg_rejects_negative_relevance():
    with pytest.raises(ValueError, match="non-negative"):
        ndcg_at_k(["a"], {"a": -1}, 1)
