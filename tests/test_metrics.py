"""Metrics tested against hand-computed answers, not against themselves.

A metric test that recomputes the metric a second way and compares is testing
that you can write the same bug twice. Every expected value below is worked out
by hand in the docstring, so the test fails if the implementation drifts toward
a different definition -- which is how nDCG quietly becomes "nDCG with linear
gain" and stops being comparable to anything published.
"""

from __future__ import annotations

import pytest

from relevance.metrics import (
    Interval,
    bootstrap,
    dcg,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    reciprocal_rank,
)


def test_precision_divides_by_k_not_by_what_was_returned():
    """Three results returned, two relevant, k=5.

    2/5 = 0.4, not 2/3. A system that returns three results when ten were asked
    for has not earned the score of one that returned ten with the same hits.
    """
    assert precision_at_k([3, 0, 2], 5) == pytest.approx(0.4)


def test_precision_threshold_separates_marginal_from_relevant():
    ranked = [1, 1, 2, 0, 0]
    assert precision_at_k(ranked, 5, threshold=1) == pytest.approx(0.6)
    assert precision_at_k(ranked, 5, threshold=2) == pytest.approx(0.2)


def test_dcg_uses_exponential_gain():
    """grades [3, 0, 1], k=3.

        (2^3 - 1)/log2(2) + (2^0 - 1)/log2(3) + (2^1 - 1)/log2(4)
      =      7/1          +        0          +      1/2
      = 7.5
    """
    assert dcg([3, 0, 1], 3) == pytest.approx(7.5)


def test_ndcg_normalises_by_the_ideal_ordering():
    """returned [1, 3], ideal grades for the query [3, 1].

      DCG  = 1/log2(2) + 7/log2(3) = 1 + 4.416508 = 5.416508
      IDCG = 7/log2(2) + 1/log2(3) = 7 + 0.630930 = 7.630930
      nDCG = 5.416508 / 7.630930   = 0.709810
    """
    assert ndcg_at_k([1, 3], 2, ideal=[3, 1]) == pytest.approx(0.709810, abs=1e-5)


def test_ndcg_ideal_comes_from_all_judgements_not_the_returned_set():
    """One marginal document returned first, while a grade-3 exists unretrieved.

    Normalising by the returned set alone scores this 1.0 -- a perfect ranking
    of the one thing it happened to find. Against the full judgement set it is
    small, which is the truthful answer.
    """
    assert ndcg_at_k([1], 10, ideal=[1]) == pytest.approx(1.0)
    assert ndcg_at_k([1], 10, ideal=[3, 3, 1]) < 0.1


def test_ndcg_is_zero_when_nothing_relevant_exists():
    assert ndcg_at_k([0, 0], 2, ideal=[0, 0]) == 0.0


def test_reciprocal_rank_is_one_over_the_first_hit():
    assert reciprocal_rank([0, 0, 2]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0, 0]) == 0.0


# ------------------------------------------------------------------ intervals


def test_a_constant_arm_has_a_zero_width_interval():
    interval = bootstrap([0.5] * 40)
    assert interval.point == pytest.approx(0.5)
    assert interval.low == pytest.approx(0.5)
    assert interval.high == pytest.approx(0.5)


def test_the_interval_contains_the_point_estimate():
    values = [0.1, 0.9, 0.3, 0.7, 0.5] * 8
    interval = bootstrap(values, resamples=2000)
    assert interval.low <= interval.point <= interval.high


def test_few_queries_are_flagged_underpowered():
    assert bootstrap([0.4, 0.6]).underpowered is True
    assert bootstrap([0.5] * 40).underpowered is False


def test_identical_arms_produce_a_difference_interval_containing_zero():
    """The guard that stops the project claiming a win it does not have."""
    arm = [0.2, 0.8, 0.4, 0.6, 0.5] * 8
    diff = paired_bootstrap(arm, list(arm), resamples=2000)
    assert diff.point == pytest.approx(0.0)
    assert diff.excludes_zero is False


def test_a_consistent_improvement_is_detected():
    """Every query improves by exactly 0.1. The paired difference has zero
    variance, so the interval is tight and excludes zero -- which is the case
    an unpaired test would blur, because the between-query spread it would have
    to account for cancels exactly here."""
    base = [0.1, 0.9, 0.3, 0.7, 0.5] * 8
    better = [v + 0.1 for v in base]
    diff = paired_bootstrap(better, base, resamples=2000)
    assert diff.point == pytest.approx(0.1)
    assert diff.excludes_zero is True


def test_a_tiny_difference_on_a_noisy_small_set_stays_inconclusive():
    """The result this whole module exists to force. A small mean gap on a
    noisy, small label set must not be reported as a win."""
    base = [0.0, 1.0] * 10
    noisy = [1.0, 0.0] * 9 + [1.0, 0.1]
    diff = paired_bootstrap(noisy, base, resamples=3000)
    assert diff.excludes_zero is False


def test_mismatched_arm_lengths_raise():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_bootstrap_is_deterministic_for_a_seed():
    values = [0.2, 0.4, 0.6, 0.8] * 10
    assert bootstrap(values, seed=7, resamples=500) == bootstrap(values, seed=7, resamples=500)


def test_interval_formats_with_its_warning():
    assert "underpowered" in str(Interval(0.5, 0.4, 0.6, underpowered=True))
    assert "underpowered" not in str(Interval(0.5, 0.4, 0.6))
