"""Agreement maths, against answers known before the code runs.

Kappa is the number that decides whether the ablation table survives, so it gets
checked against cases whose answer is fixed by definition rather than by a
second implementation: perfect agreement is 1, chance agreement is 0, and a
systematic one-grade offset sits between them.

The chance-correction is the part worth testing hardest. Most postings are
irrelevant to most queries, so two raters who both answer 0 most of the time
agree constantly while knowing nothing, and raw accuracy would call that a
triumph.
"""

from __future__ import annotations

import random

import pytest

from relevance.agreement import band, bootstrap_kappa, quadratic_weighted_kappa, report


def test_perfect_agreement_is_one():
    labels = [0, 1, 2, 3, 0, 2, 3, 1] * 4
    assert quadratic_weighted_kappa(labels, list(labels)) == pytest.approx(1.0)


def test_total_disagreement_is_strongly_negative():
    """Every 0 called a 3 and every 3 called a 0 is worse than chance, and
    kappa is signed precisely so that it can say so."""
    a = [0, 3] * 20
    b = [3, 0] * 20
    assert quadratic_weighted_kappa(a, b) < -0.9


def test_independent_raters_land_near_zero():
    """The property that makes kappa worth the arithmetic. Two raters drawing
    at random from the same distribution agree often -- on this scale roughly a
    quarter of the time -- and kappa reports that as nothing."""
    rng = random.Random(4)
    a = [rng.randint(0, 3) for _ in range(4000)]
    b = [rng.randint(0, 3) for _ in range(4000)]

    raw_agreement = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    assert raw_agreement > 0.2                      # looks like something
    assert abs(quadratic_weighted_kappa(a, b)) < 0.1  # is nothing


def test_a_skewed_corpus_does_not_manufacture_agreement():
    """The specific trap here. A pooled label set is mostly 0s, so two raters
    who both say 0 nearly always agree nearly always."""
    rng = random.Random(9)
    a = [0 if rng.random() < 0.85 else rng.randint(1, 3) for _ in range(3000)]
    b = [0 if rng.random() < 0.85 else rng.randint(1, 3) for _ in range(3000)]

    raw_agreement = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    assert raw_agreement > 0.7                       # flattering
    assert abs(quadratic_weighted_kappa(a, b)) < 0.1  # and meaningless


def test_quadratic_weighting_forgives_near_misses():
    """Ordinal scale: calling a 3 a 2 is a small disagreement, calling it a 0
    is a large one. Unweighted agreement would score both as simply wrong."""
    truth = [0, 1, 2, 3] * 15
    off_by_one = [min(3, g + 1) for g in truth]
    off_by_three = [3 - g for g in truth]

    near = quadratic_weighted_kappa(truth, off_by_one)
    far = quadratic_weighted_kappa(truth, off_by_three)
    assert near > far
    assert near > 0.5


def test_a_hand_computed_case():
    """Four pairs, worked out by hand.

    a = [0, 0, 3, 3], b = [0, 3, 0, 3]. Marginals are 2/2 for both raters, so
    every expected cell is 1. Weights on a 0-3 scale are (i-j)^2/9, so the two
    off-diagonal cells carry weight 1 and the diagonal 0.

        observed numerator = 1*1 + 1*1 = 2
        expected denominator = 1*1 + 1*1 = 2   (the same two cells)
        kappa = 1 - 2/2 = 0
    """
    assert quadratic_weighted_kappa([0, 0, 3, 3], [0, 3, 0, 3]) == pytest.approx(0.0)


def test_a_constant_rater_is_flagged_rather_than_crashing():
    """Both raters using one grade makes the chance expectation zero and kappa
    undefined. Returning 1.0 is the honest reading of 'they never disagreed',
    and the caller warns that the sample was too narrow to be informative."""
    assert quadratic_weighted_kappa([2] * 10, [2] * 10) == pytest.approx(1.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        quadratic_weighted_kappa([0, 1], [0])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        quadratic_weighted_kappa([], [])


def test_the_interval_brackets_the_estimate():
    rng = random.Random(11)
    a = [rng.randint(0, 3) for _ in range(60)]
    b = [min(3, max(0, g + rng.choice([-1, 0, 0, 1]))) for g in a]

    point = quadratic_weighted_kappa(a, b)
    lo, hi = bootstrap_kappa(a, b, resamples=800)
    assert lo <= point <= hi


def test_bands_match_their_thresholds():
    assert band(0.10) == "none to slight"
    assert band(0.30) == "fair"
    assert band(0.50) == "moderate"
    assert band(0.70) == "substantial"
    assert band(0.90) == "almost perfect"


# ------------------------------------------------------------------- report


def graded(pairs: dict, who: str) -> dict:
    return {k: (v, who) for k, v in pairs.items()}


def test_no_overlap_says_what_to_run():
    text = report(graded({("q1", "p1"): 3}, "model"), {})
    assert "No pair has been judged by both" in text
    assert "--calibrate" in text


def test_a_small_sample_is_flagged_underpowered():
    pairs = {(f"q{i}", f"p{i}"): i % 4 for i in range(10)}
    text = report(graded(pairs, "model"), graded(pairs, "human"))
    assert "UNDERPOWERED" in text


def test_low_agreement_says_withdraw_the_table():
    model = {(f"q{i}", f"p{i}"): i % 4 for i in range(40)}
    human = {k: (3 - v) for k, v in model.items()}
    text = report(graded(model, "model"), graded(human, "human"))
    assert "withdrawn" in text


def test_high_agreement_says_quote_the_figure():
    pairs = {(f"q{i}", f"p{i}"): i % 4 for i in range(40)}
    text = report(graded(pairs, "model"), graded(pairs, "human"))
    assert "measured reliability" in text


def test_a_one_sided_bias_is_named_separately_from_scatter():
    """Systematic leniency is a different fault from noise, and the fix differs:
    scatter needs more labels, bias needs the rubric rewriting."""
    model = {(f"q{i}", f"p{i}"): 3 for i in range(40)}
    human = {k: 1 for k in model}
    text = report(graded(model, "model"), graded(human, "human"))
    assert "HIGHER" in text
    assert "generous" in text
