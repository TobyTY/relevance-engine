"""Ranking metrics, and the confidence intervals that decide whether to believe them.

THE METRICS ARE THE EASY HALF. precision@k, nDCG@k and MRR are four-line
functions and nobody gets them wrong in a way that matters. What gets projects
wrong is reporting a difference between two arms as a result when the label set
is too small to distinguish it from noise.

So every number this module produces comes with an interval, and the comparison
function reports a DIFFERENCE interval rather than two separate ones. Those are
not the same test: two overlapping confidence intervals can still correspond to
a significant difference, and two arms scored on the SAME queries are paired,
which a two-sample comparison throws away. The paired bootstrap resamples
queries -- not query-score pairs -- so both arms are always scored on the same
resampled set, which is where the extra power comes from.

WHY BOOTSTRAP AND NOT A t-TEST. Per-query nDCG is bounded in [0, 1], usually
skewed, and often has a spike at 0 for queries where nothing relevant was
retrieved. A t-interval assumes none of that. The bootstrap assumes only that
the queries are exchangeable, which is the assumption already being made by
averaging them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    #: Set when the interval was computed from too few queries to mean
    #: anything. Carried rather than raised, so a caller can print the number
    #: and the warning together instead of choosing between them.
    underpowered: bool = False

    def __str__(self) -> str:
        flag = "  (underpowered)" if self.underpowered else ""
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]{flag}"

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def precision_at_k(ranked: list[int], k: int, *, threshold: int = 1) -> float:
    """Fraction of the top k that are relevant at or above `threshold`.

    `ranked` is the list of graded relevance values in rank order. On a 0-3
    scale, threshold=1 counts anything not explicitly irrelevant; threshold=2
    counts only the genuinely good ones. Both are defensible and they answer
    different questions, so the threshold is a parameter rather than a constant
    buried in the function.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked[:k]
    if not top:
        return 0.0
    # Divided by k, not by len(top). A system that returns three results when
    # ten were asked for has not earned the same score as one that returned ten
    # with the same three hits.
    return sum(1 for grade in top if grade >= threshold) / k


def dcg(ranked: list[int], k: int) -> float:
    # Gain 2^g - 1 rather than g: on a 0-3 scale it makes a grade-3 document
    # worth 7 and a grade-1 worth 1, which matches the intent of a graded scale.
    # Using g directly says one perfect result equals three marginal ones.
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ranked[:k]))


def ndcg_at_k(ranked: list[int], k: int, *, ideal: list[int] | None = None) -> float:
    """DCG normalised by the best achievable ordering of the SAME judgements.

    `ideal` must be every judged grade for the query, not just the grades the
    system returned. Normalising by the returned set scores a system that
    retrieved one marginal document and ranked it first as a perfect 1.0.
    """
    pool = sorted(ideal if ideal is not None else ranked, reverse=True)
    best = dcg(pool, k)
    return dcg(ranked, k) / best if best > 0 else 0.0


def reciprocal_rank(ranked: list[int], *, threshold: int = 1) -> float:
    for rank, grade in enumerate(ranked, start=1):
        if grade >= threshold:
            return 1.0 / rank
    return 0.0


#: Below this many queries a bootstrap interval is wide enough to be
#: uninformative, and reporting a winner from it is the mistake this whole
#: module exists to prevent.
MIN_QUERIES_FOR_INFERENCE = 20


def bootstrap(
    per_query: list[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over queries."""
    if not per_query:
        return Interval(0.0, 0.0, 0.0, underpowered=True)
    point = sum(per_query) / len(per_query)
    if len(per_query) == 1:
        return Interval(point, point, point, underpowered=True)

    rng = random.Random(seed)
    n = len(per_query)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += per_query[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    tail = (1 - confidence) / 2
    low = means[int(tail * resamples)]
    high = means[min(resamples - 1, int((1 - tail) * resamples))]
    return Interval(point, low, high, underpowered=n < MIN_QUERIES_FOR_INFERENCE)


def paired_bootstrap(
    arm_a: list[float],
    arm_b: list[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Interval on the per-query DIFFERENCE a - b.

    Paired, because both arms were scored on the same queries. An unpaired
    comparison discards that and is strictly less powerful -- it has to account
    for between-query variance that cancels exactly when the same query is
    scored twice.
    """
    if len(arm_a) != len(arm_b):
        raise ValueError(f"arms have different query counts: {len(arm_a)} vs {len(arm_b)}")
    return bootstrap(
        [a - b for a, b in zip(arm_a, arm_b)],
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
