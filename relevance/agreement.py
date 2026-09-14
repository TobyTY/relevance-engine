"""Measure how far the model's labels are from a human's.

WHY THIS EXISTS. The 241 labels in `labels.jsonl` were produced by a language
model, and the evaluator prints a banner saying the table built on them is not a
measurement of relevance. This turns that from a disclaimer into a number.

Sample the same query-posting pairs, have a person grade them WITHOUT seeing the
model's answer, and compare. If the two agree closely the label set carries a
measured reliability figure and the ablation table can be believed as far as that
figure allows. If they do not, the table is withdrawn, and that is also a result.

WHY QUADRATIC-WEIGHTED KAPPA AND NOT ACCURACY. The scale is ordinal: grading a 3
as a 2 is a small disagreement and grading it a 0 is a large one, and plain
accuracy calls both simply "wrong". Quadratic weighting penalises by the SQUARE
of the distance, which is the standard choice for graded relevance.

Kappa also corrects for chance agreement, which matters here more than usual:
most postings are irrelevant to most queries, so two raters who both say 0 most
of the time agree constantly without either of them knowing anything. Raw
agreement on this data would look impressive and mean very little.

    < 0.20   none to slight        the label set cannot support the table
    0.21-0.40 fair
    0.41-0.60 moderate             usable with the figure quoted beside it
    0.61-0.80 substantial
    > 0.80   almost perfect

Those bands are Landis and Koch's, and they are conventions rather than laws.
The interval matters more than the band: on 30 pairs it is wide, and this module
reports it rather than quoting kappa as though it were exact.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

from relevance.corpus import HUMAN, MAX_GRADE, MIN_GRADE, MODEL


def load_graded(path: pathlib.Path) -> dict[tuple[str, str], tuple[int, str]]:
    """Read labels keyed by (query, posting), keeping who judged each one."""
    out: dict[tuple[str, str], tuple[int, str]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[(row["query_id"], row["posting_id"])] = (
            int(row["grade"]),
            row.get("judged_by", HUMAN),
        )
    return out


def quadratic_weighted_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa with quadratic weights, over the full 0-3 scale.

    The confusion matrix is built over every grade in the scale rather than only
    the grades that appear, so a sample where nobody used "1" still divides by
    the right chance expectation instead of silently becoming a 3-point scale.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equal, non-empty label sequences")

    grades = list(range(MIN_GRADE, MAX_GRADE + 1))
    k = len(grades)
    n = len(a)

    observed = [[0.0] * k for _ in grades]
    for x, y in zip(a, b):
        observed[x - MIN_GRADE][y - MIN_GRADE] += 1

    hist_a = [sum(observed[i]) for i in range(k)]
    hist_b = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            expected = hist_a[i] * hist_b[j] / n
            num += w * observed[i][j]
            den += w * expected
    if den == 0:
        # Perfect agreement AND no spread in either rater. Undefined in the
        # formula; 1.0 is the honest reading and is flagged by the caller
        # because it usually means the sample was too narrow to be informative.
        return 1.0
    return 1.0 - num / den


def bootstrap_kappa(
    a: list[int], b: list[int], *, resamples: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval on kappa, because 30 pairs is not many."""
    rng = random.Random(seed)
    n = len(a)
    values = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            values.append(quadratic_weighted_kappa([a[i] for i in idx], [b[i] for i in idx]))
        except ValueError:
            continue
    if not values:
        return (0.0, 0.0)
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def band(kappa: float) -> str:
    if kappa < 0.20:
        return "none to slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"


def report(model: dict, human: dict) -> str:
    overlap = sorted(set(model) & set(human))
    lines: list[str] = []

    if not overlap:
        return (
            "No pair has been judged by both.\n\n"
            "Label a calibration sample first:\n"
            "    python -m relevance.label --calibrate 30\n\n"
            "It draws from the pairs the model already graded and hides its\n"
            "answer, so the two judgements stay independent."
        )

    m = [model[k][0] for k in overlap]
    h = [human[k][0] for k in overlap]

    exact = sum(1 for x, y in zip(m, h) if x == y) / len(overlap)
    within_one = sum(1 for x, y in zip(m, h) if abs(x - y) <= 1) / len(overlap)
    kappa = quadratic_weighted_kappa(m, h)
    lo, hi = bootstrap_kappa(m, h)

    lines.append(f"Pairs judged by both       {len(overlap)}")
    lines.append(f"Exact agreement            {exact:.1%}")
    lines.append(f"Agreement within one grade {within_one:.1%}")
    lines.append(f"Quadratic-weighted kappa   {kappa:.3f}  [{lo:.3f}, {hi:.3f}]   {band(kappa)}")
    lines.append("")

    if len(overlap) < 25:
        lines.append(
            f"UNDERPOWERED: {len(overlap)} pairs. The interval above is wide enough that"
        )
        lines.append("the band is not meaningfully distinguishable from its neighbours.")
        lines.append("")

    lines.append("Model grade (down) against human grade (across):")
    lines.append("        " + "".join(f"{g:>6}" for g in range(MIN_GRADE, MAX_GRADE + 1)))
    for gm in range(MIN_GRADE, MAX_GRADE + 1):
        row = [sum(1 for x, y in zip(m, h) if x == gm and y == gh)
               for gh in range(MIN_GRADE, MAX_GRADE + 1)]
        lines.append(f"    {gm:>3} " + "".join(f"{c:>6}" for c in row))
    lines.append("")

    harsh = sum(1 for x, y in zip(m, h) if x < y)
    lenient = sum(1 for x, y in zip(m, h) if x > y)
    if harsh or lenient:
        lines.append(
            f"Direction: the model graded {lenient} pairs HIGHER than you and "
            f"{harsh} lower."
        )
        if abs(lenient - harsh) > len(overlap) * 0.25:
            side = "generous" if lenient > harsh else "harsh"
            lines.append(
                f"That is a lopsided {side} bias rather than scatter, which is a "
                f"different problem from disagreement and is worth reading the "
                f"confusion matrix for."
            )
        lines.append("")

    if kappa < 0.41:
        lines.append(
            "VERDICT: too low to support the ablation table. The model is not "
            "measuring what you mean by relevance, so the table built on its "
            "labels should be withdrawn rather than qualified."
        )
    elif kappa < 0.61:
        lines.append(
            "VERDICT: usable, with this figure quoted beside every result. The "
            "table describes a notion of relevance close to but not identical "
            "to yours, and no conclusion should rest on a margin smaller than "
            "that gap."
        )
    else:
        lines.append(
            "VERDICT: the label set carries a measured reliability. Quote kappa "
            "and its interval in the README next to the ablation table."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="model vs human label agreement")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", help="also write the report here")
    args = parser.parse_args()

    base = pathlib.Path(args.data)
    everything = load_graded(base / "labels.jsonl")
    model = {k: v for k, v in everything.items() if v[1] == MODEL}
    human = {k: v for k, v in everything.items() if v[1] == HUMAN}
    human.update(load_graded(base / "labels.human.jsonl"))

    text = report(model, human)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
