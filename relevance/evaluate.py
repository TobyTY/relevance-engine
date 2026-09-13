"""Run the arms, score them, and print a table that refuses to overclaim.

THE GUARDS ARE THE FEATURE. Anyone can average nDCG across queries. What this
module does that a notebook does not:

  1. It reports a PAIRED difference interval against the chosen baseline, not
     two separate intervals the reader is invited to eyeball for overlap.
  2. It refuses to name a winner when the difference interval spans zero, and
     says "inconclusive at this label count" instead.
  3. It prints the number of queries behind every number, and flags the table
     as underpowered below the threshold in `metrics.py`.
  4. It records how many configurations were tried, because twenty attempts at
     a held-out set is twenty chances to find noise that flatters one of them.
  5. It scores the test set only when asked explicitly, and says so loudly.

None of that makes the ranker better. It makes the claim about the ranker
survivable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass, field

from relevance import metrics
from relevance.arms import Arm, BM25Arm, DenseArm, DenseUnavailable, HybridRRF, KeywordHeuristic
from relevance.corpus import MODEL, Corpus, SplitMismatch, load_split

CUTOFF = 10


@dataclass
class ArmResult:
    name: str
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)

    def column(self, metric: str, query_ids: list[str]) -> list[float]:
        return [self.per_query[q][metric] for q in query_ids]


def score_arm(arm: Arm, corpus: Corpus, query_ids: list[str]) -> ArmResult:
    result = ArmResult(arm.name)
    for qid in query_ids:
        ranked_ids = arm.rank(corpus.queries[qid], limit=CUTOFF)
        grades = corpus.grades_for(qid, ranked_ids)
        ideal = corpus.ideal_grades(qid)
        result.per_query[qid] = {
            "p@5": metrics.precision_at_k(grades, 5),
            "p@10": metrics.precision_at_k(grades, 10),
            "ndcg@10": metrics.ndcg_at_k(grades, 10, ideal=ideal),
            "mrr": metrics.reciprocal_rank(grades),
        }
    return result


def build_arms(corpus: Corpus) -> tuple[list[Arm], list[str]]:
    """Returns the arms that could be built and the reasons any were skipped."""
    keyword = KeywordHeuristic(corpus)
    bm25 = BM25Arm(corpus)
    arms: list[Arm] = [keyword, bm25]
    skipped: list[str] = []
    try:
        dense = DenseArm(corpus)
        arms.append(dense)
        arms.append(HybridRRF([bm25, dense]))
    except DenseUnavailable as exc:
        # Named in the output rather than quietly dropped. A table missing the
        # dense arm with no explanation reads as a dense arm that lost.
        skipped.append(f"dense and hybrid: {exc}")
    return arms, skipped


def report(
    corpus: Corpus,
    query_ids: list[str],
    *,
    baseline: str = "bm25",
    configurations_tried: int = 1,
    split_name: str = "dev",
) -> str:
    arms, skipped = build_arms(corpus)
    results = {arm.name: score_arm(arm, corpus, query_ids) for arm in arms}

    lines: list[str] = []

    # Printed above everything, because the provenance of a label set bounds
    # what its table can claim, and a disclosure kept only in a README is one
    # screenshot away from being separated from the numbers it qualifies.
    if corpus.provenance.get(MODEL):
        lines.append("=" * 72)
        lines.append("MODEL-JUDGED LABELS. This table is not a measurement of relevance.")
        lines.append("")
        lines.append("The relevance judgements below were produced by a language model, not")
        lines.append("by the person whose job search this ranker is for. The table therefore")
        lines.append("compares rankers against a model's notion of relevance, which is a")
        lines.append("weaker claim than it looks: an arm can win here by matching the judge's")
        lines.append("biases rather than by being more useful.")
        lines.append("")
        lines.append("No human-judged sample exists, so the agreement between model and human")
        lines.append("judgement is UNMEASURED. Until it is measured, treat every number below")
        lines.append("as a check that the pipeline runs, not as a result.")
        lines.append("")
        lines.append("ONE CONFOUND IN PARTICULAR. The model judged each posting largely from")
        lines.append("its title and the terms in its text, because that is what a short")
        lines.append("posting snippet makes available. That is the same signal BM25 ranks on.")
        lines.append("So a lexical arm beating a dense one on these labels is the outcome the")
        lines.append("labelling procedure was always most likely to produce, and it cannot be")
        lines.append("separated from a real lexical advantage without human labels.")
        lines.append("=" * 72)
        lines.append("")

    lines.append(f"Split          {split_name}")
    lines.append(f"Queries        {len(query_ids)}")
    lines.append(f"Labels         {corpus.label_provenance()}")
    lines.append(f"Postings       {len(corpus.postings)}")
    lines.append(f"Cutoff         {CUTOFF}")
    lines.append(f"Baseline       {baseline}")
    lines.append(f"Configurations tried before this table: {configurations_tried}")
    if configurations_tried > 1:
        lines.append(
            "  Each additional configuration is another chance for noise to "
            "flatter one arm. Read the intervals with that in mind."
        )
    lines.append("")

    if len(query_ids) < metrics.MIN_QUERIES_FOR_INFERENCE:
        lines.append(
            f"UNDERPOWERED: {len(query_ids)} queries is below the "
            f"{metrics.MIN_QUERIES_FOR_INFERENCE} this harness treats as the "
            f"minimum for inference. Every interval below is wide enough that "
            f"naming a winner from it would be reading noise."
        )
        lines.append("")

    for metric in ("p@5", "p@10", "ndcg@10", "mrr"):
        lines.append(f"  {metric}")
        base = results.get(baseline)
        for name, result in results.items():
            point = metrics.bootstrap(result.column(metric, query_ids))
            row = f"    {name:<10} {point}"
            if base is not None and name != baseline:
                diff = metrics.paired_bootstrap(
                    result.column(metric, query_ids), base.column(metric, query_ids)
                )
                verdict = (
                    ("better" if diff.point > 0 else "worse")
                    if diff.excludes_zero
                    else "inconclusive"
                )
                row += f"   vs {baseline}: {diff.point:+.4f} [{diff.low:+.4f}, {diff.high:+.4f}]  {verdict}"
            lines.append(row)
        lines.append("")

    if skipped:
        lines.append("Arms not run:")
        lines.extend(f"  {reason}" for reason in skipped)
        lines.append("")

    lines.append(
        "Reading this table: an interval spanning zero means this label set "
        "cannot separate the two arms. That is a result, not a missing result, "
        "and the honest next step is more labels rather than another model."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="ablation table over the ranking arms")
    parser.add_argument("--data", default="data", help="directory holding the jsonl files")
    parser.add_argument("--split", default="data/split.json")
    parser.add_argument("--baseline", default="bm25")
    parser.add_argument("--configurations-tried", type=int, default=1)
    parser.add_argument(
        "--test",
        action="store_true",
        help="score the HELD-OUT set. Do this once, at the end, and record that you did.",
    )
    parser.add_argument("--out", help="also write the table here")
    args = parser.parse_args()

    corpus = Corpus.load(args.data)
    if not corpus.graded_queries():
        print(
            "No graded queries. Label some postings first:\n"
            "    python -m relevance.label --data data\n"
            "The harness is complete and tested; it has nothing to score yet."
        )
        return 1

    try:
        split = load_split(corpus, args.split)
    except FileNotFoundError:
        print(f"No frozen split at {args.split}. Create one BEFORE looking at any result:\n"
              f"    python -m relevance.freeze --data {args.data} --out {args.split}")
        return 1
    except SplitMismatch as exc:
        print(f"REFUSING TO RUN: {exc}")
        return 1

    if args.test:
        print("=" * 72)
        print("SCORING THE HELD-OUT TEST SET. Record this run in the README.")
        print("=" * 72)

    query_ids = split.test if args.test else split.dev
    table = report(
        corpus,
        query_ids,
        baseline=args.baseline,
        configurations_tried=args.configurations_tried,
        split_name="test (held out)" if args.test else "dev",
    )
    print(table)
    if args.out:
        pathlib.Path(args.out).write_text(table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
