"""The labelling CLI. One keystroke per judgement, resumable, pooled.

POOLING IS WHY THIS IS A SCRIPT AND NOT A SPREADSHEET. If you label the top 10
from BM25 and then evaluate a dense arm, every good posting the dense arm found
that BM25 missed is unjudged, and unjudged scores 0 -- so the dense arm is
punished for finding things the labelling never looked at. The pool here is the
union of the top `depth` from every available arm, which shares that bias across
arms instead of aiming it at one.

IT IS APPEND-ONLY AND RESUMABLE. 250 judgements is two sittings, not one, and a
labelling tool that loses work when you close it produces 40 labels and a
project that quietly drops its evaluation.

THE SCALE, and the wording matters more than the numbers:

    0  irrelevant          wrong field, wrong seniority, or not a job you could hold
    1  marginal            adjacent -- you would skim it and move on
    2  relevant            you would read it properly and might apply
    3  strongly relevant   you would apply

Judge the POSTING AGAINST THE QUERY, not against your overall job hunt. "Would I
apply" drifts with your mood and the time of day; "does this posting answer this
query" does not. Drift is the failure mode that quietly destroys a label set,
and it is invisible afterwards.

`--calibrate N` is the other mode. It re-presents pairs the MODEL has already
graded, with its answer hidden, so the two judgements can be compared and the
label set can carry a measured reliability instead of a disclaimer. See
`relevance/agreement.py`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

from relevance.arms import BM25Arm, DenseArm, DenseUnavailable, KeywordHeuristic
from relevance.corpus import Corpus

PROMPT = "  grade [0-3, s=skip, q=save and quit]: "


def pool(corpus: Corpus, query: str, depth: int) -> list[str]:
    arms = [KeywordHeuristic(corpus), BM25Arm(corpus)]
    try:
        arms.append(DenseArm(corpus))
    except DenseUnavailable:
        # Labelling without the dense arm in the pool is allowed, and it is
        # recorded in the README, because it biases the pool toward lexical
        # matches -- which is exactly the bias the dense arm is being tested for.
        pass
    seen: list[str] = []
    for arm in arms:
        for doc_id in arm.rank(query, limit=depth):
            if doc_id not in seen:
                seen.append(doc_id)
    return seen


def show(posting, query: str, index: int, total: int) -> None:
    body = " ".join(posting.description.split())
    print("\n" + "-" * 72)
    print(f"[{index}/{total}]  query: {query}")
    print(f"  {posting.title}  --  {posting.company}  ({posting.location})")
    print(f"  {body[:400]}{'...' if len(body) > 400 else ''}")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def calibrate(corpus: Corpus, base: pathlib.Path, want: int, seed: int) -> int:
    """Re-judge a sample of already-model-graded pairs, blind.

    STRATIFIED ACROSS THE MODEL'S GRADES, not uniform. Most pairs in a pooled
    set are irrelevant, so a uniform sample would be almost entirely 0s, two
    raters would agree on nearly all of them for no better reason than that the
    corpus is mostly irrelevant, and the resulting figure would be flattering
    and meaningless. Sampling evenly across grades puts the sample where
    disagreements can actually show up.

    The model's grade is NOT shown. An anchored judgement is not an independent
    one, and comparing a judgement against the number it was anchored to
    measures nothing at all.
    """
    by_grade: dict[int, list[tuple[str, str]]] = {}
    for qid, graded in corpus.judgements.items():
        for pid, grade in graded.items():
            by_grade.setdefault(grade, []).append((qid, pid))

    if not by_grade:
        print("No labels to calibrate against. Run the labeller first.")
        return 1

    already = {
        (r["query_id"], r["posting_id"])
        for r in read_jsonl(base / "labels.human.jsonl")
    }

    rng = random.Random(seed)
    per_grade = max(1, want // max(1, len(by_grade)))
    chosen: list[tuple[str, str]] = []
    for grade in sorted(by_grade):
        candidates = [p for p in by_grade[grade] if p not in already]
        rng.shuffle(candidates)
        chosen.extend(candidates[:per_grade])
    rng.shuffle(chosen)
    chosen = chosen[:want]

    if not chosen:
        print("Every sampled pair is already judged by you. Nothing left to calibrate.")
        return 0

    print()
    print(f"{len(chosen)} pairs, sampled evenly across the grades the model used.")
    print("Its answers are hidden, so your judgement stays independent of it.")
    print()
    print("Judge each posting against ITS QUERY -- does this posting answer this")
    print("query -- not against your job hunt overall. First instinct, then move on.")

    written = 0
    answer = ""
    with (base / "labels.human.jsonl").open("a", encoding="utf-8") as out:
        for i, (qid, pid) in enumerate(chosen, start=1):
            show(corpus.postings[pid], corpus.queries[qid], i, len(chosen))
            while True:
                try:
                    answer = input(PROMPT).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nSaved.")
                    return 0
                if answer in ("q", "s"):
                    break
                if answer in ("0", "1", "2", "3"):
                    out.write(
                        json.dumps(
                            {
                                "query_id": qid,
                                "posting_id": pid,
                                "grade": int(answer),
                                "judged_by": "human",
                            }
                        )
                        + "\n"
                    )
                    out.flush()
                    written += 1
                    break
                print("  0, 1, 2, 3, s or q")
            if answer == "q":
                break

    print()
    print(f"{written} judgements recorded in {base / 'labels.human.jsonl'}.")
    print()
    print("Now measure the agreement:")
    print(f"    python -m relevance.agreement --data {base}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="label postings 0-3 against a query")
    parser.add_argument("--data", default="data")
    parser.add_argument("--query-id", help="label one query only")
    parser.add_argument("--depth", type=int, default=10, help="how deep to pool from each arm")
    parser.add_argument(
        "--target", type=int, default=250, help="stop suggesting work at this many labels"
    )
    parser.add_argument(
        "--calibrate",
        type=int,
        metavar="N",
        help="judge N pairs the MODEL has already graded, without seeing its answer, so "
        "the two can be compared. This is what turns a model-labelled set from an "
        "unvalidated one into one with a measured reliability.",
    )
    parser.add_argument("--seed", type=int, default=20260914, help="sampling seed for --calibrate")
    args = parser.parse_args()

    base = pathlib.Path(args.data)
    corpus = Corpus.load(base)
    if not corpus.postings:
        print(f"No postings in {base}/postings.jsonl. Nothing to label.")
        return 1
    if not corpus.queries:
        print(f"No queries in {base}/queries.jsonl. Nothing to label against.")
        return 1

    if args.calibrate:
        return calibrate(corpus, base, args.calibrate, args.seed)

    labels_path = base / "labels.jsonl"
    done = {(qid, pid) for qid, graded in corpus.judgements.items() for pid in graded}
    print(f"{len(done)} labels already recorded, target {args.target}.")

    query_ids = [args.query_id] if args.query_id else list(corpus.queries)
    written = 0

    with labels_path.open("a", encoding="utf-8") as out:
        for qid in query_ids:
            query = corpus.queries[qid]
            candidates = [p for p in pool(corpus, query, args.depth) if (qid, p) not in done]
            for i, pid in enumerate(candidates, start=1):
                if len(done) + written >= args.target:
                    print(f"\nReached {args.target} labels. Stopping.")
                    return 0
                show(corpus.postings[pid], query, i, len(candidates))
                while True:
                    try:
                        answer = input(PROMPT).strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\nSaved.")
                        return 0
                    if answer == "q":
                        print(f"Saved. {written} new labels this session.")
                        return 0
                    if answer == "s":
                        break
                    if answer in ("0", "1", "2", "3"):
                        # Written and flushed one line at a time. A crash costs
                        # the judgement in progress and nothing else.
                        out.write(
                            json.dumps(
                                {
                                    "query_id": qid,
                                    "posting_id": pid,
                                    "grade": int(answer),
                                    "judged_by": "human",
                                }
                            )
                            + "\n"
                        )
                        out.flush()
                        written += 1
                        break
                    print("  0, 1, 2, 3, s or q")

    print(f"\nDone. {written} new labels, {len(done) + written} total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
