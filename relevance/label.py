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
"""

from __future__ import annotations

import argparse
import json
import pathlib
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


def main() -> int:
    parser = argparse.ArgumentParser(description="label postings 0-3 against a query")
    parser.add_argument("--data", default="data")
    parser.add_argument("--query-id", help="label one query only")
    parser.add_argument("--depth", type=int, default=10, help="how deep to pool from each arm")
    parser.add_argument("--target", type=int, default=250, help="stop suggesting work at this many labels")
    args = parser.parse_args()

    base = pathlib.Path(args.data)
    corpus = Corpus.load(base)
    if not corpus.postings:
        print(f"No postings in {base}/postings.jsonl. Nothing to label.")
        return 1
    if not corpus.queries:
        print(f"No queries in {base}/queries.jsonl. Nothing to label against.")
        return 1

    labels_path = base / "labels.jsonl"
    done = {
        (qid, pid)
        for qid, graded in corpus.judgements.items()
        for pid in graded
    }
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
                            json.dumps({"query_id": qid, "posting_id": pid, "grade": int(answer)})
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
