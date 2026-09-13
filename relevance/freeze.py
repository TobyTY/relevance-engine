"""Write the dev/test split, once.

Separate entry point rather than something `evaluate.py` does on demand, because
a split created automatically on first evaluation is a split created after
somebody had a hypothesis. Running this is a deliberate act with a date on it.
"""

from __future__ import annotations

import argparse

from relevance.corpus import Corpus, SplitMismatch, freeze_split


def main() -> int:
    parser = argparse.ArgumentParser(description="freeze the dev/test split")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="data/split.json")
    parser.add_argument("--test-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    corpus = Corpus.load(args.data)
    queries = corpus.graded_queries()
    if not queries:
        print("No graded queries yet. Label first, then freeze.")
        return 1

    try:
        split = freeze_split(
            corpus, args.out, test_fraction=args.test_fraction, seed=args.seed
        )
    except SplitMismatch as exc:
        print(f"REFUSING: {exc}")
        return 1

    print(
        f"Froze {len(queries)} graded queries into "
        f"{len(split.dev)} dev / {len(split.test)} test at seed {split.seed}.\n"
        f"Fingerprint {split.fingerprint}. Written to {args.out}.\n\n"
        f"From here: tune on dev only. Score test once, at the end, with\n"
        f"    python -m relevance.evaluate --test --configurations-tried N\n"
        f"and put N in the README honestly."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
