"""Write a starter query set.

QUERIES ARE PART OF THE EXPERIMENT, not scaffolding. A query set of ten
near-identical "python developer" strings measures one thing ten times and
produces confidence intervals that look tight because the queries were not
independent. The set below deliberately spans:

  * specific skill queries       "fastapi postgres backend"
  * role-and-level queries       "graduate software engineer india"
  * domain queries               "fintech payments engineer"
  * awkward short queries        "c++"
  * queries with no good answer  "cobol mainframe patiala"

The last group matters most and is the one people leave out. A ranker is judged
partly on what it does when nothing is relevant, and a query set containing only
answerable queries cannot see that behaviour at all. They are kept in the file
and excluded from scoring automatically -- `graded_queries` drops any query with
no relevant judgement, which is recorded in the README because it changes the
headline numbers.

EDIT THIS FILE. These are a starting point aimed at a fintech/SDE fresher
profile. Queries that do not reflect searches you would actually run produce
labels that do not reflect relevance you actually feel, and the drift is
invisible once it is in the label set.
"""

from __future__ import annotations

import argparse
import json
import pathlib

STARTER_QUERIES = [
    # Skill-specific.
    "python backend fastapi postgres",
    "react typescript frontend",
    "kubernetes terraform infrastructure",
    "pytorch machine learning engineer",
    "sql data pipeline airflow",
    "c++ low latency systems",
    "golang microservices",
    "rest api design backend",
    # Role and level. The words a fresher actually types.
    "graduate software engineer",
    "entry level software developer",
    "junior backend engineer remote",
    "software engineer intern",
    "new grad sde 2027",
    # Domain -- the target sector.
    "fintech payments engineer",
    "quantitative developer trading",
    "risk analytics engineer",
    "banking settlement systems",
    "trading systems python",
    # Adjacent to the degree, in case the fintech route does not land.
    "embedded firmware engineer",
    "instrumentation control engineer",
    "iot sensor data platform",
    "can bus automotive software",
    # Deliberately awkward: very short, very broad, or ambiguous.
    "c++",
    "engineer",
    "remote",
    "data",
    # Deliberately unanswerable. Kept on purpose -- see the module docstring.
    "cobol mainframe patiala",
    "blacksmith apprenticeship",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="write the starter query set")
    parser.add_argument("--out", default="data")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing queries.jsonl. Refused by default: adding "
        "or removing queries after the split is frozen invalidates it, and "
        "load_split will refuse to run until it is re-frozen.",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "queries.jsonl"

    if target.exists() and not args.force:
        print(
            f"{target} already exists. Not overwriting.\n\n"
            f"Changing the query set after freezing a split invalidates that "
            f"split, and the evaluator will refuse to run until it is "
            f"re-frozen. Pass --force if that is what you mean."
        )
        return 1

    rows = [{"id": f"q{i:03d}", "text": text} for i, text in enumerate(STARTER_QUERIES)]
    target.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(rows)} queries to {target}.\n\n"
        f"Read them before labelling. Queries that are not searches you would "
        f"actually run produce labels that are not relevance you actually feel.\n\n"
        f"Next:\n"
        f"    python -m relevance.label --data {args.out} --target 250"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
