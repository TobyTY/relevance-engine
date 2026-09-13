"""Generate a synthetic corpus so the harness can be run before any real labels exist.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR. It shows the file format, proves the
pipeline runs end to end, and demonstrates the guards firing. It is NOT a
result. Synthetic labels are generated from a rule the ranker can rediscover, so
every arm scores implausibly well and the table means nothing about real job
postings. Every file it writes is stamped `"synthetic": true` and the evaluator
prints a banner, so a synthetic table can never be mistaken for a real one in a
screenshot.

The real corpus is yours. See the README for the two files you need to export.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

ROLES = [
    ("Backend Engineer", "python fastapi postgres docker rest api"),
    ("Data Engineer", "python spark airflow sql warehouse etl"),
    ("Embedded Engineer", "c firmware rtos i2c spi can bus microcontroller"),
    ("Frontend Engineer", "react typescript css webpack javascript"),
    ("Quant Developer", "c++ low latency market data order book python"),
    ("DevOps Engineer", "kubernetes terraform aws ci cd observability"),
    ("ML Engineer", "pytorch training inference gpu embeddings evaluation"),
    ("Site Reliability Engineer", "slo incident oncall prometheus grafana linux"),
]
COMPANIES = ["Northwind", "Kalyani Systems", "Trellis", "Arcadia Labs", "Meridian", "Satara Tech"]
CITIES = ["Bengaluru", "Pune", "Hyderabad", "Gurugram", "Chennai", "Remote"]
FILLER = (
    "We are looking for a motivated engineer to join our growing team. "
    "You will work closely with product and design. Competitive salary, "
    "health insurance, and an annual learning budget. "
)


def generate(directory: pathlib.Path, *, postings: int, queries: int, seed: int) -> None:
    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(postings):
        title, skills = ROLES[i % len(ROLES)]
        skill_words = skills.split()
        rng.shuffle(skill_words)
        # Length varies a lot, on purpose: it is what makes BM25's length
        # normalisation matter, and a synthetic corpus with uniform lengths
        # would hide a normalisation bug rather than exercise it.
        padding = FILLER * rng.randint(1, 5)
        rows.append(
            {
                "id": f"p{i:04d}",
                "title": title,
                "company": rng.choice(COMPANIES),
                "location": rng.choice(CITIES),
                "description": " ".join(skill_words) + ". " + padding,
                "role": title,
                "synthetic": True,
            }
        )
    (directory / "postings.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    query_rows = []
    for i in range(queries):
        title, skills = ROLES[i % len(ROLES)]
        terms = skills.split()
        rng.shuffle(terms)
        query_rows.append(
            {"id": f"q{i:03d}", "text": " ".join(terms[:3]), "role": title, "synthetic": True}
        )
    (directory / "queries.jsonl").write_text(
        "\n".join(json.dumps(r) for r in query_rows) + "\n", encoding="utf-8"
    )

    # Labels come from the rule "same role = relevant", with noise. The noise is
    # the point: a noiseless label set makes every arm look perfect and the
    # confidence intervals collapse, which would hide the very guards this
    # harness exists to demonstrate.
    labels = []
    by_role: dict[str, list[dict]] = {}
    for row in rows:
        by_role.setdefault(row["role"], []).append(row)

    for query in query_rows:
        same = by_role[query["role"]]
        others = [r for r in rows if r["role"] != query["role"]]
        for posting in rng.sample(same, min(6, len(same))):
            grade = rng.choices([3, 2, 1], weights=[0.5, 0.3, 0.2])[0]
            labels.append({"query_id": query["id"], "posting_id": posting["id"], "grade": grade})
        for posting in rng.sample(others, min(6, len(others))):
            grade = rng.choices([0, 1], weights=[0.85, 0.15])[0]
            labels.append({"query_id": query["id"], "posting_id": posting["id"], "grade": grade})

    (directory / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8"
    )

    print(
        f"Wrote {len(rows)} postings, {len(query_rows)} queries and {len(labels)} "
        f"SYNTHETIC labels to {directory}.\n\n"
        f"These numbers are not a result. The labels were generated from a rule a\n"
        f"ranker can rediscover, so every arm will score far better than it would\n"
        f"on real postings. Use this to check the pipeline runs, then replace it.\n\n"
        f"Next:\n"
        f"    python -m relevance.freeze --data {directory} --out {directory}/split.json\n"
        f"    python -m relevance.evaluate --data {directory} --split {directory}/split.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="write a synthetic corpus for a pipeline check")
    parser.add_argument("--out", default="data/demo")
    parser.add_argument("--postings", type=int, default=240)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    generate(pathlib.Path(args.out), postings=args.postings, queries=args.queries, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
