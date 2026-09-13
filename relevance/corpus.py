"""Postings, judgements, and the split that must be frozen before any tuning.

THE SPLIT IS THE WHOLE CREDIBILITY OF THE PROJECT. A held-out set chosen after
seeing results is not held out; it is a set that was searched until it agreed.
So the split is written to disk once, with the seed and a hash of the query set
recorded alongside it, and every later run reloads that file rather than
recomputing one. `load_split` refuses a file whose query hash no longer matches
the corpus -- which is what happens when queries are added and the "frozen"
split silently starts meaning something else.

ONE FURTHER RULE. The dev set is for choosing; the test set is scored ONCE, at
the end, and the number of configurations tried before that is recorded in the
report. Twenty configurations against a held-out set is twenty chances to find
noise that flatters one of them, and a report that does not say how many were
tried is not reporting the thing that matters.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
from dataclasses import dataclass, field

#: Relevance is graded 0-3, the scale the labelling guide in the README defines.
#: Graded rather than binary because "would apply" and "would read twice" are
#: genuinely different and nDCG can use the difference.
MIN_GRADE, MAX_GRADE = 0, 3


@dataclass(frozen=True)
class Posting:
    id: str
    title: str
    company: str
    location: str
    description: str

    def text(self) -> str:
        # Title first and repeated once. Not a tuning knob dressed up as
        # preprocessing: BM25 saturates term frequency, so repeating the title
        # raises its weight without letting a title term dominate the way a raw
        # multiplier would. It is a documented choice and it is in the ablation.
        return f"{self.title} {self.title} {self.company} {self.location} {self.description}"


@dataclass(frozen=True)
class Judgement:
    query_id: str
    posting_id: str
    grade: int

    def __post_init__(self) -> None:
        if not MIN_GRADE <= self.grade <= MAX_GRADE:
            raise ValueError(f"grade {self.grade} outside {MIN_GRADE}-{MAX_GRADE}")


@dataclass
class Corpus:
    postings: dict[str, Posting] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    #: query_id -> posting_id -> grade
    judgements: dict[str, dict[str, int]] = field(default_factory=dict)

    def graded_queries(self) -> list[str]:
        """Queries with at least one judged posting AND at least one relevant.

        A query where everything was judged irrelevant contributes 0.0 to every
        arm, which does not discriminate between them -- it only drags every
        mean down and narrows nothing. Excluding them is reported in the
        README rather than done quietly, because it does change the headline
        numbers.
        """
        return sorted(
            qid
            for qid, graded in self.judgements.items()
            if graded and any(g > 0 for g in graded.values())
        )

    def grades_for(self, query_id: str, ranked_ids: list[str]) -> list[int]:
        graded = self.judgements.get(query_id, {})
        # An unjudged posting scores 0. This is the standard assumption and it
        # is WRONG in a specific direction: an arm that surfaces a genuinely
        # good posting nobody labelled is punished for it. With a pooled label
        # set the bias is shared across arms, which is why pooling matters.
        return [graded.get(pid, 0) for pid in ranked_ids]

    def ideal_grades(self, query_id: str) -> list[int]:
        return sorted(self.judgements.get(query_id, {}).values(), reverse=True)

    @classmethod
    def load(cls, directory: str | pathlib.Path) -> "Corpus":
        base = pathlib.Path(directory)
        corpus = cls()

        postings_file = base / "postings.jsonl"
        if postings_file.exists():
            for line in postings_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                corpus.postings[row["id"]] = Posting(
                    id=row["id"],
                    title=row.get("title", ""),
                    company=row.get("company", ""),
                    location=row.get("location", ""),
                    description=row.get("description", ""),
                )

        queries_file = base / "queries.jsonl"
        if queries_file.exists():
            for line in queries_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                corpus.queries[row["id"]] = row["text"]

        labels_file = base / "labels.jsonl"
        if labels_file.exists():
            for line in labels_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                judgement = Judgement(row["query_id"], row["posting_id"], int(row["grade"]))
                corpus.judgements.setdefault(judgement.query_id, {})[
                    judgement.posting_id
                ] = judgement.grade

        return corpus

    def label_count(self) -> int:
        return sum(len(g) for g in self.judgements.values())


def query_fingerprint(query_ids: list[str]) -> str:
    joined = "\n".join(sorted(query_ids)).encode()
    return hashlib.sha256(joined).hexdigest()[:16]


@dataclass
class Split:
    dev: list[str]
    test: list[str]
    seed: int
    fingerprint: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "dev": self.dev,
                "test": self.test,
                "seed": self.seed,
                "fingerprint": self.fingerprint,
            },
            indent=2,
        )


class SplitMismatch(Exception):
    """The frozen split no longer describes this corpus.

    Raised rather than silently re-split, because silently re-splitting is
    exactly how a held-out set stops being held out.
    """


def freeze_split(
    corpus: Corpus, path: str | pathlib.Path, *, test_fraction: float = 0.4, seed: int = 20260913
) -> Split:
    """Write the split once. Refuses to overwrite an existing one."""
    target = pathlib.Path(path)
    if target.exists():
        raise SplitMismatch(
            f"{target} already exists. A split that gets rewritten is not frozen. "
            f"Delete it deliberately if the corpus genuinely changed, and say so in the README."
        )
    queries = corpus.graded_queries()
    shuffled = list(queries)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_fraction))
    split = Split(
        dev=sorted(shuffled[:cut]),
        test=sorted(shuffled[cut:]),
        seed=seed,
        fingerprint=query_fingerprint(queries),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(split.to_json(), encoding="utf-8")
    return split


def load_split(corpus: Corpus, path: str | pathlib.Path) -> Split:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    split = Split(raw["dev"], raw["test"], raw["seed"], raw["fingerprint"])
    current = query_fingerprint(corpus.graded_queries())
    if current != split.fingerprint:
        raise SplitMismatch(
            f"the corpus fingerprint is {current} but the frozen split was built "
            f"against {split.fingerprint}. Queries were added or removed, so this "
            f"split no longer partitions what is being evaluated. Re-freeze it and "
            f"report every result since the change as dev-set only."
        )
    return split
