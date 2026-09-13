"""The four ranking arms, each behind the same interface so the comparison is fair.

An arm takes a query and returns posting ids in rank order. Nothing else differs
between them -- same corpus, same queries, same cutoffs, same judgements -- so a
difference in score is a difference in ranking and not in harness.

THE FOUR:

  keyword     the heuristic being replaced. It has to be here, or the project
              claims an improvement over nothing.
  bm25        the lexical baseline, hand-rolled so it can be defended.
  dense       sentence-transformer embeddings, cosine similarity.
  hybrid      reciprocal rank fusion of bm25 and dense.

WHY RRF AND NOT A WEIGHTED SCORE SUM. BM25 scores are unbounded and
corpus-dependent; cosine similarities sit in [-1, 1] and bunch tightly near the
top. Adding them needs a normalisation, and every choice of normalisation is a
hyperparameter fitted on the same small label set the result is measured on.
RRF uses only the RANKS, so it has one parameter (k) with a conventional value
and no scale to fit. It is the honest fusion when the label set is too small to
afford another fitted knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from relevance.bm25 import BM25, tokenize
from relevance.corpus import Corpus


class Arm(Protocol):
    name: str

    def rank(self, query: str, *, limit: int) -> list[str]: ...


@dataclass
class KeywordHeuristic:
    """The incumbent: count query terms present, tie-break on title hits.

    Included because "the new ranker beats the old one" is the claim, and a
    project that omits the incumbent is comparing against a strawman it did not
    have to name.
    """

    corpus: Corpus
    name: str = "keyword"

    def rank(self, query: str, *, limit: int) -> list[str]:
        terms = set(tokenize(query))
        scored = []
        for posting in self.corpus.postings.values():
            body = set(tokenize(posting.description)) | set(tokenize(posting.company))
            title = set(tokenize(posting.title))
            hits = len(terms & (body | title))
            if hits:
                scored.append((posting.id, hits, len(terms & title)))
        scored.sort(key=lambda row: (-row[1], -row[2], row[0]))
        return [row[0] for row in scored[:limit]]


@dataclass
class BM25Arm:
    corpus: Corpus
    k1: float = 1.2
    b: float = 0.75
    name: str = "bm25"

    def __post_init__(self) -> None:
        self._index = BM25(k1=self.k1, b=self.b).index(
            {p.id: p.text() for p in self.corpus.postings.values()}
        )

    def rank(self, query: str, *, limit: int) -> list[str]:
        return [doc_id for doc_id, _ in self._index.rank(query, limit=limit)]


class DenseUnavailable(Exception):
    """sentence-transformers is not installed.

    Raised rather than silently skipped: an ablation table missing an arm
    without saying why reads as an arm that lost.
    """


@dataclass
class DenseArm:
    corpus: Corpus
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "dense"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseUnavailable(
                "pip install sentence-transformers to run the dense arm"
            ) from exc
        import numpy as np

        self._np = np
        self._model = SentenceTransformer(self.model_name)
        self._ids = list(self.corpus.postings)
        texts = [self.corpus.postings[i].text() for i in self._ids]
        # Normalised at encode time so cosine similarity is a plain dot product
        # and the ranking cannot drift on a vector-norm bug.
        self._matrix = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def rank(self, query: str, *, limit: int) -> list[str]:
        vector = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._matrix @ vector
        order = self._np.argsort(-scores)[:limit]
        return [self._ids[i] for i in order]


#: The conventional RRF constant. It flattens the contribution of the very top
#: ranks so one arm placing something first does not automatically win the
#: fusion. Left at 60 rather than tuned, because tuning it on this label set is
#: the extra fitted knob RRF was chosen to avoid.
RRF_K = 60


@dataclass
class HybridRRF:
    arms: list[Arm]
    k: int = RRF_K
    name: str = "hybrid"
    #: How deep to fuse. Shallower than the final cutoff on purpose: fusing the
    #: full ranking lets an arm's long tail outvote the other arm's head.
    depth: int = 100

    def rank(self, query: str, *, limit: int) -> list[str]:
        points: dict[str, float] = {}
        for arm in self.arms:
            for rank, doc_id in enumerate(arm.rank(query, limit=self.depth), start=1):
                points[doc_id] = points.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        ordered = sorted(points.items(), key=lambda pair: (-pair[1], pair[0]))
        return [doc_id for doc_id, _ in ordered[:limit]]
