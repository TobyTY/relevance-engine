"""BM25, written out rather than imported.

WHY HAND-ROLL IT. Not for performance -- `rank_bm25` is fine. Because the
project's claim is that a dense retriever was compared fairly against a strong
lexical baseline, and "strong" is doing real work in that sentence. A baseline
whose internals you cannot describe is a baseline you cannot defend, and the
most common way an embedding arm "wins" a portfolio project is that it was
raced against a weak BM25: no length normalisation, a stopword list that ate
half the query, or k1/b left at values that suit newswire and not job postings.

THE TWO PARAMETERS, AND WHAT THEY DO HERE.

  k1 controls term-frequency saturation. At k1=1.2 the fifth occurrence of
  "Python" in a posting adds very little over the fourth, which is right: a job
  ad repeating a skill five times is not five times more about it.

  b controls length normalisation, 0 = none, 1 = full. Job postings vary wildly
  in length for reasons unrelated to relevance -- one company's legal boilerplate
  is another's entire ad -- so b is left high. b=0 would hand every match to the
  longest posting in the corpus.

Defaults are k1=1.2, b=0.75, the standard values. They were NOT tuned on the
test set, and the count of configurations tried is recorded in the report.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

TOKEN = re.compile(r"[a-z0-9+#.]+")

#: Deliberately tiny. A big stopword list is a tuning knob wearing a disguise:
#: dropping "c" or "go" or "r" from a query silently removes the only term that
#: mattered, and on job postings those are language names.
STOPWORDS = frozenset(
    "a an and are as at be by for from has in is it its of on or that the to was were will with".split()
)


def tokenize(text: str) -> list[str]:
    # `+#.` survive the split so c++, c#, node.js and .net stay single tokens.
    # Stripping them is the standard tokenizer bug that makes a search engine
    # unable to tell C from C++ on a corpus where that distinction is the job.
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class BM25:
    k1: float = 1.2
    b: float = 0.75

    _docs: dict[str, list[str]] = field(default_factory=dict)
    _freqs: dict[str, Counter] = field(default_factory=dict)
    _doc_freq: Counter = field(default_factory=Counter)
    _avg_len: float = 0.0

    def index(self, documents: dict[str, str]) -> "BM25":
        self._docs = {doc_id: tokenize(text) for doc_id, text in documents.items()}
        self._freqs = {doc_id: Counter(tokens) for doc_id, tokens in self._docs.items()}
        self._doc_freq = Counter()
        for tokens in self._docs.values():
            self._doc_freq.update(set(tokens))
        lengths = [len(t) for t in self._docs.values()]
        self._avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        return self

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._doc_freq.get(term, 0)
        # The +0.5 smoothing form, floored at zero. Unfloored, a term appearing
        # in more than half the corpus gets a NEGATIVE idf, so a document
        # containing it scores lower than one that does not -- which is how a
        # query for "engineer" on a corpus of engineering jobs ranks the least
        # relevant postings first.
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def score(self, query: str, doc_id: str) -> float:
        tokens = self._freqs.get(doc_id)
        if tokens is None:
            return 0.0
        length = len(self._docs[doc_id])
        norm = self.k1 * (1 - self.b + self.b * (length / self._avg_len if self._avg_len else 1))
        total = 0.0
        for term in tokenize(query):
            tf = tokens.get(term, 0)
            if not tf:
                continue
            total += self._idf(term) * (tf * (self.k1 + 1)) / (tf + norm)
        return total

    def rank(self, query: str, *, limit: int | None = None) -> list[tuple[str, float]]:
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self._docs]
        # Ties break on document id, not on insertion order. Insertion order is
        # whatever the scraper happened to write, so an arm could win or lose on
        # it -- a difference that would not survive re-running the scraper.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        scored = [pair for pair in scored if pair[1] > 0]
        return scored[:limit] if limit else scored
