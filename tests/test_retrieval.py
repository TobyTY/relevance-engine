"""BM25, fusion, and the split guards.

The BM25 tests check properties rather than exact scores: saturation, length
normalisation, non-negative idf, and a tokenizer that does not destroy the terms
this corpus is actually about. Exact scores would pin the test to one parameter
setting and break the moment k1 is tuned, which is a test that gets deleted
rather than fixed.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from relevance.arms import BM25Arm, HybridRRF, KeywordHeuristic
from relevance.bm25 import BM25, tokenize
from relevance.corpus import Corpus, Posting, SplitMismatch, freeze_split, load_split


# ------------------------------------------------------------------ tokenizer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("C++ developer", ["c++", "developer"]),
        ("C# and .NET", ["c#", ".net"]),
        ("Node.js", ["node.js"]),
        ("Python 3.12", ["python", "3.12"]),
    ],
)
def test_the_tokenizer_keeps_language_names_intact(text, expected):
    """Stripping +, # and . is the standard tokenizer bug that leaves a job
    search unable to tell C from C++ on a corpus where that is the job."""
    assert tokenize(text) == expected


def test_short_stopword_list_keeps_language_names():
    """"go", "r" and "c" are stopwords in many lists and job titles here."""
    assert tokenize("go and r and c") == ["go", "r", "c"]


# ---------------------------------------------------------------------- bm25


def index_of(**docs) -> BM25:
    return BM25().index(docs)


def test_term_frequency_saturates():
    """The fifth mention of a term must add less than the second. A job ad
    repeating a skill five times is not five times more about it."""
    index = index_of(
        two="python python filler filler filler filler filler filler",
        five="python python python python python filler filler filler",
    )
    two, five = index.score("python", "two"), index.score("python", "five")
    assert five > two
    assert five < two * 2.5


def test_a_longer_document_is_penalised_for_the_same_hit_count():
    index = index_of(
        short="python engineer",
        long="python engineer " + "boilerplate " * 60,
    )
    assert index.score("python", "short") > index.score("python", "long")


def test_idf_is_never_negative():
    """A term in more than half the corpus gets a negative idf under the
    unfloored formula, so documents containing it would rank BELOW documents
    that do not -- a query for "engineer" on a corpus of engineering jobs would
    return the least relevant postings first."""
    index = index_of(**{f"d{i}": "engineer python" for i in range(10)}, odd="designer")
    assert index.score("engineer", "d0") >= 0.0


def test_a_document_without_the_term_scores_zero():
    index = index_of(a="python", b="java")
    assert index.score("python", "b") == 0.0


def test_ranking_drops_zero_scoring_documents():
    index = index_of(a="python engineer", b="pastry chef")
    assert [doc for doc, _ in index.rank("python")] == ["a"]


def test_ties_break_on_document_id_not_insertion_order():
    """Insertion order is whatever the scraper happened to write, so an arm
    could win or lose on it -- a difference that would not survive re-running
    the scraper."""
    forward = index_of(zebra="python", alpha="python").rank("python")
    backward = index_of(alpha="python", zebra="python").rank("python")
    assert [d for d, _ in forward] == [d for d, _ in backward] == ["alpha", "zebra"]


# ------------------------------------------------------------------- fusion


class FakeArm:
    def __init__(self, name, order):
        self.name = name
        self._order = order

    def rank(self, query, *, limit):
        return self._order[:limit]


def test_rrf_promotes_what_both_arms_like():
    """`c` is second on both lists and never first on either. Under RRF it wins,
    because two second places outscore one first place plus one absence -- which
    is the property that makes fusion worth doing at all."""
    a = FakeArm("a", ["x", "c", "p", "q"])
    b = FakeArm("b", ["y", "c", "r", "s"])
    assert HybridRRF([a, b]).rank("anything", limit=1) == ["c"]


def test_rrf_needs_no_score_normalisation():
    """Only ranks are used, so an arm with huge raw scores cannot dominate.
    This is why RRF was chosen over a weighted sum on a small label set: a
    normalisation would be one more knob fitted on the data being measured."""
    a = FakeArm("a", ["p", "q"])
    b = FakeArm("b", ["q", "p"])
    fused = HybridRRF([a, b]).rank("anything", limit=2)
    assert sorted(fused) == ["p", "q"]


# -------------------------------------------------------------------- corpus


def tiny_corpus() -> Corpus:
    corpus = Corpus()
    for i in range(6):
        corpus.postings[f"p{i}"] = Posting(f"p{i}", f"Engineer {i}", "Co", "Patiala", "python sql")
    for i in range(6):
        corpus.queries[f"q{i}"] = "python engineer"
        corpus.judgements[f"q{i}"] = {f"p{i}": 3, f"p{(i + 1) % 6}": 0}
    return corpus


def test_unjudged_postings_score_zero():
    corpus = tiny_corpus()
    assert corpus.grades_for("q0", ["p0", "p5", "nonexistent"]) == [3, 0, 0]


def test_queries_with_nothing_relevant_are_excluded():
    """They contribute 0.0 to every arm, so they cannot discriminate between
    them -- they only drag every mean down."""
    corpus = tiny_corpus()
    corpus.judgements["q_blank"] = {"p0": 0, "p1": 0}
    assert "q_blank" not in corpus.graded_queries()


def test_freezing_twice_is_refused(tmp_path: pathlib.Path):
    corpus = tiny_corpus()
    target = tmp_path / "split.json"
    freeze_split(corpus, target)
    with pytest.raises(SplitMismatch):
        freeze_split(corpus, target)


def test_a_split_is_reproducible_from_its_seed(tmp_path: pathlib.Path):
    corpus = tiny_corpus()
    first = freeze_split(corpus, tmp_path / "a.json", seed=42)
    second = freeze_split(corpus, tmp_path / "b.json", seed=42)
    assert first.dev == second.dev and first.test == second.test


def test_dev_and_test_do_not_overlap(tmp_path: pathlib.Path):
    split = freeze_split(tiny_corpus(), tmp_path / "split.json")
    assert not set(split.dev) & set(split.test)


def test_loading_a_split_after_the_queries_changed_is_refused(tmp_path: pathlib.Path):
    """Adding queries and reusing the old split is how a held-out set silently
    stops being held out. It has to fail loudly."""
    corpus = tiny_corpus()
    target = tmp_path / "split.json"
    freeze_split(corpus, target)
    corpus.queries["q_new"] = "sql analyst"
    corpus.judgements["q_new"] = {"p0": 3}
    with pytest.raises(SplitMismatch):
        load_split(corpus, target)


def test_a_split_loads_when_the_corpus_is_unchanged(tmp_path: pathlib.Path):
    corpus = tiny_corpus()
    target = tmp_path / "split.json"
    frozen = freeze_split(corpus, target)
    assert load_split(corpus, target).dev == frozen.dev


def test_corpus_round_trips_through_jsonl(tmp_path: pathlib.Path):
    (tmp_path / "postings.jsonl").write_text(
        json.dumps({"id": "p1", "title": "SDE", "company": "X", "location": "Y", "description": "python"})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"id": "q1", "text": "python"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "labels.jsonl").write_text(
        json.dumps({"query_id": "q1", "posting_id": "p1", "grade": 3}) + "\n", encoding="utf-8"
    )
    corpus = Corpus.load(tmp_path)
    assert corpus.label_count() == 1
    assert corpus.graded_queries() == ["q1"]


def test_a_grade_outside_the_scale_is_refused():
    from relevance.corpus import Judgement

    with pytest.raises(ValueError):
        Judgement("q", "p", 4)


# ----------------------------------------------------------------------- arms


def test_both_lexical_arms_return_something_on_a_real_query():
    corpus = tiny_corpus()
    for arm in (KeywordHeuristic(corpus), BM25Arm(corpus)):
        assert arm.rank("python engineer", limit=5), arm.name


def test_an_arm_respects_the_limit():
    corpus = tiny_corpus()
    assert len(BM25Arm(corpus).rank("python", limit=3)) <= 3
