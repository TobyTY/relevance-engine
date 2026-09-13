# relevance-engine

Four ranking arms compared on the same labelled set, with an evaluation harness
built to make overclaiming difficult.

The ranker is the ordinary half. The harness is the project: a frozen split it
refuses to re-cut, paired bootstrap intervals on the *difference* between arms,
a count of configurations tried printed above every table, and an explicit
"inconclusive at this label count" verdict when the interval spans zero.

40 tests. Every metric is checked against a value worked out by hand, not
against a second implementation of the same formula.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
set PYTHONPATH=%CD%

.venv\Scripts\python.exe -m pytest -q                 # 40 tests, <1s
.venv\Scripts\python.exe -m relevance.demo            # synthetic corpus, pipeline check
.venv\Scripts\python.exe -m relevance.freeze   --data data/demo --out data/demo/split.json
.venv\Scripts\python.exe -m relevance.evaluate --data data/demo --split data/demo/split.json
```

---

## Status

**The harness is complete and tested. The label set is not collected yet**, so
there is no result to report, and this README does not report one.

That ordering is deliberate. Building the evaluation before the labels means the
labels cannot be quietly shaped to suit a result, and the harness can be
demonstrated on synthetic data — where every number is meaningless and marked as
such — without waiting on a half-day of judgement work.

---

## The arms

| arm | what it is |
|---|---|
| `keyword` | the incumbent heuristic: count query terms present, tie-break on title hits |
| `bm25` | hand-rolled BM25, k1=1.2, b=0.75 |
| `dense` | sentence-transformer embeddings, cosine similarity (optional dependency) |
| `hybrid` | reciprocal rank fusion of `bm25` and `dense` |

**The incumbent is included on purpose.** A project that omits the thing it
claims to improve on is comparing against a strawman it did not have to name.

**BM25 is written out rather than imported.** Not for speed — the claim is that
a dense retriever was raced against a *strong* lexical baseline, and "strong" is
doing real work in that sentence. The usual way an embedding arm wins a
portfolio project is that BM25 was crippled: no length normalisation, a stopword
list that ate half the query, or an unfloored IDF that ranks documents
containing a common term *below* documents that lack it. Each of those has a
test here.

**The tokenizer keeps `+`, `#` and `.`** so `c++`, `c#`, `.net` and `node.js`
survive as single tokens, and the stopword list is deliberately tiny — `go`,
`r` and `c` are stopwords in most lists and job titles in this corpus.

**Fusion is RRF, not a weighted score sum.** BM25 scores are unbounded and
corpus-dependent; cosine similarities sit in [-1, 1] and bunch near the top.
Adding them requires a normalisation, and every choice of normalisation is a
hyperparameter fitted on the same small label set the result is measured on. RRF
uses only ranks: one parameter, conventional value, nothing to fit.

---

## Why the harness is the project

### The split is frozen, and the code refuses to unfreeze it

A held-out set chosen after seeing results is not held out — it is a set that
was searched until it agreed. So `freeze.py` writes the split once, records the
seed and a hash of the query set, and **refuses to overwrite an existing file**.
`load_split` refuses a split whose fingerprint no longer matches the corpus,
which is what happens when queries are added and a "frozen" split silently
starts meaning something else.

Both refusals have tests.

### Differences get a paired interval, not two separate ones

Reporting two confidence intervals and inviting the reader to check whether they
overlap is the wrong test twice over. Overlapping intervals can still correspond
to a significant difference, and both arms are scored on the *same* queries — a
pairing an unpaired comparison throws away.

The bootstrap resamples **queries**, so both arms are always scored on the same
resampled set. Bootstrap rather than a t-test because per-query nDCG is bounded,
skewed, and usually has a spike at zero; a t-interval assumes none of that.

### It says "inconclusive" out loud

When the difference interval spans zero, the table prints `inconclusive` rather
than a winner, and closes with:

> an interval spanning zero means this label set cannot separate the two arms.
> That is a result, not a missing result, and the honest next step is more
> labels rather than another model.

Below 20 queries the whole table is stamped **UNDERPOWERED**.

### It counts how many times you looked

`--configurations-tried N` prints above the table. Twenty attempts at a held-out
set is twenty chances for noise to flatter one of them, and a report that does
not say how many were tried is omitting the thing that decides how much to
believe it.

### A skipped arm is named, never silently dropped

If `sentence-transformers` is not installed, the table prints:

```
Arms not run:
  dense and hybrid: pip install sentence-transformers to run the dense arm
```

An ablation table missing an arm without explanation reads as an arm that lost.

---

## Labelling

```bash
.venv\Scripts\python.exe -m relevance.label --data data --target 250
```

One keystroke per judgement, append-only, resumable — 250 judgements is two
sittings, and a tool that loses work on close produces 40 labels and a project
that quietly drops its evaluation.

**The candidates are pooled across arms.** If you label BM25's top 10 and then
evaluate a dense arm, every good posting the dense arm found that BM25 missed is
unjudged, and unjudged scores 0 — so the dense arm is punished for finding
things the labelling never looked at. The pool is the union of each arm's top
`--depth`, which shares that bias instead of aiming it at one arm.

### The scale

| grade | meaning |
|---|---|
| 0 | irrelevant — wrong field, wrong seniority, or not a job you could hold |
| 1 | marginal — adjacent, you would skim it and move on |
| 2 | relevant — you would read it properly and might apply |
| 3 | strongly relevant — you would apply |

**Judge the posting against the query, not against your job hunt.** "Would I
apply" drifts with mood and time of day; "does this posting answer this query"
does not. Drift is the failure mode that quietly destroys a label set, and it is
invisible afterwards.

---

## Data format

Three JSONL files in `data/`. The corpus is yours to supply; nothing is bundled.

```jsonc
// postings.jsonl
{"id": "p0001", "title": "Backend Engineer", "company": "Northwind",
 "location": "Pune", "description": "python fastapi postgres ..."}

// queries.jsonl
{"id": "q001", "text": "python backend fresher"}

// labels.jsonl   -- written by relevance.label
{"query_id": "q001", "posting_id": "p0001", "grade": 3}
```

`relevance.demo` writes a synthetic set in this format. Every synthetic row is
stamped `"synthetic": true` and the labels come from a rule a ranker can
rediscover, so **the demo table is not a result** — it exists to prove the
pipeline runs and to show the format.

---

## Known limitations

- **Unjudged postings score 0.** The standard assumption, and wrong in a
  specific direction: an arm that surfaces a genuinely good posting nobody
  labelled is punished for it. Pooling shares the bias across arms; it does not
  remove it.
- **Queries with no relevant judgement are excluded** from scoring. They
  contribute 0.0 to every arm, so they cannot discriminate — they only drag
  every mean down. Stated here because it changes the headline numbers.
- **If the dense arm is unavailable during labelling, the pool is lexical-only**,
  which biases it toward exactly the matches the dense arm is being tested for.
- **No LLM re-ranking arm.** It would need its cost-per-query and latency
  reported beside the free baselines to be an honest comparison, and that is a
  measurement exercise of its own rather than a fifth row in this table.
- **One labeller.** With a single annotator there is no inter-annotator
  agreement figure, so the label set's own reliability is unmeasured. A second
  pass over a 30-query sample, some weeks later, would at least give a
  self-agreement number.

---

## Layout

```
relevance/corpus.py    postings, judgements, the frozen split and its guards
relevance/bm25.py      BM25 with the parameters explained, not just set
relevance/arms.py      the four arms behind one interface
relevance/metrics.py   p@k, nDCG, MRR, and the paired bootstrap
relevance/evaluate.py  the ablation table, and everything it refuses to claim
relevance/label.py     pooled, resumable, one keystroke per judgement
relevance/freeze.py    writes the split once, deliberately, with a date on it
relevance/demo.py      synthetic corpus so the pipeline can be run today
tests/                 40 tests; every metric against a hand-computed value
```
