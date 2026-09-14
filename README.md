# relevance-engine

Four ranking arms compared on the same labelled set, with an evaluation harness
built to make overclaiming difficult.

The ranker is the ordinary half. The harness is the project: a frozen split it
refuses to re-cut, paired bootstrap intervals on the *difference* between arms,
a count of configurations tried printed above every table, and an explicit
"inconclusive at this label count" verdict when the interval spans zero.

71 tests. Every metric is checked against a value worked out by hand, not
against a second implementation of the same formula.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
set PYTHONPATH=%CD%

.venv\Scripts\python.exe -m pytest -q                 # 71 tests, <1s
.venv\Scripts\python.exe -m relevance.demo            # synthetic corpus, pipeline check
.venv\Scripts\python.exe -m relevance.freeze   --data data/demo --out data/demo/split.json
.venv\Scripts\python.exe -m relevance.evaluate --data data/demo --split data/demo/split.json
```

---

## Status

Harness complete and tested. Corpus collected — 444 real postings. **241 labels
exist and every one of them was produced by a language model, not by a person.**

That is stated first because it bounds everything below it. The table in the
next section is a check that the pipeline runs end to end on real data. It is
not a measurement of relevance, and the code says so itself: the evaluator
prints a banner above any table built from model-judged labels, and
`label_provenance()` counts the sources rather than assuming one.

**The held-out test split has not been scored and will not be** until human
labels exist. It can only be spent once, and spending it on labels that cannot
support a claim would waste the one guard the project is built around.

---

## The dev table, and why it is not a result

25 graded queries, 15 in dev. Full output in [`results-dev.txt`](results-dev.txt).

| arm | p@5 | nDCG@10 | MRR | vs bm25 (nDCG@10) |
|---|---|---|---|---|
| `keyword` | 0.787 | 0.705 | 0.889 | −0.046 [−0.137, +0.041] **inconclusive** |
| `bm25` | 0.787 | 0.751 | 0.947 | baseline |
| `dense` | 0.253 | 0.387 | 0.600 | −0.364 [−0.498, −0.227] **worse** |
| `hybrid` | 0.493 | 0.643 | 0.880 | −0.108 [−0.212, −0.014] **worse** |

Read literally, this is the negative result the project was designed to be able
to report: the dense retriever loses badly to plain BM25, the hybrid loses too,
and the incumbent keyword heuristic is statistically indistinguishable from BM25
on every metric.

**Do not read it literally.** Three reasons, in order of how much they matter:

1. **The judge and the winner share a signal.** The model judged each posting
   largely from its title and the terms in its text, because that is what a
   short snippet exposes. That is the same signal BM25 ranks on. A lexical arm
   winning on these labels is the single most likely outcome of this labelling
   procedure regardless of which arm is actually more useful, and nothing in the
   data can separate the two explanations.
2. **Agreement with human judgement is unmeasured.** There is no human-labelled
   sample, so the label set has no measured reliability at all — not a low one,
   an unknown one.
3. **15 dev queries is below the 20 this harness treats as its inference
   floor**, so every interval is stamped `underpowered` and the table says so.

The honest summary is that the pipeline works on real data and produced a
plausible-looking answer that it cannot yet defend. Fixing that needs human
labels, not a better model — which is the thing the harness was built to be able
to say.

### What would make this a result

A human-labelled sample of the same query-posting pairs, judged blind, scored
for agreement against the model's grades. Two commands:

```bash
.venv\Scripts\python.exe -m relevance.label --data data --calibrate 30
.venv\Scripts\python.exe -m relevance.agreement --data data
```

The sample is **stratified across the model's grades, not uniform**. A pooled
label set is mostly 0s, so a uniform sample would be almost entirely irrelevant
pairs; two raters would agree on nearly all of them for no better reason than
that the corpus is mostly irrelevant, and the figure would be flattering and
meaningless. Sampling evenly puts the sample where disagreements can show up.

The model's grade is **not displayed**. An anchored judgement is not an
independent one, and comparing a judgement against the number it was anchored to
measures nothing.

Agreement is reported as **quadratic-weighted Cohen's kappa** with a bootstrap
interval. Quadratic because the scale is ordinal — calling a 3 a 2 is a small
disagreement and calling it a 0 is a large one, and plain accuracy scores both
as simply wrong. Kappa rather than accuracy because it corrects for chance,
which matters more here than usual: two raters who both answer 0 most of the
time agree constantly without either knowing anything. There is a test that
builds exactly that case — 85% raw agreement, kappa under 0.1.

The verdict is mechanical. Below 0.41 the table is withdrawn rather than
qualified. Between 0.41 and 0.61 it stands with the figure quoted beside it and
no conclusion may rest on a margin smaller than the gap. Above 0.61 the label
set carries a measured reliability and kappa goes in this README next to the
ablation table.

That is the only thing standing between this and a defensible finding.

---

## Getting a corpus

```bash
.venv\Scripts\python.exe -m relevance.fetch --role-filter software --pages 12 --refresh
.venv\Scripts\python.exe -m relevance.queries
.venv\Scripts\python.exe -m relevance.label --data data --target 250
```

Two public, documented JSON APIs — [Remotive](https://remotive.com) and
[Arbeitnow](https://www.arbeitnow.com) — rather than HTML scraping. Every large
job board forbids scraping in its terms, rate-limits hard, and changes its markup
often enough that a scraper becomes a maintenance job. "I read their terms and
used the interface they published" is a better answer than "it worked until they
renamed a CSS class".

Both publishers ask for attribution and light use, so every posting keeps its
`url` and `source`, responses are cached to disk, and **the network is only
touched with an explicit `--refresh`**. The cache is the rate limiter, not a
speed-up — a cache that is merely a speed-up gets bypassed the moment someone is
impatient, and there is a test asserting a cached read cannot reach the network.
`data/` is gitignored, so nothing is republished.

A current pull gives **444 postings from 1516 fetched** after deduplication
(on title+company, since the same job is cross-posted to both boards under
different ids) and a coarse role filter.

### The corpus is Europe-heavy, and that is a real limitation

Arbeitnow is a German board and Remotive is remote-first, so the postings skew
to Germany and remote-EU rather than India. It does not invalidate the
evaluation — the labelling rule is *"does this posting answer this query"*, not
*"would I take this job"* — but it does mean this corpus measures a ranker, not a
job hunt. An India-focused source would need either a paid API or scraping a
board whose terms forbid it, and neither is worth doing quietly.

**Relevance is deliberately not filtered at fetch time.** Keeping only postings
that match the queries would shrink the corpus and make every arm score better,
and it would make the evaluation meaningless: a corpus pre-filtered to relevant
documents cannot measure whether a ranker finds relevant documents.

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
relevance/fetch.py     two public job APIs, cached, attributed, rate-limited
relevance/queries.py   the starter query set, including unanswerable ones
relevance/agreement.py weighted kappa between the model's labels and yours
relevance/demo.py      synthetic corpus so the pipeline can be run today
results-dev.txt        the dev table, banner and all
tests/                 71 tests; every metric against a hand-computed value
```
