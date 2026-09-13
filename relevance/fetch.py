"""Build a posting corpus from public job APIs.

WHY APIs AND NOT HTML SCRAPING. Every large job board forbids scraping in its
terms, rate-limits aggressively, and changes its markup often enough that a
scraper is a maintenance job rather than a project. Both sources here publish a
documented JSON API intended for exactly this, so the corpus is collected the
way the publisher asked for it. That is worth more in an interview than a
cleverer parser: "I read their terms and used the interface they offered" is a
better answer than "it worked until they changed a class name".

WHAT EACH SOURCE ASKS FOR, and what this module does about it:

  Remotive    Attribution and a link back to their URL. No republishing to
              third-party job boards. No more than about four calls a day.
              So: `source` and `url` are kept on every posting, responses are
              cached to disk and reused, and `--refresh` is required to go back
              to the network. The corpus lives in `data/`, which is gitignored,
              so nothing is republished anywhere.

  Arbeitnow   "Please do not abuse. I would appreciate linking back." Same
              treatment: cached, attributed, paginated politely.

THE CACHE IS THE RATE LIMITER. A cache that is merely a speed-up gets bypassed
the moment someone is impatient. Here the network call happens only with an
explicit `--refresh`, so the default path cannot hammer anyone by accident.

RELEVANCE IS NOT FILTERED AT FETCH TIME. It is tempting to keep only postings
matching the queries -- it would make the corpus smaller and every arm score
better. It would also make the evaluation meaningless: a corpus pre-filtered to
relevant documents cannot measure whether a ranker finds relevant documents.
The `--role-filter` option exists for narrowing to an industry, and it is
deliberately coarse.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html.parser
import json
import pathlib
import re
import time
import urllib.error
import urllib.request

USER_AGENT = "relevance-engine/0.1 (portfolio project; contact via github.com/TobyTY)"
CACHE_DIR = pathlib.Path("data/.cache")

#: Polite floor between requests. Neither source demands a specific interval;
#: both ask not to be hammered, and one second costs nothing on a job that runs
#: a handful of times.
MIN_INTERVAL_SECONDS = 1.0
_last_request = 0.0


class _Stripper(html.parser.HTMLParser):
    """Turn a posting's HTML body into plain text.

    A regex over tags is the usual shortcut and it mangles exactly the postings
    that matter -- the ones with nested markup listing the skills. The stdlib
    parser is three lines more and does not guess.
    """

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("p", "br", "li", "div", "h1", "h2", "h3", "h4"):
            self._parts.append(" ")

    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._parts)).strip()


def strip_html(raw: str) -> str:
    parser = _Stripper()
    parser.feed(raw or "")
    return parser.text()


def _get(url: str, *, refresh: bool) -> dict:
    """Fetch with an on-disk cache keyed by URL."""
    global _last_request
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    cached = CACHE_DIR / f"{key}.json"

    if cached.exists() and not refresh:
        return json.loads(cached.read_text(encoding="utf-8"))

    elapsed = time.monotonic() - _last_request
    if elapsed < MIN_INTERVAL_SECONDS:
        time.sleep(MIN_INTERVAL_SECONDS - elapsed)
    _last_request = time.monotonic()

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    cached.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def from_remotive(*, limit: int, refresh: bool) -> list[dict]:
    payload = _get(f"https://remotive.com/api/remote-jobs?limit={limit}", refresh=refresh)
    rows = []
    for job in payload.get("jobs", []):
        rows.append(
            {
                "id": f"remotive-{job['id']}",
                "title": job.get("title", ""),
                "company": job.get("company_name", ""),
                "location": job.get("candidate_required_location", "") or "Remote",
                "description": strip_html(job.get("description", "")),
                "url": job.get("url", ""),
                "source": "Remotive (remotive.com)",
                "tags": job.get("tags", []),
                "fetched_at": time.strftime("%Y-%m-%d"),
            }
        )
    return rows


def from_arbeitnow(*, pages: int, refresh: bool) -> list[dict]:
    rows = []
    for page in range(1, pages + 1):
        payload = _get(
            f"https://www.arbeitnow.com/api/job-board-api?page={page}", refresh=refresh
        )
        batch = payload.get("data", [])
        if not batch:
            break
        for job in batch:
            tags = job.get("tags", [])
            if isinstance(tags, str):
                # The API returns this field as a stringified Python list.
                # literal_eval rather than eval: the value comes off the
                # network, and eval on network input is how a data loader
                # becomes a remote shell.
                try:
                    tags = ast.literal_eval(tags)
                except (ValueError, SyntaxError):
                    tags = []
            rows.append(
                {
                    "id": f"arbeitnow-{job['slug']}",
                    "title": job.get("title", ""),
                    "company": job.get("company_name", ""),
                    "location": job.get("location", "") or "Remote",
                    "description": strip_html(job.get("description", "")),
                    "url": job.get("url", ""),
                    "source": "Arbeitnow (arbeitnow.com)",
                    "tags": tags,
                    "fetched_at": time.strftime("%Y-%m-%d"),
                }
            )
    return rows


SOURCES = {"remotive": from_remotive, "arbeitnow": from_arbeitnow}

#: Coarse on purpose. Narrowing the corpus to the postings a query should match
#: would make every arm look good and measure nothing.
ROLE_TERMS = {
    "software": "engineer developer software backend frontend fullstack sdet sre devops data platform",
    "finance": "quant trading risk fintech payments banking analyst treasury settlement",
    "embedded": "embedded firmware rtos microcontroller hardware can bus automotive",
}


def matches_role(row: dict, terms: set[str]) -> bool:
    blob = f"{row['title']} {' '.join(row.get('tags') or [])}".lower()
    return any(term in blob for term in terms)


def main() -> int:
    parser = argparse.ArgumentParser(description="fetch a posting corpus from public job APIs")
    parser.add_argument("--out", default="data", help="directory to write postings.jsonl into")
    parser.add_argument(
        "--source", action="append", choices=sorted(SOURCES), help="repeatable; default is both"
    )
    parser.add_argument("--limit", type=int, default=400, help="Remotive: postings to request")
    parser.add_argument("--pages", type=int, default=3, help="Arbeitnow: pages of 250")
    parser.add_argument(
        "--role-filter",
        choices=sorted(ROLE_TERMS),
        help="keep only postings whose title or tags match this broad family",
    )
    parser.add_argument(
        "--min-description",
        type=int,
        default=200,
        help="drop postings shorter than this many characters",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="go to the network instead of the on-disk cache. Both sources ask "
        "not to be called often; without this flag nothing leaves the machine.",
    )
    args = parser.parse_args()

    chosen = args.source or sorted(SOURCES)
    rows: list[dict] = []
    for name in chosen:
        try:
            if name == "remotive":
                batch = from_remotive(limit=args.limit, refresh=args.refresh)
            else:
                batch = from_arbeitnow(pages=args.pages, refresh=args.refresh)
        except (urllib.error.URLError, TimeoutError) as exc:
            # Named, not swallowed. A corpus silently built from one source when
            # two were asked for is a corpus with a bias nobody recorded.
            print(f"  {name}: FAILED ({exc}). Skipped -- the corpus below is missing it.")
            continue
        print(f"  {name}: {len(batch)} postings")
        rows.extend(batch)

    if not rows:
        print("Nothing fetched. Re-run with --refresh if the cache is empty.")
        return 1

    before = len(rows)
    seen: set[str] = set()
    kept = []
    terms = set(ROLE_TERMS[args.role_filter].split()) if args.role_filter else None
    for row in rows:
        # Deduplicate on title+company, not on id. The same job is cross-posted
        # to both boards under different ids, and a duplicate pair in the corpus
        # is a query whose two top results are the same posting -- which
        # flatters precision@k for no reason.
        key = f"{row['title'].lower().strip()}|{row['company'].lower().strip()}"
        if key in seen:
            continue
        if len(row["description"]) < args.min_description:
            continue
        if terms and not matches_role(row, terms):
            continue
        seen.add(key)
        kept.append(row)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "postings.jsonl"
    target.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n", encoding="utf-8"
    )

    sources = sorted({r["source"] for r in kept})
    (out_dir / "SOURCES.md").write_text(
        "# Corpus provenance\n\n"
        f"Fetched {time.strftime('%Y-%m-%d')}. {len(kept)} postings after "
        f"deduplication and filtering, from {before} fetched.\n\n"
        + "".join(f"- {s}\n" for s in sources)
        + "\nEvery posting keeps its original `url` and `source`. These APIs are "
        "public and documented, and both publishers ask for attribution and "
        "light use. This corpus is local evaluation data only: it is not "
        "republished, redistributed, or committed to the repository.\n",
        encoding="utf-8",
    )

    print(
        f"\nWrote {len(kept)} postings to {target} (from {before} fetched).\n"
        f"Provenance in {out_dir / 'SOURCES.md'}.\n\n"
        f"Next:\n"
        f"    python -m relevance.queries --out {args.out}\n"
        f"    python -m relevance.label --data {args.out} --target 250"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
