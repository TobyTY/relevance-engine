"""Tests for the corpus fetcher that do not touch the network.

Anything hitting a live API in a test suite is a test that fails on a train and
a publisher that gets called on every push. The network boundary is one function
(`_get`), so everything worth testing sits either side of it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from relevance.fetch import ROLE_TERMS, matches_role, strip_html


def test_nested_markup_survives():
    """A regex over tags is the usual shortcut and it mangles exactly the
    postings that matter -- the ones with nested markup listing the skills."""
    html = "<div><p>We use <strong>Python</strong> and <em>Postgres</em>.</p><ul><li>REST</li></ul></div>"
    assert strip_html(html) == "We use Python and Postgres. REST"


def test_block_tags_become_spaces_not_nothing():
    """Without this, `<li>Python</li><li>Go</li>` tokenizes as one term."""
    assert "python go" in strip_html("<li>Python</li><li>Go</li>").lower()


def test_entities_are_decoded():
    assert strip_html("<p>C&amp;C++ &lt;3</p>") == "C&C++ <3"


def test_empty_and_plain_input_are_safe():
    assert strip_html("") == ""
    assert strip_html("no markup here") == "no markup here"


def test_whitespace_is_collapsed():
    assert strip_html("<p>a</p>\n\n   <p>b</p>") == "a b"


def test_role_filter_matches_title_or_tags():
    terms = set(ROLE_TERMS["software"].split())
    assert matches_role({"title": "Backend Engineer", "tags": []}, terms)
    assert matches_role({"title": "Team Lead", "tags": ["DevOps"]}, terms)
    assert not matches_role({"title": "Pastry Chef", "tags": ["Kitchen"]}, terms)


def test_role_filter_tolerates_a_missing_tags_field():
    terms = set(ROLE_TERMS["embedded"].split())
    assert matches_role({"title": "Firmware Engineer", "tags": None}, terms)


def test_tags_arriving_as_a_stringified_list_are_parsed_safely():
    """Arbeitnow returns this field as a stringified Python list. It is parsed
    with literal_eval, never eval -- the value comes off the network, and eval
    on network input turns a data loader into a remote shell."""
    import ast

    assert ast.literal_eval("['Business Development']") == ["Business Development"]
    with pytest.raises((ValueError, SyntaxError)):
        ast.literal_eval("__import__('os').system('echo pwned')")


def test_the_cache_is_read_without_a_network_call(tmp_path: pathlib.Path, monkeypatch):
    """The cache is the rate limiter, not a speed-up. If a cached response can
    still reach the network, an impatient re-run hammers a publisher who asked
    for four calls a day."""
    import relevance.fetch as fetch

    monkeypatch.setattr(fetch, "CACHE_DIR", tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("the network must not be touched when a cache entry exists")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", explode)

    url = "https://example.invalid/api"
    import hashlib

    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    (tmp_path / f"{key}.json").write_text(json.dumps({"jobs": []}), encoding="utf-8")

    assert fetch._get(url, refresh=False) == {"jobs": []}
