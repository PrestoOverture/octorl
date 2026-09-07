import pytest
from record_index import (
    Document, Index, tokenize, positions, excerpt,
    parse_query, filtered_search, paginate, concordance,
)


def make_index():
    """Helper to create a populated index for tests."""
    idx = Index()
    idx.add("d1", "Alpha Overview", "alpha beta gamma delta", tags=["science", "intro"])
    idx.add("d2", "Beta Summary", "alpha beta epsilon", tags=["science"])
    idx.add("d3", "Gamma Report", "gamma delta zeta eta theta", tags=["math"])
    return idx


# --- tokenize / positions / excerpt ---

def test_tokenize_basic():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_apostrophe():
    assert tokenize("don't") == ["don", "t"]


def test_positions_basic():
    pos = positions("the cat sat on the mat")
    assert pos["the"] == (0, 4)
    assert pos["cat"] == (1,)


def test_excerpt_centered():
    text = "one two three four five six seven eight nine ten"
    result = excerpt(text, "five", width=4)
    assert "five" in result
    words = result.split()
    assert len(words) <= 4


def test_excerpt_invalid_width():
    with pytest.raises(ValueError):
        excerpt("hello", "hello", width=0)


# --- Index add / get / remove ---

def test_add_and_get():
    idx = Index()
    doc = idx.add("x", "Title", "body text", tags=["Tag"])
    assert doc.identifier == "x"
    assert doc.title == "Title"
    assert doc.body == "body text"
    assert "tag" in doc.tags
    assert idx.get("x") is doc


def test_add_duplicate_raises():
    idx = Index()
    idx.add("x", "T", "B")
    with pytest.raises(ValueError):
        idx.add("x", "T2", "B2")


def test_remove():
    idx = Index()
    idx.add("x", "Title", "body word")
    idx.remove("x")
    assert len(idx) == 0
    with pytest.raises(KeyError):
        idx.get("x")


# --- replace (catches variable-misuse) ---

def test_replace_updates_tags():
    """Bug 3 (variable-misuse): replace must apply new tags, not keep old ones."""
    idx = Index()
    idx.add("x", "Old", "old body", tags=["old_tag"])
    doc = idx.replace("x", "New", "new body", tags=["new_tag"])
    assert "new_tag" in doc.tags
    assert "old_tag" not in doc.tags


def test_replace_keeps_tags_when_none():
    idx = Index()
    idx.add("x", "T", "B", tags=["keep"])
    doc = idx.replace("x", "T2", "B2")
    assert "keep" in doc.tags


# --- search (catches condition-inversion + boundary-condition-omission) ---

def test_search_mode_all():
    """Bug 1 (condition-inversion): mode='all' must require ALL query words."""
    idx = make_index()
    results = idx.search("alpha gamma", mode="all")
    assert "d1" in results
    assert "d2" not in results
    assert "d3" not in results


def test_search_mode_any():
    idx = make_index()
    results = idx.search("gamma epsilon", mode="any")
    assert "d1" in results
    assert "d2" in results
    assert "d3" in results


def test_search_mode_all_vs_any_differ():
    """Condition-inversion distinguisher: all and any must give different results."""
    idx = make_index()
    all_results = set(idx.search("alpha gamma", mode="all"))
    any_results = set(idx.search("alpha gamma", mode="any"))
    assert "d1" in all_results
    assert "d1" in any_results
    assert "d2" not in all_results
    assert "d2" in any_results


def test_search_empty_query():
    """Bug 7 (boundary-condition-omission): empty query must return [], not crash."""
    idx = make_index()
    assert idx.search("---") == []
    assert idx.search("") == []


def test_search_with_tag():
    idx = make_index()
    results = idx.search("alpha", tag="science")
    assert "d1" in results
    assert "d2" in results
    assert "d3" not in results


def test_search_with_limit():
    idx = make_index()
    results = idx.search("alpha", limit=1)
    assert len(results) == 1


def test_search_invalid_mode():
    idx = make_index()
    with pytest.raises(ValueError):
        idx.search("test", mode="invalid")


# --- frequency (catches missing-exception-handling) ---

def test_frequency_single_doc():
    idx = make_index()
    # "alpha" appears in both title ("Alpha Overview") and body of d1
    assert idx.frequency("alpha", "d1") == 2


def test_frequency_global():
    idx = make_index()
    # alpha: 2 in d1 (title + body) + 1 in d2 (body only)
    assert idx.frequency("alpha") == 3


def test_frequency_invalid_input():
    """Bug 5 (missing-exception-handling): non-word input must raise ValueError."""
    idx = make_index()
    with pytest.raises(ValueError, match="one word required"):
        idx.frequency("---")


# --- phrase ---

def test_phrase_match():
    idx = Index()
    idx.add("a", "T", "the quick brown fox")
    idx.add("b", "T", "the brown quick fox")
    results = idx.phrase("quick brown")
    assert "a" in results
    assert "b" not in results


def test_phrase_empty():
    idx = make_index()
    assert idx.phrase("") == []


# --- tag_counts ---

def test_tag_counts():
    idx = make_index()
    counts = idx.tag_counts()
    assert counts["science"] == 2
    assert counts["math"] == 1


# --- related (catches wrong-return-value) ---

def test_related_order():
    """Bug 4 (wrong-return-value): related must return most-related doc first."""
    idx = Index()
    idx.add("base", "T", "alpha beta gamma delta")
    idx.add("high", "T", "alpha beta gamma")
    idx.add("low", "T", "alpha zeta")
    results = idx.related("base")
    assert results[0] == "high"
    assert results[1] == "low"


# --- export / from_rows (catches api-parameter-error) ---

def test_export_sorted_by_identifier():
    """Bug 6 (api-parameter-error): export must sort by identifier, not title."""
    idx = Index()
    idx.add("b", "Alpha Title", "body one")
    idx.add("a", "Zeta Title", "body two")
    rows = idx.export()
    assert rows[0]["identifier"] == "a"
    assert rows[1]["identifier"] == "b"


def test_export_import_roundtrip():
    idx = make_index()
    rows = idx.export()
    rebuilt = Index.from_rows(rows)
    assert rebuilt.export() == rows
    assert rebuilt.audit()


# --- from_rows validation ---

def test_from_rows_bad_types():
    with pytest.raises(ValueError):
        Index.from_rows([{"identifier": 1, "title": "T", "body": "B"}])


def test_from_rows_bad_tags():
    with pytest.raises(ValueError):
        Index.from_rows([{"identifier": "x", "title": "T", "body": "B", "tags": "bad"}])


# --- paginate (catches off-by-one) ---

def test_paginate_page_one():
    """Bug 2 (off-by-one): page 1 must return the first items."""
    items = ["a", "b", "c", "d", "e"]
    page, has_more = paginate(items, page=1, size=2)
    assert page == ["a", "b"]
    assert has_more is True


def test_paginate_page_two():
    items = ["a", "b", "c", "d", "e"]
    page, has_more = paginate(items, page=2, size=2)
    assert page == ["c", "d"]
    assert has_more is True


def test_paginate_last_page():
    items = ["a", "b", "c"]
    page, has_more = paginate(items, page=2, size=2)
    assert page == ["c"]
    assert has_more is False


def test_paginate_invalid():
    with pytest.raises(ValueError):
        paginate([], page=0, size=1)


# --- parse_query / filtered_search ---

def test_parse_query():
    inc, exc = parse_query("hello -world foo -bar")
    assert inc == ["hello", "foo"]
    assert exc == ["world", "bar"]


def test_filtered_search():
    idx = Index()
    idx.add("a", "T", "alpha beta gamma")
    idx.add("b", "T", "alpha beta delta")
    results = filtered_search(idx, "alpha -gamma")
    assert results == ["b"]


# --- concordance ---

def test_concordance():
    idx = make_index()
    result = concordance(idx, "alpha")
    ids = [identifier for identifier, _ in result]
    assert set(ids) == {"d1", "d2"}


# --- vocabulary ---

def test_vocabulary_prefix():
    idx = make_index()
    vocab = idx.vocabulary("al")
    assert "alpha" in vocab


def test_vocabulary_all():
    idx = make_index()
    vocab = idx.vocabulary()
    assert len(vocab) > 0
    assert vocab == tuple(sorted(vocab))
