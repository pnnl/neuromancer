import pytest
import retrieve
from hypothesis import given, settings, strategies as st


def candidate(doc_id, **kwargs):
    kwargs.setdefault("document", doc_id)
    kwargs.setdefault("metadata", {})
    kwargs.setdefault("collection_name", "neuromancer_src")
    return retrieve.Candidate(id=doc_id, **kwargs)


def test_hybrid_merge_sums_rrf_across_both_lists():
    dense = [candidate("a"), candidate("b")]
    sparse = [candidate("b", sparse_score=9.0), candidate("z", sparse_score=1.0)]

    merged = retrieve.hybrid_merge(dense, sparse)
    by_id = {c.id: c for c in merged}

    assert by_id["b"].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["a"].rrf_score == pytest.approx(1 / 61)
    assert [c.id for c in merged] == ["b", "a", "z"]


def test_hybrid_merge_dense_wins_payload_on_duplicate_id():
    dense = [candidate("dup", document="from dense", dense_dist=0.25)]
    sparse = [candidate("dup", document="from sparse", sparse_score=7.0)]

    (merged,) = retrieve.hybrid_merge(dense, sparse)

    assert merged.document == "from dense"
    assert merged.dense_dist == 0.25
    assert merged.sparse_score == 7.0


def test_hybrid_merge_does_not_mutate_inputs():
    dense = [candidate("a")]

    retrieve.hybrid_merge(dense, [candidate("a", sparse_score=3.0)])

    assert dense[0].sparse_score is None
    assert dense[0].rrf_score is None


def test_hybrid_merge_truncates_to_pool_size():
    dense = [candidate(f"d{i:03d}") for i in range(40)]

    assert len(retrieve.hybrid_merge(dense, [])) == retrieve.POOL_SIZE


# this exact string is embedded with every chunk, so a change invalidates the index
def test_context_header_format():
    metadata = {"file_path": "a.py", "symbol_name": "Loss", "source_type": "api"}

    assert retrieve.context_header(metadata) == "File: a.py | Symbol: Loss | Type: api"
    assert retrieve.context_header({"file_path": "a.py"}) == "File: a.py"
    assert retrieve.context_header({}) == "File: "


def test_clean_query_collapses_whitespace():
    assert retrieve.clean_query("  what   is\n\tDPC ") == "what is DPC"


@pytest.mark.parametrize("raw", ["", "   ", "\n"])
def test_clean_query_raises_on_blank(raw):
    with pytest.raises(ValueError):
        retrieve.clean_query(raw)


@given(st.text(min_size=1).filter(lambda s: s.strip()))
@settings(max_examples=200, deadline=None)
def test_clean_query_never_exceeds_the_limit(raw):
    cleaned = retrieve.clean_query(raw)

    assert 0 < len(cleaned) <= retrieve.MAX_QUERY_CHARS
    assert "  " not in cleaned


def test_tokenize_lowercases_and_splits_on_word_chars():
    assert retrieve._tokenize("LitTrainer.fit(x)") == ["littrainer", "fit", "x"]
    assert retrieve._tokenize("!!!") == []
