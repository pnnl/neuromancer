import pickle
from dataclasses import replace

import pytest
import retrieve


def candidate(doc_id):
    return retrieve.Candidate(
        id=doc_id, document=doc_id, metadata={}, collection_name="neuromancer_src"
    )


def stub_stages(monkeypatch, pool, scores, symbol_hits=()):
    monkeypatch.setattr(retrieve, "dense_search", lambda query, k: list(pool))
    monkeypatch.setattr(retrieve, "sparse_search", lambda query, k: [])
    monkeypatch.setattr(retrieve, "symbol_search", lambda query: list(symbol_hits))

    def fake_rerank(query, candidates, top_k):
        ranked = [replace(c, rerank_score=scores[c.id]) for c in candidates]
        ranked.sort(key=lambda c: (-(c.rerank_score or 0.0), c.id))
        return ranked[:top_k]

    monkeypatch.setattr(retrieve, "rerank", fake_rerank)


def test_retrieve_applies_relative_score_floor(monkeypatch):
    pool = [candidate(f"n{i}") for i in range(5)]
    stub_stages(
        monkeypatch, pool, {"n0": 0.9, "n1": 0.5, "n2": 0.4, "n3": 0.001, "n4": 0.0005}
    )

    # floor is 0.9 * RELATIVE_RERANK_FLOOR (0.02) = 0.018
    assert [c.id for c in retrieve.retrieve("query", top_k=8)] == ["n0", "n1", "n2"]


def test_retrieve_keeps_min_chunks_even_below_floor(monkeypatch):
    pool = [candidate(f"n{i}") for i in range(4)]
    stub_stages(monkeypatch, pool, {"n0": 0.9, "n1": 0.0001, "n2": 0.0001, "n3": 0.0001})

    assert len(retrieve.retrieve("query", top_k=8)) == retrieve.MIN_KEPT_CHUNKS


# a symbol named in the query is forced in even when the reranker demotes it, by
# overwriting the lowest-ranked non-symbol slot -- so n2 is evicted, not appended
def test_retrieve_reinjects_demoted_symbol_hit(monkeypatch):
    pool = [candidate(f"n{i}") for i in range(5)]
    scores = {"n0": 0.9, "n1": 0.8, "n2": 0.7, "n3": 0.6, "n4": 0.5, "sym1": 0.1}
    stub_stages(monkeypatch, pool, scores, symbol_hits=[candidate("sym1")])

    result = retrieve.retrieve("what does LitTrainer do", top_k=3)

    assert [c.id for c in result] == ["n0", "n1", "sym1"]


def test_retrieve_caps_reinjected_symbols_and_keeps_ids_unique(monkeypatch):
    pool = [candidate(f"n{i}") for i in range(6)]
    symbols = [candidate(f"sym{i}") for i in range(4)]
    scores = {f"n{i}": 0.9 - i * 0.05 for i in range(6)}
    scores.update({f"sym{i}": 0.01 for i in range(4)})
    stub_stages(monkeypatch, pool, scores, symbol_hits=symbols)

    ids = [c.id for c in retrieve.retrieve("query", top_k=6)]

    assert len([i for i in ids if i.startswith("sym")]) <= retrieve.MAX_SYMBOL_CHUNKS
    assert len(ids) == len(set(ids))


def test_rerank_squashes_logits_into_unit_interval(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.0, -10.0]

    monkeypatch.setattr(retrieve, "_reranker", FakeCrossEncoder())

    ranked = retrieve.rerank("query", [candidate("hit"), candidate("miss")], top_k=2)
    by_id = {c.id: c for c in ranked}

    assert by_id["hit"].rerank_score == pytest.approx(0.5)
    assert 0.0 < by_id["miss"].rerank_score < 0.001


def test_load_bm25_pickle_rejects_stale_index(monkeypatch, tmp_path):
    path = tmp_path / "bm25.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {"ids": ["a"], "collection_names": ["c"], "bm25": "idx", "expected_count": 5},
            f,
        )
    monkeypatch.setattr(retrieve, "BM25_INDEX_PATH", str(path))
    monkeypatch.setattr(retrieve, "_live_chunk_count", lambda: 7)

    assert retrieve._load_bm25_pickle() is None
    with pytest.raises(RuntimeError, match="scripts/setup.sh"):
        retrieve._get_bm25_index()


def test_load_bm25_pickle_rejects_malformed_payload(monkeypatch, tmp_path):
    path = tmp_path / "bm25.pkl"
    with open(path, "wb") as f:
        pickle.dump({"ids": [], "bm25": None}, f)
    monkeypatch.setattr(retrieve, "BM25_INDEX_PATH", str(path))

    assert retrieve._load_bm25_pickle() is None


def test_sparse_search_skips_ids_missing_from_chroma(monkeypatch):
    class FakeCollection:
        def get(self, ids, include):
            present = [i for i in ids if i != "orphan"]
            return {
                "ids": present,
                "documents": [f"doc {i}" for i in present],
                "metadatas": [{"file_path": f"{i}.py"} for i in present],
            }

    class FakeBM25:
        def get_scores(self, tokens):
            return [3.0, 2.0, 1.0]

    retrieve._bm25_cache = {
        "ids": ["a", "orphan", "b"],
        "collection_names": ["neuromancer_src"] * 3,
        "bm25": FakeBM25(),
    }
    monkeypatch.setattr(
        retrieve, "_get_collections", lambda: {"neuromancer_src": FakeCollection()}
    )

    results = retrieve.sparse_search("anything", k=10)

    assert [c.id for c in results] == ["a", "b"]
    assert [c.sparse_score for c in results] == [3.0, 1.0]
