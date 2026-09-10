from __future__ import annotations

import logging
import math
import pickle
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from settings import (
    BM25_INDEX_PATH,
    CHROMA_PATH,
    COLLECTION_NAMES,
    DEFAULT_TOP_K,
    EMBED_MODEL,
    OLLAMA_URL,
)

logger = logging.getLogger(__name__)

COLLECTIONS = COLLECTION_NAMES
MAX_QUERY_CHARS = 2000
POOL_SIZE = 24
RERANK_MODEL = "BAAI/bge-reranker-base"
RRF_K = 60
NOMIC_QUERY_PREFIX = "search_query: "
# the cross-encoder truncates at 512 tokens; capping the text avoids wasting time tokenizing
RERANK_MAX_DOC_CHARS = 2000

# how many final top_k slots explicitly named symbols may claim
MAX_SYMBOL_CHUNKS = 2


RELATIVE_RERANK_FLOOR = 0.02
MIN_KEPT_CHUNKS = 3

_client: chromadb.PersistentClient | None = None
_collections: dict[str, chromadb.Collection] | None = None
_bm25_cache: dict[str, Any] | None = None
_reranker: Any | None = None
_symbol_vocab: set[str] | None = None


@dataclass
class Candidate:
    """One retrieved chunk; each score field stays None until its stage runs."""

    id: str
    document: str
    metadata: dict
    collection_name: str
    dense_dist: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


def _get_embedding_function() -> OllamaEmbeddingFunction:
    return OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=EMBED_MODEL)


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def _get_collections() -> dict[str, chromadb.Collection]:
    global _collections
    if _collections is None:
        client = _get_client()
        embed_fn = _get_embedding_function()
        _collections = {
            name: client.get_collection(name=name, embedding_function=embed_fn)
            for name in COLLECTIONS
        }
    return _collections


def clean_query(raw: str) -> str:
    """Collapse whitespace and truncate to MAX_QUERY_CHARS; raises if empty."""
    cleaned = re.sub(r"\s+", " ", raw.strip())
    if not cleaned:
        raise ValueError("Query is empty")
    if len(cleaned) > MAX_QUERY_CHARS:
        logger.warning(
            "Query exceeds %d characters (%d)",
            MAX_QUERY_CHARS,
            len(cleaned),
        )
        cleaned = cleaned[:MAX_QUERY_CHARS]
    return cleaned


@lru_cache(maxsize=256)
def _embed_query(cleaned_query: str) -> tuple[float, ...]:
    embed_fn = _get_embedding_function()
    vector = embed_fn([f"{NOMIC_QUERY_PREFIX}{cleaned_query}"])[0]
    return tuple(float(v) for v in vector)


def dense_search(query: str, k: int) -> list[Candidate]:
    """Search every collection by vector similarity; returns k hits per collection."""
    query_embedding = list(_embed_query(query))
    collections = _get_collections()
    candidates: list[Candidate] = []

    for collection_name, collection in collections.items():
        results = collection.query(query_embeddings=[query_embedding], n_results=k)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            candidates.append(
                Candidate(
                    id=doc_id,
                    document=document or "",
                    metadata=metadata or {},
                    collection_name=collection_name,
                    dense_dist=float(distance) if distance is not None else None,
                )
            )

    # id as tie-breaker keeps ordering stable when distances are near-identical
    candidates.sort(
        key=lambda c: (c.dense_dist if c.dense_dist is not None else float("inf"), c.id)
    )
    return candidates


def _get_symbol_vocabulary() -> set[str]:
    global _symbol_vocab
    if _symbol_vocab is None:
        collection = _get_collections()["neuromancer_src"]
        data = collection.get(include=["metadatas"])
        _symbol_vocab = {
            metadata["symbol_name"]
            for metadata in data["metadatas"]
            if metadata.get("symbol_name")
        }
    return _symbol_vocab


def _extract_symbols(query: str) -> list[str]:
    vocab = _get_symbol_vocabulary()
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*", query))
    return sorted(token for token in tokens if token in vocab)


def symbol_search(query: str) -> list[Candidate]:
    """Exact lookup for symbols named literally in the query, e.g. "LitTrainer.fit"."""
    symbols = _extract_symbols(query)
    if not symbols:
        return []

    collection = _get_collections()["neuromancer_src"]
    candidates: list[Candidate] = []
    for name in symbols:
        result = collection.get(
            where={"symbol_name": name},
            include=["documents", "metadatas"],
        )
        for doc_id, document, metadata in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            candidates.append(
                Candidate(
                    id=doc_id,
                    document=document or "",
                    metadata=metadata or {},
                    collection_name="neuromancer_src",
                )
            )
    return candidates


def context_header(metadata: dict) -> str:
    """Provenance header prepended to a chunk; changing it invalidates the index."""
    header = f"File: {metadata.get('file_path', '')}"
    if metadata.get("symbol_name"):
        header += f" | Symbol: {metadata['symbol_name']}"
    if metadata.get("source_type"):
        header += f" | Type: {metadata['source_type']}"
    return header


def _tokenize(text: str) -> list[str]:
    # shared with load_vector_store.py; changing this needs a BM25 index rebuild
    return re.findall(r"\w+", text.lower())


def _live_chunk_count() -> int:
    return sum(collection.count() for collection in _get_collections().values())


def _load_bm25_pickle() -> dict[str, Any] | None:
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            payload = pickle.load(f)

        required_keys = {"ids", "collection_names", "bm25"}
        if not isinstance(payload, dict) or not required_keys.issubset(payload):
            raise ValueError("malformed")

        expected = payload.get("expected_count")
        if expected is not None and expected != _live_chunk_count():
            raise ValueError("stale")

        return {
            "ids": payload["ids"],
            "collection_names": payload["collection_names"],
            "bm25": payload["bm25"],
        }
    except Exception as exc:
        logger.warning("BM25 index %s unusable (%s)", BM25_INDEX_PATH, exc)
        return None


def _get_bm25_index() -> dict[str, Any]:
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    index = _load_bm25_pickle()
    if index is None:
        raise RuntimeError(
            f"BM25 index unavailable at {BM25_INDEX_PATH}. "
            "Build it with: bash scripts/setup.sh"
        )
    _bm25_cache = index
    return _bm25_cache


def sparse_search(query: str, k: int) -> list[Candidate]:
    """Rank chunks by BM25 keyword score, refetching text from Chroma for the top k."""
    cache = _get_bm25_index()
    bm25 = cache["bm25"]
    ids = cache["ids"]
    collection_names = cache["collection_names"]

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: (-scores[i], ids[i]))[:k]

    # the pickle only stores ids; fetch document text and metadata from chroma
    by_collection: dict[str, list[int]] = {}
    for index in ranked_indices:
        by_collection.setdefault(collection_names[index], []).append(index)

    collections = _get_collections()
    fetched: dict[str, tuple[str, dict]] = {}
    for collection_name, indices in by_collection.items():
        data = collections[collection_name].get(
            ids=[ids[i] for i in indices], include=["documents", "metadatas"]
        )
        for doc_id, document, metadata in zip(
            data["ids"], data["documents"], data["metadatas"]
        ):
            fetched[doc_id] = (document or "", metadata or {})

    candidates: list[Candidate] = []
    for index in ranked_indices:
        doc_id = ids[index]
        if doc_id not in fetched:
            logger.warning("BM25 hit %s missing from chroma", doc_id)
            continue
        document, metadata = fetched[doc_id]
        candidates.append(
            Candidate(
                id=doc_id,
                document=document,
                metadata=metadata,
                collection_name=collection_names[index],
                sparse_score=float(scores[index]),
            )
        )
    return candidates


def hybrid_merge(dense: list[Candidate], sparse: list[Candidate]) -> list[Candidate]:
    """Fuse dense and sparse results with Reciprocal Rank Fusion, capped at POOL_SIZE."""
    merged: dict[str, Candidate] = {}
    rrf_scores: dict[str, float] = {}

    for rank, candidate in enumerate(dense, start=1):
        rrf_scores[candidate.id] = rrf_scores.get(candidate.id, 0.0) + 1.0 / (RRF_K + rank)
        if candidate.id not in merged:
            merged[candidate.id] = candidate

    for rank, candidate in enumerate(sparse, start=1):
        rrf_scores[candidate.id] = rrf_scores.get(candidate.id, 0.0) + 1.0 / (RRF_K + rank)
        if candidate.id not in merged:
            merged[candidate.id] = candidate
        else:
            existing = merged[candidate.id]
            merged[candidate.id] = replace(
                existing,
                sparse_score=candidate.sparse_score,
            )

    pooled = [
        replace(candidate, rrf_score=rrf_scores[candidate.id])
        for candidate in merged.values()
    ]
    pooled.sort(key=lambda c: (-(c.rrf_score or 0.0), c.id))
    return pooled[:POOL_SIZE]


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, local_files_only=True)
    return _reranker


def rerank(query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
    """Re-score candidates with the cross-encoder; scores are sigmoid-squashed to (0, 1)."""
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [
        (
            query,
            f"{context_header(candidate.metadata)}\n"
            f"{candidate.document[:RERANK_MAX_DOC_CHARS]}",
        )
        for candidate in candidates
    ]
    scores = reranker.predict(pairs)

    # BAAI/bge-reranker-base returns unbounded raw logits (often negative);
    # sigmoid maps them to (0, 1) so the relative floor below means what it says
    ranked = [
        replace(candidate, rerank_score=1.0 / (1.0 + math.exp(-float(score))))
        for candidate, score in zip(candidates, scores)
    ]
    ranked.sort(key=lambda c: (-(c.rerank_score or 0.0), c.id))
    return ranked[:top_k]


def warmup() -> None:
    """Preload every cached resource at startup, so the first query is not slow."""
    _get_collections()
    _get_bm25_index()
    _get_reranker()
    _get_symbol_vocabulary()


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[Candidate]:
    """Run the full pipeline -- dense + sparse + symbol, fused and reranked."""
    cleaned = clean_query(query)
    dense = dense_search(cleaned, k=POOL_SIZE)
    sparse = sparse_search(cleaned, k=POOL_SIZE)
    pooled = hybrid_merge(dense, sparse)

    symbol_hits = symbol_search(cleaned)
    pooled_ids = {candidate.id for candidate in pooled}
    pooled += [c for c in symbol_hits if c.id not in pooled_ids]

    ranked = rerank(cleaned, pooled, top_k=len(pooled))

    # drop chunks far below the best match for this query
    top_score = ranked[0].rerank_score or 0.0 if ranked else 0.0
    score_floor = top_score * RELATIVE_RERANK_FLOOR
    kept = [
        c
        for i, c in enumerate(ranked)
        if i < MIN_KEPT_CHUNKS or (c.rerank_score or 0.0) >= score_floor
    ]
    final = kept[:top_k]

    # cross-encoder tends to demote raw API definitions below tutorials even when users asks about that symbol by name
    symbol_ids = {c.id for c in symbol_hits}
    final_ids = {c.id for c in final}
    demoted = [c for c in ranked if c.id in symbol_ids and c.id not in final_ids]
    for extra in demoted[:MAX_SYMBOL_CHUNKS]:
        for i in range(len(final) - 1, -1, -1):
            if final[i].id not in symbol_ids:
                final[i] = extra
                break
        else:
            break

    final.sort(key=lambda c: (-(c.rerank_score or 0.0), c.id))
    return final
