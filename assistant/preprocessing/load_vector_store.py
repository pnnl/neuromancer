import argparse
import json
import pickle
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from settings import (
    BATCH_SIZE,
    BM25_INDEX_PATH,
    CHROMA_PATH,
    DOC_PREFIX,
    EMBED_MODEL,
    KNOWLEDGE_COLLECTIONS,
    OLLAMA_URL,
)
from retrieve import _tokenize, context_header

COLLECTIONS = KNOWLEDGE_COLLECTIONS

METADATA_FIELDS = ("file_path", "symbol_name", "start_line", "end_line", "source_type")

def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("id", "content", "file_path", "source_type"):
                if key not in record:
                    raise ValueError(f"{path}:{line_no} missing required field {key!r}")
            records.append(record)
    return records


def chroma_id(record: dict) -> str:
    source_type = record["source_type"]
    base_id = record["id"]
    if source_type in ("api", "impl"):
        return f"{base_id}:{source_type}"
    return base_id


def build_metadata(record: dict) -> dict:
    metadata = {}
    for key in METADATA_FIELDS:
        value = record.get(key)
        if value is not None:
            metadata[key] = value
    # implementation source carried as display payload on api chunks
    if record.get("impl"):
        metadata["impl"] = record["impl"]
    return metadata


def get_embedding_function() -> OllamaEmbeddingFunction:
    # must match the query side in app/retrieve.py, or search quality silently degrades
    return OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=EMBED_MODEL)


def collection_exists(client: chromadb.PersistentClient, name: str) -> bool:
    return name in {c.name for c in client.list_collections()}


def add_batches(collection, records: list[dict], embed_fn: OllamaEmbeddingFunction) -> None:
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        documents = [r["content"] for r in batch]
        metadatas = [build_metadata(r) for r in batch]
        # embed prefix + context header + text, but store the plain document
        embeddings = embed_fn(
            [
                f"{DOC_PREFIX}{context_header(md)}\n{doc}"
                for doc, md in zip(documents, metadatas)
            ]
        )
        collection.add(
            ids=[chroma_id(r) for r in batch],
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )


def load_collection(
    client: chromadb.PersistentClient,
    name: str,
    records: list[dict],
    embed_fn: OllamaEmbeddingFunction,
    reset: bool,
) -> chromadb.Collection:
    expected_count = len(records)

    if reset and collection_exists(client, name):
        client.delete_collection(name)

    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
    )
    current_count = collection.count()

    if not reset and current_count == expected_count:
        print(f"Skipping {name}: already loaded ({current_count} records)")
        return collection

    if not reset and current_count > 0 and current_count != expected_count:
        print(
            f"Error: {name} has {current_count} records, expected {expected_count}. "
            "Re-run with --reset to rebuild.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if reset and current_count > 0:
        client.delete_collection(name)
        collection = client.get_or_create_collection(
            name=name,
            embedding_function=embed_fn,
        )

    print(f"Loading {name} ({len(records)} records)...")
    add_batches(collection, records, embed_fn)
    return collection


def build_and_save_bm25_index(client: chromadb.PersistentClient) -> None:
    from rank_bm25 import BM25Okapi

    # store only ids + collection names; retrieval fetches document text and
    # metadata from chroma, keeping the pickle small and cold starts fast
    ids: list[str] = []
    collection_names: list[str] = []
    tokenized_corpus: list[list[str]] = []

    for _, collection_name in COLLECTIONS:
        collection = client.get_collection(name=collection_name)
        data = collection.get(include=["documents", "metadatas"])
        for doc_id, document, metadata in zip(
            data["ids"],
            data["documents"],
            data["metadatas"],
        ):
            ids.append(doc_id)
            collection_names.append(collection_name)
            text = f"{context_header(metadata or {})}\n{document or ''}"
            tokenized_corpus.append(_tokenize(text))

    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "ids": ids,
        "collection_names": collection_names,
        "bm25": bm25,
        "expected_count": len(ids),
    }
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"Built BM25 index: {len(ids)} chunks -> {BM25_INDEX_PATH}")


def main(reset: bool = False) -> None:
    """Load every collection, then build BM25 last over what actually landed in Chroma."""
    embed_fn = get_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    loaded = {}
    for jsonl_path, collection_name in COLLECTIONS:
        records = read_jsonl(jsonl_path)
        loaded[collection_name] = load_collection(
            client,
            collection_name,
            records,
            embed_fn,
            reset,
        )

    print()
    for collection_name, collection in loaded.items():
        print(f"{collection_name}: {collection.count()}")

    build_and_save_bm25_index(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load ingested JSONL chunks into a local ChromaDB vector store."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild collections before loading.",
    )
    args = parser.parse_args()
    main(reset=args.reset)
