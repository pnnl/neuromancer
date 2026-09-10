import json

import load_vector_store as lvs
import pytest
import retrieve


def record(**overrides):
    base = {
        "id": "neuromancer/loss.py:Loss:24",
        "source_type": "api",
        "file_path": "neuromancer/loss.py",
        "symbol_name": "Loss",
        "start_line": 24,
        "end_line": 62,
        "content": "class Loss: ...",
    }
    base.update(overrides)
    return base


# the indexer and the searcher must share these, or every BM25 score and every
# stored embedding is computed against different text than queries are
def test_tokenizer_and_header_are_shared_with_retrieve():
    assert lvs._tokenize is retrieve._tokenize
    assert lvs.context_header is retrieve.context_header


@pytest.mark.parametrize(
    ("source_type", "expected"),
    [
        ("api", "neuromancer/loss.py:Loss:24:api"),
        ("impl", "neuromancer/loss.py:Loss:24:impl"),
        ("doc", "neuromancer/loss.py:Loss:24"),
        ("example", "neuromancer/loss.py:Loss:24"),
    ],
)
def test_chroma_id_suffixes_only_api_and_impl(source_type, expected):
    assert lvs.chroma_id(record(source_type=source_type)) == expected


def test_build_metadata_drops_none_but_keeps_falsy():
    metadata = lvs.build_metadata(record(symbol_name=None, start_line=0, end_line=""))

    assert "symbol_name" not in metadata
    assert metadata["start_line"] == 0
    assert metadata["end_line"] == ""


@pytest.mark.parametrize(("impl", "present"), [("def f(): ...", True), ("", False)])
def test_build_metadata_adds_impl_only_when_truthy(impl, present):
    assert ("impl" in lvs.build_metadata(record(impl=impl))) is present


@pytest.mark.parametrize("missing", ["id", "content", "file_path", "source_type"])
def test_read_jsonl_raises_on_missing_required_field(tmp_path, missing):
    incomplete = record()
    del incomplete[missing]
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(incomplete) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=repr(missing)):
        lvs.read_jsonl(path)


def test_read_jsonl_round_trips_records(tmp_path):
    path = tmp_path / "corpus.jsonl"
    records = [record(), record(id="other:X:1", symbol_name="X")]
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records) + "\n", encoding="utf-8"
    )

    assert lvs.read_jsonl(path) == records


def test_add_batches_embeds_prefixed_header_but_stores_raw_document():
    embedded = []

    class FakeCollection:
        def __init__(self):
            self.added = []

        def add(self, ids, embeddings, documents, metadatas):
            self.added.append(documents)

    def embed_fn(texts):
        embedded.extend(texts)
        return [[0.0]] * len(texts)

    collection = FakeCollection()
    lvs.add_batches(collection, [record(content="body text")], embed_fn)

    assert embedded == [
        f"{lvs.DOC_PREFIX}File: neuromancer/loss.py | Symbol: Loss | Type: api\nbody text"
    ]
    assert collection.added == [["body text"]]


def test_load_collection_aborts_on_count_mismatch():
    class FakeCollection:
        def count(self):
            return 7

    class FakeClient:
        def list_collections(self):
            return []

        def get_or_create_collection(self, name, embedding_function):
            return FakeCollection()

    with pytest.raises(SystemExit):
        lvs.load_collection(
            FakeClient(), "neuromancer_src", [record()], None, reset=False
        )
