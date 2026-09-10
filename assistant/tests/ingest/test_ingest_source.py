import ast

import ingest
import pytest


def test_iter_src_symbols_yields_class_then_methods_but_not_nested():
    tree = ast.parse(
        "class Trainer:\n"
        "    def fit(self, x): ...\n"
        "def outer():\n"
        "    def inner(): ...\n"
    )

    found = [(parent, node.name) for parent, node in ingest.iter_src_symbols(tree)]

    assert found == [(None, "Trainer"), ("Trainer", "fit"), (None, "outer")]


@pytest.mark.parametrize(
    ("source", "name", "indexed"),
    [
        ("def _helper(): ...", "_helper", False),
        ("def train(): ...", "train", True),
        ('def __init__(self):\n    """Docs."""', "__init__", True),
        ("def __init__(self, lr): ...", "__init__", True),
        ("def __init__(self): ...", "__init__", False),
        ("def _set_barrier(self, barrier): ...", "_set_barrier", True),
        ('def _reset(self):\n    """Docs."""', "_reset", True),
        ("def __repr__(self): ...", "__repr__", False),
        ('def __repr__(self):\n    """Docs."""', "__repr__", False),
    ],
)
def test_should_index_symbol(source, name, indexed):
    assert ingest.should_index_symbol(name, ast.parse(source).body[0]) is indexed


def test_extract_signature_and_doc_includes_decorators_not_the_body():
    source = '@property\ndef lr(self):\n    """The learning rate."""\n    return 1'
    node = ast.parse(source).body[0]

    extracted = ingest.extract_signature_and_doc(node, source.split("\n"))

    assert extracted.startswith("@property")
    assert "The learning rate." in extracted
    assert "return 1" not in extracted


def test_chunk_src_file_splits_api_from_impl(tmp_path):
    path = tmp_path / "loss.py"
    path.write_text(
        'class Loss:\n'
        '    """Weighted objective."""\n'
        '\n'
        'def scale(x):\n'
        '    return x * 2\n',
        encoding="utf-8",
    )

    records = {r["symbol_name"]: r for r in ingest.chunk_src_file(str(path), tmp_path)}

    assert records["Loss"]["source_type"] == "api"
    assert "Weighted objective." in records["Loss"]["content"]
    assert "impl" in records["Loss"]
    assert records["scale"]["source_type"] == "impl"
    assert "impl" not in records["scale"]


def test_chunk_src_file_truncates_oversized_implementations(tmp_path):
    body = "\n".join(f"    x = {i}" for i in range(4000))
    path = tmp_path / "big.py"
    path.write_text(f'def huge():\n    """Docs."""\n{body}\n', encoding="utf-8")

    (record,) = ingest.chunk_src_file(str(path), tmp_path)

    assert record["impl"].endswith("# ... truncated ...")
    assert len(record["impl"]) < len(body)


def test_chunk_src_file_returns_empty_on_syntax_error(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("def broken(:\n", encoding="utf-8")

    assert ingest.chunk_src_file(str(path), tmp_path) == []


def test_split_large_python_example_captures_symbols_and_main_guard():
    source = (
        "import os\n"
        "\n"
        "def helper():\n"
        "    return 1\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    helper()"
    )

    records = ingest.split_large_python_example(source.split("\n"), "examples/demo.py")

    assert [r["symbol_name"] for r in records] == ["helper", "__main__"]
    assert {r["source_type"] for r in records} == {"example"}
