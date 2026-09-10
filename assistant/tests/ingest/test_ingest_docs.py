import json

import ingest
import pytest


def notebook(tmp_path, cells):
    path = tmp_path / "demo.ipynb"
    path.write_text(
        json.dumps({"cells": [{"cell_type": t, "source": s} for t, s in cells]}),
        encoding="utf-8",
    )
    return path


def test_is_autodoc_shell_distinguishes_stubs_from_real_pages():
    stub = "Loss module\n===========\n\n.. automodule:: neuromancer.loss\n   :members:\n"
    prose = "\n".join(f"Explanatory sentence {i}." for i in range(12))

    assert ingest.is_autodoc_shell(stub) is True
    assert ingest.is_autodoc_shell(f".. automodule:: x\n\n{prose}\n") is False


@pytest.mark.parametrize(
    ("name", "text", "symbols"),
    [
        ("guide.rst", "Intro\n=====\nbody\nNext\n----\nmore", ["Intro", "Next"]),
        ("guide.md", "# First\nbody a\n## Second\nbody b", ["First", "Second"]),
        ("guide.other", "Title\n=====\nbody", ["Title"]),
    ],
)
def test_split_doc_sections_dispatches_on_suffix(name, text, symbols):
    assert [s["symbol_name"] for s in ingest.split_doc_sections(text, name)] == symbols


def test_split_by_headers_emits_preamble_using_the_file_stem():
    text = "intro line\n# First\nbody"

    sections = ingest.split_by_headers(text, "guide.md", [(1, "First")])

    assert [s["symbol_name"] for s in sections] == ["guide", "First"]
    assert sections[0]["content"] == "intro line"
    assert sections[1]["start_line"] == sections[0]["end_line"] + 1


def test_chunk_doc_file_emits_a_record_per_section(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# First\nbody a\n# Second\nbody b\n", encoding="utf-8")

    records = ingest.chunk_doc_file(str(path), tmp_path)

    assert [r["symbol_name"] for r in records] == ["First", "Second"]
    assert {r["source_type"] for r in records} == {"doc"}


def test_chunk_doc_file_skips_autodoc_shells_and_other_suffixes(tmp_path):
    stub = tmp_path / "api.rst"
    stub.write_text(".. automodule:: x\n   :members:\n", encoding="utf-8")
    script = tmp_path / "script.py"
    script.write_text("print('x')\n", encoding="utf-8")

    assert ingest.chunk_doc_file(str(stub), tmp_path) == []
    assert ingest.chunk_doc_file(str(script), tmp_path) == []


# line numbers here are cell indices + 1, not source lines
def test_chunk_ipynb_groups_markdown_with_following_code(tmp_path):
    path = notebook(
        tmp_path,
        [
            ("markdown", "# Part one\n" + "prose " * 40),
            ("code", "!pip install neuromancer"),
            ("code", "model = Trainer()"),
            ("markdown", "# Part two\n" + "prose " * 40),
            ("code", "model.fit()"),
        ],
    )

    records = ingest.chunk_ipynb_file(str(path), "examples/demo.ipynb")

    assert [r["symbol_name"] for r in records] == ["Part one", "Part two"]
    assert "pip install" not in records[0]["content"]
    assert "model = Trainer()" in records[0]["content"]
    assert (records[0]["start_line"], records[0]["end_line"]) == (1, 3)


def test_chunk_ipynb_carries_short_markdown_into_the_next_chunk(tmp_path):
    path = notebook(
        tmp_path,
        [
            ("markdown", "# Brief"),
            ("markdown", "# Real section\n" + "prose " * 40),
            ("code", "x = 1"),
        ],
    )

    (record,) = ingest.chunk_ipynb_file(str(path), "examples/demo.ipynb")

    assert "# Brief" in record["content"]
    assert "# Real section" in record["content"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ingest.py:441 moves a short markdown-only buffer into `carry`; when it is "
        "the last buffer the final flush_buffer() returns via the carry branch and "
        "the text never reaches examples.jsonl. Fixing it changes chunk ids, so it "
        "needs a corpus re-ingest and re-embed rather than riding along with tests."
    ),
)
def test_chunk_ipynb_keeps_trailing_markdown_only_cells(tmp_path):
    path = notebook(
        tmp_path,
        [
            ("markdown", "# Walkthrough\n" + "prose " * 40),
            ("code", "x = 1"),
            ("markdown", "# Conclusion\nThat is the whole workflow."),
        ],
    )

    records = ingest.chunk_ipynb_file(str(path), "examples/demo.ipynb")

    assert any("That is the whole workflow." in r["content"] for r in records)
