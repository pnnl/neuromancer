from pathlib import PureWindowsPath

import ingest
import pytest
from hypothesis import given, settings, strategies as st


# this directory is named ingest/ with no __init__.py, so it could shadow
# preprocessing/ingest.py as a namespace package
def test_ingest_is_the_preprocessing_module_not_this_directory():
    assert ingest.__file__.endswith("preprocessing/ingest.py")
    assert hasattr(ingest, "make_record")


# patterns match the full walk root and carry a literal "/", so a bare
# directory name never matches
@pytest.mark.parametrize(
    ("path", "skipped"),
    [
        ("/repo/docs", True),
        ("/repo/nested/docs", True),
        ("/repo/.git", True),
        ("/repo/neuromancer.egg-info", True),
        ("/repo/src", False),
        ("/repo/examples", False),
        ("/repo/mydocs", False),
        ("docs", False),
    ],
)
def test_should_skip_directory(path, skipped):
    assert ingest.should_skip_directory(path) is skipped


# exercises the .as_posix() windows fix; on POSIX, Path keeps backslashes
def test_should_skip_directory_normalizes_windows_paths(monkeypatch):
    monkeypatch.setattr(ingest, "Path", PureWindowsPath)

    assert ingest.should_skip_directory(r"C:\repo\docs") is True


def test_should_skip_file_takes_a_bare_basename():
    assert ingest.should_skip_file("__init__.py") is True
    assert ingest.should_skip_file("pkg/__init__.py") is False
    assert ingest.should_skip_file("trainer.py") is False


def test_make_record_schema_and_id():
    record = ingest.make_record("a/b.py", "api", "Sym", 3, 9, "body")

    assert set(record) == {
        "id",
        "source_type",
        "file_path",
        "symbol_name",
        "start_line",
        "end_line",
        "content",
    }
    assert record["id"] == "a/b.py:Sym:3"
    assert (
        ingest.make_record("a/b.py", "example", None, 3, 9, "body")["id"] == "a/b.py::3"
    )


def test_split_oversized_returns_small_records_untouched():
    record = ingest.make_record("a.py", "impl", "S", 1, 2, "short")

    assert ingest.split_oversized(record) == [record]


@given(st.integers(min_value=150, max_value=600))
@settings(max_examples=25, deadline=None)
def test_split_oversized_preserves_content_and_line_ranges(line_count):
    lines = ["x" * 50 for _ in range(line_count)]
    record = ingest.make_record("a.py", "impl", "S", 1, line_count, "\n".join(lines))

    parts = ingest.split_oversized(record)

    assert "\n".join(p["content"] for p in parts) == "\n".join(lines)
    assert [p["id"] for p in parts] == [f"a.py:S:1:part{i}" for i in range(len(parts))]
    for earlier, later in zip(parts, parts[1:]):
        assert later["start_line"] == earlier["end_line"] + 1
    assert parts[-1]["end_line"] == record["end_line"]
