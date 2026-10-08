import pytest
import ui_format
from generate import RewriteResult


# the four substitutions run in a fixed order: \[..\] before bare [..], or the
# bare-bracket rule eats the display form first
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("inline \\(a+b\\) text", "inline $a+b$ text"),
        ("\\[x = y\\]", "$$\nx = y\n$$"),
        ("before \\[x = y\\] after", "before $x = y$ after"),
        ("[ \\alpha ]", "$$\n\\alpha\n$$"),
        ("text [ \\alpha ] more", "text [ \\alpha ] more"),
        ("[not tex]", "[not tex]"),
        ("see [the docs](http://x) here", "see [the docs](http://x) here"),
        ("cost $$ per unit", "cost $ $ per unit"),
        ("$$\nx = y\n$$", "$$\nx = y\n$$"),
        ("", ""),
    ],
)
def test_sanitize_markdown_delimiter_rules(raw, expected):
    assert ui_format.sanitize_markdown(raw) == expected


def test_sanitize_markdown_never_touches_fenced_code():
    text = "before \\(a\\)\n```python\nx = [\\alpha]  # $$ stays\n```\nafter \\(b\\)"

    result = ui_format.sanitize_markdown(text)

    assert "```python\nx = [\\alpha]  # $$ stays\n```" in result
    assert "$a$" in result and "$b$" in result


@pytest.mark.parametrize(
    ("rewrite", "expected"),
    [
        (RewriteResult(query="q"), "research 1.0s  ·  stream 2.0s"),
        (
            RewriteResult(query="what is DPC", rewritten=True),
            'research 1.0s  ·  stream 2.0s  ·  searched: "what is DPC"',
        ),
    ],
)
def test_build_status_caption(rewrite, expected):
    assert ui_format.build_status_caption(1.0, 2.0, rewrite) == expected


def test_build_status_caption_error_beats_rewritten():
    rewrite = RewriteResult(query="q", rewritten=True, error="boom")

    caption = ui_format.build_status_caption(1.0, 2.0, rewrite)

    assert "query rewrite unavailable (boom)" in caption
    assert "searched:" not in caption
