from __future__ import annotations

import re

from generate import RewriteResult

_CODE_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
_DISPLAY_BRACKETS = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_INLINE_PARENS = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_BRACKET_TEX = re.compile(
    r"\[\s*((?:[^\[\]]|\n)*?\\[A-Za-z]+(?:[^\[\]]|\n)*?)\s*\]",
    re.DOTALL,
)

_STRAY_DOUBLE_DOLLAR = re.compile(r"(?<!\n)\$\$(?!\n)")


def _is_alone_on_line(text: str, start: int, end: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return not text[line_start:start].strip() and not text[end:line_end].strip()


def _to_math(match: re.Match, *, only_when_alone: bool = False) -> str:
    expression = match.group(1).strip()
    if _is_alone_on_line(match.string, match.start(), match.end()):
        return f"$$\n{expression}\n$$"
    if only_when_alone:
        return match.group(0)
    return f"${expression}$"


def sanitize_markdown(text: str) -> str:
    """Normalize LaTeX delimiters to Streamlit's $/$$ form, leaving code fences alone."""
    if not text:
        return text

    parts = _CODE_FENCE.split(text)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
            continue
        part = _DISPLAY_BRACKETS.sub(_to_math, part)
        part = _INLINE_PARENS.sub(lambda m: f"${m.group(1).strip()}$", part)
        part = _BRACKET_TEX.sub(lambda m: _to_math(m, only_when_alone=True), part)
        part = _STRAY_DOUBLE_DOLLAR.sub("$ $", part)
        out.append(part)
    return "".join(out)


def build_status_caption(
    research_s: float,
    stream_s: float,
    rewrite: RewriteResult,
) -> str:
    """Caption under an answer: timings, plus how the query was actually searched."""
    #timings and rewritten query
    caption = f"research {research_s:.1f}s  ·  stream {stream_s:.1f}s"
    if rewrite.error:
        return (
            f"{caption}  ·  ⚠ query rewrite unavailable ({rewrite.error}) — "
            "searched your question as typed"
        )
    if rewrite.rewritten:
        return f'{caption}  ·  searched: "{rewrite.query}"'
    return caption
