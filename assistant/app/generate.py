from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from ollama import Client, RequestError, ResponseError
from openai import OpenAI, OpenAIError

from settings import (
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_CONTEXT_CHUNK_CHARS,
    MECHANICS_PATH,
    OLLAMA_CHAT_OPTIONS,
    OLLAMA_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_REWRITE_MODEL,
    PROMPT_PATH,
    REWRITE_PROMPT_PATH,
)
from retrieve import Candidate

logger = logging.getLogger(__name__)


MAX_REWRITTEN_QUERY_CHARS = 300
MAX_HISTORY_TURN_CHARS = 600


MAX_CONTEXT_IMPL_CHARS = 2000


REWRITE_TEMPERATURE = 0


REWRITE_ERRORS = (
    ConnectionError,
    TimeoutError,
    ValueError,
    OpenAIError,
    RequestError,
    ResponseError,
)

_ollama_client: Client | None = None


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of a query rewrite; on failure holds the original query and an error."""

    query: str
    rewritten: bool = False
    error: str | None = None


def _load_system_prompt() -> str:
    base = PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    mechanics = MECHANICS_PATH.read_text(encoding="utf-8").rstrip()
    return f"{base}\n\n{mechanics}"


def sources_from_chunks(chunks: list[Candidate]) -> list[str]:
    """Return the distinct file paths behind the chunks, for citation in the UI."""
    sources: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        path = chunk.metadata.get("file_path")
        if not path or path == "unknown" or path in seen:
            continue
        seen.add(path)
        sources.append(path)
    return sources


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... (truncated)"


def _format_chunk(chunk: Candidate) -> str:
    file_path = chunk.metadata.get("file_path", "unknown")
    symbol_name = chunk.metadata.get("symbol_name")
    header = f"### {file_path}"
    if symbol_name and symbol_name != "unknown":
        header += f" — {symbol_name}"

    section = f"{header}\n\n{_truncate(chunk.document, MAX_CONTEXT_CHUNK_CHARS)}"

    impl = chunk.metadata.get("impl")
    if not impl:
        return section
    return f"{section}\n\n```python\n{_truncate(impl, MAX_CONTEXT_IMPL_CHARS)}\n```"


def _build_messages(
    query: str,
    chunks: list[Candidate],
    history: list[dict] | None = None,
) -> list[dict]:
    chunk_sections: list[str] = []

    for chunk in chunks:
        chunk_sections.append(_format_chunk(chunk))

    context_block = "\n\n---\n\n".join(chunk_sections)
    user_content = (
        f"## Reference excerpts from the NeuroMANCER repository\n\n"
        f"{context_block}\n\n---\n\n## Question\n{query}"
    )

    messages: list[dict] = [{"role": "system", "content": _load_system_prompt()}]
    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})
    return messages


def _get_ollama_client() -> Client:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = Client(host=OLLAMA_URL)
    return _ollama_client


def _stream_openai(
    messages: list[dict],
    model: str,
    api_key: str,
) -> Iterator[str]:
    if not api_key:
        raise ValueError(
            "OpenAI API key missing. Set OPENAI_API_KEY or enter it in the sidebar."
        )

    client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def _stream_ollama(
    messages: list[dict],
    model: str,
) -> Iterator[str]:
    stream = _get_ollama_client().chat(
        model=model,
        messages=messages,
        options=OLLAMA_CHAT_OPTIONS,
        stream=True,
    )
    for part in stream:
        content = part.message.content
        if content:
            yield content


def _load_rewrite_prompt() -> str:
    return REWRITE_PROMPT_PATH.read_text(encoding="utf-8").rstrip()


def _build_rewrite_messages(query: str, history: list[dict]) -> list[dict]:
    turns: list[str] = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            label = "User" if role == "user" else "Assistant"
            turns.append(f"{label}: {content[:MAX_HISTORY_TURN_CHARS]}")

    conversation = "\n\n".join(turns)
    user_content = (
        f"## Conversation so far\n\n{conversation}\n\n"
        f"## Follow-up question\n{query}"
    )
    return [
        {"role": "system", "content": _load_rewrite_prompt()},
        {"role": "user", "content": user_content},
    ]


def _complete_openai(messages: list[dict], model: str, api_key: str) -> str:
    if not api_key:
        raise ValueError("OpenAI API key missing")

    client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=REWRITE_TEMPERATURE,
    )
    return response.choices[0].message.content or ""


def _complete_ollama(messages: list[dict], model: str) -> str:
    response = _get_ollama_client().chat(
        model=model,
        messages=messages,
        options={**OLLAMA_CHAT_OPTIONS, "temperature": REWRITE_TEMPERATURE},
    )
    return response.message.content or ""


def contextualize_query(
    query: str,
    history: list[dict] | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> RewriteResult:
    """Rewrite a follow-up into a standalone search query; falls back to the original."""
    if not history:
        return RewriteResult(query=query)

    provider = (provider or LLM_PROVIDER).strip().lower()
    messages = _build_rewrite_messages(query, history)

    try:
        if provider == "openai":
            rewritten = _complete_openai(
                messages,
                model=OPENAI_REWRITE_MODEL,
                api_key=api_key if api_key is not None else OPENAI_API_KEY,
            )
        else:
            rewritten = _complete_ollama(messages, model=model or LLM_MODEL)

        rewritten = rewritten.strip().strip('"')
        if not rewritten or len(rewritten) > MAX_REWRITTEN_QUERY_CHARS:
            raise ValueError("rewrite returned an unusable query")
    except REWRITE_ERRORS as exc:
        logger.warning("Query rewrite failed (%s); using the original query", exc)
        return RewriteResult(query=query, error=str(exc))

    return RewriteResult(query=rewritten, rewritten=rewritten != query)


def stream_from_chunks(
    query: str,
    chunks: list[Candidate],
    history: list[dict] | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Iterator[str]:
    """Stream an answer grounded in the retrieved chunks."""
    messages = _build_messages(query, chunks, history=history)
    provider = (provider or LLM_PROVIDER).strip().lower()
    model = model or LLM_MODEL

    if provider == "openai":
        yield from _stream_openai(
            messages,
            model=model,
            api_key=api_key if api_key is not None else OPENAI_API_KEY,
        )
        return

    yield from _stream_ollama(messages, model=model)
