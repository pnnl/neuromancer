import dataclasses
from types import SimpleNamespace

import generate
import pytest
import retrieve
from ollama import ResponseError
from openai import OpenAIError


# settings.py freezes env vars at import and generate binds its own names via
# `from settings import ...`, so tests patch generate, never settings
class FakeOllamaClient:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error

    def chat(self, model, messages, options, **kwargs):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))


# stands in for the OpenAI class, recording which model it was asked for
def fake_openai(calls, reply=None, deltas=None, error=None):
    def completion(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def create(model, messages, stream=False, temperature=None):
        calls["model"] = model
        calls["messages"] = messages
        calls["temperature"] = temperature
        if error is not None:
            raise error
        if stream:
            return iter(deltas or [])
        return completion(reply)

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            calls["api_key"] = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    return FakeOpenAI


def delta(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


def chunk(file_path, document="", impl=None):
    metadata = {"file_path": file_path}
    if impl is not None:
        metadata["impl"] = impl
    return retrieve.Candidate(
        id=str(file_path),
        document=document,
        metadata=metadata,
        collection_name="neuromancer_src",
    )


def test_sources_dedupes_and_preserves_order():
    chunks = [chunk("b.py"), chunk("a.py"), chunk("b.py")]

    assert generate.sources_from_chunks(chunks) == ["b.py", "a.py"]


@pytest.mark.parametrize("path", [None, "", "unknown"])
def test_sources_drops_missing_and_unknown_paths(path):
    assert generate.sources_from_chunks([chunk(path)]) == []


def test_rewrite_result_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        generate.RewriteResult(query="q").query = "other"


def test_contextualize_returns_original_without_history():
    assert generate.contextualize_query("what is DPC", history=[]) == (
        generate.RewriteResult(query="what is DPC")
    )


def test_contextualize_strips_quotes_and_flags_rewritten(monkeypatch):
    generate._ollama_client = FakeOllamaClient(
        reply='"differentiable predictive control"'
    )
    monkeypatch.setattr(generate, "LLM_PROVIDER", "ollama")

    result = generate.contextualize_query(
        "what about it", history=[{"role": "user", "content": "explain DPC"}]
    )

    assert result.query == "differentiable predictive control"
    assert result.rewritten is True
    assert result.error is None


@pytest.mark.parametrize(
    "client",
    [
        FakeOllamaClient(error=ResponseError("connection refused")),
        FakeOllamaClient(reply=""),
        FakeOllamaClient(reply="x" * (generate.MAX_REWRITTEN_QUERY_CHARS + 1)),
    ],
)
def test_contextualize_falls_back_on_bad_rewrite(monkeypatch, client):
    generate._ollama_client = client
    monkeypatch.setattr(generate, "LLM_PROVIDER", "ollama")

    result = generate.contextualize_query(
        "follow up", history=[{"role": "user", "content": "prior"}]
    )

    assert result.query == "follow up"
    assert result.rewritten is False
    assert result.error


def test_build_messages_puts_system_first_and_question_last():
    messages = generate._build_messages("why?", [chunk("a.py", document="body")])

    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("## Question\nwhy?")
    assert "### a.py" in messages[-1]["content"]


def test_build_messages_filters_malformed_history():
    history = [
        {"role": "user", "content": "keep me"},
        {"role": "system", "content": "wrong role"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "keep me too"},
    ]

    messages = generate._build_messages("q", [], history=history)

    assert [m["content"] for m in messages[1:-1]] == ["keep me", "keep me too"]


def test_build_messages_truncates_oversized_chunks():
    limit = generate.MAX_CONTEXT_CHUNK_CHARS

    content = generate._build_messages("q", [chunk("a.py", "z" * (limit + 500))])[-1]

    assert "z" * limit in content["content"]
    assert "z" * (limit + 1) not in content["content"]


def test_build_messages_includes_the_implementation_source():
    chunks = [chunk("loss.py", "BarrierLoss docstring", impl="def _set_barrier(self):")]

    content = generate._build_messages("q", chunks)[-1]["content"]

    assert "BarrierLoss docstring" in content
    assert "def _set_barrier(self):" in content


def test_build_messages_truncates_oversized_implementations():
    limit = generate.MAX_CONTEXT_IMPL_CHARS
    chunks = [chunk("a.py", "doc", impl="z" * (limit + 500))]

    content = generate._build_messages("q", chunks)[-1]["content"]

    assert "z" * limit in content
    assert "z" * (limit + 1) not in content


def test_build_messages_omits_the_code_fence_without_an_implementation():
    content = generate._build_messages("q", [chunk("a.py", "doc")])[-1]["content"]

    assert "```python" not in content


def test_contextualize_openai_uses_the_cheap_rewrite_model(monkeypatch):
    calls = {}
    monkeypatch.setattr(generate, "OpenAI", fake_openai(calls, reply="rewritten query"))

    result = generate.contextualize_query(
        "follow up",
        history=[{"role": "user", "content": "prior"}],
        provider="openai",
        model="gpt-4o",
        api_key="sk-test",
    )

    assert calls["model"] == generate.OPENAI_REWRITE_MODEL
    assert calls["model"] != "gpt-4o"
    assert result.query == "rewritten query"


def test_contextualize_openai_rewrites_deterministically(monkeypatch):
    calls = {}
    monkeypatch.setattr(generate, "OpenAI", fake_openai(calls, reply="rewritten query"))

    generate.contextualize_query(
        "follow up",
        history=[{"role": "user", "content": "prior"}],
        provider="openai",
        api_key="sk-test",
    )

    assert calls["temperature"] == 0


def test_contextualize_openai_falls_back_on_api_error(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        generate, "OpenAI", fake_openai(calls, error=OpenAIError("rate limited"))
    )

    result = generate.contextualize_query(
        "follow up",
        history=[{"role": "user", "content": "prior"}],
        provider="openai",
        api_key="sk-test",
    )

    assert result.query == "follow up"
    assert result.error


def test_stream_from_chunks_openai_honors_the_selected_model(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        generate, "OpenAI", fake_openai(calls, deltas=[delta("Hello"), delta(" world")])
    )

    streamed = list(
        generate.stream_from_chunks(
            "q", [chunk("a.py")], provider="openai", model="gpt-4o", api_key="sk-test"
        )
    )

    assert streamed == ["Hello", " world"]
    assert calls["model"] == "gpt-4o"


def test_stream_openai_skips_empty_deltas(monkeypatch):
    calls = {}
    chunks = [delta("Hello"), delta(""), delta(None), SimpleNamespace(choices=[])]
    monkeypatch.setattr(generate, "OpenAI", fake_openai(calls, deltas=chunks))

    streamed = list(generate._stream_openai([], model="gpt-4o", api_key="sk-test"))

    assert streamed == ["Hello"]


def test_complete_openai_requires_an_api_key():
    with pytest.raises(ValueError, match="API key"):
        generate._complete_openai([], model="gpt-4o", api_key="")


# a generator, so the guard only fires once the caller starts consuming it
def test_stream_openai_requires_an_api_key():
    with pytest.raises(ValueError, match="API key"):
        list(generate._stream_openai([], model="gpt-4o", api_key=""))
