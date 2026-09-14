from __future__ import annotations

import os
from pathlib import Path

ASSISTANT_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = ASSISTANT_DIR / "knowledge"
PROMPTS_DIR = ASSISTANT_DIR / "prompts"
PROMPT_PATH = PROMPTS_DIR / "prompt.txt"
MECHANICS_PATH = PROMPTS_DIR / "mechanics.txt"
REWRITE_PROMPT_PATH = PROMPTS_DIR / "rewrite.txt"

CHROMA_PATH = str(ASSISTANT_DIR / "chroma_store")
BM25_INDEX_PATH = str(ASSISTANT_DIR / "chroma_store" / "bm25_index.pkl")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

# "ollama" (local) or "openai" (uses OPENAI_API_KEY)
LLM_PROVIDER = os.environ.get("NEUROMANCER_LLM_PROVIDER", "ollama").strip().lower()
if LLM_PROVIDER == "openai":
    LLM_MODEL = os.environ.get("NEUROMANCER_LLM_MODEL", "gpt-5.6")
else:
    LLM_MODEL = os.environ.get("NEUROMANCER_LLM_MODEL", "llama3.1:8b")
    LLM_PROVIDER = "ollama"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

OLLAMA_CHAT_OPTIONS = {"num_ctx": 8192}


DEFAULT_OPENAI_MODELS = "gpt-5.6"
OPENAI_MODELS = [
    model.strip()
    for model in os.environ.get("NEUROMANCER_OPENAI_MODELS", DEFAULT_OPENAI_MODELS).split(",")
    if model.strip()
] or [DEFAULT_OPENAI_MODELS]

# not a reasoning model: this runs before every retrieval
OPENAI_REWRITE_MODEL = os.environ.get("NEUROMANCER_REWRITE_MODEL", "gpt-4o-mini")


COLLECTION_NAMES = ["neuromancer_examples", "neuromancer_docs", "neuromancer_src"]
KNOWLEDGE_COLLECTIONS = [
    (KNOWLEDGE_DIR / "examples.jsonl", "neuromancer_examples"),
    (KNOWLEDGE_DIR / "docs.jsonl", "neuromancer_docs"),
    (KNOWLEDGE_DIR / "src.jsonl", "neuromancer_src"),
]

DEFAULT_TOP_K = int(os.environ.get("NEUROMANCER_TOP_K", "8"))
MAX_CONTEXT_CHUNK_CHARS = 2500

BATCH_SIZE = 100
DOC_PREFIX = "search_document: "
