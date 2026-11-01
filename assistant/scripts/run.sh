#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_command streamlit "Install the assistant first: bash scripts/setup.sh"

cd "${ASSISTANT_DIR}"

# Required even on the OpenAI provider: query embeddings always go through it.
ensure_ollama

missing=()
[ -f "chroma_store/chroma.sqlite3" ] || missing+=("chroma_store/chroma.sqlite3")
[ -f "chroma_store/bm25_index.pkl" ] || missing+=("chroma_store/bm25_index.pkl")
compgen -G "knowledge/*.jsonl" >/dev/null || missing+=("knowledge/*.jsonl")

if [ "${#missing[@]}" -gt 0 ]; then
    die "Search index is missing (${missing[*]}). Run: bash scripts/setup.sh"
fi

log "Starting Streamlit"
if [ -n "${PORT:-}" ]; then
    exec streamlit run app/streamlit_app.py --server.port "${PORT}"
fi
exec streamlit run app/streamlit_app.py
