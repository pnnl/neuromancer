#!/usr/bin/env bash
# Start Ollama on its own; setup.sh and run.sh call ensure_ollama themselves.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_ollama
log "Ollama is serving at ${OLLAMA_URL} (log: ${OLLAMA_LOG})"
