#!/usr/bin/env bash
# Shared helpers for setup.sh, run.sh, and start_ollama.sh.

ASSISTANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_LOG="${OLLAMA_LOG:-${TMPDIR:-/tmp}/neuromancer-gpt-ollama.log}"
OLLAMA_START_TIMEOUT="${OLLAMA_START_TIMEOUT:-30}"

# chromadb and streamlit need 3.11+; neuromancer itself supports 3.9+.
REQUIRED_PYTHON="3.11"

log()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m warn\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror\033[0m %s\n' "$*" >&2; exit 1; }

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is not installed or not on PATH. $2"
}

ollama_install_hint() {
    case "$(uname -s)" in
        Darwin)               printf 'Install it with:  brew install ollama' ;;
        Linux)                printf 'Install it with:  curl -fsSL https://ollama.com/install.sh | sh' ;;
        MINGW*|MSYS*|CYGWIN*) printf 'Install it in PowerShell with:  winget install Ollama.Ollama' ;;
        *)                    printf 'Download it from https://ollama.com/download' ;;
    esac
}

ollama_is_up() {
    curl -sf -m 3 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1
}

# Safe to call repeatedly; a developer may already have one running.
ensure_ollama() {
    if ollama_is_up; then
        log "Ollama already running at ${OLLAMA_URL}"
        return 0
    fi

    require_command ollama "$(ollama_install_hint)
  Ollama is required on every path, including API mode, because query
  embeddings always run locally. See https://ollama.com/download"

    log "Starting Ollama (log: ${OLLAMA_LOG})"
    OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}" \
    OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}" \
        nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &

    local waited=0
    until ollama_is_up; do
        sleep 1
        waited=$((waited + 1))
        if [ "${waited}" -ge "${OLLAMA_START_TIMEOUT}" ]; then
            die "Ollama did not become healthy within ${OLLAMA_START_TIMEOUT}s. See ${OLLAMA_LOG}"
        fi
    done
    log "Ollama ready after ${waited}s"
}

ensure_ollama_model() {
    local model="$1"
    if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -q "^${model}\(:latest\)\?$"; then
        log "Ollama model present: ${model}"
        return 0
    fi
    log "Pulling Ollama model: ${model}"
    ollama pull "${model}" || die "Failed to pull ${model}"
}

check_python_version() {
    command -v python >/dev/null 2>&1 \
        || die "python is not on PATH. Activate the environment you want to install into first."

    python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' && return 0

    die "The assistant needs Python ${REQUIRED_PYTHON}+, but 'python' is $(python -c 'import sys; print("%d.%d" % sys.version_info[:2])').
  NeuroMANCER itself supports 3.9+, so your library environment may be older on purpose.
  Create a newer one and re-run:
      conda create -n neuromancer-gpt python=${REQUIRED_PYTHON} -y
      conda activate neuromancer-gpt"
}

# Warn rather than fail: conda environments do not always trip the venv check.
warn_if_no_env() {
    if python -c 'import os, sys; sys.exit(0 if sys.prefix != sys.base_prefix or "CONDA_PREFIX" in os.environ else 1)'; then
        return 0
    fi

    warn "No virtual environment detected. Dependencies would be installed into:"
    warn "    $(python -c 'import sys; print(sys.prefix)')"

    if [ "${NEUROMANCER_ASSUME_YES:-0}" = "1" ] || [ ! -t 0 ]; then
        warn "Continuing anyway."
        return 0
    fi

    printf '    Continue? [y/N] '
    read -r reply
    case "${reply}" in
        [yY]*) return 0 ;;
        *)     die "Aborted. Activate a venv or conda environment, then re-run." ;;
    esac
}

# settings.py is the single source of truth for model names.
setting() {
    (cd "${ASSISTANT_DIR}/app" && python -c "import settings; print(getattr(settings, '$1'))")
}
