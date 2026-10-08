from __future__ import annotations

import sys
from pathlib import Path

ASSISTANT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ASSISTANT_DIR / "app"))

from retrieve import RERANK_MODEL 


def main() -> int:
    """Download the reranker, then re-load it offline to prove the cache is complete."""
    from sentence_transformers import CrossEncoder

    print(f"Fetching reranker: {RERANK_MODEL}")
    try:
        CrossEncoder(RERANK_MODEL)
    except Exception as exc:
        print(f"error: failed to fetch {RERANK_MODEL}: {exc}", file=sys.stderr)
        return 1

    try:
        CrossEncoder(RERANK_MODEL, local_files_only=True)
    except Exception as exc:
        print(
            f"error: {RERANK_MODEL} downloaded but is not loadable offline: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Reranker ready (loads offline): {RERANK_MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
