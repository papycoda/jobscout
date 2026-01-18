"""Preload the semantic embedding model during build or startup."""

import logging
import os
import sys

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jobscout.semantic import preload_semantic_model  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if os.getenv("JOBSCOUT_SKIP_SEMANTIC_DOWNLOAD") == "1":
        logging.info("Skipping semantic model preload because JOBSCOUT_SKIP_SEMANTIC_DOWNLOAD=1")
        return 0

    if preload_semantic_model():
        logging.info("Semantic model is ready (downloaded or cached)")
        return 0

    logging.warning("Semantic model preload failed; runtime will fall back if needed")
    return 0  # Do not fail builds on preload issues


if __name__ == "__main__":
    sys.exit(main())
