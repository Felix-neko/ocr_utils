"""Точка входа: ``uv run python -m research.article_problems <команда>``."""

import sys

from research.article_problems.cli import main

if __name__ == "__main__":
    sys.exit(main())
