"""Точка входа: ``uv run python -m research.legacy.table_processing <команда>``."""

import sys

from research.legacy.table_processing.cli import main

if __name__ == "__main__":
    sys.exit(main())
