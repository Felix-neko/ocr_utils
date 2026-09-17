"""Точка входа: ``uv run python -m research.geometry_regression <команда>``."""

import sys

from research.geometry_regression.cli import main

if __name__ == "__main__":
    sys.exit(main())
