"""Точка входа: ``uv run python -m research.text_layer_fix <команда>``."""

import sys

from research.text_layer_fix.cli import main

if __name__ == "__main__":
    sys.exit(main())
