"""Точка входа: ``uv run python -m ocr_utils.text_layer_fix <команда>``."""

import sys

from ocr_utils.text_layer_fix.cli import main

if __name__ == "__main__":
    sys.exit(main())
