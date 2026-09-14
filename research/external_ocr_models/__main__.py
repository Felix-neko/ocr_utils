"""Точка входа: ``uv run python -m research.external_ocr_models <команда>``."""

import sys

from research.external_ocr_models.cli import main

if __name__ == "__main__":
    sys.exit(main())
