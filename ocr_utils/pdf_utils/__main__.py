"""CLI-точка входа: python -m ocr_utils.pdf_utils."""

from ocr_utils.pdf_utils.extract_images import main

if __name__ == "__main__":
    import sys

    sys.exit(main())
