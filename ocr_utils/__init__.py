"""ocr_utils — postprocess PDF magazine scans: split spreads into pages, add OCR layer."""

from ocr_utils.pipeline import process_single_pdf, process_directory

__all__ = ["process_single_pdf", "process_directory"]
