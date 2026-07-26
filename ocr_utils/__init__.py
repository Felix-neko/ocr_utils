"""ocr_utils — postprocess PDF magazine scans: add OCR layer keeping source images."""

# Импорт pipeline опционален: подпакеты (например, finger_removal) должны
# запускаться даже когда зависимости основного pipeline недоступны.
try:
    from ocr_utils.pipeline import process_single_pdf, process_directory

    __all__ = ["process_single_pdf", "process_directory"]
except Exception:
    __all__ = []
