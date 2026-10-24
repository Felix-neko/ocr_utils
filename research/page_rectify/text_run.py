"""Проход кадра стенда со вторым преобразованием — реэкспорт боевого ``scan_cropping.v2.rectify.text_run`` (меры текста и листы теперь там же)."""

from ocr_utils.scan_cropping.v2.rectify.text_run import (  # noqa: F401 — реэкспорт для CLI стенда
    COLOUR_SHARE,
    ILLUMINATION_FEATHER_MM,
    PageState,
    TextParams,
    TextResult,
    add_displacement,
    equalize_pages,
    page_mask,
    pass_samples,
    text_frame,
)
