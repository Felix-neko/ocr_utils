"""Границы растрового фото — реэкспорт боевого ``page_layout.raster.photo_edges`` (с 2026-10-24 ядро там; стенд держит меры, листы и разметку)."""

from ocr_utils.page_layout.raster.photo_edges import *  # noqa: F401,F403 — реэкспорт для модулей стенда
from ocr_utils.page_layout.raster.photo_edges import (  # noqa: F401 — имена, которые стенд берёт явно
    CORE_TONE_DELTA,
    SEARCH_MM,
    EdgeClass,
    Maps,
    PhotoBound,
    Side,
    SideFit,
    find_photos,
    light_map,
    object_ink,
    object_mask,
    object_outside,
    page_maps,
    robust_poly,
    step_on_profile,
)
