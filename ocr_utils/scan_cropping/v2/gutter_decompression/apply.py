"""Растяжение у корешка на итоге выпрямления разворота: буквы обеих страниц холста после поля по тексту → модели сжатия → планы с расширением → смещение расширенного холста (сложение с полем делает ``text_run``)."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.gutter_decompression.fit import fit_model
from ocr_utils.scan_cropping.v2.gutter_decompression.letters import LetterParams, ink_mask, page_letters
from ocr_utils.scan_cropping.v2.gutter_decompression.model import CompressionModel
from ocr_utils.scan_cropping.v2.gutter_decompression.widen import WidenLimits, WidenPlan, plan_widen, text_zone
from ocr_utils.scan_cropping.v2.rectify.coons import Layout

def page_plans(final: np.ndarray, layout: Layout, px_per_mm: float, enabled: tuple[bool, ...],
               limits: WidenLimits = WidenLimits(),
               params: LetterParams = LetterParams()) -> tuple[WidenPlan, WidenPlan] | None:  # fmt: skip
    """Планы растяжения левой и правой страницы выпрямленного разворота.

    Args:
        final: Итог выпрямления (Кунс + поле по тексту) BGR, холст ``layout``.
        layout: Раскладка Кунса.
        px_per_mm: Плотность, px/мм.
        enabled: Растягивать ли каждую страницу (цветная, игнорируемая и ``no-text-rectify`` — нет).
        limits: Пределы и порог применения.
        params: Параметры меры букв.

    Returns:
        Планы обеих страниц или None — кадр с одной страницей (корешка на холсте нет).
    """
    if layout.single:
        return None
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(gray, px_per_mm, params)
    plans = []
    for page, (rect, on) in enumerate(zip(layout.rects, enabled)):
        outer_x = rect.x0 if page == 0 else rect.x1
        width_mm = abs(outer_x - layout.spine_x) / px_per_mm
        if not on:
            plans.append(WidenPlan(CompressionModel(), width_mm, width_mm))
            continue
        letters = page_letters(ink, int(rect.x0), int(np.ceil(rect.x1)), page, layout.spine_x, px_per_mm, params)
        starts, y_range, outer = text_zone(letters, rect)
        # Спад — от начала зоны текста (наименьшее по высоте): концы строк ближе медианного края растягиваются по
        # модели, а не постоянным сжатием у медианного края.
        origin = min(starts) if starts else 0.0
        model = fit_model(letters, rect.y0, rect.height, px_per_mm, origin)
        plans.append(plan_widen(model, width_mm, outer, limits, starts, y_range))
    return plans[0], plans[1]


__all__ = ["page_plans"]
