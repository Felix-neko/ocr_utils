"""Зоны повёрнутого текста вне таблиц: цепочки глифов Docstrum по битональной копии, проверка tesseract.

ЧТО ИЩЕТСЯ. Боковой текст, набранный «лёжа»: подписи осей на графиках, надписи на схемах,
колонтитулы боком, отдельные повёрнутые врезки. Внутри таблиц он тоже бывает (шапки граф), но
таблицы разбираются своим ходом (``rotated_text.tables``) и сюда подаются исключениями — как и
растр, где глифов нет вовсе, зато полно пятен размером с букву.

КАК. ``docstrum.glyph_components`` берёт связные компоненты размера глифа (0.8–7 мм), а
``cluster_rotated`` — вертикальные цепочки ближайших соседей (kNN, k = 3, досягаемость 1.15
высоты глифа, не короче трёх глифов). Это ровно то, чем ``text_layer_fix.zones.free_zones``
находит зоны на растре PDF; сторона поворота (90 или 270) там определяется tesseract, здесь
не определяется.

ПРОВЕРКА ЧТЕНИЕМ (если подана серая копия). Одна геометрия давала на паке-1 две трети ложных зон
(просмотр 60 из 1171): буквы соседних строк, столбцы цифр таблиц, списки инициалов, пунктир и
штриховка. Зона читается tesseract под 0°, 90° и 270° (:func:`read_evidence`) и остаётся, если
боком читается слово, а прямо — хуже (:func:`reads_sideways`); внутри line art — мягче. До чтения
снимаются зоны уже строки (пунктир) и зоны на полутоновом растре (:func:`speck_density`).
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.regions import Region, RegionKind
from ocr_utils.page_layout.rotated_text.docstrum import cluster_rotated, glyph_components


def rotated_zones(
    bitonal: np.ndarray,
    dpi: int,
    exclude: list[Box],
    line_art: list[Box] | None = None,
    gray: np.ndarray | None = None,
    gray_dpi: int | None = None,
) -> list[Region]:
    """Зоны повёрнутого текста в пикселях поданной копии.

    Args:
        bitonal: Битональная копия (краска 0, бумага 255).
        dpi: Её разрешение.
        exclude: Рамки, внутри которых глифы не считаются (таблицы, растр).
        line_art: Рамки line art — зона внутри них помечается ``inside_line_art`` (подпись на схеме).
        gray: Серая копия для проверки чтением (:func:`read_evidence`, :func:`reads_sideways`); ``None`` — без проверки
            (прежнее поведение, только геометрия цепочек).
        gray_dpi: Разрешение ``gray``.

    Returns:
        Области ``ROTATED_TEXT``; ``info["inside_line_art"]``, при проверке — ``info["evidence"]``
        (длина лучшего слова под углами 0, 90, 270).
    """
    if bitonal is None or bitonal.size == 0:
        return []
    height, width = bitonal.shape[:2]
    stats = glyph_components(bitonal, dpi, exclude or None)
    regions: list[Region] = []
    for box in cluster_rotated(stats, dpi):
        box = box.clipped(width, height)
        if box.area <= 0:
            continue
        # Уже самой мелкой строки боком — пунктир или край линии, не подпись.
        if box.width < MIN_ZONE_WIDTH_MM / 25.4 * dpi:
            continue
        inside = any(_inside(box, art) for art in line_art or [])
        info: dict = {"inside_line_art": inside}
        if gray is not None and gray_dpi:
            gray_box = box.scaled(gray_dpi / dpi)
            if speck_density(gray, gray_box, gray_dpi) >= MAX_SPECKS_PER_MM2:
                continue
            evidence = read_evidence(gray, gray_box, gray_dpi)
            info["evidence"] = evidence
            if not reads_sideways(evidence, inside):
                continue
        regions.append(Region(box, RegionKind.ROTATED_TEXT, None, "rotated_text", False, info))
    return regions


# Самая узкая зона: поперёк цепочки — высота строки боком. На паке-1 настоящие зоны от 1.7 мм,
# пунктир и края линий, собранные в цепочку, — 1.0–1.2 мм (две выборки по 60 зон, 2026-09-27).
MIN_ZONE_WIDTH_MM = 1.4
# Полутоновый растр, не распознанный растром (фото внутри line art): в окрестности зоны
# SPECK_PAD_MM — мелкие точки меньше SPECK_MM по стороне. У растра их 2–2.6 на мм², у настоящих
# подписей не больше 0.3 (выборка из 60 зон, 2026-09-27).
SPECK_PAD_MM = 2.0
SPECK_MM = 0.25
MAX_SPECKS_PER_MM2 = 1.0
# Поле вокруг зоны при чтении, мм: буква на самом краю читается хуже.
READ_PAD_MM = 1.0
# Масштабы вырезки при чтении: tesseract на мелком боковом наборе неустойчив — одна и та же зона
# читается как есть и не читается увеличенной или наоборот (на паке-1 при чтении с 300 и с 600 dpi
# терялись разные настоящие зоны), поэтому берётся лучшее из двух чтений.
READ_SCALES = (1.0, 2.0)
# Повёрнутым текстом зона вне line art признаётся, если под 90° или 270° читается слово хотя бы
# такой длины (``evidence_at``: от 3 знаков с уверенностью от 50) и много длиннее, чем прямо.
# Буквы соседних строк («и» над «5») боком дают «слово» из 3 знаков, настоящие зоны вне line art —
# от 4 («80÷90»); просмотр 60 зон полного прогона пака-1, 2026-09-27.
MIN_SIDEWAYS_EVIDENCE = 4
# Внутри line art зона отбрасывается, только если прямо читается слово хотя бы такой длины и
# длиннее бокового: курсивные подписи в рамках схем tesseract не читает, зато вычитывает прямо
# шумовые слова из двух знаков.
MIN_UPRIGHT_EVIDENCE = 3
# Вне line art боковое чтение должно быть длиннее прямого хотя бы во столько раз. Столбик чисел
# таблицы («96 92 94 94») боком даёт «слово» из 3–5 цифр, но и прямо читается число из двух; у
# настоящих зон вне line art прямо не читается ничего (пак-1, просмотр 60 зон).
UPRIGHT_MARGIN = 3


def speck_density(gray: np.ndarray, box: Box, dpi: int) -> float:
    """Мелкие точки краски (растровые) на мм² в окрестности зоны.

    Args:
        gray: Серая копия страницы.
        box: Зона в пикселях ``gray``.
        dpi: Разрешение ``gray``.

    Returns:
        Число компонент площадью меньше ``SPECK_MM``² на мм² окрестности (поле ``SPECK_PAD_MM``).
    """
    height, width = gray.shape[:2]
    pad = int(round(SPECK_PAD_MM / 25.4 * dpi))
    crop = gray[box.padded(pad).clipped(width, height).slice]
    if crop.size == 0:
        return 0.0
    ink = (crop < 128).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    speck_area = (SPECK_MM / 25.4 * dpi) ** 2
    specks = int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] < speck_area))
    return specks / (crop.size / (dpi / 25.4) ** 2)


def read_evidence(gray: np.ndarray, box: Box, dpi: int) -> dict[int, int]:
    """Длина лучшего слова tesseract в зоне под углами 0, 90 и 270 — лучшее по ``READ_SCALES``.

    Args:
        gray: Серая копия страницы.
        box: Зона в пикселях ``gray``.
        dpi: Разрешение ``gray`` (для поля вокруг зоны).

    Returns:
        Угол → длина лучшего слова (0 — ничего не читается); пустой словарь у пустой зоны.
    """
    from ocr_utils.rotated_text.tables.orientation import evidence_at

    height, width = gray.shape[:2]
    pad = int(round(READ_PAD_MM / 25.4 * dpi))
    crop = gray[box.padded(pad).clipped(width, height).slice]
    if crop.size == 0:
        return {}
    evidence = {0: 0, 90: 0, 270: 0}
    for scale in READ_SCALES:
        # Второе чтение — по вырезке, увеличенной бикубически.
        scaled = crop if scale == 1 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled = np.ascontiguousarray(scaled)
        for angle in evidence:
            evidence[angle] = max(evidence[angle], evidence_at(scaled, angle, False))
    return evidence


def reads_sideways(evidence: dict[int, int], inside_line_art: bool) -> bool:
    """Признать ли зону повёрнутым текстом по чтению :func:`read_evidence`.

    Args:
        evidence: Угол → длина лучшего слова.
        inside_line_art: Зона внутри line art. Подписи на схемах часто курсивные и короткие («Рмин»,
            «Запсибстрой-снабсбыт» в рамке) — tesseract их не читает ни под каким углом, поэтому
            там зона отбрасывается, только если прямо читается слово от ``MIN_UPRIGHT_EVIDENCE`` знаков,
            длиннее бокового.

    Returns:
        True — зона остаётся.
    """
    if not evidence:
        return False
    sideways = max(evidence[90], evidence[270])
    if inside_line_art:
        return not (evidence[0] >= MIN_UPRIGHT_EVIDENCE and evidence[0] > sideways)
    return sideways >= MIN_SIDEWAYS_EVIDENCE and sideways >= UPRIGHT_MARGIN * evidence[0]


def _inside(box: Box, outer: Box) -> bool:
    """Лежит ли зона внутри рамки с допуском в её собственную ширину/высоту (как в ``free_zones``)."""
    return (
        box.x0 >= outer.x0 - box.width
        and box.x1 <= outer.x1 + box.width
        and box.y0 >= outer.y0 - box.height
        and box.y1 <= outer.y1 + box.height
    )


__all__ = ["MIN_SIDEWAYS_EVIDENCE", "read_evidence", "reads_sideways", "rotated_zones", "speck_density"]
