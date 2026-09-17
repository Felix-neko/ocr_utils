"""Карта локальных углов: наклон строк по тайлам полосы.

Классическая оценка перекоса (проекционный профиль контрастнее всего, когда строки лежат
горизонтально), но не по полосе целиком, а по сетке тайлов. Полоса с ПРЯМЫМИ строками
даёт один и тот же угол во всех тайлах, даже если перекошена вся; полоса с волнами или с
блоком, повёрнутым относительно остальных, — разные. Поэтому метрики — не сам угол,
а его РАЗБРОС по полосе и остаток от плоскости: перекос и трапеция описываются
плоскостью ``угол = a + bx + cy``, волны — нет.

Угол в тайле считается сдвигом профилей, а не поворотом картинки: тайл режется на узкие
столбцы, профиль каждого столбца сдвигается на ``x·tgθ``, и складывается. Без единой
интерполяции, и на порядок дешевле ``warpAffine`` на каждый угол.

Именно этот детектор ловит третий вид кривизны — локальный поворот блока, где кривизны
строк нет вовсе (эталонные 1966/01/IMG_0048_2R, IMG_0051_2R, 1972/03/IMG_0151_1L).
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.scan_markup.curved_lines.detectors.base import Detector, Frame, Measure, silent
from ocr_utils.scan_markup.curved_lines.detectors.line_fit import column_separators
from ocr_utils.scan_markup.curved_lines.fitting import plane_fit
from ocr_utils.scan_markup.orientation.detectors.ink_axis import _smear, glyph_mask
from ocr_utils.scan_markup.orientation.detectors.profile import PITCH_MAX_PX, PITCH_MIN_PX

# Сетка тайлов: 5 поперёк, 8 вдоль. На копии 150 dpi полосы пака-1 (~900x1600) тайл выходит
# 180x200 px — это 8-9 строк корпуса, достаточно для острого пика профиля, и при этом
# прогиб у корешка или повёрнутый блок занимают несколько тайлов, а не долю одного.
GRID_COLS = 5
GRID_ROWS = 8

# ТАЙЛЫ РЕЖУТСЯ ПО КОЛОНКАМ, а не по сетке поперёк всей полосы. Тайл, попавший на
# межколонник, содержит строки двух колонок, а их базовые линии сдвинуты друг относительно
# друга на произвольную долю шага; сдвиг «x·tgθ» при большом θ совмещает гребёнку левой
# колонки с гребёнкой правой, и профиль даёт ложный пик в ±4° на ровном тексте. Замер по
# паку-1: такие тайлы были главным источником одиночных ложных находок карты углов.
# Колонки берутся из ``line_fit.column_separators``; уже этой ширины колонка не режется.
MIN_COLUMN_PX = 60

# Перебор угла: грубо с шагом 0.5° в ±4°, потом тонко с шагом 0.1° вокруг лучшего.
# ±4° — с запасом: замер по паку-1 показал перекосы в пределах ±2°, а волны ±3°.
ANGLE_LIMIT_DEG = 4.0
COARSE_STEP_DEG = 0.5
FINE_STEP_DEG = 0.1

# Ширина столбца, по которому профиль сдвигается целиком. 8 px при 150 dpi — ширина буквы;
# внутри такого столбца сдвиг от наклона в 4° не превышает половины пикселя.
BLOCK_W = 8

# Тайл с меньшей долей краски (после отбора компонент размера глифа) — пустой, картинка
# или одна строка колонтитула. 0.02 — примерно три строки корпуса в тайле; при 0.01 ложные
# углы давали тайлы с номером страницы и обрывком заголовка.
MIN_INK_FRAC = 0.02

# Острота пика профиля: (лучшая дисперсия − медианная) / лучшая. Ниже — в тайле нет строк
# как таковых (таблица, чертёж, одна подпись), угол по нему не считается.
MIN_CONFIDENCE = 0.15

# Меньше стольких тайлов с углом — разброс не о чём считать.
MIN_TILES = 4

# Периодичность профиля тайла ПРИ НАЙДЕННОМ УГЛЕ (пик автокорреляции в диапазоне
# межстрочного шага): у текста 0.4-0.85, у растра фотографии и штриховки чертежа 0.03-0.15.
# Мерить надо именно выровненный профиль: у тайла с наклоном в 4° профиль без выравнивания
# размыт, и настоящий прогиб у корешка (0350_2R) давал 0.14-0.17. Замер по паку: одиночные
# ложные находки карты углов — фотографии и чертежи, тайлы которых проходили порог
# уверенности и давали ±3-4°.
MIN_PERIODICITY = 0.25


# Во сколько раз профиль столбца растягивается по y перед сдвигом. Сдвиг на дробное число
# пикселей через интерполяцию НЕЛЬЗЯ: интерполяция размывает профиль тем сильнее, чем
# дальше сдвиг от целого, и дисперсия оказывается максимальной там, где сдвиги целые, —
# то есть при нулевом угле. Первая версия с целыми сдвигами так и отвечала «−0.3°» на
# всё подряд, версия с линейной интерполяцией — «0.0°». Растяжение повторением (без
# интерполяции) и целые сдвиги в растянутых единицах дают шаг 1/UPSAMPLE px без размытия.
UPSAMPLE = 4


def _sheared_profile(blocks: np.ndarray, offsets: np.ndarray, angle_deg: float, pad: int) -> np.ndarray:
    """Профиль тайла при наклоне ``angle_deg``: столбцы сдвинуты и сложены.

    ``blocks`` уже растянуты по y в UPSAMPLE раз; сдвиг — целое число растянутых единиц.
    """
    shifts = np.rint(offsets * np.tan(np.radians(angle_deg)) * UPSAMPLE).astype(int)
    height = blocks.shape[1]
    total = np.zeros(height + 2 * pad, np.float64)
    for block, shift in zip(blocks, shifts):
        start = pad + shift
        total[start : start + height] += block
    return total


def _profile_variance(blocks: np.ndarray, offsets: np.ndarray, angle_deg: float, pad: int) -> float:
    return float(_sheared_profile(blocks, offsets, angle_deg, pad).var())


def _periodicity(profile: np.ndarray) -> float:
    """Пик автокорреляции профиля в диапазоне межстрочного шага (шаг — в растянутых единицах)."""
    profile = profile - profile.mean()
    energy = float((profile * profile).sum())
    lo, hi = PITCH_MIN_PX * UPSAMPLE, PITCH_MAX_PX * UPSAMPLE
    if energy <= 0.0 or profile.size <= hi * 2:
        return 0.0
    correlation = np.correlate(profile, profile, mode="full")[profile.size - 1 :] / energy
    window = correlation[lo : hi + 1]
    return float(window.max()) if window.size else 0.0


def tile_angle(tile: np.ndarray) -> tuple[float, float, float]:
    """Угол строк в тайле (градусы, положительный — строки опускаются вправо), уверенность
    и периодичность выровненного профиля."""
    height, width = tile.shape
    n_blocks = max(1, width // BLOCK_W)
    used = n_blocks * BLOCK_W
    blocks = tile[:, :used].astype(np.float32).reshape(height, n_blocks, BLOCK_W).sum(axis=2).T  # (блок, y)
    blocks = np.repeat(blocks, UPSAMPLE, axis=1)
    offsets = (np.arange(n_blocks) + 0.5) * BLOCK_W - used / 2.0
    pad = int(np.ceil(abs(offsets).max() * np.tan(np.radians(ANGLE_LIMIT_DEG)) * UPSAMPLE)) + 1

    coarse = np.arange(-ANGLE_LIMIT_DEG, ANGLE_LIMIT_DEG + 1e-9, COARSE_STEP_DEG)
    coarse_var = np.array([_profile_variance(blocks, offsets, angle, pad) for angle in coarse])
    best = float(coarse[int(coarse_var.argmax())])
    fine = np.arange(best - COARSE_STEP_DEG, best + COARSE_STEP_DEG + 1e-9, FINE_STEP_DEG)
    fine = fine[np.abs(fine) <= ANGLE_LIMIT_DEG + 1e-9]  # запас pad рассчитан на ±ANGLE_LIMIT_DEG
    fine_var = np.array([_profile_variance(blocks, offsets, angle, pad) for angle in fine])
    peak = float(fine_var.max())
    if peak <= 0.0:
        return 0.0, 0.0, 0.0
    # Знак: ось y вниз, сдвиг «+x·tgθ» выравнивает строку, которая опускается вправо.
    # Среди равных пиков берётся средний, а не первый: иначе на плоском пике угол
    # систематически уезжал бы к началу сетки.
    top = np.nonzero(fine_var >= peak - 1e-9)[0]
    angle = float(fine[int(top[len(top) // 2])])
    confidence = (peak - float(np.median(coarse_var))) / peak
    return angle, max(0.0, confidence), _periodicity(_sheared_profile(blocks, offsets, angle, pad))


def _tile_boxes(smeared: np.ndarray, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Прямоугольники тайлов: по строкам сетки, по колонкам — внутри каждой колонки."""
    separators = column_separators(smeared)
    edges = [0] + [x for pair in separators for x in pair] + [width]
    columns = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] - edges[i] >= MIN_COLUMN_PX]
    if not columns:
        columns = [(0, width)]
    target_w = width / GRID_COLS
    tile_h = height // GRID_ROWS
    boxes = []
    for x0, x1 in columns:
        n_cols = max(1, int(round((x1 - x0) / target_w)))
        tile_w = (x1 - x0) / n_cols
        for row in range(GRID_ROWS):
            for col in range(n_cols):
                boxes.append((int(x0 + col * tile_w), row * tile_h, int(x0 + (col + 1) * tile_w), (row + 1) * tile_h))
    return boxes


def measure(frame: Frame, keep_raw: bool = False) -> Measure:
    mask = glyph_mask(frame.gray150)
    height, width = mask.shape
    if height // GRID_ROWS < 20 or width // GRID_COLS < 20:
        return silent("полоса слишком мала для сетки")

    tiles: list[list[float]] = []  # [x0, y0, x1, y1, angle, confidence, ink]
    for x0, y0, x1, y1 in _tile_boxes(_smear(mask, horizontal=True), width, height):
        tile = mask[y0:y1, x0:x1]
        if tile.shape[1] < BLOCK_W * 3 or tile.shape[0] < 20:
            continue
        ink = float(tile.mean() / 255.0)
        if ink < MIN_INK_FRAC:
            continue
        angle, confidence, period = tile_angle(tile)
        if confidence < MIN_CONFIDENCE or period < MIN_PERIODICITY:
            continue
        tiles.append([x0, y0, x1, y1, round(angle, 2), round(confidence, 3), round(ink, 4)])

    raw = {"w": width, "h": height, "tiles": tiles} if keep_raw else None
    if len(tiles) < MIN_TILES:
        return Measure(metrics={"tiles": float(len(tiles))}, note="мало тайлов со строками", silent=True, raw=raw)

    angles = np.array([tile[4] for tile in tiles])
    weights = np.array([tile[6] for tile in tiles])
    xs = np.array([(tile[0] + tile[2]) / 2.0 for tile in tiles]) / width
    ys = np.array([(tile[1] + tile[3]) / 2.0 for tile in tiles]) / height
    median = float(np.median(angles))
    _, grad_x, grad_y, resid = plane_fit(xs, ys, angles, weights)
    metrics = {
        "tiles": float(len(tiles)),
        "median_deg": median,
        "spread_deg": float(np.percentile(angles, 90) - np.percentile(angles, 10)),
        "max_dev_deg": float(np.abs(angles - median).max()),
        "resid_deg": resid,
        "grad_x": grad_x,
        "grad_y": grad_y,
    }
    return Measure(metrics=metrics, raw=raw)


def draw(canvas: np.ndarray, raw: dict, scale: float) -> None:
    """Тайлы и штрих под найденным углом в каждом; цвет — отклонение от медианы."""
    tiles = raw.get("tiles", [])
    if not tiles:
        return
    median = float(np.median([tile[4] for tile in tiles]))
    for x0, y0, x1, y1, angle, confidence, _ in tiles:
        cv2.rectangle(
            canvas, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale)), (215, 215, 215), 1
        )
        cx, cy = (x0 + x1) / 2.0 * scale, (y0 + y1) / 2.0 * scale
        half = 0.4 * (x1 - x0) * scale
        dy = half * np.tan(np.radians(angle))
        deviation = abs(angle - median)
        colour = (0, 170, 0) if deviation < 0.7 else (0, 200, 255) if deviation < 1.5 else (0, 0, 255)
        cv2.line(canvas, (int(cx - half), int(cy - dy)), (int(cx + half), int(cy + dy)), colour, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{angle:+.1f}",
            (int(cx - half), int(cy - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colour,
            1,
            cv2.LINE_AA,
        )


ALGORITHM = Detector(
    name="skew_map",
    summary="разброс локальных углов строк по сетке тайлов (профиль проекции со сдвигом столбцов)",
    stage="cpu",
    # Пороги — по 14 эталонным полосам: max отклонение тайла от медианы у прямых 0.5-1.1°,
    # у кривых 2.3-4.8° (включая обе «лёгкие»); остаток от плоскости у прямых 0.09-0.34°,
    # у кривых 0.35-1.6°; размах p90-p10 у прямых до 0.7°, у кривых 0.9-5.0° (кроме одной
    # лёгкой с локальным прогибом в одном тайле — размах 0, отклонение 4.2°).
    # Пороги версии 2 (дробный сдвиг): на эталоне max отклонение тайла у прямых до 1.2°,
    # у кривых 3.1-4.0° (лёгкая — 0.9°); остаток от плоскости у прямых до 0.26°, у кривых
    # 0.72-1.54° (лёгкая — 0.18°); на шести ложных находках первого прогона по паку
    # (ровный текст с одним тайлом-выбросом) max отклонение 0.8-1.2°, одна — 2.4°.
    thresholds={"spread_deg": 1.5, "max_dev_deg": 2.0, "resid_deg": 0.6},
    version=4,
    run=measure,
    draw=draw,
)
