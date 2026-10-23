"""Синтез срезанных нижних строк страницы по колонкам: ось строки по донору, буквы со страницы, текст (tesseract + DeepSeek), набор и вклейка с чёткой кромкой листа."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz
from scipy import ndimage

from research.cut_lines import synth as S
from research.cut_lines.llm import line_text

log = logging.getLogger(__name__)

# Увеличение строки донора для DeepSeek (буквы около 25–30 px).
LLM_SCALE = 1.5
# Высота куска донора для DeepSeek, px.
LLM_CROP_HEIGHT = 360
# Сколько строк над нижней брать в контекст DeepSeek.
CONTEXT_ROWS = 3
# Поле кадра вокруг строки при вычислениях, px полного кадра.
BOX_PAD = 60
# Колонка не дорисовывается, если текст DeepSeek похож на черновик tesseract меньше этого (0..100): модель
# сочинила, а не прочла.
MIN_DRAFT_MATCH = 55.0
# Строка набирается, если она шире этой доли ширины страницы (колонтитул «Риск № 5» — около 8%).
MIN_COLUMN_SHARE = 0.03
# Пределы множителя силы краски букв донора.
INK_GAIN = (0.5, 1.5)
# Перебор размытия букв донора, px.
BLUR_STEPS = (0.4, 0.7, 1.0, 1.3, 1.7, 2.2)
# Слово набирается, только если библиотечных букв в нём не больше этой доли (остальные — буквы донора на месте).
MAX_LIBRARY_SHARE = 0.25
# Строка — последняя в колонке, если её краска (в зоне строчных) покрывает не меньше этой доли отрезка строки.
LAST_ROW_COVER = 0.3
# Сдвиг оси по шагу строк делается, если расхождение больше этой доли высоты буквы.
PITCH_SNAP_MIN = 0.25
# Проверки набора: сходство перечитанной tesseract-ом строки с текстом (0..100); доля видимой краски строки (1),
# накрытой набором (если видимой краски больше MIN_VISIBLE_INK px).
MIN_REREAD = 70.0
MIN_OVERLAP = 0.55
MIN_VISIBLE_INK = 300
# Слово считается видимым, если видимой бумаги (1) в его рамке не меньше этой доли; поле вокруг набранного слова, px.
WORD_VISIBLE = 0.97
WORD_PAD = 4
# Куски склеиваются в строку, если промежуток между ними по x меньше стольких высот буквы.
MERGE_GAP = 3.0
# Строка над нижней: вклейка вниз до этой глубины под базовой линией, в высотах буквы (середина промежутка).
LOWER_SEAM = 0.5
# Полоса строки над нижней шире самой строки на столько высот буквы с каждой стороны.
SIDE_PAD = 1.5


@dataclass
class ColumnReport:
    """Итог по колонке.

    Поля:
        status: ``synth`` / ``skip_text`` (текст не сошёлся с черновиком) / ``no_axis`` / ``no_paper``.
        text: Текст строки (DeepSeek).
        draft: Черновик tesseract по донору.
        match: Сходство текста с черновиком (0..100).
        missing: Буквы без образца.
    """

    status: str
    text: str = ""
    draft: str = ""
    match: float = 0.0
    missing: str = ""


@dataclass
class SynthResult:
    """Итог синтеза страницы.

    Поля:
        image: Кадр (1) после синтеза.
        columns: Отчёты по колонкам.
        boxes: Окна изменений по колонкам ``(x0, y0, x1, y1)``.
    """

    image: np.ndarray
    columns: list[ColumnReport] = field(default_factory=list)
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


def seg_to_frame(map_seg: np.ndarray, x: np.ndarray, y: np.ndarray, seg: float, donor_scale: float) -> np.ndarray:
    """Отображение масштаба совмещения в точках полного кадра (1) → координаты полного кадра (2).

    Args:
        map_seg: Карта (по x или y) на масштабе совмещения, координаты в уменьшенном (2).
        x: Абсциссы точек полного кадра (1).
        y: Ординаты.
        seg: Масштаб совмещения относительно (1).
        donor_scale: Масштаб уменьшенного (2) относительно полного (2).

    Returns:
        Координаты в полном (2) той же формы.
    """
    val = cv2.remap(map_seg, (x * seg).astype(np.float32), (y * seg).astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
    return val / donor_scale


def donor_view(donor: np.ndarray, pm, mx: np.ndarray, my: np.ndarray, seg: float, donor_scale: float) -> np.ndarray:
    """Донор в точках ``(mx, my)`` полного кадра (1) (по отображению страницы).

    Args:
        donor: Кадр (2) BGR.
        pm: ``PageMap`` страницы.
        mx: Абсциссы точек (1).
        my: Ординаты.
        seg: Масштаб совмещения.
        donor_scale: Масштаб уменьшенного (2).

    Returns:
        BGR той же формы, что ``mx``.
    """
    dx = seg_to_frame(pm.map_x, mx, my, seg, donor_scale).astype(np.float32)
    dy = seg_to_frame(pm.map_y, mx, my, seg, donor_scale).astype(np.float32)
    return cv2.remap(donor, dx, dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def column_axis(donor: np.ndarray, pm, comp_seg: np.ndarray, seg: float, donor_scale: float, xh_hint: float):
    """Ось нижней строки колонки по краске донора в координатах (1).

    Args:
        donor: Кадр (2).
        pm: ``PageMap``.
        comp_seg: Маска строки колонки на масштабе совмещения.
        seg: Масштаб совмещения.
        donor_scale: Масштаб уменьшенного (2).
        xh_hint: Ожидаемая высота буквы в полном кадре, px.

    Returns:
        ``synth.Axis`` или None.
    """
    h, w = int(round(comp_seg.shape[0] / seg)), int(round(comp_seg.shape[1] / seg))
    ys, xs = np.nonzero(comp_seg)
    x0, x1 = max(0, int(xs.min() / seg) - BOX_PAD), min(w, int(xs.max() / seg) + BOX_PAD)
    y0, y1 = max(0, int(ys.min() / seg) - BOX_PAD), min(h, int(ys.max() / seg) + BOX_PAD)
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    warped = donor_view(donor, pm, gx, gy, seg, donor_scale)
    mask = cv2.resize(comp_seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1] > 0
    ink = np.zeros((h, w), bool)
    ink[y0:y1, x0:x1] = S.ink_of(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)) & mask
    return S.fit_axis(ink, xh_hint)


def paper_tone(b_strip: np.ndarray, base_v: int, x_height: float) -> np.ndarray | None:
    """Тон бумаги полосы: по столбцам — медиана бумаги основы над строкой без краски, продлённая и сглаженная.

    Args:
        b_strip: Полоса основы BGR.
        base_v: Строка базовой линии.
        x_height: Высота буквы, px.

    Returns:
        Бумага (h, w, 3) float32 или None, если бумаги над строкой нет.
    """
    gray = cv2.cvtColor(b_strip, cv2.COLOR_BGR2GRAY)
    ink = S.ink_of(gray)
    good = ~cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    good &= gray > 0.75 * np.percentile(gray, 90)
    good[int(base_v - 1.3 * x_height) :] = False
    tone = np.full((b_strip.shape[1], 3), np.nan, np.float32)
    for u in range(b_strip.shape[1]):
        if good[:, u].sum() >= 3:
            tone[u] = np.median(b_strip[good[:, u], u], axis=0)
    valid = ~np.isnan(tone[:, 0])
    if valid.sum() < 10:
        return None
    idx = np.arange(len(tone))
    tone = np.stack([np.interp(idx, idx[valid], tone[valid, c]) for c in range(3)], axis=1)
    tone = ndimage.gaussian_filter1d(tone, 25, axis=0)
    return np.repeat(tone[None], b_strip.shape[0], axis=0).astype(np.float32)


def sheet_edge(d_strip: np.ndarray, base_v: int, x_height: float) -> np.ndarray:
    """Кромка нового листа по столбцам полосы: низ бумаги донора под строкой, сглаженный.

    Args:
        d_strip: Полоса донора BGR.
        base_v: Строка базовой линии.
        x_height: Высота буквы, px.

    Returns:
        Строка кромки по столбцам (float).
    """
    gray = cv2.cvtColor(d_strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    paper = gray > 0.6 * np.percentile(gray[:base_v], 80)
    edge = np.empty(d_strip.shape[1], float)
    for u in range(d_strip.shape[1]):
        dark = np.nonzero(~paper[base_v:, u])[0]
        edge[u] = base_v + (dark[0] if len(dark) else d_strip.shape[0] - base_v) - 1
    edge = ndimage.gaussian_filter1d(ndimage.median_filter(edge, 61, mode="nearest"), 15, mode="nearest")
    return np.clip(edge, base_v + S.EDGE_MIN * x_height, d_strip.shape[0] - 2)


def donor_crop(donor: np.ndarray, pm, axis: S.Axis, span: tuple[int, int], seg: float, donor_scale: float):
    """Кусок исходного кадра донора вокруг строки и :data:`CONTEXT_ROWS` строк над ней (для DeepSeek).

    Args:
        donor: Кадр (2).
        pm: ``PageMap``.
        axis: Ось строки в (1).
        span: Отрезок x строки в (1).
        seg: Масштаб совмещения.
        donor_scale: Масштаб уменьшенного (2).

    Returns:
        BGR вырезка, высота около :data:`LLM_CROP_HEIGHT`.
    """
    xh = axis.x_height
    xs = np.linspace(span[0], span[1], 50).astype(np.float32)
    top = axis.y(xs) - (CONTEXT_ROWS * 2.1 + 1.6) * xh
    bot = axis.y(xs) + 0.8 * xh
    px = np.concatenate([xs, xs])[None, :]
    py = np.concatenate([top, bot])[None, :].astype(np.float32)
    dx = seg_to_frame(pm.map_x, px, py, seg, donor_scale)
    dy = seg_to_frame(pm.map_y, px, py, seg, donor_scale)
    x0, x1 = int(max(0, dx.min())), int(min(donor.shape[1], dx.max()))
    y0, y1 = int(max(0, dy.min())), int(min(donor.shape[0], dy.max()))
    crop = donor[y0:y1, x0:x1]
    scale = LLM_CROP_HEIGHT / max(1, crop.shape[0])
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


def full_line_span(
    donor: np.ndarray, pm, axis: S.Axis, piece: tuple[int, int], page: tuple[int, int], seg: float, donor_scale: float
) -> tuple[int, int]:
    """Вся строка по x: краска донора в зоне строчных букв вдоль оси, от куска в обе стороны, пока промежутки
    меньше :data:`MERGE_GAP` высот буквы (видимая часть строки у корешка тоже входит в набор).

    Args:
        donor: Кадр (2).
        pm: ``PageMap``.
        axis: Ось строки.
        piece: Отрезок x срезанной части строки.
        page: Отрезок x страницы.
        seg: Масштаб совмещения.
        donor_scale: Масштаб уменьшенного (2).

    Returns:
        Отрезок x строки.
    """
    xh = axis.x_height
    maps = S.strip_maps(axis, S.GLYPH_UP * xh, S.GLYPH_DOWN * xh, page[0], page[1])
    strip = donor_view(donor, pm, maps[0], maps[1], seg, donor_scale)
    ink = S.ink_of(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY))
    top = int(S.GLYPH_UP * xh)
    cols = ink[int(top - 0.9 * xh) : int(top + 0.1 * xh)].any(axis=0)
    cols = ndimage.binary_closing(cols, structure=np.ones(int(MERGE_GAP * xh)))
    labels, _ = ndimage.label(cols)
    a, b = piece[0] - page[0], piece[1] - page[0]
    ids = {int(v) for v in np.unique(labels[max(0, a) : max(a + 1, b)]) if v > 0}
    if not ids:
        return piece
    idx = np.nonzero(np.isin(labels, list(ids)))[0]
    return page[0] + int(idx[0]), page[0] + int(idx[-1]) + 1


def last_row_axis(donor: np.ndarray, pm, axis: S.Axis, span: tuple[int, int], seg: float, donor_scale: float):
    """Ось последней строки колонки по донору: в высокой полосе донора вдоль грубой оси — самая нижняя строка с
    краской по большей части отрезка; ось подгоняется по низам её краски.

    В доноре страница целая, так что последняя строка колонки в нём — ровно та, что срезана в (1). Грубая ось
    (по маске кусков) может лечь на строку выше, если в маску попал её кусок.

    Args:
        donor: Кадр (2).
        pm: ``PageMap``.
        axis: Грубая ось.
        span: Отрезок x строки.
        seg: Масштаб совмещения.
        donor_scale: Масштаб уменьшенного (2).

    Returns:
        Ось последней строки (или грубая, если строк не нашлось).
    """
    xh = axis.x_height
    up, down = 3.5 * xh, 4.0 * xh
    maps = S.strip_maps(axis, up, down, span[0], span[1])
    strip = donor_view(donor, pm, maps[0], maps[1], seg, donor_scale)
    ink = S.ink_of(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY))
    best = None
    for b in S.all_baselines(ink, xh):
        band = ink[max(0, int(b - 0.9 * xh)) : int(b + 0.15 * xh)]
        if band.size and band.any(axis=0).mean() >= LAST_ROW_COVER:
            best = b
    if best is None:
        return axis
    band = np.zeros_like(ink)
    band[max(0, int(best - 0.9 * xh)) : int(best + 0.15 * xh)] = True
    sel = ink & band
    bottom = np.where(sel.any(axis=0), sel.shape[0] - 1 - np.argmax(sel[::-1], axis=0), -1)
    us = np.nonzero(bottom >= 0)[0]
    fx = maps[0][0, us]
    fy = axis.y(fx) + bottom[us] - int(up)
    found = S.axis_from_points(fx, fy, xh, span[0], span[1])
    return found if found is not None else axis


def snap_to_pitch(base: np.ndarray, axis: S.Axis, text_span: tuple[int, int]) -> S.Axis:
    """Поправка высоты оси по шагу строк основы: нижняя строка стоит на шаг ниже последней видимой строки над ней.

    Ось по донору у самой кромки держится на продлённом (а не измеренном) поле совмещения и может уехать на
    полстроки; строки над ней в (1) видны, и их шаг надёжен.

    Args:
        base: Кадр (1).
        axis: Ось строки по донору.
        text_span: Отрезок x строки.

    Returns:
        Ось со сдвигом (или прежняя, если строк мало или сдвиг велик).
    """
    xh = axis.x_height
    tall_up = S.TALL_UP * xh
    tmaps = S.strip_maps(axis, tall_up, S.STRIP_DOWN * xh, text_span[0], text_span[1])
    tall = cv2.remap(base, tmaps[0], tmaps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rows = S.row_baselines(S.ink_of(cv2.cvtColor(tall, cv2.COLOR_BGR2GRAY)), xh, int(tall_up))
    if len(rows) < 3:
        return axis
    pitch = float(np.median(np.diff(rows)))
    delta = rows[-1] + pitch - int(tall_up)
    if PITCH_SNAP_MIN * xh < abs(delta) < 0.6 * pitch:
        log.info("ось сдвинута по шагу строк на %.1f px", delta)
        return axis.shifted(delta)
    return axis


def glyph_depth(g: S.Glyph) -> float:
    """Глубина краски буквы: медиана (1 − отношения к бумаге) по её краске (пиксели темнее 0.75 бумаги)."""
    r = g.ratio.mean(axis=2)
    ink = r < 0.75
    return float(np.median(1 - r[ink])) if ink.sum() >= 5 else 0.0


def glyph_sharpness(g: S.Glyph) -> float:
    """Резкость буквы: средний модуль градиента в краске, делённый на глубину краски."""
    r = g.ratio.mean(axis=2).astype(np.float32)
    grad = np.hypot(cv2.Sobel(r, cv2.CV_32F, 1, 0), cv2.Sobel(r, cv2.CV_32F, 0, 1))
    edge = (r < 0.9) & (r > 0.3)
    return float(grad[edge].mean() / max(glyph_depth(g), 1e-3)) if edge.sum() >= 5 else 0.0


def blur_glyph(g: S.Glyph, sigma: float) -> S.Glyph:
    """Буква, размытая гауссом ``sigma`` (0 — как есть)."""
    if sigma <= 0:
        return g
    return S.Glyph(cv2.GaussianBlur(g.ratio, (0, 0), sigma), g.top, g.width, g.source)


def match_blur(base: list[S.Glyph], donor: list[S.Glyph]) -> float:
    """Размытие букв донора, при котором их резкость ближе всего к резкости букв основы (перебор сигм).

    Args:
        base: Буквы основы.
        donor: Буквы донора.

    Returns:
        Сигма гаусса, px.
    """
    target = np.median([glyph_sharpness(g) for g in base])
    best, best_err = 0.0, abs(np.median([glyph_sharpness(g) for g in donor]) - target)
    for sigma in BLUR_STEPS:
        err = abs(np.median([glyph_sharpness(blur_glyph(g, sigma)) for g in donor]) - target)
        if err < best_err:
            best, best_err = sigma, err
    return best


def synth_column(
    base: np.ndarray,
    donor: np.ndarray,
    pm,
    axis: S.Axis,
    span: tuple[int, int],
    text_span: tuple[int, int],
    seg: float,
    donor_scale: float,
    cache_dir: Path,
    bottom: bool = True,
    shared: S.Library | None = None,
) -> tuple[np.ndarray, ColumnReport, tuple[int, int, int, int]]:
    """Синтез нижней строки одной колонки.

    Args:
        base: Кадр (1) BGR.
        donor: Кадр (2) BGR.
        pm: ``PageMap``.
        axis: Ось строки колонки.
        span: Отрезок x полосы (колонка с половиной промежутков до соседей): здесь продлевается лист.
        text_span: Отрезок x самой строки (по краске донора): здесь читается и набирается текст.
        seg: Масштаб совмещения.
        donor_scale: Масштаб уменьшенного (2).
        cache_dir: Кэш DeepSeek (рядом — полосы для отсмотра).
        bottom: Нижняя строка колонки: лист продлевается до кромки донора; иначе вклейка — только до середины
            промежутка до строки ниже (её набирает своя полоса).
        shared: Общая библиотека букв кадра: недостающие буквы берутся из неё, свои образцы в неё добавляются.

    Returns:
        ``(кадр, отчёт, окно изменений)``.
    """
    h, w = base.shape[:2]
    xh = axis.x_height
    sx0, sx1 = span
    up, down = S.STRIP_UP * xh, S.STRIP_DOWN * xh
    base_v = int(up)
    g_up, g_down = int(S.GLYPH_UP * xh), int(S.GLYPH_DOWN * xh)
    maps = S.strip_maps(axis, up, down, sx0, sx1)
    b_strip = cv2.remap(base, maps[0], maps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    d_strip = donor_view(donor, pm, maps[0], maps[1], seg, donor_scale)
    t0, t1 = max(0, text_span[0] - sx0), text_span[1] - sx0
    # Библиотека букв и контекст: строки этой колонки над нижней в основе (выпрямлены вдоль её оси).
    library = S.Library(fallback=shared)
    tall_up = S.TALL_UP * xh
    tmaps = S.strip_maps(axis, tall_up, down, text_span[0], text_span[1])
    tall = cv2.remap(base, tmaps[0], tmaps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rows = S.row_baselines(S.ink_of(cv2.cvtColor(tall, cv2.COLOR_BGR2GRAY)), xh, int(tall_up))
    context: list[str] = []
    tall_ink = S.ink_of(cv2.cvtColor(tall, cv2.COLOR_BGR2GRAY))
    for b in rows:
        if b - g_up < 0:
            continue
        # Своя ось строки: низы краски в зоне строчных букв этой строки, пересчитанные в кадр (строки выше
        # нижней идут под своим наклоном и изгибом — выпрямлять их по оси нижней нельзя).
        band = np.zeros_like(tall_ink)
        band[int(b - 0.9 * xh) : int(b + 0.15 * xh)] = True
        col_bottom = np.where(
            (tall_ink & band).any(axis=0),
            (tall_ink & band).shape[0] - 1 - np.argmax((tall_ink & band)[::-1], axis=0),
            -1,
        )
        us = np.nonzero(col_bottom >= 0)[0]
        fx = tmaps[0][0, us]
        fy = axis.y(fx) + col_bottom[us] - int(tall_up)
        row_axis = S.axis_from_points(fx, fy, xh, text_span[0], text_span[1])
        if row_axis is None:
            continue
        rmaps = S.strip_maps(row_axis, g_up, g_down, text_span[0], text_span[1])
        sub = cv2.remap(base, rmaps[0], rmaps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        S.harvest_chars(sub, g_up, library, "base")
        context.append(" ".join(wd[0] for wd in S.tesseract_words(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY))))
    d_line = d_strip[base_v - g_up : base_v + g_down, t0:t1]
    words = S.tesseract_words(cv2.cvtColor(d_line, cv2.COLOR_BGR2GRAY))
    d_glyphs = S.line_glyphs(d_line, g_up, "donor")
    for ch, _, g in d_glyphs:
        library.add(ch, g)
    draft = " ".join(wd[0] for wd in words)
    if shared is not None:
        library.merge_into(shared)
    dbg = cache_dir.parent / "strips"
    dbg.mkdir(parents=True, exist_ok=True)
    tag = f"{sx0}_{int(axis.y(np.array([sx0]))[0])}"
    cv2.imwrite(str(dbg / f"{tag}_donor.jpg"), d_strip)
    cv2.imwrite(str(dbg / f"{tag}_base.jpg"), b_strip)
    crop = donor_crop(donor, pm, axis, text_span, seg, donor_scale)
    text = line_text(crop, "\n".join(context[-CONTEXT_ROWS:]), draft, cache_dir)
    # Проверка: модель должна прочесть строку, а не сочинить — сходство с черновиком tesseract по донору.
    match = float(fuzz.ratio(text.lower(), draft.lower())) if draft else 0.0
    box = (sx0, 0, sx1, 0)
    if match < MIN_DRAFT_MATCH:
        return base, ColumnReport("skip_text", text, draft, match), box
    missing = "".join(sorted({c for c in text if not c.isspace() and library.pick(c) is None}))
    words_abs = [(wd[0], wd[1] + t0, wd[2], wd[3], wd[4], wd[5]) for wd in words]
    # Буквы донора — к виду букв основы: размытие (по резкости перепадов краски) и сила краски (по медианной глубине).
    base_glyphs = [g for gs in library.samples.values() for g in gs if g.source == "base"]
    if len(base_glyphs) >= 20 and len(d_glyphs) >= 5:
        sigma = match_blur(base_glyphs, [g for _, _, g in d_glyphs])
        blurred = [(c, x, blur_glyph(g, sigma)) for c, x, g in d_glyphs]
        depth_b = np.median([glyph_depth(g) for g in base_glyphs])
        depth_d = np.median([glyph_depth(g) for _, _, g in blurred])
        gain = float(np.clip(depth_b / max(depth_d, 1e-3), *INK_GAIN))
        d_glyphs = [(c, x, S.Glyph(np.clip(1 - gain * (1 - g.ratio), 0, 1.2), g.top, g.width, g.source))
                    for c, x, g in blurred]  # fmt: skip
    placed = S.layout_by_donor(text, d_glyphs, library, xh, t0)
    paper = paper_tone(b_strip, base_v, xh)
    if paper is None:
        return base, ColumnReport("no_paper", text, draft, match, missing), box
    canvas, glyph_mask = S.render(paper, placed, base_v)
    # Проверка 1: набранная строка читается как текст.
    c_line = np.clip(canvas[base_v - g_up : base_v + g_down, t0:t1], 0, 255).astype(np.uint8)
    reread = " ".join(wd[0] for wd in S.tesseract_words(cv2.cvtColor(c_line, cv2.COLOR_BGR2GRAY)))
    read_match = float(fuzz.ratio(reread.lower(), text.lower()))
    if read_match < MIN_REREAD:
        return base, ColumnReport("skip_reread", text, draft, read_match, missing), box
    # Что видно в (1): бумага полосы основы (светлое, без коврика), сглаженная.
    b_gray = cv2.cvtColor(b_strip, cv2.COLOR_BGR2GRAY)
    visible = b_gray > 0.6 * np.percentile(b_gray[:base_v], 90)
    visible = cv2.morphologyEx(visible.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    visible = cv2.erode(visible, np.ones((5, 5), np.uint8)) > 0
    # Слова: видимое в (1) целиком остаётся своим; срезанное хотя бы частично набирается целиком.
    word_ids = sorted({p.word for p in placed})
    synth_words = set()
    for wi in word_ids:
        us = [p.u for p in placed if p.word == wi]
        ue = [p.u + p.glyph.ratio.shape[1] for p in placed if p.word == wi]
        a, b = max(0, min(us)), min(visible.shape[1], max(ue))
        zone = visible[int(base_v - 1.2 * xh) : int(base_v + 0.3 * xh), a:b]
        if zone.size and zone.mean() < WORD_VISIBLE:
            synth_words.add(wi)
    # Слово, собранное в основном из библиотечных букв (донор повреждён у корешка), встаёт криво — не набирается
    # (остаётся как было); остальные слова строки набираются.
    shaky = {wi for wi in synth_words
             if np.mean([p.glyph.source != "donor" for p in placed if p.word == wi]) > MAX_LIBRARY_SHARE}  # fmt: skip
    synth_words -= shaky
    log.info(
        "колонка %s: слов %d, к набору %s, сомнительных %s, библиотечных букв %d из %d",
        tag,
        len(word_ids),
        sorted(synth_words),
        sorted(shaky),
        sum(p.glyph.source != "donor" for p in placed),
        len(placed),
    )
    if not synth_words:
        return base, ColumnReport("visible", text, draft, match, missing), box
    # Набранное слово не должно ложиться на видимый текст основы, который с ним не совпадает.
    b_line_ink = S.ink_of(cv2.cvtColor(b_strip, cv2.COLOR_BGR2GRAY)) & visible
    probe, _ = S.render(np.ones_like(paper) * 255, [p for p in placed if p.word in synth_words], base_v)
    p_ink = probe.mean(axis=2) < 200
    near = cv2.dilate(p_ink.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    zone_b = b_line_ink & near
    if zone_b.sum() > MIN_VISIBLE_INK:
        hit = (zone_b & (cv2.dilate(p_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)).sum() / zone_b.sum()
        if hit < MIN_OVERLAP:
            return base, ColumnReport("skip_conflict", text, draft, float(hit) * 100, missing), box
    canvas, glyph_mask = S.render(paper, [p for p in placed if p.word in synth_words], base_v)
    word_zone = np.zeros(visible.shape, bool)
    for wi in synth_words:
        us = [p.u for p in placed if p.word == wi]
        ue = [p.u + p.glyph.ratio.shape[1] for p in placed if p.word == wi]
        a, b = max(0, min(us) - WORD_PAD), min(visible.shape[1], max(ue) + WORD_PAD)
        word_zone[int(base_v - g_up) : int(base_v + g_down), a:b] = True
    edge = sheet_edge(d_strip, base_v, xh) if bottom else np.full(d_strip.shape[1], base_v + LOWER_SEAM * xh)
    # Альфа: 1 от шва в межстрочном промежутке до кромки, рампа над швом, сглаженная кромка.
    vv = np.arange(d_strip.shape[0], dtype=np.float32)[:, None]
    seam, ramp = base_v - 1.55 * xh, S.SEAM_RAMP * xh
    band = np.clip((vv - (seam - ramp)) / ramp, 0, 1) * np.clip(edge[None, :] - vv + 0.5, 0, 1)
    # Вклейка: набранные слова и продление листа там, где в (1) не бумага; видимая бумага (1) не трогается.
    take = (word_zone | ~visible).astype(np.float32)
    take = cv2.GaussianBlur(take, (0, 0), 1.5)
    alpha = band * np.maximum(take, word_zone.astype(np.float32))
    # Обратно в кадр: пиксель (x, y) окна → u = x − sx0, v = y − base(x) + up.
    ends = axis.y(np.array([sx0, (sx0 + sx1) / 2, sx1 - 1]))
    by0, by1 = int(max(0, ends.min() - up - 2)), int(min(h, ends.max() + down + 2))
    yy, xx = np.mgrid[by0:by1, sx0:sx1].astype(np.float32)
    vmap = (yy - axis.y(xx) + up).astype(np.float32)
    umap = (xx - sx0).astype(np.float32)
    back = cv2.remap(canvas, umap, vmap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a_back = cv2.remap(alpha, umap, vmap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    log.info("колонка %s: альфа в полосе %.0f, в кадре %.0f", tag, float(alpha.sum()), float(a_back.sum()))
    out = base.copy()
    roi = out[by0:by1, sx0:sx1].astype(np.float32)
    out[by0:by1, sx0:sx1] = np.clip(roi * (1 - a_back[..., None]) + back * a_back[..., None], 0, 255).astype(np.uint8)
    return out, ColumnReport("synth", text, draft, match, missing), (sx0, by0, sx1, by1)


def merge_pieces(pieces: list[np.ndarray], x_height_seg: float) -> list[np.ndarray]:
    """Куски срезанного ряда → строки: склеиваются куски на одной оси с промежутком по x меньше :data:`MERGE_GAP`.

    Args:
        pieces: Маски кусков (масштаб совмещения).
        x_height_seg: Высота буквы на масштабе совмещения.

    Returns:
        Маски строк.
    """
    info = []
    for m in pieces:
        ys, xs = np.nonzero(m)
        deg = 2 if np.ptp(xs) > 400 else 1
        info.append(
            (xs.min(), xs.max(), np.polyfit(xs, ys, deg), float(np.median(m.sum(axis=0)[m.any(axis=0)])), xs, ys)
        )
    parent = list(range(len(pieces)))
    for i in range(len(pieces)):
        for j in range(len(pieces)):
            if i == j:
                continue
            a0, a1, poly, thick, _, _ = info[i]
            b0, b1, _, _, xs, ys = info[j]
            gap = max(0, b0 - a1, a0 - b1)
            if gap >= MERGE_GAP * x_height_seg:
                continue
            # Высота на стыке: конец куска j, ближний к i, против оси i, продлённой не дальше промежутка.
            near = xs <= b0 + 30 if b0 >= a1 else xs >= b1 - 30
            if np.median(np.abs(ys[near] - np.polyval(poly, xs[near]))) < 0.5 * thick:
                ri, rj = i, j
                while parent[ri] != ri:
                    ri = parent[ri]
                while parent[rj] != rj:
                    rj = parent[rj]
                parent[rj] = ri
    groups: dict[int, np.ndarray] = {}
    for i, m in enumerate(pieces):
        r = i
        while parent[r] != r:
            r = parent[r]
        groups[r] = groups[r] | m if r in groups else m.copy()
    return list(groups.values())


def synth_page_bottom(
    base: np.ndarray,
    donor: np.ndarray,
    pm,
    pieces: list[np.ndarray],
    page_seg: np.ndarray,
    seg: float,
    donor_scale: float,
    x_height_seg: float,
    cache_dir: Path,
    shared: S.Library | None = None,
) -> SynthResult:
    """Синтез срезанных нижних строк страницы.

    Куски ряда склеиваются в строки (каждая строка набирается целиком). Нижняя строка колонки (под ней нет другой
    срезанной строки) продлевает лист до кромки донора, её полоса по x — от середины промежутка до соседней нижней
    строки (или от края страницы): кромка идёт через всю страницу. Строки над ней (на загнутом углу срезано
    несколько строк) вклеиваются только до середины межстрочного промежутка.

    Args:
        base: Кадр (1) BGR.
        donor: Кадр (2) BGR.
        pm: ``PageMap`` страницы.
        pieces: Куски срезанного нижнего ряда (``CutResult.pieces``).
        page_seg: Маска страницы (масштаб совмещения).
        seg: Масштаб совмещения.
        donor_scale: Масштаб уменьшенного (2).
        x_height_seg: Высота буквы на масштабе совмещения.
        cache_dir: Кэш DeepSeek.
        shared: Общая библиотека букв кадра.

    Returns:
        Итог.
    """
    pcols = np.nonzero(page_seg.any(axis=0))[0]
    p0, p1 = int(pcols.min() / seg), int((pcols.max() + 1) / seg)
    lines = [m for m in merge_pieces(pieces, x_height_seg) if m.any(axis=0).sum() >= MIN_COLUMN_SHARE * len(pcols)]
    spans = []
    for m in lines:
        ys, xs = np.nonzero(m)
        spans.append((xs.min(), xs.max(), ys.mean()))
    # Нижняя строка колонки: ни одна другая строка не лежит ниже с перекрытием по x больше трети.
    bottom = []
    for i, (a0, a1, cy) in enumerate(spans):
        below = any(cy2 > cy and min(a1, b1) - max(a0, b0) > 0.33 * min(a1 - a0, b1 - b0)
                    for j, (b0, b1, cy2) in enumerate(spans) if j != i)  # fmt: skip
        bottom.append(not below)
    order = sorted(range(len(lines)), key=lambda i: spans[i][0])
    bottoms = [i for i in order if bottom[i]]
    result = SynthResult(base)
    for i in order:
        if not bottom[i]:
            # Строки над нижней не трогаются: набор только самой нижней строки колонки.
            continue
        c0, c1 = int(spans[i][0] / seg), int((spans[i][1] + 1) / seg)
        axis = column_axis(donor, pm, lines[i], seg, donor_scale, x_height_seg / seg)
        if axis is None:
            result.columns.append(ColumnReport("no_axis"))
            continue
        c0, c1 = full_line_span(donor, pm, axis, (c0, c1), (p0, p1), seg, donor_scale)
        axis = last_row_axis(donor, pm, axis, (c0, c1), seg, donor_scale)
        c0, c1 = full_line_span(donor, pm, axis, (c0, c1), (p0, p1), seg, donor_scale)
        if bottom[i]:
            k = bottoms.index(i)
            left = p0 if k == 0 else (int((spans[bottoms[k - 1]][1] + 1) / seg) + c0) // 2
            right = p1 if k == len(bottoms) - 1 else (c1 + int(spans[bottoms[k + 1]][0] / seg)) // 2
        else:
            pad = int(SIDE_PAD * axis.x_height)
            left, right = max(p0, c0 - pad), min(p1, c1 + pad)
        image, report, box = synth_column(result.image, donor, pm, axis, (left, right), (c0, c1), seg, donor_scale,
                                          cache_dir, bottom[i], shared)  # fmt: skip
        result.image = image
        result.columns.append(report)
        if report.status == "synth":
            result.boxes.append(box)
    return result
