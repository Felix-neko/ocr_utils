"""Перспективный масштаб страницы по её высоте на снимке: модель высоты вне зоны у корешка, плоская координата ``(x − x_c)/H(x)``, ширина прямоугольника первого преобразования и параметр Кунса вдоль страницы по плоскости (буквы у ближнего к камере края не шире, чем у корешка)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import PagePose
from ocr_utils.scan_cropping.v2.rectify.coons import Patch

# Отсчётов вдоль страницы для профиля высоты и таблицы параметра.
SAMPLES = 401
# Итераций перевзвешивания прямой Хубера.
HUBER_ITERATIONS = 20


class ScaleSkip(StrEnum):
    """Почему ширина и параметр страницы остались по длине дуги."""

    # Масштаб посчитан.
    NONE = ""
    # Страница приподнята или опущена (``page_pose``): у наклонённого листа сильный ракурс, высота не мера масштаба
    # (ХиП 1991, 18 разворотов: сужение до 0.80 при соседней плоской странице без изменения).
    TILTED = "tilted"
    # Вне зоны у корешка слишком короткий участок страницы для модели высоты (обрывок, полоска).
    SHORT = "short"


@dataclass(frozen=True)
class ScaleParams:
    """Параметры меры масштаба по высоте страницы.

    Поля:
        spine_zone_mm: Зона у корешка, где высота страницы на снимке не берётся, мм: глубина впадины у сгиба плавает
            от сбоев сегментации. 10, 20 и 30 мм на 934 плоских разворотах ХиП 1991 дают одно и то же (|R−L| 2.1 мм).
        min_span_mm: Наименьший участок страницы вне зоны у корешка для модели, мм.
        max_shrink: Наибольшее сужение ширины к ширине по дуге: не меньше ``1 − max_shrink``. ХиП 1991, 1868 плоских
            страниц: коэффициент p1 0.943, медиана 0.986.
        max_grow: Наибольшее расширение: не больше ``1 + max_grow``. ХиП 1991: p99 1.048 (страница, внешний край
            которой дальше от камеры, чем корешок).
        huber_share: Порог Хубера прямой высоты, доля средней высоты страницы: дальше — выброс (палец, надрыв края).
    """

    spine_zone_mm: float = 20.0
    min_span_mm: float = 30.0
    max_shrink: float = 0.10
    max_grow: float = 0.06
    huber_share: float = 0.003


@dataclass(frozen=True)
class PageScale:
    """Мера масштаба страницы и ширина её прямоугольника.

    Поля:
        arc_width: Ширина по длине дуги верха и низа, px (прежняя ширина прямоугольника).
        height: Высота прямоугольника страницы, px (у разворота — сторона у корешка).
        aspect: Ширина / высота страницы по масштабу высоты ``∫ |d mid| / H(s)`` или NaN — не считалось.
        target_width: Ширина прямоугольника, px.
        ratio: ``target_width / arc_width``.
        clamped: Отношение упёрлось в предел.
        skip: Почему ширина и параметр остались по дуге.
        u_warp: Таблица параметра вдоль страницы ``(u, t)`` для ``Patch.u_warp`` или None.
    """

    arc_width: float
    height: float
    aspect: float
    target_width: float
    ratio: float
    clamped: bool
    skip: ScaleSkip = ScaleSkip.NONE
    u_warp: tuple[np.ndarray, np.ndarray] | None = None


def height_profile(patch: Patch, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Профиль высоты страницы вдоль неё.

    Args:
        patch: Область страницы (верх и низ слева направо, параметр — доля длины дуги).
        centre: Главная точка снимка (центр кадра) в px итога.

    Returns:
        ``(t, s, x, H)``: доли длины дуги верха и низа (SAMPLES,); длина по средней линии от левого края, px;
        координата средней линии вдоль хорды страницы (слева направо) от проекции главной точки, px; расстояние между
        точками верха и низа с одной долей, px (пары идут вдоль вертикалей страницы, поворот книги в кадре и наклон
        линии корешка его не меняют).
    """
    t = np.linspace(0.0, 1.0, SAMPLES)
    top, bottom = patch.top.at(t), patch.bottom.at(t)
    mid = (top + bottom) / 2
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(mid, axis=0).T))]
    chord = mid[-1] - mid[0]
    x = (mid - np.asarray(centre, np.float64)) @ (chord / max(float(np.linalg.norm(chord)), 1e-12))
    return t, s, x, np.hypot(*(bottom - top).T)


def huber_line(x: np.ndarray, y: np.ndarray, delta: float) -> np.ndarray:
    """Прямая ``y = a·x + b`` с потерей Хубера (перевзвешенный МНК).

    Args:
        x: Абсциссы.
        y: Ординаты.
        delta: Порог Хубера, в единицах ``y``.

    Returns:
        Коэффициенты ``(a, b)`` для ``np.polyval``.
    """
    coef = np.polyfit(x, y, 1)
    for _ in range(HUBER_ITERATIONS):
        r = np.abs(y - np.polyval(coef, x))
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
        coef = np.polyfit(x, y, 1, w=np.sqrt(w))
    return coef


def page_scale(patch: Patch, height: float, spine_right: bool | None, pose: PagePose | None, centre: np.ndarray,
               px_per_mm: float, params: ScaleParams) -> PageScale:  # fmt: skip
    """Ширина прямоугольника и параметр вдоль страницы по перспективному масштабу её высоты.

    У плоской страницы под перспективой глубина ``z`` аффинна по плоскости, ``1/z`` аффинна по координате снимка ``x``,
    а высота страницы на снимке ``H(x) ∝ 1/z`` — прямая по ``x``. Координата плоскости ``X ∝ (x − x_c)·z ∝
    (x − x_c)/H(x)``, где ``x_c`` — проекция главной точки (центра кадра): так учитывается и местный масштаб, и ракурс
    (горизонтальный масштаб ``∝ 1/z²``, а не ``1/z``), фокус не нужен. Остаётся множитель ``cos θ`` наклона листа —
    второй порядок. Высота берётся вне зоны у корешка (глубина впадины у сгиба плавает от сбоев сегментации) и
    приближается прямой Хубера по ``x``, в зоне у корешка — её продолжение. Пропорции ``P(конец) − P(начало)``,
    ``P = (x − x_c)/H``; ширина — высота прямоугольника × пропорции в пределах ``params``; параметр Кунса
    ``u = (P − P(начало)) / пропорции``.

    ХиП 1991, 934 плоских разворота: разница пропорций правой и левой страницы, ×190 мм — медиана 5.0 → 1.85 мм, p90
    9.8 → 4.4 мм (``∫ dx / H`` без ракурса — 2.8 и 6.4 мм).

    Args:
        patch: Область страницы в px итога.
        height: Высота её прямоугольника, px.
        spine_right: Где корешок: True — справа (левая страница), False — слева (правая), None — нет (одна страница,
            высота берётся по всей ширине).
        pose: Положение страницы или None; приподнятая и опущенная — без коррекции.
        centre: Главная точка снимка (центр кадра) в px итога.
        px_per_mm: Плотность итога, px/мм.
        params: Параметры.

    Returns:
        Мера страницы.
    """
    arc_width = (patch.top.length + patch.bottom.length) / 2
    if pose in (PagePose.RAISED, PagePose.LOWERED):
        return PageScale(arc_width, height, float("nan"), arc_width, 1.0, False, ScaleSkip.TILTED)
    t, s, x, h = height_profile(patch, centre)
    # Расстояние до корешка по средней линии; без корешка — зона пуста.
    to_spine = (s[-1] - s if spine_right else s) if spine_right is not None else np.full_like(s, np.inf)
    keep = to_spine >= params.spine_zone_mm * px_per_mm
    if keep.sum() < 2 or np.ptp(s[keep]) < params.min_span_mm * px_per_mm:
        return PageScale(arc_width, height, float("nan"), arc_width, 1.0, False, ScaleSkip.SHORT)
    coef = huber_line(x[keep], h[keep], params.huber_share * float(h[keep].mean()))
    model = np.polyval(coef, x)
    if model.min() <= 0:
        return PageScale(arc_width, height, float("nan"), arc_width, 1.0, False, ScaleSkip.SHORT)
    # Координата плоскости в высотах страницы; монотонность — защита от изгиба средней линии у краёв.
    plane = np.maximum.accumulate(x / model)
    plane = plane - plane[0]
    aspect = float(plane[-1])
    if aspect <= 0:
        return PageScale(arc_width, height, float("nan"), arc_width, 1.0, False, ScaleSkip.SHORT)
    raw = height * aspect / arc_width
    ratio = float(np.clip(raw, 1.0 - params.max_shrink, 1.0 + params.max_grow))
    # Таблица параметра — только строго возрастающие доли: средняя линия у корешка может чуть загибаться назад
    # (координата вдоль хорды не монотонна), и повтор доли давал деление на ноль в наклоне продолжения за край.
    u = plane / aspect
    rising = np.r_[True, np.diff(u) > 1e-9]
    return PageScale(arc_width, height, aspect, arc_width * ratio, ratio, bool(ratio != raw), ScaleSkip.NONE,
                     (u[rising], t[rising]))  # fmt: skip


def scaled_patch(patch: Patch, scale: PageScale) -> Patch:
    """Область с параметром вдоль страницы по плоскости (без таблицы у меры — та же область).

    Args:
        patch: Область.
        scale: Мера страницы.

    Returns:
        Область с ``u_warp``.
    """
    return patch if scale.u_warp is None else replace(patch, u_warp=scale.u_warp)


def scale_fields(scales: tuple[PageScale, ...], px_per_mm: float) -> dict:
    """Поля сводки кадра по мере масштаба страниц.

    Args:
        scales: Меры страниц (``TextResult.scales``); пусто — мера выключена.
        px_per_mm: Плотность итога, px/мм.

    Returns:
        ``scale_<L|R|P>_{arc_mm, width_mm, ratio, clamped, skip}`` (у одной страницы — ``P``); пусто — меры нет.
    """
    names = "P" if len(scales) == 1 else "LR"
    out = {}
    for name, sc in zip(names, scales):
        out |= {f"scale_{name}_arc_mm": round(sc.arc_width / px_per_mm, 2),
                f"scale_{name}_width_mm": round(sc.target_width / px_per_mm, 2), f"scale_{name}_ratio": round(sc.ratio, 4),
                f"scale_{name}_clamped": int(sc.clamped), f"scale_{name}_skip": sc.skip.value}  # fmt: skip
    return out


def scale_note(scales: tuple[PageScale, ...]) -> str:
    """Подпись листа: коэффициенты ширины страниц мерой масштаба.

    Args:
        scales: Меры страниц; пусто — мера выключена.

    Returns:
        Строка вида ``ширина страниц по масштабу высоты: ×1.004 | ×0.928 (предел)`` или пустая.
    """
    if not scales:
        return ""
    parts = []
    for sc in scales:
        mark = " (предел)" if sc.clamped else f" ({sc.skip.value})" if sc.skip is not ScaleSkip.NONE else ""
        parts.append(f"×{sc.ratio:.3f}{mark}")
    return "ширина страниц по масштабу высоты: " + " | ".join(parts)
