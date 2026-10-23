"""Двойная грубая сегментация страниц: SAM 3 по кадру как есть (база) и по кадру с растянутыми уровнями — только в зонах у
внешних краёв книги (``--dual-levels``).

Растяжение уровней (``paper.stretch_paper_levels``) правит рваные внешние края страниц (ПХ 1975/09 IMG_0045 L,
ХиП 07/IMG_0120), но чаще даёт ямы у корешка и губит обложки (12/IMG_0085 — 0 страниц); на 240 случайных сторонах
сырой SAM 3 ≥ 1,05: 58 → 42 (stack_edges/coarse_levels/rand). Поэтому база — без растяжения, а растянутый проход берётся
лишь в полосах вдоль левой и правой кромок книги (прямые ``spread_lines.book_edge`` по грубой маске книги, до SAM 3
страниц не нужно). Явные промахи растянутого прохода обходятся: нет страниц, мало бумаги в зоне, экземпляр в зоне не
сопоставлен со страницей базы — зона остаётся на базе.

У верхних и нижних краёв страниц (``--levels-edges``, ``union_levels_edges``) — не замена, а объединение: к базе
добавляется то, что растянутый проход видит в зоне краёв, а база — нет (выемки края у корешка, дыры у кромки).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.side_fit import Side
from ocr_utils.scan_cropping.v2.spread_lines import book_edge

# Зона у кромки книги: столько наружу и внутрь от прямой кромки, мм бумаги (наружу — запас на наклон кромки; внутрь —
# торцы стопки и обложка 5–15 мм плюс языки рваного края до ~10 мм).
ZONE_OUT_MM = 5.0
ZONE_IN_MM = 30.0
# Растянутый проход в зоне теряет больше этой доли бумаги базы — промах, зона на базе.
ZONE_MAX_LOSS = 0.25
# Строки книги для пролёта прямой кромки: доли высоты книги, отбрасываемые сверху и снизу.
BOOK_ROWS_CUT = 0.02


class ZoneStatus(StrEnum):
    """Итог зоны у кромки книги."""

    USED = "растянутый"
    NO_LINE = "нет кромки книги"
    EMPTY = "растянутый пуст"
    LOSS = "мало бумаги"
    UNMATCHED = "экземпляр не сопоставлен"


@dataclass
class DualMerge:
    """Итог слияния двух проходов SAM 3.

    Поля:
        raw: Сырое объединение экземпляров (база, в принятых зонах — растянутый проход).
        inst: Карта меток экземпляров-страниц базы с заменой в принятых зонах.
        zones: Сторона → маска зоны (только построенные зоны).
        status: Сторона → итог зоны.
        frame_status: Итог кадра: пусто — слияние было; иначе причина, почему кадр целиком на базе.
    """

    raw: np.ndarray
    inst: np.ndarray
    zones: dict[Side, np.ndarray] = field(default_factory=dict)
    status: dict[Side, ZoneStatus] = field(default_factory=dict)
    frame_status: str = ""


def outer_zones(book: np.ndarray, hands: np.ndarray, px_per_mm: float = SEG_PX_PER_MM) -> dict[Side, np.ndarray]:
    """Зоны у левой и правой кромок книги по прямым края грубой маски книги.

    Пролёт прямой — строки книги без ``BOOK_ROWS_CUT`` сверху и снизу; зона — по всем строкам книги, от прямой на
    :data:`ZONE_OUT_MM` наружу и :data:`ZONE_IN_MM` внутрь. Сторона без прямой (точек мало) — без зоны.

    Args:
        book: Грубая маска книги (не фон), масштаб сегментации.
        hands: Руки и зона закраса пальцев (строки у них не идут в прямую кромки).
        px_per_mm: Масштаб, px/мм.

    Returns:
        Сторона (``Side.LEFT`` / ``Side.RIGHT``) → маска зоны (bool).
    """
    rows = np.flatnonzero(book.any(axis=1))
    if len(rows) == 0:
        return {}
    top, bot = float(rows[0]), float(rows[-1])
    cut = BOOK_ROWS_CUT * (bot - top)
    h, w = book.shape
    ys = np.arange(h, dtype=np.float64)
    xs = np.arange(w)[None, :]
    in_rows = (ys >= top) & (ys <= bot)
    out_px, in_px = ZONE_OUT_MM * px_per_mm, ZONE_IN_MM * px_per_mm
    zones = {}
    for side in (Side.LEFT, Side.RIGHT):
        edge = book_edge(book, hands, side, (top + cut, bot - cut))
        if edge.line is None:
            continue
        x = edge.line.x_at(ys)[:, None]
        # Наружу у левой кромки — влево, у правой — вправо.
        lo, hi = (x - out_px, x + in_px) if side is Side.LEFT else (x - in_px, x + out_px)
        zones[side] = (xs >= lo) & (xs <= hi) & in_rows[:, None]
    return zones


def merge_levels(
    raw: np.ndarray, inst: np.ndarray, raw_lv: np.ndarray, inst_lv: np.ndarray, zones: dict[Side, np.ndarray]
) -> DualMerge:
    """База SAM 3 с растянутым проходом в зонах у кромок книги; промахи растянутого прохода — зона на базе.

    Зона берётся из растянутого прохода, если в ней (1) растянутая бумага не пуста, (2) её не меньше чем
    ``1 − ZONE_MAX_LOSS`` бумаги базы, (3) каждый экземпляр растянутого прохода в зоне сопоставлен с экземпляром-
    страницей базы (наибольшее перекрытие по всему кадру). Метки в зоне — меток базы. Кадр целиком на базе, если у
    базы меньше двух страниц (одиночные страницы и обложки) или растянутый проход не нашёл ни одной.

    Args:
        raw: Сырое объединение экземпляров SAM 3 базы (bool).
        inst: Карта меток экземпляров-страниц базы (uint8, 0 — ничего).
        raw_lv: То же растянутого прохода.
        inst_lv: Карта меток растянутого прохода.
        zones: Зоны у кромок (``outer_zones``).

    Returns:
        Итог слияния; ``raw`` и ``inst`` — новые массивы.
    """
    out = DualMerge(raw.copy(), inst.copy(), dict(zones))
    if int(inst.max()) < 2 or len(np.unique(inst[inst > 0])) < 2:
        out.frame_status = "у базы меньше двух страниц"
        return out
    if not inst_lv.any():
        out.frame_status = "растянутый проход без страниц"
        return out
    # Сопоставление меток растянутого прохода с метками базы — по наибольшему перекрытию по всему кадру.
    mapping = {}
    for j in np.unique(inst_lv[inst_lv > 0]):
        hit = inst[inst_lv == j]
        hit = hit[hit > 0]
        if hit.size:
            mapping[int(j)] = int(np.bincount(hit).argmax())
    for side, zone in zones.items():
        base_area, lv_area = int((raw & zone).sum()), int((raw_lv & zone).sum())
        if lv_area == 0:
            out.status[side] = ZoneStatus.EMPTY
            continue
        if lv_area < (1 - ZONE_MAX_LOSS) * base_area:
            out.status[side] = ZoneStatus.LOSS
            continue
        labels = np.unique(inst_lv[zone & (inst_lv > 0)])
        if any(int(j) not in mapping for j in labels):
            out.status[side] = ZoneStatus.UNMATCHED
            continue
        lut = np.zeros(256, np.uint8)
        for j, i in mapping.items():
            lut[j] = i
        out.raw[zone] = raw_lv[zone]
        out.inst[zone] = lut[inst_lv[zone]]
        out.status[side] = ZoneStatus.USED
    for side in (Side.LEFT, Side.RIGHT):
        out.status.setdefault(side, ZoneStatus.NO_LINE)
    return out


# Зона верхних и нижних краёв страниц для объединения проходов (``union_levels_edges``): по столбцу от крайней строки
# экземпляров-страниц базы — столько наружу и внутрь, мм. Стенд ``research/gutter_strip`` (``levels-top``, ПХ 1975 и
# ХиП 1991, 1980 кадров, 2026-10-23): добавка в книге у верха — медиана 13.4 / 5.2 мм², у низа — 7.0 / 3.6 мм², на фон
# — до 2 мм²; заделывает выемки края у корешка (ПХ 1975/10 IMG_0007, 0003, 0008: верхний угол у корешка поднялся из
# ямы на 1–2.6 мм); на сильных загибах ХиП углы не сдвинулись больше 0.2 мм.
EDGES_OUT_MM = 10.0
EDGES_IN_MM = 15.0


@dataclass(frozen=True)
class EdgesUnion:
    """Объединение проходов у верхних и нижних краёв страниц.

    Поля:
        raw: Сырая маска базы с добавкой.
        inst: Карта экземпляров базы; пиксели добавки — метка ближайшего экземпляра.
        added_top: Добавка у верхних краёв.
        added_bottom: Добавка у нижних краёв.
    """

    raw: np.ndarray
    inst: np.ndarray
    added_top: np.ndarray
    added_bottom: np.ndarray


def edges_zone(pages: np.ndarray, top: bool, px_per_mm: float = SEG_PX_PER_MM) -> np.ndarray:
    """Зона верхних (``top``) или нижних краёв страниц: по столбцу от крайней строки ``pages`` на ``EDGES_OUT_MM``
    наружу и ``EDGES_IN_MM`` внутрь (столбцы без страниц — вне зоны).

    Args:
        pages: Маска страниц (объединение экземпляров базы).
        top: Верхние края (иначе нижние).
        px_per_mm: Плотность масштаба, px/мм.

    Returns:
        Маска зоны.
    """
    h, w = pages.shape
    has = pages.any(axis=0)
    first = np.argmax(pages, axis=0) if top else h - 1 - np.argmax(pages[::-1], axis=0)
    edge = np.where(has, first, 0).astype(np.float64)
    yy = np.arange(h)[:, None]
    out, inside = EDGES_OUT_MM * px_per_mm, EDGES_IN_MM * px_per_mm
    lo, hi = (edge - out, edge + inside) if top else (edge - inside, edge + out)
    return (yy >= lo[None, :]) & (yy <= hi[None, :]) & has[None, :]


def union_levels_edges(
    raw: np.ndarray,
    inst: np.ndarray,
    raw_lv: np.ndarray,
    inst_lv: np.ndarray,
    book: np.ndarray,
    raw_noink: np.ndarray | None = None,
) -> EdgesUnion:
    """Объединение базы SAM 3 с проходом по растянутой светлоте у верхних и нижних краёв страниц (``--levels-edges``).

    У краёв (``edges_zone`` по экземплярам базы) к базе добавляется то, что растянутый проход видит, а база
    (``raw`` ∪ ``raw_noink``) — нет, в пределах грубой книги. Это объединение, а не замена: растянутый проход теряет у
    краёв больше, чем добавляет (на стенде база видит, растянутый нет: медиана 30–93 мм²). Метка добавки — экземпляр
    базы, с которым больше всего перекрывается экземпляр растянутого прохода под пикселем (у корешка ближайший
    экземпляр бывает соседней страницей через щель); нет такого — ближайший экземпляр базы.

    Args:
        raw: Сырая маска базы.
        inst: Карта экземпляров-страниц базы.
        raw_lv: Сырая маска растянутого прохода.
        inst_lv: Карта экземпляров растянутого прохода.
        book: Грубая маска книги.
        raw_noink: Сырая маска второго прохода базы без крупной краски или None.

    Returns:
        Объединение.
    """
    from scipy import ndimage as ndi

    base = raw.astype(bool) | (raw_noink.astype(bool) if raw_noink is not None else False)
    pages = inst > 0
    if not pages.any():
        empty = np.zeros(raw.shape, bool)
        return EdgesUnion(raw.astype(bool), inst, empty, empty)
    added_top = raw_lv.astype(bool) & ~base & edges_zone(pages, True) & book
    added_bottom = raw_lv.astype(bool) & ~base & edges_zone(pages, False) & book
    added = added_top | added_bottom
    out_inst = inst.copy()
    if added.any():
        # Экземпляр растянутого прохода → экземпляр базы с наибольшим перекрытием.
        mapping = np.zeros(int(inst_lv.max()) + 1, inst.dtype)
        for k in np.unique(inst_lv[added]):
            if k == 0:
                continue
            labels = inst[(inst_lv == k) & pages]
            if len(labels):
                mapping[k] = np.bincount(labels).argmax()
        out_inst[added] = mapping[inst_lv[added]]
        # Без сопоставления — ближайший экземпляр базы.
        rest = added & (out_inst == 0)
        if rest.any():
            _, (iy, ix) = ndi.distance_transform_edt(~pages, return_indices=True)
            out_inst[rest] = inst[iy[rest], ix[rest]]
    return EdgesUnion(raw.astype(bool) | added, out_inst, added_top, added_bottom)
