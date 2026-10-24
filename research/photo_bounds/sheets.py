"""Листы отсмотра стенда: все фото подряд (для разметки классов глазами), спорные стороны по поясам расхождения «новая рамка − прежняя», фото по классам края."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

from research.photo_bounds.cache import GRAY_DPI, load_page
from research.photo_bounds.data import CacheLayout, PageRef
from research.photo_bounds.measure import SIDES, best_match
from ocr_utils.page_layout.overlay_frame import header_strip
from research.photo_bounds.bounds import Side
from research.photo_bounds.overlay import photo_card, sheet

# Пояса расхождения сторон «новая − прежняя рамка», мм (наибольший модуль по четырём сторонам).
DISAGREE_BANDS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9))


class SheetKind(StrEnum):
    """Какие листы строить."""

    ALL = "all"  # все эталонные фото по порядку полос (разметка классов)
    DISAGREE = "disagree"  # по поясам расхождения новой и прежней рамки
    CLASS = "class"  # по классам края нового способа
    NEW_CUT = "new_cut"  # новая рамка срезает снимок (снаружи стороны светлый растр)
    NEW_SPARE = "new_spare"  # у новой рамки внутри стороны бумага
    REF_SPARE = "ref_spare"  # у эталона внутри стороны бумага, новая рамка тесная (выборка)
    OBJECT = "object"  # объект снимка за рамкой (эталона, прежней или новой) больше 3 мм²


@dataclass(frozen=True)
class CardSpec:
    """Что нарисовать на карточке: полоса, номер эталона на полосе, подписи."""

    page: PageRef
    ref_index: int
    title: list[str]


def max_disagreement(row: dict) -> float:
    """Наибольший модуль разности сторон новой и прежней рамки, мм (по промахам против эталона)."""
    values = []
    for side in SIDES:
        a, b = row.get(f"new_{side.value}_mm"), row.get(f"det_{side.value}_mm")
        if a not in (None, "") and b not in (None, ""):
            values.append(abs(float(a) - float(b)))
    return max(values) if values else float("nan")


def card_title(row: dict) -> list[str]:
    """Строки шапки карточки по строке мер."""
    deltas = " ".join(
        f"{s.value[0]}{float(row[f'new_{s.value}_mm']) - float(row[f'det_{s.value}_mm']):+.1f}"
        for s in SIDES
        if row.get(f"new_{s.value}_mm") not in (None, "") and row.get(f"det_{s.value}_mm") not in (None, "")
    )
    return [
        f"{row['key']}  {row.get('kind', '')}  {row.get('ref_w_mm', '')}×{row.get('ref_h_mm', '')} мм",
        f"класс {row.get('new_class', '—')}; новая−прежняя, мм: {deltas}; IoU прежн./нов. с эталоном "
        f"{row.get('det_iou', '')}/{row.get('new_iou', '')}",
    ]


def build_card(layout: CacheLayout, run_dir: Path, spec: CardSpec, zoom: bool = True) -> np.ndarray:
    """Карточка одного эталонного фото: эталон, прежняя рамка и новое фото с наибольшим IoU против эталона.

    Args:
        layout: Раскладка кэша.
        run_dir: Папка прогона (``pages/<полоса>.json``).
        spec: Что рисовать.
        zoom: С увеличенными углами.

    Returns:
        Карточка BGR.
    """
    data = load_page(layout, spec.page)
    payload = json.loads((run_dir / "pages" / f"{spec.page.page_key}.json").read_text())
    ref = payload["refs"][spec.ref_index]
    di, _ = best_match(ref, [tuple(b) for b in payload["det"]])
    ni, _ = best_match(ref, [tuple(p["box"]) for p in payload["photos"]])
    det = payload["det"][di] if di >= 0 else None
    photo = payload["photos"][ni] if ni >= 0 else None
    return photo_card(data.gray, GRAY_DPI, spec.title, ref, det, photo, zoom)


def specs_for(rows: list[dict], pages: dict[str, PageRef]) -> list[CardSpec]:
    """Карточки по строкам мер (номер эталона на полосе — по ключу области).

    Args:
        rows: Строки мер эталонных фото.
        pages: Полосы по ключу.

    Returns:
        Описания карточек.
    """
    out = []
    for row in rows:
        page = pages[row["page"]]
        rid = int(row["key"].split("#")[1])
        index = next(i for i, ph in enumerate(page.photos) if ph.region_id == rid)
        out.append(CardSpec(page, index, card_title(row)))
    return out


def write_sheets(
    layout: CacheLayout, run_dir: Path, specs: list[CardSpec], out_dir: Path, name: str, header: str,
    per_sheet: int, cols: int, max_width: int, zoom: bool = True,
) -> list[Path]:  # fmt: skip
    """Листы по ``per_sheet`` карточек: ``out_dir/<name>_NN.jpg``.

    Args:
        layout: Раскладка кэша.
        run_dir: Папка прогона.
        specs: Карточки.
        out_dir: Куда писать.
        name: Префикс имени листа.
        header: Шапка листа.
        per_sheet: Карточек на лист.
        cols: Карточек в ряду.
        max_width: Ширина листа, px (уменьшается до неё).
        zoom: Карточки с увеличенными углами.

    Returns:
        Пути листов.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, start in enumerate(range(0, len(specs), per_sheet), 1):
        part = specs[start : start + per_sheet]
        cards = [build_card(layout, run_dir, s, zoom) for s in part]
        total = (len(specs) + per_sheet - 1) // per_sheet
        picture = sheet(cards, [f"{header} — лист {n}/{total}, фото {start + 1}–{start + len(part)}"], cols)
        if picture.shape[1] > max_width:
            f = max_width / picture.shape[1]
            picture = cv2.resize(picture, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        path = out_dir / f"{name}_{n:02d}.jpg"
        cv2.imwrite(str(path), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    return paths


def disagreement_groups(rows: list[dict]) -> dict[str, list[dict]]:
    """Эталонные фото по поясам расхождения новой и прежней рамки (по убыванию внутри пояса).

    Args:
        rows: Строки мер.

    Returns:
        Имя пояса → строки.
    """
    out = {}
    for lo, hi in DISAGREE_BANDS:
        part = [r for r in rows if lo <= max_disagreement(r) < hi]
        part.sort(key=max_disagreement, reverse=True)
        name = f"{lo:g}-{hi:g}mm" if hi < 1e8 else f"{lo:g}mm+"
        out[name] = part
    return out


def tight_flags(row: dict, method: str, inner: bool, threshold: float) -> list[str]:
    """Стороны строки мер, у которых полоса внутри (``inner``) покрыта растром меньше порога или снаружи — больше.

    Args:
        row: Строка мер.
        method: ``ref``, ``det``, ``new``.
        inner: Полоса внутри рамки (лишнее поле) или снаружи (срез).
        threshold: Порог доли светлого растра.

    Returns:
        Имена сторон.
    """
    out = []
    for side in SIDES:
        v = row.get(f"{method}_{side.value}_{'in' if inner else 'out'}")
        if v in (None, "", "nan"):
            continue
        v = float(v)
        if (inner and v < threshold) or (not inner and v > threshold):
            out.append(side.value)
    return out


# Вырезка середины стороны для проверки глазами: вдоль стороны, мм; поперёк — в обе стороны, мм; увеличение.
VERIFY_ALONG_MM = 6.0
VERIFY_ACROSS_MM = 4.0
VERIFY_ZOOM = 5


def side_crop(gray: np.ndarray, ref, det, photo: dict, side: Side, label: str) -> np.ndarray:
    """Увеличенная вырезка середины стороны новой рамки с эталоном, прежней и новой рамкой и линейкой 1 мм.

    Args:
        gray: Серая полоса стенда (300 dpi).
        ref, det: Эталон и прежняя рамка (px полосы) или None.
        photo: JSON нового фото.
        side: Сторона.
        label: Подпись вырезки.

    Returns:
        Вырезка BGR с подписью.
    """
    from research.photo_bounds.overlay import draw_marks

    h, w = gray.shape
    px = GRAY_DPI / 25.4
    x0, y0, x1, y1 = photo["box"]
    along, across = VERIFY_ALONG_MM * px / 2, VERIFY_ACROSS_MM * px
    if side.horizontal:
        cx, cy = (x0 + x1) / 2, (y0 if side is Side.TOP else y1)
        win = (cx - along, cy - across, cx + along, cy + across)
    else:
        cx, cy = (x0 if side is Side.LEFT else x1), (y0 + y1) / 2
        win = (cx - across, cy - along, cx + across, cy + along)
    a0, b0 = int(max(win[0], 0)), int(max(win[1], 0))
    a1, b1 = int(min(win[2], w)), int(min(win[3], h))
    crop = cv2.cvtColor(gray[b0:b1, a0:a1], cv2.COLOR_GRAY2BGR)
    crop = cv2.resize(crop, None, fx=VERIFY_ZOOM, fy=VERIFY_ZOOM, interpolation=cv2.INTER_NEAREST)
    draw_marks(crop, np.array([a0, b0], float), VERIFY_ZOOM, ref, det, photo, 2)
    # Линейка 1 мм в левом нижнем углу.
    bar = int(round(px * VERIFY_ZOOM))
    cv2.rectangle(crop, (6, crop.shape[0] - 12), (6 + bar, crop.shape[0] - 7), (0, 0, 0), -1)
    return np.vstack([header_strip([label], crop.shape[1]), crop])


def verify_sheet(layout: CacheLayout, run_dir: Path, rows: list[dict], pages: dict, out: Path) -> list[dict]:
    """Лист вырезок середин сторон выбранных фото (по 4 на фото) и таблица для оценки глазами.

    Args:
        layout: Раскладка кэша.
        run_dir: Папка прогона.
        rows: Строки мер выбранных фото.
        pages: Полосы по ключу.
        out: Путь листа (JPEG).

    Returns:
        Строки «номер вырезки, фото, сторона, промахи новой/прежней рамки против эталона» — бланк оценки.
    """
    tiles, table = [], []
    for spec, row in zip(specs_for(rows, pages), rows):
        data = load_page(layout, spec.page)
        payload = json.loads((run_dir / "pages" / f"{spec.page.page_key}.json").read_text())
        ref = payload["refs"][spec.ref_index]
        di, _ = best_match(ref, [tuple(b) for b in payload["det"]])
        ni, _ = best_match(ref, [tuple(p["box"]) for p in payload["photos"]])
        if ni < 0:
            continue
        det = payload["det"][di] if di >= 0 else None
        for side in SIDES:
            n = len(table) + 1
            tiles.append(side_crop(data.gray, ref, det, payload["photos"][ni], side, f"{n}. {row['key']} {side.value}"))
            table.append({"n": n, "key": row["key"], "side": side.value,
                          "new_mm": row.get(f"new_{side.value}_mm"), "det_mm": row.get(f"det_{side.value}_mm")})  # fmt: skip
    size = max(t.shape[1] for t in tiles), max(t.shape[0] for t in tiles)
    padded = [cv2.copyMakeBorder(t, 0, size[1] - t.shape[0], 0, size[0] - t.shape[1], cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255)) for t in tiles]  # fmt: skip
    picture = sheet(padded, [f"Проверка сторон глазами: {len(rows)} фото, {len(tiles)} сторон; линейка — 1 мм"], 6)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), picture, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return table


COLOUR_OBJ_OUT = (0, 0, 220)  # объект за новой рамкой
COLOUR_OBJ = (60, 160, 230)  # объект внутри (справочно, светлее)


def object_card(layout: CacheLayout, run_dir: Path, spec: CardSpec) -> np.ndarray:
    """Карточка «объект за рамкой»: маска объекта снимка (краска и тон, сцепленные с ним) полупрозрачно, часть за
    новой рамкой — красным; эталон, прежняя и новая рамки.

    Args:
        layout: Раскладка кэша.
        run_dir: Папка прогона.
        spec: Что рисовать.

    Returns:
        Карточка BGR.
    """
    from research.photo_bounds.bounds import SEARCH_MM, object_mask, page_maps
    from research.photo_bounds.overlay import CARD_WIDTH, draw_marks

    data = load_page(layout, spec.page)
    payload = json.loads((run_dir / "pages" / f"{spec.page.page_key}.json").read_text())
    ref = payload["refs"][spec.ref_index]
    di, _ = best_match(ref, [tuple(b) for b in payload["det"]])
    ni, _ = best_match(ref, [tuple(p["box"]) for p in payload["photos"]])
    det = payload["det"][di] if di >= 0 else None
    photo = payload["photos"][ni] if ni >= 0 else None
    pad = int(round(SEARCH_MM * GRAY_DPI / 25.4))
    window = tuple(int(v) for v in (ref[0] - pad, ref[1] - pad, ref[2] + pad, ref[3] + pad))
    maps = page_maps(data.gray, window, GRAY_DPI, tuple(int(v) for v in ref))
    core = photo["box"] if photo is not None else ref
    others = tuple(p["box"] for j, p in enumerate(payload["photos"]) if j != ni)
    obj = object_mask(maps, core, GRAY_DPI, others)
    ox, oy = maps.origin
    boxes = [b for b in (ref, det, None if photo is None else photo["box"]) if b is not None]
    ys, xs = np.nonzero(obj)
    m = int(5 * GRAY_DPI / 25.4)
    x0 = int(max(min([b[0] for b in boxes] + ([xs.min() + ox] if len(xs) else [])) - m, 0))
    y0 = int(max(min([b[1] for b in boxes] + ([ys.min() + oy] if len(ys) else [])) - m, 0))
    x1 = int(min(max([b[2] for b in boxes] + ([xs.max() + ox] if len(xs) else [])) + m, data.gray.shape[1]))
    y1 = int(min(max([b[3] for b in boxes] + ([ys.max() + oy] if len(ys) else [])) + m, data.gray.shape[0]))
    view = cv2.cvtColor(data.gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    # Маска объекта (координаты окна) → координаты вырезки: общий участок окна и вырезки.
    sub = np.zeros(view.shape[:2], bool)
    gx0, gy0 = max(x0, ox), max(y0, oy)
    gx1, gy1 = min(x1, ox + obj.shape[1]), min(y1, oy + obj.shape[0])
    if gx1 > gx0 and gy1 > gy0:
        sub[gy0 - y0 : gy1 - y0, gx0 - x0 : gx1 - x0] = obj[gy0 - oy : gy1 - oy, gx0 - ox : gx1 - ox]
    out = sub.copy()
    if photo is not None:
        bx0, by0, bx1, by1 = (int(round(v)) for v in photo["box"])
        out[max(by0 - y0, 0) : max(by1 - y0, 0), max(bx0 - x0, 0) : max(bx1 - x0, 0)] = False
    layer = view.copy()
    layer[sub] = COLOUR_OBJ
    layer[out] = COLOUR_OBJ_OUT
    cv2.addWeighted(layer, 0.45, view, 0.55, 0, view)
    scale = CARD_WIDTH / view.shape[1]
    view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    draw_marks(view, np.array([x0, y0], float), scale, ref, det, photo, 2)
    return np.vstack([header_strip(spec.title, CARD_WIDTH), view])


def object_title(row: dict) -> list[str]:
    """Шапка карточки «объект за рамкой»."""
    return [f"{row['key']}  {row.get('kind', '')}  класс {row.get('new_class', '')} {row.get('new_quad_reason', '')}",
            f"объект за рамкой, мм² (в/н/л/п): эталон {row.get('ref_obj_out_mm2')} ({row.get('ref_obj_out_sides')}), "
            f"прежняя {row.get('det_obj_out_mm2')} ({row.get('det_obj_out_sides')}), "
            f"новая {row.get('new_obj_out_mm2')} ({row.get('new_obj_out_sides')})"]  # fmt: skip


# Лист «многочлен | сплайн» стороны: увеличение поперёк стороны (отклонение от хорды), px на мм вдоль и поперёк.
FIT_PLOT_ALONG = 6.0
FIT_PLOT_ACROSS = 200.0
COLOUR_POLY = (0, 165, 255)  # многочлен 2-й степени (было) — оранжевый
COLOUR_SPLINE = (200, 60, 200)  # сплайн (стало) — фиолетовый, как четырёхугольник
COLOUR_POINT_FIT = (140, 140, 0)  # точки края — бирюзовый


def side_fit_plot(along: np.ndarray, across: np.ndarray, dpi: float, title: str) -> np.ndarray:
    """График стороны: точки края и две подгонки (многочлен 2-й степени и сплайн с узлом на ``SIDE_KNOT_MM``) как
    отклонение от хорды точек; поперёк — увеличение :data:`FIT_PLOT_ACROSS` px на мм.

    Args:
        along, across: Точки края, px.
        dpi: Разрешение.
        title: Подпись.

    Returns:
        Картинка BGR.
    """
    from ocr_utils.page_layout.raster.photo_edges import (
        FIT_HUBER_MM,
        FIT_OUTLIER_MM,
        SideModel,
        mm,
        robust_poly,
        robust_spline,
    )

    k = 25.4 / dpi
    coef, inl = robust_poly(along, across, 2, mm(FIT_HUBER_MM, dpi), mm(FIT_OUTLIER_MM, dpi))
    poly = SideModel(np.asarray(coef))
    spline, inl2 = robust_spline(along, across, poly, inl, dpi, mm(FIT_HUBER_MM, dpi), mm(FIT_OUTLIER_MM, dpi))
    a = np.linspace(along.min(), along.max(), 400)
    chord = np.polyfit(along[inl], across[inl], 1)

    def dev(v, x):  # noqa: E306 — отклонение от хорды, мм
        return (v - np.polyval(chord, x)) * k

    W = int((along.max() - along.min()) * k * FIT_PLOT_ALONG) + 40
    H = 260
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.line(img, (0, H // 2), (W, H // 2), (200, 200, 200), 1)
    for mmv in (-0.5, 0.5):
        y = int(H // 2 - mmv * FIT_PLOT_ACROSS)
        cv2.line(img, (0, y), (W, y), (230, 230, 230), 1)

    def xy(x, d):  # noqa: E306
        return np.stack([20 + (x - along.min()) * k * FIT_PLOT_ALONG, H // 2 - d * FIT_PLOT_ACROSS], 1)

    for (px, py), ok in zip(xy(along, dev(across, along)), inl2):
        cv2.circle(img, (int(px), int(py)), 2, COLOUR_POINT_FIT if ok else (0, 0, 220), -1)
    for model, colour in ((poly, COLOUR_POLY), (spline, COLOUR_SPLINE)):
        pts = xy(a, dev(model(a), a))
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, colour, 2, cv2.LINE_AA)
    rp = np.percentile(np.abs(across[inl] - poly(along[inl])), 90) * k
    rs = np.percentile(np.abs(across[inl2] - spline(along[inl2])), 90) * k
    head = [title, f"длина {np.ptp(along) * k:.0f} мм; остаток p90: многочлен {rp:.3f} мм, сплайн {rs:.3f} мм; "
                   f"поперёк ×{FIT_PLOT_ACROSS / 1:.0f} px/мм, серые линии — ±0.5 мм"]  # fmt: skip
    return np.vstack([header_strip(head, W), img])


def fit_sheet(run_dir: Path, out: Path, top: int = 10) -> Path:
    """Лист «многочлен | сплайн» по самым длинным чётким сторонам с наибольшей разницей остатков.

    Args:
        run_dir: Папка прогона (``pages/*.json`` с точками сторон).
        out: Путь листа.
        top: Сколько сторон.

    Returns:
        Путь листа.
    """
    from ocr_utils.page_layout.overlay_frame import LegendEntry, framed

    found = []
    for fn in sorted((run_dir / "pages").rglob("*.json")):
        payload = json.loads(fn.read_text())
        for p in payload["photos"]:
            if p["corners"] is None:
                continue
            for sd in p["sides"]:
                a, c, ok = np.array(sd["along"]), np.array(sd["across"]), np.array(sd["inlier"], bool)
                if ok.sum() < 40 or np.ptp(a) * 25.4 / GRAY_DPI < 80:
                    continue
                r = c[ok] - np.polyval(np.polyfit(a[ok], c[ok], 2), a[ok])
                found.append((float(np.percentile(np.abs(r), 90)), payload["page"], sd["side"], a[ok], c[ok]))
    found.sort(key=lambda t: -t[0])
    plots = [side_fit_plot(a, c, GRAY_DPI, f"{page} {side}") for _, page, side, a, c in found[:top]]
    width = max(p.shape[1] for p in plots)
    plots = [cv2.copyMakeBorder(p, 0, 10, 0, width - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for p in plots]  # fmt: skip
    legend = [
        LegendEntry("многочлен 2-й степени (было)", COLOUR_POLY),
        LegendEntry("сплайн, узел на 25 мм (стало)", COLOUR_SPLINE),
        LegendEntry("точки края", COLOUR_POINT_FIT),
        LegendEntry("точки вне подгонки", (0, 0, 220)),
    ]
    picture = framed(np.vstack(plots), ["Подгонка длинных сторон фото: отклонение от хорды, поперёк увеличено"], legend)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), picture, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out
