"""Обработка одного кадра основы: страницы из кэша v2, совмещение страниц донора, срезанные строки, вклейка, листы «было | донор | стало»."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import framed
from ocr_utils.scan_cropping.v2.glue import split_pages
from research.cut_lines.compose import compose
from research.cut_lines.frames import read_bgr
from research.cut_lines.lines import find_cut_lines
from research.cut_lines.register import SIFT_POINTS, register_page

log = logging.getLogger(__name__)

# Донор уменьшается так, чтобы его высота была этой долей высоты (1) на масштабе совмещения (книга в кадре (2)
# крупнее: высота кадров 3584 против 4268 при той же высоте книги ~ доли кадра).
DONOR_HEIGHT_SHARE = 0.95
# Доля ширины кадра донора, до которой (от своего края) берутся его точки для страницы: 0.55 — половина и запас
# через сгиб (разворот донора не всегда по центру кадра).
DONOR_SIDE_SHARE = 0.55
# Ниже этой медианы корреляции страница считается несовмещённой (чужая пара, сбой) и не дорисовывается.
MIN_NCC = 0.35
# Качество JPEG тронутых кадров.
JPEG_QUALITY = 95
# Высота вырезки на листе «было | донор | стало», px.
SHEET_HEIGHT = 520


@dataclass(frozen=True)
class PageRow:
    """Строка сводки по странице.

    Поля:
        frame: Имя кадра (1).
        donor: Имя кадра (2) или пусто.
        page: Номер страницы в кадре (по порядку слева направо).
        status: ``patched`` / ``clean`` (срезанных строк нет) / ``no_match`` (не совместилась) / ``no_pair``.
        inliers: Инлайеры гомографии.
        ncc: Медиана корреляции после совмещения.
        lines: Число срезанных строк.
        sides: Края через «;» (по строкам).
        ink_gain: Множитель силы краски донора.
    """

    frame: str
    donor: str
    page: int
    status: str
    inliers: int = 0
    ncc: float = 0.0
    lines: int = 0
    sides: str = ""
    ink_gain: float = 0.0


def sheet(base: np.ndarray, donor_patch: np.ndarray, result: np.ndarray, box: tuple[int, int, int, int], title: str):
    """Лист «было | донор после совмещения и цвета | стало» по окну вклейки.

    Args:
        base: Кадр (1) до вклейки.
        donor_patch: Донор в окне (после совмещения и подгонки цвета).
        result: Кадр после вклейки.
        box: Окно ``(x0, y0, x1, y1)``.
        title: Шапка.

    Returns:
        Картинка BGR.
    """
    x0, y0, x1, y1 = box
    tiles = [base[y0:y1, x0:x1], donor_patch, result[y0:y1, x0:x1]]
    s = SHEET_HEIGHT / max(1, y1 - y0)
    tiles = [cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC) for t in tiles]
    gap = np.full((tiles[0].shape[0], 16, 3), 255, np.uint8)
    return framed(
        np.hstack([tiles[0], gap, tiles[1], gap, tiles[2]]),
        [title, "было | донор (2) после совмещения и цвета | стало"],
        [],
    )


def process_frame(base_path: Path, donor_path: Path | None, v2_work: Path, out_root: Path) -> list[PageRow]:
    """Дорисовка срезанных строк одного кадра основы; итог пишется в ``out_root/frames``.

    Args:
        base_path: Кадр (1).
        donor_path: Кадр (2) с тем же разворотом или None.
        v2_work: Кэш scan_cropping v2 по (1).
        out_root: Корень выхода.

    Returns:
        Строки сводки по страницам.
    """
    out_path = out_root / "frames" / base_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = v2_work / base_path.stem
    if donor_path is None or not (work / "paper.npz").exists():
        shutil.copy2(base_path, out_path)
        return [PageRow(base_path.name, "", 0, "no_pair")]
    base = read_bgr(base_path)
    donor = read_bgr(donor_path)
    paper1 = np.load(work / "paper.npz")["paper"]
    book = np.load(work / "book.npz")["book"]
    inst = np.load(work / "sam3.npz")["inst"]
    h, w = paper1.shape
    seg = w / base.shape[1]
    small = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    donor_scale = h / donor.shape[0] * DONOR_HEIGHT_SHARE
    donor_small = cv2.resize(donor, None, fx=donor_scale, fy=donor_scale, interpolation=cv2.INTER_AREA)
    gray_b = cv2.cvtColor(donor_small, cv2.COLOR_BGR2GRAY)
    kp_b, desc_b = cv2.SIFT_create(nfeatures=SIFT_POINTS).detectAndCompute(gray_b, None)
    sift_b = (np.float32([k.pt for k in kp_b]), desc_b)
    # Страницы — бумага v2, разданная экземплярам SAM 3 (слева направо).
    pages = [p for p in split_pages(paper1, book, None, inst) if p.sum() > 0.05 * paper1.sum()]
    pages.sort(key=lambda p: np.nonzero(p.any(axis=0))[0].mean())
    result = base.copy()
    rows: list[PageRow] = []
    reference = float(np.median(gray_a[paper1]))
    for k, page in enumerate(pages, start=1):
        others = [p for j, p in enumerate(pages, start=1) if j != k]
        other = np.logical_or.reduce(others) if others else None
        flow_zone = cv2.dilate(page.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
        # Точки донора — только со своей половины его разворота (с запасом через сгиб): одинаковые слова соседней
        # страницы иначе забирают совпадения, и гомография страницы не находится (IMG_0572: 73 → 280 инлайеров).
        on_left = k == 1 and len(pages) > 1
        cut_x = gray_b.shape[1] * (DONOR_SIDE_SHARE if on_left else 1 - DONOR_SIDE_SHARE)
        side = (sift_b[0][:, 0] < cut_x) if on_left else (sift_b[0][:, 0] > cut_x)
        if len(pages) == 1:
            side[:] = True
        pm = register_page(gray_a, gray_b, flow_zone, page, (sift_b[0][side], sift_b[1][side]))
        if pm is None or pm.ncc < MIN_NCC:
            rows.append(PageRow(base_path.name, donor_path.name, k, "no_match", pm.inliers if pm else 0,
                                pm.ncc if pm else 0.0))  # fmt: skip
            continue
        warped = cv2.remap(donor_small, pm.map_x, pm.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        found = find_cut_lines(warped, paper1, page, other, reference, gray_a)
        if not found.lines:
            rows.append(PageRow(base_path.name, donor_path.name, k, "clean", pm.inliers, pm.ncc))
            continue
        before = result
        # Без краски донора: продление листа вне бумаги донора и вне срезанных строк (только тон бумаги).
        clean = found.extension & (~found.paper2 | ~found.text)
        comp = compose(result, donor, pm.map_x, pm.map_y, found.alpha, paper1, found.cut, clean, seg, donor_scale)
        if comp is None:
            rows.append(PageRow(base_path.name, donor_path.name, k, "clean", pm.inliers, pm.ncc))
            continue
        result = comp.image
        sides = ";".join(line.side.value for line in found.lines)
        title = (
            f"РИСК 1991/05 {base_path.name} стр. {k} ← {donor_path.name}: срезанных строк {len(found.lines)} ({sides})"
        )
        pic = sheet(before, comp.donor, result, comp.box, title)
        (out_root / "sheets").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "sheets" / f"{base_path.stem}_p{k}.jpg"), pic, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(PageRow(base_path.name, donor_path.name, k, "patched", pm.inliers, pm.ncc, len(found.lines), sides,
                            comp.ink_gain))  # fmt: skip
    if any(r.status == "patched" for r in rows):
        ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        buf.tofile(str(out_path))
    else:
        shutil.copy2(base_path, out_path)
    return rows
