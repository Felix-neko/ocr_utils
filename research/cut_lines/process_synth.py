"""Синтез срезанных нижних строк одного кадра основы: совмещение страниц с донором, нижние строки, набор из букв в два прохода, листы «было | стало»."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import framed
from ocr_utils.scan_cropping.v2.glue import split_pages
from research.cut_lines import synth as S
from research.cut_lines.frames import read_bgr
from research.cut_lines.lines import Side, find_cut_lines, glyph_height, ink_lines
from research.cut_lines.page_synth import synth_page_bottom
from research.cut_lines.process import DONOR_HEIGHT_SHARE, DONOR_SIDE_SHARE, JPEG_QUALITY, MIN_NCC
from research.cut_lines.register import SIFT_POINTS, register_page

log = logging.getLogger(__name__)

# Высота листа отсмотра «было | стало», px.
SHEET_WIDTH = 2400


@dataclass(frozen=True)
class LineRow:
    """Строка сводки по колонке.

    Поля:
        frame: Кадр (1).
        donor: Кадр (2).
        page: Страница в кадре.
        status: Итог колонки (``synth`` / ``skip_*`` / ``visible`` / ``no_axis`` / ``no_match`` / ``no_pair``).
        text: Текст строки.
        draft: Черновик tesseract по донору.
        match: Сходство (текст с черновиком или перечитанного набора с текстом).
        missing: Буквы без образца.
    """

    frame: str
    donor: str
    page: int
    status: str
    text: str = ""
    draft: str = ""
    match: float = 0.0
    missing: str = ""


@dataclass
class PagePrep:
    """Подготовка страницы: совмещение и куски срезанного нижнего ряда.

    Поля:
        page: Маска страницы (масштаб совмещения).
        pm: ``PageMap``.
        pieces: Куски нижнего ряда.
        x_height: Высота буквы (масштаб совмещения).
    """

    page: np.ndarray
    pm: object
    pieces: list
    x_height: float


def prepare_pages(base: np.ndarray, donor: np.ndarray, work: Path):
    """Совмещение страниц кадра с донором и куски срезанного нижнего ряда по страницам.

    Args:
        base: Кадр (1).
        donor: Кадр (2).
        work: Кэш scan_cropping v2 кадра.

    Returns:
        ``(подготовки страниц или None по каждой, масштаб совмещения, масштаб донора)``.
    """
    paper1 = np.load(work / "paper.npz")["paper"]
    book = np.load(work / "book.npz")["book"]
    inst = np.load(work / "sam3.npz")["inst"]
    h, w = paper1.shape
    seg = w / base.shape[1]
    gray_a = cv2.cvtColor(cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    donor_scale = h / donor.shape[0] * DONOR_HEIGHT_SHARE
    donor_small = cv2.resize(donor, None, fx=donor_scale, fy=donor_scale, interpolation=cv2.INTER_AREA)
    gray_b = cv2.cvtColor(donor_small, cv2.COLOR_BGR2GRAY)
    kp, desc = cv2.SIFT_create(nfeatures=SIFT_POINTS).detectAndCompute(gray_b, None)
    pts = np.float32([k.pt for k in kp])
    pages = [p for p in split_pages(paper1, book, None, inst) if p.sum() > 0.05 * paper1.sum()]
    pages.sort(key=lambda p: np.nonzero(p.any(axis=0))[0].mean())
    reference = float(np.median(gray_a[paper1]))
    preps: list[PagePrep | None] = []
    for k, page in enumerate(pages, start=1):
        others = [p for j, p in enumerate(pages, start=1) if j != k]
        other = np.logical_or.reduce(others) if others else None
        # Точки донора — со своей половины его разворота (одинаковые слова соседней страницы сбивают совпадения).
        if len(pages) == 1:
            side = np.ones(len(pts), bool)
        elif k == 1:
            side = pts[:, 0] < gray_b.shape[1] * DONOR_SIDE_SHARE
        else:
            side = pts[:, 0] > gray_b.shape[1] * (1 - DONOR_SIDE_SHARE)
        zone = cv2.dilate(page.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
        pm = register_page(gray_a, gray_b, zone, page, (pts[side], desc[side]))
        if pm is None or pm.ncc < MIN_NCC:
            preps.append(None)
            continue
        warped = cv2.remap(donor_small, pm.map_x, pm.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        found = find_cut_lines(warped, paper1, page, other, reference, gray_a, frozenset({Side.BOTTOM}))
        ink, _ = ink_lines(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
        preps.append(PagePrep(page, pm, found.pieces, glyph_height(ink, page & paper1)))
    return preps, seg, donor_scale


def synth_pass(base, donor, preps, seg, donor_scale, cache_dir, shared):
    """Один проход синтеза по страницам кадра.

    Args:
        base: Кадр (1).
        donor: Кадр (2).
        preps: Подготовки страниц.
        seg: Масштаб совмещения.
        donor_scale: Масштаб донора.
        cache_dir: Кэш DeepSeek.
        shared: Общая библиотека букв кадра.

    Returns:
        ``(кадр, отчёты по страницам [(номер, отчёты колонок, окна)])``.
    """
    result = base
    reports = []
    for k, prep in enumerate(preps, start=1):
        if prep is None or not prep.pieces:
            reports.append((k, None, []))
            continue
        r = synth_page_bottom(result, donor, prep.pm, prep.pieces, prep.page, seg, donor_scale, prep.x_height,
                              cache_dir, shared)  # fmt: skip
        result = r.image
        reports.append((k, r.columns, r.boxes))
    return result, reports


def sheet(before: np.ndarray, after: np.ndarray, boxes, title: str) -> np.ndarray:
    """Лист «было | стало» по окнам набора.

    Args:
        before: Кадр до.
        after: Кадр после.
        boxes: Окна ``(x0, y0, x1, y1)``.
        title: Шапка.

    Returns:
        BGR.
    """
    x0, x1 = min(b[0] for b in boxes), max(b[2] for b in boxes)
    y0, y1 = max(0, min(b[1] for b in boxes) - 150), max(b[3] for b in boxes)
    pair = np.vstack([before[y0:y1, x0:x1], np.full((12, x1 - x0, 3), 255, np.uint8), after[y0:y1, x0:x1]])
    s = min(1.0, SHEET_WIDTH / pair.shape[1])
    pair = cv2.resize(pair, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return framed(pair, [title, "сверху — было, снизу — стало"], [])


def process_frame_synth(base_path: Path, donor_path: Path | None, v2_work: Path, out_root: Path) -> list[LineRow]:
    """Синтез срезанных нижних строк кадра основы; итог — в ``out_root/frames``, листы — в ``out_root/sheets``.

    Первый проход набирает строки и собирает общую библиотеку букв кадра; если каким-то колонкам не хватило букв,
    второй проход перенабирает кадр с исходника уже по полной библиотеке (ответы DeepSeek — из кэша).

    Args:
        base_path: Кадр (1).
        donor_path: Кадр (2) с тем же разворотом или None.
        v2_work: Кэш scan_cropping v2 по (1).
        out_root: Корень выхода.

    Returns:
        Строки сводки по колонкам.
    """
    out_path = out_root / "frames" / base_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = v2_work / base_path.stem
    if donor_path is None or not (work / "paper.npz").exists():
        shutil.copy2(base_path, out_path)
        return [LineRow(base_path.name, "", 0, "no_pair")]
    base = read_bgr(base_path)
    donor = read_bgr(donor_path)
    preps, seg, donor_scale = prepare_pages(base, donor, work)
    cache_dir = out_root / "llm"
    shared = S.Library()
    result, reports = synth_pass(base, donor, preps, seg, donor_scale, cache_dir, shared)
    if any(c.missing for _, cols, _ in reports if cols for c in cols):
        result, reports = synth_pass(base, donor, preps, seg, donor_scale, cache_dir, shared)
    rows: list[LineRow] = []
    boxes_all = []
    for k, cols, boxes in reports:
        if cols is None:
            rows.append(LineRow(base_path.name, donor_path.name, k, "no_match" if preps[k - 1] is None else "clean"))
            continue
        for c in cols:
            rows.append(LineRow(base_path.name, donor_path.name, k, c.status, c.text, c.draft, c.match, c.missing))
        boxes_all += boxes
    if boxes_all:
        ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        buf.tofile(str(out_path))
        texts = "; ".join(r.text for r in rows if r.status == "synth")
        (out_root / "sheets").mkdir(parents=True, exist_ok=True)
        pic = sheet(base, result, boxes_all, f"РИСК 1991/05 {base_path.name} ← {donor_path.name}: {texts}")
        cv2.imwrite(str(out_root / "sheets" / f"{base_path.stem}.jpg"), pic, [cv2.IMWRITE_JPEG_QUALITY, 88])
    else:
        shutil.copy2(base_path, out_path)
    return rows
