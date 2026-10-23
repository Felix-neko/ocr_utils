"""Режим extend одного кадра основы: страницы из кэша v2, продление нижней кромки (пальцы — по LaMa-кадру с возвратом ядра), допечатка срезанных строк по набору, сверка текста с донором, листы «было | стало»."""

from __future__ import annotations

import csv
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.page_layout.overlay_frame import framed
from ocr_utils.scan_cropping.v2.glue import split_pages
from research.cut_lines import synth as S
from research.cut_lines.extend import (
    BottomEdge,
    extend_bottom,
    full_page_mask,
    page_bottom_edge,
    shift_down,
    stack_columns,
)
from research.cut_lines.fragments import (
    bottom_columns,
    complete_page,
    harvest_rows,
    ink_above_edge,
    load_library,
    row_geometry,
    x_height_of,
)
from research.cut_lines.frames import read_bgr
from research.cut_lines.process import JPEG_QUALITY

log = logging.getLogger(__name__)

# Ширина листа «было | стало», px; высота вырезки над старой кромкой, px.
SHEET_WIDTH = 2400
SHEET_ABOVE = 420
# Зона пальца возвращается поверх продлённого LaMa-кадра: дилатация и перо, px; окно сглаживания низа маски под
# пальцем, столбцов.
FINGER_EDGE_WIN = 61
# Зона корешка: от конца кромки левой страницы до начала правой с запасом, px.
SPINE_PAD = 80
# Нижняя полоса кадра для замера цвета фона, строк.
PAD_COLOR_ROWS = 400
FINGER_DILATE = 9
FINGER_FEATHER = 3
# Ожидаемая высота буквы в кадрах основы (9–10 pt при ~16 px/мм), px.
X_HEIGHT_HINT = 26.0
# Нижняя полоса страницы для сверки текста tesseract (обе подшивки), в долях высоты страницы.
OCR_BAND = 0.12


@dataclass(frozen=True)
class PageVerdict:
    """Строка набора: статус страницы и тексты колонок."""

    status: str
    texts: list[str]


@dataclass(frozen=True)
class ExtendRow:
    """Строка сводки по странице или колонке.

    Поля:
        frame: Кадр (1).
        page: ``L`` / ``R``.
        status: ``extended`` / ``synth`` / ``check`` / ``no_axis`` / ``no_text`` / ``no_edge`` / ``no_cache``.
        text: Заданный текст колонки.
        reread: Перечитанный tesseract текст набранной строки.
        match: Сходство (0..100).
        missing: Буквы без образца.
        text_base_ocr: Нижние строки страницы в (1) по tesseract.
        text_donor_ocr: Нижние строки той же страницы в (2).
    """

    frame: str
    page: str
    status: str
    text: str = ""
    reread: str = ""
    match: float = 0.0
    missing: str = ""
    text_base_ocr: str = ""
    text_donor_ocr: str = ""


@dataclass
class FrameResult:
    """Итог кадра: строки сводки и путь результата."""

    rows: list[ExtendRow] = field(default_factory=list)
    out_path: Path | None = None


def read_set(path: Path) -> dict[tuple[str, str], PageVerdict]:
    """Набор вердиктов: ``frame,page,status,texts`` (тексты колонок через ``|``).

    Args:
        path: CSV.

    Returns:
        Словарь «(кадр, страница) → вердикт».
    """
    out = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            texts = r["texts"].split("|") if r["texts"] else []
            out[(r["frame"], r["page"])] = PageVerdict(r["status"], texts)
    return out


def load_pages(work: Path) -> list[np.ndarray]:
    """Маски страниц кадра из кэша scan_cropping v2 (½ кадра), слева направо.

    Args:
        work: Папка кэша кадра.

    Returns:
        Маски (bool).
    """
    paper = np.load(work / "paper.npz")["paper"]
    book = np.load(work / "book.npz")["book"]
    inst = np.load(work / "sam3.npz")["inst"]
    pages = [p for p in split_pages(paper, book, None, inst) if p.sum() > 0.05 * paper.sum()]
    pages.sort(key=lambda p: np.nonzero(p.any(axis=0))[0].mean())
    return pages


def bottom_text(gray: np.ndarray, page: np.ndarray, edge: BottomEdge | None) -> str:
    """Нижние строки страницы по tesseract (psm 6) для сверки: полоса :data:`OCR_BAND` высоты страницы над
    кромкой, последние две строки.

    Args:
        gray: Серое кадра.
        page: Маска страницы (полный кадр, bool).
        edge: Кромка или None (тогда низ маски).

    Returns:
        Текст последних двух строк.
    """
    ys, xs = np.nonzero(page)
    if len(ys) < 100:
        return ""
    y1 = int(np.median(edge.y)) if edge is not None else int(ys.max())
    y0 = int(max(0, y1 - OCR_BAND * (ys.max() - ys.min())))
    band = gray[y0:y1, int(xs.min()) : int(xs.max()) + 1]
    words = S.tesseract_words(band, psm=6)
    lines: dict[int, list[str]] = {}
    for text, left, top, wid, hgt, conf in words:
        lines.setdefault(int(round(top / max(hgt, 1))), []).append((left, text))
    rows = [" ".join(t for _, t in sorted(v)) for _, v in sorted(lines.items())]
    return " / ".join(rows[-2:])


def donor_half(donor: np.ndarray, side: str) -> np.ndarray:
    """Половина разворота донора по стороне (``L`` / ``R``), серое."""
    gray = cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]
    return gray[:, : int(0.55 * w)] if side == "L" else gray[:, int(0.45 * w) :]


def donor_bottom_text(donor: np.ndarray, side: str) -> str:
    """Нижние строки страницы донора по tesseract: нижняя полоса половины разворота, последние две строки.

    Args:
        donor: Кадр (2) BGR.
        side: Сторона страницы.

    Returns:
        Текст последних двух строк.
    """
    half = donor_half(donor, side)
    h = half.shape[0]
    # Бумага донора: светлые строки; низ — последняя строка, где бумага занимает заметную долю.
    light = (half > 0.6 * np.percentile(half, 95)).mean(axis=1)
    rows = np.nonzero(light > 0.3)[0]
    if len(rows) == 0:
        return ""
    y1 = int(rows.max())
    band = half[int(max(0, y1 - OCR_BAND * (rows.max() - rows.min()))) : y1]
    words = S.tesseract_words(band, psm=6)
    lines: dict[int, list[str]] = {}
    for text, left, top, wid, hgt, conf in words:
        lines.setdefault(int(round(top / max(hgt, 1))), []).append((left, text))
    out = [" ".join(t for _, t in sorted(v)) for _, v in sorted(lines.items())]
    return " / ".join(out[-2:])


def sheet(before: np.ndarray, after: np.ndarray, edges: list[BottomEdge], title: str) -> np.ndarray:
    """Лист «было | стало»: нижние полосы кадра над старой кромкой и под ней.

    Args:
        before: Кадр до (той же высоты, что ``after``).
        after: Кадр после.
        edges: Кромки страниц.
        title: Шапка.

    Returns:
        BGR.
    """
    x0 = min(e.x0 for e in edges)
    x1 = max(e.x1 for e in edges)
    y0 = int(max(0, min(float(e.y.min()) for e in edges) - SHEET_ABOVE))
    y1 = after.shape[0]
    pair = np.vstack([before[y0:y1, x0:x1], np.full((12, x1 - x0, 3), 255, np.uint8), after[y0:y1, x0:x1]])
    s = min(1.0, SHEET_WIDTH / pair.shape[1])
    pair = cv2.resize(pair, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return framed(pair, [title, "сверху — было, снизу — стало"], [])


def process_frame_extend(
    base_path: Path,
    donor_path: Path | None,
    v2_work: Path,
    out_root: Path,
    verdicts: dict[tuple[str, str], PageVerdict],
    extend_px: int,
    library_path: Path | None = None,
) -> FrameResult:
    """Продление кромки и допечатка строк одного кадра; итог — ``out_root/frames``, лист — ``out_root/sheets``.

    Args:
        base_path: Кадр (1).
        donor_path: Кадр (2) с тем же разворотом (только для сверки текста) или None.
        v2_work: Кэш scan_cropping v2 по (1).
        out_root: Корень выхода.
        verdicts: Набор вердиктов по страницам.
        extend_px: На сколько px продлить кромку.
        library_path: Общая библиотека букв подшивки (npz от команды ``harvest``) или None.

    Returns:
        Итог кадра.
    """
    out_path = out_root / "frames" / base_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = v2_work / base_path.stem
    result = FrameResult(out_path=out_path)
    if not (work / "paper.npz").exists():
        shutil.copy2(base_path, out_path)
        result.rows.append(ExtendRow(base_path.name, "", "no_cache"))
        return result
    original = read_bgr(base_path)
    # Возвращается вся закрашенная зона пальца (``paint``, не ``core``): LaMa в ней дорисовывает похожие строки,
    # и они не должны проступать рядом с настоящими.
    finger_masks = np.load(work / "fingers.npz") if (work / "fingers.npz").exists() else None
    fingers = finger_masks["paint"] if finger_masks is not None else None
    has_fingers = fingers is not None and fingers.any() and (work / "clean.png").exists()
    # Источник пикселей кромки: LaMa-кадр без пальцев (зона пальца вернётся поверх), иначе исходник.
    source = cv2.imread(str(work / "clean.png")) if has_fingers else original
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    pages = load_pages(work)
    book_data = np.load(work / "book.npz")
    book_half = book_data["book"]
    bg_half = book_data["bg"] if "bg" in book_data else None
    sides = ["L", "R"] if len(pages) == 2 else [f"P{k}" for k in range(1, len(pages) + 1)]
    edges: list[BottomEdge | None] = []
    masks = []
    book_center = float(np.nonzero(book_half.any(axis=0))[0].mean()) if book_half.any() else book_half.shape[1] / 2
    frame_cols = np.zeros(gray.shape[1], bool)
    for side, page in zip(sides, pages):
        edge = page_bottom_edge(page, gray, bg_half)
        edges.append(edge)
        if edge is None:
            result.rows.append(ExtendRow(base_path.name, side, "no_edge"))
            continue
        masks.append(full_page_mask(page, edge, gray.shape))
        frame_cols[edge.x0 : edge.x1] |= edge.frame
        # Торец стопки за наружным краем страницы продлевается вместе с ней.
        outer = -1 if float(np.nonzero(page.any(axis=0))[0].mean()) < book_center else 1
        stack = stack_columns(book_half, gray, edge, outer)
        if stack.any():
            log.info("%s %s: торец стопки %d столбцов", base_path.name, side, int(stack.any(axis=0).sum()))
            masks.append(stack)
    if not masks:
        shutil.copy2(base_path, out_path)
        return result
    union = np.logical_or.reduce(masks)
    exclude = None
    if has_fingers:
        # Ядро пальца на странице (над кромкой) — не трогать при продлении.
        core = finger_masks["core"] & union
        exclude = cv2.dilate(core.astype(np.uint8), np.ones((FINGER_DILATE, FINGER_DILATE), np.uint8)) > 0
    good_edges = [e for e in edges if e is not None]
    spine = None
    if len(good_edges) == 2:
        spine = (
            min(good_edges[0].x1, good_edges[1].x1) - SPINE_PAD,
            max(good_edges[0].x0, good_edges[1].x0) + SPINE_PAD,
        )
    # Цвет фона для дорощенных строк вне окна продления — медиана пикселей фона v2 в нижней полосе кадра.
    pad_color = None
    if bg_half is not None and bg_half.any():
        bg_full = (
            cv2.resize(bg_half.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        )
        band = bg_full[-PAD_COLOR_ROWS:]
        if band.sum() > 1000:
            pad_color = np.median(source[-PAD_COLOR_ROWS:][band], axis=0).astype(np.uint8)
    ext = extend_bottom(source, union, extend_px, exclude, frame_cols=frame_cols, spine=spine, pad_color=pad_color)
    frame = ext.image
    before = cv2.copyMakeBorder(original, 0, ext.pad, 0, 0, cv2.BORDER_REPLICATE)
    if has_fingers:
        # Зона пальца возвращается с исходника: часть на бумаге (над кромкой) — на место, часть под кромкой — со
        # сдвигом на extend_px вместе с кромкой; новая бумага не перекрывается.
        paint = cv2.copyMakeBorder(fingers.astype(np.uint8), 0, ext.pad, 0, 0, cv2.BORDER_CONSTANT, value=0) > 0
        union_pad = np.vstack([union, np.zeros((ext.pad, union.shape[1]), bool)]) if ext.pad else union
        # «Под кромкой» — ниже низа маски страницы в столбце; где лист уходит за рамку кадра, кромки нет и низ
        # зоны пальца не сдвигается.
        hh = union_pad.shape[0]
        has = union_pad.any(axis=0)
        ub = np.where(has, hh - 1 - np.argmax(union_pad[::-1], axis=0), -1)
        # Низ маски страницы под пальцем обрывается выше пальца (маска кончается, где палец её закрыл): провал
        # низа уже FINGER_EDGE_WIN столбцов — дырка маски, а не кромка; берётся максимум по соседним столбцам.
        ub = np.where(has, ndimage.maximum_filter1d(ub, FINGER_EDGE_WIN, mode="nearest"), -1)
        rows = np.arange(hh)[:, None]
        under = (rows > ub[None, :] + 2) & has[None, :] & ~frame_cols[None, :]
        above = paint & ~under
        below = paint & under
        kernel = np.ones((FINGER_DILATE, FINGER_DILATE), np.uint8)
        above_d = (cv2.dilate(above.astype(np.uint8), kernel) > 0) & ~ext.paper
        below_d = shift_down(cv2.dilate(below.astype(np.uint8), kernel) > 0, extend_px) & ~ext.paper & ~union_pad
        alpha = cv2.GaussianBlur(above_d.astype(np.float32), (0, 0), FINGER_FEATHER)[..., None]
        frame = np.clip(frame * (1 - alpha) + before * alpha, 0, 255).astype(np.uint8)
        shifted = np.zeros_like(before)
        shifted[extend_px:] = before[: before.shape[0] - extend_px]
        alpha = cv2.GaussianBlur(below_d.astype(np.float32), (0, 0), FINGER_FEATHER)[..., None]
        frame = np.clip(frame * (1 - alpha) + shifted * alpha, 0, 255).astype(np.uint8)
    donor = read_bgr(donor_path) if donor_path is not None else None
    library = S.Library(fallback=load_library(library_path) if library_path is not None else None)
    for side, page, edge in zip(sides, pages, edges):
        if edge is None:
            continue
        verdict = verdicts.get((base_path.stem, side))
        full = full_page_mask(page, edge, gray.shape)
        full = np.vstack([full, np.zeros((ext.pad, full.shape[1]), bool)]) if ext.pad else full
        base_ocr = bottom_text(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), full, edge)
        donor_ocr = donor_bottom_text(donor, side) if donor is not None else ""
        if verdict is None or verdict.status != "cut" or not any(t.strip() for t in verdict.texts):
            result.rows.append(ExtendRow(base_path.name, side, "extended", text_base_ocr=base_ocr,
                                         text_donor_ocr=donor_ocr))  # fmt: skip
            continue
        done = complete_page(frame, full, edge, verdict.texts, X_HEIGHT_HINT, library)
        frame = done.image
        for rep in done.reports:
            result.rows.append(ExtendRow(base_path.name, side, rep.status, rep.text, rep.reread, rep.match,
                                         rep.missing, base_ocr, donor_ocr))  # fmt: skip
        if not done.reports:
            result.rows.append(ExtendRow(base_path.name, side, "no_columns", "|".join(verdict.texts),
                                         text_base_ocr=base_ocr, text_donor_ocr=donor_ocr))  # fmt: skip
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    buf.tofile(str(out_path))
    (out_root / "sheets").mkdir(parents=True, exist_ok=True)
    texts = "; ".join(r.text for r in result.rows if r.status in ("synth", "check"))
    pic = sheet(before, frame, [e for e in edges if e is not None], f"РИСК 1991/05 {base_path.name}: {texts}")
    cv2.imwrite(str(out_root / "sheets" / f"{base_path.stem}.jpg"), pic, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return result


def harvest_frame(base_path: Path, v2_work: Path) -> dict[str, list[S.Glyph]]:
    """Буквы кадра для общей библиотеки: по каждой странице — колонки нижнего ряда, строки колонки над кромкой
    (как при допечатке), буквы по рамкам tesseract.

    Args:
        base_path: Кадр (1).
        v2_work: Кэш scan_cropping v2.

    Returns:
        Образцы по буквам.
    """
    work = v2_work / base_path.stem
    if not (work / "paper.npz").exists():
        return {}
    source = cv2.imread(str(work / "clean.png")) if (work / "clean.png").exists() else read_bgr(base_path)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    library = S.Library()
    for page in load_pages(work):
        edge = page_bottom_edge(page, gray)
        if edge is None:
            continue
        full = full_page_mask(page, edge, gray.shape)
        ink = ink_above_edge(gray, full, edge)
        x_height = x_height_of(ink, X_HEIGHT_HINT)
        for column in bottom_columns(ink, edge, x_height):
            geo = row_geometry(ink, gray, column, edge, x_height)
            if geo is not None:
                harvest_rows(source, geo.axes + [geo.cut_axis], library, geo.x_height)
    return library.samples
