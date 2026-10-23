"""Лента у корешка: кадр кэша v2 и выпрямленная вдоль предварительного корешка полоса ±``BAND_MM`` — яркость бумаги без букв, оттенок, экземпляры SAM 3, печать."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.gutter_strip import BAND_MM, PAPER_WINDOW_MM, remap_band, rows_of
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.spread_lines import Line, SpineMethod, mm_px
from research.spread_lines.frames import FrameLines, load_frame


@dataclass(frozen=True)
class StripFrame:
    """Кадр кэша v2 для стенда: прямые разворота и маски, нужные способам обнаружения.

    Поля:
        frame: Кадр с прямыми по способам разделителя (``research.spread_lines.frames``).
        inst: Экземпляры-страницы SAM 3 (метки, 0 — не страница), масштаб сегментации.
        ink: Крупная печать кадра (``book.npz`` → ``ink``) или пустая маска.
        paint: Зона закраса пальцев на масштабе сегментации.
        full_shape: ``(H, W)`` исходного кадра (по маске пальцев полного размера) — пересчёт блоков surya.
        polys: Блоки surya на исходном кадре (``layout.json``), px кадра.
        core: Ядро пальца на масштабе сегментации.
    """

    frame: FrameLines
    inst: np.ndarray
    ink: np.ndarray
    paint: np.ndarray
    full_shape: tuple[int, int]
    polys: list
    core: np.ndarray

    @property
    def key(self) -> str:
        """Ключ кадра в кэше."""
        return self.frame.key

    @property
    def spine0(self) -> Line:
        """Предварительный корешок — разделитель ``auto`` v2."""
        return self.frame.lines[SpineMethod.AUTO].spine


def load_strip_frame(cache: Path, key: str) -> StripFrame | None:
    """Кадр кэша v2 для стенда; None — страниц с трапецией не две.

    Args:
        cache: Корень кэша прогона v2.
        key: Ключ кадра.

    Returns:
        Кадр или None.
    """
    from ocr_utils.scan_cropping.v2.pipeline import paint_on_seg, read_layout

    frame = load_frame(cache, key)
    if frame is None:
        return None
    work = cache / key
    inst = np.load(work / "sam3.npz")["inst"]
    with np.load(work / ("book_veto.npz" if (work / "book_veto.npz").exists() else "book.npz")) as data:
        ink = data["ink"] if "ink" in data.files else np.zeros(inst.shape, bool)
    fingers = np.load(work / "fingers.npz")
    paint = paint_on_seg(fingers, inst.shape)
    polys = read_layout(work / "layout.json") or []
    core = cv2.resize(fingers["core"].astype(np.uint8), inst.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    return StripFrame(frame, inst, ink, paint, tuple(fingers["paint"].shape), polys, core)


@dataclass(frozen=True)
class Band:
    """Лента ±``BAND_MM`` у корешка, выпрямленная по прямой ``spine``: столбец ``u`` — сдвиг поперёк от корешка.

    Поля:
        spine: Прямая, вдоль которой выпрямлена лента.
        ys: Строки кадра (сегментация) по строкам ленты.
        us: Сдвиги столбцов от корешка, px сегментации (минус — влево).
        bgr: Лента кадра без пальцев.
        light: Яркость L бумаги без букв (максимум по вертикали в ``PAPER_WINDOW_MM``), float.
        tone: Оттенок b* (жёлтый — больше), медиана по вертикали в том же окне, float.
        inst: Метки экземпляров SAM 3.
        ink: Крупная печать.
        book: Грубая маска книги.
        hands: Зона закраса пальцев.
    """

    spine: Line
    ys: np.ndarray
    us: np.ndarray
    bgr: np.ndarray
    light: np.ndarray
    tone: np.ndarray
    inst: np.ndarray
    ink: np.ndarray
    book: np.ndarray
    hands: np.ndarray

    def x_of(self, u: np.ndarray | float, rows: np.ndarray | None = None) -> np.ndarray:
        """Абсциссы кадра для сдвигов ``u`` в строках ленты ``rows`` (все строки — None)."""
        ys = self.ys if rows is None else self.ys[rows]
        return self.spine.x_at(ys) + np.asarray(u, np.float64)

    def mm(self, px: np.ndarray | float) -> np.ndarray | float:
        """px сегментации → мм бумаги."""
        return np.asarray(px, np.float64) / SEG_PX_PER_MM


def build_band(sf: StripFrame, spine: Line | None = None, band_mm: float = BAND_MM) -> Band:
    """Лента у корешка кадра: выпрямление по ``spine`` (по умолчанию — предварительный корешок) и карты бумаги.

    Args:
        sf: Кадр стенда.
        spine: Прямая выпрямления или None (``auto`` v2).
        band_mm: Полуширина ленты, мм.

    Returns:
        Лента.
    """
    spine = sf.spine0 if spine is None else spine
    img = sf.frame.img
    ys = rows_of(spine, img.shape[:2])
    half = int(round(mm_px(band_mm)))
    us = np.arange(-half, half + 1, dtype=np.float64)
    bgr = remap_band(img, spine, ys, us, nearest=False)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    win = max(3, int(round(mm_px(PAPER_WINDOW_MM))) | 1)
    # Бумага без букв: буквы темнее бумаги, максимум по вертикали их снимает; вертикальные кромки полосы остаются.
    light = cv2.dilate(lab[..., 0], np.ones((win, 1), np.uint8)).astype(np.float32)
    tone = cv2.medianBlur(np.ascontiguousarray(lab[..., 2]), 5).astype(np.float32)
    tone = cv2.blur(tone, (1, win))
    return Band(
        spine,
        ys,
        us,
        bgr,
        light,
        tone,
        remap_band(sf.inst, spine, ys, us, nearest=True),
        remap_band(sf.ink, spine, ys, us, nearest=True),
        remap_band(sf.frame.book, spine, ys, us, nearest=True),
        remap_band(sf.paint, spine, ys, us, nearest=True),
    )
