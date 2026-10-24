"""Признаки «в кадре одна страница» по кэшу прогона ``scan_cropping.v2``: число страниц и экземпляров SAM 3, ширины трапеций в мм, печать на линии разреза, пропорции бумаги и книги."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.keystone import vertical_sides
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM

# Полуширина окна у линии разреза, в котором ищется провал печати, мм (34 px сегментации при 6.9 px/мм): шире щели
# корешка с тенью (до ~12 мм у ПХ), уже поля страницы.
SPINE_WINDOW_MM = 5.0
# Ширина миниатюры кадра для листов отсмотра, px.
THUMB_WIDTH = 360
# Цвета трапеций страниц на миниатюре (BGR): левая — синяя, правая — красная, прочие — зелёная.
QUAD_COLORS = ((255, 120, 0), (0, 0, 255), (0, 200, 0))


class SplitSource(StrEnum):
    """Откуда у кадра две страницы: из экземпляров SAM 3, из провала печати (запасной ход ``glue.split_pages``) или страница одна."""

    INSTANCES = "instances"  # SAM 3 дал два и больше экземпляров-страниц
    PRINT = "print"  # экземпляр один, а страниц две — разрез по провалу печати
    SINGLE = "single"  # страница одна (или ни одной)


@dataclass(frozen=True)
class FrameFeatures:
    """Признаки кадра.

    Поля:
        dataset: Имя набора (кэша прогона).
        key: Ключ кадра в кэше (относительный путь папки кадра).
        n_pages: Страниц в ``pages.json``.
        n_inst: Экземпляров-страниц SAM 3 (карта ``inst`` после двойной сегментации, как её видит ``split_pages``).
        source: Откуда страницы (:class:`SplitSource`).
        book_w_mm: Ширина осевой рамки грубой маски книги, мм.
        book_h_mm: Высота той же рамки, мм.
        paper_w_mm: Ширина осевой рамки итоговой бумаги, мм.
        paper_h_mm: Высота той же рамки, мм.
        paper_aspect: Ширина / высота рамки бумаги (разворот ≈ 1.3–1.5, одна страница ≈ 0.7).
        w_left_mm: Ширина левой (первой) трапеции, мм (среднее верхней и нижней сторон); 0 — трапеции нет.
        w_right_mm: Ширина правой (последней) трапеции, мм; 0 — страница одна.
        h_mm: Высота трапеций, мм (наибольшая из средних длин боковых сторон).
        width_ratio: Узкая / широкая трапеция; 1 — страница одна.
        overlap_mm: Перекрытие трапеций по X, мм (плюс — правая заходит за правую сторону левой).
        narrow_keystone: Сход узкой трапеции, °.
        narrow_pose: Класс положения узкой страницы из ``pages.json``.
        print_spine: Печать (оболочки) на линии разреза: минимум профиля столбцов в окне ±:data:`SPINE_WINDOW_MM` /
            медиана профиля по бумаге; у разворота провал (≈0–0.3), у одной страницы печать сплошная (≈1).
        spine_rel: Положение линии разреза в рамке бумаги, доля ширины (0.5 — посередине).
    """

    dataset: str
    key: str
    n_pages: int
    n_inst: int
    source: str
    book_w_mm: float
    book_h_mm: float
    paper_w_mm: float
    paper_h_mm: float
    paper_aspect: float
    w_left_mm: float
    w_right_mm: float
    h_mm: float
    width_ratio: float
    overlap_mm: float
    narrow_keystone: float
    narrow_pose: str
    print_spine: float
    spine_rel: float

    def row(self) -> dict:
        """Строка CSV (числа округлены)."""
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


def bbox_mm(mask: np.ndarray) -> tuple[float, float, int, int]:
    """Осевая рамка маски в мм и её столбцы.

    Args:
        mask: Булева маска масштаба сегментации.

    Returns:
        ``(ширина мм, высота мм, первый столбец, последний столбец)``; пустая маска — нули.
    """
    if not mask.any():
        return 0.0, 0.0, 0, 0
    xs = np.nonzero(mask.any(axis=0))[0]
    ys = np.nonzero(mask.any(axis=1))[0]
    return (xs[-1] - xs[0] + 1) / SEG_PX_PER_MM, (ys[-1] - ys[0] + 1) / SEG_PX_PER_MM, int(xs[0]), int(xs[-1])


def quad_width_mm(quad: np.ndarray) -> float:
    """Ширина трапеции, мм: среднее длин верхней и нижней сторон (между концами левой и правой боковых).

    Args:
        quad: Вершины (4, 2), px сегментации.

    Returns:
        Ширина, мм.
    """
    left, right = vertical_sides(quad)
    top = np.linalg.norm(right[0] - left[0])
    bottom = np.linalg.norm(right[1] - left[1])
    return float((top + bottom) / 2 / SEG_PX_PER_MM)


def quad_height_mm(quad: np.ndarray) -> float:
    """Высота трапеции, мм: среднее длин боковых сторон.

    Args:
        quad: Вершины (4, 2), px сегментации.

    Returns:
        Высота, мм.
    """
    left, right = vertical_sides(quad)
    return float((np.linalg.norm(left[1] - left[0]) + np.linalg.norm(right[1] - right[0])) / 2 / SEG_PX_PER_MM)


def spine_x(left: np.ndarray, right: np.ndarray) -> float:
    """Абсцисса линии разреза: середина между правой стороной левой трапеции и левой стороной правой.

    Args:
        left: Трапеция левой страницы.
        right: Трапеция правой страницы.

    Returns:
        Абсцисса, px сегментации.
    """
    return float((np.sort(left[:, 0])[2:].mean() + np.sort(right[:, 0])[:2].mean()) / 2)


def print_at_spine(hulls: np.ndarray, paper: np.ndarray, x: float) -> float:
    """Печать на линии разреза относительно печати страницы.

    Профиль — число пикселей оболочек печати в каждом столбце. Берётся минимум профиля в окне ±:data:`SPINE_WINDOW_MM`
    у линии разреза и делится на медиану профиля по столбцам бумаги (без нулевых). У разворота у корешка печати нет —
    провал; у одной страницы, разрезанной по ложному корешку, печать на линии разреза та же, что на странице.

    Args:
        hulls: Оболочки своей печати (bool, масштаб сегментации).
        paper: Итоговая бумага (bool).
        x: Абсцисса линии разреза, px сегментации.

    Returns:
        Отношение (0 — провал до нуля); нет печати на бумаге — NaN.
    """
    profile = (hulls & paper).sum(axis=0).astype(float)
    base = profile[profile > 0]
    if not len(base):
        return float("nan")
    r = int(round(SPINE_WINDOW_MM * SEG_PX_PER_MM))
    lo, hi = max(int(x) - r, 0), min(int(x) + r + 1, len(profile))
    if hi <= lo:
        return float("nan")
    return float(profile[lo:hi].min() / np.median(base))


def frame_features(dataset: str, cache: Path, key: str) -> FrameFeatures:
    """Признаки одного кадра по его кэшу.

    Args:
        dataset: Имя набора.
        cache: Корень кэша прогона v2.
        key: Ключ кадра (относительный путь папки кадра в кэше).

    Returns:
        Признаки.
    """
    root = cache / key
    rows = json.loads((root / "pages.json").read_text())
    quads = [np.asarray(r["quad"], np.float64) for r in rows if r.get("quad")]
    poses = [r.get("pose", "") for r in rows if r.get("quad")]
    keystones = [float(r.get("keystone", 0.0)) for r in rows if r.get("quad")]
    inst = np.load(root / "sam3.npz")["inst"]
    n_inst = int(len([i for i in np.unique(inst) if i > 0]))
    # Версии кэша: маски шага «фон и книга» после вето (book_veto) — те, по которым строилась бумага.
    book_file = root / "book_veto.npz" if (root / "book_veto.npz").exists() else root / "book.npz"
    book_npz = np.load(book_file)
    book, hulls = book_npz["book"], book_npz["hulls"]
    paper = np.load(root / "paper.npz")["paper"]
    book_w, book_h, _, _ = bbox_mm(book)
    paper_w, paper_h, px0, px1 = bbox_mm(paper)
    widths = [quad_width_mm(q) for q in quads]
    heights = [quad_height_mm(q) for q in quads]
    source = SplitSource.SINGLE if len(rows) < 2 else SplitSource.INSTANCES if n_inst >= 2 else SplitSource.PRINT
    w_left = widths[0] if widths else 0.0
    w_right = widths[-1] if len(widths) >= 2 else 0.0
    ratio, overlap, narrow_ks, narrow_pose, spine_print, spine_rel = 1.0, 0.0, 0.0, "", float("nan"), float("nan")
    if len(quads) >= 2:
        left, right = quads[0], quads[-1]
        narrow = 0 if w_left < w_right else -1
        ratio = min(w_left, w_right) / max(w_left, w_right, 1e-6)
        overlap = float((np.sort(left[:, 0])[2:].mean() - np.sort(right[:, 0])[:2].mean()) / SEG_PX_PER_MM)
        narrow_ks, narrow_pose = keystones[narrow], poses[narrow]
        x = spine_x(left, right)
        spine_print = print_at_spine(hulls, paper, x)
        spine_rel = (x - px0) / max(px1 - px0, 1)
    return FrameFeatures(
        dataset=dataset,
        key=key,
        n_pages=len(rows),
        n_inst=n_inst,
        source=source.value,
        book_w_mm=book_w,
        book_h_mm=book_h,
        paper_w_mm=paper_w,
        paper_h_mm=paper_h,
        paper_aspect=paper_w / max(paper_h, 1e-6),
        w_left_mm=w_left,
        w_right_mm=w_right,
        h_mm=max(heights) if heights else 0.0,
        width_ratio=ratio,
        overlap_mm=overlap,
        narrow_keystone=narrow_ks,
        narrow_pose=narrow_pose,
        print_spine=spine_print,
        spine_rel=spine_rel,
    )


def frame_thumb(cache: Path, key: str, width: int = THUMB_WIDTH) -> np.ndarray:
    """Миниатюра кадра (``seg.png``) с трапециями страниц: левая синяя, правая красная.

    Args:
        cache: Корень кэша прогона v2.
        key: Ключ кадра.
        width: Ширина миниатюры, px.

    Returns:
        Картинка BGR.
    """
    root = cache / key
    img = cv2.imread(str(root / "seg.png"))
    rows = json.loads((root / "pages.json").read_text())
    quads = [np.asarray(r["quad"], np.float64) for r in rows if r.get("quad")]
    scale = width / img.shape[1]
    thumb = cv2.resize(img, (width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    for k, q in enumerate(quads):
        color = QUAD_COLORS[0] if k == 0 else QUAD_COLORS[1] if k == len(quads) - 1 else QUAD_COLORS[2]
        cv2.polylines(thumb, [np.round(q * scale).astype(np.int32)], True, color, 2, cv2.LINE_AA)
    return thumb
