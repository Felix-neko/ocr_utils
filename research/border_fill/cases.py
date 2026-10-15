"""Сборка входа закраса стенда по кадру: кадр без пальцев из кэша v2, copy back, маски бумаги и защиты, стык страниц в координатах прямоугольника."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr_utils.scan_cropping.v2.copy_back import LAYOUT_PAD_PX, copy_back_blocks
from ocr_utils.scan_cropping.v2.pipeline import read_layout, read_page_quads, to_frame
from research.border_fill.frames import key_of, load_masks, read_image
from research.border_fill.variants import FillInput


@dataclass(frozen=True)
class Case:
    """Кадр тестового множества.

    Поля:
        rel: Путь кадра относительно корня пака (как в выходе прогона).
        key: Ключ кэша v2.
        inp: Вход закраса.
        box: Рамка вырезки в координатах кадра.
    """

    rel: Path
    key: str
    inp: FillInput
    box: tuple[int, int, int, int]


def read_set(path: Path) -> list[Path]:
    """Кадры множества: по строке на кадр, путь относительно корня пака; пустые строки и ``#`` — пропуск."""
    rows = [line.split("#")[0].strip() for line in path.read_text().splitlines()]
    return [Path(r) for r in rows if r]


def spine_in_box(cache: Path, seg_to_frame: float, x0: int) -> float | None:
    """Абсцисса стыка страниц в координатах прямоугольника: середина между правой стороной левой трапеции и левой
    стороной правой (``pages.json``, px сегментации).

    Args:
        cache: Папка кадра в кэше v2.
        seg_to_frame: Множитель px сегментации → px кадра.
        x0: Левая граница рамки вырезки, px кадра.

    Returns:
        Абсцисса или None (страниц не две).
    """
    quads = read_page_quads(cache / "pages.json")
    if len(quads) != 2:
        return None
    left = np.sort(quads[0][:, 0])[2:].mean()
    right = np.sort(quads[1][:, 0])[:2].mean()
    return float((left + right) / 2 * seg_to_frame - x0)


def load_case(rel: Path, mask_cache: Path, clean_cache: Path, src_root: Path, px_per_mm: float) -> Case | None:
    """Вход закраса кадра.

    Маски бумаги и защиты, страницы и блоки surya — из кэша прогона ``mask_cache`` (как ``finish_frame``); кадр без
    пальцев — из ``clean_cache`` (прогон с ``--save-fingers``). Copy back повторяется, как в v2: блоки, задетые
    закрасом пальцев, с оригинала внутри «не фона» целиком (``--copy-back-over-fingers``).

    Args:
        rel: Путь кадра относительно корня пака.
        mask_cache: Корень кэша прогона с масками.
        clean_cache: Корень кэша прогона с ``clean.png``.
        src_root: Корень пака (исходные кадры, только чтение).
        px_per_mm: Плотность кадра.

    Returns:
        Вход или None, если бумаги нет.
    """
    key = key_of(rel)
    masks = load_masks(mask_cache / key)
    if masks.box is None:
        return None
    clean = read_image(clean_cache / key / "clean.png")
    original = read_image(src_root / rel)
    shape = clean.shape[:2]
    with np.load(mask_cache / key / "book.npz") as data:
        book_seg = data["book"]
    with np.load(clean_cache / key / "fingers.npz") as data:
        paint = data["paint"]
    polys = read_layout(mask_cache / key / "layout.json") or []
    top, _, _ = copy_back_blocks(original, clean, polys, paint, to_frame(book_seg, shape), LAYOUT_PAD_PX)
    x0, y0, x1, y1 = masks.box
    seg_to_frame = shape[0] / book_seg.shape[0]
    inp = FillInput(
        base=clean[y0:y1, x0:x1],
        top=top[y0:y1, x0:x1],
        paper=masks.crop(masks.paper),
        keep=masks.crop(masks.protected),
        px_per_mm=px_per_mm,
        spine_x=spine_in_box(mask_cache / key, seg_to_frame, x0),
    )
    return Case(rel, key, inp, masks.box)


def load_case_from_output(rel: Path, run_dir: Path, px_per_mm: float) -> Case | None:
    """Вход закраса по готовому итогу прогона v2 (без GPU-шагов и ``--save-fingers``).

    Итог — прямоугольник вырезки: бумага и защита в нём — финальные пиксели (после copy back, тени пальца и уровней),
    вне них — прежний закрас, который варианты перекрашивают заново. Опора pull-push — полоса бумаги у края, она та
    же, что видит пользователь; отличие от входа ``load_case`` — только в том, что бумага уже прошла copy back и уровни.

    Args:
        rel: Путь исходного кадра относительно корня пака.
        run_dir: Папка прогона v2 (``output/`` с итогами ``.tiff`` и ``cache/`` с масками).
        px_per_mm: Плотность кадра.

    Returns:
        Вход или None, если бумаги нет.
    """
    key = key_of(rel)
    cache = run_dir / "cache" / key
    masks = load_masks(cache)
    if masks.box is None:
        return None
    out = read_image(run_dir / "output" / rel.with_suffix(".tiff"))
    paper, keep = masks.crop(masks.paper), masks.crop(masks.protected)
    if out.shape[:2] != paper.shape:
        raise ValueError(f"размер итога {out.shape[:2]} ≠ рамке {paper.shape}")
    with np.load(cache / "book.npz") as data:
        seg_h = data["book"].shape[0]
    spine = spine_in_box(cache, masks.paper.shape[0] / seg_h, masks.box[0])
    return Case(rel, key, FillInput(out, out, paper, keep, px_per_mm, spine), masks.box)
