"""Фасад PageLayout: порядок детекторов, исключения, surya как затравка, два этапа, кэш только для чтения."""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ocr_utils.page_layout import WORK_DPI
from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
from ocr_utils.page_layout.geometry import Box, iou
from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.regions import RegionKind
from ocr_utils.page_layout.surya import OnMiss, SuryaCache, SuryaMissing, SuryaSource
from ocr_utils.page_layout.surya.blocks import from_boxes
from tests.ocr_utils.scan_markup import synthetic

DPI = 600
SIZE = (6000, 4000)  # (h, w): копия 150 dpi — 1500x1000
PHOTO = (300, 400, 1700, 1600)  # растр (x1, y1, x2, y2) в оригинале
TABLE = (2200, 400, 3700, 1500)
DRAWING = (300, 2300, 2200, 4200)
INK = 25


def _ruled_table(gray: np.ndarray, box) -> None:
    """Таблица с «буквами» в ячейках; линейки не касаются друг друга, как в печати."""
    x0, y0, x1, y1 = box
    xs = list(range(x0, x1 + 1, 375))
    ys = list(range(y0, y1 + 1, 275))
    for y in ys:
        gray[y : y + 8, x0 : x1 + 8] = INK
    for x in xs:
        gray[y0 : y1 + 8, x : x + 8] = INK
    for y in ys:
        for x in xs:
            gray[y - 12 : y + 20, x - 12 : x + 20] = synthetic.PAPER
    for top, bottom in zip(ys[:-1], ys[1:]):
        for left, right in zip(xs[:-1], xs[1:]):
            for cx in range(left + 60, right - 60, 44):
                gray[top + 90 : top + 138, cx : cx + 24] = INK


def _drawing(gray: np.ndarray, box, seed: int = 0) -> None:
    x0, y0, x1, y1 = box
    rng = np.random.default_rng(seed)
    points = rng.integers([x0, y0], [x1, y1], size=(70, 2)).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(gray, [points], False, INK, 7)


def _rotated_text(gray: np.ndarray, x: int, y: int, glyphs: int = 24, glyph=(24, 46), step: int = 30) -> None:
    """Колонка «букв» лёжа: цепочка глифов по вертикали, как боковая подпись оси."""
    w, h = glyph
    for i in range(glyphs):
        gray[y + i * step : y + i * step + w, x : x + h] = INK


def _page_array(photo=True, table=True, drawing=True, rotated: "tuple[int, int] | None" = None) -> np.ndarray:
    gray = synthetic.text_page(SIZE, line_step=90, glyph_w=24, glyph_h=48, char_step=44, margin=120)
    gray[PHOTO[1] - 100 : PHOTO[3] + 100, PHOTO[0] - 100 : PHOTO[2] + 100] = synthetic.PAPER
    gray[TABLE[1] - 100 : TABLE[3] + 100, TABLE[0] - 100 : TABLE[2] + 100] = synthetic.PAPER
    gray[DRAWING[1] - 100 : DRAWING[3] + 100, DRAWING[0] - 100 : DRAWING[2] + 100] = synthetic.PAPER
    if photo:
        synthetic.with_screen(gray, PHOTO)
    if table:
        _ruled_table(gray, TABLE)
    if drawing:
        _drawing(gray, DRAWING)
    if rotated is not None:
        x, y = rotated
        gray[y - 60 : y + 820, x - 60 : x + 120] = synthetic.PAPER
        _rotated_text(gray, x, y)
    return gray


def _image(tmp_path: Path, name: str = "1977/01/0010_2R", **kwargs) -> PageImage:
    gray = _page_array(**kwargs)
    path = tmp_path / "scan" / f"{name.replace('/', '_')}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.stack([gray] * 3, -1)).save(path, dpi=(DPI, DPI))
    return PageImage.from_file(path, Variant.SCAN, cache_name=name)


def _overlap(box: Box, target) -> float:
    return iou(box, Box(*target))


NO_SURYA = LayoutOptions(use_surya=False)


def test_order_raster_tables_then_line_art_outside_them(tmp_path: Path) -> None:
    layout = PageLayout(_image(tmp_path), options=NO_SURYA).process()
    assert layout.finished and not layout.surya_used
    assert [r.kind for r in layout.raster_pics] == [RegionKind.GRAYSCALE]
    assert _overlap(layout.raster_pics[0].box, PHOTO) > 0.5
    assert len(layout.tables) == 1 and _overlap(layout.tables[0].box, TABLE) > 0.5
    assert layout.tables[0].info["kind"] == "таблица" and layout.tables[0].confidence is not None
    assert layout.line_arts, "рисунок штрихом должен найтись"
    for art in layout.line_arts:
        assert art.kind is RegionKind.LINE_ART
        assert _overlap(art.box, PHOTO) < 0.1 and _overlap(art.box, TABLE) < 0.1, "line art не внутри растра и таблицы"
    assert any(_overlap(a.box, DRAWING) > 0.4 for a in layout.line_arts)
    assert layout.rotated_text_not_in_tables_regions == []
    for region in layout.regions:
        assert 0 <= region.box.x0 <= region.box.x1 <= layout.image.width


def test_surya_picture_over_drawing_becomes_line_art_seed(tmp_path: Path) -> None:
    """Блок Picture над штрихом — затравка line art (surya зовёт штрих фотографией), над растром — растр."""
    image = _image(tmp_path)
    w, h = image.size_at(WORK_DPI)
    k = WORK_DPI / DPI
    blocks = from_boxes(
        [
            ("Picture", 0.9, Box(*(int(v * k) for v in DRAWING))),
            ("Picture", 0.9, Box(*(int(v * k) for v in PHOTO))),
            ("Table", 0.8, Box(*(int(v * k) for v in TABLE))),
        ],
        w,
        h,
    )
    layout = PageLayout(image)
    layout.prepare(None)
    assert layout.needs_surya
    layout.finish(blocks)
    assert layout.surya_used and layout.raw_surya_content is not None
    assert layout.raw_surya_content.width == image.width
    assert [r.kind for r in layout.raster_pics] == [RegionKind.GRAYSCALE]
    art = max(layout.line_arts, key=lambda r: _overlap(r.box, DRAWING))
    assert "surya:Picture" in art.info["sources"] and art.confidence >= 2 / 3
    assert all(_overlap(a.box, PHOTO) < 0.1 for a in layout.line_arts)


def _prepare_in_child(layout: PageLayout) -> PageLayout:
    return layout.prepare(None)


def test_two_phase_across_processes_equals_single_process(tmp_path: Path) -> None:
    image = _image(tmp_path)
    with ProcessPoolExecutor(max_workers=1) as pool:
        prepared = pool.submit(_prepare_in_child, PageLayout(image, options=NO_SURYA)).result()
    assert prepared.finished, "без surya всё достраивается в воркере"
    assert prepared.image._bgr is None and prepared.image._gray is None, "полный кадр не ездит"
    single = PageLayout(_image(tmp_path), options=NO_SURYA).process()
    assert [r.box for r in prepared.regions] == [r.box for r in single.regions]

    # С surya: воркер отдаёт объект с кадром surya, родитель достраивает.
    with ProcessPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(_prepare_in_child, PageLayout(_image(tmp_path))).result()
    assert waiting.needs_surya and waiting.image.surya_frame.shape[2] == 3
    clone = pickle.loads(pickle.dumps(waiting))
    clone.finish(None)
    assert [r.box for r in clone.regions] == [r.box for r in single.regions]


def test_readonly_cache_miss_fails_loudly_unless_skip(tmp_path: Path) -> None:
    cache = SuryaCache(tmp_path / "cache", readonly=True)
    with pytest.raises(SuryaMissing):
        PageLayout(_image(tmp_path)).process(SuryaSource(cache))
    layout = PageLayout(_image(tmp_path)).process(SuryaSource(cache, on_miss=OnMiss.SKIP))
    assert layout.finished and not layout.surya_used

    class _Model:
        @staticmethod
        def predict_one(frame):
            return from_boxes([], frame.shape[1], frame.shape[0])

        @staticmethod
        def name():
            return "stub"

    writable = SuryaCache(tmp_path / "cache")
    first = PageLayout(_image(tmp_path)).process(SuryaSource(writable, _Model()))
    assert first.surya_used and writable.has(Variant.SCAN, "1977/01/0010_2R")
    second = PageLayout(_image(tmp_path)).process(SuryaSource(SuryaCache(tmp_path / "cache", readonly=True)))
    assert second.surya_used, "второй раз — из кэша, модель не нужна"


def test_rotated_text_outside_tables_only(tmp_path: Path) -> None:
    image = _image(tmp_path, rotated=(3000, 2600))
    layout = PageLayout(image, options=NO_SURYA).process()
    zones = layout.rotated_text_not_in_tables_regions
    assert zones and all(z.kind is RegionKind.ROTATED_TEXT for z in zones)
    assert any(z.box.x0 <= 3000 <= z.box.x1 and z.box.y0 <= 2700 <= z.box.y1 for z in zones)
    assert all(_overlap(z.box, TABLE) == 0 for z in zones)

    inside = _image(tmp_path, name="1977/01/0011_1L", rotated=(TABLE[0] + 500, TABLE[1] + 300))
    layout = PageLayout(inside, options=NO_SURYA).process()
    assert not any(_overlap(z.box, TABLE) > 0 for z in layout.rotated_text_not_in_tables_regions)


def test_cover_page_is_ready_without_pixels(tmp_path: Path) -> None:
    image = _image(tmp_path)
    layout = PageLayout(image, options=LayoutOptions(first_page_is_cover=True), order_index=0).prepare(None)
    assert layout.finished and layout.is_cover and image._bgr is None
    assert [r.kind for r in layout.raster_pics] == [RegionKind.COLOR] and layout.raster_pics[0].full_page
    assert layout.tables == layout.line_arts == [] and layout.best_page_orientation is not None
    assert layout.best_page_orientation.rotate_cw == 0


def test_find_subset_and_orientation(tmp_path: Path) -> None:
    image = _image(tmp_path)
    options = LayoutOptions(use_surya=False, orientation_detectors=("ink_axis",))
    layout = PageLayout(image, find={Find.TABLES, Find.ORIENTATION}, options=options).process()
    assert layout.tables and layout.raster_pics == [] and layout.line_arts == []
    assert layout.best_page_orientation is not None and "ink_axis" in layout.orientation_verdicts
