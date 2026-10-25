"""Тесты выхода страниц: части по флагам, области страниц по полю (кривая линия разреза), нарезка с закрасом, отсев кадров по сайдкарам."""

from types import SimpleNamespace

import numpy as np

from ocr_utils.frame_flags.flags import FrameFlag, FrameFlags, write_flags
from ocr_utils.scan_cropping.v2.fill import FillMethod
from ocr_utils.scan_cropping.v2.pipeline import RunParams, frame_jobs
from ocr_utils.scan_cropping.v2.rectify.coons import Layout, PageRect
from ocr_utils.scan_cropping.v2.rectify.pages import PagePart, cut_line, cut_pages, page_areas, wanted_parts


def flags(*values: FrameFlag) -> FrameFlags:
    """Флаги кадра из значений enum."""
    return FrameFlags(frozenset(values))


def layout() -> Layout:
    """Холст 200×100: страницы 10..100 и 100..190 по X, 10..90 по Y, корешок на X = 100."""
    return Layout(200, 100, (PageRect(10, 10, 100, 90), PageRect(100, 10, 190, 90)), 100.0)


def test_wanted_parts():
    assert wanted_parts(FrameFlags()) == [PagePart.LEFT, PagePart.RIGHT]
    assert wanted_parts(flags(FrameFlag.IGNORE_LEFT)) == [PagePart.RIGHT]
    assert wanted_parts(flags(FrameFlag.IGNORE_RIGHT)) == [PagePart.LEFT]
    assert wanted_parts(flags(FrameFlag.IGNORE_BOTH)) == []
    assert wanted_parts(flags(FrameFlag.NO_SPLITTING)) == [PagePart.SPREAD]
    # Запрет освещения на части не влияет.
    assert wanted_parts(flags(FrameFlag.NO_ILLUMINATION_LEFT)) == [PagePart.LEFT, PagePart.RIGHT]


def test_areas_without_field_are_rects():
    lay = layout()
    zero = (np.zeros((100, 200), np.float32), np.zeros((100, 200), np.float32))
    left, right = page_areas(lay, zero)
    assert not (left & right).any()
    assert left[50, 10] and left[50, 99] and not left[50, 100] and not left[5, 50]
    assert right[50, 100] and right[89, 189] and not right[90, 150]


def test_curved_cut_follows_field():
    lay = layout()
    # Поле по X растёт сверху вниз от −10 до +10 px: точка Кунса уходит вправо, линия разреза — влево.
    dx = np.repeat(np.linspace(-10, 10, 100, dtype=np.float32)[:, None], 200, axis=1)
    dense = (dx, np.zeros_like(dx))
    left, right = page_areas(lay, dense)
    assert not (left & right).any()
    line = cut_line(lay, dense, step=1)
    # Вверху разрез правее корешка, внизу левее: линия кривая (наклонная), а не прямая X = 100.
    assert line[0, 0] > 105 and line[-1, 0] < 95
    # Сверху вниз граница левой области смещается влево вместе с разрезом.
    assert left[20].nonzero()[0].max() > left[80].nonzero()[0].max()


def test_cut_pages_fills_to_rectangle():
    lay = layout()
    dx = np.repeat(np.linspace(-6, 6, 100, dtype=np.float32)[:, None], 200, axis=1)
    image = np.full((100, 200, 3), 230, np.uint8)
    # Правая страница темнее: закрас левой у кривого разреза не должен затянуть её краску.
    image[:, 100:] = 120
    result = SimpleNamespace(layout=lay, canvas=lay, dense=(dx, np.zeros_like(dx)), final=image)
    pages = cut_pages(result, [PagePart.LEFT, PagePart.RIGHT], 300.0, FillMethod.PULL_PUSH, 0.0)
    assert [p.part for p in pages] == [PagePart.LEFT, PagePart.RIGHT]
    left_img = pages[0].image
    assert left_img.ndim == 3 and left_img.shape[0] == 80
    # Внутри области левой страницы пиксели взяты как есть.
    assert int(left_img[40, 20, 0]) == 230
    spread = cut_pages(result, [PagePart.SPREAD], 300.0, FillMethod.PULL_PUSH, 0.0)
    assert spread[0].image.shape[1] >= 180


def test_frame_jobs_reads_and_filters(tmp_path):
    for name in ("IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg", "IMG_0004.jpg"):
        (tmp_path / name).write_bytes(b"jpeg")
    write_flags(tmp_path / "IMG_0001.jpg", flags(FrameFlag.IGNORE_BOTH))
    write_flags(tmp_path / "IMG_0002.jpg", flags(FrameFlag.IGNORE_RIGHT))
    (tmp_path / "IMG_0003.jpg.flags.json").write_text('{"schema": 1, "flags": ["ignore-rigth"]}')
    files = sorted(tmp_path.glob("*.jpg"))
    params = RunParams(tmp_path, tmp_path / "out", tmp_path / "work")
    jobs, skipped = frame_jobs(files, params)
    assert [j.rel.name for j in jobs] == ["IMG_0002.jpg", "IMG_0004.jpg"]
    assert jobs[0].flags.label() == "ignore-right" and jobs[1].flags.empty
    assert {(s.key, s.status) for s in skipped} == {("IMG_0001", "ignored"), ("IMG_0003", "bad_flags")}
    # Без чтения флагов идут все кадры.
    jobs_off, skipped_off = frame_jobs(files, RunParams(tmp_path, tmp_path / "out", tmp_path / "work", use_flags=False))
    assert len(jobs_off) == 4 and not skipped_off


def test_flags_control_equalize_and_field(monkeypatch):
    from ocr_utils.scan_cropping.v2.rectify import text_run
    from ocr_utils.scan_cropping.v2.rectify.geometry import SpreadGeometry
    from tests.ocr_utils.scan_cropping.v2.rectify.test_coons import spread

    image = np.full((400, 500, 3), 220, np.uint8)
    geometry = SpreadGeometry(spread(0.0), (0, 0, 500, 400), (1.0, 1.0))
    seen = {}

    def fake_equalize(img, masks, skip, dpi):
        seen["skip"] = list(skip)
        return img, [None, None]

    analyzed = []
    monkeypatch.setattr(text_run, "equalize_pages", fake_equalize)
    monkeypatch.setattr(text_run, "analyze_page", lambda *a, **k: analyzed.append(a[3]) or None)
    monkeypatch.setattr(text_run, "page_regions", lambda *a, **k: ())
    controls = (text_run.PageControl(equalize=False), text_run.PageControl(field=False))
    params = text_run.TextParams(passes=0)
    text_run.text_frame(image, geometry, 300.0, params, controls)
    # Серая бумага не цветная: запрет освещения — только у левой по флагу.
    assert seen["skip"] == [True, False]
    # Проход поля: правая (игнорируемая) не разбирается, левая разбирается.
    result = text_run.text_frame(image, geometry, 300.0, text_run.TextParams(passes=1), controls)
    assert len(analyzed) == 2 and all(rect == result.layout.rects[0] for rect in analyzed)
    assert result.analyses[0][1] is None


def test_finger_mask_follows_page_and_passport_roundtrip(tmp_path):
    from ocr_utils.scan_cropping.v2.rectify.pages import page_mask_of
    from ocr_utils.scan_cropping.v2.rectify.passport import PagePassport, read_passport, write_passport, zone_contours

    lay = layout()
    zero = np.zeros((100, 200), np.float32)
    # Тождественная карта: холст = итог v2.
    ys, xs = np.mgrid[0:100, 0:200].astype(np.float32)
    image = np.full((100, 200, 3), 230, np.uint8)
    result = SimpleNamespace(layout=lay, dense=(zero, zero), final=image, maps=(xs, ys))
    pages = cut_pages(result, [PagePart.LEFT, PagePart.RIGHT], 254.0, FillMethod.PULL_PUSH, 0.0)
    fingers = np.zeros((100, 200), bool)
    fingers[40:70, 150:180] = True  # «палец» на правой странице, 9 мм² при 10 px/мм
    left_zone = page_mask_of(result, fingers, pages[0])
    right_zone = page_mask_of(result, fingers, pages[1])
    assert not left_zone.any()
    # Правая страница начинается на холсте с (100, 10): квадрат — на (50..80, 30..60) страницы.
    assert right_zone.sum() == 900 and right_zone[30:60, 50:80].all()
    passport = PagePassport("1974/IMG_0001.jpg", "R", ["cover-right"], True, 254.0, 90, 80, 0.5, False, 9.0,
                            zone_contours(right_zone, 254.0))  # fmt: skip
    write_passport(tmp_path / "IMG_0001_R", passport, right_zone)
    back = read_passport(tmp_path / "IMG_0001_R.json")
    assert back == passport and (tmp_path / "IMG_0001_R.fingers.png").exists()
    assert len(back.finger_zones) == 1
