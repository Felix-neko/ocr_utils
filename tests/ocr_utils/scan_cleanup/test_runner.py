"""Прогон по паку: порядок операций, запись результата, отчёт (``scan_cleanup.runner``).

GPU здесь не нужен: закрас проверяется подставной заливкой, которая просто красит
ROI ровным цветом.
"""

import numpy as np
import pytest
from PIL import Image

from ocr_utils.scan_cleanup.inpaint import InpaintOptions, inpaint_page
from ocr_utils.scan_cleanup.runner import CleanupParams, process_page, run_cleanup, select_pages, summary
from ocr_utils.scan_cleanup.source import load_markup
from ocr_utils.scan_markup.db.models import MASK_LIBRARY_STAMP

FILL = 111


class StubModels:
    """Заглушка ``GpuModels``: заливает дыру ровным цветом и считает вызовы."""

    def __init__(self):
        self.calls = 0

    def lama_fill_roi(self, roi, roi_mask, max_side=512, hole_max_px=None):
        self.calls += 1
        return np.full_like(roi, FILL)


@pytest.fixture
def params(pack, tmp_path):
    db_path, pack_dir, pack_name = pack
    return CleanupParams(
        db_path=db_path, pack_name=pack_name, pack_dir=pack_dir, out_dir=tmp_path / "out", smooth=_small_radii()
    )


def _small_radii():
    """Радиусы под маленький синтетический кадр (400x600, а не 3500x6100)."""
    from ocr_utils.scan_cleanup.smoothing import SmoothOptions

    return SmoothOptions(dilate_px=4.0, blur_px=16.0)


def page(pack, name: str):
    db_path, _pack_dir, pack_name = pack
    return next(p for p in load_markup(db_path, pack_name) if p.rel_path.endswith(name))


# ----------------------------------------------------------------------
# Одна полоса
# ----------------------------------------------------------------------


def test_writes_result_mirroring_the_subfolders(pack, params):
    report = process_page(page(pack, "0010.tif"), params)
    out = params.out_dir / "1970/01/0010.tif"

    assert report.status == "ok"
    assert out.exists()
    assert Image.open(out).size == (400, 600)


def test_dpi_and_suffix_are_preserved(pack, params, tmp_path):
    process_page(page(pack, "0010.tif"), params)
    out = params.out_dir / "1970/01/0010.tif"
    assert out.suffix == ".tif"
    # DPI переносится со входа; у синтетического файла его нет, и это не ошибка.
    assert Image.open(out).info.get("dpi") in (None, (72.0, 72.0), (1.0, 1.0))


def test_cover_is_copied_without_blurring_but_still_inpainted(pack, params):
    """Обложка размытие пропускает, а закрас — нет: там половина всех масок пака."""
    models = StubModels()
    report = process_page(page(pack, "0020.tif"), params, models)

    assert report.status == "copied"
    assert "полосная иллюстрация" in report.reason
    assert models.calls == 1
    assert report.zones_by_kind == {MASK_LIBRARY_STAMP: 1}

    import cv2

    out = cv2.imread(str(params.out_dir / "1970/01/0020.tif"), cv2.IMREAD_COLOR)
    assert out[120, 130, 0] == FILL  # печать закрашена
    assert out[10, 10, 0] > 200  # бумага не размыта


def test_page_without_masks_never_calls_the_filler(pack, params):
    models = StubModels()
    process_page(page(pack, "0010.tif"), params, models)
    assert models.calls == 0


def test_skip_if_exists_returns_early(pack, params):
    process_page(page(pack, "0010.tif"), params)
    report = process_page(page(pack, "0010.tif"), params)
    assert report.status == "skipped"


def test_missing_file_is_reported_not_fatal(pack, params):
    markup = page(pack, "0010.tif")
    (params.pack_dir / markup.rel_path).unlink()
    report = process_page(markup, params)
    assert report.status == "missing"


def test_no_part_files_left_behind(pack, params):
    process_page(page(pack, "0010.tif"), params)
    assert list(params.out_dir.rglob("*.part")) == []


def test_gray_is_computed_after_inpainting(pack, params):
    """Закрашенная печать не должна попасть в маску контента.

    Иначе размытие защитило бы то, чего на кадре уже нет.
    """
    markup = page(pack, "0020.tif")
    # Убираем полосную иллюстрацию, чтобы полоса пошла через размытие.
    markup = type(markup)(markup.rel_path, markup.width, markup.height, markup.dpi, markup.divisor, (), markup.masks)
    bgr_before = __import__("cv2").imread(str(markup.source_path(params.pack_dir)))
    # На месте печати рисуем тёмное пятно — его-то бинаризация и поймала бы.
    bgr_before[100:140, 100:160] = 20

    filled, report = inpaint_page(bgr_before, markup, InpaintOptions(), StubModels())
    assert report.zones == 1
    assert filled[120, 130, 0] == FILL


# ----------------------------------------------------------------------
# Пак целиком
# ----------------------------------------------------------------------


def test_select_pages_filters(pack, params):
    assert len(select_pages(params)) == 2
    params.only_with_masks = True
    assert [p.rel_path for p in select_pages(params)] == ["1970/01/0020.tif"]


def test_run_cleanup_without_inpaint_processes_everything(pack, params):
    params.do_inpaint = False
    params.jobs = 1
    reports = run_cleanup(params)

    assert len(reports) == 2
    assert {r.status for r in reports} == {"ok", "copied"}
    assert (params.out_dir / "1970/01/0010.tif").exists()
    assert (params.out_dir / "1970/01/0020.tif").exists()


def test_report_csv_is_written(pack, params, tmp_path):
    params.do_inpaint = False
    params.jobs = 1
    params.report_csv = tmp_path / "report.csv"
    run_cleanup(params)

    text = params.report_csv.read_text(encoding="utf-8")
    assert "rel_path" in text
    assert "1970/01/0010.tif" in text


def test_summary_mentions_every_status(pack, params):
    params.do_inpaint = False
    params.jobs = 1
    text = summary(run_cleanup(params))
    assert "Полос обработано: 2" in text
    assert "скопировано без изменений" in text
