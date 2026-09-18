"""Экспериментальный подпакет: полосы 1×N, подсказки из CSV детектора корешка, склейка переносов по словарю."""

from pathlib import Path

import pytest
from PIL import Image

from ocr_utils.experimental.damage_hints import (
    HintedPrompts,
    PageSide,
    hint_for_rel,
    hints_from_gutter_csv,
    read_gutter_csv,
)
from ocr_utils.experimental.strips import STRIP_OVERLAP, prepare_strips, rows_for_area, strip_boxes
from ocr_utils.external_ocr_services.ocr import PageJob, RunOptions
from ocr_utils.external_ocr_services.prompts import DEFAULT_DAMAGE_NOTE
from ocr_utils.external_ocr_services.tiling import describe


def test_strip_boxes_cover_frame_in_order_with_overlap():
    boxes = strip_boxes(3452, 6096, 3)
    assert [(b.col, b.row) for b in boxes] == [(0, 0), (0, 1), (0, 2)]
    assert boxes[0].top == 0 and boxes[-1].bottom == 6096 and all(b.left == 0 and b.right == 3452 for b in boxes)
    assert boxes[0].bottom - boxes[1].top == 2 * int(6096 / 3 * STRIP_OVERLAP / 2)
    assert strip_boxes(100, 200, 1) == [strip_boxes(100, 200, 1)[0]] and strip_boxes(100, 200, 1)[0].bottom == 200


def test_prepare_strips_scales_by_width_and_describe_reports_rows(tmp_path):
    path = tmp_path / "page.jpg"
    Image.new("L", (3452, 6096), 255).save(path)
    tiles = prepare_strips(path, 5, max_model_tile=2200)
    assert len(tiles) == 5 and all(t.width == 2200 for t in tiles) and all(t.height <= 900 for t in tiles), [
        t.height for t in tiles
    ]
    info = describe(tiles)
    assert info.ncols == 1 and info.nrows == 5 and [t.row for t in info.tiles] == [0, 1, 2, 3, 4]


def test_rows_for_area_matches_deepseek_cap():
    # Полоса 1966/03 при 2200 px: 5 полос с перекрытием 15 % — 2200×893 ≈ 1.96 Мпкс (ужмут), 6 — 2200×744 ≈ 1.64 (нет).
    assert rows_for_area(3452, 6096, 2200) == 6
    assert rows_for_area(3452, 6096, 1100) == 2  # при 1100 px площадь вчетверо меньше
    assert rows_for_area(500, 500, 2200) == 1  # маленькую картинку не увеличивают


def _gutter_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "gutter.csv"
    csv_path.write_text(
        "файл,балл,вердикт,полосы,поле_L,поле_R,таблица_L,таблица_R,шаг_строк,строк,наклон_сгиба,замечание\n"
        '/x/IMG_0006.jpg,0.97,текст,"L, R",0.04,0.00,0,0,60.0,40,1.0,\n'
        '/x/IMG_0046.jpg,0.51,текст,"L, R",0.30,0.29,0,0,60.0,40,1.0,\n'
        "/x/IMG_0009.jpg,nan,ок,,,,,,,,,одиночная страница\n",
        encoding="utf-8",
    )
    return csv_path


def test_hints_from_gutter_csv_by_side_and_score(tmp_path):
    verdicts = read_gutter_csv(_gutter_csv(tmp_path))
    assert set(verdicts) == {"IMG_0006", "IMG_0046"}, "кадр без балла пропущен"
    hints = hints_from_gutter_csv(_gutter_csv(tmp_path))
    assert (
        "RIGHT ends of the lines run into the binding gutter" in hints["IMG_0006_L"] and "hidden" in hints["IMG_0006_L"]
    )
    assert "LEFT ends of the lines" in hints["IMG_0006_R"]
    assert "nothing is hidden" in hints["IMG_0046_L"], "балл выше порога, но поле 0.30 — буквы видны"
    assert hint_for_rel(hints, Path("a/b/IMG_0006_2R.jpg")) == hints["IMG_0006_R"]
    assert hint_for_rel(hints, Path("IMG_0007_L.jpg")) is None and hint_for_rel(hints, Path("cover.jpg")) is None
    assert PageSide("L") is PageSide.LEFT


def test_hinted_prompts_replace_default_note_only_when_hint_exists(tmp_path):
    hints = hints_from_gutter_csv(_gutter_csv(tmp_path))
    hinted = HintedPrompts(hints)
    Image.new("L", (400, 600), 255).save(tmp_path / "IMG_0006_L.jpg")
    tiles = prepare_strips(tmp_path / "IMG_0006_L.jpg", 2)
    system, user = hinted(PageJob(Path("IMG_0006_L.jpg")), tiles, RunOptions())
    assert hints["IMG_0006_L"] in user and DEFAULT_DAMAGE_NOTE not in user and "2 overlapping tiles" in user
    assert "Transcribe the page" in system and hinted.used == 1
    _, plain = hinted(PageJob(Path("IMG_0007_L.jpg")), tiles, RunOptions())
    assert DEFAULT_DAMAGE_NOTE in plain and hinted.used == 1


@pytest.mark.parametrize("backend", ["pymorphy3", "mawo"])
def test_join_broken_hyphens_by_dictionary(backend):
    pytest.importorskip("pymorphy3" if backend == "pymorphy3" else "mawo_pymorphy3")
    from ocr_utils.experimental.hyphen_join import JoinRule, Morph, join_broken_hyphens

    morph = Morph(backend)
    text = "взять кре-диты и ва<supplied>л</supplied>ютных, материально-техническое, когда-нибудь, Мифи-нанц"
    out, report = join_broken_hyphens(text, morph, JoinRule.D)
    assert "кредиты" in out and "материально-техническое" in out and "когда-нибудь" in out and "Мифи-нанц" in out
    assert "ва<supplied>л</supplied>ютных" in out, "слово без дефиса не трогается"
    assert report.joined == ["кре-диты"] and "когда-нибудь" in report.kept
    tagged, report = join_broken_hyphens("кре-<supplied>ди</supplied>ты", morph, JoinRule.A)
    assert tagged == "кре<supplied>ди</supplied>ты" and report.joined == ["кре-диты"], "теги внутри слова сохраняются"
