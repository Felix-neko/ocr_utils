"""Признаки «растр или штрих»: тона области и растровая сетка в спектре."""

import numpy as np
import pytest

from ocr_utils.scan_markup.detection.dots import params_for_dpi
from ocr_utils.scan_markup.detection.tone import (
    LINEART_ENTROPY_THR,
    LINEART_MID_FRAC_THR,
    LINEART_SCREEN_PEAK_THR,
    ToneStats,
    looks_like_line_art,
    screen_peak,
    tone_maps,
    tone_stats,
)
from tests.ocr_utils.scan_markup import synthetic

SIZE = (1800, 1200)  # полоса 300 dpi
DPI = 300
BOX = (200, 200, 900, 1000)


def _screen_page() -> np.ndarray:
    """Полоса с растровым пятном. Шаг 4 px при 300 dpi — это 75 lpi, настоящая линиатура."""
    page = synthetic.paper(SIZE)
    synthetic.with_screen(page, BOX, pitch=4, radius=1)
    return page


def _line_art_page() -> np.ndarray:
    page = synthetic.paper(SIZE)
    page[BOX[1] : BOX[3], BOX[0] : BOX[2]] = synthetic.line_art((BOX[3] - BOX[1], BOX[2] - BOX[0]))
    return page


def _stats(page: np.ndarray, dpi: int = DPI) -> ToneStats:
    return tone_stats(tone_maps(page, params_for_dpi(dpi)), BOX)


def test_screen_is_not_line_art() -> None:
    """Растровое пятно правило штрихом не считает."""
    assert not looks_like_line_art(_stats(_screen_page()))


def test_line_art_is_line_art() -> None:
    """Штриховой рисунок правило ловит."""
    assert looks_like_line_art(_stats(_line_art_page()))


def test_screen_has_more_middle_tones() -> None:
    """Растр даёт промежуточные тона, штрих — почти нет: краска либо есть, либо нет."""
    screen, line = _stats(_screen_page()), _stats(_line_art_page())
    assert screen.mid_frac > line.mid_frac
    assert screen.entropy > line.entropy


def test_screen_peak_finds_the_lattice() -> None:
    """Пик сетки виден у растра и не виден у штриха."""
    params = params_for_dpi(DPI)
    tile = params.cell_px * 2
    screen_tile = _screen_page()[BOX[1] : BOX[1] + tile, BOX[0] : BOX[0] + tile]
    line_tile = _line_art_page()[BOX[1] : BOX[1] + tile, BOX[0] : BOX[0] + tile]
    scale = params.cell_px / params_for_dpi(600).cell_px
    assert screen_peak(screen_tile, scale) > screen_peak(line_tile, scale)


def test_period_band_follows_dpi() -> None:
    """Полоса поиска пика масштабируется разрешением: та же линиатура — вдвое больший шаг.

    Без масштабирования сетка полосы 600 dpi (шаг 8 px) искалась бы в полосе периодов
    5.5..11 px пересчитанной под 300 dpi, то есть 2.75..5.5, и не нашлась бы вовсе.
    """
    big = (3600, 2400)
    box = (400, 400, 1800, 2000)
    page = synthetic.paper(big)
    synthetic.with_screen(page, box, pitch=8, radius=2)
    stats = tone_stats(tone_maps(page, params_for_dpi(600)), box)
    assert stats.screen_peak > LINEART_SCREEN_PEAK_THR


def test_empty_area_gives_zeros() -> None:
    """Чистая бумага: судить не по чему, но и падать не на чем."""
    stats = _stats(synthetic.paper(SIZE))
    assert stats.screen_peak == 0.0  # ни одной плитки с краской
    assert stats.mid_frac == pytest.approx(0.0)


@pytest.mark.parametrize(
    "stats, expected",
    [
        (ToneStats(0.05, 5.0, 1.2), True),  # штрих: провалены все три
        (ToneStats(0.05, 5.0, 9.0), False),  # пересвеченная фотография: спасает сетка
        (ToneStats(0.30, 7.0, 1.1), False),  # фотография без сетки: спасают тона
        (ToneStats(0.05, 7.0, 1.1), False),  # одной доли средних тонов мало
    ],
)
def test_rule_is_a_conjunction(stats: ToneStats, expected: bool) -> None:
    """Штрихом область признаётся, только если провалила ВСЕ три признака сразу.

    По отдельности ни один не разделяет: пересвеченная фотография 1969/01 IMG_0033_2R имеет
    средние тона 0.111 и энтропию 5.77 — внутри штрихового диапазона, — и держится только на
    выступе пика 4.05.
    """
    assert looks_like_line_art(stats) is expected


def test_thresholds_stay_where_they_were_measured() -> None:
    """Пороги — не круглые числа наугад, а середины замеренных зазоров.

    Замер по 31 штриховой области против 75 фотографических: энтропия у штриха до 6.37, у
    ближайшей фотографии 6.66; выступ пика у штриха до 2.16, у ближайшей фотографии 4.05.
    Тест ловит случайную правку констант — менять их можно только вместе с новым замером.
    """
    assert (LINEART_MID_FRAC_THR, LINEART_ENTROPY_THR, LINEART_SCREEN_PEAK_THR) == (0.25, 6.5, 3.0)
