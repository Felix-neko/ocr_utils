"""Пример из README работает: на приложенной картинке находятся блоки и строки.

Единственный тест подпакета на НАСТОЯЩЕЙ странице: картинка лежит в самом подпакете
(`example_page.png`, 0.5 МБ), с дисков пака ничего не читается. Тест сторожит и пример, и то, что
он показывает, — если разбор на этой странице развалится, пример перестанет быть примером.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from click.testing import CliRunner

from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.example import EXAMPLE_PAGE, main
from ocr_utils.page_layout.text_blocks.page import analyse_gray

# Что на странице 1970/02 с.90: рубрика, заголовок, вводка полужирным и два столбца корпуса.
EXPECTED_BLOCKS = 5
EXPECTED_AXES = 60  # строк там 82; берём с запасом вниз, чтобы тест не ловил шум сглаживания


def test_example_page_is_shipped_with_the_subpackage():
    """Картинка примера лежит рядом с ним и читается как серая."""
    assert EXAMPLE_PAGE.is_file()
    gray = cv2.imread(str(EXAMPLE_PAGE), cv2.IMREAD_GRAYSCALE)
    assert gray is not None and gray.ndim == 2 and min(gray.shape) > 1000


def test_example_page_gives_blocks_and_lines():
    """Разбор приложенной страницы даёт столбцы корпуса, заголовок и вводку."""
    gray = cv2.imread(str(EXAMPLE_PAGE), cv2.IMREAD_GRAYSCALE)
    analysis = analyse_gray(gray, InkEngine())
    assert len(analysis.blocks) == EXPECTED_BLOCKS
    assert len(analysis.axes) >= EXPECTED_AXES
    # Два самых высоких блока — столбцы корпуса: в каждом три десятка рядов с одним шагом.
    body = sorted(analysis.blocks, key=lambda block: block.lines, reverse=True)[:2]
    assert all(block.lines > 30 for block in body)
    assert abs(body[0].pitch_mm - body[1].pitch_mm) < 0.5
    # Контур блока держит внутри все оси своих рядов — главное обещание границы-полосы.
    for block in analysis.blocks:
        polygon = block.envelope.polygon
        assert polygon.shape[0] >= 3 and polygon.shape[1] == 2


def test_example_script_runs_and_draws_a_picture(tmp_path: Path):
    """Сам скрипт примера запускается, печатает блоки и рисует картинку разбора.

    ``--no-show`` обязателен: иначе на машине без окна пример откроет системный просмотрщик.
    """
    picture = tmp_path / "blocks.png"
    result = CliRunner().invoke(main, ["--no-show", "--picture", str(picture)])
    assert result.exit_code == 0, result.output
    assert result.output.count("блок кол.") == EXPECTED_BLOCKS
    assert picture.is_file() and picture.stat().st_size > 0


def test_example_script_can_stay_in_the_console(tmp_path: Path):
    """Без картинки пример ничего не открывает и не пишет на диск."""
    result = CliRunner().invoke(main, ["--no-show"])
    assert result.exit_code == 0, result.output
    assert "рисунок сохранён" not in result.output
