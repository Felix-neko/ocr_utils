"""Сквозная проверка CLI на синтетической папке."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import cv2

from ocr_utils.show_through_detection.cli import main
from tests.ocr_utils.show_through_detection.pages import add_show_through, draw_page, expose, scan, spread

NOISE = 1.5


@pytest.fixture
def folder(tmp_path):
    """Папка со сканами, где плохие развороты названы ``bleed_*``.

    Имена кодируют ожидание: отчёт обязан вытащить их наверх. Так тест читается без
    сверки с числами, а если порядок сломается, видно сразу, что именно всплыло.

    Args:
        tmp_path: Временная папка pytest.

    Returns:
        Путь к папке со сканами.
    """
    directory = tmp_path / "scans"
    directory.mkdir()
    for i in range(6):
        page = spread(draw_page(seed=i), draw_page(seed=100 + i))
        cv2.imwrite(str(directory / f"clean_{i:02d}.png"), expose(scan(page), noise=NOISE))
    for i in range(2):
        left = add_show_through(draw_page(seed=200 + i), 0.45, seed=300 + i)
        page = spread(left, draw_page(seed=400 + i))
        cv2.imwrite(str(directory / f"bleed_{i:02d}.png"), expose(scan(page), noise=NOISE))
    return directory


def run(*args):
    """Запускает CLI и проверяет успешный выход.

    Args:
        *args: Аргументы командной строки.

    Returns:
        Результат ``CliRunner.invoke``.
    """
    result = CliRunner().invoke(main, [*args, "--workers", "1", "--quiet"])
    assert result.exit_code == 0, result.output + str(result.exception)
    return result


# Порог для синтетики, в долях от калибровочного. Калибровка ``ghost_ink`` привязана
# к настоящей печати на настоящей бумаге, и рисованные штрихи до неё не дотягивают
# (0.0008–0.0011 против порога 0.010). Занижаем порог, иначе списки на пересканирование
# в тестах всегда пустые и проверять нечего. Чистые синтетические полосы дают ровно
# ноль, так что отделение брака от нормы остаётся честным.
SYNTHETIC_THRESHOLD = "0.05"


def halves_part(output: str) -> str:
    """Первая таблица отчёта — по полосам.

    Args:
        output: Полный вывод CLI.

    Returns:
        Кусок вывода до второй таблицы.
    """
    return output.split("== 2. КАДРЫ")[0]


def test_bleeding_halves_come_first(folder) -> None:
    """Полосы с просветом обязаны оказаться в самом верху отчёта."""
    output = halves_part(run(str(folder), "--worst-count", "4").output)
    top = [line for line in output.splitlines() if ".png" in line][:2]
    assert all("bleed_" in line for line in top), f"наверху не те полосы:\n{output}"


def test_side_of_the_spread_is_reported(folder) -> None:
    """Просвет только на левой полосе — отчёт обязан назвать именно её.

    Ради этого разворот и делится: пересканировать всё равно кадр целиком, но знать,
    какую страницу смотреть в бумажном экземпляре, нужно.
    """
    output = run(str(folder), "--worst-count", "2").output
    top = [line for line in halves_part(output).splitlines() if "bleed_" in line]
    assert top and all("[L]" in line for line in top), f"сторона названа неверно:\n{output}"


def test_all_four_outputs_are_written(folder, tmp_path) -> None:
    """Каждый заявленный выход должен появиться на диске и быть непустым."""
    txt, md, csv_path, links = (tmp_path / n for n in ("r.txt", "r.md", "r.csv", "links"))
    run(
        str(folder),
        "--threshold",
        SYNTHETIC_THRESHOLD,
        "--worst-percent",
        "20",
        "--txt-report",
        str(txt),
        "--md-report",
        str(md),
        "--csv",
        str(csv_path),
        "--link-dir",
        str(links),
    )
    for path in (txt, md, csv_path):
        assert path.exists() and path.stat().st_size > 0, f"{path} не записан"
    assert links.is_dir()
    assert list(links.iterdir()), "папка симлинков пуста — выход не проверен"
    # В CSV попадают ВСЕ полосы, а не только показанные: он для калибровки порога.
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) >= 16


def test_link_names_carry_year_and_issue(tmp_path) -> None:
    """Имя симлинка обязано говорить, откуда кадр, а не только как он назван в папке.

    В паке файлы зовутся ``09_0005.jpg`` — девятый номер какого года, по имени не
    понять, а листают именно эту папку. Год и выпуск берутся из пути относительно
    корня прогона.
    """
    root = tmp_path / "pack"
    issue = root / "1938" / "09"
    issue.mkdir(parents=True)
    left = add_show_through(draw_page(seed=1), 0.45, seed=2)
    cv2.imwrite(str(issue / "09_0005.png"), expose(scan(spread(left, draw_page(seed=3))), noise=NOISE))
    cv2.imwrite(str(issue / "09_0006.png"), expose(scan(spread(draw_page(seed=4), draw_page(seed=5))), noise=NOISE))

    links = tmp_path / "links"
    run(str(root), "--recursive", "--threshold", SYNTHETIC_THRESHOLD, "--worst-count", "1", "--link-dir", str(links))
    names = [p.name for p in links.iterdir()]
    assert names, "симлинков не создано"
    assert all("1938_09_" in n for n in names), f"год и выпуск не попали в имя: {names}"
    assert all(n.endswith(".png") for n in names)
    assert all("_L_" in n for n in names), f"сторона с просветом не названа: {names}"
    assert all(p.is_symlink() and p.resolve().exists() for p in links.iterdir())


def test_link_names_stay_short_on_a_flat_folder(folder, tmp_path) -> None:
    """На плоской папке приставки быть не должно — добавлять к имени нечего."""
    links = tmp_path / "links"
    run(str(folder), "--threshold", SYNTHETIC_THRESHOLD, "--worst-count", "2", "--link-dir", str(links))
    for path in links.iterdir():
        # <позиция>_<×порог>_<стороны>_<имя>: после трёх служебных полей идёт сразу
        # исходное имя, без папок. Считать подчёркивания нельзя — они есть и в самом имени.
        assert path.name.split("_", 3)[3] == path.resolve().name, f"лишняя приставка: {path.name}"


def test_sides_are_listed_in_page_order() -> None:
    """«L, R», а не «R, L»: порядок сторон не должен зависеть от сборки результата."""
    from ocr_utils.show_through_detection.analysis import FileResult, HalfResult

    path = Path("x.jpg")
    result = FileResult(
        path=path, halves=[HalfResult(path=path, side="R", severity=3.0), HalfResult(path=path, side="L", severity=2.0)]
    )
    assert result.worst_sides() == "L, R"


def test_csv_holds_every_half(folder, tmp_path) -> None:
    """CSV обязан содержать по строке на полосу, включая заведомо чистые."""
    csv_path = tmp_path / "r.csv"
    run(str(folder), "--worst-count", "1", "--csv", str(csv_path))
    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()[1:]
    assert len(rows) == 16, f"8 разворотов = 16 полос, а в CSV {len(rows)}"


def test_combo_requires_an_explicit_selection(folder) -> None:
    """У среднего ранга нет калибровочного порога, и молча его выдумывать нельзя."""
    result = CliRunner().invoke(main, [str(folder), "--algorithm", "combo", "--workers", "1", "--quiet"])
    assert result.exit_code != 0
    assert "combo" in result.output


def test_conflicting_selection_options_are_refused(folder) -> None:
    """--worst-count и --worst-percent взаимоисключающи."""
    result = CliRunner().invoke(main, [str(folder), "--worst-count", "3", "--worst-percent", "10"])
    assert result.exit_code != 0
    assert "взаимоисключающи" in result.output


def test_empty_folder_is_reported(tmp_path) -> None:
    """Папка без изображений — понятная ошибка, а не пустой отчёт."""
    empty = tmp_path / "empty"
    empty.mkdir()
    result = CliRunner().invoke(main, [str(empty)])
    assert result.exit_code != 0
    assert "не найдено" in result.output
