"""Сквозные проверки CLI: отбор худших, форматы файлов, отчёты."""

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from tests.ocr_utils.defocus_detection.pages import blur, draw_page

from ocr_utils.defocus_detection.cli import main


# Кадры делаются крупными: на мелкой сетке зонального отчёта тайл должен содержать
# достаточно краёв, иначе полосы не измеряются и второй отчёт остаётся пустым.
PAGE = dict(height=1536, width=1536)


@pytest.fixture
def folder(tmp_path):
    """Папка из восьми резких страниц и двух заметно размытых.

    Размытые называются ``soft_*`` — по имени и проверяем, что они всплыли наверх.
    Резкие тоже слегка размыты: у настоящего снимка край всегда шире пикселя.
    """
    directory = tmp_path / "scans"
    directory.mkdir()
    for i in range(8):
        cv2.imwrite(str(directory / f"sharp_{i:02d}.png"), blur(draw_page(seed=i, **PAGE), 0.7))
    for i in range(2):
        cv2.imwrite(str(directory / f"soft_{i:02d}.png"), blur(draw_page(seed=100 + i, **PAGE), 1.8))
    return directory


def overall_part(output: str) -> str:
    """Часть вывода до второй таблицы — чтобы считать строки только первого отчёта.

    Args:
        output: Полный вывод CLI.

    Returns:
        Текст первого отчёта.
    """
    return output.split("2. ЗОНАЛЬНЫЙ")[0]


def run(*args):
    """Запускает CLI и возвращает результат, падая с понятным текстом при ошибке.

    Args:
        *args: Аргументы командной строки.

    Returns:
        Объект ``click.testing.Result``.
    """
    result = CliRunner().invoke(main, [*args, "--workers", "1", "--quiet"])
    assert result.exit_code == 0, result.output + str(result.exception)
    return result


def test_lists_every_file_when_no_limit_given(folder) -> None:
    """Без --worst-* печатаются все файлы с числовой метрикой."""
    output = run(str(folder)).output
    assert "Показано 10 из 10 файлов" in output
    for name in (p.name for p in folder.iterdir()):
        assert name in output


def test_worst_first_ordering(folder) -> None:
    """Размытые страницы должны стоять в начале отчёта."""
    lines = [line for line in overall_part(run(str(folder)).output).splitlines() if ".png" in line]
    assert all("soft_" in line for line in lines[:2]), lines[:4]


def test_worst_count_limits_the_report(folder) -> None:
    """--worst-count оставляет ровно N строк, самых подозрительных."""
    output = overall_part(run(str(folder), "--worst-count", "2").output)
    assert "Показано 2 из 10 файлов" in output
    assert output.count("soft_") == 2
    assert "sharp_" not in output


def test_worst_percent_rounds_up(folder) -> None:
    """--worst-percent 15 на десяти файлах даёт две строки (округление вверх)."""
    assert "Показано 2 из 10 файлов" in overall_part(run(str(folder), "--worst-percent", "15").output)


def test_percent_and_count_are_mutually_exclusive(folder) -> None:
    """Одновременно заданные --worst-percent и --worst-count — ошибка."""
    result = CliRunner().invoke(main, [str(folder), "--worst-percent", "5", "--worst-count", "3"])
    assert result.exit_code != 0
    assert "взаимоисключающи" in result.output


@pytest.mark.parametrize("algorithm", ["edge_width", "reblur", "hf_mid", "moire", "laplacian", "combo"])
def test_every_algorithm_runs(folder, algorithm: str) -> None:
    """Каждый алгоритм из справки должен отработать и найти размытые кадры."""
    output = overall_part(run(str(folder), "-a", algorithm, "--no-zonal").output)
    lines = [line for line in output.splitlines() if ".png" in line]
    assert all("soft_" in line for line in lines[:2]), (algorithm, lines[:4])


def test_reads_tiff_png_and_jpeg(tmp_path) -> None:
    """Все заявленные растровые форматы читаются, включая 16-битный TIFF."""
    directory = tmp_path / "mixed"
    directory.mkdir()
    page = blur(draw_page(**PAGE), 0.7)
    cv2.imwrite(str(directory / "a.png"), page)
    cv2.imwrite(str(directory / "b.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(directory / "c.tif"), (page.astype(np.uint16) * 257))
    output = overall_part(run(str(directory)).output)
    assert "Показано 3 из 3 файлов" in output
    assert "—" not in output.split("Показано")[0].replace("--", "")


def test_writes_markdown_and_csv(folder, tmp_path) -> None:
    """Отчёты сохраняются: md — с отобранными строками, csv — со всеми файлами."""
    md = tmp_path / "out" / "report.md"
    csv_path = tmp_path / "out" / "scores.csv"
    run(str(folder), "--worst-count", "3", "--md-report", str(md), "--csv", str(csv_path))

    overall_md = md.read_text(encoding="utf-8").split("## 2. Зональный расфокус")[0]
    assert overall_md.count("| 1 |") == 1
    assert "soft_00.png" in overall_md
    # В md попадают только отобранные строки, в csv — вся папка.
    assert len([line for line in overall_md.splitlines() if line.startswith("| ")]) == 3 + 1
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) == 10 + 1


def test_unreadable_file_is_reported_not_crashed(folder) -> None:
    """Битый файл не роняет прогон и попадает в начало таблицы с пометкой."""
    (folder / "broken.png").write_bytes(b"not an image")
    output = overall_part(run(str(folder)).output)
    assert "broken.png" in output
    assert "не прочитан" in output


def test_empty_folder_is_a_clean_error(tmp_path) -> None:
    """Папка без изображений — понятная ошибка, а не трассировка."""
    result = CliRunner().invoke(main, [str(tmp_path)])
    assert result.exit_code != 0
    assert "не найдено поддерживаемых изображений" in result.output


def test_prints_both_reports(folder) -> None:
    """По умолчанию печатаются оба отчёта: общий и зональный."""
    output = run(str(folder)).output
    assert "1. ОБЩЕЕ КАЧЕСТВО ФОКУСА" in output
    assert "2. ЗОНАЛЬНЫЙ РАСФОКУС" in output


def test_zonal_report_can_be_switched_off(folder) -> None:
    """--no-zonal убирает второй отчёт целиком."""
    output = run(str(folder), "--no-zonal").output
    assert "ЗОНАЛЬНЫЙ" not in output


def test_zonal_selection_is_independent_of_the_main_one(folder) -> None:
    """Отбор во втором отчёте задаётся своими опциями и не влияет на первый."""
    output = run(str(folder), "--worst-count", "2", "--zonal-count", "4").output
    overall, zonal = output.split("2. ЗОНАЛЬНЫЙ")
    assert overall.count(".png") == 2
    assert zonal.count(".png") == 4


def test_zonal_percent_rounds_up(folder) -> None:
    """--zonal-percent считается от числа файлов с зональной оценкой."""
    assert "Показано 2 из" in run(str(folder), "--zonal-percent", "15").output.split("2. ЗОНАЛЬНЫЙ")[1]


def test_zonal_options_are_mutually_exclusive(folder) -> None:
    """--zonal-percent и --zonal-count вместе — ошибка."""
    result = CliRunner().invoke(main, [str(folder), "--zonal-percent", "5", "--zonal-count", "3"])
    assert result.exit_code != 0
    assert "взаимоисключающи" in result.output


def test_zonal_selection_conflicts_with_no_zonal(folder) -> None:
    """Просить отбор во втором отчёте и одновременно его выключать бессмысленно."""
    result = CliRunner().invoke(main, [str(folder), "--no-zonal", "--zonal-count", "3"])
    assert result.exit_code != 0
    assert "несовместим" in result.output


def test_zonal_defocus_surfaces_in_the_second_report_only(tmp_path) -> None:
    """Кадр с мягкой третью должен быть виден во втором отчёте, а не в первом.

    Это и есть смысл разделения: у такого кадра резкая часть вытягивает общий балл.
    """
    directory = tmp_path / "zonal"
    directory.mkdir()
    for i in range(6):
        cv2.imwrite(str(directory / f"flat_{i:02d}.png"), blur(draw_page(seed=i, **PAGE), 0.9))
    tilted = blur(draw_page(seed=99, **PAGE), 0.7)
    tilted[1000:, :] = cv2.GaussianBlur(tilted[1000:, :], (0, 0), 1.6)
    cv2.imwrite(str(directory / "tilted.png"), tilted)

    output = run(str(directory)).output
    overall, zonal = output.split("2. ЗОНАЛЬНЫЙ")
    assert zonal.splitlines()[3].strip().endswith("tilted.png"), zonal
    assert not overall.splitlines()[3].strip().endswith("tilted.png"), overall


def test_csv_carries_zonal_columns(folder, tmp_path) -> None:
    """В CSV попадают и зональные колонки — они нужны для калибровки."""
    csv_path = tmp_path / "scores.csv"
    run(str(folder), "--csv", str(csv_path))
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "zonal_drop" in header and "zonal_soft_band" in header


def test_markdown_has_both_tables(folder, tmp_path) -> None:
    """Md-отчёт содержит обе таблицы с кликабельными ссылками."""
    md = tmp_path / "r.md"
    run(str(folder), "--worst-count", "3", "--zonal-count", "2", "--md-report", str(md))
    text = md.read_text(encoding="utf-8")
    assert "## 1. Общее качество фокуса" in text
    assert "## 2. Зональный расфокус" in text
    overall, zonal = text.split("## 2. Зональный расфокус")
    assert overall.count("](") == 3
    assert zonal.count("](") == 2
