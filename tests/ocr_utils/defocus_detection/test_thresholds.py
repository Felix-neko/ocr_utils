"""Проверки абсолютных порогов: присвоение тегов, разбор опции, поведение CLI."""

import cv2
import pytest
from click.testing import CliRunner
from tests.ocr_utils.defocus_detection.pages import blur, draw_page

from ocr_utils.defocus_detection.cli import main
from ocr_utils.defocus_detection.thresholds import HEAVY, LIGHT, MEDIUM, PRESETS, Preset, parse_thresholds

PAGE = dict(height=1536, width=1536)


@pytest.fixture
def preset():
    """Набор порогов с круглыми числами — чтобы проверять границы, а не калибровку."""
    return Preset(
        name="проверочный",
        algorithm="dom",
        aggregation="best",
        quantile=0.8,
        basis="raw",
        heavy=1.0,
        medium=2.0,
        light=3.0,
        source="тест",
    )


@pytest.fixture
def folder(tmp_path):
    """Папка из восьми резких страниц и двух заметно размытых."""
    directory = tmp_path / "scans"
    directory.mkdir()
    for i in range(8):
        cv2.imwrite(str(directory / f"sharp_{i:02d}.png"), blur(draw_page(seed=i, **PAGE), 0.7))
    for i in range(2):
        cv2.imwrite(str(directory / f"soft_{i:02d}.png"), blur(draw_page(seed=100 + i, **PAGE), 1.8))
    return directory


def test_tag_boundaries_are_strictly_below(preset):
    """Порог означает «строго ниже»: балл, равный порогу, к этому уровню не относится."""
    assert preset.tag(0.5) == HEAVY
    assert preset.tag(1.0) == MEDIUM
    assert preset.tag(1.5) == MEDIUM
    assert preset.tag(2.0) == LIGHT
    assert preset.tag(2.999) == LIGHT
    assert preset.tag(3.0) == ""


def test_unmeasured_frame_is_not_tagged(preset):
    """NaN — это «не измерено», а не «всё хорошо»: тега такой кадр не получает."""
    assert preset.tag(float("nan")) == ""


def test_preset_rejects_thresholds_out_of_order():
    """Пороги обязаны идти по возрастанию балла, иначе уровень недостижим."""
    with pytest.raises(ValueError, match="по возрастанию"):
        Preset(
            name="кривой",
            algorithm="dom",
            aggregation="best",
            quantile=0.8,
            basis="raw",
            heavy=3.0,
            medium=2.0,
            light=1.0,
            source="тест",
        )


def test_parse_thresholds_requires_all_levels():
    """Частичный набор порогов не принимается: недосказанность тут дороже отказа."""
    assert parse_thresholds("heavy=1,medium=2,light=3") == {"heavy": 1.0, "medium": 2.0, "light": 3.0}
    with pytest.raises(ValueError, match="не заданы уровни"):
        parse_thresholds("heavy=1,light=3")
    with pytest.raises(ValueError, match="неизвестный уровень"):
        parse_thresholds("heavy=1,medium=2,light=3,extra=4")


def test_shipped_presets_are_consistent():
    """Каждый готовый пресет ссылается на существующий алгоритм и несёт происхождение."""
    from ocr_utils.defocus_detection.metrics import ALGORITHMS
    from ocr_utils.defocus_detection.scoring import AGGREGATIONS

    for preset in PRESETS.values():
        assert preset.algorithm in ALGORITHMS
        assert preset.aggregation in AGGREGATIONS
        assert preset.source


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


def test_threshold_report_is_independent_of_worst_percent(folder):
    """Порог и доля худших — два разных списка, и печатаются оба.

    Доля выбирает пять кадров из десяти независимо от их качества, порог — только те два,
    что действительно размыты. Ровно в этом расхождении и смысл всей затеи.
    """
    output = run(
        str(folder),
        "--algorithm",
        "dom",
        "--aggregate",
        "best",
        "--tag-thresholds",
        "heavy=2.0,medium=2.25,light=3.0",
        "--worst-percent",
        "50",
    ).output
    assert "3. ПРЕВЫСИЛИ АБСОЛЮТНЫЙ ПОРОГ" in output
    threshold_part = output.split("3. ПРЕВЫСИЛИ")[1]
    assert threshold_part.count("soft_") == 2
    assert "sharp_" not in threshold_part


def test_tagged_only_replaces_the_percentage_report(folder):
    """С --tagged-only первый отчёт сам строится по порогам, и третий не дублируется."""
    output = run(
        str(folder),
        "--algorithm",
        "dom",
        "--aggregate",
        "best",
        "--tag-thresholds",
        "heavy=2.0,medium=2.25,light=3.0",
        "--tagged-only",
    ).output
    assert "3. ПРЕВЫСИЛИ" not in output
    assert output.split("2. ЗОНАЛЬНЫЙ")[0].count("soft_") == 2


def test_preset_settings_conflict_is_an_error(folder):
    """Пресет на чужой агрегации — отказ: порог на другой шкале ничего не значит."""
    result = CliRunner().invoke(main, [str(folder), "--tag-preset", "dom-si", "--aggregate", "worst", "--quiet"])
    assert result.exit_code != 0
    assert "откалиброван" in result.output


def test_retake_list_is_written_even_when_empty(folder, tmp_path):
    """Пустой файл значит «переснимать нечего», отсутствие файла — «прогон не доехал»."""
    path = tmp_path / "retake.txt"
    run(
        str(folder),
        "--algorithm",
        "dom",
        "--aggregate",
        "best",
        "--tag-thresholds",
        "heavy=0.5,medium=0.8,light=1.0",
        "--retake-list",
        str(path),
    )
    assert path.exists() and path.read_text(encoding="utf-8") == ""
