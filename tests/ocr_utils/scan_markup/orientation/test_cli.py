"""Сквозной прогон CLI по крошечному синтетическому дереву."""

from __future__ import annotations

import csv

import cv2
from click.testing import CliRunner

from ocr_utils.scan_markup.orientation.cli import main
from ocr_utils.scan_markup.orientation.detectors.base import rotate_cw
from tests.ocr_utils.scan_markup.orientation.synthetic import text_page


def build_tree(root):
    """Выпуск из трёх полос: две прямые, одна повёрнутая на 90 по часовой."""
    issue = root / "1967" / "01"
    issue.mkdir(parents=True)
    cv2.imwrite(str(issue / "IMG_0001_1L.png"), text_page())
    cv2.imwrite(str(issue / "IMG_0001_2R.png"), text_page())
    cv2.imwrite(str(issue / "IMG_0002_1L.png"), rotate_cw(text_page(), 90))
    # Служебный каталог ScanTailor: его содержимое полосами не считается.
    cache = issue / "cache"
    cache.mkdir()
    cv2.imwrite(str(cache / "thumb.png"), text_page())
    return issue


def run(*args):
    result = CliRunner().invoke(main, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


def test_run_finds_the_rotated_page_and_writes_every_artefact(tmp_path):
    build_tree(tmp_path / "pack")
    links, csv_path, report = tmp_path / "links", tmp_path / "out.csv", tmp_path / "out.md"
    output = run(
        "run",
        "--root",
        str(tmp_path / "pack"),
        "--detectors",
        "ink_axis",
        "--jobs",
        "1",
        "--source-dpi",
        "300",
        "--link-dir",
        str(links),
        "--csv",
        str(csv_path),
        "--md-report",
        str(report),
    )
    assert "Полос: 3" in output, "содержимое cache/ полосой считаться не должно"

    found = sorted(path.name for path in (links / "combo").iterdir())
    assert found == ["1967_01_IMG_0002_1L_p003_ccw90.png"]

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 3
    assert {row["combo"] for row in rows} == {"прямо", "ccw90"}
    assert "Полосы под поворот" in report.read_text(encoding="utf-8")


def test_only_takes_the_named_pages(tmp_path):
    build_tree(tmp_path / "pack")
    output = run(
        "run",
        "--root",
        str(tmp_path / "pack"),
        "--only",
        "1967/01/IMG_0002_1L.png",
        "--detectors",
        "ink_axis",
        "--jobs",
        "1",
        "--source-dpi",
        "300",
    )
    assert "Полос: 1" in output


def test_unknown_detector_is_refused(tmp_path):
    build_tree(tmp_path / "pack")
    result = CliRunner().invoke(main, ["run", "--root", str(tmp_path / "pack"), "--detectors", "нетакого"])
    assert result.exit_code != 0
    assert "нет детектора" in result.output


def test_validate_recovers_synthetic_rotations(tmp_path):
    build_tree(tmp_path / "pack")
    output = run(
        "validate",
        "--root",
        str(tmp_path / "pack"),
        "--detectors",
        "ink_axis",
        "--jobs",
        "1",
        "--sample",
        "2",
        "--source-dpi",
        "300",
    )
    assert "Проверка детекторов ориентации" in output
    assert "ink_axis" in output
