"""Сквозной прогон CLI по крошечному синтетическому дереву."""

from __future__ import annotations

import csv

import cv2
from click.testing import CliRunner

from ocr_utils.scan_markup.curved_lines.cli import main
from tests.ocr_utils.scan_markup.curved_lines.synthetic import bow_page, straight_page


def build_tree(root):
    issue = root / "1967" / "01"
    issue.mkdir(parents=True)
    cv2.imwrite(str(issue / "IMG_0001_1L.png"), straight_page(seed=1))
    cv2.imwrite(str(issue / "IMG_0001_2R.png"), straight_page(seed=2))
    cv2.imwrite(str(issue / "IMG_0002_1L.png"), bow_page())
    cache = issue / "cache"  # служебный каталог ScanTailor — не полоса
    cache.mkdir()
    cv2.imwrite(str(cache / "thumb.png"), straight_page())
    labels = root / "labels.csv"
    labels.write_text("1967/01/IMG_0001_1L,straight\n1967/01/IMG_0002_1L,curved\n", encoding="utf-8")
    return labels


def run(*args):
    result = CliRunner().invoke(main, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


def test_run_flags_the_bow_page_and_writes_every_artefact(tmp_path):
    labels = build_tree(tmp_path / "pack")
    common = [
        "--root",
        str(tmp_path / "pack"),
        "--detectors",
        "skew_map,line_fit,strip_shift",
        "--jobs",
        "1",
        "--source-dpi",
        "300",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--labels",
        str(labels),
    ]
    output = run(
        "run",
        *common,
        "--link-dir",
        str(tmp_path / "links"),
        "--csv",
        str(tmp_path / "out.csv"),
        "--md-report",
        str(tmp_path / "out.md"),
        "--overlay-dir",
        str(tmp_path / "overlay"),
        "--overlay",
        "flagged",
        "--suggest-thresholds",
    )
    assert "Полос: 3" in output, "содержимое cache/ полосой считаться не должно"
    combo = sorted(p.name for p in (tmp_path / "links" / "combo").iterdir())
    assert combo == [
        "1967_01_IMG_0002_1L_p003_s{:.2f}.png".format(_score(tmp_path / "out.csv", "1967/01/IMG_0002_1L.png"))
    ]
    rows = list(csv.DictReader((tmp_path / "out.csv").open(encoding="utf-8")))
    assert len(rows) == 3 and {row["combo"] for row in rows} == {"да", ""}
    assert "Разметка" in (tmp_path / "out.md").read_text(encoding="utf-8")
    assert len(list((tmp_path / "overlay").iterdir())) == 1
    labelled = sorted(p.name for p in (tmp_path / "links" / "разметка").iterdir())
    assert any(name.endswith("_curved_ok.png") for name in labelled) and any(
        name.endswith("_straight_ok.png") for name in labelled
    )

    # Второй прогон берёт всё из кэша: время детекторов нулевое, результат тот же.
    output = run("run", *common, "--csv", str(tmp_path / "again.csv"))
    again = list(csv.DictReader((tmp_path / "again.csv").open(encoding="utf-8")))
    assert [r["combo"] for r in again] == [r["combo"] for r in rows]

    # Пересборка из CSV со строгими порогами опустошает combo без пересчёта.
    strict = [
        "--thr",
        "skew_map.max_dev_deg=100",
        "--thr",
        "skew_map.spread_deg=100",
        "--thr",
        "skew_map.resid_deg=100",
    ]
    strict += [
        "--thr",
        "line_fit.sagitta_rel_p90=100",
        "--thr",
        "line_fit.sagitta_rel_max3=100",
        "--thr",
        "line_fit.slope_spread_deg=100",
    ]
    strict += [
        "--thr",
        "line_fit.slope_resid_deg=100",
        "--thr",
        "strip_shift.resid_max_rel=100",
        "--thr",
        "strip_shift.angle_max_dev_deg=100",
    ]
    run(
        "report",
        "--csv",
        str(tmp_path / "out.csv"),
        "--root",
        str(tmp_path / "pack"),
        "--link-dir",
        str(tmp_path / "links"),
        *strict,
    )
    assert list((tmp_path / "links" / "combo").iterdir()) == []


def _score(csv_path, rel):
    for row in csv.DictReader(csv_path.open(encoding="utf-8")):
        if row["полоса"] == rel:
            return float(row["combo_score"])
    raise AssertionError(rel)


def test_only_and_list_thresholds(tmp_path):
    build_tree(tmp_path / "pack")
    output = run(
        "run",
        "--root",
        str(tmp_path / "pack"),
        "--only",
        "1967/01/IMG_0002_1L.png",
        "--detectors",
        "skew_map",
        "--jobs",
        "1",
        "--source-dpi",
        "300",
    )
    assert "Полос: 1" in output
    output = run("run", "--root", str(tmp_path / "pack"), "--detectors", "skew_map", "--list-thresholds")
    assert "max_dev_deg" in output
