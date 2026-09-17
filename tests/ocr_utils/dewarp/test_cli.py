"""Сквозной прогон CLI движком textline по крошечному дереву."""

from __future__ import annotations

import csv

import cv2
from click.testing import CliRunner

from ocr_utils.dewarp.cli import main
from tests.ocr_utils.dewarp.synthetic import curl_page


def test_run_textline_writes_every_artefact(tmp_path):
    issue = tmp_path / "pack" / "1967" / "01"
    issue.mkdir(parents=True)
    warped, _ = curl_page()
    cv2.imwrite(str(issue / "IMG_0001_1L.png"), warped)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--root",
            str(tmp_path / "pack"),
            "--only",
            "1967/01/IMG_0001_1L.png",
            "--engines",
            "textline",
            "--out-dir",
            str(out),
            "--source-dpi",
            "300",
            "--jobs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    assert (out / "textline" / "1967_01_IMG_0001_1L.jpg").is_file()
    assert (out / "original" / "1967_01_IMG_0001_1L.jpg").is_file()
    assert (out / "compare" / "textline" / "1967_01_IMG_0001_1L.jpg").is_file()
    rows = list(csv.DictReader((out / "quality.csv").open(encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["ok"] == "1"
    assert float(rows[0]["line_fit.sagitta_rel_p90_after"]) < float(rows[0]["line_fit.sagitta_rel_p90_before"])
    assert "textline" in (out / "quality.md").read_text(encoding="utf-8")
