from __future__ import annotations

from pathlib import Path

import pytest

from ocr_utils.scan_markup.curved_lines import report
from ocr_utils.scan_markup.curved_lines.analysis import PageResult
from ocr_utils.scan_markup.curved_lines.detectors.base import Measure
from ocr_utils.page_layout.orientation.pdf_pages import PdfPage


def _result(rel: str, tmp_path: Path, flag: bool, score: float, label: str = "") -> PageResult:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    result = PageResult(rel_path=rel, path=path, label=label)
    result.measures["a"] = Measure(metrics={"m": score}, flag=flag, score=score)
    result.combo = Measure(metrics={"votes": float(flag)}, flag=flag, score=score)
    return result


def test_link_name_carries_year_issue_stem_page_and_score(tmp_path):
    result = _result("1967/01/IMG_0043_2R.jpg", tmp_path, True, 2.345)
    assert (
        report.link_name(result, PdfPage("full_1967_01.pdf", 80), 2.345, None, ".tif")
        == "1967_01_IMG_0043_2R_p080_s2.35.tif"
    )
    assert (
        report.link_name(result, None, 0.5, 7, ".jpg", "_curved_ok") == "1967_01_IMG_0043_2R_p007_s0.50_curved_ok.jpg"
    )


def test_write_link_dir_lays_out_detector_combo_candidates_and_labels(tmp_path):
    results = [
        _result("1967/01/a.png", tmp_path / "src", True, 1.5, "curved"),
        _result("1967/01/b.png", tmp_path / "src", False, 0.3, "straight"),
        _result("1967/02/c.png", tmp_path / "src", False, 0.2, "curved"),
    ]
    counts = report.write_link_dir(tmp_path / "links", results, ["a"], {})
    assert counts == {"a": 1, "combo": 1, report.CANDIDATES_DIR: 1, report.LABELS_DIR: 3}
    names = sorted(p.name for p in (tmp_path / "links" / report.LABELS_DIR).iterdir())
    assert names == [
        "1967_01_a_p001_s1.50_curved_ok.png",
        "1967_01_b_p002_s0.30_straight_ok.png",
        "1967_02_c_p003_s0.20_curved_missed.png",
    ]
    link = tmp_path / "links" / "combo" / "1967_01_a_p001_s1.50.png"
    assert link.is_symlink() and link.resolve() == (tmp_path / "src/1967/01/a.png").resolve()
    # Повторный прогон переписывает свои симлинки, а на чужой файл отказывается.
    report.write_link_dir(tmp_path / "links", results, ["a"], {})
    (tmp_path / "links" / "combo" / "stranger.txt").write_text("x")
    with pytest.raises(report.LinkDirError):
        report.write_link_dir(tmp_path / "links", results, ["a"], {})


def test_csv_roundtrip_keeps_metrics_labels_and_silence(tmp_path):
    results = [
        _result("1967/01/a.png", tmp_path / "src", True, 1.5, "curved"),
        _result("1967/01/b.png", tmp_path / "src", False, 0.3),
    ]
    results[1].measures["a"] = Measure(note="мало строк", silent=True)
    path = tmp_path / "out.csv"
    report.write_csv(path, results, ["a"], {})
    loaded, names = report.read_csv(path, tmp_path / "src")
    assert names == ["a"]
    assert loaded[0].measures["a"].metrics == {"m": 1.5}
    assert loaded[0].label == "curved"
    assert loaded[1].measures["a"].silent and loaded[1].measures["a"].note == "мало строк"
