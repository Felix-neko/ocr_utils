import fitz
import pytest

from research.geometry_regression.render import pair_pdfs
from research.geometry_regression.scoring import Thresholds


def _pdf(path, pages: int) -> None:
    document = fitz.open()
    for _ in range(pages):
        document.new_page()
    document.save(path)


def test_pair_pdfs_skips_mismatched_page_counts(tmp_path):
    geo, nogeo = tmp_path / "geo", tmp_path / "nogeo"
    geo.mkdir()
    nogeo.mkdir()
    _pdf(geo / "full_1966_01.pdf", 3)
    _pdf(nogeo / "full_1966_01.pdf", 3)
    _pdf(geo / "full_1966_02.pdf", 3)
    _pdf(nogeo / "full_1966_02.pdf", 4)
    _pdf(geo / "full_1966_03.pdf", 2)
    pairs, notes = pair_pdfs(geo, nogeo)
    assert [p.name for p in pairs] == ["full_1966_01"]
    assert pairs[0].pages == 3 and pairs[0].year == "1966"
    assert any("full_1966_02" in n for n in notes) and any("full_1966_03" in n for n in notes)


def test_thresholds_parse_and_score():
    thresholds = Thresholds.parse(("line_wobble_delta_max=0.1",))
    verdict = thresholds.apply({"line_wobble_delta_max": 0.25, "edge_tilt_max_delta": 0.25})
    assert verdict.flag and verdict.reason == "wobble" and verdict.score == pytest.approx(2.5)
    assert set(verdict.flags) == {"line_wobble_delta_max"}
    with pytest.raises(KeyError):
        Thresholds({"нет_такой": 1.0})
