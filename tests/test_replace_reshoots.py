"""Тесты подмены пересъёмкой (replace_reshoots.py): имена подложенных кадров и выбор оригинала."""

from pathlib import Path

import numpy as np

from replace_reshoots import best_matches, replacement_name


def test_replacement_name_keeps_original_marks_and_adds_reshoot():
    """Имя = оригинал (с пометками) + снимок пересъёмки + .replacement[.gutter_loss.light]."""
    assert replacement_name(Path("DSCF1023.RAF"), Path("DSCF0015.RAF"), False) == "DSCF1023.DSCF0015.replacement.RAF"
    assert (
        replacement_name(Path("DSCF1023.RAF"), Path("DSCF0015.RAF"), True)
        == "DSCF1023.DSCF0015.replacement.gutter_loss.light.RAF"
    )
    assert (
        replacement_name(Path("DSCF1257_defocus.RAF"), Path("DSCF0002.RAF"), False)
        == "DSCF1257_defocus.DSCF0002.replacement.RAF"
    )


def test_best_matches_takes_top_and_drops_weak():
    """Берётся самый похожий оригинал; при сходстве ниже порога пары нет."""
    reshoots = [Path("a.RAF"), Path("b.RAF")]
    originals = [Path("o1.RAF"), Path("o2.RAF"), Path("o3.RAF")]
    sim = np.array([[0.1, 0.6, 0.2], [0.02, 0.01, 0.03]])
    first, second = best_matches(reshoots, originals, sim, min_sim=0.05)
    assert (first.original, first.runner_up) == (Path("o2.RAF"), Path("o3.RAF"))
    assert np.isclose(first.second, 0.2)
    assert second.original is None
