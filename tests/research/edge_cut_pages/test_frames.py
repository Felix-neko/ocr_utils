"""Тесты обхода пака и отбора кадров: скрытые папки, схема журнал/год/выпуск, решение по строке сводки."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.edge_cut_pages.cli import is_positive
from research.edge_cut_pages.frames import iter_frames


def test_iter_frames_skips_hidden(tmp_path: Path) -> None:
    """Кадры из скрытых папок и вне схемы журнал/год/выпуск не попадают в обход."""
    for rel in (
        "РИСК/1991/05 x/IMG_1.jpg",
        "РИСК/1991/05 x/.covers/IMG_2.jpg",
        "РИСК/1991/IMG_3.jpg",
        "МТС/1990/01/IMG_4.JPG",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    frames = iter_frames(tmp_path)
    assert [str(frame.rel) for frame in frames] == ["МТС/1990/01/IMG_4.JPG", "РИСК/1991/05 x/IMG_1.jpg"]
    assert frames[1].issue == "05 x" and frames[1].name == "IMG_1"


def test_is_positive() -> None:
    """Отбор: срез вдоль с достаточной мерой или наружный край; обложка и только корешок — нет."""
    base = {"is_cover": False, "n_bottom": 0, "n_top": 0, "n_outer": 0, "n_spine": 0, "score": 0}
    assert not is_positive(pd.Series(base), min_score=3, min_outer=1)
    assert is_positive(pd.Series({**base, "n_bottom": 1, "score": 5}), min_score=3, min_outer=1)
    assert not is_positive(pd.Series({**base, "n_bottom": 1, "score": 2}), min_score=3, min_outer=1)
    assert is_positive(pd.Series({**base, "n_outer": 1}), min_score=3, min_outer=1)
    assert not is_positive(pd.Series({**base, "n_spine": 9}), min_score=3, min_outer=1)
    assert not is_positive(pd.Series({**base, "n_bottom": 2, "score": 30, "is_cover": True}), min_score=3, min_outer=1)
