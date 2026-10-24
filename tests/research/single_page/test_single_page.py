"""Тесты стенда «в кадре одна страница»: признаки трапеций, правило, разметка, отбор краёв выпуска, сверка."""

from __future__ import annotations

import numpy as np
import pytest

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from research.single_page.cli import evaluation_lines, frames_at_issue_edges, issue_of
from research.single_page.features import bbox_mm, print_at_spine, quad_height_mm, quad_width_mm, spine_x
from research.single_page.labels import Label, read_labels
from research.single_page.rule import RuleParams, Verdict, typical_widths, verdict_of


def rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Прямоугольная «трапеция» по часовой стрелке."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)


def test_quad_sizes_in_mm():
    """Ширина и высота трапеции переводятся в мм по плотности сегментации."""
    q = rect(0, 0, 69, 138)
    assert quad_width_mm(q) == pytest.approx(69 / SEG_PX_PER_MM)
    assert quad_height_mm(q) == pytest.approx(138 / SEG_PX_PER_MM)


def test_spine_x_between_pages():
    """Линия разреза — посередине между правой стороной левой трапеции и левой стороной правой."""
    assert spine_x(rect(0, 0, 100, 50), rect(110, 0, 200, 50)) == pytest.approx(105)


def test_bbox_mm_empty_and_full():
    """Рамка пустой маски — нули; непустой — размер в мм и крайние столбцы."""
    assert bbox_mm(np.zeros((10, 10), bool)) == (0.0, 0.0, 0, 0)
    m = np.zeros((100, 100), bool)
    m[10:79, 20:89] = True
    w, h, x0, x1 = bbox_mm(m)
    assert (w, h) == (pytest.approx(69 / SEG_PX_PER_MM), pytest.approx(69 / SEG_PX_PER_MM))
    assert (x0, x1) == (20, 88)


def test_print_at_spine_gap_and_solid():
    """У разворота на линии разреза провал печати (0), у сплошной печати — 1."""
    paper = np.ones((50, 200), bool)
    hulls = np.ones((50, 200), bool)
    assert print_at_spine(hulls, paper, 100) == pytest.approx(1.0)
    hulls[:, 90:111] = False
    assert print_at_spine(hulls, paper, 100) == pytest.approx(0.0)


def row(**kw) -> dict:
    """Строка признаков обычного разворота с заменой полей ``kw``."""
    base = dict(dataset="d", key="k", n_pages=2, n_inst=2, paper_h_mm=260.0, paper_aspect=1.2, w_left_mm=160.0,
                w_right_mm=158.0, width_ratio=0.99)  # fmt: skip
    return base | kw


@pytest.mark.parametrize(
    "kw, expected",
    [
        ({}, Verdict.SPREAD),
        # Поднятая страница: самый узкий разворот наборов.
        (dict(width_ratio=0.714, paper_aspect=0.807, w_right_mm=110.0), Verdict.SPREAD),
        # Крышка подшивки, разрезанная по провалу печати (ПХ 1975/10 IMG_0085).
        (dict(n_inst=1, width_ratio=0.10, paper_aspect=0.58, w_right_mm=15.0), Verdict.SINGLE),
        # Страница с полоской соседней у настоящего корешка (МТС 1991/04 «по страницам»).
        (dict(width_ratio=0.14, paper_aspect=0.66, w_right_mm=20.0), Verdict.SLIVER),
        # Одна трапеция.
        (dict(n_pages=1, n_inst=1, w_right_mm=0.0, width_ratio=1.0, paper_aspect=0.6), Verdict.SINGLE),
        # Ярлык на закрытой крышке, разрезанный на две «страницы».
        (dict(n_inst=1, paper_h_mm=54.0, paper_aspect=1.38, width_ratio=0.28), Verdict.NO_PAGE),
        # Бумаги нет.
        (dict(n_pages=0, n_inst=0, w_left_mm=0.0, w_right_mm=0.0, paper_h_mm=0.0), Verdict.NO_PAGE),
        # Узкий кусок, но рамка бумаги широкая — не одна страница (оба условия обязательны).
        (dict(width_ratio=0.3, paper_aspect=1.1), Verdict.SPREAD),
    ],
)
def test_verdict_of(kw, expected):
    """Вердикт правила по поясам признаков."""
    assert verdict_of(row(**kw), 160.0, RuleParams()) is expected


def test_typical_widths_uses_even_spreads():
    """Типичная ширина — по ровным разворотам набора; кадры с узкой трапецией не в счёт."""
    rows = [row(w_left_mm=160.0, w_right_mm=162.0), row(w_left_mm=150.0, w_right_mm=10.0, width_ratio=0.07)]
    assert typical_widths(rows, RuleParams())["d"] == pytest.approx(161.0)


def test_read_labels(tmp_path):
    """Разметка: комментарии пропускаются, метка — первое слово после «#», ключи с пробелами целы."""
    f = tmp_path / "l.txt"
    f.write_text("# заголовок\n\nhp/01 оплачено/IMG_1 # no_page — ярлык\nph/IMG_2 # sliver\n")
    assert read_labels(f) == {"hp/01 оплачено/IMG_1": Label.NO_PAGE, "ph/IMG_2": Label.SLIVER}
    f.write_text("ph/IMG_3\n")
    with pytest.raises(ValueError):
        read_labels(f)


def test_issue_of_and_edges():
    """Выпуск — ключ без последней части; с каждого края выпуска берётся по N кадров."""
    assert issue_of("1975/01__IMG_0001") == "1975/01"
    assert issue_of("МТС__1990__01__IMG_0828") == "МТС/1990/01"
    rows = [dict(dataset="d", key=f"01__IMG_{i:04d}") for i in range(10)]
    picked = [r["key"][-4:] for r in frames_at_issue_edges(rows, 2)]
    assert picked == ["0000", "0001", "0008", "0009"]


def test_evaluation_lines_counts_unlabeled_as_spread():
    """Неразмеченный кадр — разворот; промах выводится строкой «✗»."""
    rows = [dict(dataset="d", key="a", verdict="single"), dict(dataset="d", key="b", verdict="single")]
    lines = evaluation_lines(rows, {"d/a": Label.SINGLE})
    assert lines[-1].startswith("  ✗ d/b")
    assert any(line.startswith("промахов: 1") for line in lines)
