"""Тесты склейки страниц разворота (``research.bound_volumes.glue``) на синтетике."""

import numpy as np
import pytest

from research.bound_volumes.glue import GlueMethod, glue


def spread(gap: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Две страницы 300×280 со щелью ``gap`` px и книга вокруг с полями 30 px.

    Args:
        gap: Ширина щели, px.

    Returns:
        ``(маска бумаги, маска книги)``.
    """
    paper = np.zeros((400, 700), bool)
    paper[50:350, 50:330] = True
    paper[50:350, 330 + gap : 610 + gap] = True
    book = np.zeros_like(paper)
    book[20:380, 20 : 640 + gap] = True
    return paper, book


@pytest.mark.parametrize("method", list(GlueMethod))
def test_gap_between_pages_glued(method: GlueMethod) -> None:
    """Щель 20 px между страницами закрывается любым способом."""
    paper, book = spread(20)
    out = glue(paper, book, method).paper
    assert out[200, 340]


@pytest.mark.parametrize("method", list(GlueMethod))
def test_outer_edge_not_grown(method: GlueMethod) -> None:
    """Наружу от страниц (на поле книги — обложку) маска не растёт."""
    paper, book = spread(20)
    out = glue(paper, book, method).paper
    assert not out[200, 40]
    assert not out[30, 200]


@pytest.mark.parametrize("method", list(GlueMethod))
def test_single_page_untouched(method: GlueMethod) -> None:
    """Одиночная страница не меняется."""
    paper = np.zeros((400, 400), bool)
    paper[50:350, 50:330] = True
    book = np.zeros_like(paper)
    book[20:380, 20:380] = True
    assert (glue(paper, book, method).paper == paper).all()


@pytest.mark.parametrize("method", list(GlueMethod))
def test_nothing_above_spine_junction(method: GlueMethod) -> None:
    """Над корешком (страницы сходятся вершинами вниз, между ними треугольник «торца») склейка не добавляет ничего."""
    paper = np.zeros((400, 700), bool)
    yy, xx = np.mgrid[0:400, 0:700]
    # Верхние края страниц опускаются к корешку (x=340): треугольная выемка над ним, как у тугой подшивки.
    top = 110 - np.abs(xx - 340) * 0.15
    paper[(yy >= top) & (yy < 350) & (xx >= 60) & (xx < 336)] = True
    paper[(yy >= top) & (yy < 350) & (xx >= 344) & (xx < 620)] = True
    book = np.zeros_like(paper)
    book[20:380, 30:650] = True
    r = glue(paper, book, method)
    assert r.paper[200, 340]  # щель между страницами закрыта
    assert not r.added[(yy < top - 10)].any()  # над верхними краями ничего не добавлено (кроме кончика щели)
