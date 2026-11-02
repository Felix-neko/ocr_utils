"""Тесты стенда псевдоглифов: правила по признакам, разбор разметки, кандидаты и фильтр на синтетической странице с линией сгиба у края."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from research.pseudo_glyphs.candidates import CandidateKind, EdgeSide, components_of, page_candidates, stroke_mm
from research.pseudo_glyphs.cli import parse_raw_labels
from research.pseudo_glyphs.crops import analyze
from research.pseudo_glyphs.prototype import FilterMode, filter_page
from research.pseudo_glyphs.rules import PseudoReason, RuleParams, reason_of, reasons


def _features(**over) -> dict:
    """Признаки обычной буквы в середине ряда; ``over`` подменяет нужные."""
    base = dict(kind="row_left", edge_mm=10.0, stroke_mm=0.45, w_mm=1.3, h_mm=1.7, aspect=1.3, tone_rel=0.0,
                h_rel_block=0.8, gap_mm=0.3, gap_rel=0.1)  # fmt: skip
    base.update(over)
    return base


def test_letter_is_kept():
    """Обычная буква у края ряда, но далеко от края картинки, не трогается."""
    assert reason_of(_features()) == ""


def test_far_tall_is_not_a_reason():
    """Высокое далеко от ряда — не причина (на разметке 0 из 3 сработавших — рисунки и буквицы)."""
    assert reason_of(_features(gap_mm=6.0, gap_rel=2.0, h_rel_block=2.0, h_mm=4.0)) == ""


def test_cut_letter_at_edge_is_kept():
    """Буква, срезанная краем вырезки (краска как у текста, штрих обычный, высота буквы), не трогается."""
    assert reason_of(_features(edge_mm=0.0, w_mm=0.5, aspect=3.4)) == ""


def test_fold_line_at_edge():
    """Тонкая высокая черта у края — линия сгиба или нитка."""
    assert reason_of(_features(edge_mm=0.1, stroke_mm=0.08, w_mm=0.1, h_mm=2.5, aspect=25.0)) == PseudoReason.EDGE_LINE


def test_grey_at_edge():
    """Серая компонента у края — тень сгиба или обложка."""
    assert reason_of(_features(edge_mm=0.2, tone_rel=0.4)) == PseudoReason.EDGE_GREY


def test_far_grey_pencil():
    """Серая компонента, оторванная от ряда дальше максимума пробела, — карандаш."""
    assert reason_of(_features(gap_mm=6.0, gap_rel=2.0, tone_rel=0.45)) == PseudoReason.FAR_GREY


def test_far_black_letter_is_kept():
    """Чёрная буква обычной высоты за большим пробелом (номер страницы, слово после разгона) не трогается."""
    assert reason_of(_features(gap_mm=6.0, gap_rel=2.0)) == ""


def test_craft_vote_only_for_doubtful():
    """CRAFT голосует только у спорных (у края или далеко), серых и не брусков, и только если правила промолчали."""
    frame = pd.DataFrame(
        [
            _features(edge_mm=3.0, tone_rel=0.22, craft=0.1),  # серая у края — CRAFT решает
            _features(edge_mm=3.0, tone_rel=0.0, craft=0.1),  # тёмная (срезанная буква) — CRAFT молчит
            _features(edge_mm=3.0, tone_rel=0.22, w_mm=1.2, h_mm=0.4, craft=0.1),  # брусок-тире — CRAFT молчит
            _features(tone_rel=0.22, craft=0.1),  # в середине строки — не спорная
        ]
    )
    out = reasons(frame, RuleParams(craft_doubt=0.3))
    assert list(out) == [PseudoReason.CRAFT.value, "", "", ""]


def test_parse_raw_labels_default_and_explicit():
    """Класс по умолчанию листа и явные метки с заметкой."""
    raw = parse_raw_labels("# раунд\n1 default G\n3 SH тень\n5 HY\n")
    assert raw[-1] == ("G", "")
    assert raw[3] == ("SH", "тень")
    assert raw[5] == ("HY", "")


def test_stroke_of_line():
    """Толщина штриха прямоугольника 3 × 60 px при 300 dpi — около 3 px (0.25 мм)."""
    mask = np.zeros((70, 10), bool)
    mask[5:65, 3:6] = True
    assert stroke_mm(mask) == pytest.approx(0.254, abs=0.05)


def _synthetic_page() -> tuple[np.ndarray, np.ndarray]:
    """Синтетическая страница 300 dpi: колонка строк текста и тонкая серая линия сгиба у левого края, касающаяся строк.

    Returns:
        ``(серый для разбора, серый без утолщения)`` — здесь одинаковые.
    """
    gray = np.full((900, 1100), 255, np.uint8)
    words = "the quick brown fox jumps over lazy dog"
    for i in range(14):
        y = 80 + i * 50
        cv2.putText(gray, words, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3, cv2.LINE_AA)
    # Линия сгиба: серая (тон 100 при краске 0 — темнее порога Оцу, но серее букв), толщиной 2 px, у самого левого края, во всю высоту строк.
    gray[40:760, 0:2] = 100
    # Крючки линии, заходящие к строкам: короткие отростки той же серости на уровне двух строк.
    gray[255:262, 0:30] = 100
    gray[505:512, 0:30] = 100
    return gray, gray.copy()


def test_candidates_and_filter_on_synthetic_page():
    """Линия сгиба у края находится кандидатом края с касанием и снимается фильтром; буквы остаются."""
    gray, raw = _synthetic_page()
    analysis = analyze(gray)
    frame = pd.DataFrame(page_candidates(analysis, gray, raw, EdgeSide.LEFT))
    assert not frame.empty
    at_edge = frame[frame.edge_mm <= 0.2]
    assert len(at_edge) >= 1
    assert at_edge.spine_side.all()
    result = filter_page(gray, raw, EdgeSide.LEFT, FilterMode.RULES, first=analysis)
    assert result.drops, "линия сгиба должна быть закрашена"
    assert all(d.box[0] <= 2 for d in result.drops), "закрашено только у края"
    # Буквы не тронуты: краска правее 40 px осталась той же.
    assert np.array_equal(result.gray300[:, 40:], gray[:, 40:])
    assert len(components_of(result.gray300).stats) < len(components_of(gray).stats)


def test_kinds_enum_values():
    """Значения видов кандидатов — строки CSV."""
    assert {k.value for k in CandidateKind} == {"row_left", "row_right", "edge"}
