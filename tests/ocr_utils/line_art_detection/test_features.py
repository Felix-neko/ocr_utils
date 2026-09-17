"""Проверки правил детектора на синтетике — без обращения к паку."""

import cv2
import numpy as np
import pytest

from ocr_utils.line_art_detection.features import analyse_gray, params_for_dpi, rule_clusters

PAGE = (6733, 4165)  # страница пака-1 при 600 dpi


@pytest.fixture
def params():
    return params_for_dpi(600)


def blank():
    """Чистая белая полоса."""
    return np.full(PAGE, 255, np.uint8)


def put_text_block(page, x=400, y=400, columns=30, rows=40):
    """Сетка прямоугольников размером с букву — имитация набора."""
    for row in range(rows):
        for col in range(columns):
            left, top = x + col * 45, y + row * 89
            cv2.rectangle(page, (left, top), (left + 28, top + 60), 0, -1)
    return page


def test_чистый_текст_ничего_не_находит(params):
    findings = analyse_gray(put_text_block(blank()), params)
    assert findings.coverage == 0.0
    assert findings.candidates == []


def test_штриховой_рисунок_находится(params):
    page = put_text_block(blank(), rows=10)
    rng = np.random.default_rng(0)
    points = rng.integers([600, 2000], [3400, 5600], size=(60, 2))
    cv2.polylines(page, [points.reshape(-1, 1, 2).astype(np.int32)], False, 0, 7)
    findings = analyse_gray(page, params)
    assert findings.coverage > 0.05
    assert any(c.source == "ink" for c in findings.candidates)


def test_сплошная_клякса_отбрасывается_по_заполнению(params):
    """Залитый прямоугольник не проходит даже ворота: заполнение рамки у него единица."""
    page = blank()
    cv2.rectangle(page, (0, 1500), (700, 4000), 0, -1)
    findings = analyse_gray(page, params)
    assert findings.coverage == 0.0
    assert findings.candidates == []


def test_рваный_потёк_у_края_отбрасывается(params):
    """Потёк вдоль края скана рыхлее сплошной заливки и до ворот по заполнению доходит.

    Снимает его отдельное правило ``border_fill`` (приём ``pixFindPageForeground``).
    """
    page = blank()
    cv2.rectangle(page, (0, 1500), (700, 4000), 0, -1)
    rng = np.random.default_rng(7)
    holes = rng.integers([0, 1500], [700, 4000], size=(4000, 2))
    for x, y in holes:  # выедаем краску до заполнения около 0.55
        cv2.circle(page, (int(x), int(y)), 9, 255, -1)
    findings = analyse_gray(page, params)
    assert findings.coverage == 0.0
    assert "клякса у края" in findings.dropped


def test_типографские_линейки_поодиночке_не_находка(params):
    """Горизонтальные линейки без вертикальных — это оглавление, а не таблица."""
    page = blank()
    for row in range(12):
        cv2.rectangle(page, (600, 1000 + row * 200), (3400, 1016 + row * 200), 0, -1)
    findings = analyse_gray(page, params)
    assert findings.coverage == 0.0


def ruled_table(page):
    """Сетка таблицы, линейки которой ДРУГ ДРУГА НЕ КАСАЮТСЯ.

    Так эта сетка и напечатана в журнале: на эталонной стр. 47 крупная таблица не даёт
    ни одного связного пятна, проходящего ворота по размеру. Поэтому в местах пересечений
    краска стирается — иначе синтетика оказалась бы связнее настоящей печати и проверяла
    бы не тот путь.
    """
    for row in range(10):
        cv2.rectangle(page, (600, 1000 + row * 300), (3400, 1010 + row * 300), 0, -1)
    for col in range(5):
        cv2.rectangle(page, (600 + col * 700, 1000), (610 + col * 700, 3700), 0, -1)
    for row in range(10):
        for col in range(5):
            x, y = 600 + col * 700, 1000 + row * 300
            cv2.rectangle(page, (x - 12, y - 12), (x + 22, y + 22), 255, -1)
    return page


def test_разлинованная_таблица_находится_по_скоплению_линеек(params):
    findings = analyse_gray(ruled_table(blank()), params)
    assert findings.coverage > 0.05
    assert any(c.source == "rules" for c in findings.candidates)


def test_скопление_только_горизонталей_таблицей_не_считается(params):
    page = blank()
    for row in range(10):
        cv2.rectangle(page, (600, 1000 + row * 300), (3400, 1010 + row * 300), 0, -1)
    ink = (page < 128).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    assert rule_clusters(stats[1:], params) == []


def test_разметка_вычитает_находку(params):
    """Прямоугольник растра из базы снимает найденное под ним."""
    page = blank()
    rng = np.random.default_rng(1)
    points = rng.integers([600, 2000], [3400, 5600], size=(60, 2))
    cv2.polylines(page, [points.reshape(-1, 1, 2).astype(np.int32)], False, 0, 7)
    assert analyse_gray(page, params).coverage > 0.05
    covered = analyse_gray(page, params, exclude_boxes=[(500, 1900, 3500, 5700)])
    assert covered.coverage == 0.0
    assert "растр по разметке" in covered.dropped


def test_пороги_пересчитываются_под_разрешение():
    at600, at300 = params_for_dpi(600), params_for_dpi(300)
    assert at300.pitch_px == at600.pitch_px // 2
    assert at300.min_area_px == pytest.approx(at600.min_area_px / 4, rel=0.01)
    assert at300.max_aspect == at600.max_aspect  # доли не масштабируются


def test_чёрная_рамка_скана_отбрасывается(params):
    """Неосвещённое поле вокруг полосы бинаризуется в кайму по периметру скана.

    До края СТРАНИЦЫ она не достаёт — промежуточный PDF собран с белыми полями, — поэтому
    правило края её не берёт. Опознаётся она по плотности: сплошная масса переживает
    эрозию диском в два штриха, а сам штрих исчезает целиком.
    """
    page = blank()
    cv2.rectangle(page, (288, 144), (3877, 6589), 0, -1)  # скан целиком
    cv2.rectangle(page, (348, 204), (3817, 6529), 255, -1)  # его светлая середина
    put_text_block(page, x=600, y=600, columns=20, rows=30)
    findings = analyse_gray(page, params)
    assert findings.coverage == 0.0
    assert "сплошная масса (кайма скана, клякса)" in findings.dropped


def test_находка_внутри_чёрной_рамки_не_теряется(params):
    """Кайму выбрасываем, но всё, что она окружает, по-прежнему ищется."""
    page = blank()
    cv2.rectangle(page, (288, 144), (3877, 6589), 0, -1)
    cv2.rectangle(page, (348, 204), (3817, 6529), 255, -1)
    rng = np.random.default_rng(3)
    points = rng.integers([700, 2000], [3400, 5600], size=(60, 2))
    cv2.polylines(page, [points.reshape(-1, 1, 2).astype(np.int32)], False, 0, 7)
    findings = analyse_gray(page, params)
    assert findings.coverage > 0.05
    assert "сплошная масса (кайма скана, клякса)" in findings.dropped


def test_декоративная_рамка_с_одной_вертикалью_не_таблица(params):
    """Полоса выходных данных: рамка вокруг набора и один разделитель колонок.

    Внешне это «две горизонтали и две вертикали», но внутренняя вертикаль всего одна —
    у настоящей таблицы их минимум три (замер в ``TABLE_MIN_RULES``).
    """
    page = blank()
    cv2.rectangle(page, (500, 800), (3600, 5800), 0, 12)  # рамка вокруг набора
    cv2.rectangle(page, (2050, 900), (2062, 5700), 0, -1)  # разделитель колонок
    for row in range(2):  # пара линеек внизу
        cv2.rectangle(page, (600, 5300 + row * 120), (3500, 5312 + row * 120), 0, -1)
    findings = analyse_gray(page, params)
    assert not any(c.source == "rules" for c in findings.candidates)


def test_разделитель_колонок_перебитый_надвое_не_таблица(params):
    """Полоса содержания: одна вертикаль, перебитая горизонталью на два куска.

    По штукам это «две вертикали», но стоят они на одном x. Сетку делает число РАЗНЫХ
    колонок, а не число обрезков (замер в ``TABLE_MIN_RULES``).
    """
    page = blank()
    cv2.rectangle(page, (500, 800), (3600, 5800), 0, 12)
    cv2.rectangle(page, (2050, 900), (2062, 2600), 0, -1)  # верхний кусок разделителя
    cv2.rectangle(page, (2050, 2800), (2062, 5700), 0, -1)  # нижний кусок того же
    for row in range(2):
        cv2.rectangle(page, (600, 2650 + row * 900), (3500, 2662 + row * 900), 0, -1)
    findings = analyse_gray(page, params)
    assert not any(c.source == "rules" for c in findings.candidates)
