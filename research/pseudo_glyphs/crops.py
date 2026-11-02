"""Вырезки холста Кунса страниц (PNG + паспорт JSON) и их разбор, как во втором преобразовании ``scan_cropping.v2``: серый → 300 dpi → утолщение штрихов → гашение линеек → ``analyse_gray``."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks import alignment as alignment_module
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.hints import LayoutHints, masked_ink
from ocr_utils.page_layout.text_blocks.page import PageAnalysis, analyse_gray
from ocr_utils.scan_cropping.v2.rectify.text_field import rule_ink, thicken_strokes

# Корень вырезок ХиП 1991 (выпуски 01–07): ``<выпуск>/<кадр>_<L|R>.png`` + ``.json``.
HP1991_CANVAS = Path("/mnt/hotstore/scan_processing/line_art_boxes/hp_1991/canvas")
# Корень выхода стенда.
OUT_ROOT = Path("/mnt/hotstore/scan_processing/pseudo_glyphs")


@dataclass(frozen=True)
class PageCrop:
    """Паспорт вырезки страницы холста (копия ``research/line_art_boxes/canvas.PageCrop``, пакет не в git).

    Поля:
        part: ``L`` или ``R`` — левая или правая страница разворота.
        box: Вырезка на холсте ``(x0, y0, x1, y1)``.
        dpi: Плотность холста.
        plain: Страница шла в поле (не цветная, не тонированная).
        field: Флаги кадра разрешают поле по строкам.
    """

    part: str
    box: tuple[int, int, int, int]
    dpi: float
    plain: bool
    field: bool


@dataclass(frozen=True)
class PreparedPage:
    """Вырезка, приготовленная к разбору.

    Поля:
        key: Ключ страницы ``<выпуск>/<кадр>_<часть>``.
        gray300: Серый 300 dpi, как его видит ``analyse_gray`` во втором преобразовании (утолщённый, линейки погашены).
        raw300: Тот же серый 300 dpi без утолщения и гашения — для тона краски.
        dpi: Плотность холста (для пересчёта в мм на холсте).
        part: Часть разворота ``L``/``R`` (у ``L`` корешок справа, у ``R`` — слева).
    """

    key: str
    gray300: np.ndarray
    raw300: np.ndarray
    dpi: float
    part: str


def load_crop(path: Path) -> tuple[np.ndarray, PageCrop]:
    """Прочитать вырезку страницы и её паспорт.

    Args:
        path: PNG вырезки (паспорт — тот же путь с ``.json``).

    Returns:
        ``(картинка BGR, паспорт)``.
    """
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["box"] = tuple(meta["box"])
    return cv2.imread(str(path), cv2.IMREAD_COLOR), PageCrop(**meta)


def page_key(path: Path) -> str:
    """Ключ страницы по пути PNG: ``<номер выпуска>/<кадр>_<часть>`` (номер — первое слово папки выпуска).

    Args:
        path: PNG вырезки.

    Returns:
        Ключ, например ``07/IMG_0099_R``.
    """
    return f"{path.parent.name.split()[0]}/{path.stem}"


def list_crops(root: Path = HP1991_CANVAS, plain_only: bool = True) -> list[Path]:
    """Все вырезки под корнем, по порядку выпусков и кадров.

    Args:
        root: Корень вырезок.
        plain_only: Только страницы, которые шли в поле по строкам (``plain`` и ``field``) — их и разбирает v2.

    Returns:
        Пути PNG.
    """
    out = []
    for path in sorted(root.glob("*/*.png")):
        if plain_only:
            meta = json.loads(path.with_suffix(".json").read_text())
            if not (meta.get("plain") and meta.get("field")):
                continue
        out.append(path)
    return out


def prepare(path: Path) -> PreparedPage:
    """Приготовить вырезку к разбору ровно как ``rectify.text_field.analyze_page`` (без областей рисунков).

    Области line art, растра и таблиц в кэше вырезок не хранятся, поэтому преград-рисунков нет; на разбор строк у
    краёв страницы это почти не влияет.

    Args:
        path: PNG вырезки.

    Returns:
        :class:`PreparedPage`.
    """
    bgr, crop = load_crop(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    f = RENDER_DPI / crop.dpi
    raw300 = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    # Утолщение штрихов по вертикали и гашение линеек — как в analyze_page.
    gray300 = masked_ink(thicken_strokes(raw300), ~rule_ink(gray, crop.dpi))
    return PreparedPage(page_key(path), gray300, raw300, crop.dpi, crop.part)


@contextlib.contextmanager
def edge_touch(enabled: bool) -> Iterator[None]:
    """Включить или выключить правило «ряд ближе ``EDGE_TOUCH_MM`` к краю картинки — обрезан» на время разбора.

    Правило (``alignment.EDGE_TOUCH_MM`` = 0.3 мм) пользователь отверг как замену фильтру псевдоглифов; стенд меряет
    разбор и с ним, и без него. Выключение — отрицательный порог: ни один ряд к краю ближе не подходит.

    Args:
        enabled: ``True`` — как в боевом коде, ``False`` — без правила.
    """
    saved = alignment_module.EDGE_TOUCH_MM
    alignment_module.EDGE_TOUCH_MM = saved if enabled else -1.0
    try:
        yield
    finally:
        alignment_module.EDGE_TOUCH_MM = saved


def analyze(gray300: np.ndarray, edge_rule: bool = False) -> PageAnalysis:
    """Разбор серого 300 dpi движком по краске, как во втором преобразовании (без карт CRAFT).

    Args:
        gray300: Серый 300 dpi (:attr:`PreparedPage.gray300` или он же после фильтра).
        edge_rule: Держать ли правило 0.3 мм у края картинки (:func:`edge_touch`).

    Returns:
        Разбор страницы.
    """
    with edge_touch(edge_rule):
        return analyse_gray(gray300, InkEngine(), hints=LayoutHints())


__all__ = ["HP1991_CANVAS", "OUT_ROOT", "PageCrop", "PreparedPage", "analyze", "edge_touch", "list_crops",
           "load_crop", "page_key", "prepare"]  # fmt: skip
