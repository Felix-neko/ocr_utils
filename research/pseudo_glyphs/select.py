"""Выборки кандидатов на разметку по поясам признаков (касание края, отрыв, высота, тон, тонкий штрих) плюс случайные концы рядов; классы разметки."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

SEED = 0


class Label(StrEnum):
    """Класс разметки кандидата глазами (``sets/labels.csv``, поле ``label``)."""

    # Псевдоглифы — то, что фильтр должен убрать.
    SHADOW = "SH"  # тень сгиба у корешка
    COVER = "CV"  # кусок обложки или подложки из-под края листа
    THREAD = "TH"  # нитки прошивки, крючки у края
    PENCIL = "PN"  # карандашная или чернильная пометка на периферии блока
    SPECK = "SP"  # сор, пятно, точка-грязь
    # Настоящие знаки — их терять нельзя.
    GLYPH = "G"  # буква, слово
    HYPHEN = "HY"  # дефис, тире
    BRACKET = "BR"  # скобка, кавычка
    NUMBER = "NU"  # цифра, номер пункта
    PUNCT = "PU"  # точка, запятая, двоеточие
    INITIAL = "IN"  # буквица, крупная буква
    # Ни то ни другое.
    NEIGHBOR = "NB"  # текст соседней страницы, попавший в вырезку
    FIGURE = "FG"  # рисунок, линейка, рамка — не текст, но законная краска
    UNCLEAR = "U"


PSEUDO = {Label.SHADOW, Label.COVER, Label.THREAD, Label.PENCIL, Label.SPECK}
TRUE = {Label.GLYPH, Label.HYPHEN, Label.BRACKET, Label.NUMBER, Label.PUNCT, Label.INITIAL}


class Band(StrEnum):
    """Пояс выборки на разметку."""

    ROW_EDGE = "row_edge"  # конец ряда ближе 1 мм к краю вырезки
    EDGE_ONLY = "edge_only"  # компонента у края вне рядов
    FAR = "far"  # отрыв от ряда больше робастного максимума пробела
    TALL = "tall"  # выше 1.4 p95 букв блока
    GREY = "grey"  # тон светлее 0.25 (от краски к бумаге)
    THIN = "thin"  # вытянутая (h/w ≥ 3) с тонким штрихом ≤ 0.25 мм
    RANDOM = "random"  # случайный конец ряда
    EDGE_DARK = "edge_dark"  # у края (≤ 1 мм), тёмная как краска (тон < 0.15), не черта (ширина ≥ 0.4 мм)
    EDGE_WIDE = "edge_wide"  # у края, шире 2 мм (кусок обложки, буква)
    FAR_BIG = "far_big"  # далеко от ряда и не мелочь (высота ≥ 1 мм)
    GREY_BIG = "grey_big"  # не у края, серая (тон > 0.2) и не мелочь (высота ≥ 1.2 мм)
    FIRED = "fired"  # правила сработали (колонка ``reason``) — оценка точности по паку
    FIRED_ROW = (
        "fired_row"  # правила сработали на крайней компоненте ряда (её закрас меняет ряды) — точность там, где важно
    )
    DOUBT = "doubt"  # правила молчат, но кандидат спорный: у края или далеко и CRAFT < 0.4 (колонка ``craft``)


def band_masks(frame: pd.DataFrame) -> dict[Band, pd.Series]:
    """Маски поясов по таблице кандидатов (пояса пересекаются).

    Args:
        frame: Кандидаты (``candidates.csv``).

    Returns:
        ``пояс → булева маска``.
    """
    row = frame.kind != "edge"
    masks = {
        Band.ROW_EDGE: row & (frame.edge_mm <= 1.0),
        Band.EDGE_ONLY: ~row,
        Band.FAR: row & (frame.gap_rel > 1.0),
        Band.TALL: row & (frame.h_rel_block > 1.4),
        Band.GREY: row & (frame.tone_rel > 0.25),
        Band.THIN: (frame.aspect >= 3.0) & (frame.stroke_mm <= 0.25),
        Band.RANDOM: row,
        Band.EDGE_DARK: (frame.edge_mm <= 1.0) & (frame.tone_rel < 0.15) & (frame.w_mm >= 0.4),
        Band.EDGE_WIDE: (frame.edge_mm <= 1.0) & (frame.w_mm >= 2.0),
        Band.FAR_BIG: row & (frame.gap_rel > 1.0) & (frame.h_mm >= 1.0),
        Band.GREY_BIG: (frame.edge_mm > 1.0) & (frame.tone_rel > 0.2) & (frame.h_mm >= 1.2),
    }
    if "reason" in frame:
        masks[Band.FIRED] = frame.reason != ""
        masks[Band.FIRED_ROW] = (frame.reason != "") & (frame.kind != "edge")
        if "craft" in frame:
            far = (frame.gap_rel.fillna(0.0) > 1.0) & (frame.gap_mm.fillna(0.0) >= 2.0)
            masks[Band.DOUBT] = (frame.reason == "") & ((frame.edge_mm <= 4.0) | far) & (frame.craft < 0.4)
    return masks


def sample_bands(frame: pd.DataFrame, per_band: dict[Band, int], exclude: set[str] = frozenset()) -> pd.DataFrame:
    """Выборка на разметку: по ``per_band[пояс]`` кандидатов из каждого пояса, без повторов компонент.

    Args:
        frame: Кандидаты (``cid`` уникален; одна компонента может быть кандидатом нескольких рядов).
        per_band: Сколько брать из пояса.
        exclude: Уже размеченные ``cid``.

    Returns:
        Кандидаты с колонкой ``band``, перемешанные внутри пояса.
    """
    rng = np.random.default_rng(SEED)
    unique = frame.drop_duplicates(["key", "comp"])
    unique = unique[~unique.cid.isin(exclude)]
    masks = band_masks(unique)
    taken: set[str] = set()
    parts = []
    for band, count in per_band.items():
        pool = unique[masks[band] & ~unique.cid.isin(taken)]
        if band in (Band.FIRED, Band.FIRED_ROW) and len(pool) > count:
            # Сработавшие — поровну по причинам: иначе выборку забивает самая частая (черта у края).
            groups = list(pool.groupby("reason"))
            share = max(1, count // len(groups))
            parts_band = [g.iloc[rng.choice(len(g), min(share, len(g)), replace=False)] for _, g in groups]
            picked = pd.concat(parts_band)
            rest = pool[~pool.cid.isin(picked.cid)]
            extra = count - len(picked)
            if extra > 0 and len(rest):
                picked = pd.concat([picked, rest.iloc[rng.choice(len(rest), min(extra, len(rest)), replace=False)]])
            pool = picked
        elif len(pool) > count:
            pool = pool.iloc[rng.choice(len(pool), count, replace=False)]
        pool = pool.assign(band=band.value)
        taken.update(pool.cid)
        parts.append(pool)
    return pd.concat(parts)


__all__ = ["PSEUDO", "TRUE", "Band", "Label", "band_masks", "sample_bands"]
