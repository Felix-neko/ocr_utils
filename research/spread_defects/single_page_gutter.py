"""Уход текста под корешок на постраничной съёмке газетной подшивки: сгиб у края кадра, отступ строк, обрубки слов.

Съёмка ЭГ (Fujifilm, превью RAF 4416×2944): в кадре одна полоса. На кадре, повёрнутом по EXIF (Orientation 8),
корешок слева (там прищепки); чтение (``read_preview``, ``read_raw``) поворачивает кадр по EXIF и затем корешком
вверх (``fold_line.to_spine_top``) — вся геометрия ниже и кэши ведутся в этой канонической системе «корешок у
ВЕРХНЕГО края». Нечётные полосы (``_1``) — правые, сгиб у начала строк; чётные (``_2``) — левые, сгиб у
концов строк. Признаки потери:

* **отступ** — между краем строк крайней колонки и линией сгиба не остаётся бумаги (в высотах строки);
* **обрубки** — крайнее у сгиба слово незнакомо словарю (``truncated_words.check_lines``: переносы склеиваются со
  смежной строкой, контроль — дальний край тех же строк).

Сгиб ищется как тёмная складка-тень бумаги у верхнего края кадра по кускам вдоль него с подгонкой параболы; всё
за сгибом (загиб соседней полосы) закрашивается цветом бумаги, иначе surya склеивает строки через сгиб.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from research.spread_defects.fold_line import LIBRAW_FLIP_TO_EXIF, SpineSide, spine_side_from_exif, to_spine_top
from research.spread_defects.truncated_words import adjacency, check_lines, clean, is_known

# ЭГ 1985 10-12: корешок у верхнего края снимка «как снято» (у сенсора). На кадре по EXIF это слева у ``_1``
# (Orientation 8) и справа у ``_2`` (Orientation 6) — сторона выводится из тега каждого кадра.
EG_1985_RAW_SPINE = SpineSide.TOP

# Поиск сгиба: глубина поиска от верхнего края кадра (px превью), ширина куска вдоль сгиба, глубина провала
# тени против бумаги ниже него (уровни яркости), окно «бумаги ниже» (px от провала).
FOLD_MAX_ROW, FOLD_CHUNK, FOLD_MIN_DIP = 900, 96, 15
FOLD_AFTER = (30, 120)
# Минимум кусков с уверенным провалом, чтобы сгиб считался найденным.
FOLD_MIN_CHUNKS = 8
# Лента у корешка для surya: столько px превью от сгиба вглубь полосы (≈ 2–3 колонки).
STRIP_DEPTH = 1350
# Крайняя колонка у сгиба: край строки не дальше стольких высот строки от сгиба (на нормальной полосе отступ
# 2–5 высот, плюс запас на перекос).
COLUMN_REACH = 14.0
# Ширина крайней колонки по краю у сгиба: строки не дальше стольких высот строки от ближнего края колонки (10-й
# перцентиль отступов в зоне поиска). На газете следующая колонка начинается в ~7–14 высотах строки; без этого
# сужения строки двух колонок перемежаются по высоте и хвосты переносов не склеиваются со своими строками —
# ложные «обрубки» на правых полосах (0180_1, 0480_1, 0920_1 в v2).
COLUMN_WIDTH = 3.0
# Строка «доходит до края колонки»: её край у сгиба не дальше стольких высот строки от опорного края колонки.
AT_EDGE = 1.5
# Строка «упирается в сгиб»: отступ меньше стольких высот строки.
TOUCH = 0.5
# Маска за сгибом не доходит до линии сгиба на столько px: линия — середина тени, хвосты букв лежат рядом, а
# линия ошибается вниз до ~10 px (эталон линии ЭГ 1985 10-12). Стереть букву хуже, чем оставить край загиба.
FOLD_KEEP = 14
# Вариант линии сгиба (``fold_line``): хребет складки + обход прищепок + RANSAC, запрет проходить сквозь строки
# surya — лучший на эталоне (reports/eg_1985_10_12_gutter.md, «Линия сгиба»).
FOLD_VARIANT, FOLD_BOXES = "C+F+2", "строки"

_NAME = re.compile(r"_(\d)$")


class Side(str, Enum):
    """Сторона полосы в развороте (как в ``truncated_words``): ``R`` — сгиб у начала строк, ``L`` — у концов."""

    LEFT = "L"
    RIGHT = "R"


def page_side(stem: str) -> Side:
    """Сторона полосы по имени кадра ЭГ: ``…_1`` — нечётная правая полоса, ``…_2`` — чётная левая.

    Args:
        stem: Имя кадра без расширения (``0030_1``).

    Returns:
        ``Side.RIGHT`` для ``_1``, ``Side.LEFT`` для ``_2``.

    Raises:
        ValueError: Имя без суффикса ``_1``/``_2``.
    """
    match = _NAME.search(stem)
    if not match or match.group(1) not in "12":
        raise ValueError(f"не понять сторону полосы по имени {stem!r}")
    return Side.RIGHT if match.group(1) == "1" else Side.LEFT


def find_fold(gray: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Линия сгиба у верхнего края кадра: строка кадра как парабола от столбца.

    По каждому куску ширины ``FOLD_CHUNK`` берётся профиль уровня бумаги по строкам (80-й перцентиль — текст не
    мешает), сглаживается, и самая тёмная строка в верхних ``FOLD_MAX_ROW`` считается складкой, если она темнее
    бумаги ниже неё на ``FOLD_MIN_DIP``. По уверенным кускам робастно (три прохода с отбросом выбросов)
    подгоняется парабола.

    Args:
        gray: Полутоновый кадр (float32 или uint8), корешок сверху.

    Returns:
        (коэффициенты параболы ``np.polyval`` или None, если уверенных кусков мало; x кусков; строки провалов;
        глубины провалов).
    """
    xs, rows, dips = [], [], []
    for x0 in range(0, gray.shape[1] - FOLD_CHUNK, FOLD_CHUNK):
        # Уровень бумаги по строке куска, сглаженный вдоль строк.
        level = np.percentile(gray[:FOLD_MAX_ROW, x0 : x0 + FOLD_CHUNK], 80, axis=1).astype(np.float32)
        level = cv2.GaussianBlur(level.reshape(-1, 1), (1, 9), 0).ravel()
        # Самая тёмная строка (первые 10 строк — край кадра, пропускаем).
        row = int(np.argmin(level[10:])) + 10
        lo, hi = row + FOLD_AFTER[0], row + FOLD_AFTER[1]
        after = float(np.median(level[lo:hi])) if hi < len(level) else float(level[-1])
        xs.append(x0 + FOLD_CHUNK / 2)
        rows.append(row)
        dips.append(after - float(level[row]))
    xs, rows, dips = np.asarray(xs), np.asarray(rows, dtype=float), np.asarray(dips)
    ok = dips > FOLD_MIN_DIP
    if ok.sum() < FOLD_MIN_CHUNKS:
        return None, xs, rows, dips
    coef = np.polyfit(xs[ok], rows[ok], 2)
    for _ in range(3):
        # Отброс кусков, далёких от текущей параболы (прищепки, заголовки, загиб соседней полосы).
        residual = np.abs(np.polyval(coef, xs) - rows)
        ok = ok & (residual < max(25.0, 3.0 * float(np.median(residual[ok]))))
        if ok.sum() < FOLD_MIN_CHUNKS:
            return None, xs, rows, dips
        coef = np.polyfit(xs[ok], rows[ok], 2)
    return coef, xs, rows, dips


@dataclass(frozen=True)
class GutterStrip:
    """Лента полосы у корешка, повёрнутая к прямому чтению, и сгиб в её координатах.

    Attributes:
        image: RGB uint8, текст прямо; за сгибом закрашено цветом бумаги.
        fold_col: x сгиба в ленте для каждой строки ленты (float, длина = высота ленты).
        side: Сторона полосы: у ``R`` сгиб слева, у ``L`` — справа.
    """

    image: np.ndarray
    fold_col: np.ndarray
    side: Side


def gutter_strip(
    rgb: np.ndarray, fold_rows: np.ndarray, side: Side, depth: int = STRIP_DEPTH, keep: float = FOLD_KEEP
) -> GutterStrip:
    """Вырезает ленту у корешка от сгиба вглубь полосы и поворачивает её к прямому чтению.

    Верхние строки кадра до ``max(сгиб) + depth``; всё выше сгиба закрашивается уровнем бумаги ленты. ``_1``
    (правая) поворачивается на 90° против часовой — сгиб уходит влево; ``_2`` (левая) — по часовой, сгиб справа.

    Args:
        rgb: Кадр RGB uint8, корешок сверху.
        fold_rows: Строка сгиба для каждого столбца кадра (``fold_line.find_line(...).rows`` или
            ``np.polyval`` параболы ``find_fold``).
        side: Сторона полосы (``page_side``).
        depth: Глубина ленты от самого глубокого места сгиба, px.
        keep: Маска загиба не доходит до линии сгиба на столько px (``FOLD_KEEP`` — для превью; для полного RAW
            масштабируется вместе с кадром).

    Returns:
        ``GutterStrip``.
    """
    fold_rows = np.asarray(fold_rows, dtype=float)
    height = int(min(rgb.shape[0], np.ceil(fold_rows.max()) + depth))
    strip = rgb[:height].copy()
    # Уровень бумаги: светлый перцентиль ленты ниже сгиба.
    paper = np.percentile(strip[int(fold_rows.max()) :], 90, axis=(0, 1)).astype(np.uint8)
    above = np.arange(height)[:, None] < (fold_rows - keep)[None, :]
    strip[above] = paper
    if side is Side.RIGHT:
        # np.rot90 k=1 (против часовой): строка ленты i ↔ столбец кадра W-1-i, столбец ленты j ↔ строка кадра j.
        image = np.ascontiguousarray(np.rot90(strip, 1))
        fold_col = fold_rows[::-1].copy()
    else:
        # np.rot90 k=-1 (по часовой): строка ленты i ↔ столбец кадра i, столбец ленты j ↔ строка кадра H-1-j.
        image = np.ascontiguousarray(np.rot90(strip, -1))
        fold_col = (height - 1) - fold_rows
    return GutterStrip(image=image, fold_col=fold_col, side=side)


@dataclass(frozen=True)
class GutterFeatures:
    """Признаки потери текста у корешка по строкам крайней колонки.

    Attributes:
        lines: Строк surya в крайней колонке у сгиба.
        line_height: Медианная высота строки, px.
        margin_p10, margin_median: 10-й перцентиль и медиана отступа края строк (доходящих до края колонки) от
            сгиба, в высотах строки; NaN, если строк нет.
        touch_share: Доля строк колонки с отступом меньше ``TOUCH`` высоты строки.
        fold_checked, fold_unknown, outer_checked, outer_unknown: Проверка крайних слов словарём
            (``truncated_words.EdgeCheck``).
        inner_checked, inner_unknown: Слова в середине тех же строк (без крайних) и сколько из них незнакомы —
            уровень ошибок OCR полосы; крайние «внешние» слова для этого не годятся: через перенос они склеиваются
            с обрубком у сгиба соседней строки.
        examples: Незнакомые слова у сгиба (до 8).
    """

    lines: int
    line_height: float
    margin_p10: float
    margin_median: float
    touch_share: float
    fold_checked: int
    fold_unknown: int
    outer_checked: int
    outer_unknown: int
    inner_checked: int
    inner_unknown: int
    examples: tuple[str, ...]

    @property
    def fold_share(self) -> float:
        """Доля незнакомых словарю слов у сгиба (NaN при < 8 проверенных)."""
        return self.fold_unknown / self.fold_checked if self.fold_checked >= 8 else float("nan")

    @property
    def outer_share(self) -> float:
        """Доля незнакомых словарю слов у дальнего края тех же строк (NaN при < 8 проверенных)."""
        return self.outer_unknown / self.outer_checked if self.outer_checked >= 8 else float("nan")

    @property
    def inner_share(self) -> float:
        """Доля незнакомых словарю слов в середине строк колонки (NaN при < 20 проверенных)."""
        return self.inner_unknown / self.inner_checked if self.inner_checked >= 20 else float("nan")

    @property
    def excess(self) -> float:
        """Превышение доли незнакомых у сгиба над дальним краем (NaN, если проверено мало)."""
        return self.fold_share - self.outer_share


def inner_words(rows: list[list[str]]) -> tuple[int, int]:
    """Проверка словарём слов в середине строк (без первого и последнего и без слов с переносом).

    Args:
        rows: Слова каждой строки как прочитаны.

    Returns:
        (проверено, незнакомо).
    """
    checked = unknown = 0
    for row in rows:
        for token in row[1:-1]:
            word = clean(token)
            if word is None or word.endswith("-"):
                continue
            checked += 1
            unknown += not is_known(word.strip("-"))
    return checked, unknown


def line_margin(line: dict, fold_col: np.ndarray, side: Side) -> float:
    """Отступ края строки у сгиба от линии сгиба на высоте центра строки.

    Args:
        line: Строка surya в координатах ленты (``bbox`` = [x0, y0, x1, y1]).
        fold_col: x сгиба для каждой строки ленты.
        side: Сторона полосы: у ``R`` край у сгиба — x0, у ``L`` — x1.

    Returns:
        Отступ, px; отрицательный — строка заходит за сгиб.
    """
    x0, y0, x1, y1 = line["bbox"]
    fold_x = float(fold_col[int(np.clip((y0 + y1) / 2, 0, len(fold_col) - 1))])
    return x0 - fold_x if side is Side.RIGHT else fold_x - x1


def fold_column(lines: list[dict], fold_col: np.ndarray, side: Side, line_height: float) -> list[dict]:
    """Строки крайней колонки у сгиба: край у сгиба в зоне ``COLUMN_REACH`` и не дальше ``COLUMN_WIDTH`` высот строки
    от ближнего края колонки.

    Args:
        lines: Строки surya в координатах ленты.
        fold_col: x сгиба для каждой строки ленты.
        side: Сторона полосы.
        line_height: Медианная высота строки, px.

    Returns:
        Строки колонки (порядок входа сохраняется); пусто, если у сгиба строк нет.
    """
    margins = np.array([line_margin(line, fold_col, side) for line in lines]) / line_height
    near = margins < COLUMN_REACH
    if not near.any():
        return []
    edge = float(np.percentile(margins[near], 10))
    return [line for line, m in zip(lines, margins) if m < min(COLUMN_REACH, edge + COLUMN_WIDTH)]


def gutter_features(lines: list[dict], fold_col: np.ndarray, side: Side) -> GutterFeatures:
    """Признаки по строкам surya ленты: отступ от сгиба и обрубки крайних слов.

    Крайняя колонка — ``fold_column`` (зона ``COLUMN_REACH``, ширина ``COLUMN_WIDTH`` от ближнего края). Опорный край колонки —
    20-й перцентиль положения краёв к сгибу; строка «у края», если её край не дальше ``AT_EDGE`` высот от опорного.
    Словарная проверка — ``truncated_words.check_lines``: у левой полосы проверяется только конец строк у края
    (концевые строки абзацев короче), у правой — начало строк у края (без абзацного отступа).

    Args:
        lines: Строки surya в координатах ленты: ``{"bbox": [x0, y0, x1, y1], "text": str, "conf": float}``.
        fold_col: x сгиба для каждой строки ленты (``GutterStrip.fold_col``).
        side: Сторона полосы.

    Returns:
        ``GutterFeatures`` (нулевые и NaN, если строк у сгиба нет).
    """
    empty = GutterFeatures(0, float("nan"), float("nan"), float("nan"), float("nan"), 0, 0, 0, 0, 0, 0, ())
    text_lines = [line for line in lines if len(line["text"].strip()) >= 3]
    if not text_lines:
        return empty
    heights = np.array([line["bbox"][3] - line["bbox"][1] for line in text_lines])
    line_height = float(np.median(heights))
    column = fold_column(text_lines, fold_col, side, line_height)
    if not column:
        return GutterFeatures(0, line_height, *([float("nan")] * 3), 0, 0, 0, 0, 0, 0, ())
    margins = np.array([line_margin(line, fold_col, side) for line in column]) / line_height
    # Опорный край колонки и строки, доходящие до него.
    reference = float(np.percentile(margins, 20))
    at_edge = margins < reference + AT_EDGE
    order = np.argsort([(line["bbox"][1] + line["bbox"][3]) / 2 for line in column])
    column = [column[n] for n in order]
    margins, at_edge = margins[order], at_edge[order]
    centers = [(line["bbox"][1] + line["bbox"][3]) / 2 for line in column]
    words = [line["text"].replace("<br>", " ").split() for line in column]
    # Строки не у края колонки у сгиба не проверяем, но оставляем соседями для переносов: у левой полосы короткая
    # концевая строка абзаца не несёт переноса, её слова не нужны; у правой — нужен конец (перенос на следующую).
    texts = [w if edge else ([] if side is Side.LEFT else w) for w, edge in zip(words, at_edge)]
    check = check_lines(texts, side.value, adjacency(centers), lost_hyphen=True, skip_capitalized=True)
    if side is Side.RIGHT:
        # Начало строки с абзацным отступом не обрубок: у сгиба считаем только строки у края колонки.
        partial = check_lines(
            [w if edge else [] for w, edge in zip(words, at_edge)],
            side.value,
            adjacency(centers),
            lost_hyphen=True,
            skip_capitalized=True,
        )
        fold_checked, fold_unknown, examples = partial.fold_checked, partial.fold_unknown, partial.examples
    else:
        fold_checked, fold_unknown, examples = check.fold_checked, check.fold_unknown, check.examples
    edge_margins = margins[at_edge]
    inner_checked, inner_unknown = inner_words(words)
    return GutterFeatures(
        lines=len(column),
        line_height=line_height,
        margin_p10=float(np.percentile(edge_margins, 10)),
        margin_median=float(np.median(edge_margins)),
        touch_share=float(np.mean(margins < TOUCH)),
        fold_checked=fold_checked,
        fold_unknown=fold_unknown,
        outer_checked=check.outer_checked,
        outer_unknown=check.outer_unknown,
        inner_checked=inner_checked,
        inner_unknown=inner_unknown,
        examples=examples,
    )


def read_raw(path: Path, spine_side: SpineSide | None = None) -> np.ndarray:
    """Полный кадр RAF (rawpy, баланс белого камеры, 8 бит), повёрнутый по метаданным, затем корешком вверх.

    Args:
        path: Путь RAF.
        spine_side: Сторона корешка на кадре, повёрнутом по метаданным; None — из метаданных кадра по правилу
            ЭГ 1985 (корешок у верха снимка «как снято», :data:`EG_1985_RAW_SPINE`).

    Returns:
        RGB uint8 полного разрешения в канонической системе «корешок сверху» (у X100VI ЭГ 1985 — 5178×7752).
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        upright = raw.postprocess(use_camera_wb=True, output_bps=8)
        orientation = LIBRAW_FLIP_TO_EXIF[raw.sizes.flip]
    side = spine_side if spine_side is not None else spine_side_from_exif(orientation, EG_1985_RAW_SPINE)
    return to_spine_top(upright, side)


def scale_fold(fold_rows: np.ndarray, preview_shape: tuple[int, int], full_shape: tuple[int, int]) -> np.ndarray:
    """Линия сгиба превью → линия сгиба полного кадра (масштаб по осям; превью и RAW совпадают с точностью до
    нескольких px превью — проверено фазовой корреляцией на 1350_2 и 1200_1, это покрывает запас маски).

    Args:
        fold_rows: Строка сгиба для каждого столбца превью.
        preview_shape: (высота, ширина) превью.
        full_shape: (высота, ширина) полного кадра.

    Returns:
        Строка сгиба для каждого столбца полного кадра.
    """
    kx, ky = full_shape[1] / preview_shape[1], full_shape[0] / preview_shape[0]
    columns = np.arange(full_shape[1]) / kx
    return np.interp(columns, np.arange(len(fold_rows)), fold_rows) * ky


def read_preview(path: Path, spine_side: SpineSide | None = None) -> np.ndarray:
    """Превью кадра (JPEG, извлечённый из RAF) как RGB uint8: повёрнуто по EXIF, затем корешком вверх.

    Args:
        path: Путь JPEG.
        spine_side: Сторона корешка на превью, повёрнутом по EXIF; None — из тега Orientation кадра по правилу
            ЭГ 1985 (корешок у верха снимка «как снято», :data:`EG_1985_RAW_SPINE`).

    Returns:
        Массив H×W×3 uint8 в канонической системе «корешок сверху».

    Raises:
        OSError: Файл не читается.
    """
    # По EXIF (cv2.imread поворачивает сам), затем каноническая система детектора сгиба — корешок сверху.
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"не читается {path}")
    side = spine_side if spine_side is not None else spine_side_from_exif(exif_orientation(path), EG_1985_RAW_SPINE)
    return to_spine_top(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), side)


def exif_orientation(path: Path) -> int:
    """Тег EXIF Orientation JPEG (1, если тега нет).

    Args:
        path: Путь JPEG.

    Returns:
        Значение тега.
    """
    from PIL import Image

    with Image.open(path) as img:
        return int(img.getexif().get(0x0112, 1))


class Verdict(str, Enum):
    """Решение по кадру; значение — имя папки симлинков."""

    LOSS = "есть_потеря_текста_у_корешка"
    NORMAL = "текст_у_корешка_в_норме"
    UNCLEAR = "непонятно"


# Пороги решения (ЭГ 1985 10-12, v5). Доля незнакомых словарю крайних у сгиба слов — по согласию чтений RAW и превью.
LOSS_SHARE, NORMAL_SHARE = 0.16, 0.05
# Спорный пояс по пикселям: доля обрубков, с которой — «потеря».
BAND_LOSS_SHARE = 0.10
# Геометрическая норма: строки крайней колонки не касаются складки (доля касающихся и 10-й перцентиль отступа в
# высотах строки). У 13 из 14 норм пользователя из «непонятно» v4 — так; ни у одной из 23 его потерь.
GEO_TOUCH, GEO_MARGIN = 0.4, 0.35
# Медиана просвета крайнего знака до тени складки, высоты строки: распределение по папке двугорбое — около 0 (буквы
# целы, в тень уходит лишь знак переноса) и ≤ −0.10 (буквы тонут в тени); между −0.04 и −0.02 почти пусто.
GAP_LOSS, GAP_NORMAL = -0.10, -0.03
# Мало слов у сгиба для решения (фото, реклама, заголовки) и слишком шумный OCR полосы; мало строк у складки.
MIN_CHECKED, MAX_INNER_SHARE, MIN_TOUCHING = 15, 0.08, 5


@dataclass(frozen=True)
class FrameSigns:
    """Признаки кадра для решения.

    Attributes:
        fold_checked, fold_share, inner_share: Словарь (``GutterFeatures``; доля и шум — по согласию чтений).
        touch_share, margin_p10: Геометрия строк крайней колонки (``GutterFeatures``).
        touching, gap_median: Пиксели у складки (``PixelFeatures``); NaN — не измерено.
    """

    fold_checked: int
    fold_share: float
    inner_share: float
    touch_share: float = float("nan")
    margin_p10: float = float("nan")
    touching: int = 0
    gap_median: float = float("nan")


def decide(
    signs: FrameSigns, loss_share: float = LOSS_SHARE, normal_share: float = NORMAL_SHARE
) -> tuple[Verdict, str]:
    """Решение по признакам кадра.

    Порядок: геометрическая норма (текст не касается складки) → словарь (уверенная потеря) → пиксели (буквы тонут
    в тени / целы) → спорный пояс пикселей решает словарь → иначе «непонятно».

    Args:
        signs: Признаки кадра.
        loss_share: Доля обрубков, начиная с которой — «есть потеря».
        normal_share: Доля обрубков, ниже которой словарь говорит «в норме».

    Returns:
        (решение, причина для отчёта).
    """
    share, inner, gap = signs.fold_share, signs.inner_share, signs.gap_median
    if signs.touch_share < GEO_TOUCH and signs.margin_p10 >= GEO_MARGIN:
        return Verdict.NORMAL, f"строки не касаются складки (касаются {signs.touch_share:.2f})"
    noisy = not np.isnan(inner) and inner > MAX_INNER_SHARE
    enough = signs.fold_checked >= MIN_CHECKED and not np.isnan(share)
    if enough and share >= loss_share and not noisy:
        return Verdict.LOSS, f"обрубков у сгиба {share:.2f}"
    if signs.touching >= MIN_TOUCHING and not np.isnan(gap):
        if gap <= GAP_LOSS:
            return Verdict.LOSS, f"буквы уходят в тень складки (просвет {gap:+.2f})"
        if not enough or noisy:
            return Verdict.UNCLEAR, f"просвет {gap:+.2f}, слов у сгиба {signs.fold_checked}, шум OCR {inner:.2f}"
        if gap > GAP_NORMAL and share < loss_share:
            return Verdict.NORMAL, f"буквы целы у складки (просвет {gap:+.2f}), обрубков {share:.2f}"
        if share >= BAND_LOSS_SHARE:
            return Verdict.LOSS, f"просвет {gap:+.2f} (спорный), обрубков {share:.2f}"
        if share < normal_share:
            return Verdict.NORMAL, f"просвет {gap:+.2f} (спорный), обрубков {share:.2f}"
        return Verdict.UNCLEAR, f"просвет {gap:+.2f} и обрубков {share:.2f} — оба спорные"
    if not enough:
        return Verdict.UNCLEAR, f"мало слов у сгиба ({signs.fold_checked}) и строк у складки ({signs.touching})"
    if noisy:
        return Verdict.UNCLEAR, f"шумный OCR полосы (в середине строк {inner:.2f})"
    if share < normal_share:
        return Verdict.NORMAL, f"обрубков у сгиба {share:.2f}"
    return Verdict.UNCLEAR, f"обрубков у сгиба {share:.2f} — между порогами"


# Куски колонки для DeepSeek-OCR-2: высота куска и перекрытие в высотах строки, заступ за сгиб и запас за дальним
# краем колонки в высотах строки (на ЭСХ по ~12–14 строк в куске модель не склеивает строки).
DS_CHUNK, DS_OVERLAP, DS_OVER_FOLD, DS_FAR_PAD = 13.0, 1.0, 1.0, 1.0


def deepseek_crops(strip: GutterStrip, lines: list[dict], stem: str, crop_dir: Path) -> list[dict]:
    """Куски крайней колонки у сгиба для второго читателя (DeepSeek-OCR-2), PNG на диск.

    Колонка — ``fold_column``; дальний край колонки —
    медиана дальних краёв этих строк. Колонка режется по высоте на куски ``DS_CHUNK`` строк с перекрытием.

    Args:
        strip: Лента у корешка (``gutter_strip``) в полном разрешении превью.
        lines: Строки surya этой ленты.
        stem: Имя кадра (префикс id заданий).
        crop_dir: Папка вырезок кадра.

    Returns:
        Задания воркера ``{"id": "<кадр>|<сторона>|<номер>", "crop": путь}``; пусто, если колонки нет.
    """
    text_lines = [line for line in lines if len(line["text"].strip()) >= 3]
    if not text_lines:
        return []
    line_height = float(np.median([line["bbox"][3] - line["bbox"][1] for line in text_lines]))
    column = fold_column(text_lines, strip.fold_col, strip.side, line_height)
    if len(column) < 5:
        return []
    height, width = strip.image.shape[:2]
    gray = cv2.cvtColor(strip.image, cv2.COLOR_RGB2GRAY)
    top = int(min(line["bbox"][1] for line in column))
    bottom = int(max(line["bbox"][3] for line in column))
    step, size = int((DS_CHUNK - DS_OVERLAP) * line_height), int(DS_CHUNK * line_height)
    crop_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for n, y0 in enumerate(range(top, max(top + 1, bottom - size // 3), step)):
        y1 = min(height, y0 + size)
        fold_x = float(np.median(strip.fold_col[y0:y1]))
        if strip.side is Side.RIGHT:
            far = float(np.median([line["bbox"][2] for line in column]))
            x0, x1 = int(fold_x - DS_OVER_FOLD * line_height), int(far + DS_FAR_PAD * line_height)
        else:
            far = float(np.median([line["bbox"][0] for line in column]))
            x0, x1 = int(far - DS_FAR_PAD * line_height), int(fold_x + DS_OVER_FOLD * line_height)
        x0, x1 = max(0, x0), min(width, x1)
        path = crop_dir / f"{n:02d}.png"
        if not path.exists():
            cv2.imwrite(str(path), gray[y0:y1, x0:x1])
        jobs.append({"id": f"{stem}|{strip.side.value}|{n:02d}", "crop": str(path)})
    return jobs


def deepseek_check(answers: dict[str, str], stem: str, side: Side):
    """Словарная проверка крайних у сгиба слов по ответам DeepSeek на куски колонки кадра.

    Args:
        answers: Ответы воркера (``deepseek_fold.read_answers``).
        stem: Имя кадра.
        side: Сторона полосы.

    Returns:
        ``truncated_words.EdgeCheck`` (с учётом потерянного дефиса).
    """
    texts = []
    for chunk in sorted(k for k in answers if k.startswith(f"{stem}|")):
        for raw in answers[chunk].replace("­", "-").splitlines():
            if raw.split():
                texts.append(raw.split())
    return check_lines(texts, side.value, lost_hyphen=True, skip_capitalized=True)


# --- Пиксельные признаки у складки: срезана ли крайняя буква, затемнены ли буквы ---------------------------------

# Строка «касается складки»: край строки surya ближе стольких высот строки к линии сгиба.
TOUCH_LINE = 0.5
# Бумага для маски краски: максимум (закрытие) в окне такой доли высоты строки, затем сглаживание. Окно шире штриха
# (~0.1 высоты строки), но уже буквы: тень складки нарастает плавно и на таком окне уходит в бумагу (окно 0.8 —
# тень у края считалась краской и «срезанным знаком» в каждой строке, 0810_2).
PAPER_WINDOW = 0.35
# Краска — темнее локальной бумаги больше чем на такую долю.
INK_DEPTH = 0.18
# Знак: компонента краски выше стольких долей высоты строки (отсекает точки и сор) и не шире 1.5 высоты.
GLYPH_MIN_HEIGHT, GLYPH_MAX_WIDTH = 0.25, 1.5
# Просвета нет: крайний знак ближе стольких долей высоты строки к линии маски загиба (куда видна страница).
NO_GAP = 0.12
# Зона «у складки» для затемнения: столько высот строки от края строки у сгиба.
DARK_ZONE = 1.5
# Крайний знак «глубоко в тени»: просвет меньше минус столько высот строки.
DEEP = 0.2
# Тень складки на строке: уровень бумаги ниже такой доли бумаги строки и до конца видимой ленты.
SHADOW_LEVEL = 0.8


@dataclass(frozen=True)
class LineEdge:
    """Крайний у складки знак одной строки и затемнение букв у складки.

    Attributes:
        box: Рамка строки в ленте (x0, y0, x1, y1).
        glyphs: Рамки знаков строки в зоне у складки (x0, y0, x1, y1), в ленте.
        edge: Индекс крайнего у складки знака в ``glyphs`` или -1 (знаков нет).
        gap: Просвет от крайнего знака до начала тени складки, в высотах строки (≤ 0 — знак заходит в тень).
        width_ratio: Ширина крайнего знака к медиане ширины знаков строки (срезанная буква уже).
        contrast_ratio: Контраст букв у складки к контрасту букв в середине строки (затемнение — меньше 1).
        shadow: x начала тени складки на строке (граница видимой страницы), в ленте; NaN — не измерено.
    """

    box: tuple[float, float, float, float]
    glyphs: tuple[tuple[int, int, int, int], ...]
    edge: int
    gap: float
    width_ratio: float
    contrast_ratio: float
    shadow: float


def ink_mask(gray: np.ndarray, line_height: float) -> np.ndarray:
    """Маска краски относительно локальной бумаги (устойчива к тени складки).

    Args:
        gray: Полутоновый фрагмент uint8.
        line_height: Высота строки, px.

    Returns:
        Маска bool той же формы.
    """
    k = max(3, int(PAPER_WINDOW * line_height) | 1)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    paper = cv2.GaussianBlur(paper.astype(np.float32), (0, 0), k / 3)
    return gray.astype(np.float32) < (1 - INK_DEPTH) * paper


def _contrast(gray: np.ndarray) -> float:
    """Контраст букв во фрагменте: 90-й перцентиль (бумага) минус 5-й (краска); NaN для пустого."""
    return float(np.percentile(gray, 90) - np.percentile(gray, 5)) if gray.size else float("nan")


def shadow_start(gray: np.ndarray, side: Side, line_height: float) -> tuple[int, float]:
    """Где на строке бумага уходит в тень складки: первый со стороны страницы столбец, после которого уровень бумаги
    держится ниже ``SHADOW_LEVEL`` бумаги строки.

    Args:
        gray: Фрагмент строки (полутон) от страницы до видимого края; у ``L`` складка справа, у ``R`` — слева.
        side: Сторона полосы.
        line_height: Высота строки, px.

    Returns:
        (столбец начала тени во фрагменте — ширина фрагмента, если тени нет; уровень бумаги строки).
    """
    profile = np.percentile(gray, 90, axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), max(1.0, line_height / 12)).ravel()
    if side is Side.RIGHT:
        profile = profile[::-1]
    width = len(profile)
    paper = float(np.percentile(profile[: max(1, width // 2)], 75))
    dark = profile < SHADOW_LEVEL * paper
    # Первый столбец, с которого и до конца фрагмента бумага темнее порога (тень не прерывается).
    tail_light = np.flip(np.cumsum(np.flip(~dark))) > 0  # есть ли светлый столбец правее
    start = width
    for x in range(width):
        if dark[x] and not (tail_light[x + 1] if x + 1 < width else False):
            start = x
            break
    if side is Side.RIGHT:
        start = width - start
    return start, paper


def line_edge(
    gray: np.ndarray, line: dict, fold_col: np.ndarray, side: Side, line_height: float, keep: float
) -> LineEdge:
    """Крайний знак строки у складки: просвет до тени складки, ширина знака, затемнение.

    Граница видимой страницы на строке — начало тени складки (``shadow_start``), а не линия сгиба: страница уходит в
    складку постепенно, и буквы тонут в тени раньше линии (0810_2). Знак, заходящий в тень, — срезанный.

    Args:
        gray: Лента у сгиба (полутон, прямой текст) — та же, что читал surya.
        line: Строка surya в координатах ленты.
        fold_col: x сгиба для каждой строки ленты.
        side: Сторона полосы (у ``L`` сгиб справа, у ``R`` — слева).
        line_height: Медианная высота строки, px.
        keep: Запас маски загиба, px (как в ``gutter_strip``).

    Returns:
        ``LineEdge``; ``shadow`` — x начала тени в ленте.
    """
    x0, y0, x1, y1 = (float(v) for v in line["bbox"])
    height, width = gray.shape
    yc = int(np.clip((y0 + y1) / 2, 0, len(fold_col) - 1))
    # Видимый край ленты (дальше — закраска загиба).
    visible = float(fold_col[yc]) + (keep if side is Side.RIGHT else -keep)
    reach = int(DARK_ZONE * 2 * line_height)
    r0, r1 = int(max(0, y0)), int(min(height, y1))
    if side is Side.LEFT:
        c0, c1 = int(max(0, visible - reach)), int(min(width, visible - 1))
    else:
        c0, c1 = int(max(0, visible + 2)), int(min(width, visible + reach))
    empty = LineEdge((x0, y0, x1, y1), (), -1, float("nan"), float("nan"), float("nan"), float("nan"))
    if r1 - r0 < 4 or c1 - c0 < 4:
        return empty
    crop = gray[r0:r1, c0:c1]
    start, _ = shadow_start(crop, side, line_height)
    shadow = float(c0 + start)
    ink = ink_mask(crop, line_height).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    glyphs = []
    for left, top, w, h, area in stats[1:count]:
        gx0, gx1 = int(left + c0), int(left + w + c0)
        # Знак: достаточной высоты, не шире буквы и начинается до тени (в самой тени — только хвосты букв и сор).
        starts_lit = gx0 < shadow if side is Side.LEFT else gx1 > shadow
        if h >= GLYPH_MIN_HEIGHT * line_height and w <= GLYPH_MAX_WIDTH * line_height and area >= 6 and starts_lit:
            glyphs.append((gx0, int(top + r0), gx1, int(top + h + r0)))
    if not glyphs:
        return LineEdge((x0, y0, x1, y1), (), -1, float("nan"), float("nan"), float("nan"), shadow)
    edge = (
        max(range(len(glyphs)), key=lambda n: glyphs[n][2])
        if side is Side.LEFT
        else min(range(len(glyphs)), key=lambda n: glyphs[n][0])
    )
    gx0, _, gx1, _ = glyphs[edge]
    gap = (shadow - gx1) if side is Side.LEFT else (gx0 - shadow)
    widths = np.array([g[2] - g[0] for g in glyphs], float)
    width_ratio = float(widths[edge] / np.median(widths))
    # Затемнение: контраст букв у складки против середины строки.
    zone = DARK_ZONE * line_height
    near = (
        gray[r0:r1, int(max(0, shadow - zone)) : int(shadow)]
        if side is Side.LEFT
        else gray[r0:r1, int(shadow) : int(min(width, shadow + zone))]
    )
    mid = int((x0 + x1) / 2)
    middle = gray[r0:r1, int(max(0, mid - line_height)) : int(min(width, mid + line_height))]
    contrast_ratio = _contrast(near) / max(_contrast(middle), 1.0)
    return LineEdge((x0, y0, x1, y1), tuple(glyphs), edge, gap / line_height, width_ratio, contrast_ratio, shadow)


@dataclass(frozen=True)
class PixelFeatures:
    """Пиксельные признаки кадра по строкам крайней колонки, касающимся складки.

    Attributes:
        touching: Строк, касающихся складки (край строки ближе ``TOUCH_LINE`` высоты строки).
        no_gap_share: Доля из них, где крайний знак упирается в видимый край страницы (просвет < ``NO_GAP``).
        narrow_share: Доля из них, где крайний знак уже 0.6 медианного знака строки.
        contrast_median, contrast_p10: Медиана и 10-й перцентиль контраста букв у складки к середине строки.
        gap_median: Медиана просвета крайнего знака до тени складки, в высотах строки (отрицательный — буквы тонут в
            тени: потеря или сильное затемнение; у плотной, но целой нормы в тень уходит лишь знак переноса, ~0).
        deep_share: Доля строк, где крайний знак уходит в тень глубже ``DEEP`` высоты строки.
    """

    touching: int
    no_gap_share: float
    narrow_share: float
    contrast_median: float
    contrast_p10: float
    gap_median: float
    deep_share: float


def pixel_features(edges: list[LineEdge]) -> PixelFeatures:
    """Сводка ``LineEdge`` строк, касающихся складки, по кадру.

    Args:
        edges: Края строк (только касающиеся складки).

    Returns:
        ``PixelFeatures`` (NaN-доли, если строк нет).
    """
    good = [e for e in edges if e.edge >= 0]
    if not good:
        return PixelFeatures(0, *([float("nan")] * 6))
    gaps = np.array([e.gap for e in good])
    narrow = np.array([e.width_ratio for e in good])
    contrast = np.array([e.contrast_ratio for e in good if np.isfinite(e.contrast_ratio)])
    return PixelFeatures(
        touching=len(good),
        no_gap_share=float(np.mean(gaps < NO_GAP)),
        narrow_share=float(np.mean(narrow < 0.6)),
        contrast_median=float(np.median(contrast)) if len(contrast) else float("nan"),
        contrast_p10=float(np.percentile(contrast, 10)) if len(contrast) else float("nan"),
        gap_median=float(np.median(gaps)),
        deep_share=float(np.mean(gaps < -DEEP)),
    )
