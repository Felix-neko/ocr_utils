"""Ксерокопии вместо страниц: признаки бумаги, тонера и фона по текстовым плиткам каждой полосы разворота.

Что видно на примерах (ВЭ пак-3: 1965/12 «ксера статьи Кронрода», 1968/03 IMG_0066–0067, 1964/12-copy
IMG_0024–0032). Ксерокопия — отдельный лист, вклеенный вместо вырванной страницы. Бумага белее и холоднее
пожелтевшей книжной, без волокна и просвета. Краска (тонер) чернее и жёстче по краю. На плохих копиях фон
серый, неровный, с крапом тонера. Цвет кадра сам по себе ненадёжен: у камеры автобаланс белого. Поэтому
признаки здесь только меряются по полосам, а сравнение с медианой своего выпуска и с соседней полосой
того же кадра делает ранжирование (``photocopy_scores``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from research.spread_defects.motion import TILE, line_share, midtone_share, tile_ink

# Уровень бумаги: максимум в окне (больше межстрочья) и сглаживание, px полного кадра (как в ``exposure``).
PAPER_WINDOW, PAPER_SIGMA = 41, 15
# Пиксель — бумага, если его отражение (доля уровня бумаги) не ниже этого.
PAPER_REFL = 0.93
# Краска — отражение ниже этого.
INK_REFL = 0.55
# Зерно бумаги: СКО остатка после вычитания размытия с этой сигмой, px, только по пикселям бумаги.
GRAIN_SIGMA = 3.0
# Крап: тёмная компонента площадью в этих пределах (px полного кадра, 450 dpi: 2–40 px ≈ 0.08–0.35 мм)
# на расстоянии не меньше ``SPECK_CLEAR`` px от краски букв.
SPECK_MIN, SPECK_MAX, SPECK_CLEAR = 2, 40, 9
# Крап — темнее бумаги хотя бы на эту долю (слабее — зерно и пыль на бумаге оригинала).
SPECK_REFL = 0.75
# Плитка не годится, если прямых длинных линий много (линейки, рамки таблиц, край листа).
MAX_LINES = 0.05
# Минимум текстовых плиток на полосе, чтобы считать сводку.
MIN_TILES = 8
# Ширина зоны у сгиба, которую пропускаем (тень, изгиб листа), px полного кадра.
FOLD_MARGIN = 200
# Край выпуска: столько первых и последних кадров (обложки, реклама, вклейки) не входят в опорные медианы.
EDGE_FRAMES = 6

SIDES = ("L", "R")
# Признаки плитки (и полосы — медианы по плиткам).
FEATURES = ("paper_l", "paper_a", "paper_b", "grain", "speck", "ink_refl", "midtone", "unevenness")


@dataclass(frozen=True)
class CopyTiles:
    """Сетка признаков «ксерокопии» по плиткам ``TILE`` кадра.

    Attributes:
        text: bool[gy, gx] — текстовая плитка (``motion.tile_ink``, без линеек и рамок).
        paper_l, paper_a, paper_b: float32[gy, gx] — среднее L*, a*, b* (OpenCV Lab, a/b со сдвигом −128)
            по пикселям бумаги.
        grain: float32[gy, gx] — СКО высокочастотного остатка по пикселям бумаги, уровни 0..255.
        speck: float32[gy, gx] — число отдельных тёмных пятнышек на 1000 пикселей бумаги.
        ink_refl: float32[gy, gx] — медиана отражения пикселей краски (0 — чёрная, 1 — как бумага).
        midtone: float32[gy, gx] — доля полутонов среди краски (``motion.midtone_share``).
        unevenness: float32[gy, gx] — размах уровня бумаги в плитке (p95 − p5) к его медиане.
    """

    text: np.ndarray
    paper_l: np.ndarray
    paper_a: np.ndarray
    paper_b: np.ndarray
    grain: np.ndarray
    speck: np.ndarray
    ink_refl: np.ndarray
    midtone: np.ndarray
    unevenness: np.ndarray


def speck_mask(ink: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Крап: мелкие тёмные компоненты вдали от букв.

    Args:
        ink: uint8-маска краски букв (отражение < ``INK_REFL``).
        dark: uint8-маска всего тёмного (отражение < ``SPECK_REFL``), включает ``ink``.

    Returns:
        bool-маска пикселей крапа (компоненты ``dark`` площадью ``SPECK_MIN``–``SPECK_MAX``, не касающиеся
        расширенной на ``SPECK_CLEAR`` px краски крупных компонент).
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    small = (area >= SPECK_MIN) & (area <= SPECK_MAX)
    small[0] = False
    # Крупная краска (буквы, линейки) и окрестность вокруг неё: точки, запятые и обрывки букв рядом не крап.
    big = (~small)[labels] & (dark > 0)
    near = cv2.dilate(big.astype(np.uint8), np.ones((2 * SPECK_CLEAR + 1, 2 * SPECK_CLEAR + 1), np.uint8)) > 0
    touched = np.unique(labels[near & (dark > 0)])
    small[touched] = False
    return small[labels]


def copy_tiles(rgb: np.ndarray) -> CopyTiles:
    """Признаки «ксерокопии» по плиткам кадра полного разрешения.

    Args:
        rgb: Кадр H×W×3 uint8.

    Returns:
        ``CopyTiles`` с сеткой ``H // TILE`` × ``W // TILE`` (та же, что у ``motion_tiles`` и ``exposure_tiles``).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    # Уровень бумаги в окрестности каждого пикселя и отражение относительно него.
    paper = cv2.dilate(gray, np.ones((PAPER_WINDOW, PAPER_WINDOW), np.uint8)).astype(np.float32)
    paper = cv2.GaussianBlur(paper, (0, 0), PAPER_SIGMA)
    refl = gray.astype(np.float32) / np.maximum(paper, 1.0)
    ink = (refl < INK_REFL).astype(np.uint8)
    dark = (refl < SPECK_REFL).astype(np.uint8)
    # Высокочастотный остаток для зерна бумаги.
    residual = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), GRAIN_SIGMA)
    gy, gx = gray.shape[0] // TILE, gray.shape[1] // TILE
    text = np.zeros((gy, gx), bool)
    grids = {name: np.full((gy, gx), np.nan, np.float32) for name in FEATURES}
    for row in range(gy):
        for col in range(gx):
            ys, xs = slice(row * TILE, (row + 1) * TILE), slice(col * TILE, (col + 1) * TILE)
            # Только плитки печатного текста на бумаге: без обложек, пальцев, линеек и рамок.
            if tile_ink(hsv[ys, xs]) is None:
                continue
            value = hsv[ys, xs][..., 2]
            if line_share(value, float(np.percentile(value, 90))) > MAX_LINES:
                continue
            is_paper = refl[ys, xs] >= PAPER_REFL
            # Межбуквенные пиксели бумаги, отодвинутые от краски (ореол букв не зерно).
            clean = is_paper & (cv2.dilate(ink[ys, xs], np.ones((5, 5), np.uint8)) == 0)
            is_ink = ink[ys, xs].astype(bool)
            if clean.sum() < 2000 or is_ink.sum() < 500:
                continue
            text[row, col] = True
            tile_lab = lab[ys, xs]
            grids["paper_l"][row, col] = float(np.mean(tile_lab[..., 0][clean]))
            grids["paper_a"][row, col] = float(np.mean(tile_lab[..., 1][clean])) - 128.0
            grids["paper_b"][row, col] = float(np.mean(tile_lab[..., 2][clean])) - 128.0
            grids["grain"][row, col] = float(np.std(residual[ys, xs][clean]))
            specks = speck_mask(ink[ys, xs], dark[ys, xs])
            n_specks = cv2.connectedComponents(specks.astype(np.uint8), connectivity=8)[0] - 1
            grids["speck"][row, col] = 1000.0 * n_specks / float(clean.sum())
            grids["ink_refl"][row, col] = float(np.median(refl[ys, xs][is_ink]))
            grids["midtone"][row, col] = midtone_share(gray[ys, xs])
            level = paper[ys, xs]
            p5, p50, p95 = np.percentile(level, (5, 50, 95))
            grids["unevenness"][row, col] = float((p95 - p5) / max(p50, 1.0))
    return CopyTiles(text=text, **grids)


def side_summary(tiles: CopyTiles, fold_x: float) -> dict:
    """Сводка признаков по полосам разворота: медианы по текстовым плиткам каждой стороны сгиба.

    Args:
        tiles: Сетка кадра.
        fold_x: x сгиба посередине высоты, px полного кадра.

    Returns:
        Словарь ``<сторона>_<признак>`` (медиана по плиткам полосы) и ``<сторона>_tiles`` (число плиток);
        признаки полосы, где плиток меньше ``MIN_TILES``, не пишутся.
    """
    out = {}
    centers = (np.arange(tiles.text.shape[1]) + 0.5) * TILE
    for side, cols in (("L", centers < fold_x - FOLD_MARGIN), ("R", centers > fold_x + FOLD_MARGIN)):
        mask = tiles.text & cols[None, :]
        out[f"{side}_tiles"] = int(mask.sum())
        if mask.sum() < MIN_TILES:
            continue
        for name in FEATURES:
            out[f"{side}_{name}"] = round(float(np.nanmedian(getattr(tiles, name)[mask])), 4)
    return out


def pages_table(frames: pd.DataFrame) -> pd.DataFrame:
    """Сводка по кадрам → таблица по полосам (строка на сторону кадра).

    Args:
        frames: Сводка прогона ``copies`` (``key, issue, index, path, L_*, R_*``).

    Returns:
        Таблица ``key, issue, index, path, side`` + признаки ``FEATURES`` + ``tiles``; полосы без признаков
        (мало текстовых плиток) отброшены.
    """
    parts = []
    for side in SIDES:
        columns = {f"{side}_{name}": name for name in (*FEATURES, "tiles")}
        present = [c for c in columns if c in frames.columns]
        part = frames[["key", "issue", "index", "path", *present]].rename(columns=columns)
        part.insert(4, "side", side)
        parts.append(part)
    pages = pd.concat(parts, ignore_index=True)
    return pages.dropna(subset=[f for f in FEATURES if f in pages.columns]).reset_index(drop=True)


def relative_features(pages: pd.DataFrame) -> pd.DataFrame:
    """Признаки полос относительно своего выпуска и соседней полосы того же кадра.

    Опора — только «тело» выпуска: первые и последние ``EDGE_FRAMES`` кадров (обложки, реклама, вклейки,
    оглавление на другой бумаге) в медианы не входят и помечаются ``edge``. Ксерокопий в выпуске всегда
    меньшинство, поэтому медиана тела — это книжная бумага. Для каждого признака ``f`` добавляются:

    * ``f_iz`` — робастный z к медиане тела выпуска по той же стороне (свет и баланс белого у левой и правой
      страницы разные, но в выпуске стабильны);
    * ``f_pair`` — разность с другой полосой того же кадра (NaN, если её нет).

    Args:
        pages: Таблица ``pages_table``.

    Returns:
        Копия с новыми колонками и булевой ``edge``.
    """
    out = pages.copy()
    # Порядок кадров в выпуске и признак «край выпуска».
    order = out[["issue", "index", "key"]].drop_duplicates("key").sort_values(["issue", "index", "key"])
    order["position"] = order.groupby("issue").cumcount()
    order["count"] = order.groupby("issue")["key"].transform("size")
    order["edge"] = (order.position < EDGE_FRAMES) | (order.position >= order["count"] - EDGE_FRAMES)
    out = out.merge(order[["key", "position", "edge"]], on="key", how="left")
    for name in FEATURES:
        issue_z = np.full(len(out), np.nan)
        scales = {}
        for group_key, group in out.groupby(["issue", "side"]):
            # Опора — тело выпуска; совсем короткая папка (доскан из 5 кадров) — целиком.
            body = group if (~group.edge).sum() < 3 else group[~group.edge]
            values = body[name].to_numpy(float)
            center = float(np.median(values))
            scales[group_key] = (center, float(np.median(np.abs(values - center))))
        # Нижняя граница разброса — медиана MAD по всем выпускам: почти постоянный признак не даёт огромных z.
        floor = max(float(np.median([mad for _, mad in scales.values()])), 1e-6)
        for group_key, group in out.groupby(["issue", "side"]):
            center, mad = scales[group_key]
            issue_z[group.index] = (group[name].to_numpy(float) - center) / (1.4826 * max(mad, floor))
        out[f"{name}_iz"] = issue_z
        # Значение другой полосы того же кадра: подменяем сторону и присоединяем по (кадр, сторона).
        other = out[["key", "side", name]].rename(columns={name: "partner"})
        other["side"] = other.side.map({"L": "R", "R": "L"})
        partner = out[["key", "side"]].merge(other, on=["key", "side"], how="left")["partner"].to_numpy()
        out[f"{name}_pair"] = out[name].to_numpy() - partner
    return out


# Читаемость: вырезка полосы — от сгиба (с отступом) наружу на эту ширину, px полного кадра (450 dpi).
READ_SKIP, READ_WIDTH = 150, 1900
# Тело полосы по высоте для чтения (без колонтитула и пальцев у нижнего обреза), доли высоты кадра.
READ_TOP, READ_BOTTOM = 0.15, 0.85
# Слово в счёт — кириллица не короче этого (короткие предлоги словарь знает и в мусоре).
READ_MIN_LETTERS = 4
# Уверенное слово tesseract — conf не ниже этого.
READ_CONF = 80
_CYRILLIC_WORD = re.compile(r"^[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*$")


@dataclass(frozen=True)
class TextQuality:
    """Читаемость текста полосы по tesseract и словарю.

    Attributes:
        words: Число кириллических слов не короче ``READ_MIN_LETTERS`` букв.
        known: Доля этих слов, которые знает словарь pymorphy3.
        confident: Доля этих слов с conf tesseract ≥ ``READ_CONF``.
        conf: Медиана conf по этим словам.
    """

    words: int
    known: float
    confident: float
    conf: float


def read_crop(gray: np.ndarray, fold_x: float, side: str) -> np.ndarray:
    """Вырезка тела полосы для чтения.

    Args:
        gray: Полутоновый кадр полного разрешения.
        fold_x: x сгиба посередине высоты, px.
        side: ``L`` или ``R``.

    Returns:
        Вырезка uint8 (ширина до ``READ_WIDTH``, высота — тело ``READ_TOP``–``READ_BOTTOM``).
    """
    height, width = gray.shape
    y0, y1 = int(height * READ_TOP), int(height * READ_BOTTOM)
    if side == "R":
        x0 = int(fold_x) + READ_SKIP
        return gray[y0:y1, x0 : min(width, x0 + READ_WIDTH)]
    x1 = int(fold_x) - READ_SKIP
    return gray[y0:y1, max(0, x1 - READ_WIDTH) : x1]


def text_quality(gray: np.ndarray, fold_x: float, side: str) -> TextQuality:
    """Читаемость полосы: tesseract (rus, psm 6) по телу полосы и проверка слов словарём.

    Args:
        gray: Полутоновый кадр полного разрешения.
        fold_x: x сгиба посередине высоты, px.
        side: ``L`` или ``R``.

    Returns:
        ``TextQuality``; при пустом чтении — нули и NaN.
    """
    from ocr_utils.rotated_text.tables.ocr import tesseract_tsv
    from research.spread_defects.truncated_words import is_known

    rows = tesseract_tsv(read_crop(gray, fold_x, side), lang="rus", psm=6)
    words, confs = [], []
    for line in rows:
        cells = line.split("\t")
        if len(cells) < 12:
            continue
        text = cells[11].strip(".,;:!?«»()\"'—–")
        if len(text.replace("-", "")) < READ_MIN_LETTERS or not _CYRILLIC_WORD.match(text):
            continue
        words.append(text)
        confs.append(float(cells[10]))
    if not words:
        return TextQuality(0, float("nan"), float("nan"), float("nan"))
    confs = np.array(confs)
    return TextQuality(
        words=len(words),
        known=float(np.mean([is_known(w) for w in words])),
        confident=float(np.mean(confs >= READ_CONF)),
        conf=float(np.median(confs)),
    )


# Признаки ручного судьи в сочетании с эмбеддингами: отклонение от выпуска и от соседней полосы.
HAND_COLUMNS = tuple(f"{name}_iz" for name in FEATURES) + tuple(f"{name}_pair" for name in FEATURES)
# Регуляризация логистической регрессии (вход — 768 признаков DINOv2 + ручные, после стандартизации).
REGULARIZATION = 0.1
# Число складок проверки: выпуски с размеченными копиями раскладываются по разным складкам.
FOLDS = 5


def labeled_pages(out_roots: list[Path], labels: pd.DataFrame) -> pd.DataFrame:
    """Полосы прогонов ``copies`` + ``copies-embed`` с относительными признаками, эмбеддингом и разметкой.

    Args:
        out_roots: Корни выхода стенда (по паку): там ``copies.csv`` и ``copies_embed.npz``.
        labels: Разметка ``sets/photocopy.csv`` (key, side, is_copy).

    Returns:
        Таблица полос: ``pack`` (имя папки корня), ключ, сторона, признаки ``FEATURES`` и ``HAND_COLUMNS``,
        колонки ``e0``…``e767`` эмбеддинга, ``y`` — размечена ли полоса как ксерокопия.
    """
    parts = []
    for root in out_roots:
        # Относительные признаки — внутри пака (выпуски разных паков не смешиваются, но и не пересекаются).
        pages = relative_features(pages_table(pd.read_csv(root / "copies.csv")))
        with np.load(root / "copies_embed.npz") as data:
            embed = pd.DataFrame(data["vector"], columns=[f"e{k}" for k in range(data["vector"].shape[1])])
            embed["key"], embed["side"] = data["key"], data["side"]
        pages = pages.merge(embed, on=["key", "side"])
        pages.insert(0, "pack", root.name)
        parts.append(pages)
    pages = pd.concat(parts, ignore_index=True)
    positive = set(zip(labels.key[labels.is_copy == 1], labels.side[labels.is_copy == 1]))
    pages["y"] = [(k, s) in positive for k, s in zip(pages.key, pages.side)]
    pages[[f"{name}_pair" for name in FEATURES]] = pages[[f"{name}_pair" for name in FEATURES]].fillna(0.0)
    return pages


def out_of_fold_scores(pages: pd.DataFrame) -> np.ndarray:
    """Вероятность «ксерокопия» по каждой полосе от модели, не видевшей выпуска этой полосы.

    Выпуски — единица складки: соседние полосы одной вставки похожи друг на друга, и проверка внутри
    выпуска завышала бы качество. Выпуски с размеченными копиями раскладываются по разным складкам.

    Args:
        pages: Таблица ``labeled_pages``.

    Returns:
        float[len(pages)] — вероятность класса «ксерокопия».
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    columns = [*HAND_COLUMNS, *[c for c in pages.columns if c.startswith("e") and c[1:].isdigit()]]
    issue_keys = pages.pack + "/" + pages.issue
    positive_issues = sorted(issue_keys[pages.y].unique())
    other_issues = sorted(set(issue_keys) - set(positive_issues))
    # Детерминированная раскладка: позитивные выпуски по кругу, затем остальные по кругу.
    fold_of = {issue: k % FOLDS for k, issue in enumerate(positive_issues + other_issues)}
    folds = issue_keys.map(fold_of).to_numpy()
    features = pages[columns].to_numpy(dtype=float)
    scores = np.zeros(len(pages))
    for fold in range(FOLDS):
        test = folds == fold
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=REGULARIZATION, class_weight="balanced", max_iter=5000)
        )
        model.fit(features[~test], pages.y.to_numpy()[~test])
        scores[test] = model.predict_proba(features[test])[:, 1]
    return scores


def page_numbers(pages_dir: Path) -> pd.DataFrame:
    """Номера полос кадров из прогона ``pages``: прочитанный номер, иначе выведенный по соседям.

    Args:
        pages_dir: Папка прогона ``pages`` (``frames.csv``).

    Returns:
        Таблица ``key, side, page`` (page — int или NaN).
    """
    from research.spread_defects.sequence import frame_numbers, offset_candidates

    frames = pd.read_csv(pages_dir / "frames.csv").sort_values(["issue", "index", "key"])
    rows = []
    for _, group in frames.groupby("issue"):
        group = group.reset_index(drop=True)
        candidates = [
            offset_candidates(k, None if pd.isna(l) else int(l), None if pd.isna(r) else int(r))
            for k, (l, r) in enumerate(zip(group.left, group.right))
        ]
        inferred = frame_numbers(candidates)
        for k, row in group.iterrows():
            left = inferred[k] if inferred[k] is not None else (None if pd.isna(row.left) else int(row.left))
            rows.append({"key": row.key, "side": "L", "page": left})
            rows.append({"key": row.key, "side": "R", "page": None if left is None else left + 1})
    return pd.DataFrame(rows)


def copy_regions(scored: pd.DataFrame, threshold: float, extend: float) -> pd.DataFrame:
    """Участки ксерокопий: подряд идущие полосы выпуска с оценкой не ниже порога.

    Гистерезис: участок — подряд идущие полосы с оценкой не ниже ``extend``, в котором хотя бы одна полоса
    не ниже ``threshold``. Соседние полосы одной вставки модель оценивает очень по-разному (таблица, формулы,
    палец сбивают оценку), поэтому ядро ищется по строгому порогу, а края — по мягкому. Порядок полос —
    кадры по номеру, в кадре левая, затем правая. Одна полоса ниже ``extend`` внутри участка его не рвёт.

    Args:
        scored: Полосы с колонками ``pack, issue, index, key, side, score`` и ``page`` (может быть NaN).
        threshold: Порог ядра участка.
        extend: Порог расширения участка на соседние полосы.

    Returns:
        Таблица участков: pack, issue, first_key, last_key, pages (число полос), page_from, page_to,
        score_median, edge (участок целиком в первых/последних ``EDGE_FRAMES`` кадрах).
    """
    rows = []
    for (pack, issue), group in scored.groupby(["pack", "issue"]):
        group = group.sort_values(["index", "key", "side"]).reset_index(drop=True)
        hit = (group.score >= extend).to_numpy()
        # Затягиваем одиночные провалы между двумя срабатываниями.
        bridged = hit.copy()
        bridged[1:-1] |= hit[:-2] & hit[2:]
        start = None
        for k in range(len(group) + 1):
            if k < len(group) and bridged[k]:
                start = k if start is None else start
                continue
            if start is not None:
                part = group.iloc[start:k]
                if (part.score < threshold).all():
                    start = None
                    continue
                rows.append(
                    {
                        "pack": pack,
                        "issue": issue,
                        "first_key": f"{part.key.iloc[0]}:{part.side.iloc[0]}",
                        "last_key": f"{part.key.iloc[-1]}:{part.side.iloc[-1]}",
                        "pages": len(part),
                        "page_from": part.page.min(),
                        "page_to": part.page.max(),
                        "score_median": round(float(part.score.median()), 3),
                        # Участок целиком в краю выпуска (обложки, реклама) — кандидат в ложные.
                        "edge": bool(part.edge.all()) if "edge" in part else False,
                    }
                )
                start = None
    return pd.DataFrame(rows)
