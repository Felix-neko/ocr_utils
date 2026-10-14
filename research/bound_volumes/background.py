"""Сегментация фона (тёмный коврик) вокруг подшивки: несколько способов, маска рук; книга = не фон − руки."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.background import (  # noqa: F401 — ядро, ре-экспорт для стенда
    BG_SCALE,
    BORDER_PX,
    MAT_SAMPLE_MAX_GRAY,
    MAT_MAX_CHROMA,
    MAT_TAIL_PCT,
    MAT_TAIL_MARGIN,
    MIN_HOLE_PX,
    CLOSE_PX,
    SKIN_CR,
    SKIN_MIN_CR_MINUS_CB,
    SKIN_CB,
    SKIN_MIN_Y,
    HAND_CORE_DILATE_PX,
    border_band,
    chroma,
    mat_seeds,
    book_seeds,
    hand_mask,
    clean_background,
    bg_dark,
    bg_flood,
    book_from_background,
)



class BgMethod(str, Enum):
    """Способ сегментации фона."""

    # Порог яркости (Оцу по образцам у рамки и середине кадра) + хрома, связность с рамкой.
    DARK = "dark"
    # Цветовая модель коврика + GrabCut с затравками «полоса у рамки» и «печать».
    GRABCUT = "gmm_grabcut"
    # Водораздел по модулю градиента с маркерами «полоса у рамки» и «печать».
    FLOOD = "flood"
    # SAM2.1: положительные точки на коврике, отрицательные — на печати.
    SAM2 = "sam2_bg"
    # Большинство способов DARK, GRABCUT, FLOOD, SAM2.
    VOTE = "vote"
















def bg_grabcut(bgr: np.ndarray, content: np.ndarray, iters: int = 4) -> np.ndarray:
    """Фон через GrabCut: коврик у рамки — фон наверняка, печать — книга наверняка, остальное — «вероятно».

    Начальное «вероятно фон / вероятно книга» — по порогу ``bg_dark``, так GMM стартует с разумного разбиения.

    Args:
        bgr: Кадр BGR.
        content: Своя печать (bool).
        iters: Итераций GrabCut.

    Returns:
        Маска фона.
    """
    rough = bg_dark(bgr, content)
    mask = np.where(rough, cv2.GC_PR_BGD, cv2.GC_PR_FGD).astype(np.uint8)
    mask[mat_seeds(bgr)] = cv2.GC_BGD
    mask[book_seeds(content)] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    return clean_background((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD))




def bg_sam2(bgr: np.ndarray, content: np.ndarray, sam) -> np.ndarray:
    """Фон через SAM2.1: точки на коврике (+), на печати (−), без бокса; маска — объединение кандидатов-фона.

    Args:
        bgr: Кадр BGR.
        content: Своя печать (bool).
        sam: ``geometry.GeometryModels`` (нужен ``sam_masks``).

    Returns:
        Маска фона.
    """
    h, w = content.shape
    scale = 1024 / max(h, w)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mat = cv2.resize(mat_seeds(bgr).astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    books = cv2.resize(
        book_seeds(content).astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    rng = np.random.default_rng(0)
    pos = np.argwhere(mat > 0)
    neg = np.argwhere(books > 0)
    pos = pos[rng.choice(len(pos), min(12, len(pos)), replace=False)] if len(pos) else pos
    neg = neg[rng.choice(len(neg), min(6, len(neg)), replace=False)] if len(neg) else neg
    points = [[float(x), float(y)] for y, x in pos] + [[float(x), float(y)] for y, x in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    masks, scores = sam.sam_masks(rgb, points, labels, None)
    # Кандидат, который лучше всех покрывает затравки коврика и меньше всех — печать.
    cover = [(m & (mat > 0)).sum() / max((mat > 0).sum(), 1) - 5 * (m & (books > 0)).sum() / max((books > 0).sum(), 1)
             for m in masks]  # fmt: skip
    best = masks[int(np.argmax(cover))]
    bg = cv2.resize(best.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR) > 0
    return clean_background(bg)




def bg_inputs(frame, source: str = "glyph") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Входы сегментации кадра в масштабе :data:`BG_SCALE`: кадр без пальцев, подложка оверлея, печать, ядра пальцев.

    Args:
        frame: Кадр стенда.
        source: Вариант закраса пальцев (``glyph``, ``inkback`` …). Для варианта, отличного от ``glyph``, подложка
            оверлея — тот же кадр после закраса, а ядра пальцев не отдаются (пальцы уже закрашены; руки ищутся по
            коже на этом кадре).

    Returns:
        ``(без пальцев BGR, подложка BGR, печать bool, ядра пальцев bool)``.
    """
    from research.bound_volumes.fingers import Guard, out_paths
    from research.bound_volumes.frames import OUT_ROOT, read_bgr

    clean = read_bgr(out_paths(Guard(source), frame)[0])
    img = cv2.resize(clean, None, fx=BG_SCALE, fy=BG_SCALE, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if source != Guard.GLYPH.value:
        return img, img, content_mask(frame, (h, w)), np.zeros((h, w), bool)
    src = cv2.resize(read_bgr(frame.work_path), (w, h), interpolation=cv2.INTER_AREA)
    content = cv2.imread(str(OUT_ROOT / "geometry" / "content" / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE)
    content = cv2.resize(content, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    core = cv2.imread(str(out_paths(Guard.GLYPH, frame)[1]) + "_core.png", cv2.IMREAD_GRAYSCALE)
    core = np.zeros((h, w), bool) if core is None else cv2.resize(core, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return img, src, content, core


def content_mask(frame, shape: tuple[int, int]) -> np.ndarray:
    """Своя печать кадра (кэш защиты глифов) в размере ``shape``.

    Args:
        frame: Кадр.
        shape: ``(H, W)``.

    Returns:
        Булева маска.
    """
    from research.bound_volumes.frames import OUT_ROOT

    content = cv2.imread(str(OUT_ROOT / "geometry" / "content" / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(content, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


# Способы, считаемые на CPU (в пул), и их функции.
CPU_METHODS = {BgMethod.DARK: bg_dark, BgMethod.GRABCUT: bg_grabcut, BgMethod.FLOOD: bg_flood}


# SAM 3: текстовые подсказки книги; экземпляр берётся, если накрывает не меньше SAM3_MIN_PRINT своей печати.
SAM3_PROMPTS = ("open book", "book")
SAM3_MIN_PRINT = 0.05


class Sam3Book:
    """SAM 3 (ultralytics ``SAM3SemanticPredictor``) по текстовым подсказкам книги; фон = не книга."""

    def __init__(self, weights: str = "finger_models/sam3/sam3.pt") -> None:
        """Грузит модель.

        Args:
            weights: Путь к ``sam3.pt``.
        """
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(conf=0.25, task="segment", mode="predict", model=weights, half=True, verbose=False, save=False)
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def __call__(self, bgr: np.ndarray, content: np.ndarray) -> np.ndarray:
        """Маска фона: всё, что не входит в экземпляры «книга», накрывающие свою печать.

        Args:
            bgr: Кадр BGR (масштаб сегментации).
            content: Своя печать (bool).

        Returns:
            Маска фона (без чистки: при пустом выходе SAM 3 — фон весь кадр, кроме печати).
        """
        h, w = content.shape
        self.predictor.set_image(bgr)
        results = self.predictor(text=list(SAM3_PROMPTS))
        book = np.zeros((h, w), bool)
        if results and results[0].masks is not None:
            for m in results[0].masks.data.cpu().numpy() > 0.5:
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if (m & content).sum() >= SAM3_MIN_PRINT * max(content.sum(), 1):
                    book |= m
        return clean_background(~(book | content))


# SAM 3 по фону: текстовые подсказки коврика; экземпляр отбрасывается, если накрывает больше SAM3_BG_MAX_PRINT печати.
SAM3_BG_PROMPTS = ("black background", "black fabric", "dark table surface")
SAM3_BG_MAX_PRINT = 0.01


class Sam3Background(Sam3Book):
    """SAM 3 по текстовым подсказкам фона (чёрный коврик); фон = объединение экземпляров без своей печати."""

    def __call__(self, bgr: np.ndarray, content: np.ndarray) -> np.ndarray:
        """Маска фона: объединение экземпляров «чёрный фон / ткань / стол», не накрывающих печать.

        Args:
            bgr: Кадр BGR (масштаб сегментации).
            content: Своя печать (bool).

        Returns:
            Маска фона.
        """
        h, w = content.shape
        self.predictor.set_image(bgr)
        results = self.predictor(text=list(SAM3_BG_PROMPTS))
        bg = np.zeros((h, w), bool)
        if results and results[0].masks is not None:
            for m in results[0].masks.data.cpu().numpy() > 0.5:
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if (m & content).sum() <= SAM3_BG_MAX_PRINT * max(content.sum(), 1):
                    bg |= m
        return clean_background(bg & ~content)


# Честное сравнение SAM 3: родное разрешение SAM 3 (1008 px, кратно патчу 14) вместо 640 по умолчанию ultralytics.
SAM3_IMGSZ = 1008


def sam3_overrides(weights: str) -> dict:
    """Настройки предикторов SAM 3 ultralytics.

    Args:
        weights: Путь к ``sam3.pt``.

    Returns:
        Словарь ``overrides``.
    """
    return dict(conf=0.25, task="segment", mode="predict", model=weights, half=True, verbose=False, save=False,
                imgsz=SAM3_IMGSZ)  # fmt: skip


class Sam3TextBg:
    """SAM 3 по тексту фона (как ``Sam3Background``), но в родном разрешении 1008."""

    def __init__(self, weights: str = "finger_models/sam3/sam3.pt") -> None:
        """Грузит модель.

        Args:
            weights: Путь к весам.
        """
        from ultralytics.models.sam import SAM3SemanticPredictor

        self.predictor = SAM3SemanticPredictor(overrides=sam3_overrides(weights))

    def __call__(self, bgr: np.ndarray, content: np.ndarray) -> np.ndarray:
        """Маска фона по тексту.

        Args:
            bgr: Кадр BGR.
            content: Своя печать.

        Returns:
            Маска фона.
        """
        h, w = content.shape
        self.predictor.set_image(bgr)
        bg = np.zeros((h, w), bool)
        for prompt in SAM3_BG_PROMPTS:
            results = self.predictor(text=[prompt])
            if not results or results[0].masks is None:
                continue
            for m in results[0].masks.data.cpu().numpy() > 0.5:
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if (m & content).sum() <= SAM3_BG_MAX_PRINT * max(content.sum(), 1):
                    bg |= m
        return clean_background(bg & ~content)


def sam_point_prompts(bgr: np.ndarray, content: np.ndarray, seed: int = 0) -> tuple[list[list[float]], list[int]]:
    """Те же точечные подсказки, что у ``bg_sam2``: 12 точек на коврике (+), 6 на печати (−), в px кадра.

    Args:
        bgr: Кадр BGR.
        content: Своя печать.
        seed: Зерно выбора точек.

    Returns:
        ``(точки, метки)``.
    """
    rng = np.random.default_rng(seed)
    pos = np.argwhere(mat_seeds(bgr))
    neg = np.argwhere(book_seeds(content))
    pos = pos[rng.choice(len(pos), min(12, len(pos)), replace=False)] if len(pos) else pos
    neg = neg[rng.choice(len(neg), min(6, len(neg)), replace=False)] if len(neg) else neg
    points = [[float(x), float(y)] for y, x in pos] + [[float(x), float(y)] for y, x in neg]
    return points, [1] * len(pos) + [0] * len(neg)


class Sam3Points:
    """SAM 3 в интерактивном режиме (``SAM3Predictor``) с теми же точками, что у SAM 2, и тем же выбором маски."""

    def __init__(self, weights: str = "finger_models/sam3/sam3.pt") -> None:
        """Грузит модель.

        Args:
            weights: Путь к весам.
        """
        from ultralytics.models.sam import SAM3Predictor

        self.predictor = SAM3Predictor(overrides=sam3_overrides(weights))

    def __call__(self, bgr: np.ndarray, content: np.ndarray) -> np.ndarray:
        """Маска фона по точкам.

        Args:
            bgr: Кадр BGR.
            content: Своя печать.

        Returns:
            Маска фона.
        """
        h, w = content.shape
        import torch

        points, labels = sam_point_prompts(bgr, content)
        # Без inference_mode и сброса признаков кадра интерактивный SAM 3 копит граф и падает по памяти на 2-м кадре.
        with torch.inference_mode():
            self.predictor.set_image(bgr)
            results = self.predictor(points=[points], labels=[labels], multimask_output=True)
            masks = results[0].masks.data.cpu().numpy() > 0.5
        self.predictor.reset_image()
        torch.cuda.empty_cache()
        mat, books = mat_seeds(bgr), book_seeds(content)
        best, best_score = None, -np.inf
        for m in masks:
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            score = (m & mat).sum() / max(mat.sum(), 1) - 5 * (m & books).sum() / max(books.sum(), 1)
            if score > best_score:
                best, best_score = m, score
        return clean_background(best)
