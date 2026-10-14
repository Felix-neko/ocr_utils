"""Сегментация бумаги внутри маски книги: цветовая модель, GrabCut, SAM2 по точкам, SAM 3 по тексту, DINOv2 + kNN."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.book import disk
from ocr_utils.scan_cropping.v2.paper import (  # noqa: F401 — ядро, ре-экспорт для стенда
    CLOSE_PX,
    INK_DROP,
    L_MARGIN,
    MAHAL_THR,
    PAPER_BG_PX,
    SAM3_MIN_PRINT,
    SMOOTH_PX,
    PaperModel,
    paper_seeds,
    postprocess_plain,
    smooth_lab,
)

# Затравки «не бумаги»: полоса у края книги шириной EDGE_BAND px с цветом дальше NONPAPER_DIST от модели бумаги.
EDGE_BAND = 12
NONPAPER_DIST = 6.0


class PaperMethod(str, Enum):
    """Способ сегментации бумаги."""

    LAB = "lab_paper"
    GRABCUT = "grabcut_paper"
    SAM2 = "sam2_paper"
    SAM3 = "sam3_page"
    DINO = "dino_knn"
    VOTE = "vote"






def nonpaper_seeds(lab: np.ndarray, book: np.ndarray, model: PaperModel) -> np.ndarray:
    """Затравки «не бумаги»: полоса у края книги, далёкая по оттенку от модели бумаги.

    Args:
        lab: Кадр Lab.
        book: Маска книги.
        model: Модель бумаги.

    Returns:
        Булева маска.
    """
    band = book & ~(cv2.erode(book.astype(np.uint8), disk(EDGE_BAND)) > 0)
    return band & (model.distance(lab) > NONPAPER_DIST)


def finish(paper: np.ndarray, book: np.ndarray, hulls: np.ndarray) -> np.ndarray:
    """Чистка маски бумаги: внутри книги, закрытие, компоненты, связанные с печатью, заливка дыр, + оболочки печати.

    Args:
        paper: Сырая маска бумаги.
        book: Маска книги.
        hulls: Оболочки своей печати.

    Returns:
        Маска бумаги.
    """
    paper = paper & book
    paper = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_CLOSE, disk(CLOSE_PX)) > 0
    paper = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_OPEN, disk(3)) > 0
    labels, _ = ndimage.label(paper | hulls)
    keep = np.unique(labels[hulls & (labels > 0)])
    if len(keep) == 0:
        return paper & book
    paper = np.isin(labels, keep)
    return ndimage.binary_fill_holes(paper) & book


def paper_lab(bgr: np.ndarray, book: np.ndarray, hulls: np.ndarray) -> np.ndarray:
    """Бумага по цветовой модели из затравок.

    Args:
        bgr: Кадр BGR (½).
        book: Маска книги.
        hulls: Оболочки своей печати.

    Returns:
        Маска бумаги.
    """
    lab = smooth_lab(bgr)
    model = PaperModel(lab, paper_seeds(bgr, hulls, book))
    return finish(model.is_paper(lab), book, hulls)


def paper_grabcut(bgr: np.ndarray, book: np.ndarray, hulls: np.ndarray, iters: int = 4) -> np.ndarray:
    """Бумага через GrabCut: вне книги и «не бумага» у края — фон, затравки бумаги — объект, остальное — по цвету.

    Args:
        bgr: Кадр BGR.
        book: Маска книги.
        hulls: Оболочки своей печати.
        iters: Итераций.

    Returns:
        Маска бумаги.
    """
    lab = smooth_lab(bgr)
    seeds = paper_seeds(bgr, hulls, book)
    model = PaperModel(lab, seeds)
    rough = model.is_paper(lab) & book
    mask = np.where(rough, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    mask[~book] = cv2.GC_BGD
    mask[nonpaper_seeds(lab, book, model)] = cv2.GC_BGD
    mask[seeds | hulls] = cv2.GC_FGD
    ys, xs = np.nonzero(book)
    pad = 10
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
    sub = mask[y0:y1, x0:x1].copy()
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(np.ascontiguousarray(bgr[y0:y1, x0:x1]), sub, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    out = np.zeros(book.shape, bool)
    out[y0:y1, x0:x1] = (sub == cv2.GC_FGD) | (sub == cv2.GC_PR_FGD)
    return finish(out, book, hulls)


def sample_points(mask: np.ndarray, n: int, rng: np.random.Generator) -> list[list[float]]:
    """Случайные точки маски.

    Args:
        mask: Маска.
        n: Сколько.
        rng: Генератор.

    Returns:
        ``[[x, y], ...]``.
    """
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return []
    pick = pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]
    return [[float(x), float(y)] for y, x in pick]


def paper_sam2(bgr: np.ndarray, book: np.ndarray, hulls: np.ndarray, models) -> np.ndarray:
    """Бумага через SAM2.1: бокс по книге, + точки на затравках бумаги, − на «не бумаге» у края; лучший кандидат.

    Args:
        bgr: Кадр BGR.
        book: Маска книги.
        hulls: Оболочки печати.
        models: ``geometry.GeometryModels``.

    Returns:
        Маска бумаги.
    """
    h, w = book.shape
    scale = 1024 / max(h, w)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = smooth_lab(bgr)
    seeds = paper_seeds(bgr, hulls, book)
    model = PaperModel(lab, seeds)
    neg = nonpaper_seeds(lab, book, model)
    rng = np.random.default_rng(0)
    pos_pts = sample_points(seeds, 18, rng)
    neg_pts = sample_points(neg, 10, rng)
    points = [[x * scale, y * scale] for x, y in pos_pts + neg_pts]
    labels = [1] * len(pos_pts) + [0] * len(neg_pts)
    ys, xs = np.nonzero(book)
    box = [xs.min() * scale, ys.min() * scale, xs.max() * scale, ys.max() * scale]
    masks, _ = models.sam_masks(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), points, labels, box)
    best, best_score = None, -np.inf
    for m in masks:
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR) > 0
        score = (m & seeds).sum() / max(seeds.sum(), 1) - 3 * (m & neg).sum() / max(neg.sum(), 1)
        if score > best_score:
            best, best_score = m, score
    return finish(best, book, hulls)


class Sam3Page:
    """SAM 3 по тексту «book page» / «paper page» (1008 px): экземпляры, накрывающие свою печать."""

    PROMPTS = ("book page", "paper page")

    def __init__(self, weights: str = "finger_models/sam3/sam3.pt") -> None:
        """Грузит модель.

        Args:
            weights: Путь к весам.
        """
        from ultralytics.models.sam import SAM3SemanticPredictor

        from research.bound_volumes.background import sam3_overrides

        self.predictor = SAM3SemanticPredictor(overrides=sam3_overrides(weights))

    def __call__(self, bgr: np.ndarray, book: np.ndarray, hulls: np.ndarray) -> np.ndarray:
        """Маска бумаги.

        Args:
            bgr: Кадр BGR.
            book: Маска книги.
            hulls: Оболочки печати.

        Returns:
            Маска бумаги.
        """
        import torch

        h, w = book.shape
        out = np.zeros((h, w), bool)
        with torch.inference_mode():
            self.predictor.set_image(bgr)
            for prompt in self.PROMPTS:
                results = self.predictor(text=[prompt])
                if not results or results[0].masks is None:
                    continue
                for m in results[0].masks.data.cpu().numpy() > 0.5:
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    if (m & hulls).sum() >= 0.05 * max(hulls.sum(), 1):
                        out |= m
        torch.cuda.empty_cache()
        return finish(out, book, hulls)


class DinoKnn:
    """DINOv2-base: признаки патчей кадра, логистическая регрессия на затравках этого кадра, уточнение по цвету."""

    SIDE = 1022  # кратно патчу 14

    def __init__(self, device: str = "cuda") -> None:
        """Грузит модель.

        Args:
            device: Устройство torch.
        """
        import torch
        from transformers import AutoModel

        self.device = device
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval().half()
        self.torch = torch

    def __call__(self, bgr: np.ndarray, book: np.ndarray, hulls: np.ndarray) -> np.ndarray:
        """Маска бумаги.

        Args:
            bgr: Кадр BGR.
            book: Маска книги.
            hulls: Оболочки печати.

        Returns:
            Маска бумаги.
        """
        from sklearn.linear_model import LogisticRegression

        torch = self.torch
        h, w = book.shape
        side = self.SIDE
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (side, side), interpolation=cv2.INTER_AREA)
        x = (rgb.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float16, device=self.device)
        with torch.inference_mode():
            feats = self.model(pixel_values=t).last_hidden_state[0, 1:].float().cpu().numpy()
        g = side // 14
        feats = feats.reshape(g, g, -1)
        lab = smooth_lab(bgr)
        seeds = paper_seeds(bgr, hulls, book)
        model = PaperModel(lab, seeds)
        neg = nonpaper_seeds(lab, book, model) | (cv2.dilate((~book).astype(np.uint8), disk(3)) > 0) & ~book

        def grid(mask: np.ndarray) -> np.ndarray:
            return cv2.resize(mask.astype(np.float32), (g, g), interpolation=cv2.INTER_AREA)

        pos_g, neg_g = grid(seeds) > 0.3, grid(neg) > 0.3
        X = np.r_[feats[pos_g], feats[neg_g]]
        y = np.r_[np.ones(pos_g.sum()), np.zeros(neg_g.sum())]
        if len(set(y)) < 2:
            return finish(model.is_paper(lab), book, hulls)
        clf = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced").fit(X, y)
        prob = clf.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1].reshape(g, g)
        prob = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        coarse = prob > 0.5
        # Уточнение границы патча: в полосе ±14 px (в ½) решает цветовая модель.
        band = (cv2.dilate(coarse.astype(np.uint8), disk(14)) > 0) & ~(cv2.erode(coarse.astype(np.uint8), disk(14)) > 0)
        paper = np.where(band, model.is_paper(lab), coarse)
        return finish(paper, book, hulls)


# --- Доводка маски SAM 3: сырой выход, «полная печать», заливка вогнутостей --------------------------------------

# Краска для «полной печати»: окно и сдвиг адаптивного порога (½ кадра), пределы площади компоненты, px².
INK_BLOCK, INK_OFFSET = 31, 18
INK_MIN_AREA, INK_MAX_AREA = 6, 6000
# Краска берётся, если лежит не дальше этого от сырой маски SAM 3, px (~9 мм): номер страницы в углу, сноска.
INK_REACH = 60
# Раздутие краски перед включением в бумагу, px.
INK_PAD = 6
# Вогнутость у края заливается, если её глубина (до выпуклой оболочки страницы) не больше этого, px (~6 мм).
CONCAVITY_DEPTH = 40


class PostVariant(str, Enum):
    """Постобработка сырой маски SAM 3."""

    # Как было: finish() с насильным включением выпуклых оболочек печати.
    HULLS = "hulls"
    # Без оболочек: закрытие, открытие, компоненты с печатью, заливка дыр.
    PLAIN = "plain"
    # PLAIN + «полная печать»: вся краска рядом со страницей (номера страниц, сноски).
    INK = "ink"
    # INK + заливка неглубоких вогнутостей у края.
    INK_CONCAVE = "ink_concave"


def sam3_raw(predictor, bgr: np.ndarray, hulls: np.ndarray, prompts=("book page", "paper page")) -> np.ndarray:
    """Сырое объединение экземпляров SAM 3, накрывающих не меньше :data:`SAM3_MIN_PRINT` печати.

    Args:
        predictor: ``SAM3SemanticPredictor``.
        bgr: Кадр BGR (½).
        hulls: Оболочки печати (для отбора экземпляров).
        prompts: Текстовые подсказки.

    Returns:
        Булева маска.
    """
    import torch

    h, w = hulls.shape
    out = np.zeros((h, w), bool)
    with torch.inference_mode():
        predictor.set_image(bgr)
        for prompt in prompts:
            results = predictor(text=[prompt])
            if not results or results[0].masks is None:
                continue
            for m in results[0].masks.data.cpu().numpy() > 0.5:
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if (m & hulls).sum() >= SAM3_MIN_PRINT * max(hulls.sum(), 1):
                    out |= m
    torch.cuda.empty_cache()
    return out


def ink_components(bgr: np.ndarray, book: np.ndarray) -> np.ndarray:
    """Вся краска книги размера знака: адаптивный порог, компоненты с площадью в пределах, без линий торца.

    Args:
        bgr: Кадр BGR (½).
        book: Маска книги.

    Returns:
        Булева маска краски.
    """
    from research.bound_volumes.stripes import drop_edge_lines

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, INK_BLOCK, INK_OFFSET) > 0
    ink &= book
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    keep = (area >= INK_MIN_AREA) & (area <= INK_MAX_AREA)
    keep[0] = False
    return drop_edge_lines(keep[labels])


def fill_concavities(mask: np.ndarray, depth: int = CONCAVITY_DEPTH) -> np.ndarray:
    """Заливка неглубоких вогнутостей у края: по каждой компоненте «выпуклая оболочка − маска», части глубиной ≤ depth.

    Args:
        mask: Маска бумаги.
        depth: Предельная глубина, px.

    Returns:
        Маска с залитыми вогнутостями.
    """
    out = mask.copy()
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    for i in range(1, count):
        comp = labels == i
        pts = cv2.findNonZero(comp.astype(np.uint8))
        hull = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(hull, [cv2.convexHull(pts)], 1)
        gap = (hull > 0) & ~comp
        # Глубина вогнутости — расстояние от пикселя выемки до края выпуклой оболочки.
        dist = cv2.distanceTransform(hull, cv2.DIST_L2, 5)
        n, gl = cv2.connectedComponents(gap.astype(np.uint8))
        for j in range(1, n):
            part = gl == j
            if dist[part].max() <= depth:
                out |= part
    return out


def page_hulls(mask: np.ndarray, min_area: int = 20000) -> np.ndarray:
    """Выпуклые оболочки крупных компонент маски (по странице).

    Args:
        mask: Маска.
        min_area: Компоненты меньше этого, px², не берутся.

    Returns:
        Маска оболочек.
    """
    out = np.zeros(mask.shape, np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cv2.fillPoly(out, [cv2.convexHull(cv2.findNonZero((labels == i).astype(np.uint8)))], 1)
    return out > 0


def postprocess_sam3(raw: np.ndarray, bgr: np.ndarray, book: np.ndarray, hulls: np.ndarray, variant: PostVariant) -> np.ndarray:
    """Маска бумаги из сырой маски SAM 3 в заданном варианте.

    Args:
        raw: Сырая маска (``sam3_raw``).
        bgr: Кадр BGR (½).
        book: Маска книги.
        hulls: Оболочки печати.
        variant: Вариант.

    Returns:
        Маска бумаги.
    """
    if variant == PostVariant.HULLS:
        return finish(raw, book, hulls)
    if variant == PostVariant.PLAIN:
        # Боевой вариант — в ядре scan_cropping v2.
        return postprocess_plain(raw, book, hulls)
    paper = raw & book
    paper = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_CLOSE, disk(CLOSE_PX)) > 0
    paper = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_OPEN, disk(3)) > 0
    if variant in (PostVariant.INK, PostVariant.INK_CONCAVE):
        ink = ink_components(bgr, book)
        # Краска берётся только внутри выпуклой оболочки каждой страницы SAM 3: номер страницы в скруглённом углу
        # туда попадает, а штрихи торца стопки и обложки (снаружи страницы) — нет.
        hull = page_hulls(paper)
        paper |= (cv2.dilate((ink & hull).astype(np.uint8), disk(INK_PAD)) > 0) & hull & book
        paper = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_CLOSE, disk(CLOSE_PX)) > 0
    # Компоненты, задевающие печать.
    labels, _ = ndimage.label(paper)
    keep = np.unique(labels[hulls & paper])
    paper = np.isin(labels, keep[keep > 0]) if len(keep) else paper
    paper = ndimage.binary_fill_holes(paper)
    if variant == PostVariant.INK_CONCAVE:
        paper = fill_concavities(paper) & book
    return paper & book


def sam3_instances(predictor, bgr: np.ndarray, hulls: np.ndarray, prompts=("book page", "paper page")) -> np.ndarray:
    """Экземпляры SAM 3 (страницы) картой меток: 0 — ничего, 1..N — экземпляры, накрывающие печать.

    Экземпляры двух подсказок, совпадающие на ≥80% (IoU), считаются одной страницей; пересечения отдаются экземпляру
    с большей оценкой.

    Args:
        predictor: ``SAM3SemanticPredictor``.
        bgr: Кадр BGR (½).
        hulls: Своя печать (отбор экземпляров).
        prompts: Подсказки.

    Returns:
        Карта меток uint8.
    """
    import torch

    h, w = hulls.shape
    found: list[tuple[float, np.ndarray]] = []
    with torch.inference_mode():
        predictor.set_image(bgr)
        for prompt in prompts:
            results = predictor(text=[prompt])
            if not results or results[0].masks is None:
                continue
            scores = results[0].boxes.conf.cpu().numpy()
            for m, sc in zip(results[0].masks.data.cpu().numpy() > 0.5, scores):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if (m & hulls).sum() < SAM3_MIN_PRINT * max(hulls.sum(), 1):
                    continue
                dup = [k for k, (_, o) in enumerate(found) if (m & o).sum() / max((m | o).sum(), 1) >= 0.8]
                if dup:
                    k = dup[0]
                    found[k] = (max(found[k][0], float(sc)), found[k][1] | m)
                else:
                    found.append((float(sc), m))
    torch.cuda.empty_cache()
    labels = np.zeros((h, w), np.uint8)
    for k, (_, m) in sorted(enumerate(found), key=lambda t: t[1][0]):
        labels[m] = k + 1
    return labels
