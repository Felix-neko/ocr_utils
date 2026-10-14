"""Бумага внутри книги: SAM 3 по тексту «book page» / «paper page» (1288 px), постобработка ``plain``, цветовая модель бумаги."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.gpu_models import MODELS_DIR
from ocr_utils.scan_cropping.v2.book import disk

# Веса SAM 3 (ultralytics ``SAM3SemanticPredictor``): закрытые веса взяты с зеркала ModelScope со сверкой SHA
# (``research/newspaper_crop/README.md``).
SAM3_WEIGHTS = MODELS_DIR / "sam3" / "sam3.pt"
# Сторона входа SAM 3, px: 1288 (кратно патчу 14) лучше родных 1008 держит кромку страницы (стенд подшивок).
SAM3_IMGSZ = 1288
# Порог уверенности экземпляра SAM 3.
SAM3_CONF = 0.25
# Текстовые подсказки страницы.
SAM3_PROMPTS = ("book page", "paper page")
# Порог доли печати, накрытой экземпляром SAM 3, чтобы взять его в бумагу.
SAM3_MIN_PRINT = 0.05
# Экземпляры двух подсказок с IoU не меньше этого — одна страница.
SAM3_SAME_IOU = 0.8
# Затравки бумаги: пиксели внутри оболочек своей печати, не краска (серое выше локального фона минус INK_DROP).
INK_DROP = 25
# Окно локального фона бумаги (серое закрытие), px масштаба сегментации.
PAPER_BG_PX = 15
# Порог расстояния Махаланобиса (в ab Lab) для бумаги и запас по яркости ниже квантиля бумаги (тень изгиба).
MAHAL_THR = 4.0
L_MARGIN = 70
# Сглаживание кадра перед цветовой моделью (медиана), px — гасит краску и шум.
SMOOTH_PX = 7
# Закрытие маски бумаги, px.
CLOSE_PX = 9


def sam3_overrides(weights: str | Path = SAM3_WEIGHTS, imgsz: int = SAM3_IMGSZ) -> dict:
    """Настройки предиктора SAM 3 ultralytics.

    Args:
        weights: Путь к ``sam3.pt``.
        imgsz: Сторона входа сети, px.

    Returns:
        Словарь ``overrides``.
    """
    return dict(conf=SAM3_CONF, task="segment", mode="predict", model=str(weights), half=True, verbose=False,
                save=False, imgsz=imgsz)  # fmt: skip


def load_sam3(weights: str | Path = SAM3_WEIGHTS, imgsz: int = SAM3_IMGSZ):
    """Предиктор SAM 3 по тексту (грузится один раз на прогон).

    Args:
        weights: Путь к ``sam3.pt``.
        imgsz: Сторона входа сети, px.

    Returns:
        ``SAM3SemanticPredictor``.
    """
    from ultralytics.models.sam import SAM3SemanticPredictor

    return SAM3SemanticPredictor(overrides=sam3_overrides(weights, imgsz))


def sam3_pages(predictor, bgr: np.ndarray, hulls: np.ndarray, prompts=SAM3_PROMPTS) -> tuple[np.ndarray, np.ndarray]:
    """Страницы SAM 3 одним проходом: сырое объединение экземпляров и карта меток экземпляров-страниц.

    Берутся экземпляры, накрывающие не меньше :data:`SAM3_MIN_PRINT` своей печати. В карте меток экземпляры двух
    подсказок, совпадающие на ≥ :data:`SAM3_SAME_IOU` (IoU), считаются одной страницей; пересечения отдаются
    экземпляру с большей оценкой. Это то же, что ``sam3_raw`` и ``sam3_instances`` стенда подшивок по отдельности,
    но кадр кодируется и подсказки прогоняются один раз.

    Args:
        predictor: ``SAM3SemanticPredictor``.
        bgr: Кадр BGR (масштаб сегментации).
        hulls: Оболочки своей печати (отбор экземпляров).
        prompts: Текстовые подсказки.

    Returns:
        ``(сырая маска bool, карта меток uint8: 0 — ничего, 1..N — страницы)``.
    """
    import torch

    h, w = hulls.shape
    raw = np.zeros((h, w), bool)
    found: list[tuple[float, np.ndarray]] = []
    need = SAM3_MIN_PRINT * max(hulls.sum(), 1)
    with torch.inference_mode():
        predictor.set_image(bgr)
        for prompt in prompts:
            results = predictor(text=[prompt])
            if not results or results[0].masks is None:
                continue
            scores = results[0].boxes.conf.cpu().numpy()
            for m, sc in zip(results[0].masks.data.cpu().numpy() > 0.5, scores):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if (m & hulls).sum() < need:
                    continue
                raw |= m
                # Тот же экземпляр по другой подсказке — склеить, иначе новая страница.
                dup = [k for k, (_, o) in enumerate(found) if (m & o).sum() / max((m | o).sum(), 1) >= SAM3_SAME_IOU]
                if dup:
                    k = dup[0]
                    found[k] = (max(found[k][0], float(sc)), found[k][1] | m)
                else:
                    found.append((float(sc), m))
    # Сбросить картинку предиктора и кэш CUDA: иначе на следующем кадре растёт видеопамять.
    if hasattr(predictor, "reset_image"):
        predictor.reset_image()
    torch.cuda.empty_cache()
    labels = np.zeros((h, w), np.uint8)
    # Пересечения — экземпляру с большей оценкой: он кладётся последним.
    for k, (_, m) in sorted(enumerate(found), key=lambda t: t[1][0]):
        labels[m] = k + 1
    return raw, labels


def postprocess_plain(raw: np.ndarray, book: np.ndarray, hulls: np.ndarray) -> np.ndarray:
    """Маска бумаги из сырой маски SAM 3 (вариант ``plain``): закрытие, открытие, компоненты с печатью, заливка дыр.

    Args:
        raw: Сырая маска SAM 3.
        book: Маска книги.
        hulls: Оболочки своей печати.

    Returns:
        Маска бумаги.
    """
    paper = raw & book
    paper = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_CLOSE, disk(CLOSE_PX)) > 0
    paper = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_OPEN, disk(3)) > 0
    # Компоненты, задевающие печать.
    labels, _ = ndimage.label(paper)
    keep = np.unique(labels[hulls & paper])
    paper = np.isin(labels, keep[keep > 0]) if len(keep) else paper
    paper = ndimage.binary_fill_holes(paper)
    return paper & book


def paper_seeds(bgr: np.ndarray, hulls: np.ndarray, book: np.ndarray) -> np.ndarray:
    """Затравки бумаги: внутри оболочек своей печати и книги, без краски.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        hulls: Оболочки своей печати.
        book: Маска книги.

    Returns:
        Булева маска.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    local = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, disk(PAPER_BG_PX))
    not_ink = gray.astype(np.int16) > local.astype(np.int16) - INK_DROP
    inner = cv2.erode((hulls & book).astype(np.uint8), disk(3)) > 0
    return inner & not_ink & (cv2.erode(not_ink.astype(np.uint8), disk(2)) > 0)


class PaperModel:
    """Робастная модель цвета бумаги в Lab: медиана и ковариация (a, b), нижний квантиль L."""

    def __init__(self, lab: np.ndarray, seeds: np.ndarray) -> None:
        """Строит модель по затравкам.

        Args:
            lab: Кадр Lab float32 (сглаженный).
            seeds: Затравки бумаги.
        """
        ab = lab[..., 1:][seeds]
        med = np.median(ab, axis=0)
        # Ядро затравок: 90% ближайших к медиане — выбросы (краска, тень) не портят ковариацию.
        d = np.linalg.norm(ab - med, axis=1)
        core = ab[d <= np.percentile(d, 90)]
        self.mean = core.mean(axis=0)
        self.icov = np.linalg.inv(np.cov(core.T) + np.eye(2) * 1.0)
        self.l_low = float(np.percentile(lab[..., 0][seeds], 2))

    def distance(self, lab: np.ndarray) -> np.ndarray:
        """Расстояние Махаланобиса по (a, b) от модели.

        Args:
            lab: Кадр Lab float32.

        Returns:
            Карта float32.
        """
        d = lab[..., 1:] - self.mean
        return np.sqrt(np.einsum("...i,ij,...j->...", d, self.icov, d)).astype(np.float32)

    def is_paper(self, lab: np.ndarray) -> np.ndarray:
        """Пиксели, похожие на бумагу: близко по оттенку и не темнее нижнего квантиля бумаги минус запас.

        Args:
            lab: Кадр Lab float32.

        Returns:
            Булева маска.
        """
        return (self.distance(lab) < MAHAL_THR) & (lab[..., 0] > self.l_low - L_MARGIN)


def smooth_lab(bgr: np.ndarray) -> np.ndarray:
    """Кадр в Lab float32 после медианы (без краски и шума).

    Args:
        bgr: Кадр BGR.

    Returns:
        Lab float32.
    """
    return cv2.cvtColor(cv2.medianBlur(bgr, SMOOTH_PX), cv2.COLOR_BGR2LAB).astype(np.float32)
