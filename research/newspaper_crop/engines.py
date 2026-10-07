"""Движки грубой маски газетной бумаги: SAM2.1 по точкам и боксу, SAM 3 по тексту, BiRefNet, кластеры DINOv2, цвет бумаги, стек scan_cropping."""

from __future__ import annotations

import logging
from enum import Enum

import cv2
import numpy as np
import torch
from PIL import Image

from research.newspaper_crop.zone import clean_paper_mask, work_shape_of

log = logging.getLogger(__name__)

# Положительные точки SAM: сетка 3×3 в центральной части кадра (доли ширины и высоты). Полоса
# в этой съёмке занимает почти весь кадр, центр всегда на бумаге; края не трогаем — там пол,
# крышка подшивки и пальцы.
CENTRE_GRID = (0.35, 0.5, 0.65)
CENTRE_GRID_Y = (0.3, 0.5, 0.7)
# Нижняя граница предсказанного SAM IoU для маски-кандидата «весь лист».
SAM_MIN_SCORE = 0.5
# Отступ бокса-подсказки от края кадра, доля стороны.
BOX_INSET = 0.01
# Входная сторона BiRefNet (квадрат; модель учена на 1024).
BIREFNET_SIDE = 1024
# Сторона DINOv2 по длинной стороне кадра (кратно патчу 14): 1344 → сетка 96×64 патча.
DINO_LONG_SIDE = 1344
DINO_PATCH = 14
# Число кластеров патчей DINOv2.
DINO_CLUSTERS = 5
# Текстовые подсказки SAM 3: что брать и что заведомо не бумага.
SAM3_PAPER_PROMPT = "newspaper page"


class EngineName(str, Enum):
    """Имена движков стенда (подпапки прогона)."""

    SAM2_POINTS = "sam2_points"
    SAM2_BOX = "sam2_box"
    SAM3_TEXT = "sam3_text"
    BIREFNET = "birefnet"
    DINO_KMEANS = "dino_kmeans"
    DINO_SAM2 = "dino_sam2"
    COLOR = "color"
    SCAN_CROPPING = "scan_cropping"
    INK_HULL = "ink_hull"


def work_shape(rgb: np.ndarray) -> tuple[int, int]:
    """Размер рабочей маски (H, W) для кадра (см. :func:`zone.work_shape_of`).

    Args:
        rgb: Кадр превью.

    Returns:
        (H, W) на масштабе :data:`WORK_SCALE`.
    """
    return work_shape_of(rgb)


def to_work(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Приводит маску любого размера к рабочему (ближайший сосед для булевой, билинейно для вероятностей).

    Args:
        mask: Маска (bool или float в [0, 1]).
        shape: Целевой (H, W).

    Returns:
        Булева маска рабочего размера (вероятности — порог 0.5).
    """
    m = mask.astype(np.float32)
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return m >= 0.5


class Sam2Engine:
    """SAM2.1 Hiera-L (transformers): подсказка — центральные точки (и бокс почти во весь кадр); берётся самая крупная уверенная маска."""

    def __init__(self, with_box: bool, device: str = "cuda") -> None:
        """Грузит модель.

        Args:
            with_box: Добавлять ли бокс во весь кадр (с отступом :data:`BOX_INSET`) к точкам.
            device: Устройство torch.
        """
        from transformers import Sam2Model, Sam2Processor

        self.with_box = with_box
        self.device = device
        self.model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device).eval()
        self.processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

    def masks(
        self, rgb: np.ndarray, points: list[list[float]], labels: list[int], box: list[float] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Три маски-кандидата SAM2 и их оценки по произвольным подсказкам.

        Args:
            rgb: Кадр RGB.
            points: Точки [[x, y], ...] в пикселях кадра.
            labels: Метки точек (1 — объект, 0 — фон).
            box: Бокс [x0, y0, x1, y1] или None.

        Returns:
            (маски (3, H, W) bool в размере кадра, оценки IoU (3,)).
        """
        image = Image.fromarray(rgb)
        kwargs = {"input_points": [[points]], "input_labels": [[labels]]}
        if box is not None:
            kwargs["input_boxes"] = [[box]]
        inputs = self.processor(images=image, return_tensors="pt", **kwargs).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy()
        return masks.astype(bool), out.iou_scores[0, 0].float().cpu().numpy()

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Маска бумаги на рабочем масштабе.

        Args:
            rgb: Кадр превью RGB.

        Returns:
            Булева маска.
        """
        h, w = rgb.shape[:2]
        points = [[w * x, h * y] for x in CENTRE_GRID for y in CENTRE_GRID_Y]
        box = [w * BOX_INSET, h * BOX_INSET, w * (1 - BOX_INSET), h * (1 - BOX_INSET)] if self.with_box else None
        masks, scores = self.masks(rgb, points, [1] * len(points), box)
        return to_work(pick_sheet_mask(masks, scores), work_shape(rgb))


def pick_sheet_mask(masks: np.ndarray, scores: np.ndarray, min_score: float = SAM_MIN_SCORE) -> np.ndarray:
    """Из кандидатов SAM выбирает «весь лист»: самую крупную маску с оценкой не ниже порога.

    Args:
        masks: Маски (K, H, W) bool.
        scores: Оценки (K,).
        min_score: Порог оценки; если ему не отвечает ни одна — берётся лучшая по оценке.

    Returns:
        Булева маска (H, W).
    """
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    good = np.where(scores >= min_score)[0]
    if len(good) == 0:
        return masks[int(np.argmax(scores))]
    return masks[good[int(np.argmax(areas[good]))]]


class Sam3Engine:
    """SAM 3 (ultralytics ``SAM3SemanticPredictor``) по текстовой подсказке; берётся экземпляр, накрывающий центр кадра."""

    def __init__(self, weights: str) -> None:
        """Грузит модель.

        Args:
            weights: Путь к ``sam3.pt`` (gated-веса facebook/sam3).
        """
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(conf=0.25, task="segment", mode="predict", model=weights, half=True, verbose=False, save=False)
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Маска бумаги на рабочем масштабе: экземпляр «newspaper page», содержащий центр кадра (или крупнейший).

        Args:
            rgb: Кадр превью RGB.

        Returns:
            Булева маска (пустая, если ничего не найдено).
        """
        shape = work_shape(rgb)
        self.predictor.set_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        results = self.predictor(text=[SAM3_PAPER_PROMPT])
        if not results or results[0].masks is None:
            return np.zeros(shape, bool)
        masks = results[0].masks.data.cpu().numpy() > 0.5
        h, w = masks.shape[1:]
        centre = [m for m in masks if m[h // 2, w // 2]]
        pick = max(centre or list(masks), key=lambda m: m.sum())
        return to_work(pick, shape)


class BiRefNetEngine:
    """BiRefNet (дихотомическая сегментация, «главный объект» кадра) на квадрате 1024."""

    def __init__(self, device: str = "cuda") -> None:
        """Грузит модель (код модели — trust_remote_code из ZhengPeng7/BiRefNet).

        Args:
            device: Устройство torch.
        """
        from transformers import AutoModelForImageSegmentation

        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        self.model.to(device).eval().half()

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Маска бумаги на рабочем масштабе (порог вероятности 0.5).

        Args:
            rgb: Кадр превью RGB.

        Returns:
            Булева маска.
        """
        x = cv2.resize(rgb, (BIREFNET_SIDE, BIREFNET_SIDE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        t = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float16)).to(self.device)
        with torch.no_grad():
            prob = self.model(t)[-1].sigmoid()[0, 0].float().cpu().numpy()
        return to_work(prob, work_shape(rgb))


class DinoEngine:
    """DINOv2-base: патч-признаки → k-средних → кластеры, занимающие центр кадра, — «бумага»; по желанию уточнение SAM2."""

    def __init__(self, refine: Sam2Engine | None, device: str = "cuda") -> None:
        """Грузит модель.

        Args:
            refine: SAM2 для уточнения края грубой маски (бокс + точки внутри/снаружи) или None.
            device: Устройство torch.
        """
        from transformers import AutoModel

        self.device = device
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval().half()
        self.refine = refine

    def patch_features(self, rgb: np.ndarray) -> np.ndarray:
        """Нормированные признаки патчей DINOv2.

        Args:
            rgb: Кадр RGB.

        Returns:
            Массив (gh, gw, C) float32, L2-нормированный по C.
        """
        h, w = rgb.shape[:2]
        scale = DINO_LONG_SIDE / max(h, w)
        gh, gw = int(round(h * scale / DINO_PATCH)), int(round(w * scale / DINO_PATCH))
        x = cv2.resize(rgb, (gw * DINO_PATCH, gh * DINO_PATCH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        t = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float16)).to(self.device)
        with torch.no_grad():
            tokens = self.model(pixel_values=t).last_hidden_state[0, 1:].float().cpu().numpy()
        feats = tokens.reshape(gh, gw, -1)
        return feats / (np.linalg.norm(feats, axis=2, keepdims=True) + 1e-6)

    def coarse(self, rgb: np.ndarray) -> np.ndarray:
        """Грубая маска на сетке патчей: кластеры, чья доля в центральной трети кадра выше, чем на кайме.

        Args:
            rgb: Кадр RGB.

        Returns:
            Булева маска (gh, gw).
        """
        feats = self.patch_features(rgb)
        gh, gw, c = feats.shape
        data = feats.reshape(-1, c).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        cv2.setRNGSeed(0)
        _, labels, _ = cv2.kmeans(data, DINO_CLUSTERS, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(gh, gw)
        centre = np.zeros((gh, gw), bool)
        centre[gh // 4 : 3 * gh // 4, gw // 4 : 3 * gw // 4] = True
        rim = np.zeros((gh, gw), bool)
        rim[:2], rim[-2:], rim[:, :2], rim[:, -2:] = True, True, True, True
        keep = []
        for k in range(DINO_CLUSTERS):
            in_centre = (labels[centre] == k).mean()
            on_rim = (labels[rim] == k).mean()
            # Кластер бумаги/печати заметен в центре; фон, крышка и пальцы — только у края.
            if in_centre > 0.05 and in_centre >= 0.5 * on_rim:
                keep.append(k)
        return np.isin(labels, keep)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Маска бумаги на рабочем масштабе.

        Args:
            rgb: Кадр превью RGB.

        Returns:
            Булева маска.
        """
        shape = work_shape(rgb)
        coarse = clean_paper_mask(to_work(self.coarse(rgb), shape), open_px=0)
        if self.refine is None or not coarse.any():
            return coarse
        return self._refine(rgb, coarse)

    def _refine(self, rgb: np.ndarray, coarse: np.ndarray) -> np.ndarray:
        """Уточняет край грубой маски SAM2: бокс маски, точки в глубине маски и за её краем.

        Args:
            rgb: Кадр RGB.
            coarse: Грубая маска на рабочем масштабе.

        Returns:
            Булева маска на рабочем масштабе.
        """
        h, w = rgb.shape[:2]
        sx, sy = w / coarse.shape[1], h / coarse.shape[0]
        ys, xs = np.nonzero(coarse)
        box = [xs.min() * sx, ys.min() * sy, xs.max() * sx, ys.max() * sy]
        # Глубина маски: точки «внутри» — в самой глубокой части, «снаружи» — далеко за краем.
        dist_in = cv2.distanceTransform(coarse.astype(np.uint8), cv2.DIST_L2, 5)
        dist_out = cv2.distanceTransform((~coarse).astype(np.uint8), cv2.DIST_L2, 5)
        rng = np.random.default_rng(0)
        inner = np.argwhere(dist_in > 0.6 * dist_in.max())
        outer = np.argwhere(dist_out > 40)
        pos = inner[rng.choice(len(inner), min(8, len(inner)), replace=False)]
        neg = outer[rng.choice(len(outer), min(6, len(outer)), replace=False)] if len(outer) else np.empty((0, 2), int)
        points = [[float(x * sx), float(y * sy)] for y, x in np.vstack([pos, neg])]
        labels = [1] * len(pos) + [0] * len(neg)
        masks, scores = self.refine.masks(rgb, points, labels, box)
        return to_work(pick_sheet_mask(masks, scores), coarse.shape)


def color_paper_mask(rgb: np.ndarray, close_px: int = 9, delta: float = 14.0) -> np.ndarray:
    """Цветовая модель бумаги без нейросетей: эталон цвета — светлые пиксели центра кадра, маска — близкие по Lab.

    Печать сначала стирается серым замыканием (краска тоньше окна), потом в Lab считается расстояние до
    медианы бумаги центральной трети кадра по хроме (a, b) и светлоте с весом 0.5.

    Args:
        rgb: Кадр превью RGB.
        close_px: Радиус замыкания на рабочем масштабе (стирает буквы; толще — рисунки).
        delta: Порог расстояния в Lab.

    Returns:
        Булева маска на рабочем масштабе.
    """
    shape = work_shape(rgb)
    small = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
    clean = cv2.morphologyEx(small, cv2.MORPH_CLOSE, k)
    lab = cv2.cvtColor(clean, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 3)
    h, w = shape
    centre = lab[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3].reshape(-1, 3)
    ref = np.median(centre[centre[:, 0] >= np.quantile(centre[:, 0], 0.5)], axis=0)
    diff = lab - ref
    dist = np.sqrt(diff[..., 1] ** 2 + diff[..., 2] ** 2 + (0.5 * diff[..., 0]) ** 2)
    return dist < delta


class ScanCroppingEngine:
    """Нынешний стек журналов: YOLO-World «page/paper…» → SAM-B → ``page_mask`` из ``scan_cropping.page_detection``."""

    def __init__(self) -> None:
        """Грузит ``GpuModels`` без LaMa."""
        from ocr_utils.scan_cropping.gpu_models import GpuModels

        self.models = GpuModels(with_lama=False)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Маска разворота ``page_mask`` на рабочем масштабе.

        Args:
            rgb: Кадр превью RGB.

        Returns:
            Булева маска.
        """
        from ocr_utils.scan_cropping.page_detection import page_mask

        mask = page_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.models)
        return to_work(mask > 0, work_shape(rgb))


def make_engine(name: EngineName, sam3_weights: str | None = None):
    """Создаёт движок по имени.

    Args:
        name: Имя движка.
        sam3_weights: Путь к весам SAM 3 (только для ``sam3_text``).

    Returns:
        Вызываемый объект ``rgb -> bool mask`` на рабочем масштабе.
    """
    if name is EngineName.SAM2_POINTS:
        return Sam2Engine(with_box=False)
    if name is EngineName.SAM2_BOX:
        return Sam2Engine(with_box=True)
    if name is EngineName.SAM3_TEXT:
        if not sam3_weights:
            raise click_error("для sam3_text нужен --sam3-weights")
        return Sam3Engine(sam3_weights)
    if name is EngineName.BIREFNET:
        return BiRefNetEngine()
    if name is EngineName.DINO_KMEANS:
        return DinoEngine(refine=None)
    if name is EngineName.DINO_SAM2:
        return DinoEngine(refine=Sam2Engine(with_box=False))
    if name is EngineName.COLOR:
        return color_paper_mask
    if name is EngineName.INK_HULL:
        from research.newspaper_crop.ink_guard import ink_hull_mask

        return ink_hull_mask
    return ScanCroppingEngine()


def click_error(message: str) -> Exception:
    """Ошибка CLI с сообщением (без импорта click в модуль движков).

    Args:
        message: Текст.

    Returns:
        Исключение ``click.UsageError``.
    """
    import click

    return click.UsageError(message)
