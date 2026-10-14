"""Кэш геометрии кадра на рабочем масштабе: маски-кандидаты SAM2.1 (бокс + точки) и глубина/нормали MoGe-2."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from research.bound_volumes.frames import OUT_ROOT, Frame

log = logging.getLogger(__name__)

# Рабочий масштаб кэша: ¼ кадра (5813×4419 → 1453×1105, ~90 dpi) — MoGe-2 и SAM2 всё равно работают на ~1–1.5 Мп.
WORK_SCALE = 0.25
# Корень кэша: <GEOMETRY_ROOT>/<источник>/<плоский ключ>.npz.
GEOMETRY_ROOT = OUT_ROOT / "geometry"
# Подсказки SAM2: сетка точек в середине кадра и бокс почти во весь кадр (как в газетном стенде, ``Sam2Engine``).
CENTRE_GRID_X = (0.3, 0.5, 0.7)
CENTRE_GRID_Y = (0.35, 0.5, 0.65)
BOX_INSET = 0.01
# Отрицательные подсказки на коврике: отступ от рамки (доля стороны), точек на сторону, предел серого коврика.
MAT_INSET = 0.03
MAT_PER_SIDE = 7
MAT_MAX_GRAY = 45
# Веса MoGe-2 (MIT): ViT-L с головой нормалей.
MOGE_WEIGHTS = "Ruicheng/moge-2-vitl-normal"
# Уровень разрешения MoGe-2 (0–9): 9 — самая подробная сетка токенов.
MOGE_RESOLUTION_LEVEL = 9


@dataclass(frozen=True)
class Geometry:
    """Геометрия кадра на рабочем масштабе.

    Поля:
        sam_masks: Маски-кандидаты SAM2 ``(3, H, W)`` bool.
        sam_scores: Оценки IoU кандидатов ``(3,)``.
        depth: Глубина MoGe-2 ``(H, W)`` float32 (относительные метры), NaN вне валидной области.
        normal: Нормали MoGe-2 ``(H, W, 3)`` float32 в системе камеры (z — к камере со знаком модели).
    """

    sam_masks: np.ndarray
    sam_scores: np.ndarray
    depth: np.ndarray
    normal: np.ndarray


def work_image(bgr: np.ndarray) -> np.ndarray:
    """Кадр на рабочем масштабе.

    Args:
        bgr: Кадр BGR в исходном разрешении.

    Returns:
        Уменьшенный кадр BGR.
    """
    return cv2.resize(bgr, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)


def cache_path(source: str, frame: Frame) -> Path:
    """Путь файла кэша геометрии.

    Args:
        source: Имя источника кадра (``orig`` или вариант закраса пальцев).
        frame: Кадр.

    Returns:
        Путь ``.npz``.
    """
    return GEOMETRY_ROOT / source / f"{frame.flat_key}.npz"


def mat_points(gray: np.ndarray) -> list[list[float]]:
    """Отрицательные подсказки SAM: точки на тёмном коврике у краёв кадра.

    Кандидаты — сетка точек по периметру с отступом :data:`MAT_INSET` от рамки; берутся те, где медиана серого
    в окне 15×15 ниже :data:`MAT_MAX_GRAY` (коврик), а не книга или рука.

    Args:
        gray: Кадр рабочего масштаба в сером.

    Returns:
        Точки ``[[x, y], ...]``.
    """
    h, w = gray.shape
    blurred = cv2.medianBlur(gray, 15)
    out = []
    for f in np.linspace(0.05, 0.95, MAT_PER_SIDE):
        for x, y in ((f * w, MAT_INSET * h), (f * w, (1 - MAT_INSET) * h), (MAT_INSET * w, f * h), ((1 - MAT_INSET) * w, f * h)):
            if blurred[int(y), int(x)] < MAT_MAX_GRAY:
                out.append([float(x), float(y)])
    return out


class GeometryModels:
    """SAM2.1 Hiera-L (transformers) и MoGe-2 ViT-L, загруженные один раз на прогон."""

    def __init__(self, device: str = "cuda") -> None:
        """Грузит обе модели.

        Args:
            device: Устройство torch.
        """
        from moge.model.v2 import MoGeModel
        from transformers import Sam2Model, Sam2Processor

        self.device = device
        self.sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device).eval()
        self.processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
        self.moge = MoGeModel.from_pretrained(MOGE_WEIGHTS).to(device).eval().half()

    def sam_masks(
        self, rgb: np.ndarray, points: list[list[float]], labels: list[int], box: list[float] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Три маски-кандидата SAM2 по подсказкам.

        Args:
            rgb: Картинка RGB рабочего масштаба.
            points: Точки ``[[x, y], ...]`` в px картинки.
            labels: Метки точек (1 — объект, 0 — фон).
            box: Бокс ``[x0, y0, x1, y1]`` или None.

        Returns:
            ``(маски (3, H, W) bool, оценки (3,))``.
        """
        from PIL import Image

        kwargs = {"input_points": [[points]], "input_labels": [[labels]]}
        if box is not None:
            kwargs["input_boxes"] = [[box]]
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt", **kwargs).to(self.device)
        with torch.no_grad():
            out = self.sam(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy()
        return masks.astype(bool), out.iou_scores[0, 0].float().cpu().numpy()

    def moge_maps(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Глубина и нормали MoGe-2.

        Args:
            rgb: Картинка RGB рабочего масштаба.

        Returns:
            ``(глубина (H, W) float32 c NaN вне маски модели, нормали (H, W, 3) float32)``.
        """
        tensor = torch.tensor(rgb / 255.0, dtype=torch.float16, device=self.device).permute(2, 0, 1)
        with torch.no_grad():
            out = self.moge.infer(tensor, resolution_level=MOGE_RESOLUTION_LEVEL)
        depth = out["depth"].float().cpu().numpy()
        depth[~out["mask"].cpu().numpy().astype(bool)] = np.nan
        return depth, out["normal"].float().cpu().numpy()

    def __call__(self, bgr_work: np.ndarray) -> Geometry:
        """Геометрия кадра рабочего масштаба.

        Args:
            bgr_work: Кадр BGR рабочего масштаба.

        Returns:
            Маски SAM2 и карты MoGe-2.
        """
        rgb = cv2.cvtColor(bgr_work, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        points = [[w * x, h * y] for x in CENTRE_GRID_X for y in CENTRE_GRID_Y]
        # Отрицательные точки на коврике (``mat_points``) отвергнуты: SAM2 с ними ставит всем кандидатам оценку 0
        # (95 из 95 кадров); захват коврика отсекает выбор маски по доле тёмного (``layers.book_mask``).
        labels = [1] * len(points)
        box = [w * BOX_INSET, h * BOX_INSET, w * (1 - BOX_INSET), h * (1 - BOX_INSET)]
        masks, scores = self.sam_masks(rgb, points, labels, box)
        depth, normal = self.moge_maps(rgb)
        return Geometry(masks, scores, depth, normal)


def save_geometry(path: Path, geo: Geometry) -> None:
    """Пишет геометрию в ``.npz`` (маски упакованы битами, карты — float16).

    Args:
        path: Путь файла.
        geo: Геометрия.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sam_masks=np.packbits(geo.sam_masks, axis=-1),
        sam_shape=np.array(geo.sam_masks.shape),
        sam_scores=geo.sam_scores,
        depth=geo.depth.astype(np.float16),
        normal=geo.normal.astype(np.float16),
    )


def load_geometry(path: Path) -> Geometry:
    """Читает геометрию из ``.npz``.

    Args:
        path: Путь файла.

    Returns:
        Геометрия (карты в float32).
    """
    data = np.load(path)
    shape = tuple(data["sam_shape"])
    masks = np.unpackbits(data["sam_masks"], axis=-1, count=shape[-1]).astype(bool).reshape(shape)
    return Geometry(masks, data["sam_scores"], data["depth"].astype(np.float32), data["normal"].astype(np.float32))
