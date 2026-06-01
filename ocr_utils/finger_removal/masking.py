"""Построение маски пальцев на сканах.

Стратегия (по умолчанию режим ``auto``):
  1. Нейросетевой детектор (``neural_hand_mask``): open-vocabulary YOLO-World
     находит боксы руки/пальца, затем SAM по этим боксам строит точный силуэт.
     Нейромаска — семантический «затвор»: если рука не найдена, маска пустая.
  2. Скин-прайор (``skin_color_mask``): детектор кожи в YCrCb+HSV.
  3. Добор кожей ТОЛЬКО там, где она пересекает нейромаску
     (``keep_seeded_components``) — подхватывает мягкие края/ноготь, недобранные
     детектором. Скин-компоненты, лишь касающиеся края кадра, но не связанные с
     рукой (цветная кромка бумаги, текст), НЕ берём — иначе ложные маски.

Режим ``skin`` (без нейросети) использует краевое ограничение
(``keep_border_components``): оставляет компоненты у рамки кадра нужной площади.

Веса нейромоделей качаются автоматически в ``finger_models/`` (корень проекта).
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FingerDetectionsBySide(BaseModel):
    """Детекции пальцев, сгруппированные по сторонам изображения."""
    top: list[np.ndarray] = []
    bottom: list[np.ndarray] = []
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    
    class Config:
        arbitrary_types_allowed = True

# Папка для весов нейромоделей (корень проекта, по аналогии с dewarp_models/)
MODELS_DIR = Path(__file__).resolve().parents[2] / "finger_models"

# Имена ассетов ultralytics (качаются по basename, если файла нет)
DEFAULT_YOLO_WORLD = "yolov8x-worldv2.pt"
DEFAULT_SAM = "sam_b.pt"

# Классы open-vocabulary детектора, описывающие пальцы/руку.
# Без "person" — он матчит всю страницу/фото и приводит к огромным ложным маскам.
HAND_CLASSES = ["hand", "finger", "thumb", "fingertip", "human hand"]

# Палец никогда не занимает больше этой доли кадра — отсекаем гигантские ложные маски
MAX_FINGER_AREA_FRAC = 0.12

# Кэш загруженных моделей (чтобы не грузить заново на каждом кадре)
_MODEL_CACHE: dict = {}


# ============================================================
# Скин-прайор (детектор кожи) + краевое ограничение
# ============================================================


def skin_color_mask(rgb: np.ndarray) -> np.ndarray:
    """Бинарная маска кожи (uint8 0/255) по правилам в YCrCb и HSV (пересечение)."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    # Классический диапазон кожи в YCrCb
    skin_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    # Тёплый красно-розовый оттенок кожи. Порог насыщенности повышен (>=55),
    # чтобы НЕ цеплять бледную кремовую бумагу (низкая насыщенность) и не давать
    # ложных масок на кадрах вообще без пальцев.
    skin_hsv = ((h <= 20) | (h >= 172)) & (s >= 55) & (s <= 190) & (v >= 50)

    mask = (skin_ycrcb & skin_hsv).astype(np.uint8) * 255
    return mask


def morph_cleanup(mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Морфологическое закрытие+открытие для удаления шума и заполнения дыр."""
    if ksize <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def keep_border_components(
    mask: np.ndarray,
    edge_frac: float = 0.12,
    min_area_frac: float = 0.0015,
    max_area_frac: float = MAX_FINGER_AREA_FRAC,
    seed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Оставляет только компоненты, относящиеся к пальцу.

    Компонента сохраняется, если её площадь в диапазоне
    ``[min_area_frac, max_area_frac]`` от кадра И выполнено хотя бы одно из условий:
      - её пиксели пересекают краевую рамку шириной ``edge_frac`` (палец входит с края);
      - она пересекает ``seed_mask`` (например, нейромаску) — чтобы подхватить
        внутренние части пальца, доходящие далеко от края.

    Верхний порог площади отсекает гигантские ложные блобы (вся страница/обложка).
    """
    h, w = mask.shape[:2]
    frame_w = max(1, int(edge_frac * w))
    frame_h = max(1, int(edge_frac * h))
    min_area = int(min_area_frac * h * w)
    max_area = int(max_area_frac * h * w)

    border_frame = np.zeros((h, w), dtype=bool)
    border_frame[:frame_h, :] = True
    border_frame[h - frame_h :, :] = True
    border_frame[:, :frame_w] = True
    border_frame[:, w - frame_w :] = True

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        comp = labels == i
        touches_border = bool(np.any(comp & border_frame))
        touches_seed = bool(seed_mask is not None and np.any(comp & (seed_mask > 0)))
        if touches_border or touches_seed:
            out[comp] = 255
    return out


def keep_seeded_components(
    mask: np.ndarray, seed_mask: np.ndarray, min_area_frac: float = 0.0015, max_area_frac: float = MAX_FINGER_AREA_FRAC
) -> np.ndarray:
    """Оставляет компоненты маски, пересекающиеся с ``seed_mask`` (без условия края).

    В отличие от ``keep_border_components``, НЕ пропускает компоненты только из-за
    касания рамки кадра — нужно именно пересечение с seed (нейромаской).
    """
    h, w = mask.shape[:2]
    min_area = int(min_area_frac * h * w)
    max_area = int(max_area_frac * h * w)
    seed = seed_mask > 0
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        comp = labels == i
        if bool(np.any(comp & seed)):
            out[comp] = 255
    return out


def skin_edge_mask(rgb: np.ndarray, edge_frac: float = 0.12, min_area_frac: float = 0.0015) -> np.ndarray:
    """Полный скин-прайор: цвет кожи → морфология → краевые компоненты."""
    mask = skin_color_mask(rgb)
    mask = morph_cleanup(mask, ksize=max(3, int(0.006 * min(rgb.shape[:2]))))
    mask = keep_border_components(mask, edge_frac=edge_frac, min_area_frac=min_area_frac)
    return mask


# ============================================================
# Нейросетевой детектор: YOLO-World (боксы) → SAM (силуэт)
# ============================================================


def _load_yolo_world(model_name: str, device: str):
    """Ленивая загрузка YOLO-World с кэшем; веса в MODELS_DIR."""
    key = f"yolo:{model_name}"
    if key not in _MODEL_CACHE:
        from ultralytics import YOLOWorld

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model = YOLOWorld(str(MODELS_DIR / model_name))
        model.set_classes(HAND_CLASSES)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _load_sam(model_name: str):
    """Ленивая загрузка SAM с кэшем; веса в MODELS_DIR."""
    key = f"sam:{model_name}"
    if key not in _MODEL_CACHE:
        from ultralytics import SAM

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _MODEL_CACHE[key] = SAM(str(MODELS_DIR / model_name))
    return _MODEL_CACHE[key]


def neural_hand_mask(
    rgb: np.ndarray,
    device: str = "cuda",
    conf: float = 0.05,
    yolo_model: str = DEFAULT_YOLO_WORLD,
    sam_model: str = DEFAULT_SAM,
    max_box_frac: float = 0.30,
    max_area_frac: float = MAX_FINGER_AREA_FRAC,
) -> np.ndarray:
    """Маска пальца через YOLO-World→SAM. Возвращает uint8 0/255 (может быть пустой).

    Боксы крупнее ``max_box_frac`` кадра и SAM-маски крупнее ``max_area_frac``
    отбраковываются — палец не занимает половину снимка, такие срабатывания ложные.
    """
    h, w = rgb.shape[:2]
    img_area = h * w
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    yolo = _load_yolo_world(yolo_model, device)
    det = yolo.predict(bgr, conf=conf, device=device, verbose=False)
    boxes = det[0].boxes.xyxy.cpu().numpy() if det and det[0].boxes is not None else np.empty((0, 4))

    # Отбрасываем слишком большие боксы (вся страница/обложка)
    if len(boxes) > 0:
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        keep = (bw * bh) <= (max_box_frac * img_area)
        boxes = boxes[keep]
    if len(boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    sam = _load_sam(sam_model)
    seg = sam.predict(bgr, bboxes=boxes, device=device, verbose=False)
    mask = np.zeros((h, w), dtype=np.uint8)
    if seg and seg[0].masks is not None:
        data = seg[0].masks.data.cpu().numpy()  # (N, H, W), float/bool
        for m in data:
            m_bin = m > 0.5
            if m_bin.shape != (h, w):
                m_bin = cv2.resize(m_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            # Отбрасываем отдельные маски, покрывающие слишком большую долю кадра
            if m_bin.sum() > max_area_frac * img_area:
                continue
            mask[m_bin] = 255
    return mask


def get_box_edge_overlap(box: np.ndarray, img_shape: tuple[int, int], edge_frac: float = 0.12) -> dict[str, float]:
    """Вычисляет длину пересечения бокса с каждым краем изображения.
    
    Args:
        box: [x1, y1, x2, y2]
        img_shape: (height, width)
        edge_frac: ширина краевой зоны (доля от размера изображения)
    
    Returns:
        dict с ключами 'top', 'bottom', 'left', 'right' и длинами пересечений
    """
    h, w = img_shape
    x1, y1, x2, y2 = box
    
    edge_h = int(edge_frac * h)
    edge_w = int(edge_frac * w)
    
    overlaps = {
        'top': 0.0,
        'bottom': 0.0,
        'left': 0.0,
        'right': 0.0,
    }
    
    if y1 < edge_h:
        overlaps['top'] = x2 - x1
    if y2 > h - edge_h:
        overlaps['bottom'] = x2 - x1
    if x1 < edge_w:
        overlaps['left'] = y2 - y1
    if x2 > w - edge_w:
        overlaps['right'] = y2 - y1
    
    return overlaps


def get_dominant_side(box: np.ndarray, img_shape: tuple[int, int], edge_frac: float = 0.12) -> Optional[str]:
    """Определяет, к какой стороне изображения бокс прилегает сильнее всего.
    
    Returns:
        'top', 'bottom', 'left', 'right' или None, если не прилегает ни к одной стороне
    """
    overlaps = get_box_edge_overlap(box, img_shape, edge_frac)
    max_overlap = max(overlaps.values())
    
    if max_overlap == 0:
        return None
    
    for side, overlap in overlaps.items():
        if overlap == max_overlap:
            return side
    
    return None


def neural_hand_mask_batch(
    rgb_list: list[np.ndarray],
    device: str = "cuda",
    conf: float = 0.05,
    yolo_model: str = DEFAULT_YOLO_WORLD,
    sam_model: str = DEFAULT_SAM,
    max_box_frac: float = 0.30,
    max_area_frac: float = MAX_FINGER_AREA_FRAC,
) -> list[np.ndarray]:
    """Батчевая версия neural_hand_mask. Возвращает список масок uint8 0/255."""
    if not rgb_list:
        return []

    bgr_list = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in rgb_list]
    
    yolo = _load_yolo_world(yolo_model, device)
    det_results = yolo.predict(bgr_list, conf=conf, device=device, verbose=False)
    
    sam = _load_sam(sam_model)
    masks = []
    
    for idx, (rgb, bgr, det) in enumerate(zip(rgb_list, bgr_list, det_results)):
        h, w = rgb.shape[:2]
        img_area = h * w
        
        boxes = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else np.empty((0, 4))
        
        if len(boxes) > 0:
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            keep = (bw * bh) <= (max_box_frac * img_area)
            boxes = boxes[keep]
        
        if len(boxes) == 0:
            masks.append(np.zeros((h, w), dtype=np.uint8))
            continue
        
        seg = sam.predict(bgr, bboxes=boxes, device=device, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        if seg and seg[0].masks is not None:
            data = seg[0].masks.data.cpu().numpy()
            for m in data:
                m_bin = m > 0.5
                if m_bin.shape != (h, w):
                    m_bin = cv2.resize(m_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if m_bin.sum() > max_area_frac * img_area:
                    continue
                mask[m_bin] = 255
        masks.append(mask)
    
    return masks


def neural_hand_mask_batch_double_pass(
    rgb_list: list[np.ndarray],
    device: str = "cuda",
    conf_high: float = 0.15,
    conf_low: float = 0.03,
    edge_frac: float = 0.12,
    dilate_px: int = 10,
    yolo_model: str = DEFAULT_YOLO_WORLD,
    sam_model: str = DEFAULT_SAM,
    max_box_frac: float = 0.30,
    max_area_frac: float = MAX_FINGER_AREA_FRAC,
) -> tuple[list[np.ndarray], list[np.ndarray], list[FingerDetectionsBySide]]:
    """Двухпроходная батчевая детекция с высоким и низким порогами confidence.
    
    Returns:
        (masks_high, masks_low, detections_by_side) - маски с высоким conf (красные),
        маски с низким conf (синие), детекции по сторонам для каждого изображения
    """
    if not rgb_list:
        return [], [], []
    
    bgr_list = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in rgb_list]
    yolo = _load_yolo_world(yolo_model, device)
    sam = _load_sam(sam_model)
    
    det_high = yolo.predict(bgr_list, conf=conf_high, device=device, verbose=False)
    det_low = yolo.predict(bgr_list, conf=conf_low, device=device, verbose=False)
    
    masks_high = []
    masks_low = []
    detections_by_side_list = []
    
    for rgb, bgr, dh, dl in zip(rgb_list, bgr_list, det_high, det_low):
        h, w = rgb.shape[:2]
        img_area = h * w
        
        detections_by_side = FingerDetectionsBySide()
        
        boxes_high = dh.boxes.xyxy.cpu().numpy() if dh.boxes is not None else np.empty((0, 4))
        boxes_low = dl.boxes.xyxy.cpu().numpy() if dl.boxes is not None else np.empty((0, 4))
        
        if len(boxes_high) > 0:
            bw = boxes_high[:, 2] - boxes_high[:, 0]
            bh = boxes_high[:, 3] - boxes_high[:, 1]
            keep = (bw * bh) <= (max_box_frac * img_area)
            boxes_high = boxes_high[keep]
        
        if len(boxes_low) > 0:
            bw = boxes_low[:, 2] - boxes_low[:, 0]
            bh = boxes_low[:, 3] - boxes_low[:, 1]
            keep = (bw * bh) <= (max_box_frac * img_area)
            boxes_low = boxes_low[keep]
        
        occupied_sides = set()
        
        for box in boxes_high:
            side = get_dominant_side(box, (h, w), edge_frac)
            if side:
                getattr(detections_by_side, side).append(box)
                occupied_sides.add(side)
        
        for box in boxes_low:
            side = get_dominant_side(box, (h, w), edge_frac)
            if side and side not in occupied_sides:
                getattr(detections_by_side, side).append(box)
        
        mask_high_raw = np.zeros((h, w), dtype=np.uint8)
        if len(boxes_high) > 0:
            seg = sam.predict(bgr, bboxes=boxes_high, device=device, verbose=False)
            if seg and seg[0].masks is not None:
                data = seg[0].masks.data.cpu().numpy()
                for m in data:
                    m_bin = m > 0.5
                    if m_bin.shape != (h, w):
                        m_bin = cv2.resize(m_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    if m_bin.sum() > max_area_frac * img_area:
                        continue
                    mask_high_raw[m_bin] = 255
        
        boxes_low_filtered = []
        for box in boxes_low:
            side = get_dominant_side(box, (h, w), edge_frac)
            if side and side not in occupied_sides:
                boxes_low_filtered.append(box)
        
        mask_low_raw = np.zeros((h, w), dtype=np.uint8)
        if len(boxes_low_filtered) > 0:
            seg = sam.predict(bgr, bboxes=np.array(boxes_low_filtered), device=device, verbose=False)
            if seg and seg[0].masks is not None:
                data = seg[0].masks.data.cpu().numpy()
                for m in data:
                    m_bin = m > 0.5
                    if m_bin.shape != (h, w):
                        m_bin = cv2.resize(m_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    if m_bin.sum() > max_area_frac * img_area:
                        continue
                    mask_low_raw[m_bin] = 255
        
        mask_high = mask_high_raw
        mask_low = mask_low_raw
        
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            if int(np.count_nonzero(mask_high)) > 0:
                mask_high = cv2.dilate(mask_high, kernel, iterations=1)
            if int(np.count_nonzero(mask_low)) > 0:
                mask_low = cv2.dilate(mask_low, kernel, iterations=1)
        
        masks_high.append(mask_high)
        masks_low.append(mask_low)
        detections_by_side_list.append(detections_by_side)
    
    return masks_high, masks_low, detections_by_side_list


# ============================================================
# Итоговая сборка маски
# ============================================================


def build_finger_mask(
    rgb: np.ndarray,
    method: str = "auto",
    edge_frac: float = 0.12,
    dilate_px: int = 12,
    min_area_frac: float = 0.0015,
    device: str = "cuda",
    conf: float = 0.05,
) -> tuple[np.ndarray, str]:
    """Строит итоговую маску пальца. Возвращает (mask uint8 0/255, краткое описание).

    На кадрах без пальцев должна возвращаться пустая маска — поэтому ``auto``
    требует подтверждения от нейросети и не выдумывает маску на «тёплой» бумаге.

    method:
      - ``neural`` — только нейросеть (YOLO-World→SAM);
      - ``skin``   — только скин-прайор (склонен к ложным срабатываниям на бумаге);
      - ``auto``   — нейросеть как семантический «затвор»: берём нейромаску и
                     расширяем её скин-компонентами, которые её касаются. Если
                     нейросеть руку не нашла — маска пустая (палец считаем отсутствующим).
    """
    h, w = rgb.shape[:2]
    info = method

    if method == "skin":
        mask = skin_edge_mask(rgb, edge_frac=edge_frac, min_area_frac=min_area_frac)
    elif method == "neural":
        mask = neural_hand_mask(rgb, device=device, conf=conf)
        mask = keep_border_components(mask, edge_frac=edge_frac, min_area_frac=min_area_frac)
    elif method == "auto":
        nm = neural_hand_mask(rgb, device=device, conf=conf)
        if int(np.count_nonzero(nm)) == 0:
            mask = np.zeros((h, w), dtype=np.uint8)
            info = "auto(пусто)"
        else:
            # Расширяем нейромаску ТОЛЬКО скин-компонентами, которые её пересекают
            # (мягкие края/части пальца, недобранные детектором). Скин-полосы у
            # края, не связанные с рукой (цветная кромка бумаги, текст), не берём —
            # иначе появляются ложные маски там, где пальца нет.
            sm = morph_cleanup(skin_color_mask(rgb), ksize=max(3, int(0.006 * min(h, w))))
            sm_near = keep_seeded_components(sm, seed_mask=nm, min_area_frac=min_area_frac)
            mask = cv2.bitwise_or(nm, sm_near)
            info = "auto(neural+skin)"
    else:
        raise ValueError(f"Неизвестный метод маскирования: {method}")

    if int(np.count_nonzero(mask)) > 0:
        # Заполняем только внутренние дыры компонент (силуэт пальца должен быть
        # сплошным), НО не используем выпуклую оболочку — она раздувает маску в
        # треугольник, и инпейнтер заливает большую дыру доминирующим фоном.
        mask = fill_holes(mask)
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, info


def build_finger_mask_batch(
    rgb_list: list[np.ndarray],
    method: str = "auto",
    edge_frac: float = 0.12,
    dilate_px: int = 12,
    min_area_frac: float = 0.0015,
    device: str = "cuda",
    conf: float = 0.05,
) -> list[tuple[np.ndarray, str]]:
    """Батчевая версия build_finger_mask. Возвращает список (mask, info)."""
    if not rgb_list:
        return []
    
    results = []
    
    if method == "skin":
        for rgb in rgb_list:
            mask = skin_edge_mask(rgb, edge_frac=edge_frac, min_area_frac=min_area_frac)
            results.append((mask, method))
    elif method in ("neural", "auto"):
        neural_masks = neural_hand_mask_batch(rgb_list, device=device, conf=conf)
        
        for rgb, nm in zip(rgb_list, neural_masks):
            h, w = rgb.shape[:2]
            
            if method == "neural":
                mask = keep_border_components(nm, edge_frac=edge_frac, min_area_frac=min_area_frac)
                info = method
            else:
                if int(np.count_nonzero(nm)) == 0:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    info = "auto(пусто)"
                else:
                    sm = morph_cleanup(skin_color_mask(rgb), ksize=max(3, int(0.006 * min(h, w))))
                    sm_near = keep_seeded_components(sm, seed_mask=nm, min_area_frac=min_area_frac)
                    mask = cv2.bitwise_or(nm, sm_near)
                    info = "auto(neural+skin)"
            
            if int(np.count_nonzero(mask)) > 0:
                mask = fill_holes(mask)
                if dilate_px > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
                    mask = cv2.dilate(mask, kernel, iterations=1)
            
            results.append((mask, info))
    else:
        raise ValueError(f"Неизвестный метод маскирования: {method}")
    
    return results


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Заполняет внутренние дыры в компонентах маски (силуэт становится сплошным)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Возвращает RGB с полупрозрачной красной заливкой по маске (для отладки)."""
    out = rgb.copy()
    red = np.zeros_like(rgb)
    red[:, :, 0] = 255
    m = mask > 0
    out[m] = (alpha * red[m] + (1.0 - alpha) * rgb[m]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 3)
    return out


def overlay_mask_double(
    rgb: np.ndarray, mask_high: np.ndarray, mask_low: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Возвращает RGB с красной заливкой для mask_high и синей для mask_low."""
    out = rgb.copy()
    
    red = np.zeros_like(rgb)
    red[:, :, 0] = 255
    m_high = mask_high > 0
    out[m_high] = (alpha * red[m_high] + (1.0 - alpha) * rgb[m_high]).astype(np.uint8)
    
    blue = np.zeros_like(rgb)
    blue[:, :, 2] = 255
    m_low = mask_low > 0
    out[m_low] = (alpha * blue[m_low] + (1.0 - alpha) * rgb[m_low]).astype(np.uint8)
    
    contours_high, _ = cv2.findContours(mask_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours_high, -1, (0, 255, 0), 3)
    
    contours_low, _ = cv2.findContours(mask_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours_low, -1, (255, 255, 0), 3)
    
    return out
