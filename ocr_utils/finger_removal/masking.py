"""Построение маски пальцев на сканах.

Стратегия (по умолчанию режим ``auto``):
  1. Нейросетевой детектор (``neural_hand_mask``): open-vocabulary YOLO-World
     находит боксы руки/пальца (боксы, почти целиком вложенные в более уверенный
     бокс того же места — синонимичные классы hand/human hand/fingernail —
     отбрасываются, см. ``_suppress_nested_boxes``), затем SAM по этим боксам
     строит точный силуэт. Нейромаска — семантический «затвор»: если рука не
     найдена, маска пустая.
  2. Маска берётся как есть (без скин-цветного добора — раньше он подхватывал
     кожу, пересекающую нейромаску, но целым связным компонентом, который иногда
     оказывался огромным, если скин-тон совпадал с обложкой книги у края кадра).

Режим ``skin`` (без нейросети) использует краевое ограничение
(``keep_border_components``): оставляет компоненты у рамки кадра нужной площади.

Веса нейромоделей качаются автоматически в ``finger_models/`` (корень проекта).
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Папка для весов нейромоделей (корень проекта, по аналогии с dewarp_models/)
MODELS_DIR = Path(__file__).resolve().parents[2] / "finger_models"

# Имена ассетов ultralytics (качаются по basename, если файла нет)
DEFAULT_YOLO_WORLD = "yolov8x-worldv2.pt"
DEFAULT_SAM = "sam_b.pt"

# Классы open-vocabulary детектора, описывающие пальцы/руку.
# Без "person" — он матчит всю страницу/фото и приводит к огромным ложным маскам.
HAND_CLASSES = ["hand", "finger", "thumb", "fingertip", "human hand", "fingernail", "nail"]

# Классы-«детали» (часть пальца, не весь палец/рука) — см. _suppress_nested_boxes:
# такой бокс часто увереннее (нейросети легче узнать сам ноготь), но он МЕНЬШЕ
# настоящего пальца, поэтому не должен «побеждать» целые hand/finger-боксы.
HAND_PART_CLASSES = {"fingertip", "fingernail", "nail"}

# Во сколько раз площадь принятого "целого" бокса может расти относительно самого
# уверенного из них — отсекает боксы, раздутые за счёт смазанного/тёмного края.
FINGER_BOX_GROWTH_FACTOR = 1.5

# Классы open-vocabulary детектора для зажимов/биндеров, которыми прижимают край
# страницы. Это мелкие объекты — детектору нужно повышенное разрешение и низкий
# порог уверенности (см. detect_fingers.py: DEFAULT_CLAMP_WORK_SIDE/CONF).
CLAMP_CLASSES = ["binder clip", "colored clip", "office clip", "clip", "clamp", "bulldog clip"]

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


def _suppress_nested_boxes(
    boxes: np.ndarray,
    confs: np.ndarray,
    containment_thresh: float = 0.8,
    growth_factor: float = FINGER_BOX_GROWTH_FACTOR,
) -> np.ndarray:
    """Индексы боксов, оставленных после подавления избыточных/раздутых боксов.

    YOLO-World часто выдаёт на одно и то же место НЕСКОЛЬКО боксов разного
    масштаба (варьирующих по классу/уверенности) — от компактного вокруг
    настоящего пальца до расползающихся вдоль края кадра. У таких вложенных
    боксов IoU низкий (разный масштаб), поэтому обычный NMS их не объединяет.

    Жадно берём боксы по убыванию confidence:
      - отбрасываем кандидата, если бОльшая часть (≥ ``containment_thresh``)
        ЕГО СОБСТВЕННОЙ площади уже покрыта ранее принятыми боксами —
        асимметрично (в отличие от простого IoU/вложенности), чтобы не
        отбрасывать легитимно БОЛЬШИЙ бокс только из-за того, что внутри него
        оказался уже принятый, но более мелкий (например, самый уверенный бокс
        — это плотный кончик ногтя, а настоящий палец крупнее и должен войти
        в маску целиком, см. IMG_0049/IMG_0052);
      - ограничиваем разрастание: кандидат отбрасывается, если его площадь
        превышает площадь самого уверенного УЖЕ ПРИНЯТОГО бокса, С КОТОРЫМ ОН
        ПЕРЕСЕКАЕТСЯ, более чем в ``growth_factor`` раз. Якорь берётся ЛОКАЛЬНО
        (по первому пересекающемуся уже принятому боксу — он же самый уверенный
        в этом месте, т.к. боксы перебираются по убыванию confidence), А НЕ
        глобально по всему кадру — иначе один палец на одном краю кадра (с
        случайно более уверенным, но некрупным боксом) становится «якорем» для
        совершенно другого, пространственно не связанного пальца на другом
        краю, и все его боксы отбрасываются целиком (см. IMG_0153.jpg: боксы
        правого пальца росли от 0.1036 до 0.056/0.0527 conf, но отбрасывались
        из-за анкера от постороннего левого пальца — итоговая маска правого
        пальца схлопывалась до размера одного ногтя).
    """
    order = np.argsort(-confs)
    keep: list[int] = []
    for i in order:
        bi = boxes[i]
        area_i = max(1.0, float((bi[2] - bi[0]) * (bi[3] - bi[1])))
        redundant = False
        local_anchor_area: Optional[float] = None
        for j in keep:
            bj = boxes[j]
            ix1, iy1 = max(bi[0], bj[0]), max(bi[1], bj[1])
            ix2, iy2 = min(bi[2], bj[2]), min(bi[3], bj[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            if local_anchor_area is None:
                # Первый пересекающийся уже принятый бокс — самый уверенный в этом месте
                local_anchor_area = max(1.0, float((bj[2] - bj[0]) * (bj[3] - bj[1])))
            if inter / area_i >= containment_thresh:
                redundant = True
                break
        if redundant:
            continue
        if local_anchor_area is not None and area_i > growth_factor * local_anchor_area:
            continue
        keep.append(i)
    return np.array(keep, dtype=int)


def _select_finger_boxes(
    boxes: np.ndarray,
    confs: np.ndarray,
    cls: np.ndarray,
    containment_thresh: float = 0.8,
    growth_factor: float = FINGER_BOX_GROWTH_FACTOR,
) -> np.ndarray:
    """Отбирает финальные боксы: подавление избыточных — только среди «целых»
    классов (hand/finger/thumb/human hand), боксы-«детали» (``HAND_PART_CLASSES``
    — fingernail/fingertip/nail) включаются безусловно.

    Деталь пальца детектор часто узнаёт увереннее, чем весь палец целиком (ноготь
    — куда более характерный паттерн), но она заведомо МЕНЬШЕ настоящего пальца.
    Если пускать боксы-детали в общую конкуренцию по confidence в
    ``_suppress_nested_boxes``, самый уверенный (но мелкий) ноготь «побеждает» и
    отбрасывает более крупный (но менее уверенный) целый бокс руки — итоговая
    маска получается по размеру ногтя, а не пальца (см. IMG_0049/IMG_0052).
    """
    if len(boxes) == 0:
        return boxes
    is_part = np.array([HAND_CLASSES[k] in HAND_PART_CLASSES for k in cls])
    whole_idx = np.where(~is_part)[0]
    part_idx = np.where(is_part)[0]
    if len(whole_idx) > 0:
        kept_whole = whole_idx[_suppress_nested_boxes(boxes[whole_idx], confs[whole_idx], containment_thresh, growth_factor)]
    else:
        kept_whole = np.empty((0,), dtype=int)
    keep_idx = np.concatenate([kept_whole, part_idx]).astype(int)
    return boxes[keep_idx]


def neural_hand_mask(
    rgb: np.ndarray,
    device: str = "cuda",
    conf: float = 0.05,
    yolo_model: str = DEFAULT_YOLO_WORLD,
    sam_model: str = DEFAULT_SAM,
    max_box_frac: float = 0.30,
    max_area_frac: float = MAX_FINGER_AREA_FRAC,
    containment_thresh: float = 0.8,
    return_boxes: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """Маска пальца через YOLO-World→SAM. Возвращает uint8 0/255 (может быть пустой).

    Боксы крупнее ``max_box_frac`` кадра и SAM-маски крупнее ``max_area_frac``
    отбраковываются — палец не занимает половину снимка, такие срабатывания ложные.
    Боксы, почти целиком вложенные в уже принятый более уверенный бокс, тоже
    отбрасываются (см. ``_suppress_nested_boxes``) — иначе синонимичные классы
    (hand/human hand/fingernail) дают один и тот же палец боксами разного
    масштаба, и самый большой из них раздувает итоговую маску.

    ``return_boxes=True`` дополнительно возвращает «сырые» боксы YOLO-World —
    после фильтра ``max_box_frac``, но ДО ``_select_finger_boxes`` (т.е. все
    кандидаты, включая позже отсеянные вложенные/раздутые) — специально для
    debug-оверлея в ``detect_and_crop.py``, чтобы не гонять YOLO-World ещё раз
    только ради визуализации (раньше это был отдельный повторный проход).
    """
    h, w = rgb.shape[:2]
    img_area = h * w
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    yolo = _load_yolo_world(yolo_model, device)
    det = yolo.predict(bgr, conf=conf, device=device, verbose=False)
    boxes = det[0].boxes.xyxy.cpu().numpy() if det and det[0].boxes is not None else np.empty((0, 4))
    confs = det[0].boxes.conf.cpu().numpy() if det and det[0].boxes is not None else np.empty((0,))
    cls = det[0].boxes.cls.cpu().numpy().astype(int) if det and det[0].boxes is not None else np.empty((0,), dtype=int)

    # Отбрасываем слишком большие боксы (вся страница/обложка)
    if len(boxes) > 0:
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        keep = (bw * bh) <= (max_box_frac * img_area)
        boxes, confs, cls = boxes[keep], confs[keep], cls[keep]

    debug_boxes = boxes

    if len(boxes) > 0:
        boxes = _select_finger_boxes(boxes, confs, cls, containment_thresh)
    if len(boxes) == 0:
        mask = np.zeros((h, w), dtype=np.uint8)
        return (mask, debug_boxes) if return_boxes else mask

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
    return (mask, debug_boxes) if return_boxes else mask


def neural_hand_mask_batch(
    rgb_list: list[np.ndarray],
    device: str = "cuda",
    conf: float = 0.05,
    yolo_model: str = DEFAULT_YOLO_WORLD,
    sam_model: str = DEFAULT_SAM,
    max_box_frac: float = 0.30,
    max_area_frac: float = MAX_FINGER_AREA_FRAC,
    containment_thresh: float = 0.8,
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
        confs = det.boxes.conf.cpu().numpy() if det.boxes is not None else np.empty((0,))
        cls = det.boxes.cls.cpu().numpy().astype(int) if det.boxes is not None else np.empty((0,), dtype=int)

        if len(boxes) > 0:
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            keep = (bw * bh) <= (max_box_frac * img_area)
            boxes, confs, cls = boxes[keep], confs[keep], cls[keep]
        if len(boxes) > 0:
            boxes = _select_finger_boxes(boxes, confs, cls, containment_thresh)

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
    return_boxes: bool = False,
) -> "tuple[np.ndarray, str] | tuple[np.ndarray, str, np.ndarray]":
    """Строит итоговую маску пальца. Возвращает (mask uint8 0/255, краткое описание).

    На кадрах без пальцев должна возвращаться пустая маска — поэтому ``auto``
    требует подтверждения от нейросети и не выдумывает маску на «тёплой» бумаге.

    method:
      - ``neural`` — только нейросеть (YOLO-World→SAM);
      - ``skin``   — только скин-прайор (склонен к ложным срабатываниям на бумаге);
      - ``auto``   — нейросеть как семантический «затвор»: если нейросеть руку не
                     нашла — маска пустая (палец считаем отсутствующим), иначе
                     берём нейромаску как есть. Раньше здесь добавлялся ещё
                     скин-цветной добор (``keep_seeded_components``), но он давал
                     пренебрежимо малый выигрыш там, где нейромаска и так хороша
                     (+0.4% площади на проверенном кадре), и катастрофически
                     раздувал маску там, где скин-тона совпадали с обложкой книги
                     у края кадра — связный компонент скин-маски вбирал в себя
                     всё, что касалось нейромаски, целиком. Убран.

    ``return_boxes=True`` дополнительно возвращает «сырые» боксы YOLO-World (см.
    ``neural_hand_mask``, только для ``method`` != ``skin``, иначе — пустой массив):
    debug-оверлей в ``detect_and_crop.py`` берёт их отсюда вместо повторного
    прогона YOLO-World специально ради визуализации.
    """
    h, w = rgb.shape[:2]
    info = method
    debug_boxes = np.empty((0, 4), dtype=np.float32)

    if method == "skin":
        mask = skin_edge_mask(rgb, edge_frac=edge_frac, min_area_frac=min_area_frac)
    elif method == "neural":
        mask, debug_boxes = neural_hand_mask(rgb, device=device, conf=conf, return_boxes=True)
        mask = keep_border_components(mask, edge_frac=edge_frac, min_area_frac=min_area_frac)
    elif method == "auto":
        nm, debug_boxes = neural_hand_mask(rgb, device=device, conf=conf, return_boxes=True)
        if int(np.count_nonzero(nm)) == 0:
            mask = np.zeros((h, w), dtype=np.uint8)
            info = "auto(пусто)"
        else:
            mask = nm
            info = "auto(neural)"
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

    return (mask, info, debug_boxes) if return_boxes else (mask, info)


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
                    mask = nm
                    info = "auto(neural)"

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
    # Контур маски для наглядности
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 3)
    return out
