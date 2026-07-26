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

Сами сети (YOLO-World, SAM) живут в ``scan_cropping.gpu_models.GpuModels``; здесь
только правила отбора боксов и обработка масок.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ocr_utils.scan_cropping.finger_removal.asymmetric_dilation import (
    DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO,
    dilate_finger_zones,
)
from ocr_utils.scan_cropping.gpu_models import HAND_CLASSES
from ocr_utils.timing import log_timing

logger = logging.getLogger(__name__)

# Классы-«детали» (часть пальца, не весь палец/рука) — см. _suppress_nested_boxes:
# такой бокс часто увереннее (нейросети легче узнать сам ноготь), но он МЕНЬШЕ
# настоящего пальца, поэтому не должен «побеждать» целые hand/finger-боксы.
HAND_PART_CLASSES = {"fingertip", "fingernail", "nail"}

# Во сколько раз площадь принятого "целого" бокса может расти относительно самого
# уверенного из них — отсекает боксы, раздутые за счёт смазанного/тёмного края.
FINGER_BOX_GROWTH_FACTOR = 1.5

# Палец никогда не занимает больше этой доли кадра — отсекаем гигантские ложные маски
MAX_FINGER_AREA_FRAC = 0.12

# Доля площади СЫРОГО (до дилатации) блоба пальца, лежащая внутри контентного блока
# Surya, при которой блоб считается ложным пальцем на печатном контенте (лицо на
# портрете). См. drop_fingers_on_content.
FINGER_LAYOUT_OVERLAP_DROP = 0.50


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


def drop_fingers_on_content(
    mask: np.ndarray,
    predilate: np.ndarray,
    layout_mask: np.ndarray,
    dilate_px: int,
    asymmetric_dilation_ratio: float,
    edge_frac: float,
    overlap_thr: float = FINGER_LAYOUT_OVERLAP_DROP,
) -> "tuple[np.ndarray, np.ndarray, int]":
    """Убирает ложные «пальцы», распознанные на печатном контенте.

    Компонент СЫРОЙ (до дилатации) маски ``predilate`` считается ложным, если
    ≥``overlap_thr`` его площади лежит внутри контентных блоков ``layout_mask`` И
    он НЕ касается краевой рамки кадра (ширина ``edge_frac``). Логика: настоящий
    палец входит С КРАЯ кадра по чистой бумаге, поэтому касается рамки и в блок
    Surya не попадает; а лицо на напечатанном портрете — внутреннее и целиком
    внутри блока Surya (детектор пальца принимает его за кожу).

    Оценка ведётся ДО дилатации: она склеивает ложный блоб (портрет) с настоящим
    пальцем в один компонент, и на раздутой маске доля «внутри блока» размывается.

    Убрав ложные компоненты из сырой маски, ЗАНОВО строим дилатированную маску (те
    же параметры дилатации + отсев не-краевых компонент через
    ``keep_border_components``), чтобы вместе с ложным ядром ушла и его раздутая
    кайма.

    Возвращает (дилатированная маска, сырая маска, число убранных компонентов).
    """
    if int(np.count_nonzero(predilate)) == 0 or int(np.count_nonzero(layout_mask)) == 0:
        return mask, predilate, 0

    h, w = predilate.shape[:2]
    frame_w = max(1, int(edge_frac * w))
    frame_h = max(1, int(edge_frac * h))
    border = np.zeros((h, w), dtype=bool)
    border[:frame_h, :] = True
    border[h - frame_h :, :] = True
    border[:, :frame_w] = True
    border[:, w - frame_w :] = True
    layout_bool = layout_mask > 0

    num, labels = cv2.connectedComponents((predilate > 0).astype(np.uint8), connectivity=8)
    cleaned = predilate.copy()
    dropped = 0
    for i in range(1, num):
        comp = labels == i
        area = int(comp.sum())
        if area == 0:
            continue
        inside = int(np.count_nonzero(comp & layout_bool))
        if inside / area < overlap_thr:
            continue
        if bool(np.any(comp & border)):  # касается рамки — настоящий палец, не трогаем
            continue
        cleaned[comp] = 0
        dropped += 1

    if dropped == 0:
        return mask, predilate, 0
    if int(np.count_nonzero(cleaned)) == 0:
        return cleaned, cleaned, dropped  # все компоненты оказались ложными — пустая маска

    new_mask = (
        dilate_finger_zones(cleaned, dilate_px, max_ratio=asymmetric_dilation_ratio) if dilate_px > 0 else cleaned
    )
    new_mask = keep_border_components(new_mask, edge_frac=edge_frac)
    return new_mask, cleaned, dropped


def skin_edge_mask(rgb: np.ndarray, edge_frac: float = 0.12, min_area_frac: float = 0.0015) -> np.ndarray:
    """Полный скин-прайор: цвет кожи → морфология → краевые компоненты."""
    mask = skin_color_mask(rgb)
    mask = morph_cleanup(mask, ksize=max(3, int(0.006 * min(rgb.shape[:2]))))
    mask = keep_border_components(mask, edge_frac=edge_frac, min_area_frac=min_area_frac)
    return mask


# ============================================================
# Нейросетевой детектор: YOLO-World (боксы) → SAM (силуэт)
# ============================================================


def _box_uncovered_fraction(cand: np.ndarray, others: list[np.ndarray], grid: int = 256) -> float:
    """Доля площади бокса ``cand``, НЕ покрытая объединением боксов ``others``.

    Считается на грубом растре внутри ``cand`` (бОльшая сторона до ``grid`` ячеек):
    точная площадь объединения прямоугольников не нужна, боксов единицы. Возвращает
    1.0, если ``cand`` не пересекается ни с одним из ``others`` (вся площадь новая),
    и 0.0, если полностью ими покрыт.
    """
    x1, y1, x2, y2 = cand[0], cand[1], cand[2], cand[3]
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return 0.0
    scale = grid / max(cw, ch)
    gw, gh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    cov = np.zeros((gh, gw), dtype=bool)
    for bj in others:
        ix1, iy1 = max(x1, bj[0]), max(y1, bj[1])
        ix2, iy2 = min(x2, bj[2]), min(y2, bj[3])
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        gx1, gx2 = int((ix1 - x1) * scale), int(round((ix2 - x1) * scale))
        gy1, gy2 = int((iy1 - y1) * scale), int(round((iy2 - y1) * scale))
        cov[gy1:gy2, gx1:gx2] = True
    return 1.0 - float(cov.mean())


def _suppress_nested_boxes(
    boxes: np.ndarray,
    confs: np.ndarray,
    containment_thresh: float = 0.8,
    growth_factor: float = FINGER_BOX_GROWTH_FACTOR,
    keep_new_area_frac: Optional[float] = None,
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

    ``keep_new_area_frac`` (по умолчанию None — выключено, поведение для пальцев
    не меняется): исключение из ограничителя разрастания для ДЕТЕКЦИИ СТРАНИЦ.
    Бокс, переросший локальный якорь сверх ``growth_factor``, всё же оставляется,
    если он добавляет ≥ этой доли площади, ЕЩЁ НЕ покрытой принятыми боксами
    (``_box_uncovered_fraction``). Для пальца «раздутый» бокс — это тот же палец,
    расползшийся вдоль края (новой площади добавляет мало полезного, его надо
    резать), а для разворота YOLO-World нередко НЕ даёт отдельного бокса на
    страницу, целиком занятую фото/иллюстрацией, и покрывает её только «широким»
    боксом на весь разворот; такой бокс переростает якорь-соседа (одну страницу),
    но добавляет цельную вторую страницу как новую площадь — его нужно сохранить
    (см. detect_and_crop.detect_page_mask, IMG_0004.jpg 1972/04).
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
            # Перерос локального якоря. По умолчанию режем (защита от раздувания
            # маски пальца). Для страниц (keep_new_area_frac задан) оставляем, если
            # бокс добавляет достаточно НЕ покрытой принятыми боксами площади —
            # это не раздутый дубль, а отдельная крупная область (вторая страница).
            if keep_new_area_frac is None or _box_uncovered_fraction(bi, [boxes[j] for j in keep]) < keep_new_area_frac:
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
        kept_whole = whole_idx[
            _suppress_nested_boxes(boxes[whole_idx], confs[whole_idx], containment_thresh, growth_factor)
        ]
    else:
        kept_whole = np.empty((0,), dtype=int)
    keep_idx = np.concatenate([kept_whole, part_idx]).astype(int)
    return boxes[keep_idx]


def neural_hand_mask(
    rgb: np.ndarray,
    models,
    conf: float = 0.05,
    max_box_frac: float = 0.30,
    max_area_frac: float = MAX_FINGER_AREA_FRAC,
    containment_thresh: float = 0.8,
    return_boxes: bool = False,
    log_name: str = "",
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """Маска пальца через YOLO-World→SAM. Возвращает uint8 0/255 (может быть пустой).

    ``models`` — ``scan_cropping.gpu_models.GpuModels``: сети живут там, здесь —
    только правила отбора боксов и масок.

    Боксы крупнее ``max_box_frac`` кадра и SAM-маски крупнее ``max_area_frac``
    отбраковываются — палец не занимает половину снимка, такие срабатывания ложные.
    Боксы, почти целиком вложенные в уже принятый более уверенный бокс, тоже
    отбрасываются (см. ``_suppress_nested_boxes``) — иначе синонимичные классы
    (hand/human hand/fingernail) дают один и тот же палец боксами разного
    масштаба, и самый большой из них раздувает итоговую маску.

    ``return_boxes=True`` дополнительно возвращает «сырые» боксы YOLO-World —
    после фильтра ``max_box_frac``, но ДО ``_select_finger_boxes`` (т.е. все
    кандидаты, включая позже отсеянные вложенные/раздутые) — специально для
    debug-оверлея в ``scan_cropping.overlay``, чтобы не гонять YOLO-World ещё раз
    только ради визуализации (раньше это был отдельный повторный проход).
    """
    h, w = rgb.shape[:2]
    img_area = h * w
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    with log_timing("yolo_predict", log_name, log=logger):
        boxes, confs, cls = models.detect_hand_boxes(bgr, conf=conf)

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

    with log_timing("sam_predict", log_name, log=logger):
        sam_masks = models.segment_boxes(bgr, boxes)
    mask = np.zeros((h, w), dtype=np.uint8)
    for m_bin in sam_masks:
        # Отбрасываем отдельные маски, покрывающие слишком большую долю кадра
        if m_bin.sum() > max_area_frac * img_area:
            continue
        mask[m_bin] = 255
    return (mask, debug_boxes) if return_boxes else mask


# ============================================================
# Итоговая сборка маски
# ============================================================


def build_finger_mask(
    rgb: np.ndarray,
    models,
    method: str = "auto",
    edge_frac: float = 0.12,
    dilate_px: int = 12,
    min_area_frac: float = 0.0015,
    conf: float = 0.05,
    return_boxes: bool = False,
    return_predilate: bool = False,
    asymmetric_dilation_ratio: float = DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO,
    log_name: str = "",
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
    debug-оверлей в ``scan_cropping.overlay`` берёт их отсюда вместо повторного
    прогона YOLO-World специально ради визуализации.

    ``return_predilate=True`` дополнительно возвращает (последним элементом) маску
    ДО дилатации — первичную зону пальца после SAM и заливки дыр. Нужна debug-оверлею,
    чтобы показать первичную и раздутую зоны разным стилем линий.
    """
    h, w = rgb.shape[:2]
    info = method
    debug_boxes = np.empty((0, 4), dtype=np.float32)

    if method == "skin":
        mask = skin_edge_mask(rgb, edge_frac=edge_frac, min_area_frac=min_area_frac)
    elif method == "neural":
        with log_timing("neural_hand_mask", log_name, log=logger):
            mask, debug_boxes = neural_hand_mask(rgb, models, conf=conf, return_boxes=True, log_name=log_name)
        mask = keep_border_components(mask, edge_frac=edge_frac, min_area_frac=min_area_frac)
    elif method == "auto":
        with log_timing("neural_hand_mask", log_name, log=logger):
            nm, debug_boxes = neural_hand_mask(rgb, models, conf=conf, return_boxes=True, log_name=log_name)
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
        with log_timing("fill_holes", log_name, log=logger):
            mask = fill_holes(mask)
        mask_predilate = mask.copy()  # первичная зона (после SAM + заливки дыр) — для debug-оверлея
        if dilate_px > 0:
            # Асимметричная дилатация: каждую зону пальца растим сильнее вдоль той
            # стороны кадра, к которой она прилегает (там и лежит её тень).
            with log_timing("dilate_finger_zones", log_name, log=logger):
                mask = dilate_finger_zones(mask, dilate_px, max_ratio=asymmetric_dilation_ratio)
    else:
        mask_predilate = mask.copy()

    result = (mask, info, debug_boxes) if return_boxes else (mask, info)
    return (*result, mask_predilate) if return_predilate else result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Заполняет внутренние дыры в компонентах маски (силуэт становится сплошным)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out
