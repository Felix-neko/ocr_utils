"""Детекция разворота (YOLO-World + SAM), его правильный поворот и crop.

Пайплайн на каждый кадр:
  1. YOLO-World находит боксы страницы/разворота, SAM строит криволинейный силуэт,
     ``refine_page_mask`` оставляет крупнейшую область и заполняет дыры.
  2. ``min_area_rotated_bbox`` ищет «правильный поворот»: вокруг центра тяжести маски
     перебираются углы ±``ROT_RANGE_DEG`` с шагом ``ROT_STEP_DEG``, выбирается угол с
     минимальной площадью осевого bounding box.
  3. К bbox применяются припуски ``--x-margins`` / ``--y-margins`` (пиксели; >0 —
     расширить, <0 — сжать) → финальная crop-зона.
  4. Исходный кадр поворачивается на найденный угол вокруг центра тяжести, из него
     вырезается crop-зона (выпрямленный прямоугольник) и кладётся в ``--output-dir``
     под тем же именем файла.

Если задана ``--debug-dir`` — туда пишется кадр с оверлеями (всегда JPEG, ДО
удаления пальцев и компенсации уровней): зелёная граница разворота, синий
min-area bbox, фиолетовая crop-зона, красная граница обнаруженного пальца,
жёлтая ROI-рамка контекста, переданного в LaMa.

Дополнительные опции:
  - ``--output-format`` (png/tiff) — формат файлов в ``--output-dir``; по умолчанию
    как у входного файла.
  - ``--compensate-levels`` — растягивает уровни (по общей интенсивности, не по
    каналам отдельно) по перцентилям внутри маски страницы, эрозированной на
    ``--erosion-px`` (по умолчанию 20).
  - ``--upscale`` — увеличивает выходной холст перед поворотом/кропом (по
    умолчанию не задан — апскейл вообще не считается); сэмплирование всегда
    из исходного кадра.
  - ``--remove-fingers/--no-remove-fingers`` (включено по умолчанию) — перед
    детекцией разворота и кропом детектирует и закрашивает через LaMa палец,
    придерживающий страницу (``ocr_utils.finger_removal``), чтобы он не искажал
    силуэт/bbox страницы и не попадал в финальный кроп.

    uv run python -m ocr_utils.detect_and_crop \\
        --input-dir IN --output-dir OUT --debug-dir DBG --x-margins -150 --y-margins -150
"""

import logging
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import torch
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from ocr_utils.finger_removal.finger_inpaint import lama_inpaint, roi_bounds_list
from ocr_utils.finger_removal.masking import build_finger_mask, keep_border_components
from ocr_utils.finger_removal.masking import DEFAULT_YOLO_WORLD as FINGER_YOLO_WORLD
from ocr_utils.finger_removal.masking import _load_yolo_world as _load_finger_yolo_world
from ocr_utils.finger_removal.masking import _suppress_nested_boxes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Папка для весов нейромоделей (корень проекта, рядом с finger_models)
MODELS_DIR = Path(__file__).resolve().parents[1] / "finger_models"

# Цвета оверлеев в BGR (OpenCV)
COLOR_PAGE = (0, 255, 0)  # ярко-зелёный — криволинейная граница разворота
COLOR_ROT_BBOX = (255, 0, 0)  # ярко-синий — min-area повёрнутый bounding box
COLOR_CROP = (211, 0, 148)  # фиолетовый — финальная crop-зона с припусками
COLOR_FINGER = (0, 0, 255)  # красный — обнаруженная область пальца
COLOR_LAMA_ROI = (0, 255, 255)  # жёлтый — контекстная ROI-рамка, переданная в LaMa

# Поиск правильного поворота разворота: перебор углов ± предела с шагом (градусы)
ROT_RANGE_DEG = 35
ROT_STEP_DEG = 1

# Веса по умолчанию (лежат/качаются в finger_models/)
DEFAULT_YOLO_WORLD = "yolov8x-worldv2.pt"
DEFAULT_SAM = "sam_b.pt"

# Классы open-vocabulary детектора, описывающие страницу/разворот книги.
PAGE_CLASSES = ["page", "book page", "open book", "sheet of paper", "paper", "document"]

# Классы фона/подложки — конкурируют с PAGE_CLASSES за боксы, чтобы боксы,
# распознанные как ткань/подложка, не попадали в маску страницы (см. detect_page_mask).
FABRIC_CLASSES = ["fabric", "cloth", "fabric backdrop", "tablecloth"]

# Поддерживаемые форматы входных изображений (без учёта регистра расширения)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Параметры детекции
CONF = 0.05  # порог уверенности YOLO-World (выше — не хватает всю картинку как «страницу»)
WORK_SIDE = 2048  # сторона уменьшенной копии для детекции (выше = точнее контур SAM)
MIN_PAGE_FRAC = 0.05  # бокс/маска меньше этой доли кадра — это не страница
MAX_PAGE_FRAC = 1.0  # верхний предел не ставим: страница может занимать весь кадр

# Компенсация уровней: перцентили по общей интенсивности внутри маски (минус эрозия)
N_EROSION_PX = 20
LEVELS_LOW_PCT = 1.0
LEVELS_HIGH_PCT = 98.0

# Заливка фона за пределами силуэта книги (перед rotated-crop): эрозия маски
# книги перед расчётом усреднённого цвета заливки, пикс. — чтобы не захватывать
# шумную/смазанную границу силуэта.
BG_FILL_EROSION_PX = 100

# Удаление пальцев (finger_removal) перед детекцией книги/кропом
# Низкий порог нужен для recall (слабые боксы на смазанных/неярких пальцах, см.
# IMG_0028.jpg — лучший бокс conf=0.046, ниже стандартного 0.05); раздутая маска
# была из-за скин-добора и невложенных дублей боксов — то и другое уже устранено
# (skin-добор убран, _suppress_nested_boxes в neural_hand_mask), так что низкий
# conf теперь безопасен.
FINGER_CONF = 0.01
# Дилатация маски пальца (build_finger_mask default=12) — тонкая мягкая тень по
# краю силуэта (полутона на стыке кожа/бумага) иначе не докрашивается.
FINGER_DILATE_PX = 40
FINGER_EDGE_FRAC = 0.12  # доля кадра для проверки контакта с рамкой (реальный палец всегда входит с края)
FINGER_PADDING = 64  # контекст вокруг маски пальца для LaMa, пикс. (как в finger_inpaint.py)
# ROI для LaMa увеличивается в FINGER_ROI_SCALE раз от центра (после padding) —
# без этого LaMa не видит достаточно кромки/фона и заливает дыру доминирующим
# цветом (см. finger_inpaint.py, коммит "Сделали хороший закрас с помощью lama").
FINGER_ROI_SCALE = 1.5

_MODEL_CACHE: dict = {}


# ============================================================
# Удаление пальцев (перед детекцией книги/кропом)
# ============================================================


def finger_yolo_boxes(rgb: np.ndarray, device: str, conf: float, max_box_frac: float = 0.30) -> np.ndarray:
    """Боксы YOLO-World для пальцев ДО SAM (только для debug-оверлея).

    Повторяет фильтр по площади из ``neural_hand_mask`` (``masking.py``), чтобы
    показывать именно те боксы, что реально ушли в SAM.
    """
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    yolo = _load_finger_yolo_world(FINGER_YOLO_WORLD, device)
    det = yolo.predict(bgr, conf=conf, device=device, verbose=False)
    boxes = det[0].boxes.xyxy.cpu().numpy() if det and det[0].boxes is not None else np.empty((0, 4))
    if len(boxes) > 0:
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[(bw * bh) <= (max_box_frac * h * w)]
    return boxes


def remove_fingers(
    bgr: np.ndarray, device: str, conf: float = FINGER_CONF, want_boxes: bool = False
) -> tuple[np.ndarray, np.ndarray, Optional[list], Optional[np.ndarray], str]:
    """Детектирует и закрашивает пальцы (finger_removal.masking/finger_inpaint) в BGR-кадре.

    Возвращает (bgr, finger_mask, lama_roi_bboxes, yolo_boxes, info) — маска,
    список ROI-боксов LaMa (по одному на компоненту маски) и боксы YOLO-World
    нужны только для debug-оверлея, на итоговый bgr не влияют. ``yolo_boxes``
    считается лишним проходом YOLO-World и запрашивается только при
    ``want_boxes=True`` (т.е. когда включён ``--debug-dir``), чтобы не удваивать
    инференс на обычных прогонах без debug.

    Палец может исказить детекцию разворота и итоговый кроп, поэтому закраска
    выполняется до ``page_mask``/``crop_rotated``. ``build_finger_mask("auto", ...)``
    не проверяет контакт нейромаски с рамкой кадра — из-за этого крупные ФОТО
    людей/рук на самой странице (в глубине кадра, не с края) иногда ложно
    принимаются за палец. Настоящий палец всегда входит С КРАЯ кадра, поэтому
    дополнительно отсекаем компоненты, не касающиеся рамки, через
    ``keep_border_components``. Если палец не найден — кадр возвращается без
    изменений.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    yolo_boxes = finger_yolo_boxes(rgb, device, conf) if want_boxes else None
    mask, info = build_finger_mask(rgb, method="auto", device=device, conf=conf, dilate_px=FINGER_DILATE_PX)
    if int(np.count_nonzero(mask)) > 0:
        mask = keep_border_components(mask, edge_frac=FINGER_EDGE_FRAC)
        if int(np.count_nonzero(mask)) == 0:
            info = "auto(отсеяно: не у края)"
    if int(np.count_nonzero(mask)) == 0:
        return bgr, mask, None, yolo_boxes, info

    roi_bboxes = roi_bounds_list(mask, padding=FINGER_PADDING, roi_scale=FINGER_ROI_SCALE)
    rgb_clean = lama_inpaint(rgb, mask, device=device, padding=FINGER_PADDING, roi_scale=FINGER_ROI_SCALE)
    return cv2.cvtColor(rgb_clean, cv2.COLOR_RGB2BGR), mask, roi_bboxes, yolo_boxes, info


# ============================================================
# Модели и маска разворота
# ============================================================


def resolve_model_path(name: str) -> str:
    """Путь к весам в finger_models/ (качает ассет ultralytics по имени, если нужно)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODELS_DIR / name)


def load_yolo_world(name: str):
    """Ленивая загрузка YOLO-World с классами страницы + классами ткани/фона.

    Классы фона нужны, чтобы они конкурировали за боксы с классами страницы —
    тогда фон/подложка, ошибочно захваченные в один бокс со страницей, скорее
    получат класс из ``FABRIC_CLASSES`` и будут отфильтрованы в ``detect_page_mask``.
    """
    key = f"world:{name}"
    if key not in _MODEL_CACHE:
        from ultralytics import YOLOWorld

        model = YOLOWorld(resolve_model_path(name))
        model.set_classes(PAGE_CLASSES + FABRIC_CLASSES)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def load_sam(name: str):
    """Ленивая загрузка SAM."""
    key = f"sam:{name}"
    if key not in _MODEL_CACHE:
        from ultralytics import SAM

        _MODEL_CACHE[key] = SAM(resolve_model_path(name))
    return _MODEL_CACHE[key]


def refine_page_mask(mask: np.ndarray) -> np.ndarray:
    """Крупнейшая связная область + заливка дыр + сглаживание границы."""
    if int(np.count_nonzero(mask)) == 0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == biggest, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def detect_page_mask(bgr: np.ndarray, device: str) -> np.ndarray:
    """Бинарная маска (uint8 0/255) области страниц: YOLO-World боксы → SAM силуэт.

    Боксы класса из ``FABRIC_CLASSES`` (ткань/подложка) отбрасываются сразу.
    Оставшиеся боксы дополнительно прогоняются через ``_suppress_nested_boxes``
    (та же логика, что и для пальцев в ``masking.py``) — низкоуверенный бокс,
    почти целиком содержащий в себе более уверенный (например, «вся подложка +
    книга» вместо «только книга»), отбрасывается в пользу более точного.
    """
    h, w = bgr.shape[:2]
    img_area = h * w

    yolo = load_yolo_world(DEFAULT_YOLO_WORLD)
    det = yolo.predict(bgr, conf=CONF, device=device, verbose=False)
    if not det or det[0].boxes is None or len(det[0].boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    boxes = det[0].boxes.xyxy.cpu().numpy()
    confs = det[0].boxes.conf.cpu().numpy()
    cls = det[0].boxes.cls.cpu().numpy().astype(int)

    # Отбрасываем боксы, распознанные как ткань/подложка (классы после PAGE_CLASSES)
    is_page_cls = cls < len(PAGE_CLASSES)
    boxes, confs = boxes[is_page_cls], confs[is_page_cls]
    if len(boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    area = bw * bh
    size_ok = (area >= MIN_PAGE_FRAC * img_area) & (area <= MAX_PAGE_FRAC * img_area)
    boxes, confs = boxes[size_ok], confs[size_ok]
    if len(boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    boxes = boxes[_suppress_nested_boxes(boxes, confs)]
    if len(boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    sam = load_sam(DEFAULT_SAM)
    seg = sam.predict(bgr, bboxes=boxes, device=device, verbose=False)
    mask = np.zeros((h, w), dtype=np.uint8)
    if seg and seg[0].masks is not None:
        for m in seg[0].masks.data.cpu().numpy():
            m_bin = (m > 0.5).astype(np.uint8)
            if m_bin.shape != (h, w):
                m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            if MIN_PAGE_FRAC * img_area <= m_bin.sum() <= MAX_PAGE_FRAC * img_area:
                mask = cv2.bitwise_or(mask, m_bin * 255)
    return mask


def page_mask(bgr: np.ndarray, device: str) -> np.ndarray:
    """Полная маска разворота в разрешении кадра (детекция на уменьшенной копии)."""
    h, w = bgr.shape[:2]
    scale = WORK_SIDE / max(h, w) if max(h, w) > WORK_SIDE else 1.0
    work = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
    mask = detect_page_mask(work, device)
    if scale != 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return refine_page_mask(mask)


# ============================================================
# Компенсация уровней
# ============================================================


def compensate_levels(
    bgr: np.ndarray,
    mask: np.ndarray,
    erosion_px: int,
    low_pct: float = LEVELS_LOW_PCT,
    high_pct: float = LEVELS_HIGH_PCT,
) -> np.ndarray:
    """Растягивает уровни по общей интенсивности (одинаково для всех каналов).

    Перцентили считаются по пикселям внутри маски страницы, эрозированной на
    ``erosion_px`` (чтобы не захватывать край страницы/фон). Диапазон общий для
    B/G/R — это не независимая цветокоррекция по каналам, а контраст-стретч,
    сохраняющий цветовой баланс.
    """
    eroded = mask
    if erosion_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1))
        eroded = cv2.erode(mask, k)
    sel = eroded > 0
    if not np.any(sel):
        return bgr

    bgr_f = bgr.astype(np.float32) / 255.0
    lo, hi = np.percentile(bgr_f[sel], (low_pct, high_pct))
    if hi <= lo:
        return bgr

    out = rescale_intensity(bgr_f, in_range=(lo, hi), out_range=(0.0, 1.0))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# Геометрия: правильный поворот, повёрнутый bbox, crop
# ============================================================


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    """Матрица поворота 2×2 на ``angle_deg`` градусов."""
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def min_area_rotated_bbox(mask: np.ndarray) -> Optional[tuple]:
    """Возвращает (cx, cy, angle, (minx, miny, maxx, maxy)) или None.

    Центр тяжести — среднее X и Y по всем пикселям маски. Перебираем углы поворота
    вокруг центра и берём тот, у которого осевой bbox повёрнутых точек минимален по
    площади. ``ext`` — в повёрнутой системе координат (относительно центра).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)  # (N, 2) в (x, y)
    rel = pts - np.array([cx, cy])

    best = None
    for ang in range(-ROT_RANGE_DEG, ROT_RANGE_DEG + 1, ROT_STEP_DEG):
        rot = rel @ _rotation_matrix(ang).T
        mn = rot.min(axis=0)
        mx = rot.max(axis=0)
        area = (mx[0] - mn[0]) * (mx[1] - mn[1])
        if best is None or area < best[0]:
            best = (area, ang, (mn[0], mn[1], mx[0], mx[1]))

    _, angle, ext = best
    return cx, cy, angle, ext


def _ext_with_margins(ext: tuple, mx: int, my: int) -> tuple:
    """Применяет припуски к ext (minx, miny, maxx, maxy): >0 расширяет, <0 сжимает."""
    minx, miny, maxx, maxy = ext
    return (minx - mx, miny - my, maxx + mx, maxy + my)


def _bbox_corners(cx: float, cy: float, angle: float, ext: tuple) -> np.ndarray:
    """4 угла повёрнутого bbox в координатах исходного кадра (порядок TL,TR,BR,BL)."""
    minx, miny, maxx, maxy = ext
    corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]], dtype=np.float64)
    # Обратно в исходный кадр: rel = rot @ R(angle), затем + центр
    return (corners @ _rotation_matrix(angle) + np.array([cx, cy])).astype(np.float32)


def fill_outside_mask(bgr: np.ndarray, mask: np.ndarray, erosion_px: int = BG_FILL_EROSION_PX) -> np.ndarray:
    """Закрашивает всё вне ``mask`` усреднённым цветом внутри неё.

    Криволинейная маска страницы не идеально совпадает с осевым min-area bbox
    (неровные/загнутые края) — в углы повёрнутого кропа может попасть кусок
    чёрного фона. Заранее закрасив фон усреднённым цветом книги, получаем
    ровный угол вместо чёрного пятна, даже если crop-зона чуть шире силуэта.
    Цвет считается по маске, эрозированной на ``erosion_px`` — чтобы не задеть
    шумную/смазанную границу силуэта (там же соседствует фон).
    """
    sel = mask > 0
    if not np.any(sel):
        return bgr
    sample_sel = sel
    if erosion_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1))
        eroded = cv2.erode(mask, k)
        if np.any(eroded > 0):
            sample_sel = eroded > 0
    avg_color = bgr[sample_sel].mean(axis=0)
    out = bgr.copy()
    out[~sel] = avg_color.astype(np.uint8)
    return out


def crop_rotated(
    bgr: np.ndarray, cx: float, cy: float, angle: float, ext: tuple, mx: int, my: int, upscale: Optional[float] = None
) -> np.ndarray:
    """Поворот вокруг центра тяжести + вырез crop-зоны → выпрямленный прямоугольник.

    Берём 4 угла crop-зоны в исходном кадре и перспективным преобразованием
    отображаем их в осевой прямоугольник нужного размера (это и есть поворот кадра
    на найденный угол с одновременным вырезом области). ``upscale`` увеличивает
    только выходной холст (источник сэмплирования — всегда исходный полноразмерный
    кадр), поэтому апскейл получается за один интерполяционный проход, без потерь
    от промежуточного ресайза целого кадра. ``None`` — апскейл вообще не считается
    (экономит время: без умножения размеров и без INTER_CUBIC).
    """
    minx, miny, maxx, maxy = _ext_with_margins(ext, mx, my)
    if upscale is None:
        out_w = max(1, int(round(maxx - minx)))
        out_h = max(1, int(round(maxy - miny)))
        flags = cv2.INTER_LINEAR
    else:
        out_w = max(1, int(round((maxx - minx) * upscale)))
        out_h = max(1, int(round((maxy - miny) * upscale)))
        flags = cv2.INTER_CUBIC
    src = _bbox_corners(cx, cy, angle, (minx, miny, maxx, maxy))
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, m, (out_w, out_h), flags=flags)


def _draw_dashed_line(
    img: np.ndarray, pt1: tuple, pt2: tuple, color: tuple, thickness: int, dash_len: int = 20, gap_len: int = 14
) -> None:
    """Пунктирная линия pt1→pt2 (cv2 не умеет рисовать пунктир нативно)."""
    x1, y1 = pt1
    x2, y2 = pt2
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 1:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    draw = True
    while pos < length:
        seg_end = min(pos + (dash_len if draw else gap_len), length)
        if draw:
            p1 = (int(round(x1 + dx * pos)), int(round(y1 + dy * pos)))
            p2 = (int(round(x1 + dx * seg_end)), int(round(y1 + dy * seg_end)))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        pos = seg_end
        draw = not draw


def _draw_dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, thickness: int) -> None:
    """Пунктирный прямоугольник (используется для YOLO-World bbox пальца до SAM)."""
    for p1, p2 in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        _draw_dashed_line(img, p1, p2, color, thickness)


def draw_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    geom: Optional[tuple],
    mx: int,
    my: int,
    finger_mask: Optional[np.ndarray] = None,
    lama_roi_bboxes: Optional[list] = None,
    finger_boxes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Кадр с оверлеями: зелёная граница разворота, синий min-bbox, фиолетовая crop-зона,
    красная граница обнаруженного пальца (после SAM), красный пунктирный bbox от
    YOLO-World (до SAM), жёлтая ROI-рамка контекста для LaMa.

    Рисуется поверх ``bgr`` ДО удаления пальцев и компенсации уровней — оверлей
    должен показывать, что было найдено, а не результат обработки.
    """
    h, w = bgr.shape[:2]
    out = bgr.copy()
    thickness = max(2, int(round(max(h, w) / 500)))
    if int(np.count_nonzero(mask)) > 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_PAGE, thickness, lineType=cv2.LINE_AA)
    if geom is not None:
        cx, cy, angle, ext = geom
        bbox = _bbox_corners(cx, cy, angle, ext).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [bbox], True, COLOR_ROT_BBOX, thickness, cv2.LINE_AA)
        crop = _bbox_corners(cx, cy, angle, _ext_with_margins(ext, mx, my)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [crop], True, COLOR_CROP, thickness, cv2.LINE_AA)
    if lama_roi_bboxes is not None:
        for x1, y1, x2, y2 in lama_roi_bboxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_LAMA_ROI, thickness, cv2.LINE_AA)
    if finger_boxes is not None:
        for bx in finger_boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in bx)
            _draw_dashed_rect(out, x1, y1, x2, y2, COLOR_FINGER, thickness)
    if finger_mask is not None and int(np.count_nonzero(finger_mask)) > 0:
        contours, _ = cv2.findContours(finger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_FINGER, thickness, lineType=cv2.LINE_AA)
    return out


# ============================================================
# Сбор файлов и сохранение
# ============================================================


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    """Собирает изображения (по расширению, без учёта регистра).

    ``recursive=False`` — только верхний уровень каталога; ``True`` — рекурсивно.
    """
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _imwrite_params(suffix: str) -> list[int]:
    """Параметры cv2.imwrite под формат (качество JPEG / сжатие PNG)."""
    s = suffix.lower()
    if s in (".jpg", ".jpeg"):
        return [cv2.IMWRITE_JPEG_QUALITY, 95]
    if s == ".png":
        return [cv2.IMWRITE_PNG_COMPRESSION, 3]
    return []


def _resolve_output_suffix(orig_suffix: str, output_format: Optional[str]) -> str:
    """Суффикс выходного файла: как у входа, если ``output_format`` не задан."""
    if output_format is None:
        return orig_suffix
    return ".png" if output_format.lower() == "png" else ".tiff"


# ============================================================
# CLI
# ============================================================


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Каталог с исходными изображениями (JPG/PNG)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Куда сохранять повёрнутые и обрезанные развороты (имя файла сохраняется)",
)
@click.option(
    "--debug-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Если задана — сюда кадр с оверлеями (граница/min-bbox/crop-зона)",
)
@click.option("--x-margins", default=0, show_default=True, help="Припуск crop-зоны по X, пикс. (>0 шире, <0 уже)")
@click.option("--y-margins", default=0, show_default=True, help="Припуск crop-зоны по Y, пикс. (>0 шире, <0 уже)")
@click.option("--recursive", is_flag=True, default=False, help="Рекурсивно обходить подкаталоги в поисках картинок")
@click.option(
    "--output-format",
    type=click.Choice(["png", "tiff"], case_sensitive=False),
    default=None,
    help="Формат файлов в output-dir (по умолчанию — как у входного файла)",
)
@click.option(
    "--compensate-levels/--no-compensate-levels",
    "do_compensate_levels",
    default=False,
    show_default=True,
    help="Растягивать уровни по перцентилям внутри маски страницы (минус эрозия)",
)
@click.option(
    "--erosion-px",
    default=N_EROSION_PX,
    show_default=True,
    help="Эрозия маски страницы (пикс.) перед расчётом уровней (--compensate-levels)",
)
@click.option(
    "--upscale",
    default=None,
    type=float,
    show_default=True,
    help="Апскейл выходного изображения перед поворотом/кропом (по умолчанию — без апскейла)",
)
@click.option(
    "--remove-fingers/--no-remove-fingers",
    "do_remove_fingers",
    default=True,
    show_default=True,
    help="Детектировать и закрашивать пальцы (finger_removal) перед детекцией книги/кропом",
)
def main(
    input_dir: Path,
    output_dir: Path,
    debug_dir: Optional[Path],
    x_margins: int,
    y_margins: int,
    recursive: bool,
    output_format: Optional[str],
    do_compensate_levels: bool,
    erosion_px: int,
    upscale: Optional[float],
    do_remove_fingers: bool,
) -> None:
    """Находит разворот, выпрямляет его поворотом и вырезает crop-зону в OUTPUT_DIR."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = collect_images(input_dir, recursive)
    if not files:
        logger.warning("Изображения не найдены в %s", input_dir)
        return

    logger.info(
        "Файлов: %d | устройство: %s | margins: x=%d y=%d | recursive: %s | "
        "output-format: %s | compensate-levels: %s (erosion-px=%d) | upscale: %s | remove-fingers: %s",
        len(files),
        device,
        x_margins,
        y_margins,
        recursive,
        output_format or "как у входа",
        do_compensate_levels,
        erosion_px,
        upscale if upscale is not None else "без апскейла",
        do_remove_fingers,
    )

    for path in tqdm(files, desc="Crop", unit="img"):
        try:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                tqdm.write(f"  Не удалось загрузить: {path.name}")
                continue

            bgr_orig = bgr  # для debug-оверлея: без удаления пальцев и без компенсации уровней
            finger_mask: Optional[np.ndarray] = None
            lama_roi_bboxes: Optional[list] = None
            finger_boxes: Optional[np.ndarray] = None
            if do_remove_fingers:
                bgr, finger_mask, lama_roi_bboxes, finger_boxes, finger_info = remove_fingers(
                    bgr, device, want_boxes=debug_dir is not None
                )
                if int(np.count_nonzero(finger_mask)) > 0:
                    tqdm.write(f"  Пальцы: {finger_info} ({path.name})")

            mask = page_mask(bgr, device)
            geom = min_area_rotated_bbox(mask)
            bgr_leveled = compensate_levels(bgr, mask, erosion_px) if do_compensate_levels else bgr

            # При recursive — зеркалим подкаталоги; формат — из --output-format либо как у входа
            rel = path.relative_to(input_dir)
            out_suffix = _resolve_output_suffix(path.suffix, output_format)
            params = _imwrite_params(out_suffix)
            out_path = (output_dir / rel).with_suffix(out_suffix)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if geom is None:
                # Разворот не найден — кладём оригинал, чтобы не терять файл в пайплайне
                tqdm.write(f"  Разворот не найден, сохраняю оригинал: {rel}")
                cv2.imwrite(str(out_path), bgr_leveled, params)
            else:
                cx, cy, angle, ext = geom
                bgr_for_crop = fill_outside_mask(bgr_leveled, mask)
                crop = crop_rotated(bgr_for_crop, cx, cy, angle, ext, x_margins, y_margins, upscale)
                cv2.imwrite(str(out_path), crop, params)

            if debug_dir is not None:
                dbg_path = (debug_dir / rel).with_suffix(".jpg")
                dbg_path.parent.mkdir(parents=True, exist_ok=True)
                overlay = draw_overlay(
                    bgr_orig, mask, geom, x_margins, y_margins, finger_mask, lama_roi_bboxes, finger_boxes
                )
                cv2.imwrite(str(dbg_path), overlay, _imwrite_params(".jpg"))

        except Exception as e:
            tqdm.write(f"  Ошибка {path.name}: {e}")
            import traceback

            tqdm.write(traceback.format_exc())

    logger.info("Готово. Crop → %s%s", output_dir, f" | debug → {debug_dir}" if debug_dir else "")


if __name__ == "__main__":
    main()
