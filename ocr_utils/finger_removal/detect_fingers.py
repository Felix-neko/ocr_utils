"""Обводка пальцев/рук на сырых сканах криволинейной красной границей.

Детектор YOLO находит боксы руки, затем SAM по этим боксам строит точный силуэт;
рисуется КРАСНЫЙ контур силуэта (а не bounding box).

Поддерживаются два режима детектора (опция ``--finger-model``):
  - обычная YOLO closed-vocabulary, напр. ``Bingsu/adetailer:hand_yolov8s.pt`` или
    ``EtanHey/hand-detection-3class:model.pt``;
  - open-vocabulary YOLO-World — префикс ``world:`` (классы берутся из
    ``HAND_CLASSES`` в masking.py: рука/палец/ноготь…), напр.
    ``world:yolov8x-worldv2.pt``.

Модели задаются как ``repo_id:filename`` (качается из HuggingFace в
``finger_models/``) либо путём к локальному ``.pt``.

Пример:
    uv run python -m ocr_utils.finger_removal.detect_fingers \\
        --input-dir  /путь/к/сканам \\
        --output-dir ocr_utils/finger_removal/detected_fingers \\
        --finger-model world:yolov8x-worldv2.pt --conf 0.05 --limit 5
"""

import logging
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import torch
from tqdm import tqdm

from ocr_utils.finger_removal.finger_inpaint import (
    DEFAULT_ROI_SCALE,
    DEFAULT_SD_MODEL as SD_DEFAULT_MODEL,
    DEFAULT_SD_NEGATIVE as SD_DEFAULT_NEGATIVE,
    DEFAULT_SD_PROMPT as SD_DEFAULT_PROMPT,
    inpaint_fingers,
    roi_bounds_list,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Папка для весов нейромоделей (корень проекта, рядом с finger_models)
MODELS_DIR = Path(__file__).resolve().parents[2] / "finger_models"

# Цвета в BGR (OpenCV)
COLOR_FINGER = (0, 0, 255)  # красный — граница маски пальца
COLOR_ROI = (0, 255, 255)  # жёлтый — ROI, скармливаемый инпейнтеру
COLOR_CLAMP = (255, 0, 0)  # синий — граница маски зажима/биндера для бумаги

# Дефолтные спецификации моделей. По умолчанию — open-vocabulary YOLO-World
# (классы из HAND_CLASSES, включая «ноготь»).
DEFAULT_FINGER_MODEL = "world:yolov8x-worldv2.pt"
DEFAULT_SAM_MODEL = "sam_b.pt"  # локальный вес в finger_models/

# Дилатация объединённой маски пальцев по умолчанию, пикс. (полное разрешение)
DEFAULT_DILATE_PX = 24

# --- Детекция зажимов (биндеров/скрепок) для бумаги -----------------------
# Зажим ищется тем же стеком, что и пальцы: open-vocabulary YOLO-World (классы
# CLAMP_CLASSES из masking.py) находит бокс, SAM строит силуэт. Зажим — мелкий
# объект, поэтому детекции нужно повышенное разрешение (DEFAULT_CLAMP_WORK_SIDE)
# и низкий порог уверенности (DEFAULT_CLAMP_CONF), иначе он теряется на скане.
DEFAULT_CLAMP_MODEL = "world:yolov8x-worldv2.pt"  # YOLO-World с классами CLAMP_CLASSES
DEFAULT_CLAMP_CONF = 0.05  # порог уверенности YOLO-World для зажима (мелкий объект — низкий)
DEFAULT_CLAMP_WORK_SIDE = 2400  # сторона уменьшенной копии для детекции зажима, пикс.
MAX_CLAMP_BOX_FRAC = 0.20  # зажим не занимает больше этой доли кадра — отсекаем ложные боксы
DEFAULT_CLAMP_DILATE_PX = 36  # дилатация маски зажима, пикс. (захватить металлические усики)
# Усики зажима откинуты на внешнюю сторону — торчат ОТ центра листа к краю
# кадра (поверх тёмного фона), SAM их не обводит. Достраиваем маску
# прямоугольником от внешней грани тела к ближайшему краю кадра на эту долю
# ширины/высоты тела (0 — не достраивать).
DEFAULT_CLAMP_HANDLE_EXTEND = 1.0

# Палец/рука не занимает больше этой доли кадра — отсекаем ложные гигантские боксы
MAX_FINGER_BOX_FRAC = 0.35

# Кэш загруженных моделей
_MODEL_CACHE: dict = {}


# ============================================================
# Загрузка моделей
# ============================================================


def resolve_model_path(spec: str) -> str:
    """Превращает спецификацию модели в путь к локальному файлу весов.

    Поддерживает два формата:
      - ``repo_id:filename`` — качает файл из HuggingFace Hub в ``finger_models/``;
      - путь к существующему ``.pt`` — возвращается как есть.
    """
    # Локальный файл?
    p = Path(spec)
    if p.exists():
        return str(p)

    # Просто имя файла, уже лежащего в finger_models/?
    local = MODELS_DIR / spec
    if local.exists():
        return str(local)

    if ":" not in spec:
        raise ValueError(f"Не удалось разрешить модель '{spec}'. Укажите путь к .pt или формат 'repo_id:filename'.")

    repo_id, filename = spec.split(":", 1)
    from huggingface_hub import hf_hub_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Качаю %s из %s …", filename, repo_id)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(MODELS_DIR))
    return path


def load_yolo(spec: str):
    """Ленивая загрузка обычной YOLO-модели (детекция) с кэшем."""
    key = f"yolo:{spec}"
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO

        _MODEL_CACHE[key] = YOLO(resolve_model_path(spec))
    return _MODEL_CACHE[key]


def load_yolo_world(spec: str, classes: Optional[list[str]] = None, cache_tag: str = "hand"):
    """Ленивая загрузка open-vocabulary YOLO-World с кэшем.

    Классы задаются ``classes``; по умолчанию берутся ``HAND_CLASSES`` (masking.py)
    — рука/палец/ноготь и т.п. ``cache_tag`` разделяет кэш для разных наборов
    классов (один и тот же вес можно настроить и на пальцы, и на зажимы).
    """
    key = f"world:{cache_tag}:{spec}"
    if key not in _MODEL_CACHE:
        from ultralytics import YOLOWorld

        if classes is None:
            from ocr_utils.finger_removal.masking import HAND_CLASSES

            classes = HAND_CLASSES

        model = YOLOWorld(resolve_model_path(spec))
        model.set_classes(classes)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def load_sam(spec: str):
    """Ленивая загрузка SAM с кэшем."""
    key = f"sam:{spec}"
    if key not in _MODEL_CACHE:
        from ultralytics import SAM

        _MODEL_CACHE[key] = SAM(resolve_model_path(spec))
    return _MODEL_CACHE[key]


# ============================================================
# Детекция пальцев → SAM-силуэт
# ============================================================


def _yolo_sam_masks(
    bgr: np.ndarray,
    yolo,
    sam_model: Optional[str],
    device: str,
    conf: float,
    max_box_frac: float,
    drop_negative_names: bool = False,
    imgsz: Optional[int] = None,
) -> list[np.ndarray]:
    """Общий конвейер «YOLO-бокс → SAM-силуэт». Возвращает список масок uint8 0/255.

    YOLO-детектор ``yolo`` находит боксы, слишком крупные (> ``max_box_frac`` кадра)
    отбрасываются; при ``drop_negative_names`` выкидываются классы вроде
    ``not_hand``/``background``. По оставшимся боксам SAM строит криволинейный
    силуэт; без SAM (``sam_model is None``) маской служит прямоугольник бокса.
    """
    h, w = bgr.shape[:2]
    img_area = h * w

    det = yolo.predict(bgr, conf=conf, device=device, verbose=False, imgsz=imgsz or max(h, w))
    if not det or det[0].boxes is None or len(det[0].boxes) == 0:
        return []

    boxes = det[0].boxes.xyxy.cpu().numpy()
    cls = det[0].boxes.cls.cpu().numpy().astype(int)
    names = det[0].names

    keep = []
    for i in range(len(boxes)):
        if drop_negative_names:
            name = str(names.get(cls[i], "")).lower()
            if "not" in name or "background" in name:
                continue
        bw = boxes[i, 2] - boxes[i, 0]
        bh = boxes[i, 3] - boxes[i, 1]
        if bw * bh > max_box_frac * img_area:
            continue
        keep.append(i)
    boxes = boxes[keep]
    if len(boxes) == 0:
        return []

    if sam_model is None:
        masks = []
        for x1, y1, x2, y2 in boxes.astype(int):
            m = np.zeros((h, w), dtype=np.uint8)
            m[y1:y2, x1:x2] = 255
            masks.append(m)
        return masks

    sam = load_sam(sam_model)
    seg = sam.predict(bgr, bboxes=boxes, device=device, verbose=False)
    masks = []
    if seg and seg[0].masks is not None:
        data = seg[0].masks.data.cpu().numpy()
        for m in data:
            m_bin = (m > 0.5).astype(np.uint8)
            if m_bin.shape != (h, w):
                m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(m_bin * 255)
    return masks


def detect_finger_masks(
    bgr: np.ndarray, finger_model: str, sam_model: Optional[str], device: str, conf: float, world: bool = False
) -> list[np.ndarray]:
    """Возвращает список бинарных масок (uint8 0/255) пальцев/рук.

    Сначала YOLO-детектор находит боксы руки, затем (если задан) SAM строит по ним
    криволинейный силуэт. Без SAM маской служит прямоугольник бокса. При ``world``
    используется open-vocabulary YOLO-World с классами ``HAND_CLASSES``.
    """
    yolo = load_yolo_world(finger_model) if world else load_yolo(finger_model)
    return _yolo_sam_masks(bgr, yolo, sam_model, device, conf, MAX_FINGER_BOX_FRAC, drop_negative_names=True)


# ============================================================
# Детекция зажимов (биндеров) для бумаги: YOLO-World → SAM
# ============================================================


def detect_clamp_masks(
    bgr: np.ndarray, clamp_model: str, sam_model: Optional[str], device: str, conf: float
) -> list[np.ndarray]:
    """Возвращает список бинарных масок (uint8 0/255) зажимов/биндеров для бумаги.

    Тот же стек, что и для пальцев, но YOLO-World настроена на классы
    ``CLAMP_CLASSES`` (биндер/зажим/скрепка). Зажим — мелкий объект, поэтому
    вызывать стоит на копии повышенного разрешения (``DEFAULT_CLAMP_WORK_SIDE``)
    и с низким ``conf`` (``DEFAULT_CLAMP_CONF``), иначе детектор его не видит.
    """
    from ocr_utils.finger_removal.masking import CLAMP_CLASSES

    spec = clamp_model
    if spec.lower().startswith("world:"):
        spec = spec[len("world:") :]
    yolo = load_yolo_world(spec, classes=CLAMP_CLASSES, cache_tag="clamp")
    return _yolo_sam_masks(bgr, yolo, sam_model, device, conf, MAX_CLAMP_BOX_FRAC, drop_negative_names=False)


def extend_clamp_handles(mask: np.ndarray, extend_frac: float) -> np.ndarray:
    """Достраивает маску зажима наружу, к краю кадра — захватывает усики.

    Зажим телом прижимает край страницы, а его металлические усики откинуты на
    обратную (внешнюю) сторону — торчат ОТ центра листа к краю кадра, поверх
    тёмного фона. SAM обводит только цветное тело; чтобы покрыть и усики, к
    каждой компоненте маски достраивается прямоугольник от её внешней грани к
    ближайшему краю кадра (за него и держится зажим). Длина прямоугольника —
    ``extend_frac`` × ширины/высоты тела по соответствующей оси. Заодно это
    безопасно: за краем страницы тёмный фон, лишний текст не затирается.
    """
    if extend_frac <= 0:
        return mask
    h, w = mask.shape[:2]
    out = mask.copy()
    num, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        # Ближайший край кадра держит зажим; усики смотрят НАРУЖУ — к этому краю
        dist = {"left": x, "right": w - (x + bw), "top": y, "bottom": h - (y + bh)}
        side = min(dist, key=dist.get)
        if side == "left":
            out[y : y + bh, max(0, x - int(round(extend_frac * bw))) : x] = 255
        elif side == "right":
            out[y : y + bh, x + bw : min(w, x + bw + int(round(extend_frac * bw)))] = 255
        elif side == "top":
            out[max(0, y - int(round(extend_frac * bh))) : y, x : x + bw] = 255
        else:  # bottom
            out[y + bh : min(h, y + bh + int(round(extend_frac * bh))), x : x + bw] = 255
    return out


# ============================================================
# Отрисовка
# ============================================================


def draw_contours(img: np.ndarray, contours: list[np.ndarray], color: tuple, thickness: int) -> None:
    """Рисует контуры заданным цветом (на месте)."""
    for c in contours:
        cv2.drawContours(img, [c], -1, color, thickness, lineType=cv2.LINE_AA)


def masks_to_contours(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Извлекает внешние контуры из списка бинарных масок."""
    out = []
    for m in masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.extend(cnts)
    return out


# ============================================================
# Обработка одного изображения
# ============================================================


def annotate_image(
    bgr: np.ndarray,
    finger_model: str,
    sam_model: Optional[str],
    device: str,
    conf: float,
    work_side: int,
    dilate_px: int = DEFAULT_DILATE_PX,
    finger_world: bool = False,
    detect_clamps: bool = True,
    clamp_model: str = DEFAULT_CLAMP_MODEL,
    clamp_conf: float = DEFAULT_CLAMP_CONF,
    clamp_work_side: int = DEFAULT_CLAMP_WORK_SIDE,
    clamp_dilate_px: int = DEFAULT_CLAMP_DILATE_PX,
    clamp_handle_extend: float = DEFAULT_CLAMP_HANDLE_EXTEND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Возвращает (BGR с обводкой, маска пальцев, маска зажимов, n_пальцев, n_зажимов).

    Детекция пальцев идёт на уменьшенной копии (сторона ``work_side``). Маски
    сливаются в одну, поднимаются в полное разрешение и расширяются дилатацией
    (``dilate_px``); их контуры рисуются КРАСНЫМ. Параллельно тем же стеком
    YOLO-World→SAM ищутся зажимы для бумаги (``detect_clamp_masks`` с классами
    ``CLAMP_CLASSES``) — на ОТДЕЛЬНОЙ копии повышенного разрешения
    (``clamp_work_side``), т.к. зажим мелкий. У них СВОЯ маска: силуэт тела
    достраивается наружу, к краю кадра (``clamp_handle_extend``, чтобы захватить
    откинутые усики), расширяется дилатацией (``clamp_dilate_px``) и обводится СИНИМ.
    Обе маски — uint8 0/255 в полном разрешении, пригодны для инпейнтинга.
    """
    h, w = bgr.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    if scale < 1.0:
        work = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        work = bgr

    out = bgr.copy()
    thickness = max(2, int(round(max(h, w) / 600)))

    # --- Пальцы/руки: YOLO(+SAM) → красный контур ---
    finger_mask = np.zeros((h, w), dtype=np.uint8)
    n_fingers = 0
    masks = detect_finger_masks(work, finger_model, sam_model, device, conf, world=finger_world)
    if masks:
        combined = np.zeros(work.shape[:2], dtype=np.uint8)
        for m in masks:
            combined = cv2.bitwise_or(combined, m)
        if scale != 1.0:
            combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
        # Дилатация склеивает близкие компоненты в одну связную область
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            combined = cv2.dilate(combined, kernel, iterations=1)
        finger_mask = combined
        cnts, _ = cv2.findContours(finger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        draw_contours(out, list(cnts), COLOR_FINGER, thickness)
        n_fingers = len(cnts)

    # --- Зажимы/биндеры: YOLO-World(CLAMP_CLASSES)→SAM → синий контур ---
    # Отдельная копия повышенного разрешения: зажим мелкий, на work_side пальцев
    # (обычно меньше) детектор его теряет.
    clamp_mask = np.zeros((h, w), dtype=np.uint8)
    n_clamps = 0
    if detect_clamps:
        cscale = clamp_work_side / max(h, w) if max(h, w) > clamp_work_side else 1.0
        cwork = (
            cv2.resize(bgr, (int(w * cscale), int(h * cscale)), interpolation=cv2.INTER_AREA) if cscale < 1.0 else bgr
        )
        cmasks = detect_clamp_masks(cwork, clamp_model, sam_model, device, clamp_conf)
        if cmasks:
            cm = np.zeros(cwork.shape[:2], dtype=np.uint8)
            for m in cmasks:
                cm = cv2.bitwise_or(cm, m)
            # Достраиваем маску к центру кадра — захватываем металлические усики
            cm = extend_clamp_handles(cm, clamp_handle_extend)
            if cscale != 1.0:
                cm = cv2.resize(cm, (w, h), interpolation=cv2.INTER_NEAREST)
            if clamp_dilate_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * clamp_dilate_px + 1, 2 * clamp_dilate_px + 1)
                )
                cm = cv2.dilate(cm, kernel, iterations=1)
            clamp_mask = cm
            c_cnts, _ = cv2.findContours(clamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            draw_contours(out, list(c_cnts), COLOR_CLAMP, thickness)
            n_clamps = len(c_cnts)

    return out, finger_mask, clamp_mask, n_fingers, n_clamps


# ============================================================
# CLI
# ============================================================


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Каталог с исходными сканами (JPG/PNG)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(__file__).resolve().parent / "detected_fingers",
    show_default=True,
    help="Куда сохранять итоговые обработанные изображения (с закрашенными пальцами, без подсветки границ)",
)
@click.option(
    "--debug-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Если задана — сюда сохраняются отладочные JPG с красным оверлеем маски пальцев",
)
@click.option(
    "--finger-model",
    default=DEFAULT_FINGER_MODEL,
    show_default=True,
    help="Модель пальцев/рук: 'repo_id:file' или путь. Префикс 'world:' включает "
    "open-vocabulary YOLO-World (классы из HAND_CLASSES), напр. world:yolov8x-worldv2.pt. "
    "Альт.: EtanHey/hand-detection-3class:model.pt",
)
@click.option(
    "--sam-model",
    default=DEFAULT_SAM_MODEL,
    show_default=True,
    help="SAM для криволинейного силуэта пальца ('none' — рисовать боксом)",
)
@click.option("--device", default=None, help="cuda / cpu (по умолчанию авто)")
@click.option("--conf", default=0.25, show_default=True, help="Порог уверенности YOLO")
@click.option("--work-side", default=1600, show_default=True, help="Сторона уменьшенной копии для детекции, пикс.")
@click.option(
    "--dilate",
    "dilate_px",
    default=DEFAULT_DILATE_PX,
    show_default=True,
    help="Дилатация объединённой маски, пикс. (склеивает близкие области; 0 — выкл.)",
)
@click.option(
    "--detect-clamps/--no-detect-clamps",
    default=True,
    show_default=True,
    help="Искать зажимы/биндеры для бумаги (YOLO-World с классами CLAMP_CLASSES → SAM)",
)
@click.option(
    "--clamp-model",
    default=DEFAULT_CLAMP_MODEL,
    show_default=True,
    help="YOLO-World для зажимов (классы CLAMP_CLASSES): 'world:file' или путь к .pt",
)
@click.option(
    "--clamp-conf",
    default=DEFAULT_CLAMP_CONF,
    show_default=True,
    help="Порог уверенности YOLO-World для зажима (мелкий объект — держите низким)",
)
@click.option(
    "--clamp-work-side",
    default=DEFAULT_CLAMP_WORK_SIDE,
    show_default=True,
    help="Сторона копии для детекции зажима, пикс. (выше, чем у пальцев — зажим мелкий)",
)
@click.option(
    "--clamp-dilate",
    "clamp_dilate_px",
    default=DEFAULT_CLAMP_DILATE_PX,
    show_default=True,
    help="Дилатация маски зажима, пикс. (0 — выкл.)",
)
@click.option(
    "--clamp-handle-extend",
    default=DEFAULT_CLAMP_HANDLE_EXTEND,
    show_default=True,
    help="Достроить маску зажима наружу, к краю кадра, на эту долю его ширины/высоты "
    "(захват откинутых металлических усиков; 0 — только тело)",
)
@click.option(
    "--inpaint",
    type=click.Choice(["none", "lama", "sd"]),
    default="lama",
    show_default=True,
    help="Убрать пальцы инпейнтингом: none — только обводка; lama / sd — модель инпейнтинга",
)
@click.option("--padding", default=64, show_default=True, help="Контекст вокруг маски для инпейнтинга, пикс.")
@click.option("--feather", default=9, show_default=True, help="Растушёвка краёв вклейки инпейнтинга, пикс.")
@click.option("--sd-model", default=SD_DEFAULT_MODEL, show_default=True, help="HF-id модели SD inpainting")
@click.option("--sd-steps", default=30, show_default=True, help="Число шагов SD")
@click.option("--sd-guidance", default=3.0, show_default=True, help="guidance_scale SD (ниже = меньше выдумок)")
@click.option("--sd-prompt", default=SD_DEFAULT_PROMPT, show_default=True, help="Промпт для SD-инпейнтинга")
@click.option("--sd-negative", default=SD_DEFAULT_NEGATIVE, show_default=True, help="Негативный промпт для SD")
@click.option("--limit", default=0, show_default=True, help="Обработать только первые N файлов (0 — все)")
def main(
    input_dir: Path,
    output_dir: Path,
    debug_dir: Optional[Path],
    finger_model: str,
    sam_model: str,
    device: Optional[str],
    conf: float,
    work_side: int,
    dilate_px: int,
    detect_clamps: bool,
    clamp_model: str,
    clamp_conf: float,
    clamp_work_side: int,
    clamp_dilate_px: int,
    clamp_handle_extend: float,
    inpaint: str,
    padding: int,
    feather: int,
    sd_model: str,
    sd_steps: int,
    sd_guidance: float,
    sd_prompt: str,
    sd_negative: str,
    limit: int,
) -> None:
    """Убирает пальцы и зажимы со сканов из INPUT_DIR (инпейнтинг) → OUTPUT_DIR.

    Пальцы/руки и зажимы для бумаги ищутся одним стеком YOLO-World→SAM (зажимы —
    отдельным набором классов и на повышенном разрешении). В ``--output-dir`` —
    финальные изображения без подсветки границ. Если задана ``--debug-dir`` —
    туда кладутся отладочные JPG: КРАСНАЯ граница пальцев, СИНЯЯ граница зажимов,
    жёлтый ROI инпейнтинга.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    finger_world = False
    finger = finger_model
    if finger.lower().startswith("world:"):
        finger_world = True
        finger = finger[len("world:") :]
    sam = None if sam_model.lower() == "none" else sam_model

    extensions = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    files: list[Path] = []
    for ext in extensions:
        files.extend(p for p in input_dir.glob(ext) if p.is_file())
    files = sorted(files)
    if limit > 0:
        files = files[:limit]
    if not files:
        logger.warning("Изображения не найдены в %s", input_dir)
        return

    logger.info(
        "Файлов: %d | устройство: %s | пальцы: %s%s | sam: %s | dilate: %d px | зажимы: %s | inpaint: %s",
        len(files),
        device,
        finger,
        " (YOLO-World)" if finger_world else "",
        sam,
        dilate_px,
        f"{clamp_model} @conf={clamp_conf}, ws={clamp_work_side}" if detect_clamps else "выкл",
        inpaint,
    )

    for path in tqdm(files, desc="Пальцы", unit="img"):
        try:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                tqdm.write(f"  Не удалось загрузить: {path.name}")
                continue
            out, finger_mask, clamp_mask, n_fingers, n_clamps = annotate_image(
                bgr,
                finger,
                sam,
                device,
                conf,
                work_side,
                dilate_px=dilate_px,
                finger_world=finger_world,
                detect_clamps=detect_clamps,
                clamp_model=clamp_model,
                clamp_conf=clamp_conf,
                clamp_work_side=clamp_work_side,
                clamp_dilate_px=clamp_dilate_px,
                clamp_handle_extend=clamp_handle_extend,
            )

            # Маска для инпейнтинга — объединение пальцев и зажимов (оба убираем)
            mask = cv2.bitwise_or(finger_mask, clamp_mask)
            n = n_fingers + n_clamps

            # Отладочный оверлей: красная граница пальцев + синяя граница зажимов
            # (нарисованы в annotate_image) + жёлтый ROI каждой компоненты
            if debug_dir is not None:
                if inpaint != "none" and n > 0:
                    thickness = max(2, int(round(max(bgr.shape[:2]) / 600)))
                    for x1, y1, x2, y2 in roi_bounds_list(mask, padding=padding, roi_scale=DEFAULT_ROI_SCALE):
                        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_ROI, thickness)
                dbg_file = debug_dir / f"{path.stem}_overlay.jpg"
                cv2.imwrite(str(dbg_file), out, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # Финальное изображение без подсветки границ: инпейнтинг пальцев либо оригинал
            if inpaint == "none":
                result_bgr = bgr
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result_rgb = inpaint_fingers(
                    rgb,
                    mask,
                    method=inpaint,
                    device=device,
                    padding=padding,
                    feather=feather,
                    sd_prompt=sd_prompt,
                    sd_negative=sd_negative,
                    sd_steps=sd_steps,
                    sd_guidance=sd_guidance,
                    sd_model=sd_model,
                )
                result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

            out_file = output_dir / f"{path.stem}.jpg"
            cv2.imwrite(str(out_file), result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            tqdm.write(
                f"  {path.name} → {out_file.name} | пальцев={n_fingers} | зажимов={n_clamps} | inpaint={inpaint}"
            )
        except Exception as e:
            tqdm.write(f"  Ошибка {path.name}: {e}")
            import traceback

            tqdm.write(traceback.format_exc())

    logger.info("Готово. Результаты в %s", output_dir)


if __name__ == "__main__":
    main()
