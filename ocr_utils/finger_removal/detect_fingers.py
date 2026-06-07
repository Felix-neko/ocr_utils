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

# Дефолтные спецификации моделей. По умолчанию — open-vocabulary YOLO-World
# (классы из HAND_CLASSES, включая «ноготь»).
DEFAULT_FINGER_MODEL = "world:yolov8x-worldv2.pt"
DEFAULT_SAM_MODEL = "sam_b.pt"  # локальный вес в finger_models/

# Дилатация объединённой маски пальцев по умолчанию, пикс. (полное разрешение)
DEFAULT_DILATE_PX = 24

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


def load_yolo_world(spec: str):
    """Ленивая загрузка open-vocabulary YOLO-World с кэшем.

    Классы берутся из ``HAND_CLASSES`` (masking.py) — рука/палец/ноготь и т.п.,
    поэтому модель ищет именно пальцы, а не произвольные объекты.
    """
    key = f"world:{spec}"
    if key not in _MODEL_CACHE:
        from ultralytics import YOLOWorld

        from ocr_utils.finger_removal.masking import HAND_CLASSES

        model = YOLOWorld(resolve_model_path(spec))
        model.set_classes(HAND_CLASSES)
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


def detect_finger_masks(
    bgr: np.ndarray,
    finger_model: str,
    sam_model: Optional[str],
    device: str,
    conf: float,
    world: bool = False,
) -> list[np.ndarray]:
    """Возвращает список бинарных масок (uint8 0/255) пальцев/рук.

    Сначала YOLO-детектор находит боксы руки, затем (если задан) SAM строит по ним
    криволинейный силуэт. Без SAM маской служит прямоугольник бокса. При ``world``
    используется open-vocabulary YOLO-World с классами ``HAND_CLASSES``.
    """
    h, w = bgr.shape[:2]
    img_area = h * w

    yolo = load_yolo_world(finger_model) if world else load_yolo(finger_model)
    det = yolo.predict(bgr, conf=conf, device=device, verbose=False)
    if not det or det[0].boxes is None or len(det[0].boxes) == 0:
        return []

    boxes = det[0].boxes.xyxy.cpu().numpy()
    cls = det[0].boxes.cls.cpu().numpy().astype(int)
    names = det[0].names

    # Оставляем только «руку/палец», выбрасываем классы вроде not_hand/background
    keep = []
    for i in range(len(boxes)):
        name = str(names.get(cls[i], "")).lower()
        if "not" in name or "background" in name:
            continue
        bw = boxes[i, 2] - boxes[i, 0]
        bh = boxes[i, 3] - boxes[i, 1]
        if bw * bh > MAX_FINGER_BOX_FRAC * img_area:
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
) -> tuple[np.ndarray, np.ndarray, int]:
    """Возвращает (BGR с красной обводкой, объединённая маска uint8 0/255, число областей).

    Детекция идёт на уменьшенной копии (сторона ``work_side``). Все найденные
    маски сливаются в одну, поднимаются в полное разрешение и расширяются
    дилатацией на ``dilate_px`` пикселей — близкие/касающиеся области сливаются в
    одну связную (объединение связных областей). Контуры берутся уже с полной
    маски, поэтому рисуются в исходном разрешении; маска пригодна для инпейнтинга.
    """
    h, w = bgr.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    if scale < 1.0:
        work = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        work = bgr

    out = bgr.copy()
    thickness = max(2, int(round(max(h, w) / 600)))

    masks = detect_finger_masks(work, finger_model, sam_model, device, conf, world=finger_world)
    if not masks:
        return out, np.zeros((h, w), dtype=np.uint8), 0

    # Объединяем все маски в одну (логическое ИЛИ), поднимаем в полное разрешение
    combined = np.zeros(work.shape[:2], dtype=np.uint8)
    for m in masks:
        combined = cv2.bitwise_or(combined, m)
    if scale != 1.0:
        combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)

    # Дилатация склеивает близкие компоненты в одну связную область
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        combined = cv2.dilate(combined, kernel, iterations=1)

    # Контуры объединённой маски — каждый соответствует одной связной области
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw_contours(out, list(cnts), COLOR_FINGER, thickness)

    return out, combined, len(cnts)


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
    """Убирает пальцы со сканов из INPUT_DIR (инпейнтинг) и пишет результат в OUTPUT_DIR.

    В ``--output-dir`` — финальные изображения без подсветки границ. Если задана
    ``--debug-dir`` — туда кладутся отладочные JPG с красным оверлеем маски пальцев.
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
        "Файлов: %d | устройство: %s | пальцы: %s%s | sam: %s | dilate: %d px | inpaint: %s",
        len(files),
        device,
        finger,
        " (YOLO-World)" if finger_world else "",
        sam,
        dilate_px,
        inpaint,
    )

    for path in tqdm(files, desc="Пальцы", unit="img"):
        try:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                tqdm.write(f"  Не удалось загрузить: {path.name}")
                continue
            out, mask, n = annotate_image(
                bgr, finger, sam, device, conf, work_side, dilate_px=dilate_px, finger_world=finger_world
            )

            # Отладочный оверлей: красная граница маски + жёлтый ROI каждой компоненты
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
            tqdm.write(f"  {path.name} → {out_file.name} | пальцев={n} | inpaint={inpaint}")
        except Exception as e:
            tqdm.write(f"  Ошибка {path.name}: {e}")
            import traceback

            tqdm.write(traceback.format_exc())

    logger.info("Готово. Результаты в %s", output_dir)


if __name__ == "__main__":
    main()
