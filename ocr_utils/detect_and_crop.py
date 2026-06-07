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

Если задана ``--debug-dir`` — туда пишется кадр с оверлеями: зелёная граница
разворота, синий min-area bbox, фиолетовая crop-зона.

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
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Папка для весов нейромоделей (корень проекта, рядом с finger_models)
MODELS_DIR = Path(__file__).resolve().parents[1] / "finger_models"

# Цвета оверлеев в BGR (OpenCV)
COLOR_PAGE = (0, 255, 0)  # ярко-зелёный — криволинейная граница разворота
COLOR_ROT_BBOX = (255, 0, 0)  # ярко-синий — min-area повёрнутый bounding box
COLOR_CROP = (211, 0, 148)  # фиолетовый — финальная crop-зона с припусками

# Поиск правильного поворота разворота: перебор углов ± предела с шагом (градусы)
ROT_RANGE_DEG = 35
ROT_STEP_DEG = 1

# Веса по умолчанию (лежат/качаются в finger_models/)
DEFAULT_YOLO_WORLD = "yolov8x-worldv2.pt"
DEFAULT_SAM = "sam_b.pt"

# Классы open-vocabulary детектора, описывающие страницу/разворот книги.
PAGE_CLASSES = ["page", "book page", "open book", "sheet of paper", "paper", "document"]

# Поддерживаемые форматы входных изображений (без учёта регистра расширения)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Параметры детекции
CONF = 0.10  # порог уверенности YOLO-World (выше — не хватает всю картинку как «страницу»)
WORK_SIDE = 2048  # сторона уменьшенной копии для детекции (выше = точнее контур SAM)
MIN_PAGE_FRAC = 0.05  # бокс/маска меньше этой доли кадра — это не страница
MAX_PAGE_FRAC = 1.0  # верхний предел не ставим: страница может занимать весь кадр

_MODEL_CACHE: dict = {}


# ============================================================
# Модели и маска разворота
# ============================================================


def resolve_model_path(name: str) -> str:
    """Путь к весам в finger_models/ (качает ассет ultralytics по имени, если нужно)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODELS_DIR / name)


def load_yolo_world(name: str):
    """Ленивая загрузка YOLO-World с классами страницы."""
    key = f"world:{name}"
    if key not in _MODEL_CACHE:
        from ultralytics import YOLOWorld

        model = YOLOWorld(resolve_model_path(name))
        model.set_classes(PAGE_CLASSES)
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
    """Бинарная маска (uint8 0/255) области страниц: YOLO-World боксы → SAM силуэт."""
    h, w = bgr.shape[:2]
    img_area = h * w

    yolo = load_yolo_world(DEFAULT_YOLO_WORLD)
    det = yolo.predict(bgr, conf=CONF, device=device, verbose=False)
    if not det or det[0].boxes is None or len(det[0].boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    boxes = det[0].boxes.xyxy.cpu().numpy()
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    area = bw * bh
    boxes = boxes[(area >= MIN_PAGE_FRAC * img_area) & (area <= MAX_PAGE_FRAC * img_area)]
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


def crop_rotated(bgr: np.ndarray, cx: float, cy: float, angle: float, ext: tuple, mx: int, my: int) -> np.ndarray:
    """Поворот вокруг центра тяжести + вырез crop-зоны → выпрямленный прямоугольник.

    Берём 4 угла crop-зоны в исходном кадре и перспективным преобразованием
    отображаем их в осевой прямоугольник нужного размера (это и есть поворот кадра
    на найденный угол с одновременным вырезом области).
    """
    minx, miny, maxx, maxy = _ext_with_margins(ext, mx, my)
    out_w = max(1, int(round(maxx - minx)))
    out_h = max(1, int(round(maxy - miny)))
    src = _bbox_corners(cx, cy, angle, (minx, miny, maxx, maxy))
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, m, (out_w, out_h), flags=cv2.INTER_LINEAR)


def draw_overlay(bgr: np.ndarray, mask: np.ndarray, geom: Optional[tuple], mx: int, my: int) -> np.ndarray:
    """Кадр с оверлеями: зелёная граница разворота, синий min-bbox, фиолетовая crop-зона."""
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
def main(
    input_dir: Path,
    output_dir: Path,
    debug_dir: Optional[Path],
    x_margins: int,
    y_margins: int,
    recursive: bool,
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
        "Файлов: %d | устройство: %s | margins: x=%d y=%d | recursive: %s",
        len(files),
        device,
        x_margins,
        y_margins,
        recursive,
    )

    for path in tqdm(files, desc="Crop", unit="img"):
        try:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                tqdm.write(f"  Не удалось загрузить: {path.name}")
                continue

            mask = page_mask(bgr, device)
            geom = min_area_rotated_bbox(mask)

            # Формат и имя сохраняем как у входа; при recursive — зеркалим подкаталоги
            rel = path.relative_to(input_dir)
            params = _imwrite_params(path.suffix)
            out_path = output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if geom is None:
                # Разворот не найден — кладём оригинал, чтобы не терять файл в пайплайне
                tqdm.write(f"  Разворот не найден, сохраняю оригинал: {rel}")
                cv2.imwrite(str(out_path), bgr, params)
            else:
                cx, cy, angle, ext = geom
                crop = crop_rotated(bgr, cx, cy, angle, ext, x_margins, y_margins)
                cv2.imwrite(str(out_path), crop, params)

            if debug_dir is not None:
                dbg_path = debug_dir / rel
                dbg_path.parent.mkdir(parents=True, exist_ok=True)
                overlay = draw_overlay(bgr, mask, geom, x_margins, y_margins)
                cv2.imwrite(str(dbg_path), overlay, params)

        except Exception as e:
            tqdm.write(f"  Ошибка {path.name}: {e}")
            import traceback

            tqdm.write(traceback.format_exc())

    logger.info("Готово. Crop → %s%s", output_dir, f" | debug → {debug_dir}" if debug_dir else "")


if __name__ == "__main__":
    main()
