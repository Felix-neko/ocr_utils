"""Воркер CRAFT: карты символов и связей → осевые линии слов/строк в JSON. Запускается ЧУЖИМ питоном, наш пакет не импортирует.

Аргументы: ``<png> <out.json> [--weights PTH] [--src DIR] [--canvas-size N] [--mag-ratio R] [--fp16] [--overlay PNG]``.

Модель — исходный код clovaai/CRAFT-pytorch (каталог ``--src``) и веса ``craft_mlt_25k.pth``. Сеть отдаёт
две карты в половинном разрешении входа: «регион» (центр символа) и «связь» (промежуток между соседними
символами одного слова). Из них без подстройки под наши полосы, с порогами CRAFT по умолчанию
(text_threshold 0.7, link_threshold 0.4, low_text 0.4), строится:

* компонента = связная область ``(регион ≥ low_text) ∪ (связь ≥ link_threshold)`` (4-связность), в которой
  максимум региона ≥ text_threshold — ровно группировка слов самого CRAFT. Если карта связей соединяет
  слова между собой, компонента — целая строка; если нет — слово выдаётся отдельной «строкой»;
* символы = связные пятна ``регион ≥ text_threshold``; центр пятна — центр масс, взвешенный значением
  карты региона; пятно относится к компоненте, в которую попадает его пиковый пиксель;
* ``centre`` — ломаная центров символов компоненты по возрастанию x; ``boundary`` — внешний контур
  компоненты; ``height`` — медиана по столбцам вертикальной толщины маски компоненты (устойчива к изгибу
  строки, в отличие от высоты рамки).

Выход: ``{"lines": [{"baseline": [], "boundary": [[x, y], …], "centre": [[x, y], …], "height": h}],
"regions": [], "meta": {…}}`` в пикселях поданного изображения.
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch

# Порог CRAFT по умолчанию (test.py исходного репозитория) — не подстраиваем.
TEXT_THRESHOLD = 0.7
LINK_THRESHOLD = 0.4
LOW_TEXT = 0.4
# Минимальная площадь компоненты в пикселях карты — как в craft_utils.getDetBoxes_core.
MIN_COMPONENT_AREA = 10
# Нормировка ImageNet, как в imgproc.normalizeMeanVariance.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

DEFAULT_ROOT = "/mnt/hotstore/scan_processing/mts_markup/line_axis_engines"


def parse_args() -> argparse.Namespace:
    """Разобрать командную строку.

    Возвращает пространство имён: ``png`` — входная полоса, ``out`` — путь JSON, ``weights`` — файл весов,
    ``src`` — каталог клона CRAFT-pytorch, ``canvas_size`` — предел длинной стороны (CRAFT ужимает вход,
    если он больше), ``mag_ratio`` — увеличение входа, ``fp16`` — вывод в половинной точности (меньше
    видеопамяти), ``overlay`` — путь отладочной картинки (пусто — не рисовать).
    """
    parser = argparse.ArgumentParser(description="CRAFT → осевые линии в JSON")
    parser.add_argument("png")
    parser.add_argument("out")
    parser.add_argument("--weights", default=os.path.join(DEFAULT_ROOT, "models/craft/craft_mlt_25k.pth"))
    parser.add_argument("--src", default=os.path.join(DEFAULT_ROOT, "src/craft"))
    parser.add_argument("--canvas-size", type=int, default=3840)
    parser.add_argument("--mag-ratio", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--overlay", default="")
    return parser.parse_args()


def load_model(src: str, weights: str, device: torch.device) -> torch.nn.Module:
    """Собрать сеть CRAFT из исходников клона и загрузить веса.

    Аргументы: ``src`` — каталог CRAFT-pytorch (там ``craft.py`` и ``basenet``), ``weights`` — путь к
    ``craft_mlt_25k.pth``, ``device`` — куда положить модель. Возвращает модель в режиме eval.
    """
    # basenet/vgg16_bn.py импортирует torchvision.models.vgg.model_urls, которого в новом torchvision нет;
    # подкладываем словарь-заглушку, чтобы не править клон. Предобученный VGG не качаем (pretrained=False),
    # все веса берутся из чекпойнта CRAFT.
    import torchvision.models.vgg as tv_vgg

    if not hasattr(tv_vgg, "model_urls"):
        tv_vgg.model_urls = {"vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"}
    sys.path.insert(0, src)
    from craft import CRAFT  # noqa: E402  (модуль из клона)

    net = CRAFT(pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    # Чекпойнт сохранён из DataParallel: ключи начинаются с "module." — срезаем.
    clean = OrderedDict((key[7:] if key.startswith("module.") else key, value) for key, value in state.items())
    net.load_state_dict(clean)
    return net.to(device).eval()


def prepare_input(gray: np.ndarray, canvas_size: int, mag_ratio: float) -> tuple[np.ndarray, float]:
    """Привести полосу к входу сети так же, как imgproc.resize_aspect_ratio CRAFT.

    Аргументы: ``gray`` — полоса в оттенках серого (H×W uint8), ``canvas_size`` — предел длинной стороны,
    ``mag_ratio`` — увеличение. Возвращает (``tensor_hwc`` — нормированный float32 H'×W'×3 с дополнением до
    кратного 32, ``ratio`` — во сколько раз вход сети больше исходника). Отличие от оригинала: поле
    дополнения заливается белым (фон бумаги), а не чёрным, чтобы край не давал ложного «текста».
    """
    height, width = gray.shape
    # Длинная сторона после увеличения, но не больше canvas_size — логика CRAFT.
    target = min(mag_ratio * max(height, width), canvas_size)
    ratio = target / max(height, width)
    new_h, new_w = int(height * ratio), int(width * ratio)
    resized = gray if (new_h, new_w) == (height, width) else cv2.resize(gray, (new_w, new_h), cv2.INTER_LINEAR)
    # Дополнение до кратного 32 (пять понижений разрешения в VGG).
    pad_h, pad_w = (32 - new_h % 32) % 32, (32 - new_w % 32) % 32
    canvas = np.full((new_h + pad_h, new_w + pad_w), 255, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    # Серое → три одинаковых канала RGB и нормировка ImageNet.
    rgb = np.repeat(canvas[:, :, None], 3, axis=2).astype(np.float32)
    return (rgb - MEAN) / STD, ratio


def run_network(net: torch.nn.Module, image: np.ndarray, device: torch.device, fp16: bool) -> np.ndarray:
    """Прогнать сеть по всей полосе целиком, с повтором при нехватке видеопамяти.

    Аргументы: ``net`` — модель, ``image`` — нормированный вход H×W×3, ``device`` — устройство, ``fp16`` —
    считать в автокасте половинной точности. Возвращает массив (H/2)×(W/2)×2: канал 0 — регион, 1 — связь.
    """
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    for attempt in range(6):
        try:
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                output, _ = net(tensor.to(device))
            return output[0].float().cpu().numpy()
        except torch.OutOfMemoryError:
            # Видеокарту делят с другими процессами: освобождаем кэш и ждём, потом пробуем снова.
            torch.cuda.empty_cache()
            print(f"нехватка видеопамяти, повтор {attempt + 1}", file=sys.stderr)
            time.sleep(20)
    raise RuntimeError("не хватило видеопамяти после шести попыток")


def weighted_centroids(region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Найти пятна символов (регион ≥ text_threshold) и их взвешенные центры.

    Аргумент ``region`` — карта региона (h×w float). Возвращает (``centres`` — N×2 (x, y) в пикселях карты,
    центр масс с весом значения карты; ``peaks`` — N×2 (row, col) пикового пикселя пятна; ``heights`` — N
    высот пятен в пикселях карты).
    """
    blob_mask = (region >= TEXT_THRESHOLD).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blob_mask, connectivity=8)
    if count <= 1:
        return np.zeros((0, 2)), np.zeros((0, 2), dtype=int), np.zeros(0)
    rows, cols = np.nonzero(labels)
    ids = labels[rows, cols]
    weights = region[rows, cols].astype(np.float64)
    # Взвешенные суммы координат по пятнам одним проходом bincount.
    total = np.bincount(ids, weights, minlength=count)
    cx = np.bincount(ids, weights * cols, minlength=count) / np.maximum(total, 1e-9)
    cy = np.bincount(ids, weights * rows, minlength=count) / np.maximum(total, 1e-9)
    # Пиковый пиксель каждого пятна: сортируем по (id, значение) и берём последний в группе.
    order = np.lexsort((weights, ids))
    last = np.r_[np.nonzero(np.diff(ids[order]))[0], len(order) - 1]
    peaks = np.stack([rows[order][last], cols[order][last]], axis=1)
    centres = np.stack([cx[1:], cy[1:]], axis=1)
    return centres, peaks, stats[1:, cv2.CC_STAT_HEIGHT].astype(float)


def to_input(points: np.ndarray, scale: float) -> list[list[float]]:
    """Перевести точки (x, y) из пикселей карты в пиксели исходника.

    Аргументы: ``points`` — N×2 координаты в пикселях карты, ``scale`` — сколько пикселей исходника в одном
    пикселе карты (2 / ratio). Центр пикселя карты i лежит в (i + 0.5)·scale − 0.5 пикселя исходника.
    Возвращает список [x, y].
    """
    return [[float((x + 0.5) * scale - 0.5), float((y + 0.5) * scale - 0.5)] for x, y in points]


def column_thickness(mask: np.ndarray) -> float:
    """Медиана по столбцам числа пикселей маски — толщина полосы компоненты поперёк строки.

    Аргумент ``mask`` — булева маска компоненты (обрезанная по рамке). Возвращает медианную толщину в
    пикселях маски (0, если маска пуста).
    """
    per_column = mask.sum(axis=0)
    per_column = per_column[per_column > 0]
    return float(np.median(per_column)) if per_column.size else 0.0


def build_lines(score: np.ndarray, ratio: float) -> tuple[list[dict], dict]:
    """Собрать компоненты CRAFT и перевести их в строки выходного JSON.

    Аргументы: ``score`` — выход сети (h×w×2), ``ratio`` — масштаб входа сети относительно исходника.
    Возвращает (список строк ``{"baseline", "boundary", "centre", "height"}`` в пикселях исходника,
    словарь счётчиков для meta).
    """
    region, link = score[:, :, 0], score[:, :, 1]
    text_mask = region >= LOW_TEXT
    link_mask = link >= LINK_THRESHOLD
    # Группировка слов CRAFT: объединение масок региона и связи, 4-связность.
    combined = (text_mask | link_mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=4)
    centres, peaks, _ = weighted_centroids(region)
    # Каждое пятно символа — в компоненту, где лежит его пик.
    blob_component = labels[peaks[:, 0], peaks[:, 1]] if len(peaks) else np.zeros(0, dtype=int)
    # Пиксель карты (i) покрывает пиксели входа сети 2i..2i+1: центр — 2i + 0.5; затем делим на ratio.
    scale = 2.0 / ratio

    lines = []
    skipped_small = skipped_weak = 0
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < MIN_COMPONENT_AREA:
            skipped_small += 1
            continue
        crop = labels[y : y + h, x : x + w] == label
        # Порог CRAFT: в компоненте должен быть хоть один уверенный пиксель региона.
        if region[y : y + h, x : x + w][crop].max() < TEXT_THRESHOLD:
            skipped_weak += 1
            continue
        # Внешний контур компоненты (самый длинный, если их несколько из-за касаний по диагонали).
        contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=len).reshape(-1, 2) + np.array([x, y])
        # Центры символов этой компоненты по возрастанию x.
        own = centres[blob_component == label]
        own = own[np.argsort(own[:, 0])]
        lines.append(
            {
                "baseline": [],
                "boundary": to_input(contour, scale),
                "centre": to_input(own, scale),
                "height": column_thickness(crop) * scale,
            }
        )
    counters = {"components": count - 1, "skipped_small": skipped_small, "skipped_weak": skipped_weak}
    counters["char_blobs"] = int(len(centres))
    return lines, counters


def draw_overlay(gray: np.ndarray, lines: list[dict], path: str) -> None:
    """Нарисовать отладочную картинку: контуры компонент синим, осевые ломаные красным, тонко.

    Аргументы: ``gray`` — исходная полоса, ``lines`` — строки выходного JSON, ``path`` — куда сохранить PNG.
    """
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for line in lines:
        boundary = np.round(np.array(line["boundary"])).astype(np.int32)
        cv2.polylines(canvas, [boundary], True, (255, 0, 0), 1, cv2.LINE_AA)
        centre = np.round(np.array(line["centre"])).astype(np.int32)
        if len(centre) >= 2:
            cv2.polylines(canvas, [centre], False, (0, 0, 255), 2, cv2.LINE_AA)
        elif len(centre) == 1:
            cv2.circle(canvas, tuple(int(v) for v in centre[0]), 2, (0, 0, 255), -1)
    # Легенда в левом верхнем углу.
    cv2.rectangle(canvas, (8, 8), (420, 70), (255, 255, 255), -1)
    cv2.putText(canvas, "CRAFT: centre (red)", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(canvas, "component boundary (blue)", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imwrite(path, canvas)


def main() -> None:
    """Точка входа: прочитать полосу, прогнать CRAFT, собрать строки, записать JSON (и оверлей)."""
    args = parse_args()
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gray = cv2.imread(args.png, cv2.IMREAD_GRAYSCALE)
    net = load_model(args.src, args.weights, device)
    image, ratio = prepare_input(gray, args.canvas_size, args.mag_ratio)
    score = run_network(net, image, device, args.fp16)
    lines, counters = build_lines(score, ratio)
    meta = {
        "model": "CRAFT craft_mlt_25k (clovaai/CRAFT-pytorch)",
        "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "params": {
            "text_threshold": TEXT_THRESHOLD,
            "link_threshold": LINK_THRESHOLD,
            "low_text": LOW_TEXT,
            "canvas_size": args.canvas_size,
            "mag_ratio": args.mag_ratio,
            "input_ratio": ratio,
            "network_input_hw": list(image.shape[:2]),
            "heatmap_hw": list(score.shape[:2]),
            "fp16": args.fp16,
            "height": "медиана по столбцам толщины маски компоненты (регион∪связь), пиксели исходника",
        },
        "counters": counters,
        "seconds": round(time.time() - started, 2),
    }
    if device.type == "cuda":
        meta["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": [], "meta": meta}, handle, ensure_ascii=False)
    if args.overlay:
        draw_overlay(gray, lines, args.overlay)


if __name__ == "__main__":
    main()
