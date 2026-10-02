"""Воркер TextSnake (MMOCR 1.x, CTW1500): полигоны строк и осевые линии по дискам в JSON. Запускается ЧУЖИМ питоном.

Наш пакет не импортирует. Аргументы: ``<png> <out.json> [--long-side N] [--cpu] [--min-center 0.2]``.
Выход: ``{"lines": [{"baseline": [], "boundary": [[x, y], …], "centre": [[x, y], …], "height": h}], "regions": [],
"meta": {...}}`` в пикселях поданного изображения.

Сеть предсказывает пять карт: «текст», «осевая зона» (TCL), sin/cos направления и радиус диска. Разбор карт —
тот же, что у ``TextSnakePostprocessor`` MMOCR (скелет осевой зоны → центровка → слияние дисков → контур
объединения кругов), только дополнительно отдаёт центры дисков (``centre``) и их радиус (``height`` = 2 × медиана
радиуса). Центры дисков у MMOCR неупорядочены: для ломаной ``centre`` сортируем их по главной оси облака точек.
"""

import argparse
import json
import os
import sys
import time

# Все кэши (torch hub, openmmlab) — в каталоге моделей стенда, а не в ~/.cache.
MODELS_DIR = "/mnt/hotstore/scan_processing/mts_markup/line_axis_engines/models/textsnake"
os.environ.setdefault("TORCH_HOME", os.path.join(MODELS_DIR, "torch"))
os.environ.setdefault("XDG_CACHE_HOME", MODELS_DIR)
# Чекпойнт OpenMMLab хранит объекты mmengine (история лога): torch>=2.6 по умолчанию грузит только тензоры.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner import load_checkpoint  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

from mmocr.models.textdet.postprocessors.textsnake_postprocessor import TextSnakePostprocessor  # noqa: E402
from mmocr.registry import MODELS  # noqa: E402
from mmocr.utils import fill_hole, register_all_modules  # noqa: E402

CHECKPOINT = os.path.join(MODELS_DIR, "textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500_20221101_134814-a216e5b2.pth")
# Модель как в configs/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500.py (без init_cfg:
# все веса берутся из чекпойнта, предобученный бэкбон качать не нужно).
MODEL_CFG = dict(
    type="TextSnake",
    backbone=dict(type="CLIPResNet"),
    neck=dict(type="FPN_UNet", in_channels=[256, 512, 1024, 2048], out_channels=32),
    det_head=dict(
        type="TextSnakeHead",
        in_channels=32,
        module_loss=dict(type="TextSnakeModuleLoss"),
        postprocessor=dict(type="TextSnakePostprocessor", text_repr_type="poly"),
    ),
    data_preprocessor=dict(
        type="TextDetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
)
# Нормировка входа (RGB) из data_preprocessor конфига.
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
# Родной масштаб теста MMOCR: Resize(scale=(1333, 736), keep_ratio=True).
NATIVE_LONG, NATIVE_SHORT = 1333, 736


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    :param argv: аргументы без имени скрипта.
    :return: пространство имён: ``png`` — страница, ``out`` — куда писать JSON, ``long_side`` — большая сторона
        входа сети в пикселях (``0`` — родное правило теста MMOCR 1333×736), ``cpu`` — считать на процессоре,
        ``min_center`` — порог карты «осевая зона» × «текст» (как у постпроцессора MMOCR; порог карты «текст»
        MMOCR вычисляет, но не применяет, поэтому его здесь нет),
        ``min_center_area`` — минимальная площадь осевой зоны в пикселях входа сети.
    """
    parser = argparse.ArgumentParser(description="TextSnake: полигоны и осевые линии строк в JSON")
    parser.add_argument("png")
    parser.add_argument("out")
    parser.add_argument("--long-side", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--min-center", type=float, default=0.2)
    parser.add_argument("--min-center-area", type=int, default=30)
    return parser.parse_args(argv)


def build_model(device: str) -> torch.nn.Module:
    """Собрать TextSnake из реестра MMOCR и загрузить веса CTW1500.

    :param device: ``"cuda"`` или ``"cpu"``.
    :return: модель в режиме ``eval`` на устройстве ``device``.
    """
    register_all_modules()
    model = MODELS.build(Config(dict(model=MODEL_CFG)).model)
    load_checkpoint(model, CHECKPOINT, map_location="cpu", logger="silent")
    return model.to(device).eval()


def prepare_input(gray: np.ndarray, long_side: int) -> tuple[torch.Tensor, float]:
    """Отмасштабировать страницу, нормировать и дополнить до кратного 32.

    :param gray: страница, ``uint8``, ``H×W``.
    :param long_side: большая сторона входа сети; ``0`` — родное правило MMOCR (вписать в 1333×736).
    :return: тензор ``1×3×H'×W'`` и коэффициент масштаба «пиксели сети / пиксели страницы».
    """
    height, width = gray.shape
    if long_side > 0:
        scale = long_side / max(height, width)
    else:
        scale = min(NATIVE_LONG / max(height, width), NATIVE_SHORT / min(height, width))
    resized = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    # Серый канал размножаем в RGB и нормируем как в data_preprocessor.
    rgb = np.repeat(resized[:, :, None], 3, axis=2).astype(np.float32)
    rgb = (rgb - MEAN) / STD
    # Дополнение нулями (после нормировки) справа и снизу до кратного 32, как pad_size_divisor.
    pad_h = (-rgb.shape[0]) % 32
    pad_w = (-rgb.shape[1]) % 32
    rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)))
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)[None].copy())
    return tensor, scale


def predict_maps(model: torch.nn.Module, tensor: torch.Tensor, device: str) -> np.ndarray:
    """Прогнать сеть; при нехватке видеопамяти (карту делят с другими) подождать и повторить.

    :param model: модель TextSnake.
    :param tensor: вход ``1×3×H×W``.
    :param device: устройство модели.
    :return: карты ``5×H×W`` (``float32``): логиты «текст» и «осевая зона», sin, cos, радиус.
    """
    for attempt in range(6):
        try:
            with torch.no_grad():
                return model._forward(tensor.to(device))[0].float().cpu().numpy()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            time.sleep(10 * (attempt + 1))
    with torch.no_grad():
        return model._forward(tensor.to(device))[0].float().cpu().numpy()


def order_along_axis(points: np.ndarray) -> np.ndarray:
    """Упорядочить центры дисков вдоль главной оси облака (для ломаной осевой линии).

    :param points: точки ``N×2`` (x, y).
    :return: те же точки, отсортированные по проекции на первую главную компоненту, слева направо.
    """
    if len(points) < 2:
        return points
    centered = points - points.mean(axis=0)
    # Первая главная компонента — направление строки.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0] if vt[0][0] >= 0 else -vt[0]
    return points[np.argsort(centered @ axis)]


def decode(maps: np.ndarray, args: argparse.Namespace) -> list[dict]:
    """Разобрать карты сети в экземпляры строк — алгоритм ``TextSnakePostprocessor.get_text_instances`` MMOCR.

    :param maps: карты ``5×H×W`` из :func:`predict_maps`.
    :param args: пороги (``min_center``, ``min_center_area``).
    :return: список ``{"boundary": N×2, "disks": M×4 (x, y, r, score), "score": float}`` в пикселях входа сети.
    """
    helper = TextSnakePostprocessor(text_repr_type="poly")
    text_score = 1.0 / (1.0 + np.exp(-maps[0]))
    center_score = 1.0 / (1.0 + np.exp(-maps[1])) * text_score
    center_mask = center_score > args.min_center
    sin, cos, radius_map = maps[2], maps[3], maps[4]
    # Нормируем (sin, cos) на единичную длину.
    norm = np.sqrt(1.0 / (sin**2 + cos**2 + 1e-8))
    sin, cos = sin * norm, cos * norm
    shape = text_score.shape

    center_mask = fill_hole(center_mask).astype(np.uint8)
    contours, _ = cv2.findContours(center_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    instances = []
    for contour in contours:
        if cv2.contourArea(contour) < args.min_center_area:
            continue
        # Скелет осевой зоны экземпляра.
        instance_center = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(instance_center, [contour], -1, 1, -1)
        skeleton_yx = np.argwhere(skeletonize(instance_center) > 0)
        y, x = skeleton_yx[:, 0], skeleton_yx[:, 1]
        # Центровка точек скелета поперёк строки (по нормали) внутри осевой зоны.
        centre_yx = helper._centralize(
            skeleton_yx,
            cos[y, x].reshape((-1, 1)),
            -sin[y, x].reshape((-1, 1)),
            radius_map[y, x].reshape((-1, 1)),
            instance_center,
        )
        y, x = centre_yx[:, 0], centre_yx[:, 1]
        radius = (radius_map[y, x] * helper.radius_shrink_ratio).reshape((-1, 1))
        score = center_score[y, x].reshape((-1, 1))
        disks = helper._merge_disks(np.hstack([np.fliplr(centre_yx), radius, score]), helper.disk_overlap_thr)
        # Полигон строки — внешний контур объединения дисков.
        instance_mask = np.zeros(shape, dtype=np.uint8)
        for disk_x, disk_y, disk_r, _ in disks:
            if disk_r > 1:
                cv2.circle(instance_mask, (int(disk_x), int(disk_y)), int(disk_r), 1, -1)
        outlines, _ = cv2.findContours(instance_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        line_score = float(np.sum(instance_mask * text_score) / (np.sum(instance_mask) + 1e-8))
        if outlines and cv2.contourArea(outlines[0]) > 0 and outlines[0].size > 8:
            instances.append({"boundary": outlines[0].reshape(-1, 2), "disks": disks, "score": line_score})
    return instances


def main() -> None:
    """Точка входа: модель → карты → экземпляры → JSON в пикселях страницы."""
    args = parse_args(sys.argv[1:])
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model = build_model(device)

    gray = cv2.imread(args.png, cv2.IMREAD_GRAYSCALE)
    tensor, scale = prepare_input(gray, args.long_side)

    started = time.time()
    maps = predict_maps(model, tensor, device)
    instances = decode(maps, args)
    seconds = time.time() - started

    # Пересчёт из пикселей входа сети в пиксели страницы.
    lines = []
    for instance in instances:
        disks = instance["disks"]
        centre = order_along_axis(disks[:, :2]) / scale
        lines.append(
            {
                "baseline": [],
                "boundary": (instance["boundary"] / scale).round(1).tolist(),
                "centre": centre.round(1).tolist(),
                "height": float(2.0 * np.median(disks[:, 2]) / scale),
                "confidence": instance["score"],
            }
        )
    meta = {
        "model": "textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500",
        "device": device,
        "params": {
            "scale": round(scale, 4),
            "net_input_hw": list(tensor.shape[2:]),
            "min_center": args.min_center,
            "min_center_area": args.min_center_area,
        },
        "seconds": round(seconds, 2),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": [], "meta": meta}, handle)


if __name__ == "__main__":
    main()
