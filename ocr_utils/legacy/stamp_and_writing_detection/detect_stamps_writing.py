#!/usr/bin/env python3
"""Детектирование библиотечных штампов и рукописных надписей на сканах.

Скрипт принимает входную директорию с изображениями, прогоняет выбранный
метод детектирования и сохраняет копии изображений с наложенными оверлеями:
криволинейная граница вокруг найденных объектов (красная — штампы,
зелёная — рукописные надписи) и подпись с классом и уверенностью.

Методы (--method):
  * classic              — классический CV: сегментация чернил по цвету + морфология.
  * yolo-stamp           — YOLOv8 детектор штампов (PiDinoSauR/Stamp_Detection_10_12_2024).
  * yolo-stamp-finetuned — TorchScript-детектор штампов (stamps-labs/yolov8-finetuned).
  * yolo-signature       — YOLOv8 детектор рукописных подписей
                           (tech4humans/yolov8s-signature-detector).
  * all                  — прогнать все методы по очереди.

Выходная директория по умолчанию берётся рядом со входной, внутри неё
создаётся поддиректория по имени метода/модели.

Пример:
    python detect_stamps_writing.py --input-dir test_pics --method yolo-stamp
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Поддерживаемые расширения изображений.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Цвета оверлеев в BGR.
COLOR_STAMP = (0, 0, 255)  # красный — штампы
COLOR_WRITING = (0, 200, 0)  # зелёный — рукописные надписи
COLOR_BY_LABEL = {"stamp": COLOR_STAMP, "handwriting": COLOR_WRITING}

# Максимальная сторона для внутренней обработки classic-метода (для скорости);
# найденные контуры масштабируются обратно к исходному разрешению.
CLASSIC_MAX_SIDE = 2000


@dataclass
class Detection:
    """Одна найденная область.

    label   — "stamp" или "handwriting".
    polygon — контур области в координатах исходного изображения, shape (N, 2).
    score   — уверенность [0, 1] (для classic — эвристическая).
    """

    label: str
    polygon: np.ndarray
    score: float = 1.0


# --------------------------------------------------------------------------- #
#  Метод 1. Классический CV: цвет + морфология                                #
# --------------------------------------------------------------------------- #
def detect_classic(image_bgr: np.ndarray) -> list[Detection]:
    """Находит цветные чернильные области и делит их на штампы и рукопись.

    Идея: бумага и чёрный печатный текст слабонасыщенные, а чернила штампов
    и ручки/маркера — насыщенные (фиолетовый, синий, красный). Берём маску
    насыщенных пикселей, чистим морфологией, ищем связные компоненты и
    классифицируем их по форме: компактные плотные блоки — штампы, тонкие
    вытянутые штрихи — рукопись.

    Замечание: метод по своей природе путает крупные *печатные* цветные
    заголовки со штампами — это baseline, см. MODELS_REPORT.md.
    """
    h0, w0 = image_bgr.shape[:2]
    # Масштабируем для скорости, запоминаем коэффициент для обратного пересчёта.
    scale = min(1.0, CLASSIC_MAX_SIDE / max(h0, w0))
    if scale < 1.0:
        work = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = image_bgr
    h, w = work.shape[:2]

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Маска насыщенных, не слишком тёмных чернил.
    ink = ((s > 60) & (v > 40) & (v < 250)).astype(np.uint8) * 255

    # Морфология: убираем шум, затем слегка соединяем близкие штрихи.
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    ink_closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    contours, _ = cv2.findContours(ink_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(h * w)
    detections: list[Detection] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.0004:  # слишком мелкие — шум
            continue
        if area > img_area * 0.25:  # слишком крупные — вероятно фон/разворот
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        rect_area = float(bw * bh)
        solidity = area / rect_area if rect_area > 0 else 0.0
        aspect = bw / bh if bh > 0 else 0.0

        # Плотность заполнения цветом внутри бокса (доля чернил).
        roi = ink[y : y + bh, x : x + bw]
        fill = float(np.count_nonzero(roi)) / rect_area if rect_area > 0 else 0.0

        # Классификация по форме:
        #   штамп   — крупный, плотно заполненный, не сильно вытянутый;
        #   рукопись — тонкие/вытянутые штрихи с низкой плотностью заливки.
        is_stamp = solidity > 0.35 and fill > 0.18 and 0.25 < aspect < 4.0 and area > img_area * 0.0015
        label = "stamp" if is_stamp else "handwriting"
        score = float(np.clip(fill * solidity * 2.5, 0.2, 0.95))

        poly = cnt.reshape(-1, 2).astype(np.float32)
        if scale < 1.0:
            poly = poly / scale
        detections.append(Detection(label=label, polygon=poly.astype(np.int32), score=score))

    return detections


# --------------------------------------------------------------------------- #
#  Методы 2-3. YOLOv8 детекторы (ultralytics + веса с HuggingFace)            #
# --------------------------------------------------------------------------- #
# Конфигурация YOLO-методов: repo на HF, имя файла весов и метка класса.
YOLO_METHODS = {
    "yolo-stamp": {"repo_id": "PiDinoSauR/Stamp_Detection_10_12_2024", "filename": "weights/best.pt", "label": "stamp"},
    "yolo-signature": {
        "repo_id": "tech4humans/yolov8s-signature-detector",
        "filename": "yolov8s.pt",
        "label": "handwriting",
    },
}

# Кеш загруженных моделей в пределах одного запуска.
_yolo_cache: dict[str, object] = {}


def _load_yolo(method: str):
    """Скачивает (при необходимости) и загружает YOLO-модель для метода."""
    if method in _yolo_cache:
        return _yolo_cache[method]

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    cfg = YOLO_METHODS[method]
    weights = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"])
    model = YOLO(weights)
    _yolo_cache[method] = model
    return model


def _refine_box_to_ink_contour(image_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """Уточняет YOLO-бокс до криволинейного контура чернил внутри него.

    Возвращает (polygon, ink_fraction). polygon — контур чернил в координатах
    всего изображения (или None, если чернил в боксе практически нет —
    например, бокс пришёлся на палец/бумагу). ink_fraction — доля насыщенных
    «чернильных» пикселей в боксе, используется для отсева ложных детекций.
    """
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_bgr.shape[1], x2), min(image_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0

    roi = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    # Чернила штампа/ручки — насыщенные и не чёрные; кожа пальца сюда не попадает.
    mask = ((s > 50) & (v > 40) & (v < 250)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    ink_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, ink_fraction

    # Берём выпуклую оболочку всех значимых контуров — единая криволинейная граница.
    big = [c for c in contours if cv2.contourArea(c) > 0.001 * mask.size]
    if not big:
        return None, ink_fraction
    hull = cv2.convexHull(np.vstack(big))
    poly = hull.reshape(-1, 2)
    poly[:, 0] += x1
    poly[:, 1] += y1
    return poly.astype(np.int32), ink_fraction


def detect_yolo(image_bgr: np.ndarray, method: str, conf: float = 0.25) -> list[Detection]:
    """Прогоняет YOLO-детектор; боксы уточняются до криволинейного контура чернил.

    Если внутри бокса почти нет насыщенных чернил (ложное срабатывание на
    пальце/бумаге), детекция отбрасывается. Если уточнить не удалось, но
    чернила есть — используется исходный прямоугольник.
    """
    model = _load_yolo(method)
    label = YOLO_METHODS[method]["label"]

    # ultralytics ждёт RGB / путь / numpy; передаём BGR-массив, device=0 (GPU).
    results = model.predict(image_bgr, conf=conf, device=0, verbose=False)

    # Минимальная доля чернил в боксе, чтобы считать детекцию настоящей.
    min_ink_fraction = 0.01

    detections: list[Detection] = []
    for res in results:
        if res.boxes is None:
            continue
        for box in res.boxes:
            x1, y1, x2, y2 = (int(round(c)) for c in box.xyxy[0].tolist())
            score = float(box.conf[0])
            poly, ink = _refine_box_to_ink_contour(image_bgr, x1, y1, x2, y2)
            if ink < min_ink_fraction:
                continue  # бокс без чернил — почти наверняка палец/фон
            if poly is None:
                poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            detections.append(Detection(label=label, polygon=poly, score=score))
    return detections


# --------------------------------------------------------------------------- #
#  Метод 4. TorchScript-детектор штампов stamps-labs/yolov8-finetuned         #
# --------------------------------------------------------------------------- #
# Это не обычный ultralytics-чекпойнт, а экспортированная в TorchScript обёртка
# WrapperModel2: на вход — тензор [1,3,640,640] (RGB, /255) и порог conf, на
# выходе — Optional[(x2, boxes, scores)] с кандидатами ДО NMS (или None, если
# детекций нет). Реальные xyxy-координаты лежат в x2[:, :4] (в системе входа
# 640), conf — x2[:, 4]. NMS применяем сами через torchvision.

FINETUNED_STAMP = {"repo_id": "stamps-labs/yolov8-finetuned", "filename": "weights.pt", "label": "stamp"}
FINETUNED_IMGSZ = 640


def _load_finetuned_stamp():
    """Скачивает и загружает TorchScript-модель штампов на GPU (с кешем)."""
    if "yolo-stamp-finetuned" in _yolo_cache:
        return _yolo_cache["yolo-stamp-finetuned"]

    import torch
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(repo_id=FINETUNED_STAMP["repo_id"], filename=FINETUNED_STAMP["filename"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(weights, map_location=device).eval()
    _yolo_cache["yolo-stamp-finetuned"] = (model, device)
    return model, device


def _letterbox(image_bgr: np.ndarray, new: int = FINETUNED_IMGSZ, color: int = 114):
    """Масштабирует с сохранением пропорций и дополняет до квадрата new×new.

    Возвращает (canvas, ratio, pad_x, pad_y) для обратного пересчёта координат.
    """
    h, w = image_bgr.shape[:2]
    ratio = min(new / h, new / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    pad_x, pad_y = (new - nw) // 2, (new - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, ratio, pad_x, pad_y


def _finetuned_run_on_crop(crop_bgr: np.ndarray, conf: float, iou: float) -> np.ndarray:
    """Прогоняет TorchScript-модель на одном кропе. Возвращает [N, 5]: xyxy+conf
    в координатах кропа (или пустой массив)."""
    import torch
    import torchvision

    model, device = _load_finetuned_stamp()
    canvas, ratio, pad_x, pad_y = _letterbox(crop_bgr)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x, conf)
    if out is None:
        return np.zeros((0, 5), dtype=np.float32)

    x2, boxes, scores = out
    keep = torchvision.ops.nms(boxes, scores, iou)
    dets = x2[keep].cpu().numpy()
    res = dets[:, :5].astype(np.float32).copy()
    # Возврат из letterbox-системы 640 в координаты кропа.
    res[:, [0, 2]] = (res[:, [0, 2]] - pad_x) / ratio
    res[:, [1, 3]] = (res[:, [1, 3]] - pad_y) / ratio
    return res


def detect_yolo_finetuned(
    image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45, tile: int = 1100, overlap: float = 0.25
) -> list[Detection]:
    """Детектор штампов stamps-labs/yolov8-finetuned (TorchScript) с тайлингом.

    Модель трассирована жёстко под вход 640×640, поэтому крупные сканы режутся
    на перекрывающиеся плитки tile×tile (+полнокадровый проход для больших
    штампов); детекции собираются в общие координаты и объединяются глобальным
    NMS. Боксы уточняются до криволинейного контура чернил, как в detect_yolo.
    """
    import torch
    import torchvision

    h, w = image_bgr.shape[:2]
    step = max(1, int(tile * (1 - overlap)))

    parts = [_finetuned_run_on_crop(image_bgr, conf, iou)]  # полнокадровый проход
    ys = sorted({*range(0, max(1, h - tile + 1), step), max(0, h - tile)})
    xs = sorted({*range(0, max(1, w - tile + 1), step), max(0, w - tile)})
    for y in ys:
        for x in xs:
            d = _finetuned_run_on_crop(image_bgr[y : y + tile, x : x + tile], conf, iou)
            if len(d):
                d[:, [0, 2]] += x
                d[:, [1, 3]] += y
                parts.append(d)

    all_dets = np.vstack(parts)
    if len(all_dets) == 0:
        return []

    # Глобальный NMS, чтобы убрать дубли из перекрытий плиток.
    keep = torchvision.ops.nms(torch.tensor(all_dets[:, :4]), torch.tensor(all_dets[:, 4]), 0.3)
    all_dets = all_dets[keep.numpy()]

    min_ink_fraction = 0.01
    detections: list[Detection] = []
    for bx1, by1, bx2, by2, score in all_dets:
        x1, y1, x2i, y2i = int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))
        poly, ink = _refine_box_to_ink_contour(image_bgr, x1, y1, x2i, y2i)
        if ink < min_ink_fraction:
            continue
        if poly is None:
            poly = np.array([[x1, y1], [x2i, y1], [x2i, y2i], [x1, y2i]], dtype=np.int32)
        detections.append(Detection(label="stamp", polygon=poly, score=float(score)))
    return detections


# --------------------------------------------------------------------------- #
#  Отрисовка оверлеев                                                          #
# --------------------------------------------------------------------------- #
def draw_overlays(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Рисует криволинейные границы и подписи поверх копии изображения."""
    out = image_bgr.copy()
    overlay = image_bgr.copy()
    thickness = max(2, int(round(max(image_bgr.shape[:2]) / 600)))
    font_scale = max(0.5, max(image_bgr.shape[:2]) / 2500)

    for det in detections:
        color = COLOR_BY_LABEL.get(det.label, (0, 165, 255))
        poly = det.polygon.reshape(-1, 1, 2).astype(np.int32)
        # Полупрозрачная заливка + контурная (криволинейная) граница.
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Смешиваем заливку с лёгкой прозрачностью.
    out = cv2.addWeighted(overlay, 0.20, out, 0.80, 0)

    # Подписи рисуем поверх (после blend), чтобы текст был контрастным.
    for det in detections:
        color = COLOR_BY_LABEL.get(det.label, (0, 165, 255))
        x, y = det.polygon.min(axis=0)
        label_text = f"{det.label} {det.score:.2f}"
        cv2.putText(
            out,
            label_text,
            (int(x), max(0, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return out


# --------------------------------------------------------------------------- #
#  Прогон по директории                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class MethodStats:
    """Накопленная статистика по одному методу для итогового отчёта."""

    images: int = 0
    stamps: int = 0
    handwriting: int = 0
    seconds: float = 0.0
    per_image: list[tuple[str, int, int]] = field(default_factory=list)


def run_method(input_dir: Path, output_root: Path, method: str, conf: float) -> MethodStats:
    """Прогоняет один метод по всем изображениям директории."""
    out_dir = output_root / method
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    stats = MethodStats()
    print(f"\n=== Метод: {method} | изображений: {len(images)} | выход: {out_dir} ===")

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  !! не удалось прочитать {img_path.name}")
            continue

        t0 = time.time()
        if method == "classic":
            dets = detect_classic(image)
        elif method == "yolo-stamp-finetuned":
            dets = detect_yolo_finetuned(image, conf=conf)
        else:
            dets = detect_yolo(image, method, conf=conf)
        dt = time.time() - t0

        n_stamp = sum(1 for d in dets if d.label == "stamp")
        n_hand = sum(1 for d in dets if d.label == "handwriting")
        stats.images += 1
        stats.stamps += n_stamp
        stats.handwriting += n_hand
        stats.seconds += dt
        stats.per_image.append((img_path.name, n_stamp, n_hand))

        out_img = draw_overlays(image, dets)
        out_path = out_dir / f"{img_path.stem}_overlay.jpg"
        cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {img_path.name}: штампы={n_stamp}, рукопись={n_hand} ({dt:.2f}s)")

    return stats


def write_results_report(report_path: Path, all_stats: dict[str, MethodStats]) -> None:
    """Сохраняет сводный отчёт по прогону всех методов."""
    lines = ["# Результаты прогона детекторов\n"]
    lines.append("| Метод | Изобр. | Штампы | Рукопись | Время, с | с/изобр. |")
    lines.append("|---|---|---|---|---|---|")
    for method, st in all_stats.items():
        per = st.seconds / st.images if st.images else 0.0
        lines.append(f"| {method} | {st.images} | {st.stamps} | {st.handwriting} | {st.seconds:.1f} | {per:.2f} |")
    for method, st in all_stats.items():
        lines.append(f"\n## {method}\n")
        lines.append("| Изображение | Штампы | Рукопись |")
        lines.append("|---|---|---|")
        for name, ns, nh in st.per_image:
            lines.append(f"| {name} | {ns} | {nh} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nСводный отчёт: {report_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Детекция штампов и рукописных надписей на сканах.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Входная директория с изображениями.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Корень выходной директории (по умолчанию рядом со входной: <input>_detected).",
    )
    parser.add_argument(
        "--method",
        default="all",
        choices=["classic", "yolo-stamp", "yolo-stamp-finetuned", "yolo-signature", "all"],
        help="Метод детектирования.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Порог уверенности для YOLO.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        print(f"Входная директория не найдена: {input_dir}", file=sys.stderr)
        return 1

    # Выходной корень рядом со входной директорией, если не задан явно.
    output_root: Path = args.output_dir or input_dir.parent / f"{input_dir.name}_detected"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.method == "all":
        methods = ["classic", "yolo-stamp", "yolo-stamp-finetuned", "yolo-signature"]
    else:
        methods = [args.method]

    all_stats: dict[str, MethodStats] = {}
    for method in methods:
        try:
            all_stats[method] = run_method(input_dir, output_root, method, args.conf)
        except Exception as exc:  # noqa: BLE001 — хотим продолжить с другими методами
            print(f"  !! метод {method} упал: {type(exc).__name__}: {exc}", file=sys.stderr)

    if all_stats:
        write_results_report(output_root / "RESULTS.md", all_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
