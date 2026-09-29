"""Собственный разбор вёрстки pero (ParseNet) по полосам: регионы — текстовые блоки pero, строки — базовые линии и контуры; JSON и оверлеи."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.page_layout.text_blocks.engines.external import run_worker
from ocr_utils.page_layout.text_blocks.engines.pero import PeroEngine
from research.edge_marks.candidates import Candidate

# Цвета BGR (палитра навыка draw-overlay): граница блока — синий, контур строки — приглушённый, ось строки — зелёный.
COLOR_REGION = (220, 90, 20)
COLOR_LINE_BOUNDARY = (150, 170, 120)
COLOR_BASELINE = (40, 170, 40)
COLOR_JUNK = (0, 0, 220)
# Заливка регионов — полупрозрачно, чтобы буквы читались.
REGION_ALPHA = 0.18
# Толщины линий, пиксели полосы 300 dpi.
REGION_THICKNESS, LINE_THICKNESS, BASELINE_THICKNESS = 4, 1, 2


def pero_page(gray: np.ndarray, engine: PeroEngine) -> dict:
    """Разбор полосы воркером pero (модель и конфиг — как у движка ``pero`` детектора текстовых блоков).

    Args:
        gray: Полоса в оттенках серого, 300 dpi.
        engine: Движок pero (интерпретатор окружения и ``config.ini``).

    Returns:
        ``{"lines": [{"baseline", "boundary", "height"}], "regions": [полигон]}`` в пикселях полосы.
    """
    return run_worker(engine.python, "pero_worker.py", gray, extra=[str(engine.config)])


def draw(gray: np.ndarray, layout: dict, title: str, junk: list[Candidate]) -> np.ndarray:
    """Оверлей разбора pero: регионы, контуры и базовые линии строк, рамки размеченного сора; шапка и легенда в полях.

    Args:
        gray: Полоса, 300 dpi.
        layout: Ответ :func:`pero_page`.
        title: Шапка.
        junk: Размеченные кандидаты-сор полосы (рамки в тех же пикселях).

    Returns:
        Картинка BGR.
    """
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    regions = [np.asarray(r, dtype=np.int32) for r in layout.get("regions", []) if len(r) >= 3]
    # Заливка регионов — отдельным слоем с прозрачностью.
    layer = canvas.copy()
    for polygon in regions:
        cv2.fillPoly(layer, [polygon], COLOR_REGION)
    cv2.addWeighted(layer, REGION_ALPHA, canvas, 1 - REGION_ALPHA, 0, canvas)
    # Строки: контур тонко, базовая линия — насыщенно.
    for line in layout.get("lines", []):
        if line.get("boundary"):
            cv2.polylines(canvas, [np.asarray(line["boundary"], np.int32)], True, COLOR_LINE_BOUNDARY, LINE_THICKNESS)
        cv2.polylines(canvas, [np.asarray(line["baseline"], np.int32)], False, COLOR_BASELINE, BASELINE_THICKNESS)
    # Контур регионов поверх строк.
    for polygon in regions:
        cv2.polylines(canvas, [polygon], True, COLOR_REGION, REGION_THICKNESS)
    for c in junk:
        x0, y0, x1, y1 = c.box
        cv2.rectangle(canvas, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), COLOR_JUNK, 2)
    legend = [
        LegendEntry("регион pero (текстовый блок)", COLOR_REGION, alpha=REGION_ALPHA, style=SampleStyle.BOX),
        LegendEntry("базовая линия строки pero", COLOR_BASELINE),
        LegendEntry("контур строки pero", COLOR_LINE_BOUNDARY),
        LegendEntry("размеченный сор / пометка", COLOR_JUNK),
    ]
    header = [title, f"регионов {len(regions)}, строк {len(layout.get('lines', []))}"]
    return framed(canvas, header, legend)


def run(pages_dir: Path, candidates: list[Candidate], out_dir: Path, engine: PeroEngine) -> int:
    """Разобрать pero все полосы ``pages_dir/*.png``: ``out_dir/json/<полоса>.json`` и ``out_dir/overlays/<полоса>.jpg``.

    Готовый JSON не пересчитывается (повторный запуск только перерисовывает оверлеи).

    Args:
        pages_dir: Полосы PNG 300 dpi, имя файла — ключ полосы.
        candidates: Кандидаты тестового множества (для рамок сора).
        out_dir: Куда писать.
        engine: Движок pero.

    Returns:
        Число полос.
    """
    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    paths = sorted(pages_dir.glob("*.png"))
    for path in paths:
        key = path.stem
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        cached = out_dir / "json" / f"{key}.json"
        if cached.exists():
            layout = json.loads(cached.read_text())
        else:
            layout = pero_page(gray, engine)
            cached.write_text(json.dumps(layout))
        junk = [c for c in candidates if c.key == key and c.truth == "junk"]
        picture = draw(gray, layout, f"{key} — разбор вёрстки pero (регионы и строки), nogeo 300 dpi", junk)
        cv2.imwrite(str(out_dir / "overlays" / f"{key}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{key}: регионов {len(layout.get('regions', []))}, строк {len(layout.get('lines', []))}", flush=True)
    return len(paths)


__all__ = ["draw", "pero_page", "run"]
