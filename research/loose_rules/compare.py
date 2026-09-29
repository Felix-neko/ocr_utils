"""Текстовые блоки полосы с прежними сиротами и без отбракованных: числа осей и блоков, склейка «было | стало»."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, header_strip, legend_strip
from ocr_utils.page_layout.pack_analysis.final import text_blocks
from ocr_utils.page_layout.pack_analysis.stages import PageTask, load_image, page_key
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks import overlay as blocks_overlay
from ocr_utils.page_layout.text_blocks.columns import GutterMode
from ocr_utils.page_layout.text_blocks.page import AxisKind
from ocr_utils.page_layout.text_blocks.sides_overlay import side_by_side

# Ширина оверлея полосы (как у разбора пака: ``pack_analysis.final.OVERLAY_WIDTH``).
OVERLAY_WIDTH = 1600
# Отброшенная сирота на «было» — красный «отвергнуто» палитры (``stages.COLOUR_BAD``), поверх её же барьера.
COLOUR_DROPPED = (0, 0, 220)


def _record(final: dict, rules: list[dict]) -> dict:
    """Запись полосы для ``pack_analysis.final.text_blocks`` из итогового JSON с заданным списком сирот.

    Args:
        final: Итоговый JSON разбора пака.
        rules: Сироты, которые пойдут в барьеры.

    Returns:
        Запись с ``page``, ``size``, ``dpi``, ``loose_rules`` и ``rotate_cw`` (как ``gutter_crossing.page_record``).
    """
    orientation = final.get("orientation") or {}
    return {
        "page": final["page"],
        "size": final["size"],
        "dpi": final["dpi"],
        "loose_rules": rules,
        "rotate_cw": orientation.get("rotate_cw", 0) if orientation.get("apply") else 0,
    }


def _draw(analysis, hints, gray300: np.ndarray, dropped: list[dict], dpi: int, title: list[str]) -> np.ndarray:
    """Оверлей разбора текстовых блоков с барьерами и отброшенными сиротами; шапка и легенды в полях.

    Args:
        analysis: ``PageAnalysis`` полосы (пиксели рабочей копии).
        hints: Подсказки разбора (барьеры-линейки рисуются ими).
        gray300: Серый рендер ``RENDER_DPI``.
        dropped: Отброшенные сироты (JSON, пиксели скана) — рисуются красным; пусто — не рисуются.
        dpi: Разрешение скана (для пересчёта точек сирот).
        title: Строки шапки.

    Returns:
        Картинка BGR.
    """
    scale = OVERLAY_WIDTH / analysis.width
    small = cv2.resize(gray300, (OVERLAY_WIDTH, int(round(analysis.height * scale))), interpolation=cv2.INTER_AREA)
    canvas = blocks_overlay.draw(analysis, small, scale=scale, hints=hints)
    # Точки сирот — в пикселях скана: в рабочую копию (``WORK_DPI``), затем в холст.
    to_canvas = WORK_DPI / dpi * scale
    for item in dropped:
        points = (np.asarray(item["points"], dtype=np.float64) * to_canvas).astype(np.int32)
        cv2.polylines(canvas, [points], False, COLOUR_DROPPED, 3, cv2.LINE_AA)
    width = canvas.shape[1]
    return np.vstack(
        [
            header_strip(title, width),
            canvas,
            legend_strip([LegendEntry("сирота отброшена стендом", COLOUR_DROPPED)], width, "стенд"),
            legend_strip(blocks_overlay.legend_entries(body_axis=True, hints=hints), width, "текстовые блоки"),
        ]
    )


def compare_page(key: str, drop: set[int], pack_dir: Path, sharpened_dir: Path, out_dir: Path) -> dict:
    """Разобрать полосу с прежними сиротами и без отброшенных, записать склейку «было | стало».

    Args:
        key: Полоса.
        drop: Номера отброшенных сирот в ``loose_rules`` итогового JSON.
        pack_dir: Разбор пака.
        sharpened_dir: Заострённые сканы.
        out_dir: Куда писать ``pages/<полоса>.jpg``.

    Returns:
        Строка сводки: оси и блоки до и после, число отброшенных сирот.
    """
    final = json.loads((pack_dir / "pages" / f"{page_key(key)}.json").read_text(encoding="utf-8"))
    rules = final["loose_rules"]
    kept = [rule for index, rule in enumerate(rules) if index not in drop]
    dropped = [rule for index, rule in enumerate(rules) if index in drop]
    image = load_image(PageTask(key, sharpened_dir / f"{key}.jpg"), _record(final, rules)["rotate_cw"])
    gray300 = image.gray_at(RENDER_DPI)
    results = []
    for chosen in (rules, kept):
        # Те же ключи, что у прогона v4 (``run_pack1_analysis_v4.sh``): вторая ось строки, межколонники short.
        results.append(text_blocks(image, _record(final, chosen), final["objects"], AxisKind.BODY, GutterMode.SHORT))
    (before, before_hints), (after, after_hints) = results
    row = {
        "key": key,
        "dropped": len(dropped),
        "axes_before": len(before.axes),
        "axes_after": len(after.axes),
        "blocks_before": len(before.blocks),
        "blocks_after": len(after.blocks),
    }
    title = [key, f"сирот {len(rules)}, отброшено {len(dropped)}"]
    pictures = [
        _draw(
            before,
            before_hints,
            gray300,
            dropped,
            final["dpi"],
            title + [f"осей {row['axes_before']}, блоков {row['blocks_before']}"],
        ),
        _draw(
            after,
            after_hints,
            gray300,
            [],
            final["dpi"],
            title + [f"осей {row['axes_after']}, блоков {row['blocks_after']}"],
        ),
    ]
    target = out_dir / "pages" / f"{page_key(key)}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(target),
        side_by_side(pictures, ["было: все сироты v4", "стало: без отброшенных"]),
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    return row
