"""Что именно уходит в LaMa: перехват входа сети (окно после уменьшения и маска дыры) в первом проходе A и вариантах B–D, листы по кадрам."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.inpainting.roi import roi_bounds_list
from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed, header_strip
from ocr_utils.scan_cropping.finger_removal.removal import FINGER_PADDING, FINGER_ROI_SCALE, brighten_finger_zone
from ocr_utils.scan_cropping.gpu_models import LAMA_ROI_MAX_SIDE
from ocr_utils.scan_cropping.v2.fingers import LAMA_HOLE_MAX_PX, TEXT_BLOCK_GUARD_PX, TIGHT_DILATE_PX, text_free_context
from ocr_utils.scan_cropping.v2.glyphs import glyph_protect
from ocr_utils.scan_cropping.v2.scale import Scale
from research.border_fill.frames import key_of
from research.finger_repaint.cli import PACK6_PX_PER_MM, read_set
from research.finger_repaint.repaint import (
    TIGHT_PADDING,
    TIGHT_ROI_SCALE,
    RepaintInput,
    Variant,
    load_input,
    repaint,
    repaint_hole,
)

log = logging.getLogger(__name__)

# Высветление зоны пальца первого прохода (как в pack6_params), уровни.
LIGHT_INCREMENT = (20.0, 20.0)
# Высота плитки окна на листе, px; миниатюра кадра — той же высоты.
TILE_H = 420
# Цвета: контур дыры на входе сети, рамки окон на миниатюре (по окну свой цвет).
COLOR_HOLE = (0, 0, 230)
COLOR_ROI = [(220, 90, 20), (0, 150, 0), (180, 105, 255), (0, 165, 255), (200, 60, 200), (140, 140, 0)]
# Варианты с LaMa (E, G — без сети, F — окна B).
LAMA_VARIANTS = (Variant.REPAINT, Variant.TIGHT, Variant.NO_MAT)


@dataclass(frozen=True)
class NetCall:
    """Один прогон сети: вход как он ушёл в LaMa (RGB uint8, уже уменьшенный и добитый до кратности 8), маска, выход."""

    image: np.ndarray
    mask: np.ndarray
    output: np.ndarray


class LamaRecorder:
    """Обёртка над TorchScript-моделью LaMa: пропускает вызов насквозь и запоминает вход и выход (``calls``)."""

    def __init__(self, lama) -> None:
        """Args: lama — исходная модель ``GpuModels._lama``."""
        self.lama = lama
        self.calls: list[NetCall] = []

    def __call__(self, image, mask):
        """Прогон сети; вход (1,3,H,W) и маска (1,1,H,W) float 0..1 на устройстве — запоминаются в uint8."""
        out = self.lama(image, mask)
        to_u8 = lambda t: np.clip(t[0].permute(1, 2, 0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)  # noqa
        self.calls.append(NetCall(to_u8(image), to_u8(mask)[..., 0], to_u8(out)))
        return out


def first_pass(inp: RepaintInput, models, scale: Scale) -> np.ndarray:
    """Повтор LaMa первого прохода (``fingers.remove_fingers_inkback``) по маске пальца из кэша; возвращает закрас BGR.

    Args:
        inp: Вход кадра (оригинал, зона пальца).
        models: ``GpuModels`` с LaMa (обёрнутой ``LamaRecorder``).
        scale: Масштабы прогона.

    Returns:
        Кадр BGR: зона пальца — заливка LaMa (без возврата краски).
    """
    rgb = cv2.cvtColor(inp.original, cv2.COLOR_BGR2RGB)
    paint = inp.paint.astype(np.uint8) * 255
    dilate_px = scale.frame_px(TIGHT_DILATE_PX)
    bright = brighten_finger_zone(inp.original, paint, LIGHT_INCREMENT, 2 * dilate_px)
    context = text_free_context(bright, glyph_protect(rgb, TEXT_BLOCK_GUARD_PX, scale.glyph) > 0, inp.paint)
    filled = models.inpaint(
        cv2.cvtColor(context, cv2.COLOR_BGR2RGB),
        paint,
        padding=FINGER_PADDING,
        roi_scale=FINGER_ROI_SCALE,
        roi_max_side=LAMA_ROI_MAX_SIDE,
        hole_max_px=LAMA_HOLE_MAX_PX,
    )
    return cv2.cvtColor(filled, cv2.COLOR_RGB2BGR)


def fit_height(img: np.ndarray, height: int) -> np.ndarray:
    """Картинка, приведённая к высоте ``height`` с сохранением пропорций (INTER_AREA вниз, INTER_CUBIC вверх)."""
    k = height / img.shape[0]
    interp = cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, round(img.shape[1] * k)), height), interpolation=interp)


def call_tiles(call: NetCall, color: tuple[int, int, int]) -> np.ndarray:
    """Плитка окна: вход сети с контуром дыры | выход сети, рамка цвета окна, подпись с размером входа."""
    image = cv2.cvtColor(call.image, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours((call.mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    shown = image.copy()
    cv2.drawContours(shown, contours, -1, COLOR_HOLE, max(1, image.shape[0] // 300))
    pair = [fit_height(shown, TILE_H), fit_height(cv2.cvtColor(call.output, cv2.COLOR_RGB2BGR), TILE_H)]
    tile = np.hstack([pair[0], np.full((TILE_H, 6, 3), 255, np.uint8), pair[1]])
    tile = cv2.copyMakeBorder(tile, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)
    caption = header_strip([f"вход сети {call.image.shape[1]}×{call.image.shape[0]} px  |  выход"], tile.shape[1])
    return np.vstack([caption, tile])


def frame_thumb(img: np.ndarray, hole: np.ndarray, rois: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Миниатюра кадра (контекст LaMa) с дырой и рамками окон в px кадра."""
    k = TILE_H / img.shape[0]
    thumb = fit_height(img, TILE_H)
    small_hole = cv2.resize(hole.astype(np.uint8), (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(small_hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(thumb, contours, -1, COLOR_HOLE, 1)
    for i, (x1, y1, x2, y2) in enumerate(rois):
        p1, p2 = (round(x1 * k), round(y1 * k)), (round(x2 * k) - 1, round(y2 * k) - 1)
        cv2.rectangle(thumb, p1, p2, COLOR_ROI[i % len(COLOR_ROI)], 2)
    return thumb


def variant_row(
    title: str, context: np.ndarray, hole: np.ndarray, rois: list[tuple[int, int, int, int]], calls: list[NetCall]
) -> np.ndarray:
    """Строка листа: подпись варианта, миниатюра кадра с окнами, плитки входов сети по окнам."""
    parts = [frame_thumb(context, hole, rois)]
    for i, call in enumerate(calls):
        tile = call_tiles(call, COLOR_ROI[i % len(COLOR_ROI)])
        parts.append(np.full((tile.shape[0], 12, 3), 255, np.uint8))
        parts.append(tile)
    height = max(p.shape[0] for p in parts)
    parts = [
        cv2.copyMakeBorder(p, 0, height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in parts
    ]
    row = np.hstack(parts)
    return np.vstack([header_strip([title], row.shape[1]), row])


def variant_context(inp: RepaintInput, variant: Variant, scale: Scale) -> np.ndarray:
    """Контекст варианта для миниатюры (то, из чего режутся окна): кадр первого прохода или с бумагой вместо коврика."""
    from ocr_utils.scan_cropping.v2.fill import fill_non_paper_pull_push

    if variant == Variant.NO_MAT:
        return fill_non_paper_pull_push(inp.clean, inp.book, scale.px_per_mm, inp.spine_x)
    return inp.clean


def frame_sheet(key: str, inp: RepaintInput, models, recorder: LamaRecorder, scale: Scale) -> np.ndarray:
    """Лист кадра: строки A (первый проход по маске пальца), B, C, D — окна на кадре и вход/выход сети по окнам."""
    rows = []
    recorder.calls.clear()
    first_pass(inp, models, scale)
    rois = roi_bounds_list(inp.paint.astype(np.uint8), FINGER_PADDING, FINGER_ROI_SCALE)
    title = (
        f"A — первый проход: дыра = вся раздутая зона пальца, контекст = оригинал с высветлением +20 и без блоков "
        f"текста; окно = рамка компоненты + {FINGER_PADDING} px, ×{FINGER_ROI_SCALE}"
    )
    rows.append(variant_row(title, inp.original, inp.paint, rois, list(recorder.calls)))
    hole = repaint_hole(inp, scale)
    titles = {
        Variant.REPAINT: f"B — дыра = палец ∩ книга (+1 мм), контекст = итог A без блоков текста; окно + {FINGER_PADDING} px, ×{FINGER_ROI_SCALE}",
        Variant.TIGHT: f"C — как B, окно + {TIGHT_PADDING} px, ×{TIGHT_ROI_SCALE}",
        Variant.NO_MAT: "D — как B, но вне книги в контексте вместо коврика цвет бумаги (pull-push)",
    }
    for variant in LAMA_VARIANTS:
        recorder.calls.clear()
        repaint(inp, variant, models, scale)
        padding, roi_scale = (
            (TIGHT_PADDING, TIGHT_ROI_SCALE) if variant == Variant.TIGHT else (FINGER_PADDING, FINGER_ROI_SCALE)
        )
        rois = roi_bounds_list(hole.astype(np.uint8), padding, roi_scale)
        rows.append(
            variant_row(titles[variant], variant_context(inp, variant, scale), hole, rois, list(recorder.calls))
        )
    width = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 16, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows
    ]
    legend = [
        LegendEntry("граница дыры (что LaMa закрашивает)", COLOR_HOLE),
        LegendEntry("окна LaMa (рамка плитки — того же цвета)", COLOR_ROI[0], style=SampleStyle.BOX),
    ]
    header = [
        f"{key} — что подаётся в LaMa",
        f"вход сети = окно кадра, уменьшенное так, чтобы рамка дыры ≤ {LAMA_HOLE_MAX_PX} px, добитое до кратности 8; "
        "под дырой сеть сама обнуляет пиксели (image·(1−mask)). E, G — без LaMa; F — окна B.",
    ]
    return framed(np.vstack(rows), header, legend)


@click.command()
@click.option("--base", "base_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Прогон base стенда (кэш с clean.png и fingers.npz).")  # fmt: skip
@click.option("--src-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень входа (только чтение).")  # fmt: skip
@click.option("--set", "set_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Файл множества.")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Куда листы.")
@click.option("--limit", default=0, show_default=True, help="Сколько кадров с конца множества (0 — все).")
def main(base_dir: Path, src_root: Path, set_path: Path, out_dir: Path, limit: int) -> None:
    """Листы «что подаётся в LaMa» по кадрам множества (GPU, по кадру)."""
    from ocr_utils.scan_cropping.gpu_models import GpuModels

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rels = read_set(set_path)
    rels = rels[-limit:] if limit else rels
    scale = Scale(PACK6_PX_PER_MM)
    out_dir.mkdir(parents=True, exist_ok=True)
    with GpuModels(with_detection=False, with_lama=True) as models:
        recorder = LamaRecorder(models._lama)
        models._lama = recorder
        for i, rel in enumerate(rels, 1):
            key = key_of(rel)
            inp = load_input(src_root / rel, base_dir / "cache" / key)
            if not inp.paint.any():
                continue
            sheet = frame_sheet(key, inp, models, recorder, scale)
            cv2.imwrite(str(out_dir / f"{i:02d}_{key}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
            log.info("%d/%d %s", i, len(rels), key)


if __name__ == "__main__":
    main()
