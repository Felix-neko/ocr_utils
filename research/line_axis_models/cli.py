"""Команды стенда: ``score`` — таблица мер движков по прогону, ``curls`` — вырезки концов кривых строк всеми движками, ``grid`` — сетка полос."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks.overlay import COLOUR_AXIS
from ocr_utils.page_layout.text_blocks.page import render_page
from research.line_axis_models.measures import REFERENCE, markdown_table, read_run, score_engine

# Подписи — тёмным на белой подложке, как в оверлеях text_blocks.
COLOUR_TEXT = (20, 20, 20)
# Поле вырезки вокруг конца строки, мм по x и по y.
CROP_MM = (22.0, 7.0)
# Увеличение вырезки: оси при 300 dpi видны до пикселя.
CROP_ZOOM = 3


@click.group()
def main() -> None:
    """Стенд сравнения чужих движков осей строк (``reports/line_axis_models.md``)."""


@main.command()
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--variant", type=click.Choice(["geo", "nogeo", "all"]), default="all", show_default=True)
def score(run_dir: Path, variant: str) -> None:
    """Таблица мер всех движков прогона ``text_blocks analyze`` (печать и ``<run-dir>/scores_<variant>.md``)."""
    pages = [page for page in read_run(run_dir) if variant == "all" or page.key[2] == variant]
    engines = sorted({page.engine for page in pages}, key=lambda name: (name != REFERENCE, name))
    scores = [score_engine(engine, pages) for engine in engines]
    table = markdown_table(scores)
    (run_dir / f"scores_{variant}.md").write_text(table, encoding="utf-8")
    click.echo(table)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    """Картинка с белой полосой-подписью сверху (подпись латиницей и цифрами — cv2 не рисует кириллицу)."""
    band = np.full((28, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(band, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOUR_TEXT, 1, cv2.LINE_AA)
    return np.vstack([band, image])


@main.command()
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--geo-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--nogeo-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--per-page", default=2, show_default=True, help="сколько самых изогнутых строк опоры брать с полосы")
def curls(run_dir: Path, geo_dir: Path, nogeo_dir: Path, per_page: int) -> None:
    """Вырезки концов самых изогнутых строк: один и тот же участок, по вырезке на движок, в ряд.

    Строки выбираются по |сагитте| оси опоры (``ink``); у строки берётся конец, где ось дальше от
    хорды. Выход — ``<run-dir>/curls/<полоса>_<n>.jpg``.
    """
    pages = read_run(run_dir)
    groups: dict[tuple, dict] = defaultdict(dict)
    for page in pages:
        groups[page.key][page.engine] = page
    engines = sorted({page.engine for page in pages}, key=lambda name: (name != REFERENCE, name))
    out = run_dir / "curls"
    out.mkdir(parents=True, exist_ok=True)
    dirs = {"geo": geo_dir, "nogeo": nogeo_dir}
    for key, group in sorted(groups.items()):
        if REFERENCE not in group:
            continue
        reference = group[REFERENCE]
        gray = render_page(dirs[key[2]] / f"{key[0]}.pdf", key[1])
        k = RENDER_DPI / reference.dpi
        chosen = sorted(reference.axes, key=lambda axis: -abs(axis.sagitta_mm))[:per_page]
        for number, target in enumerate(chosen):
            # Конец строки, где ось дальше всего от своей хорды: там и завиток.
            ys = target.points[:, 1]
            chord = np.interp(target.points[:, 0], [target.x0, target.x1], [ys[0], ys[-1]])
            end_x = target.x0 if np.argmax(np.abs(ys - chord)) < len(ys) / 2 else target.x1
            half_w, half_h = (CROP_MM[0] / 25.4 * RENDER_DPI, CROP_MM[1] / 25.4 * RENDER_DPI)
            cx, cy = end_x * k, float(target.y_at(np.array([end_x]))[0]) * k
            x0, x1 = int(max(0, cx - half_w)), int(min(gray.shape[1], cx + half_w))
            y0, y1 = int(max(0, cy - half_h)), int(min(gray.shape[0], cy + half_h))
            tiles = []
            for engine in engines:
                crop = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
                crop = cv2.resize(crop, None, fx=CROP_ZOOM, fy=CROP_ZOOM, interpolation=cv2.INTER_CUBIC)
                for axis in group[engine].axes if engine in group else []:
                    points = (axis.points * k - [x0, y0]) * CROP_ZOOM
                    cv2.polylines(crop, [np.round(points).astype(np.int32)], False, COLOUR_AXIS, 2, cv2.LINE_AA)
                tiles.append(_label(crop, engine if engine in group else f"{engine}: no run"))
            name = f"{key[0]}_p{key[1]:03d}_{key[2]}_{number}.jpg"
            cv2.imwrite(str(out / name), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
            click.echo(name)


@main.command()
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--width", default=700, show_default=True, help="ширина одной полосы в сетке, px")
def grid(run_dir: Path, width: int) -> None:
    """Сетка «движок × полоса» из готовых оверлеев ``analyze``: по файлу на полосу и вариант в ``<run-dir>/grid``."""
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted((run_dir / "overlays").glob("*.jpg")):
        stem, _, engine = path.stem.rpartition("_")
        groups[stem][engine] = path
    out = run_dir / "grid"
    out.mkdir(parents=True, exist_ok=True)
    for stem, files in sorted(groups.items()):
        engines = sorted(files, key=lambda name: (name != REFERENCE, name))
        tiles = []
        for engine in engines:
            image = cv2.imread(str(files[engine]))
            tiles.append(cv2.resize(image, (width, int(image.shape[0] * width / image.shape[1])), cv2.INTER_AREA))
        height = max(tile.shape[0] for tile in tiles)
        tiles = [np.vstack([tile, np.full((height - tile.shape[0], width, 3), 255, np.uint8)]) for tile in tiles]
        # По четыре полосы в ряд: иначе сетка на десяток движков не помещается на экран.
        rows = [
            np.hstack(tiles[i : i + 4] + [np.full_like(tiles[0], 255)] * (4 - len(tiles[i : i + 4])))
            for i in range(0, len(tiles), 4)
        ]
        cv2.imwrite(str(out / f"{stem}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        click.echo(stem)


__all__ = ["main"]
