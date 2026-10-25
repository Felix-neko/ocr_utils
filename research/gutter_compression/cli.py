"""CLI стенда «сжатие букв у корешка»: мера по выпрямленным итогам наборов (``scan``), отбор поясов и листы (``select``, ``band-sheets``), проверка мер на синтетическом сжатии (``synth``), буквы tesseract (``letters``), выпрямление с растяжением у корешка и листы «было | стало» (``stretch-run``), форма амплитуды по высоте (``shape``)."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from enum import StrEnum
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd

from ocr_utils.scan_cropping.v2.pipeline import init_worker, read_bgr
from research.gutter_compression.frames import FrameRef, FrameSet, frame_geometry, frame_layout, set_frames
from research.gutter_compression.measure import (
    HORIZONTAL,
    MeasureParams,
    Method,
    Side,
    ink_mask,
    page_windows,
    spread_windows,
)
from research.gutter_compression.profile import (
    NEAR_MM,
    LinePosition,
    CompressionModel,
    fit_model,
    ratio_column,
    side_summary,
)
from research.gutter_compression.letters import page_words, paired_windows
from research.gutter_compression.letter_fit import FitSource
from research.gutter_compression.letter_profile import spread_letters
from research.gutter_compression.run import StretchConfig, stretch_frame, summary_row, synth_frame, write_sheet
from research.gutter_compression.stretch import StretchLimits, StretchMode
from research.gutter_compression.sheets import band_sheet, gutter_crop
from research.gutter_compression.shape_run import band_shape_table, curves_sheet, shape_side, side_tasks, summarize
from research.gutter_compression.shape_run import write_sheet as write_curves

log = logging.getLogger(__name__)

# Воркеров по умолчанию: кадр читает оригинал с Я.Диска (NTFS-3G, для областей страниц по кэшу) и держит ~1 ГБ
# картинок — упор в диск и память раньше, чем в 16 ядер; машина к тому же занята параллельным прогоном v2.
DEFAULT_JOBS = 8


class Status(StrEnum):
    """Исход кадра в ``scan``."""

    OK = "ok"
    MISSING = "missing"  # нет выпрямленного итога
    NO_LAYOUT = "no_layout"  # областей страниц нет (не две страницы, нет бумаги)
    MISMATCH = "mismatch"  # холст раскладки не совпал с выпрямленным итогом (кэш перезаписан)
    ERROR = "error"


def windows_path(out_root: Path, ref: FrameRef) -> Path:
    """Таблица окон кадра в выходе ``scan``."""
    return out_root / "windows" / ref.frame_set.label / f"{ref.job.key}.csv.gz"


def scan_frame(args: tuple[FrameRef, Path, MeasureParams]) -> dict:
    """Мера одного кадра (для пула): раскладка по кэшу v2, окна строк выпрямленного итога, сводка сторон.

    Args:
        args: ``(кадр, корень выхода, параметры меры)``.

    Returns:
        Строка сводки: key, status, spine_x, page_top/page_height по сторонам и поля ``side_summary`` с приставкой
        ``L_``/``R_``.
    """
    ref, out_root, params = args
    row: dict = {"key": ref.key, "status": Status.OK.value}
    t0 = time.perf_counter()
    try:
        if not ref.rectified_path.exists() or not ref.v2_path.exists():
            return row | {"status": Status.MISSING.value}
        v2 = read_bgr(ref.v2_path)
        layout = frame_layout(frame_geometry(ref), v2.shape[:2], ref.frame_set.dpi)
        if layout is None:
            return row | {"status": Status.NO_LAYOUT.value}
        gray = cv2.cvtColor(read_bgr(ref.rectified_path), cv2.COLOR_BGR2GRAY)
        if gray.shape != (layout.height, layout.width):
            return row | {"status": Status.MISMATCH.value}
        table = spread_windows(gray, layout.rects, layout.spine_x, ref.frame_set.px_per_mm, params)
        path = windows_path(out_root, ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.4g")
        row["spine_x"] = round(layout.spine_x, 1)
        for side, rect in zip(Side, layout.rects):
            row[f"{side.value}_top"] = round(rect.y0, 1)
            row[f"{side.value}_height"] = round(rect.height, 1)
            part = table[table.side == side.value] if len(table) else table
            if len(part):
                row.update({f"{side.value}_{k}": v for k, v in side_summary(part).items()})
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка", ref.key)
        row.update(status=Status.ERROR.value, message=repr(exc))
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


def parse_sets(sets: tuple[tuple[str, Path, Path, Path, Path, float], ...]) -> list[FrameSet]:
    """Наборы из опций ``--set``."""
    return [FrameSet(label, inp, cache, v2, rect, ppm) for label, inp, cache, v2, rect, ppm in sets]


SET_OPTION = click.option(
    "--set",
    "sets",
    type=(str, click.Path(path_type=Path, exists=True, file_okay=False), click.Path(path_type=Path, exists=True),
          click.Path(path_type=Path, exists=True), click.Path(path_type=Path, exists=True), float),
    multiple=True,
    required=True,
    help="Набор: ПОДПИСЬ ВХОД КЭШ_V2 ИТОГИ_V2 ВЫПРЯМЛЕННЫЕ PX_PER_MM; повторяется.",
)  # fmt: skip


@click.group()
def main() -> None:
    """Стенд «сжатие букв у корешка» на выпрямленных разворотах."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command()
@SET_OPTION
@click.option("--out-root", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда.")
@click.option("--window-mm", type=float, default=MeasureParams.window_mm, show_default=True,
              help="Ширина окна вдоль строки, мм.")  # fmt: skip
@click.option("--far-mm", type=float, default=MeasureParams.far_mm, show_default=True,
              help="Опора строки — окна дальше этого от края текста у корешка, мм.")  # fmt: skip
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def scan(sets: tuple, out_root: Path, window_mm: float, far_mm: float, jobs: int) -> None:
    """Мера сжатия у корешка по всем кадрам наборов: windows/<набор>/<кадр>.csv.gz и scan.csv."""
    params = MeasureParams(window_mm=window_mm, far_mm=far_mm)
    refs = [ref for fs in parse_sets(sets) for ref in set_frames(fs)]
    log.info("кадров: %d, воркеров: %d", len(refs), jobs)
    rows = []
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=init_worker) as pool:
        for i, row in enumerate(pool.map(scan_frame, [(r, out_root, params) for r in refs]), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(refs):
                log.info("%d/%d", i, len(refs))
    out_root.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "scan.csv", index=False)
    log.info("готово: %s; %s", out_root / "scan.csv", table.status.value_counts().to_dict())


class Band(StrEnum):
    """Пояс сжатия стороны по мере у края текста (шаг штрихов по автокорреляции к опоре строки)."""

    STRONG = "1_сильное_до_0.70"
    CLEAR = "2_заметное_0.70-0.80"
    MILD = "3_слабое_0.80-0.88"
    SLIGHT = "4_едва_0.88-0.95"
    NONE = "5_нет_от_0.97"


# Нижние границы поясов (верхняя — у предыдущего); 0.95–0.97 — зазор между «едва» и контролем.
BAND_EDGES = ((Band.STRONG, 0.0, 0.70), (Band.CLEAR, 0.70, 0.80), (Band.MILD, 0.80, 0.88),
              (Band.SLIGHT, 0.88, 0.95), (Band.NONE, 0.97, 9.0))  # fmt: skip
# Сторона в счёт, если у края текста не меньше стольких строк.
MIN_SIDE_LINES = 8


def sides_table(scan_csv: Path) -> pd.DataFrame:
    """Стороны кадров из сводки ``scan`` с мерами и поясом.

    Args:
        scan_csv: Сводка ``scan.csv``.

    Returns:
        Таблица: key, side, lines, near (мера у края), mid (8–14 мм от края), edge_mm, spine_x, top, height, band.
    """
    table = pd.read_csv(scan_csv)
    table = table[table.status == Status.OK.value]
    parts = []
    for side in Side:
        p = side.value
        part = table[["key", "spine_x", f"{p}_lines", f"{p}_near_acf", f"{p}_mid_acf", f"{p}_edge_mm", f"{p}_top",
                      f"{p}_height"]].copy()  # fmt: skip
        part.columns = ["key", "spine_x", "lines", "near", "mid", "edge_mm", "top", "height"]
        part.insert(1, "side", p)
        parts.append(part)
    sides = pd.concat(parts, ignore_index=True)
    sides = sides[sides.lines >= MIN_SIDE_LINES].dropna(subset=["near"])
    sides["band"] = None
    for band, lo, hi in BAND_EDGES:
        sides.loc[(sides.near >= lo) & (sides.near < hi), "band"] = band.value
    return sides


@main.command()
@click.option("--scan-root", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True,
              help="Выход ``scan`` (scan.csv).")  # fmt: skip
@click.option("--rectified-root", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True,
              help="Корень выпрямленных итогов: <корень>/<ключ>.tiff (output/ прогона text-run).")  # fmt: skip
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Папка листов поясов и sides.csv.")
@click.option("--per-band", type=int, default=16, show_default=True, help="Сторон на лист пояса (случайно).")
@click.option("--px-per-mm", type=float, default=17.0, show_default=True,
              help="Плотность для размера вырезки, px/мм (ХиП 1991 — 16.3–17.9).")  # fmt: skip
def select(scan_root: Path, rectified_root: Path, out_dir: Path, per_band: int, px_per_mm: float) -> None:
    """Пояса сжатия по сторонам кадров: sides.csv и лист вырезок у корешка на пояс (середина высоты страницы)."""
    sides = sides_table(scan_root / "scan.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    sides.to_csv(out_dir / "sides.csv", index=False)
    log.info("сторон: %d; по поясам: %s", len(sides), sides.band.value_counts().sort_index().to_dict())
    rng = np.random.default_rng(0)
    for band in Band:
        part = sides[sides.band == band.value]
        if part.empty:
            continue
        pick = part.iloc[np.sort(rng.choice(len(part), min(per_band, len(part)), replace=False))]
        crops = []
        for row in pick.itertuples():
            image = read_bgr(rectified_root / f"{row.key}.tiff")
            crop = gutter_crop(image, row.spine_x, row.top + 0.5 * row.height, px_per_mm, 22.0, 8.0, 0.8)
            crops.append((crop, f"{row.key.split('/', 1)[1][:2]}/{Path(row.key).name} {row.side}: "
                                f"у края {row.near:.2f}, 8–14 мм {row.mid:.2f}"))  # fmt: skip
        header = [f"Пояс «{band.value}»: {len(part)} сторон, показано {len(pick)} (случайно); ХиП 1991, итоги v4",
                  "мера — шаг штрихов у края текста у корешка (окна 5 мм, первые 3 мм от края) к той же строке "
                  "дальше 25 мм от края; вырезки ±22 мм от линии корешка, середина высоты страницы"]  # fmt: skip
        cv2.imwrite(str(out_dir / f"band_{band.value}.jpg"), band_sheet(crops, 2, header),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])  # fmt: skip
        log.info("лист пояса %s", out_dir / f"band_{band.value}.jpg")


# Сетка синтетических моделей сжатия: амплитуда на линии корешка и длина спада, мм.
SYNTH_AMPLITUDES = (0.0, 0.15, 0.3, 0.45)
SYNTH_LENGTHS_MM = (5.0, 12.0)
# Сетка сверки профиля: от края текста до 15 мм наружу, мм.
SYNTH_SPAN_MM = 15.0


def outer_windows(gray: np.ndarray, layout, px_per_mm: float, params: MeasureParams) -> pd.DataFrame:
    """Нулевая проверка меры: окна от ВНЕШНЕГО края текста внутрь (там сгиба нет, отношение должно быть ≈ 1).

    Внешний край страницы подставляется вместо линии корешка: у левой страницы текст правее него (как у правой
    страницы от корешка), у правой — левее.

    Args:
        gray: Выпрямленный холст.
        layout: Раскладка.
        px_per_mm: Плотность, px/мм.
        params: Параметры меры.

    Returns:
        Окна обеих страниц; ``side`` — какая страница (L/R).
    """
    ink = ink_mask(gray, px_per_mm, params)
    margin = int(round(px_per_mm))
    left, right = layout.rects
    a = page_windows(ink, int(left.x0) + margin, int(round(layout.spine_x)), left.x0, Side.RIGHT, px_per_mm, params)
    b = page_windows(ink, int(round(layout.spine_x)), int(right.x1) - margin, right.x1, Side.LEFT, px_per_mm, params)
    a["side"], b["side"] = Side.LEFT.value, Side.RIGHT.value
    return pd.concat([a, b], ignore_index=True)


def synth_one(args: tuple[FrameRef, MeasureParams]) -> list[dict]:
    """Синтетика одного кадра (для пула): нулевая проверка у внешнего края и восстановление моделей сжатия.

    Args:
        args: ``(кадр контроля без сжатия, параметры меры)``.

    Returns:
        Строки: key, side, case (``outer`` или ``A=…,L=…``), method, true_amp, true_len_mm, fit_amp, fit_len_mm,
        near (медиана отношения у края), true_near, rms (ошибка профиля c(d) от края до +15 мм), excess_err_mm
        (ошибка лишней длины на тех же 15 мм).
    """
    ref, params = args
    rows: list[dict] = []
    try:
        v2 = read_bgr(ref.v2_path)
        layout = frame_layout(frame_geometry(ref), v2.shape[:2], ref.frame_set.dpi)
        gray = cv2.cvtColor(read_bgr(ref.rectified_path), cv2.COLOR_BGR2GRAY)
        if layout is None or gray.shape != (layout.height, layout.width):
            return rows
        ppm = ref.frame_set.px_per_mm
        outer = outer_windows(gray, layout, ppm, params)
        for side in Side:
            part = outer[outer.side == side.value]
            for method in Method:
                near = part[part.from_edge_mm <= NEAR_MM[1]][ratio_column(method)]
                rows.append({"key": ref.key, "side": side.value, "case": "outer", "method": method.value,
                             "near": float(near.median()) if len(near) else np.nan})  # fmt: skip
        models = [CompressionModel(a, l) for a in SYNTH_AMPLITUDES for l in SYNTH_LENGTHS_MM]
        edges, tables = synth_frame(gray, layout, ppm, models, params)
        for model, table in zip(models, tables):
            for side, rect, edge in zip(Side, layout.rects, edges):
                part = table[table.side == side.value]
                if part.empty:
                    continue
                model = CompressionModel(model.amplitude, model.length_mm, edge_mm=edge)
                grid = np.linspace(edge, edge + SYNTH_SPAN_MM, 61)
                truth = model.scale(grid)
                for method in HORIZONTAL:
                    fit = fit_model(part, method, rect.height, rect.y0, use_tilt=False)
                    est = fit.scale(grid)
                    near = part[part.from_edge_mm <= NEAR_MM[1]]
                    rows.append({
                        "key": ref.key, "side": side.value, "case": f"A={model.amplitude},L={model.length_mm}",
                        "method": method.value, "true_amp": model.amplitude, "true_len_mm": model.length_mm,
                        "fit_amp": fit.amplitude, "fit_len_mm": fit.length_mm,
                        "near": float(near[ratio_column(method)].median()) if len(near) else np.nan,
                        "true_near": float(np.mean(model.scale(near.d_mm))) if len(near) else np.nan,
                        "rms": float(np.sqrt(np.mean((est - truth) ** 2))),
                        "excess_err_mm": float(np.trapezoid(1 / est - 1 / truth, grid)),
                    })  # fmt: skip
    except Exception:  # кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка синтетики", ref.key)
    return rows


@main.command()
@SET_OPTION
@click.option("--sides-csv", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True,
              help="sides.csv из ``select``: контроль — кадры, у которых обе стороны в поясе «нет».")  # fmt: skip
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Папка synth.csv.")
@click.option("--frames", "n_frames", type=int, default=40, show_default=True, help="Кадров контроля (случайно).")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def synth(sets: tuple, sides_csv: Path, out_dir: Path, n_frames: int, jobs: int) -> None:
    """Проверка мер: нулевая у внешнего края и восстановление известного синтетического сжатия кадров контроля."""
    sides = pd.read_csv(sides_csv)
    both = sides[sides.band == Band.NONE.value].groupby("key").side.nunique()
    keys = sorted(both[both == 2].index)
    rng = np.random.default_rng(0)
    keys = set(rng.choice(keys, min(n_frames, len(keys)), replace=False))
    refs = [ref for fs in parse_sets(sets) for ref in set_frames(fs) if ref.key in keys]
    log.info("кадров контроля: %d", len(refs))
    rows: list[dict] = []
    ctx = get_context("forkserver")
    params = MeasureParams()
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=init_worker) as pool:
        for i, part in enumerate(pool.map(synth_one, [(r, params) for r in refs]), 1):
            rows += part
            log.info("%d/%d", i, len(refs))
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "synth.csv", index=False)
    log.info("готово: %s, строк %d", out_dir / "synth.csv", len(table))


def letters_one(args: tuple[FrameRef, str, str]) -> list[dict]:
    """Сверка одной стороны кадра (для пула): слова tesseract с отношением ширины к ожидаемой и меры окон под ними.

    Args:
        args: ``(кадр, сторона L/R, пояс)``.

    Returns:
        Строки по словам: key, side, band, text, near_mm, far_mm, tess (ширина к ожидаемой), меры окон под словом.
    """
    ref, side_value, band = args
    side = Side(side_value)
    try:
        v2 = read_bgr(ref.v2_path)
        layout = frame_layout(frame_geometry(ref), v2.shape[:2], ref.frame_set.dpi)
        gray = cv2.cvtColor(read_bgr(ref.rectified_path), cv2.COLOR_BGR2GRAY)
        if layout is None or gray.shape != (layout.height, layout.width):
            return []
        ppm = ref.frame_set.px_per_mm
        params = MeasureParams()
        rect = layout.rects[0 if side is Side.LEFT else 1]
        margin = int(round(ppm))
        x_lo, x_hi = ((int(rect.x0) + margin, int(round(layout.spine_x))) if side is Side.LEFT
                      else (int(round(layout.spine_x)), int(rect.x1) - margin))  # fmt: skip
        ink = ink_mask(gray, ppm, params)
        win = page_windows(ink, x_lo, x_hi, layout.spine_x, side, ppm, params)
        if win.empty:
            return []
        edge = float(win.edge_mm.median())
        table = page_words(gray, x_lo, x_hi, layout.spine_x, side, edge, ppm)
        if table.empty:
            return []
        for method in HORIZONTAL:
            table[method.value] = paired_windows(table, win, ratio_column(method), 0.8 * 1.6 * ppm)
        table = table.rename(columns={"ratio": "tess"}).drop(columns=["side"])
        table.insert(0, "band", band)
        table.insert(0, "side", side.value)
        table.insert(0, "key", ref.key)
        return table.to_dict("records")
    except Exception:  # сторона с ошибкой не роняет прогон
        log.exception("%s %s: ошибка сверки по словам", ref.key, side_value)
        return []


@main.command()
@SET_OPTION
@click.option("--sides-csv", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True,
              help="sides.csv из ``select``.")  # fmt: skip
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Папка letters.csv.")
@click.option("--per-band", type=int, default=8, show_default=True, help="Сторон на пояс (случайно).")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def letters(sets: tuple, sides_csv: Path, out_dir: Path, per_band: int, jobs: int) -> None:
    """Сверка мер окна со словами tesseract: ширина слова к ожидаемой по буквам против мер окон под словом."""
    sides = pd.read_csv(sides_csv).dropna(subset=["band"])
    picks = pd.concat([part.sample(min(per_band, len(part)), random_state=0) for _, part in sides.groupby("band")])
    refs = {ref.key: ref for fs in parse_sets(sets) for ref in set_frames(fs)}
    tasks = [(refs[r.key], r.side, r.band) for r in picks.itertuples() if r.key in refs]
    log.info("сторон: %d", len(tasks))
    rows: list[dict] = []
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=init_worker) as pool:
        for i, part in enumerate(pool.map(letters_one, tasks), 1):
            rows += part
            log.info("%d/%d", i, len(tasks))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "letters.csv", index=False)
    log.info("готово: %s, слов %d", out_dir / "letters.csv", len(rows))


def read_keys(paths: tuple[Path, ...]) -> list[str]:
    """Ключи кадров из файлов множеств (строка — ключ ``<набор>/<кадр>``; ``#`` — комментарий).

    Args:
        paths: Файлы множеств.

    Returns:
        Ключи по порядку без повторов.
    """
    keys: list[str] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            key = line.split("#", 1)[0].strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def stretch_one(args: tuple[FrameRef, StretchConfig, Path]) -> dict:
    """Растяжение одного кадра (для пула): итог, меры, окна до и после, лист «было | стало».

    Args:
        args: ``(кадр, параметры, папка выхода)``.

    Returns:
        Строка сводки (``run.summary_row`` плюс мера у края на кадре только после Кунса).
    """
    ref, cfg, out_dir = args
    t0 = time.perf_counter()
    try:
        out = stretch_frame(ref, cfg)
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка растяжения", ref.key)
        return {"key": ref.key, "status": Status.ERROR.value, "message": repr(exc)}
    row = {"key": ref.key, "status": Status.OK.value} | summary_row(ref.key, out)
    # Где мерить: та же мера на кадре только после Кунса (строки ещё волнистые) — для сравнения с итогом поля.
    layout = out.result.layout
    first = spread_windows(cv2.cvtColor(out.result.first, cv2.COLOR_BGR2GRAY), layout.rects, layout.spine_x,
                           ref.frame_set.px_per_mm, cfg.measure)  # fmt: skip
    for side in Side:
        part = first[first.side == side.value] if len(first) else first
        if len(part):
            summ = side_summary(part)
            row[f"{side.value}_coons_near_acf"] = summ["near_acf"]
            row[f"{side.value}_coons_lines"] = summ["lines"]
    stem = f"{ref.frame_set.label}/{ref.job.key}"
    for label, table in (("before", out.before), ("after", out.after)):
        path = out_dir / "windows" / label / f"{stem}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.4g")
    for label, table in (("before", out.letters_before), ("after", out.letters_after)):
        path = out_dir / "letters" / label / f"{stem}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.4g")
    write_sheet(out_dir / "sheets" / f"{stem}.jpg", ref.key, out, ref.frame_set.px_per_mm)
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


@main.command("stretch-run")
@SET_OPTION
@click.option("--keys", "key_files", type=click.Path(path_type=Path, exists=True, dir_okay=False), multiple=True,
              required=True, help="Файл множества кадров (ключ <набор>/<кадр> в строке); повторяется.")  # fmt: skip
@click.option("--out-dir", type=click.Path(path_type=Path), required=True,
              help="Папка выхода: stretch.csv, sheets/, windows/{before,after}/.")  # fmt: skip
@click.option("--method", type=click.Choice([m.value for m in HORIZONTAL]), default=Method.ACF.value,
              show_default=True, help="Мера, по которой подгоняется модель сжатия.")  # fmt: skip
@click.option("--tilt/--no-tilt", default=False, show_default=True,
              help="Амплитуда сжатия меняется по высоте страницы (наклон).")  # fmt: skip
@click.option("--max-stretch", type=float, default=StretchLimits.max_stretch, show_default=True,
              help="Наибольшее растяжение 1/c.")  # fmt: skip
@click.option("--edge-safety-mm", type=float, default=StretchLimits.edge_safety_mm, show_default=True,
              help="Край текста после растяжения не ближе этого к линии корешка, мм.")  # fmt: skip
@click.option("--min-amplitude", type=float, default=StretchLimits.min_amplitude, show_default=True,
              help="Модель со сжатием у края текста меньше этого не применяется.")  # fmt: skip
@click.option("--min-excess-mm", type=float, default=StretchLimits.min_excess_mm, show_default=True,
              help="Модель с лишней длиной текста меньше этой не применяется, мм.")  # fmt: skip
@click.option("--far-mm", type=float, default=MeasureParams.far_mm, show_default=True,
              help="Опора строки — окна дальше этого от края текста у корешка, мм.")  # fmt: skip
@click.option("--position-bias", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None,
              help="position_bias.csv команды outer: поправка мер на место в строке (начала/концы строк).")  # fmt: skip
@click.option("--max-body-squeeze", type=float, default=StretchLimits.max_body_squeeze, show_default=True,
              help="Наибольшее равномерное сжатие тела страницы (доля) под лишнюю длину сверх поля у корешка.")  # fmt: skip
@click.option("--mode", type=click.Choice([m.value for m in StretchMode]), default=StretchMode.SHIFT.value,
              show_default=True,
              help="shift — ширина страницы прежняя (сдвиг края текста, сжатие тела); widen — страница шире со стороны "
                   "корешка на лишнюю длину, поле у корешка только растягивается.")  # fmt: skip
@click.option("--tilt-max", type=float, default=None,
              help="Предел модуля наклона амплитуды по высоте (по умолчанию 0.3; шаг shape: у 56 % сжатых сторон "
                   "перепад больше).")  # fmt: skip
@click.option("--spline-penalty", type=float, default=None,
              help="Амплитуда по высоте — P-сплайн с этим штрафом (шаг shape: 30 — не хуже линии на 99 % сжатых "
                   "сторон); только с --mode widen.")  # fmt: skip
@click.option("--fit-source", type=click.Choice([f.value for f in FitSource]), default=FitSource.WINDOWS.value,
              show_default=True,
              help="По чему подгоняется модель: windows — окна 5 мм; letters — ширина каждой буквы, спад от начала зоны "
                   "текста (только с --mode widen).")  # fmt: skip
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def stretch_run(sets: tuple, key_files: tuple[Path, ...], out_dir: Path, method: str, tilt: bool, max_stretch: float,
                edge_safety_mm: float, min_amplitude: float, min_excess_mm: float, far_mm: float,
                position_bias: Path | None, max_body_squeeze: float, mode: str, tilt_max: float | None,
                spline_penalty: float | None, fit_source: str, jobs: int) -> None:  # fmt: skip
    """Выпрямление с растяжением у корешка по множествам кадров: сводка, окна и буквы до/после, листы «было | стало»."""
    if spline_penalty is not None and StretchMode(mode) is not StretchMode.WIDEN:
        raise click.ClickException("--spline-penalty — только с --mode widen")
    if FitSource(fit_source) is FitSource.LETTERS and StretchMode(mode) is not StretchMode.WIDEN:
        raise click.ClickException("--fit-source letters — только с --mode widen")
    keys = read_keys(key_files)
    refs = {ref.key: ref for fs in parse_sets(sets) for ref in set_frames(fs)}
    missing = [k for k in keys if k not in refs]
    if missing:
        raise click.ClickException(f"нет в наборах: {missing[:5]}")
    cfg = StretchConfig(method=Method(method), use_tilt=tilt,
                        limits=StretchLimits(max_stretch, edge_safety_mm, min_amplitude, max_body_squeeze, min_excess_mm),
                        measure=MeasureParams(far_mm=far_mm),
                        bias=pd.read_csv(position_bias) if position_bias else None, mode=StretchMode(mode),
                        tilt_max=tilt_max, spline_penalty=spline_penalty, fit_source=FitSource(fit_source))  # fmt: skip
    log.info("кадров: %d, мера %s, подгонка по %s, режим %s, наклон %s (предел %s), сплайн %s, пределы %s, поправка на "
             "место в строке %s", len(keys), method, fit_source, mode, tilt, tilt_max, spline_penalty, cfg.limits,
             position_bias)  # fmt: skip
    rows = []
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=init_worker) as pool:
        for i, row in enumerate(pool.map(stretch_one, [(refs[k], cfg, out_dir) for k in keys]), 1):
            rows.append(row)
            log.info("%d/%d %s", i, len(keys), row["key"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "stretch.csv", index=False)
    log.info("готово: %s", out_dir / "stretch.csv")


# Пояса расстояния от края текста для профиля «до | после» в сводке, мм.
PROFILE_BINS = (0.0, 3.0, 6.0, 10.0, 15.0, 25.0, 60.0)
# Частей высоты страницы для остатка у края в сводке (пятые — видно и середину, и края).
HEIGHT_PARTS = 5


def side_rows(table: pd.DataFrame, compressed: set[str]) -> pd.DataFrame:
    """Строки сторон из ``stretch.csv``: мера у края до и после, план, принадлежность множеству.

    Args:
        table: Сводка ``stretch-run``.
        compressed: Ключи кадров множества со сжатием (остальные — контроль).

    Returns:
        Таблица: key, side, group (``compressed`` / ``control``), applied, before, after, excess_mm, squeeze, edge_mm.
    """
    parts = []
    for side in Side:
        p = side.value
        # У режима widen нет сжатия тела — столбец заполняется NaN.
        part = table.reindex(columns=["key", f"{p}_applied", f"{p}_before_near_acf", f"{p}_after_near_acf",
                                      f"{p}_excess_mm", f"{p}_squeeze", f"{p}_edge_mm"]).copy()  # fmt: skip
        part.columns = ["key", "applied", "before", "after", "excess_mm", "squeeze", "edge_mm"]
        part.insert(1, "side", p)
        parts.append(part)
    rows = pd.concat(parts, ignore_index=True)
    rows["group"] = np.where(rows.key.isin(compressed), "compressed", "control")
    return rows


@main.command()
@click.option("--run", "runs", type=(str, click.Path(path_type=Path, exists=True, file_okay=False)), multiple=True,
              required=True, help="ПОДПИСЬ ПАПКА — выход stretch-run; повторяется.")  # fmt: skip
@click.option("--compressed", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True,
              help="Множество кадров со сжатием (остальные кадры прогона — контроль).")  # fmt: skip
def summary(runs: tuple[tuple[str, Path], ...], compressed: Path) -> None:
    """Сводка прогонов растяжения: меры у края до/после по сторонам, прямизна строк и сторон, профиль по поясам."""
    keys = set(read_keys((compressed,)))
    pd.set_option("display.width", 250)
    for label, folder in runs:
        table = pd.read_csv(folder / "stretch.csv")
        table = table[table.status == Status.OK.value]
        sides = side_rows(table, keys)
        sides["moved"] = (sides.after - sides.before).abs() > 0.02
        click.echo(f"=== {label}: кадров {len(table)}")
        agg = sides.groupby(["group", "applied"]).agg(
            sides=("key", "size"), before=("before", "median"), after=("after", "median"),
            gap_before=("before", lambda v: float((1 - v).abs().median())),
            gap_after=("after", lambda v: float((1 - v).abs().median())),
            excess_mm=("excess_mm", "median"), squeeze=("squeeze", "median"), edge_mm=("edge_mm", "median"))  # fmt: skip
        click.echo(agg.round(3).to_string())
        lines = table[["lines_before_line_p95_mm", "lines_after_line_p95_mm", "lines_before_side_p95_mm",
                       "lines_after_side_p95_mm"]]  # fmt: skip
        click.echo("прямизна, медиана по кадрам (мм): " + ", ".join(f"{c} {lines[c].median():.3f}" for c in lines))
        worse_lines = int((lines.lines_after_line_p95_mm > lines.lines_before_line_p95_mm + 0.05).sum())
        worse_sides = int((lines.lines_after_side_p95_mm > lines.lines_before_side_p95_mm + 0.05).sum())
        click.echo(f"хуже на 0.05 мм и больше: строки {worse_lines}, стороны {worse_sides} кадров из {len(table)}")
        # Профиль по поясам от края текста — по окнам сторон, где растяжение применено.
        applied = sides[(sides.applied == 1) & (sides.group == "compressed")]
        profile = []
        for row in applied.itertuples():
            stem = row.key
            for stage in ("before", "after"):
                path = folder / "windows" / stage / f"{stem}.csv.gz"
                if not path.exists():
                    continue
                win = pd.read_csv(path)
                win = win[win.side == row.side]
                # Доля высоты — по крайним строкам стороны (раскладка страницы в сводку не пишется).
                span = max(float(win.y.max() - win.y.min()), 1.0)
                win = win.assign(stage=stage, bin=pd.cut(win.from_edge_mm, PROFILE_BINS),
                                 part=np.minimum(((win.y - win.y.min()) / span * HEIGHT_PARTS).astype(int),
                                                 HEIGHT_PARTS - 1), key=row.key, side=row.side)  # fmt: skip
                profile.append(win[["key", "side", "stage", "bin", "part", "from_edge_mm", "acf_ratio", "stem_ratio",
                                    "comp_ratio"]])  # fmt: skip
        if profile:
            windows = pd.concat(profile)
            prof = windows.drop(columns=["key", "side", "part", "from_edge_mm"]).groupby(
                ["bin", "stage"], observed=True
            )
            click.echo("профиль по поясам от края текста (стороны с растяжением, медианы окон):")
            click.echo(prof.median().unstack("stage").round(3).to_string())
            # Остаток у края по частям высоты: медиана стороны в части, затем медиана |1 − мера| по сторонам.
            near = windows[windows.from_edge_mm <= NEAR_MM[1]]
            per_side = near.groupby(["key", "side", "stage", "part"]).acf_ratio.median().reset_index()
            per_side["gap"] = (1.0 - per_side.acf_ratio).abs()
            parts = per_side.groupby(["stage", "part"]).gap.agg(["median", lambda v: v.quantile(0.9)]).unstack("part")
            click.echo(f"|1 − acf| у края по {HEIGHT_PARTS} частям высоты (0 — верх; медиана и p90 по сторонам):")
            click.echo(parts.round(3).to_string())


# Тренд отношения у внешнего края — прямая по поясам от TREND_MM[0] до TREND_MM[1] мм; ближе EDGE_EFFECT_MM к краю
# отклонение от тренда — место в строке (прописные в начале, знаки препинания в конце). Сам тренд (у внешних краёв
# весь пояс 2–24 мм уже опоры строки на 2–3 %: страница к краям чуть мельче) к корешку не переносится.
TREND_MM = (5.0, 20.0)
EDGE_EFFECT_MM = 5.0


def edge_effect(bias: pd.DataFrame) -> pd.DataFrame:
    """Доля места в строке в отношениях мер у внешнего края: отклонение от прямой тренда по дальним поясам.

    Args:
        bias: Медианы по поясам (position, bin, ``<мера>_ratio``).

    Returns:
        Та же таблица плюс ``<мера>_edge``: отношение к тренду в поясах ближе ``EDGE_EFFECT_MM``, дальше — 1.
    """
    out = []
    for _, part in bias.groupby("position"):
        part = part.sort_values("bin").copy()
        trend_rows = part[(part.bin >= TREND_MM[0]) & (part.bin <= TREND_MM[1])]
        for method in Method:
            column = ratio_column(method)
            coef = np.polyfit(trend_rows.bin, trend_rows[column], 1)
            effect = part[column] / np.polyval(coef, part.bin)
            part[f"{method.value}_edge"] = np.where(part.bin < EDGE_EFFECT_MM, effect, 1.0)
        out.append(part)
    return pd.concat(out, ignore_index=True)


# Пояса расстояния от края текста для поправки на место в строке, мм (шаг окон у края — 1 мм).
BIAS_STEP_MM = 1.0
BIAS_REACH_MM = 25.0


def outer_frame(args: tuple[FrameRef, MeasureParams]) -> pd.DataFrame:
    """Окна у внешнего края текста одного кадра (для пула) — без сжатия у корешка, только место в строке.

    У левой страницы внешний край — начала строк (прописные, отступы абзацев), у правой — концы строк (знаки
    препинания, дефисы, короткие концы абзацев).

    Args:
        args: ``(кадр, параметры меры)``.

    Returns:
        Окна ближе ``BIAS_REACH_MM`` к краю: position (``start``/``end``), from_edge_mm и отношения мер.
    """
    ref, params = args
    try:
        if not ref.rectified_path.exists() or not ref.v2_path.exists():
            return pd.DataFrame()
        v2 = read_bgr(ref.v2_path)
        layout = frame_layout(frame_geometry(ref), v2.shape[:2], ref.frame_set.dpi)
        gray = cv2.cvtColor(read_bgr(ref.rectified_path), cv2.COLOR_BGR2GRAY)
        if layout is None or gray.shape != (layout.height, layout.width):
            return pd.DataFrame()
        win = outer_windows(gray, layout, ref.frame_set.px_per_mm, params)
    except Exception:  # кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка окон у внешнего края", ref.key)
        return pd.DataFrame()
    if win.empty:
        return win
    win = win[win.from_edge_mm <= BIAS_REACH_MM]
    position = np.where(win.side == Side.LEFT.value, LinePosition.START.value, LinePosition.END.value)
    cols = ["from_edge_mm"] + [ratio_column(m) for m in Method]
    out = win[cols].astype(np.float32)
    out.insert(0, "position", position)
    return out


@main.command()
@SET_OPTION
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Папка position_bias.csv.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def outer(sets: tuple, out_dir: Path, jobs: int) -> None:
    """Поправка на место в строке: медианы отношений мер у внешних краёв текста (начала и концы строк) по поясам."""
    refs = [ref for fs in parse_sets(sets) for ref in set_frames(fs)]
    log.info("кадров: %d", len(refs))
    parts = []
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=init_worker) as pool:
        for i, part in enumerate(pool.map(outer_frame, [(r, MeasureParams()) for r in refs]), 1):
            if len(part):
                parts.append(part)
            if i % 50 == 0 or i == len(refs):
                log.info("%d/%d", i, len(refs))
    table = pd.concat(parts, ignore_index=True)
    table["bin"] = (table.from_edge_mm // BIAS_STEP_MM) * BIAS_STEP_MM
    bias = table.groupby(["position", "bin"]).median(numeric_only=True).drop(columns=["from_edge_mm"])
    bias["n"] = table.groupby(["position", "bin"]).size()
    bias = edge_effect(bias.reset_index())
    out_dir.mkdir(parents=True, exist_ok=True)
    bias.to_csv(out_dir / "position_bias.csv", index=False, float_format="%.4f")
    log.info("готово: %s, окон %d", out_dir / "position_bias.csv", len(table))


@main.command()
@click.option("--scan-root", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True,
              help="Выход scan (scan.csv, windows/).")  # fmt: skip
@click.option("--position-bias", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None,
              help="position_bias.csv команды outer (поправка на место в строке).")  # fmt: skip
@click.option("--keys", "key_files", type=click.Path(path_type=Path, exists=True, dir_okay=False), multiple=True,
              help="Множества кадров для листа кривых (sets/*.txt).")  # fmt: skip
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Папка таблиц и листов.")
@click.option("--jobs", type=int, default=16, show_default=True,
              help="Воркеров пула (только CPU, окна уже на SSD — упор в счёт).")  # fmt: skip
def shape(scan_root: Path, position_bias: Path | None, key_files: tuple[Path, ...], out_dir: Path, jobs: int) -> None:
    """Форма зависимости амплитуды сжатия от высоты: семейства B(y), перекрёстная проверка по строкам, листы."""
    bias = pd.read_csv(position_bias) if position_bias else None
    tasks = side_tasks(scan_root, bias)
    log.info("сторон: %d, воркеров: %d", len(tasks), jobs)
    rows = []
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=init_worker) as pool:
        for i, part in enumerate(pool.map(shape_side, tasks, chunksize=4), 1):
            rows += part
            if i % 200 == 0 or i == len(tasks):
                log.info("%d/%d", i, len(tasks))
    table = pd.DataFrame(rows)
    bands = table[table.candidate == "bands"].dropna(axis=1, how="all")
    fits = table[table.candidate != "bands"].dropna(axis=1, how="all")
    out_dir.mkdir(parents=True, exist_ok=True)
    fits.to_csv(out_dir / "shape_fits.csv", index=False, float_format="%.5g")
    bands.to_csv(out_dir / "shape_bands.csv", index=False, float_format="%.5g")
    summary_table, parts = summarize(fits)
    summary_table.to_csv(out_dir / "shape_summary.csv", index=False, float_format="%.5g")
    parts.to_csv(out_dir / "shape_parts.csv", float_format="%.5g")
    band_shape_table(bands).to_csv(out_dir / "shape_band_medians.csv", float_format="%.4g")
    # Штраф P-сплайна — с наименьшей медианной ошибкой по всем сторонам.
    splines = summary_table[(summary_table.belt == "all") & summary_table.candidate.str.startswith("spline_")]
    best = splines.sort_values("cv_error_median").iloc[0].candidate
    penalty = float(best.split("_", 1)[1])
    log.info("штраф P-сплайна по паку: %g", penalty)
    # Листы: стороны множеств (обе стороны кадра) и по 16 случайных сторон каждого пояса сжатия.
    keys = set(read_keys(key_files))
    chosen = bands[bands.key.isin(keys)][["key", "side"]].itertuples(index=False, name=None)
    write_curves(out_dir / "sheets" / "sets.jpg", curves_sheet(fits, bands, list(chosen), penalty))
    for belt, group in bands.groupby("belt"):
        pick = group.sample(min(16, len(group)), random_state=0)[["key", "side"]].itertuples(index=False, name=None)
        write_curves(out_dir / "sheets" / f"belt_{belt}.jpg", curves_sheet(fits, bands, list(pick), penalty))
    view = summary_table[summary_table.belt.isin(["all", "compressed_1-3"])]
    log.info("сводка:\n%s", view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log.info("готово: %s", out_dir)


def edge_profile_frame(args: tuple[FrameRef, Path, MeasureParams]) -> dict:
    """Буквы и узкие окна одного кадра (для пула): профиль сжатия у края текста с разрешением в букву.

    Args:
        args: ``(кадр, корень выхода, параметры узких окон)``.

    Returns:
        Строка сводки: key, status, верх и высота страниц, число букв.
    """
    ref, out_root, fine = args
    row: dict = {"key": ref.key, "status": Status.OK.value}
    try:
        if not ref.rectified_path.exists() or not ref.v2_path.exists():
            return row | {"status": Status.MISSING.value}
        v2 = read_bgr(ref.v2_path)
        layout = frame_layout(frame_geometry(ref), v2.shape[:2], ref.frame_set.dpi)
        if layout is None:
            return row | {"status": Status.NO_LAYOUT.value}
        gray = cv2.cvtColor(read_bgr(ref.rectified_path), cv2.COLOR_BGR2GRAY)
        if gray.shape != (layout.height, layout.width):
            return row | {"status": Status.MISMATCH.value}
        ppm = ref.frame_set.px_per_mm
        letters = spread_letters(gray, layout.rects, layout.spine_x, ppm)
        windows = spread_windows(gray, layout.rects, layout.spine_x, ppm, fine)
        for name, table in (("letters", letters), ("fine", windows)):
            path = out_root / name / ref.frame_set.label / f"{ref.job.key}.csv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False, float_format="%.4g")
        row["letters"] = len(letters)
        for side, rect in zip(Side, layout.rects):
            row[f"{side.value}_top"] = round(rect.y0, 1)
            row[f"{side.value}_height"] = round(rect.height, 1)
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка", ref.key)
        row.update(status=Status.ERROR.value, message=repr(exc))
    return row


@main.command("edge-profile")
@SET_OPTION
@click.option("--out-root", type=click.Path(path_type=Path), required=True, help="Корень выхода (letters/, fine/).")
@click.option("--fine-window-mm", type=float, default=2.0, show_default=True,
              help="Ширина узкого окна вдоль строки, мм (2 мм = 34 px, 4–5 штрихов строчных ХиП 1991).")  # fmt: skip
@click.option("--fine-step-mm", type=float, default=0.5, show_default=True, help="Шаг узких окон у корешка, мм.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def edge_profile(sets: tuple, out_root: Path, fine_window_mm: float, fine_step_mm: float, jobs: int) -> None:
    """Профиль сжатия у края текста по буквам и узким окнам по всем кадрам наборов: letters/, fine/, frames.csv."""
    fine = MeasureParams(window_mm=fine_window_mm, near_step_mm=fine_step_mm)
    refs = [ref for fs in parse_sets(sets) for ref in set_frames(fs)]
    log.info("кадров: %d, воркеров: %d, узкие окна %s мм с шагом %s мм", len(refs), jobs, fine_window_mm, fine_step_mm)
    rows = []
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=init_worker) as pool:
        for i, row in enumerate(pool.map(edge_profile_frame, [(r, out_root, fine) for r in refs]), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(refs):
                log.info("%d/%d", i, len(refs))
    out_root.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "frames.csv", index=False)
    log.info("готово: %s; %s", out_root / "frames.csv", table.status.value_counts().to_dict())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
