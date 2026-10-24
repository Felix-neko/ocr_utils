"""Прогон команды ``shape`` по окнам ``scan``: подгонка семейств ``B(y)`` и перекрёстная проверка по каждой стороне, сводка по поясам сжатия и частям высоты, лист кривых амплитуды по высоте."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from research.gutter_compression.measure import Method, Side
from research.gutter_compression.profile import MIN_LINES, correct_position, side_summary
from research.gutter_compression.shape import (
    EVAL_PARTS,
    HEIGHT_BANDS,
    MIN_WINDOWS,
    SPLINE_LAMBDAS,
    HeightShape,
    ShapeFit,
    band_amplitudes,
    curve_panel,
    cv_errors,
    edge_excursion,
    fit_shape,
    legend_strip,
    side_data,
)

# Пояса сжатия стороны по мере у края текста (как у ``select``): нижняя граница, верхняя граница, подпись.
BELTS = ((0.0, 0.70, "1_strong"), (0.70, 0.80, "2_clear"), (0.80, 0.88, "3_mild"), (0.88, 0.95, "4_slight"),
         (0.95, 9.0, "5_none"))  # fmt: skip


def belt_of(near: float) -> str:
    """Пояс сжатия по мере у края текста (``near_acf``)."""
    for lo, hi, label in BELTS:
        if lo <= near < hi:
            return label
    return "unknown"


def candidates() -> list[tuple[HeightShape, float]]:
    """Семейства с параметром штрафа: у P-сплайна — по каждому штрафу сетки, у прочих — 0."""
    return [(s, 0.0) for s in HeightShape if s is not HeightShape.SPLINE] + [
        (HeightShape.SPLINE, lam) for lam in SPLINE_LAMBDAS
    ]


def candidate_label(shape: HeightShape, penalty: float) -> str:
    """Подпись кандидата: семейство, у P-сплайна — со штрафом."""
    return f"{shape.value}_{penalty:g}" if shape is HeightShape.SPLINE else shape.value


def shape_side(args: tuple[Path, str, float, float, str, pd.DataFrame | None]) -> list[dict]:
    """Подгонка и перекрёстная проверка семейств на одной стороне (для пула).

    Args:
        args: ``(таблица окон кадра, сторона L/R, верх страницы px, высота страницы px, ключ, поправка на место в
            строке или None)``.

    Returns:
        Строки: по одной на кандидата (ошибка, ошибки по частям высоты, параметры полной подгонки) и одна строка
        ``bands`` с амплитудой по поясам высоты; пусто — окон мало.
    """
    path, side, top, height, key, bias = args
    try:
        table = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return []
    part = correct_position(table[table.side == side], bias)
    if part.empty or part.line.nunique() < MIN_LINES:
        return []
    data = side_data(part, Method.ACF, top, height)
    if len(data.r) < MIN_WINDOWS:
        return []
    near = side_summary(part)["near_acf"]
    base = {"key": key, "side": side, "near_acf": near, "belt": belt_of(near), "lines": int(part.line.nunique()),
            "windows": len(data.r)}  # fmt: skip
    rows = []
    const_length = None
    for shape, penalty in candidates():
        error, by_part = cv_errors(data, shape, penalty)
        fit = fit_shape(data.x, data.y, data.r, shape, penalty)
        if shape is HeightShape.CONST:
            const_length = fit.length_mm
        a = fit.amplitude(np.array([0.0, 0.5, 1.0]))
        rows.append(base | {"candidate": candidate_label(shape, penalty), "shape": shape.value, "penalty": penalty,
                            "cv_error": error, **{f"cv_part{i}": v for i, v in enumerate(by_part)},
                            "length_mm": fit.length_mm, "residual": fit.residual, "a_top": a[0], "a_mid": a[1],
                            "a_bottom": a[2], "edge_excursion": edge_excursion(fit),
                            "coef": json.dumps([round(c, 5) for c in fit.coef])})  # fmt: skip
    bands = band_amplitudes(data, const_length)
    rows.append(base | {"candidate": "bands", "length_mm": const_length,
                        **{f"band{i}": v for i, v in enumerate(bands)}})  # fmt: skip
    return rows


def side_tasks(scan_root: Path, bias: pd.DataFrame | None) -> list[tuple]:
    """Задачи пула: стороны всех кадров ``scan`` с раскладкой.

    Args:
        scan_root: Выход ``scan`` (``scan.csv``, ``windows/``).
        bias: Поправка на место в строке.

    Returns:
        Аргументы :func:`shape_side`.
    """
    scan = pd.read_csv(scan_root / "scan.csv")
    scan = scan[scan.status == "ok"]
    tasks = []
    for row in scan.itertuples(index=False):
        path = scan_root / "windows" / f"{row.key}.csv.gz"
        if not path.exists():
            continue
        for side in Side:
            top, height = getattr(row, f"{side.value}_top"), getattr(row, f"{side.value}_height")
            if np.isfinite(top) and np.isfinite(height):
                tasks.append((path, side.value, float(top), float(height), row.key, bias))
    return tasks


def summarize(fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сводка кандидатов по поясам сжатия: ошибка на отложенных строках, выигрыш у константы и линии.

    Args:
        fits: Строки кандидатов всех сторон (без строк ``bands``).

    Returns:
        (сводка по поясам и кандидатам; ошибки по частям высоты у сжатых сторон).
    """
    wide = fits.pivot_table(index=["key", "side"], columns="candidate", values="cv_error")
    meta = fits.drop_duplicates(["key", "side"]).set_index(["key", "side"])[["belt"]]
    wide = wide.join(meta)
    rows = []
    labels = [c for c in wide.columns if c != "belt"]
    for belt, group in [("all", wide), ("compressed_1-3", wide[wide.belt <= "3_mild"])] + list(wide.groupby("belt")):
        for label in labels:
            gain_const = group["const"] - group[label]
            gain_linear = group["linear"] - group[label]
            rows.append({"belt": belt, "candidate": label, "sides": int(group[label].notna().sum()),
                         "cv_error_median": group[label].median(), "cv_error_mean": group[label].mean(),
                         "gain_vs_const_median": gain_const.median(), "wins_vs_const": (gain_const > 0).mean(),
                         "gain_vs_linear_median": gain_linear.median(),
                         "wins_vs_linear": (gain_linear > 0).mean()})  # fmt: skip
    summary = pd.DataFrame(rows)
    compressed = fits[fits.belt <= "3_mild"]
    parts = compressed.groupby("candidate")[[f"cv_part{i}" for i in range(EVAL_PARTS)]].mean()
    return summary, parts


def band_shape_table(bands: pd.DataFrame) -> pd.DataFrame:
    """Амплитуда по поясам высоты без допущения о форме: медианы по поясам сжатия.

    Args:
        bands: Строки ``bands`` всех сторон.

    Returns:
        Медиана амплитуды каждого пояса высоты и отклонения от средней стороны, по поясам сжатия.
    """
    columns = [f"band{i}" for i in range(HEIGHT_BANDS)]
    dev = bands[columns].sub(bands[columns].mean(axis=1), axis=0).add_prefix("dev_")
    joined = pd.concat([bands[["belt"]], bands[columns], dev], axis=1)
    out = joined.groupby("belt").median(numeric_only=True)
    out["sides"] = joined.groupby("belt").size()
    return out


def fit_from_row(row: pd.Series) -> ShapeFit:
    """Подгонка из строки таблицы кандидатов (для листа)."""
    return ShapeFit(HeightShape(row["shape"]), tuple(json.loads(row["coef"])), float(row["length_mm"]),
                    float(row["penalty"]))  # fmt: skip


def curves_sheet(fits: pd.DataFrame, bands: pd.DataFrame, sides: list[tuple[str, str]], penalty: float,
                 columns: int = 4) -> np.ndarray:  # fmt: skip
    """Лист кривых амплитуды по высоте для выбранных сторон: точки поясов и кривые всех семейств.

    Args:
        fits: Строки кандидатов.
        bands: Строки ``bands``.
        sides: Стороны ``(ключ, L/R)`` в порядке панелей.
        penalty: Штраф P-сплайна, выбранный по паку (на листе — только он).
        columns: Панелей в ряду.

    Returns:
        BGR-лист с легендой внизу.
    """
    panels = []
    for key, side in sides:
        part = fits[(fits.key == key) & (fits.side == side)]
        band_row = bands[(bands.key == key) & (bands.side == side)]
        if part.empty or band_row.empty:
            continue
        chosen = part[(part["shape"] != HeightShape.SPLINE.value) | (part.penalty == penalty)]
        values = band_row.iloc[0][[f"band{i}" for i in range(HEIGHT_BANDS)]].to_numpy(np.float64)
        title = f"{key.split('/')[1][:2]}/{key.rsplit('/', 1)[-1]} {side} acf {band_row.iloc[0].near_acf:.2f}"
        panels.append(curve_panel(title, values, [fit_from_row(r) for _, r in chosen.iterrows()]))
    if not panels:
        return np.full((40, 400, 3), 255, np.uint8)
    blank = np.full_like(panels[0], 255)
    panels += [blank] * (-len(panels) % columns)
    grid = np.vstack([np.hstack(panels[i : i + columns]) for i in range(0, len(panels), columns)])
    return np.vstack([grid, legend_strip(grid.shape[1])])


def write_sheet(path: Path, sheet: np.ndarray) -> None:
    """Запись листа JPEG (папка создаётся)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
