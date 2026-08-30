"""Отчёты: консольная таблица, CSV, markdown, папка симлинков, лист врезок.

ЗАЧЕМ ЛИСТ ВРЕЗОК. Проверять такой детектор, открывая кадры по одному, дорого и долго:
на паке речь о тысячах разворотов. Врезка шириной в пару сантиметров вокруг сгиба
показывает ровно то, по чему принимается решение — режет строки сгиб или нет, — и
полсотни таких врезок на одном листе просматриваются за минуту.
"""

import csv
import math
from pathlib import Path
from urllib.parse import quote

import numpy as np

import cv2

from ocr_utils.gutter_loss_detection.analysis import FileResult
from ocr_utils.gutter_loss_detection.geometry import LEFT, analyze_spread, read_work_gray

CODES = ("таблица", "текст", "ок")

# Размер одной врезки на контактном листе, в пикселях исходного кадра.
TILE_W, TILE_H = 420, 620


def _num(value: float, digits: int = 2) -> str:
    """Число или прочерк для NaN."""
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _side_cell(result: FileResult, side: str) -> str:
    """Поле полосы и пометка таблицы."""
    for s in result.sides:
        if s.side == side:
            mark = " (табл.)" if s.tabular else ""
            return _num(s.margin) + mark
    return "—"


def console_table(results: list[FileResult], limit: int) -> str:
    """Собирает текстовую таблицу для консоли.

    Args:
        results: Результаты, уже отсортированные.
        limit: Сколько строк показать.

    Returns:
        Готовый текст таблицы.
    """
    head = f"{'#':>3}  {'балл':>5}  {'вердикт':<8}  {'полосы':<6}  {'поле L':>9}  {'поле R':>9}  файл"
    lines = [head, "-" * len(head)]
    for i, r in enumerate(results[:limit], 1):
        note = r.error or r.problem
        name = r.path.name if not note else f"{r.path.name}  [{note}]"
        lines.append(
            f"{i:>3}  {_num(r.score):>5}  {r.code:<8}  {r.bitten_sides:<6}  "
            f"{_side_cell(r, LEFT):>9}  {_side_cell(r, 'R'):>9}  {name}"
        )
    return "\n".join(lines)


def write_csv(path: Path, results: list[FileResult]) -> None:
    """Пишет полный отчёт в CSV.

    Args:
        path: Куда писать.
        results: Результаты прогона.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "файл",
                "балл",
                "вердикт",
                "полосы",
                "поле_L",
                "поле_R",
                "таблица_L",
                "таблица_R",
                "шаг_строк",
                "строк",
                "наклон_сгиба",
                "замечание",
            ]
        )
        for r in results:
            by = {s.side: s for s in r.sides}
            writer.writerow(
                [
                    str(r.path),
                    _num(r.score),
                    r.code,
                    r.bitten_sides,
                    _num(by[LEFT].margin) if LEFT in by else "",
                    _num(by["R"].margin) if "R" in by else "",
                    int(by[LEFT].tabular) if LEFT in by else "",
                    int(by["R"].tabular) if "R" in by else "",
                    _num(r.pitch, 1),
                    r.lines,
                    _num(r.tilt, 1),
                    r.error or r.problem,
                ]
            )


def markdown_report(results: list[FileResult], limit: int, threshold: float, base: Path | None = None) -> str:
    """Собирает markdown-отчёт со ссылками на кадры.

    Args:
        results: Результаты, уже отсортированные.
        limit: Сколько строк показать.
        threshold: Порог по баллу.
        base: Корень прогона — от него строятся относительные подписи.

    Returns:
        Текст markdown.
    """
    hit = [r for r in results if np.isfinite(r.score) and r.score >= threshold]
    tables = [r for r in hit if r.code == "таблица"]
    text = [r for r in hit if r.code == "текст"]
    out = [
        "# Текст, уходящий под корешок",
        "",
        f"Кадров измерено: {sum(1 for r in results if np.isfinite(r.score))} из {len(results)}. "
        f"Порог по баллу: {threshold:g}.",
        "",
        f"- **Пересканировать обязательно** (у корешка таблица): {len(tables)}",
        f"- **Можно восстановить по контексту** (связный текст): {len(text)}",
        "",
        "| # | балл | вердикт | полосы | поле L | поле R | кадр |",
        "|--:|-----:|---------|--------|-------:|-------:|------|",
    ]
    for i, r in enumerate(results[:limit], 1):
        label = r.path.relative_to(base) if base and base in r.path.parents else r.path.name
        link = f"[{label}](file://{quote(str(r.path))})"
        out.append(
            f"| {i} | {_num(r.score)} | {r.code} | {r.bitten_sides} | "
            f"{_side_cell(r, LEFT)} | {_side_cell(r, 'R')} | {link} |"
        )
    return "\n".join(out) + "\n"


def write_link_dir(root: Path, results: list[FileResult]) -> tuple[Path, int]:
    """Раскладывает кадры симлинками, пронумерованными по рейтингу.

    Args:
        root: Куда складывать; создаётся, если нет.
        results: Отобранные кадры в порядке отчёта.

    Returns:
        Пара (путь к папке, сколько ссылок создано).
    """
    root.mkdir(parents=True, exist_ok=True)
    made = 0
    width = max(2, len(str(len(results))))
    for i, r in enumerate(results, 1):
        score = "—" if not math.isfinite(r.score) else f"{r.score:.2f}"
        name = f"{i:0{width}d}_{score}_{r.code}_{r.path.name}"
        link = root / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(r.path)
        made += 1
    return root, made


def contact_sheet(path: Path, results: list[FileResult], columns: int = 6) -> Path:
    """Собирает лист врезок вокруг сгиба для быстрой глазной проверки.

    Args:
        path: Куда сохранить PNG.
        results: Кадры в порядке отчёта.
        columns: Сколько врезок в ряду.

    Returns:
        Путь к сохранённому файлу.
    """
    tiles = []
    for r in results:
        image = cv2.imread(str(r.path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gray, scale = read_work_gray(r.path)
        fold = int(analyze_spread(gray, scale).fold_at_middle * scale) or width // 2
        y0 = max(0, height // 2 - TILE_H // 2)
        x0 = max(0, fold - TILE_W // 2)
        tile = image[y0 : y0 + TILE_H, x0 : x0 + TILE_W].copy()
        if tile.size == 0:
            continue
        tile = cv2.resize(tile, (TILE_W // 2, TILE_H // 2), interpolation=cv2.INTER_AREA)
        cv2.line(tile, (TILE_W // 4, 0), (TILE_W // 4, 8), (0, 0, 255), 2)
        tile = cv2.copyMakeBorder(tile, 24, 4, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        score = "—" if not math.isfinite(r.score) else f"{r.score:.2f}"
        cv2.putText(
            tile, f"{r.path.name[:16]} {score} {r.code}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 180), 1
        )
        tiles.append(tile)
    if not tiles:
        raise ValueError("нечего показывать: ни один кадр не прочитался")
    rows = []
    for i in range(0, len(tiles), columns):
        row = tiles[i : i + columns]
        while len(row) < columns:
            row.append(np.full_like(tiles[0], 255))
        rows.append(np.hstack(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    return path
