"""Отчёты по прогону: каталог симлинков, CSV, markdown, контактный лист.

Главный артефакт — КАТАЛОГ СИМЛИНКОВ: находок ожидаются сотни, и решать по ним всё равно
придётся глазами. CSV нужен для калибровки порогов (там все полосы и все метрики), markdown —
чтобы видеть, где детекторы разошлись и кому верить.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

import cv2
import numpy as np

from ocr_utils.scan_markup.curved_lines import labels as labelling
from ocr_utils.scan_markup.curved_lines.analysis import PageResult, flagged_counts
from ocr_utils.scan_markup.curved_lines.detectors import DETECTORS
from ocr_utils.scan_markup.curved_lines.detectors.base import Measure
from ocr_utils.scan_markup.curved_lines.flags import COMBO, Thresholds
from ocr_utils.page_layout.orientation.pdf_pages import PdfPage, lookup
from ocr_utils.scan_cropping.image_io import IMAGE_EXTS

logger = logging.getLogger(__name__)

# Полосы, которые отметил ХОТЬ ОДИН детектор, — вместе с отвергнутыми сводом. По разности с
# combo/ видно, что свод отсеял.
CANDIDATES_DIR = "кандидаты"
# Размеченные полосы с вердиктом в имени: ok / missed / false.
LABELS_DIR = "разметка"

SILENT_MARK = "молчит"
FLAG_MARK = "да"


class LinkDirError(Exception):
    """В каталоге симлинков нашлось что-то, кроме симлинков."""


def _slug(text: str) -> str:
    return re.sub(r"[\s/\\]+", "_", text).strip("_")


def link_name(
    result: PageResult, pdf: PdfPage | None, score: float, ordinal: int | None, suffix: str, extra: str = ""
) -> str:
    """``год_выпуск_имя_pNNN_sX.XX[extra].ext``: год и выпуск впереди собирают находки одного
    номера рядом, score — чтобы сортировка по имени давала «самые кривые сверху» внутри
    номера видно не будет, зато в имени сразу видно, насколько уверенно."""
    parts = Path(result.rel_path).parts
    origin = "_".join(_slug(part) for part in parts[:-1])
    stem = _slug(Path(result.rel_path).stem)
    number = pdf.page_number if pdf is not None else ordinal
    page = f"p{number:03d}" if number is not None else "pXXX"
    pieces = [piece for piece in (origin, stem, page, f"s{score:.2f}") if piece]
    return "_".join(pieces) + extra + suffix


def _target(result: PageResult, link_root: Path | None) -> Path:
    """Куда целить симлинк: на оригинал под ``--link-root`` (по основе имени, расширение
    может отличаться), иначе на разобранный файл."""
    if link_root is None:
        return result.path.resolve()
    direct = link_root / result.rel_path
    if direct.exists():
        return direct.resolve()
    relative = Path(result.rel_path)
    siblings = sorted(
        path for path in (link_root / relative.parent).glob(f"{relative.stem}.*") if path.suffix.lower() in IMAGE_EXTS
    )
    return siblings[0].resolve() if siblings else result.path.resolve()


def _clear(directory: Path) -> None:
    if directory.exists():
        for entry in directory.iterdir():
            if not entry.is_symlink():
                raise LinkDirError(f"в {directory} лежит не симлинк ({entry.name}) — каталог не тронут")
        for entry in directory.iterdir():
            entry.unlink()
    directory.mkdir(parents=True, exist_ok=True)


def _verdict(result: PageResult) -> str:
    """Для размеченной полосы: ok / missed / false по сводному флагу."""
    curved = result.label == labelling.CURVED
    if curved == result.flagged:
        return "ok"
    return "missed" if curved else "false"


def write_link_dir(
    root: Path,
    results: Sequence[PageResult],
    detector_names: Sequence[str],
    pdf_pages: dict[str, PdfPage],
    link_root: Path | None = None,
) -> dict[str, int]:
    """Симлинки: подкаталог на детектор, combo, кандидаты, разметка. Повторный прогон
    чистит ТОЛЬКО свои симлинки и отказывается трогать каталог с обычными файлами."""
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    labelled = any(result.label for result in results)
    for name in [*detector_names, COMBO, CANDIDATES_DIR, *([LABELS_DIR] if labelled else [])]:
        directory = root / name
        _clear(directory)
        written = 0
        for ordinal, result in enumerate(results, start=1):
            extra = ""
            if name == COMBO:
                measure = result.combo
                if measure is None or not measure.flag:
                    continue
            elif name == CANDIDATES_DIR:
                if not any(m.flag for m in result.measures.values()):
                    continue
                measure = result.combo or Measure()
            elif name == LABELS_DIR:
                if not result.label:
                    continue
                measure = result.combo or Measure()
                extra = f"_{result.label}_{_verdict(result)}"
            else:
                measure = result.measures.get(name)
                if measure is None or not measure.flag:
                    continue
            pdf = lookup(pdf_pages, result.rel_path)
            target = _target(result, link_root)
            link = directory / link_name(result, pdf, measure.score, ordinal, target.suffix, extra)
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)
            written += 1
        counts[name] = written
    return counts


def metric_columns(results: Sequence[PageResult], detector_names: Sequence[str]) -> dict[str, list[str]]:
    columns: dict[str, list[str]] = {name: [] for name in detector_names}
    for result in results:
        for name, measure in result.measures.items():
            if name in columns:
                for key in measure.metrics:
                    if key not in columns[name]:
                        columns[name].append(key)
    return columns


def write_csv(
    path: Path, results: Sequence[PageResult], detector_names: Sequence[str], pdf_pages: dict[str, PdfPage]
) -> None:
    """Все полосы и все метрики: по этому файлу калибруются пороги и работает ``report``."""
    metrics = metric_columns(results, detector_names)
    header = ["полоса", "pdf", "стр", "ширина", "высота", "метка", "combo", "combo_score", "combo_votes"]
    for name in detector_names:
        header += [f"{name}_flag", f"{name}_score", f"{name}_замечание"]
        header += [f"{name}_{key}" for key in metrics[name]]
    header.append("ошибка")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for result in results:
            pdf = lookup(pdf_pages, result.rel_path)
            combo = result.combo
            row = [
                result.rel_path,
                pdf.pdf_name if pdf else "",
                pdf.page_number if pdf else "",
                result.width,
                result.height,
                result.label,
                "" if combo is None or combo.silent else (FLAG_MARK if combo.flag else ""),
                f"{combo.score:.3f}" if combo is not None and not combo.silent else "",
                f"{combo.metrics.get('votes', 0):.0f}" if combo is not None and not combo.silent else "",
            ]
            for name in detector_names:
                measure = result.measures.get(name)
                if measure is None:
                    row += ["", "", ""] + [""] * len(metrics[name])
                    continue
                row += [
                    SILENT_MARK if measure.silent else (FLAG_MARK if measure.flag else ""),
                    "" if measure.silent else f"{measure.score:.3f}",
                    measure.note,
                ]
                row += [f"{measure.metrics[key]:.4f}" if key in measure.metrics else "" for key in metrics[name]]
            row.append(result.error)
            writer.writerow(row)
    logger.info("CSV: %s", path)


def read_csv(path: Path, root: Path, detector_names: Sequence[str] | None = None) -> tuple[list[PageResult], list[str]]:
    """Полосы и их метрики из CSV прогона — для ``report`` без пересчёта.

    Флаги и score из файла НЕ берутся: их заново проставят пороги. Имена детекторов
    восстанавливаются по колонкам ``*_flag``.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        names = [column[: -len("_flag")] for column in header if column.endswith("_flag")]
        if detector_names is not None:
            names = [name for name in names if name in detector_names]
        results: list[PageResult] = []
        for row in reader:
            rel = row["полоса"]
            result = PageResult(
                rel_path=rel,
                path=root / rel,
                width=int(row.get("ширина") or 0),
                height=int(row.get("высота") or 0),
                label=row.get("метка", ""),
                error=row.get("ошибка", ""),
            )
            for name in names:
                flag_cell = row.get(f"{name}_flag")
                note = row.get(f"{name}_замечание", "")
                metrics: dict[str, float] = {}
                prefix = f"{name}_"
                for column, value in row.items():
                    if not column or not column.startswith(prefix) or value in ("", None):
                        continue
                    key = column[len(prefix) :]
                    if key in ("flag", "score", "замечание"):
                        continue
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        continue
                if flag_cell is None or (flag_cell == "" and not metrics and not note):
                    continue  # детектор эту полосу не смотрел
                result.measures[name] = Measure(metrics=metrics, note=note, silent=flag_cell == SILENT_MARK)
            results.append(result)
    return results, names


def _render(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    line = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([line, rule, *body])


def counts_table(results: Sequence[PageResult], detector_names: Sequence[str]) -> str:
    counts = flagged_counts(results, detector_names)
    rows = [
        [name, flagged, silent, errors, len(results) - flagged - silent - errors]
        for name, (flagged, silent, errors) in counts.items()
    ]
    return _render(["детектор", "флаг", "молчит", "нет меры", "прямо"], rows)


def agreement_table(results: Sequence[PageResult], detector_names: Sequence[str]) -> str:
    """Жаккар флагов попарно — по полосам, где ВЫСКАЗАЛИСЬ ОБА. Молчание — не мнение."""
    headers = ["", *detector_names]
    rows = []
    for left in detector_names:
        row = [left]
        for right in detector_names:
            if left == right:
                row.append("—")
                continue
            union = both = 0
            for result in results:
                a, b = result.measures.get(left), result.measures.get(right)
                if a is None or b is None or a.silent or b.silent:
                    continue
                if a.flag or b.flag:
                    union += 1
                    both += a.flag and b.flag
            row.append(f"{both}/{union}" if union else "—")
        rows.append(row)
    return _render(headers, rows)


def distribution_table(results: Sequence[PageResult], detector_names: Sequence[str], thresholds: Thresholds) -> str:
    """Перцентили каждой метрики по паку: видно, где стоит порог относительно массы полос."""
    columns = metric_columns(results, detector_names)
    rows = []
    for name in detector_names:
        for key in columns[name]:
            values = np.array(
                [
                    r.measures[name].metrics[key]
                    for r in results
                    if name in r.measures and not r.measures[name].silent and key in r.measures[name].metrics
                ]
            )
            if values.size == 0:
                continue
            threshold = thresholds.values.get(name, {}).get(key)
            above = int((values >= threshold).sum()) if threshold is not None else ""
            rows.append(
                [
                    name,
                    key,
                    f"{threshold:g}" if threshold is not None else "",
                    *(f"{np.percentile(values, q):.3f}" for q in (50, 90, 95, 99)),
                    f"{values.max():.3f}",
                    above,
                ]
            )
    return _render(["детектор", "метрика", "порог", "p50", "p90", "p95", "p99", "max", "≥ порога"], rows)


def _link(result: PageResult) -> str:
    return f"[{result.rel_path}](file://{quote(str(result.path.resolve()))})"


def findings_table(
    results: Sequence[PageResult], pdf_pages: dict[str, PdfPage], limit: int, detector_names: Sequence[str]
) -> str:
    headers = ["полоса", "pdf", "стр", "combo", "голосов", *detector_names]
    rows = []
    ordered = sorted((r for r in results if r.flagged), key=lambda r: -(r.combo.score if r.combo else 0.0))
    for result in ordered[:limit]:
        pdf = lookup(pdf_pages, result.rel_path)
        row = [
            _link(result),
            pdf.pdf_name if pdf else "",
            pdf.page_number if pdf else "",
            f"{result.combo.score:.2f}",
            f"{result.combo.metrics.get('votes', 0):.0f}",
        ]
        for name in detector_names:
            measure = result.measures.get(name)
            if measure is None:
                row.append("—")
            elif measure.silent:
                row.append(SILENT_MARK)
            else:
                row.append(f"{'✔ ' if measure.flag else ''}{measure.score:.2f}")
        rows.append(row)
    return _render(headers, rows)


def labels_section(results: Sequence[PageResult], thresholds: Thresholds, detector_names: Sequence[str]) -> str:
    labelled = [r for r in results if r.label]
    if not labelled:
        return ""
    headers = ["полоса", "метка", "combo", *detector_names]
    rows = []
    for result in labelled:
        combo = result.combo
        combo_text = "—" if combo is None or combo.silent else f"{'✔ ' if combo.flag else ''}{combo.score:.2f}"
        row = [_link(result), result.label, f"{combo_text} ({_verdict(result)})"]
        for name in detector_names:
            measure = result.measures.get(name)
            if measure is None:
                row.append("—")
            elif measure.silent:
                row.append(SILENT_MARK)
            else:
                row.append(f"{'✔ ' if measure.flag else ''}{measure.score:.2f}")
        rows.append(row)
    pairs = [(r.label, r.metrics_by_detector()) for r in labelled]
    items = labelling.separation(pairs, thresholds.values, detector_names)
    ok = sum(1 for r in labelled if _verdict(r) == "ok")
    return "\n".join(
        [
            f"Размечено полос: {len(labelled)} (кривых {sum(1 for r in labelled if r.label == labelling.CURVED)}, "
            f"прямых {sum(1 for r in labelled if r.label == labelling.STRAIGHT)}). Сводный вердикт верен на {ok} из {len(labelled)}.",
            "",
            _render(headers, rows),
            "",
            "Разделение по метрикам (только флаговые имеют порог). «Зазор» — min по кривым минус max по прямым: "
            "положительный значит, что метрика разводит классы чисто; «предложить» — середина зазора.",
            "",
            labelling.separation_table(items),
        ]
    )


def markdown_report(
    results: Sequence[PageResult],
    detector_names: Sequence[str],
    pdf_pages: dict[str, PdfPage],
    root: Path,
    link_dir: Path | None,
    link_counts: dict[str, int],
    thresholds: Thresholds,
    votes: int,
    strong: float,
    limit: int = 400,
) -> str:
    errors = [r for r in results if r.error]
    found = [r for r in results if r.flagged]
    blocks = [
        f"# Полосы с кривыми строками: {root}",
        "",
        f"Полос: {len(results)}. Сводный флаг: {len(found)}. Ошибок чтения: {len(errors)}.",
        "",
    ]
    if link_dir is not None:
        blocks += [f"Симлинки: `{link_dir}`", "", _render(["каталог", "полос"], sorted(link_counts.items())), ""]
    blocks += [
        "## 1. Пороги",
        "",
        f"Флаг детектора: хоть одна флаговая метрика ≥ порога (score = max метрика/порог). Сводный флаг: "
        f"голосов ≥ {votes} ИЛИ max score ≥ {strong:g}.",
        "",
        thresholds.table(),
        "",
        "## 2. Что нашёл каждый детектор",
        "",
        counts_table(results, detector_names),
        "",
        "«Молчит» — детектору не по чему мерить: мало строк, нет периодичности. Это честный отказ, а не «прямо».",
        "",
        "## 3. Попарное согласие",
        "",
        agreement_table(results, detector_names),
        "",
        "В клетке «оба флагнули / хоть один флагнул» среди полос, где высказались оба (Жаккар).",
        "",
        "## 4. Распределение метрик по всем полосам",
        "",
        distribution_table(results, detector_names, thresholds),
        "",
    ]
    section = 5
    labels_text = labels_section(results, thresholds, detector_names)
    if labels_text:
        blocks += [f"## {section}. Разметка", "", labels_text, ""]
        section += 1
    blocks += [
        f"## {section}. Находки по сводному score (до {limit})",
        "",
        findings_table(results, pdf_pages, limit, detector_names),
    ]
    section += 1
    if errors:
        blocks += [
            "",
            f"## {section}. Не прочитались",
            "",
            _render(["полоса", "ошибка"], [[r.rel_path, r.error] for r in errors[:limit]]),
        ]
    seconds = {name: [r.seconds[name] for r in results if r.seconds.get(name)] for name in detector_names}
    timing = [[name, f"{np.median(values):.3f}" if values else "—", len(values)] for name, values in seconds.items()]
    blocks += [
        "",
        f"## {section + 1}. Время на полосу (медиана, только посчитанные заново)",
        "",
        _render(["детектор", "с", "полос"], timing),
    ]
    return "\n".join(blocks) + "\n"


# --- Контактный лист --------------------------------------------------------
SHEET_TILE_PX = 300
SHEET_COLUMNS = 6
SHEET_CAPTION_PX = 22
SHEET_PER_PAGE = 60
# Реперные горизонтали: на миниатюре в 300 px кривизна строк не видна, а отклонение
# строки от горизонтали рядом с линией — видно.
SHEET_GUIDES = 12


def _thumbnail(path: Path, side: int) -> np.ndarray | None:
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.draft("L", (side, side))
            gray = np.asarray(image.convert("L"))
    except Exception:
        return None
    height, width = gray.shape
    scale = side / max(height, width)
    return cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def _tile(gray: np.ndarray, caption: str) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index in range(1, SHEET_GUIDES):
        y = int(canvas.shape[0] * index / SHEET_GUIDES)
        cv2.line(canvas, (0, y), (canvas.shape[1], y), (0, 0, 255), 1)
    pad_x = SHEET_TILE_PX - canvas.shape[1]
    canvas = cv2.copyMakeBorder(
        canvas,
        SHEET_CAPTION_PX,
        SHEET_TILE_PX - canvas.shape[0],
        pad_x // 2,
        pad_x - pad_x // 2,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    cv2.putText(canvas, caption[:44], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(160, 160, 160))


def _grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    blank = np.full_like(tiles[0], 255)
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def contact_sheet(
    base_path: Path,
    results: Sequence[PageResult],
    pdf_pages: dict[str, PdfPage],
    columns: int = SHEET_COLUMNS,
    per_page: int = SHEET_PER_PAGE,
) -> list[Path]:
    """Листы миниатюр полос со сводным флагом, по убыванию score, с реперными горизонталями."""
    chosen = sorted((r for r in results if r.flagged), key=lambda r: -(r.combo.score if r.combo else 0.0))
    written: list[Path] = []
    for sheet_index in range(0, len(chosen), per_page):
        tiles = []
        for result in chosen[sheet_index : sheet_index + per_page]:
            gray = _thumbnail(result.path, SHEET_TILE_PX)
            if gray is None:
                continue
            pdf = lookup(pdf_pages, result.rel_path)
            page = f"p{pdf.page_number:03d}" if pdf else "p???"
            votes = result.combo.metrics.get("votes", 0) if result.combo else 0
            tiles.append(_tile(gray, f"{Path(result.rel_path).stem} {page} s{result.combo.score:.2f} v{votes:.0f}"))
        if not tiles:
            continue
        path = base_path.with_name(f"{base_path.stem}_{sheet_index // per_page + 1:02d}{base_path.suffix}")
        cv2.imwrite(str(path), _grid(tiles, columns))
        written.append(path)
    logger.info("Контактных листов: %d", len(written))
    return written
