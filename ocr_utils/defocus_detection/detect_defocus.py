"""CLI детектора расфокуса для сканов газет (RAF с JPEG-превью или обычные JPEG).

Это инструмент СКРИНИНГА/РАНЖИРОВАНИЯ, а не точный классификатор: он поднимает наверх
кандидатов на расфокус (в т.ч. зональный), а финальное решение — за человеком.

Запуск (см. README.md этого подпакета):
    uv run python -m ocr_utils.defocus_detection "/путь/к/выпуску"          # режим по умолчанию
    uv run python -m ocr_utils.defocus_detection "/путь" --quiet            # только подозрительные

Конвейер (реализации методов — в соседних модулях moire.py / laplacian.py / fft_hf.py,
оркестрация — в pipeline.py):
- A (`--normalize`): нормировка муара на «кол-во резких переходов» (убирает зависимость
  от количества текста). По умолчанию `structure`; `gradient` — тупиковый.
- B (`--min-edge-density`): отсев обложек/пустых листов без типографского растра.
- C (`--zone`): подозрение по СВЯЗНОЙ 2D-зоне мягких тайлов (а не по всей полосе).
- E (`--cross-check`): второй детектор FFT HF/MID; согласие обоих → высокая уверенность.

ВАЖНО: базовый уровень муара различается между выпусками (бумага, растр) — запускайте
ПО ОДНОМУ выпуску. Метод muire надёжно ловит сильную «мыльность», но для тонкого
зонального расфокуса он менее чувствителен, чем FFT HF/MID; для такого поиска опирайтесь
на голос FFT (колонка вердикта). Параметры `--zone-*` стоит калибровать на размеченной пачке.
"""

import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

import click
import numpy as np
from tqdm import tqdm

from ocr_utils.defocus_detection.image_io import collect_inputs, read_image_gray, save_heatmap
from ocr_utils.defocus_detection.pipeline import VERDICTS, analyze, verdict


def _md_link(path: Path) -> str:
    """Markdown-ссылка на файл: подпись — имя, цель — абсолютный путь без `file://`.

    Путь URL-кодируется (пробелы, кириллица, скобки), чтобы ссылка не ломалась,
    но схема `file://` НЕ добавляется — так путь остаётся кликабельным локально и
    читаемым (как в focus_detection_report.md).
    """
    target = quote(str(path.resolve()))
    return f"[{path.name}]({target})"


@click.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path())
@click.option(
    "--method",
    type=click.Choice(["moire", "laplacian"]),
    default="moire",
    show_default=True,
    help="Метод оценки резкости.",
)
@click.option("--factor", type=float, default=3.0, show_default=True, help="Коэффициент уменьшения для метода moire.")
@click.option(
    "--normalize",
    type=click.Choice(["none", "structure", "gradient", "global_contrast"]),
    default="structure",
    show_default=True,
    help="Нормировка муара на кол-во резких переходов в тайле (убирает зависимость "
    "от доли текста): none — сырой муар; structure (A1) — делить на std AREA-тайла; "
    "gradient (A2) — делить на RMS полноразмерного градиента; "
    "global_contrast (A1+, экспериментальная) — structure + нормировка на центральный "
    "std изображения (пытается убрать зависимость от общего динамического диапазона).",
)
@click.option("--grid-x", type=int, default=16, show_default=True, help="Число тайлов по горизонтали.")
@click.option("--grid-y", type=int, default=11, show_default=True, help="Число тайлов по вертикали.")
@click.option(
    "--min-structure",
    type=float,
    default=8.0,
    show_default=True,
    help="Порог контраста тайла (std), ниже — считается пустым полем.",
)
@click.option(
    "--min-edge-density",
    type=float,
    default=0.045,
    show_default=True,
    help="B (гейт обложек): файлы с плотностью краёв Canny ниже порога помечаются "
    "ОБЛОЖКА/ПУСТО (нет типографского растра) и исключаются из ранжирования.",
)
@click.option(
    "--zone/--no-zone",
    "zone_mode",
    default=True,
    show_default=True,
    help="C: выносить подозрение по связной 2D-зоне расфокуса (только moire+нормировка). "
    "При --no-zone используется прежнее ранжирование по худшей зоне.",
)
@click.option(
    "--zone-margin", type=int, default=1, show_default=True, help="C: сколько крайних рядов тайлов (поля) исключить."
)
@click.option(
    "--zone-k-abs",
    type=float,
    default=0.6,
    show_default=True,
    help="C: абсолютный порог «мягкости» нормированного муара.",
)
@click.option(
    "--zone-k-rel",
    type=float,
    default=0.6,
    show_default=True,
    help="C: доля медианы полосы, ниже которой тайл «мягкий».",
)
@click.option(
    "--zone-grad-rel",
    type=float,
    default=0.7,
    show_default=True,
    help="C: доля медианы градиента, ниже которой тайл считается реально размытым "
    "(отсекает резкие нерастровые края). Большое значение (напр. 99) отключает гейт.",
)
@click.option("--zone-min-rows", type=int, default=2, show_default=True, help="C: минимальная высота зоны в тайлах.")
@click.option("--zone-min-cols", type=int, default=3, show_default=True, help="C: минимальная ширина зоны в тайлах.")
@click.option(
    "--cross-check/--no-cross-check",
    default=True,
    show_default=True,
    help="E: кросс-проверка вторым детектором FFT HF/MID (чувствительнее к тонкому "
    "зональному расфокусу). Совпадение обоих → высокая уверенность; только муар → "
    "вероятно ложняк (цветной декор/край); только FFT → расфокус, муар не увидел.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Абсолютный порог резкости: файлы ниже помечаются как подозрительные. "
    "По умолчанию: moire без нормировки=15.0, moire с нормировкой=0.5, laplacian=900.0. "
    "Игнорируется при --relative.",
)
@click.option(
    "--relative/--absolute",
    default=True,
    show_default=True,
    help="relative: адаптивный порог относительно выборки (выпуска) — устойчив к "
    "разной бумаге/растру между выпусками. absolute: фиксированный --threshold.",
)
@click.option(
    "--z",
    type=float,
    default=2.0,
    show_default=True,
    help="Для --relative: на сколько робастных сигм (MAD) ниже медианы выборки "
    "считать файл подозрительным. Меньше — больше срабатываний.",
)
@click.option(
    "--debug-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Каталог для сохранения тепловых карт тайлов (отладка).",
)
@click.option(
    "--md-report",
    type=click.Path(dir_okay=False),
    default=None,
    help="Записать md-отчёт с таблицей и кликабельными ссылками на файлы по этому пути.",
)
@click.option("--quiet", is_flag=True, help="Печатать только подозрительные файлы (в консоль; md-отчёт всегда полный).")
def main(
    inputs: tuple[str, ...],
    method: str,
    factor: float,
    normalize: str,
    grid_x: int,
    grid_y: int,
    min_structure: float,
    min_edge_density: float,
    zone_mode: bool,
    zone_margin: int,
    zone_k_abs: float,
    zone_k_rel: float,
    zone_grad_rel: float,
    zone_min_rows: int,
    zone_min_cols: int,
    cross_check: bool,
    threshold: float | None,
    relative: bool,
    z: float,
    debug_dir: str | None,
    md_report: str | None,
    quiet: bool,
) -> None:
    """Ищет расфокусные сканы среди INPUTS (файлы и/или одна директория, нерекурсивно).

    Режим по умолчанию (moire + нормировка + --zone): подозрение выносится по СВЯЗНОЙ
    2D-зоне расфокуса (этап C), обложки/пустые листы без растра отсеиваются по плотности
    краёв (этап B). Файлы сортируются: сначала подозрительные (по глубине зоны), затем
    остальные (по «здоровью» полосы), в конце — отсеянные обложки.

    Режим --no-zone (а также --normalize none и --method laplacian): прежнее ранжирование
    по «худшей зоне» с адаптивным порогом (--relative/--z), устойчивым к разной бумаге
    между выпусками. Запускайте по одному выпуску.
    """
    if threshold is None:
        if method == "laplacian":
            threshold = 900.0
        elif normalize == "none":
            threshold = 15.0
        else:
            # Нормированная метрика лежит примерно в [0, 1.2]; «здоровая» полоса ~0.7.
            threshold = 0.5

    use_zone = zone_mode and method == "moire" and normalize != "none"
    zone_params = None
    if use_zone:
        zone_params = dict(
            margin=zone_margin,
            k_abs=zone_k_abs,
            k_rel=zone_k_rel,
            g_rel=zone_grad_rel,
            min_rows=zone_min_rows,
            min_cols=zone_min_cols,
        )

    files = collect_inputs(inputs)
    if not files:
        click.echo("Нет входных файлов.", err=True)
        sys.exit(1)

    debug_path = None
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

    # vmax для шкалы тепловых карт: «нормальная» резкость ≈ порог * 1.5
    vmax = threshold * 1.5

    rows = []
    for f in tqdm(files, desc="Анализ", unit="файл"):
        gray = read_image_gray(f)
        if gray is None:
            click.echo(f"Предупреждение: не удалось прочитать — {f.name}", err=True)
            continue
        res = analyze(gray, method, factor, grid_x, grid_y, min_structure, normalize, zone_params, cross_check)
        rows.append((f, res))
        if debug_path is not None:
            save_heatmap(debug_path / f"{f.stem}_heat.png", res, vmax)

    # B: отделяем обложки/пустые листы (нет типографского растра) — их нельзя
    # ранжировать вместе с полосами.
    covers = [(f, res) for f, res in rows if res["edge_density"] < min_edge_density]
    pages = [(f, res) for f, res in rows if res["edge_density"] >= min_edge_density]

    md_path = Path(md_report) if md_report else None
    if use_zone:
        _report_zone(pages, covers, quiet, cross_check, md_path)
    else:
        _report_worst_zone(pages, covers, threshold, relative, z, method, quiet, md_path)
    if md_path is not None:
        click.echo(f"\nMd-отчёт записан: {md_path}")


def _report_zone(pages: list, covers: list, quiet: bool, cross_check: bool, md_path: Path | None = None) -> None:
    """Печатает результат в режиме 2D-зоны (этап C) с кросс-проверкой FFT (этап E)."""

    def depth_of(res: dict) -> float:
        return res["zone"]["depth"] if res["zone"] else 1e18

    def health_of(res: dict) -> float:
        return np.nan_to_num(res["inner_median"], nan=1e18)

    # Сортировка: по уровню уверенности (both → fft → moire → ok), внутри — по глубине
    # зоны муара, затем по «здоровью» полосы.
    pages.sort(key=lambda r: (VERDICTS[verdict(r[1], cross_check)][0], depth_of(r[1]), health_of(r[1])))

    click.echo("")
    head = "связная 2D-зона (C)" + ("  +  кросс-проверка FFT HF/MID (E)" if cross_check else "")
    click.echo(f"Метод: moire (нормировка)   Решение: {head}")
    click.echo(f"{'резкость':>9} {'зона':>7} {'глубина':>8} {'отн.':>5} {'FFT':>4}  файл / вердикт")
    click.echo("-" * 78)

    counts = {"both": 0, "fft": 0, "moire": 0, "ok": 0}
    md_rows: list[str] = []
    for f, res in pages:
        v = verdict(res, cross_check)
        counts[v] += 1
        zone = res["zone"]
        if zone is not None:
            inner = res["inner_median"]
            rel = zone["depth"] / inner if inner and not np.isnan(inner) else float("nan")
            zcol, dcol, rcol = f"{zone['rows']}x{zone['cols']}", f"{zone['depth']:.2f}", f"{rel:.2f}"
        else:
            zcol = dcol = rcol = "—"
        fcol = ("да" if res.get("fft_defocus") else "нет") if cross_check else "—"
        tag = VERDICTS[v][1]
        md_rows.append(
            f"| {_md_link(f)} | {tag or 'ok'} | {res['sharpness']:.2f} | {zcol} | {dcol} | {rcol} | {fcol} |"
        )
        if v == "ok" and quiet:
            continue
        suffix = f"  <-- {tag}" if tag else ""
        click.echo(f"{res['sharpness']:9.2f} {zcol:>7} {dcol:>8} {rcol:>5} {fcol:>4}  {f.name}{suffix}")

    if covers and not quiet:
        click.echo("")
        click.echo(f"ОБЛОЖКА/ПУСТО (нет растра, исключены из ранжирования): {len(covers)}")
        for f, res in sorted(covers, key=lambda r: r[1]["edge_density"]):
            click.echo(f"   edge={res['edge_density']:.4f}  {f.name}")

    click.echo("-" * 78)
    if cross_check:
        summary = (
            f"оба детектора={counts['both']}, только FFT={counts['fft']}, "
            f"только муар={counts['moire']}, чисто={counts['ok']}; обложек={len(covers)} из {len(pages)} полос"
        )
        click.echo(f"Итог: {summary}")
    else:
        summary = (
            f"подозрительных (зона муара): {counts['both'] + counts['moire']} из {len(pages)} полос; "
            f"обложек: {len(covers)}"
        )
        click.echo(f"Итог: {summary}")

    if md_path is not None:
        cols = "| файл | вердикт | резкость | зона | глубина | отн.полосы | FFT |"
        align = "|---|---|--:|:--:|--:|--:|:--:|"
        _write_md_report(md_path, f"moire (нормировка) + {head}", cols, align, md_rows, covers, summary)


def _report_worst_zone(
    pages: list,
    covers: list,
    threshold: float,
    relative: bool,
    z: float,
    method: str,
    quiet: bool,
    md_path: Path | None = None,
) -> None:
    """Печатает результат в прежнем режиме: ранжирование по «худшей зоне» (этап C выключен)."""
    pages.sort(key=lambda r: (np.nan_to_num(r[1]["worst_zone"], nan=1e18)))
    worst_vals = np.array([r[1]["worst_zone"] for r in pages if not np.isnan(r[1]["worst_zone"])])

    if relative and worst_vals.size >= 4:
        med = float(np.median(worst_vals))
        mad = max(float(np.median(np.abs(worst_vals - med))) * 1.4826, 1e-6)
        eff_threshold = med - z * mad
        header = f"Порог (relative): медиана {med:.2f} − {z:g}·MAD = {eff_threshold:.2f}"

        def strength_of(s: float) -> float:
            return float(max(0.0, (med - s) / mad))

        strength_label = "сигм"
    else:
        eff_threshold = threshold

        def strength_of(s: float) -> float:
            return float(np.clip((eff_threshold - s) / eff_threshold, 0.0, 1.0))

        strength_label = "0..1"
        header = f"Порог (absolute): {eff_threshold:g}"

    click.echo("")
    click.echo(f"Метод: {method}   {header}   (ранжирование по худшей зоне; выше = резче)")
    click.echo(f"{'резкость':>10} {'худш.зона':>10} {'подозр,' + strength_label:>10}  файл")
    click.echo("-" * 62)

    suspects = []
    md_rows: list[str] = []
    for f, res in pages:
        sharp = res["sharpness"]
        worst = res["worst_zone"]
        if np.isnan(worst):
            strength, is_suspect = 0.0, False
        else:
            strength, is_suspect = strength_of(worst), worst < eff_threshold
        if is_suspect:
            suspects.append((f, worst, strength))
        md_rows.append(f"| {_md_link(f)} | {sharp:.2f} | {worst:.2f} | {strength:.2f} | {'да' if is_suspect else ''} |")
        if quiet and not is_suspect:
            continue
        mark = "  <-- РАСФОКУС?" if is_suspect else ""
        click.echo(f"{sharp:10.2f} {worst:10.2f} {strength:10.2f}  {f.name}{mark}")

    if covers and not quiet:
        click.echo(f"\nОБЛОЖКА/ПУСТО (нет растра, исключены): {len(covers)}")
        for f, res in sorted(covers, key=lambda r: r[1]["edge_density"]):
            click.echo(f"   edge={res['edge_density']:.4f}  {f.name}")

    click.echo("-" * 62)
    summary = f"подозрительных: {len(suspects)} из {len(pages)} полос; обложек: {len(covers)}"
    click.echo(f"Итог: {summary}")

    if md_path is not None:
        cols = f"| файл | резкость | худш.зона | подозр.({strength_label}) | расфокус? |"
        align = "|---|--:|--:|--:|:--:|"
        _write_md_report(md_path, f"{method} — {header}", cols, align, md_rows, covers, summary)


def _write_md_report(
    md_path: Path, method_line: str, cols: str, align: str, md_rows: list[str], covers: list, summary: str
) -> None:
    """Записывает md-отчёт: заголовок, таблица результатов с кликабельными ссылками,
    отдельная таблица отсеянных обложек и итоговая сводка.

    Args:
        md_path: Путь выходного .md.
        method_line: Строка описания метода/решения (в шапку).
        cols: Строка заголовка таблицы результатов (markdown).
        align: Строка выравнивания столбцов (markdown).
        md_rows: Готовые строки таблицы результатов.
        covers: Отсеянные обложки [(Path, res), ...].
        summary: Итоговая сводка одной строкой.
    """
    lines = [
        "# Отчёт детекции расфокуса",
        "",
        f"- **Дата:** {date.today().isoformat()}",
        f"- **Метод:** {method_line}",
        f"- **Итог:** {summary}",
        "",
        "Ссылки кликабельны (абсолютные пути файлов). Сортировка — самые подозрительные сверху.",
        "",
        "## Результаты",
        "",
        cols,
        align,
        *md_rows,
    ]
    if covers:
        lines += [
            "",
            "## Обложки/пусто (нет растра, исключены из ранжирования)",
            "",
            "| файл | edge |",
            "|---|--:|",
            *[
                f"| {_md_link(f)} | {res['edge_density']:.4f} |"
                for f, res in sorted(covers, key=lambda r: r[1]["edge_density"])
            ],
        ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
