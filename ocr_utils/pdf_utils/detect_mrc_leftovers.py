"""Поиск страниц, где при MRC-сжатии часть текста «провалилась» в фоновый слой.

Сканы в PDF часто сжаты по схеме MRC (Mixed Raster Content): страница разбита на
три слоя — фон (`/Bg`, JPEG2000, обычно 150 dpi), цвет переднего плана (`/CL`,
JPEG2000, ~75 dpi) и бинарная маска-трафарет (`/Mask`, JBIG2, 300 dpi). Маска
задаёт форму букв: где у неё есть чернила, видно резкий передний план, в
остальных местах — сильно сглаженный фон.

Сегментатор кодировщика иногда не опознаёт часть символов как текст: обычно это
цифры в оглавлении, знаки препинания, индексы у формул, целые повёрнутые
таблицы. Такие символы в маску не попадают и остаются только в фоновом слое —
то есть показываются в четверть разрешения и после агрессивного JPEG2000. Внешне
это выглядит как несколько «размазанных» букв посреди резкого текста.

Дефект необратим: резкого варианта этих символов в файле просто нет, страницу
нужно пересканировать. Модуль ищет такие страницы, чтобы пересканировать их
выборочно.

Как ищем: берём фоновый слой, вычитаем локальный уровень бумаги и находим тёмные
компактные пятна там, где у маски чернил нет. Дополнительно требуем, чтобы пятно
лежало на текстовой строке (рядом по горизонтали есть чернила маски) и чтобы
вокруг него была светлая бумага — это отсекает корешок, края скана и пыль.

Примеры использования:

    # Одна папка с PDF, отчёт в CSV
    uv run python -m ocr_utils.pdf_utils.detect_mrc_leftovers \\
        --input-dir "/mnt/.../1987" --report-csv mrc_leftovers.csv

    # С картинками-подсказками для 40 худших страниц
    uv run python -m ocr_utils.pdf_utils.detect_mrc_leftovers \\
        --input-dir "/mnt/.../1987" --report-csv mrc.csv \\
        --debug-dir mrc_debug --debug-top 40
"""

from __future__ import annotations

import csv
import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

import click
import cv2
import fitz
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Порог «пятно темнее бумаги» для попадания в кандидаты, в уровнях серого 0..255
DARKNESS_THRESHOLD = 30
# Насколько тёмным должен быть самый тёмный пиксель пятна, чтобы счесть его буквой,
# а не призраком уже вытащенного в маску текста (тот в фоне остаётся еле заметным)
PEAK_DARKNESS_THRESHOLD = 45
# Кольцо шириной PAD вокруг пятна должно быть светлой бумагой не темнее этого
RING_DARKNESS_LIMIT = 18
RING_PAD = 4
# Допустимый размер пятна в пикселях фонового слоя (150 dpi: буква ~8x12, цифра ~10x14)
MIN_BLOB_AREA = 5
MAX_BLOB_AREA = 400
MAX_BLOB_SIDE = 45
# Одиночное мелкое пятно — почти всегда пылинка на стекле, а не потерянный символ:
# настоящие потери идут подряд вдоль строки (слово, число, формула). Пятно крупнее
# LONE_BLOB_MIN_AREA считаем символом даже без соседей.
LONE_BLOB_MIN_AREA = 25
NEIGHBOUR_MAX_DX = 25
NEIGHBOUR_MAX_DY = 8


@dataclass
class Blob:
    """Одно найденное пятно — предположительно символ, оставшийся в фоне."""

    x: int
    y: int
    w: int
    h: int
    area: int
    peak: int


@dataclass
class PageResult:
    """Результат анализа одной страницы PDF."""

    pdf: Path
    page_no: int  # 1-based, как в PDF-просмотрщике
    n_blobs: int = 0
    blob_area: int = 0
    max_blob_area: int = 0
    left_blobs: int = 0
    right_blobs: int = 0
    bg_width: int = 0
    bg_height: int = 0
    status: str = "ok"
    blobs: list[Blob] = field(default_factory=list)

    @property
    def severity(self) -> float:
        """Оценка тяжести дефекта: сумма корней из площадей пятен.

        Корень, а не сама площадь, чтобы одна большая слипшаяся клякса не
        перевешивала десяток отдельных потерянных букв.
        """
        return sum(float(np.sqrt(b.area)) for b in self.blobs)


def _find_mrc_layers(doc: fitz.Document, page: fitz.Page) -> tuple[int | None, list[int], list[int]]:
    """Разобрать страницу на слои MRC.

    Returns:
        Кортеж (xref фонового слоя, список xref'ов масок-трафаретов,
        список xref'ов прочих картинок — врезок, которые надо исключить из анализа).
        Фоновым считается самая большая картинка без `/Mask`.
    """
    bg_xref: int | None = None
    bg_pixels = 0
    stencils: list[int] = []
    others: list[int] = []

    for info in page.get_images(full=True):
        xref, width, height = info[0], info[2], info[3]
        obj = doc.xref_object(xref)
        if "/Mask" in obj:
            key = doc.xref_get_key(xref, "Mask")
            try:
                stencils.append(int(str(key[1]).split()[0]))
            except (ValueError, IndexError):
                logger.debug("не удалось разобрать /Mask у xref %s: %r", xref, key)
            continue
        if width * height > bg_pixels:
            if bg_xref is not None:
                others.append(bg_xref)
            bg_xref, bg_pixels = xref, width * height
        else:
            others.append(xref)

    return bg_xref, stencils, others


def _pixmap_gray(doc: fitz.Document, xref: int) -> np.ndarray:
    """Прочитать изображение по xref и вернуть его как одноканальный uint8-массив."""
    px = fitz.Pixmap(doc, xref)
    arr = np.frombuffer(px.samples, dtype=np.uint8).reshape(px.height, px.width, px.n)
    if px.n >= 3:
        return cv2.cvtColor(np.ascontiguousarray(arr[:, :, :3]), cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(arr[:, :, 0])


def _stencil_coverage(doc: fitz.Document, stencils: list[int], shape: tuple[int, int]) -> np.ndarray:
    """Свести все маски-трафареты страницы в одну карту покрытия чернилами.

    Маски лежат в 300 dpi, фон — в 150 dpi, поэтому ужимаем маски до размера фона
    через INTER_AREA: получается доля чернил в каждом пикселе фона.
    """
    height, width = shape
    coverage = np.zeros((height, width), np.float32)
    for xref in stencils:
        px = fitz.Pixmap(doc, xref)
        mask = np.frombuffer(px.samples, dtype=np.uint8).reshape(px.height, px.width)
        ink = (mask >= 128).astype(np.float32)
        # У `/ImageMask` полярность зависит от `/Decode`; чернил на текстовой
        # странице всегда меньшинство, поэтому ориентируемся на это.
        if ink.mean() > 0.5:
            ink = 1.0 - ink
        coverage = np.maximum(coverage, cv2.resize(ink, (width, height), interpolation=cv2.INTER_AREA))
    return coverage


def _excluded_by_insets(doc: fitz.Document, page: fitz.Page, bg_xref: int, others: list[int], shape: tuple[int, int]):
    """Построить маску областей, занятых отдельными картинками-врезками.

    Иллюстрации кладутся в PDF отдельными XObject'ами поверх фона; растр внутри
    них — законно нерезкий, и анализировать его не надо.
    """
    height, width = shape
    excluded = np.zeros((height, width), np.uint8)
    bg_rects = page.get_image_rects(bg_xref)
    if not bg_rects:
        return excluded
    bg_rect = bg_rects[0]
    if bg_rect.width <= 0 or bg_rect.height <= 0:
        return excluded

    for xref in others:
        for rect in page.get_image_rects(xref):
            x0 = int((rect.x0 - bg_rect.x0) / bg_rect.width * width)
            x1 = int((rect.x1 - bg_rect.x0) / bg_rect.width * width)
            y0 = int((rect.y0 - bg_rect.y0) / bg_rect.height * height)
            y1 = int((rect.y1 - bg_rect.y0) / bg_rect.height * height)
            x0, x1 = max(0, min(x0, x1)), min(width, max(x0, x1))
            y0, y1 = max(0, min(y0, y1)), min(height, max(y0, y1))
            excluded[y0:y1, x0:x1] = 1
    return excluded


def _drop_isolated_specks(blobs: list[Blob]) -> list[Blob]:
    """Выбросить одиночные мелкие пятна — пыль и соринки на стекле сканера.

    Потерянный текст почти всегда идёт группой: соседние буквы слова, цифры
    числа, символы формулы. Пятно оставляем, если оно либо достаточно крупное
    само по себе, либо у него есть сосед на той же строке.
    """
    kept: list[Blob] = []
    for blob in blobs:
        if blob.area >= LONE_BLOB_MIN_AREA:
            kept.append(blob)
            continue
        centre_y = blob.y + blob.h / 2
        has_neighbour = False
        for other in blobs:
            if other is blob:
                continue
            if abs((other.y + other.h / 2) - centre_y) > NEIGHBOUR_MAX_DY:
                continue
            # Отрицательный зазор — пятна перекрываются по горизонтали
            gap = max(other.x - (blob.x + blob.w), blob.x - (other.x + other.w))
            if gap <= NEIGHBOUR_MAX_DX:
                has_neighbour = True
                break
        if has_neighbour:
            kept.append(blob)
    return kept


def analyze_page(doc: fitz.Document, pdf_path: Path, page_no: int) -> PageResult:
    """Проанализировать одну страницу и найти символы, оставшиеся в фоновом слое.

    Args:
        doc: Открытый документ PyMuPDF
        pdf_path: Путь к PDF (только для отчёта)
        page_no: Номер страницы, 1-based — как показывает PDF-просмотрщик

    Returns:
        PageResult со списком найденных пятен. Если страница не в MRC-формате,
        возвращается результат со `status` != "ok" и пустым списком.
    """
    result = PageResult(pdf=pdf_path, page_no=page_no)
    page = doc[page_no - 1]
    bg_xref, stencils, others = _find_mrc_layers(doc, page)
    if bg_xref is None:
        result.status = "no-background"
        return result
    if not stencils:
        result.status = "no-mrc-mask"
        return result

    bg = _pixmap_gray(doc, bg_xref)
    height, width = bg.shape
    result.bg_width, result.bg_height = width, height

    coverage = _stencil_coverage(doc, stencils, bg.shape)
    # Чернила маски плюс запас: по краям штрихов фон тоже подтемнён
    covered = cv2.dilate((coverage > 0.002).astype(np.uint8), np.ones((5, 5), np.uint8))
    # Пятно должно лежать на текстовой строке: ищем чернила маски в пределах
    # ±near_px по горизонтали и ±4 px по вертикали
    near_px = max(20, width // 25)
    near_text = cv2.dilate((coverage > 0.02).astype(np.uint8), np.ones((9, 2 * near_px + 1), np.uint8))

    # Уровень бумаги — локальный максимум яркости, сглаженный по той же окрестности
    paper = cv2.blur(cv2.dilate(bg, np.ones((31, 31), np.uint8)).astype(np.float32), (31, 31))
    darkness = np.clip(paper - bg.astype(np.float32), 0, 255)

    excluded = _excluded_by_insets(doc, page, bg_xref, others, bg.shape)
    candidates = ((darkness > DARKNESS_THRESHOLD) & (covered == 0) & (near_text > 0) & (excluded == 0)).astype(np.uint8)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (MIN_BLOB_AREA <= area <= MAX_BLOB_AREA):
            continue
        if w > MAX_BLOB_SIDE or h > MAX_BLOB_SIDE or w < 2 or h < 3:
            continue

        selection = labels[y : y + h, x : x + w] == i
        peak = int(darkness[y : y + h, x : x + w][selection].max())
        if peak < PEAK_DARKNESS_THRESHOLD:
            continue

        # Вокруг настоящей потерянной буквы — чистая бумага. В корешке, на краю
        # скана и внутри растровой картинки окружение тоже тёмное.
        y0, y1 = max(0, y - RING_PAD), min(height, y + h + RING_PAD)
        x0, x1 = max(0, x - RING_PAD), min(width, x + w + RING_PAD)
        ring = darkness[y0:y1, x0:x1].astype(np.float32).copy()
        ring[y - y0 : y - y0 + h, x - x0 : x - x0 + w][selection] = np.nan
        if np.nanmedian(ring) > RING_DARKNESS_LIMIT:
            continue

        result.blobs.append(Blob(x=x, y=y, w=w, h=h, area=area, peak=peak))

    result.blobs = _drop_isolated_specks(result.blobs)
    result.n_blobs = len(result.blobs)
    result.blob_area = sum(b.area for b in result.blobs)
    result.max_blob_area = max((b.area for b in result.blobs), default=0)
    # Разворот книги: полезно знать, какую из двух бумажных страниц пересканировать
    result.left_blobs = sum(1 for b in result.blobs if b.x + b.w // 2 < width // 2)
    result.right_blobs = result.n_blobs - result.left_blobs
    return result


def analyze_pdf(pdf_path: Path) -> list[PageResult]:
    """Проанализировать все страницы одного PDF."""
    results: list[PageResult] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 — битый файл не должен ронять весь прогон
        logger.warning("не удалось открыть %s: %s", pdf_path, exc)
        return [PageResult(pdf=pdf_path, page_no=0, status=f"open-error: {exc}")]

    with doc:
        for page_no in range(1, doc.page_count + 1):
            try:
                results.append(analyze_page(doc, pdf_path, page_no))
            except Exception as exc:  # noqa: BLE001
                logger.warning("ошибка на %s стр. %d: %s", pdf_path, page_no, exc)
                results.append(PageResult(pdf=pdf_path, page_no=page_no, status=f"error: {exc}"))
    return results


def save_debug_image(result: PageResult, output_path: Path) -> None:
    """Сохранить фоновый слой страницы с обведёнными пятнами — для глазной проверки.

    Фон растянут по контрасту: в оригинале потерянные буквы бледные, и на
    неподготовленной картинке разметку трудно оценить.
    """
    with fitz.open(result.pdf) as doc:
        page = doc[result.page_no - 1]
        bg_xref, _, _ = _find_mrc_layers(doc, page)
        if bg_xref is None:
            return
        bg = _pixmap_gray(doc, bg_xref)

    low, high = np.percentile(bg, [1, 99.5])
    stretched = np.clip((bg.astype(np.float32) - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
    for blob in result.blobs:
        cv2.rectangle(vis, (blob.x - 3, blob.y - 3), (blob.x + blob.w + 3, blob.y + blob.h + 3), (0, 0, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def _worker(pdf_path: Path) -> list[PageResult]:
    return analyze_pdf(pdf_path)


@click.command()
@click.option(
    "--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Папка с PDF"
)
@click.option("--recursive/--no-recursive", default=False, help="Искать PDF во вложенных папках")
@click.option("--report-csv", type=click.Path(path_type=Path), default=None, help="Куда сохранить CSV-отчёт")
@click.option("--debug-dir", type=click.Path(path_type=Path), default=None, help="Папка для картинок с разметкой")
@click.option("--debug-top", type=int, default=30, help="Сколько худших страниц отрисовать в --debug-dir")
@click.option("--min-blobs", type=int, default=8, help="Порог: со скольких пятен считать страницу дефектной")
@click.option("--jobs", type=int, default=0, help="Число процессов (0 — по числу ядер)")
def main(
    input_dir: Path,
    recursive: bool,
    report_csv: Path | None,
    debug_dir: Path | None,
    debug_top: int,
    min_blobs: int,
    jobs: int,
) -> int:
    """Найти страницы, где часть текста осталась в фоновом слое MRC."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not pdfs:
        click.echo(f"В {input_dir} не найдено PDF-файлов", err=True)
        return 1

    jobs = jobs or mp.cpu_count()
    results: list[PageResult] = []
    if jobs > 1 and len(pdfs) > 1:
        with mp.Pool(min(jobs, len(pdfs))) as pool:
            for page_results in tqdm(pool.imap(_worker, pdfs), total=len(pdfs), desc="PDF"):
                results.extend(page_results)
    else:
        for pdf in tqdm(pdfs, desc="PDF"):
            results.extend(analyze_pdf(pdf))

    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status != "ok"]
    ok.sort(key=lambda r: (r.severity, r.n_blobs), reverse=True)

    if report_csv:
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        with report_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["pdf", "page", "severity", "n_blobs", "left", "right", "blob_area", "max_blob", "status"])
            for r in ok + skipped:
                writer.writerow(
                    [
                        r.pdf.name,
                        r.page_no,
                        f"{r.severity:.1f}",
                        r.n_blobs,
                        r.left_blobs,
                        r.right_blobs,
                        r.blob_area,
                        r.max_blob_area,
                        r.status,
                    ]
                )
        click.echo(f"Отчёт: {report_csv}")

    if debug_dir:
        for r in tqdm(ok[:debug_top], desc="Отладочные картинки"):
            save_debug_image(r, Path(debug_dir) / f"{r.severity:07.1f}_{r.pdf.stem}_p{r.page_no:03d}.png")
        click.echo(f"Картинки: {debug_dir}")

    suspect = [r for r in ok if r.n_blobs >= min_blobs]
    click.echo(f"\nПроанализировано страниц: {len(ok)} (пропущено не-MRC: {len(skipped)})")
    click.echo(f"Подозрительных (>= {min_blobs} пятен): {len(suspect)}\n")
    for r in suspect[:40]:
        side = "лев" if r.left_blobs > r.right_blobs else "прав"
        click.echo(f"  {r.pdf.name} стр.{r.page_no:>4}  severity={r.severity:7.1f}  пятен={r.n_blobs:>4} ({side})")
    if len(suspect) > 40:
        click.echo(f"  ... ещё {len(suspect) - 40}, полный список в CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
