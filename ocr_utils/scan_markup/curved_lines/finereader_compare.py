"""Пары «было | стало»: исходная полоса против страницы, в которую её превратил FineReader.

ЗАЧЕМ. Детектор отметил полосы с кривыми строками; вопрос пользователя — хорошо ли на
них сработало распрямление строк FineReader. Ответ можно дать только глазами, поэтому
на каждую находку собирается одна картинка: слева полоса, справа страница PDF, обе с
реперными горизонталями через 1/12 высоты — по ним видно, легли ли строки ровно.

СООТВЕТСТВИЕ СТРАНИЦ. Промежуточный PDF выпуска собран из полос в порядке имён файлов в
папке выпуска, и FineReader этот порядок сохранил: полоса с индексом k в отсортированном
списке — страница k+1 в PDF. Перед прогоном число страниц PDF сверяется с числом полос
в папке; несовпадение — выпуск пропускается с сообщением, а не выравнивается наугад.
FineReader обработал не все выпуски — для полос без PDF пары нет.
"""

from __future__ import annotations

import logging
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import click
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.scan_markup.scan_tree import issue_images

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

DEFAULT_DPI = 150
GUIDES = 12
GAP_PX = 12
CAPTION_PX = 36
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PDF_NAME = "full_{year}_{issue}.pdf"


@dataclass(frozen=True)
class Pair:
    rel_path: str  # год/выпуск/полоса.tif относительно пака
    source: Path  # что показывать слева (заострённая копия или оригинал)
    pdf: Path
    page_index: int  # с нуля
    out_path: Path


def _issue_index(pack_root: Path, rel_path: str) -> tuple[str, str, int | None]:
    relative = Path(rel_path)
    year, issue = relative.parts[0], relative.parts[1]
    names = [p.name for p in issue_images(pack_root / year / issue)]
    try:
        return year, issue, names.index(relative.name)
    except ValueError:
        return year, issue, None


def _page_count(pdf: Path) -> int:
    import fitz

    with fitz.open(pdf) as document:
        return document.page_count


def plan(
    links: Sequence[Path], pack_root: Path, source_root: Path | None, pdf_dir: Path, out_dir: Path
) -> tuple[list[Pair], dict[str, int], list[str]]:
    """Пары для каждого симлинка; выпуски без PDF и с расхождением числа страниц — в отчёт."""
    pairs: list[Pair] = []
    skipped: dict[str, int] = {}
    notes: list[str] = []
    checked: dict[Path, bool] = {}
    for link in links:
        target = link.resolve()
        try:
            rel_path = target.relative_to(pack_root.resolve()).as_posix()
        except ValueError:
            notes.append(f"{link.name}: цель {target} не под {pack_root}")
            continue
        year, issue, index = _issue_index(pack_root, rel_path)
        pdf = pdf_dir / PDF_NAME.format(year=year, issue=issue)
        if not pdf.is_file():
            skipped[f"{year}/{issue}"] = skipped.get(f"{year}/{issue}", 0) + 1
            continue
        if index is None:
            notes.append(f"{rel_path}: полосы нет в списке выпуска")
            continue
        if pdf not in checked:
            n_files = len(issue_images(pack_root / year / issue))
            n_pages = _page_count(pdf)
            checked[pdf] = n_files == n_pages
            if not checked[pdf]:
                notes.append(f"{pdf.name}: страниц {n_pages}, полос в папке {n_files} — выпуск пропущен")
        if not checked[pdf]:
            skipped[f"{year}/{issue} (расхождение)"] = skipped.get(f"{year}/{issue} (расхождение)", 0) + 1
            continue
        source = target
        if source_root is not None:
            candidates = sorted((source_root / year / issue).glob(f"{Path(rel_path).stem}.*"))
            if candidates:
                source = candidates[0]
        pairs.append(Pair(rel_path, source, pdf, index, out_dir / (link.name.rsplit(".", 1)[0] + ".jpg")))
    return pairs, skipped, notes


def _load_source(path: Path, height: int) -> Image.Image:
    with Image.open(path) as image:
        image.draft("L", (image.width // 4, image.height // 4))
        gray = image.convert("L")
        scale = height / gray.height
        return gray.resize((max(1, round(gray.width * scale)), height), Image.LANCZOS)


def _render_pdf_page(pdf: Path, index: int, dpi: int) -> Image.Image:
    import fitz

    with fitz.open(pdf) as document:
        page = document[index]
        zoom = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
        return Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)


def compose(before: Image.Image, after: Image.Image, caption_left: str, caption_right: str) -> Image.Image:
    """Две панели одной высоты, реперные горизонтали через обе, подписи сверху."""
    height = after.height
    if before.height != height:
        before = before.resize((max(1, round(before.width * height / before.height)), height), Image.LANCZOS)
    width = before.width + GAP_PX + after.width
    canvas = Image.new("RGB", (width, height + CAPTION_PX), (255, 255, 255))
    canvas.paste(before.convert("RGB"), (0, CAPTION_PX))
    canvas.paste(after.convert("RGB"), (before.width + GAP_PX, CAPTION_PX))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([before.width, CAPTION_PX, before.width + GAP_PX - 1, height + CAPTION_PX], fill=(120, 120, 120))
    for step in range(1, GUIDES):
        y = CAPTION_PX + height * step // GUIDES
        draw.line([(0, y), (width, y)], fill=(255, 0, 0), width=1)
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 8), caption_left, fill=(0, 0, 0), font=font)
    draw.text((before.width + GAP_PX + 6, 8), caption_right, fill=(0, 0, 0), font=font)
    return canvas


def _write_pair(args: tuple[Pair, int]) -> str:
    pair, dpi = args
    try:
        after = _render_pdf_page(pair.pdf, pair.page_index, dpi)
        before = _load_source(pair.source, after.height)
        image = compose(before, after, f"было: {pair.rel_path}", f"стало: {pair.pdf.name}, стр. {pair.page_index + 1}")
        pair.out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(pair.out_path, quality=85)
    except Exception as error:
        return f"{pair.rel_path}: {error}"
    return ""


def write_pairs(pairs: Sequence[Pair], dpi: int, jobs: int) -> list[str]:
    from tqdm import tqdm

    tasks = [(pair, dpi) for pair in pairs]
    if jobs <= 1 or len(tasks) <= 1:
        outcomes = map(_write_pair, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"))
        outcomes = pool.map(_write_pair, tasks)
    return [outcome for outcome in tqdm(outcomes, total=len(tasks), desc="пары", unit="полоса") if outcome]


def _sort_key(name: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--links-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="каталог симлинков детектора (например combo/)",
)
@click.option(
    "--pack-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="пак: год/выпуск/полоса — сюда целят симлинки",
)
@click.option(
    "--source-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="откуда брать левую панель (заострённые копии); по умолчанию — цель симлинка",
)
@click.option(
    "--pdf-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="PDF после FineReader, full_ГГГГ_НН.pdf",
)
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--md-report", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--dpi", default=DEFAULT_DPI, show_default=True, type=int, help="разрешение рендера страницы PDF и панелей"
)
@click.option("--jobs", default=8, show_default=True, type=int)
@click.option("--limit", type=int)
def main(links_dir, pack_root, source_root, pdf_dir, out_dir, md_report, dpi, jobs, limit) -> None:
    """Пары «было | стало» для находок детектора: полоса против страницы FineReader."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    links = sorted((p for p in links_dir.iterdir() if p.is_symlink()), key=lambda p: _sort_key(p.name))
    if limit:
        links = links[:limit]
    pairs, skipped, notes = plan(links, pack_root, source_root, pdf_dir, out_dir)
    click.echo(f"Симлинков: {len(links)}. Пар с PDF: {len(pairs)}. Без PDF: {sum(skipped.values())}.")
    errors = write_pairs(pairs, dpi, jobs)
    for error in errors[:20]:
        click.echo(f"  {error}")
    lines = [
        f"# Было | стало: находки детектора против страниц FineReader",
        "",
        f"Симлинков: {len(links)}. Пар записано: {len(pairs) - len(errors)}. Ошибок: {len(errors)}. "
        f"Полос без PDF: {sum(skipped.values())}.",
        "",
        f"Пары: `{out_dir}`; PDF: `{pdf_dir}`; рендер {dpi} dpi.",
        "",
        "## Выпуски без PDF (FineReader их не обрабатывал)",
        "",
        "| выпуск | находок |",
        "|---|---|",
        *(f"| {issue} | {count} |" for issue, count in sorted(skipped.items())),
    ]
    if notes:
        lines += ["", "## Замечания", "", *(f"* {note}" for note in notes)]
    if errors:
        lines += ["", "## Ошибки", "", *(f"* {error}" for error in errors)]
    text = "\n".join(lines) + "\n"
    if md_report is not None:
        md_report.write_text(text, encoding="utf-8")
        click.echo(f"Отчёт: {md_report}")
    click.echo("\n".join(lines[6:]))


if __name__ == "__main__":
    main()
