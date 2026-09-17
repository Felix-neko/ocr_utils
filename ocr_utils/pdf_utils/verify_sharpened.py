"""Сверка заострённых копий с их полосами: та ли картинка лежит под этим именем.

ЗАЧЕМ. Capture One обрабатывает выпуски пачками и параллельно, а имена полос в разных
выпусках ПОВТОРЯЮТСЯ: ``IMG_0034_1L`` есть в тридцати выпусках пака-1. Однажды это уже
кончилось тем, что в ``1975/04/IMG_0034_1L.jpg`` легла полоса из ``1968/07`` — с чужим
текстом и чужими размерами.

Размер ловит такую подмену только между выпусками РАЗНОЙ геометрии. Внутри одинаковых
(а таких большинство: пак снимался одной серией) подменённый файл по размеру не отличить
никак, и заметить его можно было бы разве что глазами в готовом PDF. Поэтому проверка
идёт по СТРУКТУРЕ полосы.

КАК УСТРОЕНА ПРОВЕРКА. Мер две, и ни одной по отдельности не хватает.

* ПРОФИЛИ — «доля тёмного» по строкам и по столбцам. Они ловят рисунок полосы: где
  заголовок, где абзацные отступы, где картинка, где пустой низ. На тексте это очень
  сильный признак, но на полосе, занятой одной фотографией, профили почти плоские, и
  корреляция начинает мерить шум.
* ПИКСЕЛИ — корреляция полос, приведённых к 64x64. На тексте она вырождается (все полосы
  на эскизе одинаково серые), зато на фотографии и обложке работает.

Решение принимает МАКСИМУМ из двух: своей паре достаточно совпасть хоть по одному
признаку, а чужая проваливает оба. Замер по всему паку-1 (12 135 полос): у подменённой
полосы меры дали 0.20 и 0.35, то есть максимум 0.35; у худшей ИЗ СВОИХ — 0.78; у обычной
— 0.92-0.99. Между 0.35 и 0.78 провал, и порог 0.6 стоит в его середине.

Обе меры нормируются: заострение сильно меняет ТОН (обложки Capture One вообще
обесцветил и вывернул в жёсткий контраст), и сравнивать надо форму, а не уровень —
иначе метрика мерила бы силу фильтра, а не совпадение полос.

Полосы из полосы 0.6-0.8 подменой НЕ считаются, но выводятся отдельным списком «глянуть
глазами»: там оседают обложки и почти пустые полосы, у которых обе меры законно просели.

Запуск::

    python -m ocr_utils.pdf_utils.verify_sharpened --db base.sqlite --pack-name пак-1 \\
        --originals-dir .../blurred --sharpened-dir .../sharpened
"""

import csv
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
from PIL import Image
from tqdm import tqdm

from ocr_utils.pdf_utils.jpeg_pdf import read_jpeg_info

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

DEFAULT_JOBS = 16
# Сторона, к которой приводится полоса перед снятием профилей. 256 выбрано замером на
# паке-1: на 64 метрика вырождается (все текстовые полосы дают одинаковые 15 единиц
# разницы), на 256 своя пара уверенно отделяется от чужой.
PROFILE_SIDE = 256
# Сторона для ПИКСЕЛЬНОЙ меры. Мельче профильной намеренно: на 64x64 текст размывается в
# ровный серый (потому одной этой меры и не хватает), зато фотография и обложка остаются
# узнаваемыми, а мелкие расхождения заострения уже не мешают.
PIXEL_SIDE = 64
# Ниже этой корреляции (по максимуму из двух мер) пара считается разъехавшейся. Замер по
# всему паку-1: подменённая полоса 0.35, худшая ИЗ СВОИХ 0.72. Порог в середине провала.
#
# Он намеренно НЕ поднят ближе к своим: цена ложной тревоги — ручная проверка одной полосы,
# а цена пропуска — чужая страница в готовом документе, которую уже никто не заметит.
# Поэтому всё, что не дотянуло до DEFAULT_SUSPECT, показывается отдельным списком.
DEFAULT_THRESHOLD = 0.6

# Полоса «вроде своя, но просела»: обложки, перекрашенные Capture One до неузнаваемости, и
# почти пустые полосы, где обе меры вырождаются. Прогон из-за них не падает.
#
# 0.8, а не 0.9: по паку-1 порог 0.9 даёт 171 полосу — столько глазами никто смотреть не
# станет, и список перестаёт работать. При 0.8 остаётся горстка тех, что ближе всего к
# провалу, а это и есть то, ради чего список нужен.
DEFAULT_SUSPECT = 0.8
# Сколько байт от начала JPEG читать ради размеров. Мегабайт: у Capture One перед кадром
# лежат EXIF, миниатюра и ICC-профиль, и в первые килобайты маркер SOF не попадает.
JPEG_HEADER_BYTES = 1 << 20


@dataclass(frozen=True)
class PageCheck:
    """Что проверяем по одной полосе."""

    rel_path: str
    original: Path
    sharpened: Path
    width: int
    height: int


@dataclass
class CheckResult:
    """Итог по одной полосе."""

    rel_path: str
    status: str = "ok"  # ok | missing | size | content | suspect | error
    reason: str = ""
    correlation: float = float("nan")  # максимум из двух мер, по нему и решение
    profile_correlation: float = float("nan")
    pixel_correlation: float = float("nan")
    sharpened_size: str = ""


@dataclass
class VerifyStats:
    ok: int = 0
    missing: int = 0
    size: int = 0
    content: int = 0
    suspect: int = 0
    failed: int = 0
    bad: "list[CheckResult]" = field(default_factory=list)

    @property
    def broken(self) -> int:
        """Сколько полос действительно негодны. ``suspect`` сюда не входит: это к глазам."""
        return self.missing + self.size + self.content + self.failed

    def summary(self) -> str:
        return (
            f"сошлось {self.ok}; нет копии {self.missing}; разошёлся размер {self.size}; "
            f"ЧУЖАЯ ПОЛОСА {self.content}; глянуть глазами {self.suspect}; ошибок чтения {self.failed}"
        )


def _gray(path: Path, side: int) -> np.ndarray:
    """Полоса в оттенках серого, приведённая к ``side`` x ``side`` и нормированная.

    ``draft`` просит JPEG-декодер сразу выдать уменьшенную картинку — он умеет это делать
    прямо из DCT, кратно 1/2, 1/4, 1/8, и на полосе 3840x6692 экономит почти всё время
    декодирования. Для TIFF это не работает, и его приходится разжимать целиком.
    """
    with Image.open(path) as image:
        if image.format == "JPEG":
            image.draft("L", (side * 2, side * 2))
        array = np.asarray(image.convert("L").resize((side, side))).astype(np.float32)
    # Нормируем: заострение меняет тон и контраст, а сравнивать надо форму полосы.
    return (array - array.mean()) / (array.std() + 1e-6)


def structure_profile(path: Path, side: int = PROFILE_SIDE) -> np.ndarray:
    """Профили заполненности полосы по строкам и столбцам, нормированные."""
    array = _gray(path, side)
    return np.concatenate([array.mean(1), array.mean(0)])


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Корреляция двух наборов; NaN у вырожденного (полностью ровного) — это 0."""
    value = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return value if np.isfinite(value) else 0.0


def compare(original: Path, sharpened: Path) -> "tuple[float, float]":
    """Две меры сходства полосы и её заострённой копии: по профилям и по пикселям."""
    profiles = _correlation(structure_profile(original), structure_profile(sharpened))
    pixels = _correlation(_gray(original, PIXEL_SIDE), _gray(sharpened, PIXEL_SIDE))
    return profiles, pixels


def check_page(page: PageCheck, threshold: float = DEFAULT_THRESHOLD, suspect: float = DEFAULT_SUSPECT) -> CheckResult:
    """Проверяет одну полосу: есть ли копия, того ли она размера и та ли это полоса."""
    if not page.sharpened.exists():
        return CheckResult(page.rel_path, "missing", "нет заострённой копии")
    try:
        with page.sharpened.open("rb") as handle:
            info = read_jpeg_info(handle.read(JPEG_HEADER_BYTES))
        size = f"{info.width}x{info.height}"
        if (info.width, info.height) != (page.width, page.height):
            return CheckResult(
                page.rel_path, "size", f"копия {size}, а полоса {page.width}x{page.height}", sharpened_size=size
            )
        profiles, pixels = compare(page.original, page.sharpened)
    except Exception as exc:  # noqa: BLE001 — одна битая полоса не должна ронять прогон
        return CheckResult(page.rel_path, "error", str(exc))

    # Максимум, а не среднее: своей паре достаточно совпасть хоть по одному признаку —
    # на тексте работают профили, на фотографии пиксели, — а чужая проваливает оба.
    score = max(profiles, pixels)
    reason = f"профили {profiles:.3f}, пиксели {pixels:.3f}"
    status = "ok"
    if score < threshold:
        status = "content"
    elif score < suspect:
        status = "suspect"
    return CheckResult(page.rel_path, status, reason if status != "ok" else "", score, profiles, pixels, size)


def _run_job(job: "tuple[PageCheck, float, float]") -> CheckResult:
    return check_page(*job)


def load_pages(db_path: Path, pack_name: str, originals_dir: Path, sharpened_dir: Path) -> "list[PageCheck]":
    """Полосы пака с путями к обеим версиям; база читается один раз, в родителе."""
    from ocr_utils.db.repo import iter_pages, require_pack
    from ocr_utils.db.session import open_db

    pages: "list[PageCheck]" = []
    with open_db(db_path, create=False)() as session:
        pack = require_pack(session, pack_name)
        for _year, _issue, page in iter_pages(pack):
            rel = page.sharpened_text_pic_rel_path or Path(page.source_rel_path).with_suffix(".jpg").as_posix()
            pages.append(
                PageCheck(
                    rel_path=page.source_rel_path,
                    original=originals_dir / page.source_rel_path,
                    sharpened=sharpened_dir / rel,
                    width=int(page.width or 0),
                    height=int(page.height or 0),
                )
            )
    return pages


def run_verify(
    pages: "list[PageCheck]",
    threshold: float = DEFAULT_THRESHOLD,
    suspect: float = DEFAULT_SUSPECT,
    jobs: int = DEFAULT_JOBS,
    progress: bool = True,
) -> VerifyStats:
    """Сверяет все полосы пулом процессов; работа CPU-интенсивная (декодирование)."""
    stats = VerifyStats()
    jobs_list = [(page, threshold, suspect) for page in pages]
    bar = tqdm(
        total=len(jobs_list),
        disable=not progress,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    with bar:
        if jobs > 1 and len(jobs_list) > 1:
            with ProcessPoolExecutor(max_workers=min(jobs, len(jobs_list))) as pool:
                results = []
                for result in pool.map(_run_job, jobs_list, chunksize=4):
                    bar.update(1)
                    results.append(result)
        else:
            results = []
            for job in jobs_list:
                bar.update(1)
                results.append(_run_job(job))

    # Каждому нештатному исходу — свой счётчик; имена полей повторяют статусы, кроме
    # "error": в отчёте он читается как "ошибок чтения".
    counters = {"missing": "missing", "size": "size", "content": "content", "suspect": "suspect", "error": "failed"}
    for result in results:
        if result.status == "ok":
            stats.ok += 1
            continue
        stats.bad.append(result)
        field_name = counters[result.status]
        setattr(stats, field_name, getattr(stats, field_name) + 1)
    return stats


def write_report(path: Path, results: "list[CheckResult]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rel_path", "status", "reason", "score", "profiles", "pixels", "sharpened_size"])
        for row in results:
            writer.writerow(
                [
                    row.rel_path,
                    row.status,
                    row.reason,
                    f"{row.correlation:.4f}",
                    f"{row.profile_correlation:.4f}",
                    f"{row.pixel_correlation:.4f}",
                    row.sharpened_size,
                ]
            )


@click.command()
@click.option("--db", "db_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--pack-name", required=True, help="Имя пака в базе.")
@click.option("--originals-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--threshold",
    default=DEFAULT_THRESHOLD,
    show_default=True,
    type=float,
    help="Ниже этого сходства пара считается разъехавшейся (по максимуму из двух мер).",
)
@click.option(
    "--suspect",
    default=DEFAULT_SUSPECT,
    show_default=True,
    type=float,
    help="Ниже этого — в список «глянуть глазами»; прогон из-за таких не падает.",
)
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True, type=int, help="Процессов на декодирование.")
@click.option(
    "--report-csv", default=None, type=click.Path(dir_okay=False, path_type=Path), help="Куда положить отчёт."
)
@click.option("--no-progress", is_flag=True)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def main(
    db_path: Path,
    pack_name: str,
    originals_dir: Path,
    sharpened_dir: Path,
    threshold: float,
    suspect: float,
    jobs: int,
    report_csv: "Path | None",
    no_progress: bool,
    log_level: str,
) -> None:
    """Проверяет, что под каждым именем в заострённых копиях лежит своя полоса."""
    logging.basicConfig(level=log_level.upper(), format="%(levelname)s: %(message)s")
    pages = load_pages(db_path, pack_name, originals_dir, sharpened_dir)
    stats = run_verify(pages, threshold=threshold, suspect=suspect, jobs=jobs, progress=not no_progress)

    # Сначала негодные, потом «глянуть глазами»: иначе первые тонут во вторых.
    for result in stats.bad:
        if result.status != "suspect":
            click.echo(f"{result.status.upper():8s} {result.rel_path}: {result.reason}")
    for result in stats.bad:
        if result.status == "suspect":
            click.echo(f"глянуть  {result.rel_path}: {result.reason}")
    click.echo(stats.summary())
    if report_csv is not None:
        write_report(report_csv, stats.bad)
        click.echo(f"отчёт: {report_csv}")
    # Падаем только на настоящих бедах: «глянуть глазами» — это не повод остановить конвейер.
    raise SystemExit(1 if stats.broken else 0)


if __name__ == "__main__":
    main()
