"""Промежуточные PDF по выпускам — то, что уходит в пакетное распознание FineReader.

По каждому выпуску собираются две PDF:

* ``full_{год}_{выпуск}.pdf`` — все полосы. Её в FineReader гонят через бинаризацию И
  распрямление строк: так лучше распознаётся текст, но геометрия страницы после этого
  уже не та, что была;
* ``pages_with_pics_only_{год}_{выпуск}.pdf`` — только полосы, на которых размечены
  иллюстрации или цветной набор. Её гонят через бинаризацию БЕЗ распрямления, чтобы
  геометрия осталась прежней и в финальную PDF можно было вернуть иллюстрации ровно на
  их места.

ЧТО КЛАДЁТСЯ НА СТРАНИЦУ. Основа — заострённая копия полосы (Capture One усилил детали
текста), и её байты в PDF не перекодируются вовсе: см. ``pdf_utils.jpeg_pdf``. Если на
полосе размечены иллюстрации, поверх основы кладутся соответствующие куски ОРИГИНАЛА:
заострение вытягивает штрих и тем самым портит полутоновый растр, а иллюстрация в итоговом
документе должна остаться такой, какой была. Текстовая часть страницы при этом не
перекодируется — врезка кладётся отдельной картинкой, а не впечатывается в основу.

Полоса, у которой размеченная иллюстрация занимает ВЕСЬ кадр (обложка, вкладка), собирается
из одного оригинала: заострённая копия под ней всё равно не видна ни одним пикселем, и
встраивать её значило бы добавить к выпуску десяток лишних мегабайт на каждую такую полосу.

ПОЛЯ (``--page-margin-x-mm`` / ``--page-margin-y-mm``). Они добавляются ТОЛЬКО в полную PDF.
Распрямляя строки, FineReader увеличивает кадр, но MediaBox оставляет прежним и сажает кадр
в левый нижний угол — всё, что вылезло вправо и вверх, обрезается краем страницы. По паку-1
это 38 % полос, обычно 2–3 мм вправо и 6 мм вверх, в худшем случае 12,7 мм, и режет оно уже
не поля, а концы строк: чаще всего гибнет дефис переноса и последняя буква. Поле шириной
с этот вылет даёт распрямлению куда расти, и обрезается белое, а не текст.

В PDF типа PAGES_WITH_PICS_ONLY полей НЕТ и быть не должно: её гонят без распрямления,
кадр там не растёт, а лишние поля сдвинули бы иллюстрации относительно разметки.

Поля вставляются БЕЗ перекодирования исходного JPEG — переносом DCT-блоков, см.
``pdf_utils.padding``. Цвет заливки берётся не белый, а оценённый цвет бумаги этой самой
полосы, слегка осветлённый: сплошное белое поле рядом с сероватой бумагой — это ступенька
яркости у самого края, а по краю кадра как раз и работают автояркость с бинаризацией.

ПАМЯТЬ. Единица работы пула — выпуск целиком, и до записи он держится в памяти: сотня полос
по 12 МиБ — это порядка полутора гигабайт на воркер. При ``--jobs 8`` закладывайте до
двадцати гигабайт; если памяти меньше, убавляйте ``--jobs``, а не размер выпуска.

Запуск::

    python -m ocr_utils.pdf_utils.intermediate_pdfs --db base.sqlite --pack-name пак-1 \\
        --originals-dir .../blurred --sharpened-dir .../sharpened \\
        --full-pdf-dir .../full_intermediate_pdfs \\
        --pics-only-pdf-dir .../intermediate_pdfs_pages_with_pics_only \\
        --page-margin-x-mm 12 --page-margin-y-mm 6
"""

import logging
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import click
import cv2
import numpy as np
import pikepdf
from PIL import Image
from tqdm import tqdm

from ocr_utils.pdf_utils.jpeg_pdf import Overlay, crop, encode_jpeg, load_image, make_page, read_jpeg_info
from ocr_utils.pdf_utils.padding import (
    LosslessPaddingError,
    brighten_color,
    estimate_paper_color,
    pad_jpeg_lossless_xy,
    set_jpeg_dpi,
)
from ocr_utils.db.models import COLOR_PICTURE_KINDS, PICTURE_KINDS
from ocr_utils.scan_markup.rotation import rotate_box, rotate_size

logger = logging.getLogger(__name__)

DEFAULT_JOBS = 8
# Качество JPEG для врезок в ПРОМЕЖУТОЧНОЙ PDF. Выше, чем 75 у финальной, намеренно: этот
# файл — полуфабрикат, из него ничего не собирается заново, а вот глазами по нему проверяют,
# на месте ли иллюстрации, и артефакты сжатия тут только мешают смотреть.
DEFAULT_PICTURE_QUALITY = 90

# Поля полной PDF, в миллиметрах: слева и справа, сверху и снизу. Числа не с потолка —
# это замер вылета распрямлённого кадра за MediaBox по 6951 полосе пака-1: вправо p95 = 5,3
# мм и p99 = 8,4 мм, вверх p95 = 7,5 мм и p99 = 9,4 мм. Полем в 12 мм по горизонтали и 6 мм
# по вертикали режется около 4 % полос — почти все за счёт вертикали, у которой прирост
# почти постоянный, 6–9 мм. Если после пробного прогона окажется, что верх всё ещё режет,
# первым делом поднимать ``--page-margin-y-mm`` до 10, а не горизонталь.
DEFAULT_MARGIN_X_MM = 12.0
DEFAULT_MARGIN_Y_MM = 6.0

# На сколько тонов из 256 осветлить оценённый цвет бумаги под заливку полей. Ровно цвет
# бумаги брать не стоит: поле сплошное, а бумага в кадре пёстрая, и поле того же среднего
# тона читается как грязное пятно по краю. Осветление на полтона делает его чуть светлее
# самой светлой бумаги — тогда оно уходит в фон и для глаза, и для бинаризации.
DEFAULT_FILL_BRIGHTEN = 12

# По какой стороне уменьшается копия полосы для оценки цвета бумаги. ``Image.draft``
# распаковывает JPEG прямо из DCT-коэффициентов в уменьшенном виде — это в разы дешевле
# полного декодирования, а цвет бумаги от масштаба не зависит. Мельче брать нельзя: строки
# текста сольются в серую массу и утянут оценку в тень.
PAPER_ANALYSIS_MAX_SIDE = 1024

# С какого запаса прямоугольник считается покрывающим полосу целиком. Полпроцента стороны:
# у полосных иллюстраций пака-1 рамка совпадает с кадром точь-в-точь, и допуск нужен только
# на случай разметки, сделанной руками по уменьшенной копии.
#
# Флаг ``full_page`` из базы для этого НЕ ГОДИТСЯ: он ставится при площади от 90% кадра, и
# полоса с такой рамкой осталась бы с незакрытой десятой частью листа.
COVER_MARGIN_FRAC = 0.005

# В имя файла PDF пускаются только эти знаки: FineReader разбирает папку пакетом, и пробел
# или скобка из имени переscanного выпуска ``05 (2)`` там ни к чему.
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_name(value: str) -> str:
    """Кусок имени файла: всё небезопасное схлопывается в подчёркивание."""
    return _UNSAFE_NAME_RE.sub("_", value).strip("_") or "x"


@dataclass(frozen=True)
class PicturePlan:
    """Размеченная иллюстрация: место в пикселях оригинала и цветность."""

    x1: int
    y1: int
    x2: int
    y2: int
    kind: str

    @property
    def rect(self) -> "tuple[int, int, int, int]":
        return self.x1, self.y1, self.x2, self.y2

    @property
    def gray(self) -> bool:
        return self.kind not in COLOR_PICTURE_KINDS

    def covers(self, width: int, height: int) -> bool:
        """Занимает ли прямоугольник весь кадр (с допуском :data:`COVER_MARGIN_FRAC`)."""
        margin_x, margin_y = width * COVER_MARGIN_FRAC, height * COVER_MARGIN_FRAC
        return (
            self.x1 <= margin_x and self.y1 <= margin_y and self.x2 >= width - margin_x and self.y2 >= height - margin_y
        )


@dataclass(frozen=True)
class PagePlan:
    """Полоса и всё, что нужно, чтобы положить её на страницу PDF."""

    page_id: int
    # Путь ОЧИЩЕННОЙ полосы относительно --originals-dir: из неё режутся иллюстрации.
    # Имя может отличаться от исходного отпечатком, см. ``scan_cleanup.naming``.
    original_rel_path: str
    sharpened_rel_path: str
    width: int
    height: int
    dpi: int
    pictures: "tuple[PicturePlan, ...]"
    full_pdf_page_idx: int
    pages_with_pics_only_pdf_page_idx: "int | None"
    # На сколько ПО ЧАСОВОЙ повёрнуты файлы этой полосы на диске. ``width``/``height`` и
    # координаты врезок описывают ОРИГИНАЛ, поэтому и то и другое пересчитывается здесь.
    rotate_cw: int = 0

    @property
    def file_size(self) -> "tuple[int, int]":
        """Размер файлов на диске: у повёрнутой полосы стороны поменяны местами."""
        return rotate_size(self.width, self.height, self.rotate_cw)

    def placed_pictures(self) -> "tuple[PicturePlan, ...]":
        """Врезки в координатах ФАЙЛА. Без пересчёта легли бы на другое место полосы."""
        if not self.rotate_cw % 360:
            return self.pictures
        return tuple(
            PicturePlan(*rotate_box((p.x1, p.y1, p.x2, p.y2), self.width, self.height, self.rotate_cw), p.kind)
            for p in self.pictures
        )


@dataclass(frozen=True)
class IssuePlan:
    """Выпуск целиком — одна задача пула."""

    issue_id: int
    year_name: str
    issue_name: str
    pages: "tuple[PagePlan, ...]"

    @property
    def rel_path(self) -> str:
        return f"{self.year_name}/{self.issue_name}"

    @property
    def full_pdf_name(self) -> str:
        return f"full_{safe_name(self.year_name)}_{safe_name(self.issue_name)}.pdf"

    @property
    def pics_pdf_name(self) -> "str | None":
        if not any(page.pages_with_pics_only_pdf_page_idx is not None for page in self.pages):
            return None
        return f"pages_with_pics_only_{safe_name(self.year_name)}_{safe_name(self.issue_name)}.pdf"


@dataclass
class IssueReport:
    """Что случилось с одним выпуском."""

    rel_path: str
    issue_id: int
    status: str = "ok"  # ok | skipped | error
    reason: str = ""
    pages: int = 0
    pages_with_pics: int = 0
    # Поля полной PDF, фактически применённые (в миллиметрах). Не то же самое, что
    # запрошенное: ширина округляется вверх до кратной размеру MCU, а он зависит от
    # субдискретизации конкретного JPEG. Заполняются только при status == "ok" —
    # у пропущенного выпуска файлы не трогали и знать про них нечего.
    margin_left_mm: "float | None" = None
    margin_right_mm: "float | None" = None
    margin_top_mm: "float | None" = None
    margin_bottom_mm: "float | None" = None


@dataclass
class BuildParams:
    """Параметры прогона, едущие в воркер вместе с планом выпуска."""

    originals_dir: Path
    sharpened_dir: Path
    full_pdf_dir: Path
    pics_only_pdf_dir: Path
    picture_quality: int = DEFAULT_PICTURE_QUALITY
    skip_if_exists: bool = True
    # Поля ТОЛЬКО полной PDF; в PAGES_WITH_PICS_ONLY их нет никогда — см. шапку модуля.
    margin_x_mm: float = DEFAULT_MARGIN_X_MM
    margin_y_mm: float = DEFAULT_MARGIN_Y_MM
    fill_brighten: int = DEFAULT_FILL_BRIGHTEN
    # Разрешить пересжатие полосы, если беспотерьная вставка полей не вышла. По умолчанию
    # запрещено: весь смысл промежуточной PDF в том, что байты заострённой копии едут в неё
    # как есть, и тихо потерять это на паре полос хуже, чем упасть с внятной ошибкой.
    allow_jpeg_recode: bool = False


@dataclass
class BuildStats:
    """Итоги прогона."""

    issues: int = 0
    skipped: int = 0
    failed: int = 0
    pages: int = 0
    pages_with_pics: int = 0
    reports: "list[IssueReport]" = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"выпусков собрано {self.issues}, пропущено {self.skipped}, с ошибкой {self.failed}; "
            f"полос {self.pages}, из них с иллюстрациями {self.pages_with_pics}"
        )


MM_PER_INCH = 25.4


def _paper_fill(jpeg: bytes, brighten: int) -> "tuple[int, int, int]":
    """Цвет заливки полей: оценённый цвет бумаги этой полосы, слегка осветлённый."""
    with Image.open(BytesIO(jpeg)) as source:
        source.draft("RGB", (PAPER_ANALYSIS_MAX_SIDE, PAPER_ANALYSIS_MAX_SIDE))
        preview = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    return brighten_color(estimate_paper_color(preview), brighten)


def pad_page_jpeg(
    jpeg: bytes, margin_x_px: int, margin_y_px: int, dpi: int, brighten: int, allow_recode: bool, label: str
) -> "tuple[bytes, int, int]":
    """Добавить полосе поля, по возможности не перекодируя её.

    Args:
        jpeg: Байты полосы
        margin_x_px: Поле слева и справа в пикселях
        margin_y_px: Поле сверху и снизу в пикселях
        dpi: Разрешение, которое проставить результату
        brighten: На сколько тонов осветлить цвет бумаги под заливку
        allow_recode: Разрешено ли пересжатие, если беспотерьная вставка не вышла
        label: Как назвать полосу в сообщении об ошибке

    Returns:
        Тройка (байты результата, фактическое поле по X, фактическое поле по Y)

    Raises:
        LosslessPaddingError: Беспотерьная вставка не вышла, а пересжатие не разрешено
    """
    if margin_x_px <= 0 and margin_y_px <= 0:
        return jpeg, 0, 0

    color = _paper_fill(jpeg, brighten)
    try:
        return pad_jpeg_lossless_xy(jpeg, margin_x_px, margin_y_px, color, dpi)
    except LosslessPaddingError as error:
        if not allow_recode:
            raise LosslessPaddingError(
                f"{label}: поля без перекодирования не вставились ({error}). "
                "Либо чинить полосу, либо разрешить пересжатие ключом --allow-jpeg-recode"
            ) from error
        logger.warning("Перекодирование всей полосы %s: беспотерьная вставка полей невозможна (%s)", label, error)

    # Запасной путь: полное декодирование и пересжатие ИСХОДНЫМИ таблицами квантования —
    # хоть какое-то качество этим сохраняется. Число каналов тоже сохраняется: очищенные
    # полосы пака-1 почти все одноканальные, и разложить такую в RGB значило бы утроить
    # её вес на ровном месте.
    with Image.open(BytesIO(jpeg)) as source:
        quantization = source.quantization
        mode = "L" if source.mode == "L" else "RGB"
        image = np.asarray(source.convert(mode))
    fill = round(0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]) if mode == "L" else color
    padded = cv2.copyMakeBorder(
        image, margin_y_px, margin_y_px, margin_x_px, margin_x_px, cv2.BORDER_CONSTANT, value=fill
    )
    buffer = BytesIO()
    Image.fromarray(padded, mode).save(buffer, format="JPEG", qtables=quantization)
    return set_jpeg_dpi(buffer.getvalue(), dpi), margin_x_px, margin_y_px


def shift_overlays(overlays: "list[Overlay]", dx: int, dy: int) -> "tuple[Overlay, ...]":
    """Сдвинуть врезки на ширину полей: их места размечены в пикселях полосы БЕЗ полей."""
    if not dx and not dy:
        return tuple(overlays)
    return tuple(
        Overlay(o.jpeg, (o.rect_px[0] + dx, o.rect_px[1] + dy, o.rect_px[2] + dx, o.rect_px[3] + dy)) for o in overlays
    )


def build_issue(plan: IssuePlan, params: BuildParams) -> IssueReport:
    """Собирает обе PDF одного выпуска. Ошибка роняет ВЕСЬ выпуск, а не одну полосу.

    Именно весь: PDF с пропущенной полосой ничем не отличается на вид от целой, а номера
    страниц в базе после такого пропуска перестают отвечать содержимому — и обнаружится
    это уже на сборке финальной PDF, когда исправлять поздно.
    """
    report = IssueReport(rel_path=plan.rel_path, issue_id=plan.issue_id)
    full_path = params.full_pdf_dir / plan.full_pdf_name
    pics_name = plan.pics_pdf_name
    pics_path = params.pics_only_pdf_dir / pics_name if pics_name else None

    if params.skip_if_exists and full_path.exists() and (pics_path is None or pics_path.exists()):
        report.status = "skipped"
        return report

    try:
        full_pdf = pikepdf.Pdf.new()
        pics_pdf = pikepdf.Pdf.new() if pics_path is not None else None

        for page_plan in plan.pages:
            sharpened_path = params.sharpened_dir / page_plan.sharpened_rel_path
            jpeg = sharpened_path.read_bytes()
            info = read_jpeg_info(jpeg)
            expected = page_plan.file_size
            if (info.width, info.height) != expected:
                turned = " (полоса помечена под поворот)" if page_plan.rotate_cw else ""
                raise ValueError(
                    f"{page_plan.sharpened_rel_path}: заострённая копия {info.width}x{info.height} "
                    f"не совпадает с полосой {expected[0]}x{expected[1]}{turned}"
                )

            overlays: "list[Overlay]" = []
            base = jpeg
            if page_plan.pictures:
                original = load_image(params.originals_dir / page_plan.original_rel_path)
                if (original.shape[1], original.shape[0]) != expected:
                    raise ValueError(
                        f"{page_plan.original_rel_path}: оригинал {original.shape[1]}x{original.shape[0]} "
                        f"не совпадает с полосой {expected[0]}x{expected[1]}"
                    )
                # Врезки — в координатах ФАЙЛА: у повёрнутой полосы это не то же, что в базе.
                placed = page_plan.placed_pictures()
                covering = next((p for p in placed if p.covers(*expected)), None)
                rest = placed
                if covering is not None:
                    base = encode_jpeg(original, covering.gray, params.picture_quality)
                    rest = tuple(p for p in placed if p is not covering)
                overlays = [
                    Overlay(encode_jpeg(crop(original, p.rect), p.gray, params.picture_quality), p.rect)
                    for p in rest
                    if p.x2 > p.x1 and p.y2 > p.y1
                ]
                del original

            # Поля — только в полной PDF. В PAGES_WITH_PICS_ONLY кладём ту же полосу как
            # есть: её распрямлять не будут, кадр не вырастет, а поля лишь увели бы
            # иллюстрации от размеченных для них мест.
            margin_x_px = int(round(params.margin_x_mm / MM_PER_INCH * page_plan.dpi))
            margin_y_px = int(round(params.margin_y_mm / MM_PER_INCH * page_plan.dpi))
            full_base, used_x, used_y = pad_page_jpeg(
                base,
                margin_x_px,
                margin_y_px,
                page_plan.dpi,
                params.fill_brighten,
                params.allow_jpeg_recode,
                page_plan.sharpened_rel_path,
            )
            full_page_px = (expected[0] + 2 * used_x, expected[1] + 2 * used_y)
            # По выпуску пишем самое широкое поле из применённых. Обычно они у всех полос
            # одинаковы, но округление идёт до размера MCU конкретного JPEG, и полоса с
            # другой субдискретизацией цветности могла бы получить поле на несколько пикселей
            # шире; в базе честнее держать верхнюю границу, чем как повезёт с последней.
            report.margin_left_mm = report.margin_right_mm = max(
                report.margin_left_mm or 0.0, used_x / page_plan.dpi * MM_PER_INCH
            )
            report.margin_top_mm = report.margin_bottom_mm = max(
                report.margin_top_mm or 0.0, used_y / page_plan.dpi * MM_PER_INCH
            )

            full_pdf.pages.append(
                make_page(full_pdf, full_base, page_plan.dpi, shift_overlays(overlays, used_x, used_y), full_page_px)
            )
            if pics_pdf is not None and page_plan.pages_with_pics_only_pdf_page_idx is not None:
                pics_pdf.pages.append(make_page(pics_pdf, base, page_plan.dpi, tuple(overlays), expected))

            report.pages += 1
            if page_plan.pictures:
                report.pages_with_pics += 1

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_pdf.save(full_path)
        if pics_pdf is not None and pics_path is not None:
            pics_path.parent.mkdir(parents=True, exist_ok=True)
            pics_pdf.save(pics_path)
    except Exception as exc:  # noqa: BLE001 — одна битая полоса не должна ронять весь прогон
        logger.exception("Ошибка на выпуске %s", plan.rel_path)
        return IssueReport(rel_path=plan.rel_path, issue_id=plan.issue_id, status="error", reason=str(exc))

    return report


def _run_job(job: "tuple[IssuePlan, BuildParams]") -> IssueReport:
    """Точка входа воркера: пул умеет передавать только один аргумент."""
    return build_issue(*job)


def load_plans(
    db_path: Path, pack_name: str, *, only_year: "str | None" = None, only_issue: "str | None" = None
) -> "list[IssuePlan]":
    """Читает базу ОДИН раз и отдаёт плоские, пиклуемые планы выпусков.

    Номера страниц в обеих PDF считаются ЗДЕСЬ, а не в воркере, и это важно в двух местах.
    Во-первых, они однозначно определяются порядком полос и наличием у них иллюстраций, то
    есть воркеру их выводить нечего. Во-вторых, при ``--skip-if-exists`` воркер файлы не
    трогает вовсе, а номера всё равно должны попасть в базу — иначе повторный прогон стёр
    бы то, что записал первый.
    """
    from ocr_utils.db.repo import require_pack
    from ocr_utils.db.session import open_db

    plans: "list[IssuePlan]" = []
    session_factory = open_db(db_path, create=False)
    with session_factory() as session:
        pack = require_pack(session, pack_name)
        for year in pack.year_packages:
            if only_year is not None and year.name != only_year:
                continue
            for issue in year.issues:
                if only_issue is not None and issue.name != only_issue:
                    continue
                pages: "list[PagePlan]" = []
                pics_index = 0
                for full_index, page in enumerate(issue.pages):
                    if not page.width or not page.height:
                        raise ValueError(f"{page.source_rel_path}: нет размеров кадра, сначала нужен detect")
                    pictures = tuple(
                        PicturePlan(int(r.x1), int(r.y1), int(r.x2), int(r.y2), r.kind)
                        for r in page.rect_regions
                        if r.kind in PICTURE_KINDS
                    )
                    sharpened = page.sharpened_text_pic_rel_path
                    if sharpened is None:
                        # Запасной путь на случай, если collect_sharpened гоняли без --db:
                        # выгрузка Capture One отличается от очищенной полосы только
                        # расширением, включая отпечаток в имени.
                        base = page.cleaned_rel_path or page.source_rel_path
                        sharpened = Path(base).with_suffix(".jpg").as_posix()
                    pages.append(
                        PagePlan(
                            page_id=page.id,
                            # Очищенный файл, если он записан прогоном очистки; иначе
                            # исходное имя — так пак, очищенный до появления отпечатков в
                            # именах, продолжает собираться без правок.
                            original_rel_path=page.cleaned_rel_path or page.source_rel_path,
                            sharpened_rel_path=sharpened,
                            width=int(page.width),
                            height=int(page.height),
                            dpi=int(page.dpi or 600),
                            pictures=pictures,
                            full_pdf_page_idx=full_index,
                            pages_with_pics_only_pdf_page_idx=pics_index if pictures else None,
                            rotate_cw=int(page.rotate_cw or 0),
                        )
                    )
                    if pictures:
                        pics_index += 1
                if pages:
                    plans.append(
                        IssuePlan(issue_id=issue.id, year_name=year.name, issue_name=issue.name, pages=tuple(pages))
                    )
    return plans


def save_plans(
    db_path: Path,
    pack_name: str,
    plans: "list[IssuePlan]",
    params: BuildParams,
    built: "set[int]",
    reports: "list[IssueReport] | None" = None,
) -> None:
    """Записывает корни пака, имена собранных PDF, номера страниц в них и поля.

    ``built`` — выпуски, чьи файлы действительно лежат на диске (собранные сейчас плюс
    пропущенные как уже готовые). Имена PDF у выпусков, упавших с ошибкой, не пишутся:
    записанное имя означает «файл есть», и врать тут нельзя — на нём стоит вся сборка
    финальной PDF.

    Поля пишутся только по выпускам, СОБРАННЫМ этим прогоном: у пропущенного как уже
    готовый файлы не трогали, и что за поля в них лежат — знает предыдущая запись в базе,
    а не текущие ключи командной строки. Затирать её сегодняшним значением нельзя.
    """
    from ocr_utils.db.repo import require_pack
    from ocr_utils.db.session import open_db

    by_issue = {plan.issue_id: plan for plan in plans}
    margins = {r.issue_id: r for r in (reports or []) if r.status == "ok"}
    session_factory = open_db(db_path)
    with session_factory() as session:
        pack = require_pack(session, pack_name)
        pack.cleaned_pics_root = str(params.originals_dir)
        pack.sharpened_text_pics_root = str(params.sharpened_dir)
        pack.full_intermediate_pdf_root = str(params.full_pdf_dir)
        pack.pages_with_pics_only_intermediate_pdf_root = str(params.pics_only_pdf_dir)

        for year in pack.year_packages:
            for issue in year.issues:
                plan = by_issue.get(issue.id)
                if plan is None or issue.id not in built:
                    continue
                issue.full_intermediate_pdf_name = plan.full_pdf_name
                issue.pages_with_pics_only_intermediate_pdf_name = plan.pics_pdf_name
                report = margins.get(issue.id)
                if report is not None:
                    issue.full_intermediate_pdf_margin_left_mm = report.margin_left_mm
                    issue.full_intermediate_pdf_margin_right_mm = report.margin_right_mm
                    issue.full_intermediate_pdf_margin_top_mm = report.margin_top_mm
                    issue.full_intermediate_pdf_margin_bottom_mm = report.margin_bottom_mm
                indices = {p.page_id: p for p in plan.pages}
                for page in issue.pages:
                    page_plan = indices.get(page.id)
                    if page_plan is None:
                        continue
                    page.full_pdf_page_idx = page_plan.full_pdf_page_idx
                    page.pages_with_pics_only_pdf_page_idx = page_plan.pages_with_pics_only_pdf_page_idx
        session.commit()


def run_build(
    db_path: Path,
    pack_name: str,
    params: BuildParams,
    *,
    only_year: "str | None" = None,
    only_issue: "str | None" = None,
    jobs: int = DEFAULT_JOBS,
    progress: bool = True,
) -> BuildStats:
    """Полный прогон: план по базе, сборка PDF пулом, запись результатов обратно в базу."""
    plans = load_plans(db_path, pack_name, only_year=only_year, only_issue=only_issue)
    stats = BuildStats()
    built: "set[int]" = set()

    jobs_list = [(plan, params) for plan in plans]
    bar = tqdm(
        total=len(jobs_list),
        disable=not progress,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    with bar:
        if jobs > 1 and len(jobs_list) > 1:
            with ProcessPoolExecutor(max_workers=min(jobs, len(jobs_list))) as pool:
                results = list(iter_with_progress(pool.map(_run_job, jobs_list), bar))
        else:
            results = list(iter_with_progress((_run_job(job) for job in jobs_list), bar))

    for report in results:
        stats.reports.append(report)
        if report.status == "error":
            stats.failed += 1
            logger.error("Выпуск %s: %s", report.rel_path, report.reason)
            continue
        built.add(report.issue_id)
        stats.pages += report.pages
        stats.pages_with_pics += report.pages_with_pics
        if report.status == "skipped":
            stats.skipped += 1
        else:
            stats.issues += 1

    save_plans(db_path, pack_name, plans, params, built, results)
    return stats


def iter_with_progress(iterable, bar):
    """Прогоняет итератор, двигая полосу прогресса. Общий помощник обоих сборщиков."""
    for item in iterable:
        bar.update(1)
        yield item


DIR_IN = click.Path(exists=True, file_okay=False, path_type=Path)
DIR_OUT = click.Path(file_okay=False, path_type=Path)
FILE_IN = click.Path(exists=True, dir_okay=False, path_type=Path)


@click.command()
@click.option("--db", "db_path", required=True, type=FILE_IN, help="База разметки (та, что после from-cvat).")
@click.option("--pack-name", required=True, help="Имя пака в базе.")
@click.option("--originals-dir", required=True, type=DIR_IN, help="Корень оригиналов: из них режутся иллюстрации.")
@click.option("--sharpened-dir", required=True, type=DIR_IN, help="Корень заострённых копий (после Capture One).")
@click.option("--full-pdf-dir", required=True, type=DIR_OUT, help="Куда класть PDF со всеми полосами.")
@click.option("--pics-only-pdf-dir", required=True, type=DIR_OUT, help="Куда класть PDF только с полосами с растром.")
@click.option(
    "--picture-quality",
    default=DEFAULT_PICTURE_QUALITY,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Качество JPEG у врезанных иллюстраций.",
)
@click.option(
    "--page-margin-x-mm",
    default=DEFAULT_MARGIN_X_MM,
    show_default=True,
    type=click.FloatRange(0.0, 50.0),
    help="Поле слева и справа в ПОЛНОЙ PDF, мм. 0 — без полей.",
)
@click.option(
    "--page-margin-y-mm",
    default=DEFAULT_MARGIN_Y_MM,
    show_default=True,
    type=click.FloatRange(0.0, 50.0),
    help="Поле сверху и снизу в ПОЛНОЙ PDF, мм. 0 — без полей.",
)
@click.option(
    "--fill-brighten",
    default=DEFAULT_FILL_BRIGHTEN,
    show_default=True,
    type=click.IntRange(0, 255),
    help="На сколько тонов осветлить цвет бумаги под заливку полей.",
)
@click.option(
    "--allow-jpeg-recode",
    is_flag=True,
    help="Разрешить пересжатие полосы, если поля не вставились без перекодирования.",
)
@click.option("--only-year", default=None, help="Только этот год.")
@click.option("--only-issue", default=None, help="Только этот выпуск.")
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True, type=int, help="Воркеров; выпуск на воркер.")
@click.option("--skip-if-exists/--no-skip-if-exists", default=True, show_default=True)
@click.option("--no-progress", is_flag=True, help="Без полосы прогресса.")
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
    full_pdf_dir: Path,
    pics_only_pdf_dir: Path,
    picture_quality: int,
    page_margin_x_mm: float,
    page_margin_y_mm: float,
    fill_brighten: int,
    allow_jpeg_recode: bool,
    only_year: "str | None",
    only_issue: "str | None",
    jobs: int,
    skip_if_exists: bool,
    no_progress: bool,
    log_level: str,
) -> None:
    """Собирает промежуточные PDF по выпускам пака."""
    logging.basicConfig(level=log_level.upper(), format="%(levelname)s: %(message)s")
    params = BuildParams(
        originals_dir=originals_dir,
        sharpened_dir=sharpened_dir,
        full_pdf_dir=full_pdf_dir,
        pics_only_pdf_dir=pics_only_pdf_dir,
        picture_quality=picture_quality,
        skip_if_exists=skip_if_exists,
        margin_x_mm=page_margin_x_mm,
        margin_y_mm=page_margin_y_mm,
        fill_brighten=fill_brighten,
        allow_jpeg_recode=allow_jpeg_recode,
    )
    stats = run_build(
        db_path, pack_name, params, only_year=only_year, only_issue=only_issue, jobs=jobs, progress=not no_progress
    )
    click.echo(stats.summary())
    raise SystemExit(1 if stats.failed else 0)


if __name__ == "__main__":
    main()
