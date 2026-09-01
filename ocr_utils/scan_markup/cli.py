"""CLI подсистемы разметки полос: ``python -m ocr_utils.scan_markup <команда>``.

Три команды по трём шагам конвейера::

    detect     оригиналы          -> SQLite (предварительная разметка)
    to-cvat    SQLite             -> уменьшенные копии + проект CVAT с предразметкой
    from-cvat  CVAT               -> SQLite той же схемы (уточнённая разметка)
"""

import logging
from pathlib import Path

import click

from ocr_utils.scan_markup.cvat.client import CvatSettings
from ocr_utils.scan_markup.cvat.export import ExportParams, run_export
from ocr_utils.scan_markup.cvat.publish import PublishParams, run_publish
from ocr_utils.scan_markup.db.session import open_db
from ocr_utils.scan_markup.detection.boxes import FULL_PAGE_FRAC, MIN_REGION_FRAC
from ocr_utils.scan_markup.detection.color_kind import (
    CHROMA_SELF_FRAC_THR,
    CHROMA_SPREAD_THR,
    CHROMA_THR,
    COLOR_FRAC_THR,
)
from ocr_utils.scan_markup.detection.recolor import RecolorParams, run_mark_covers, run_recolor
from ocr_utils.scan_markup.detection.page import PageOptions
from ocr_utils.scan_markup.detection.tone import (
    LINEART_ENTROPY_THR,
    LINEART_MID_FRAC_THR,
    LINEART_SCREEN_PEAK_THR,
    STAMP_INK_CONTRAST_THR,
)
from ocr_utils.scan_markup.detection.regions import (
    FULL_PAGE_COLOR_FRAC,
    GROW_PAPER_MARGIN,
    LEADER_EMPTY_ROWS_THR,
    LINEART_MAX_DOT_FRAC,
    LEADER_PERIODICITY_THR,
    LEADER_TONE_SPREAD_THR,
    LINEART_PICTURE_MIN_FRAC,
    SAFETY_MIN_FRAC,
    SURYA_LINEART_P99_PX,
)
from ocr_utils.scan_markup.detection.run import DetectParams, run_detect
from ocr_utils.scan_markup.geometry import CVAT_DPI
from ocr_utils.scan_markup.pen_marks import DEFAULT_WEIGHTS, PEN_CHROMA_THR, fix_pages
from ocr_utils.scan_markup.validation.report import console_lines, write_csv, write_markdown
from ocr_utils.scan_markup.validation.run import ValidateParams, run_validate

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def _set_log_level(level: str) -> None:
    logging.getLogger().setLevel(level.upper())


def _cvat_options(func):
    """Общие опции подключения к CVAT: они одинаковы у ``to-cvat`` и ``from-cvat``.

    Пустые значения означают «взять из docker/.env» — см. ``cvat.client.CvatSettings``.
    """
    func = click.option("--cvat-url", default=None, help="Адрес CVAT; по умолчанию из docker/.env.")(func)
    func = click.option("--cvat-user", default=None, help="Логин; по умолчанию ADMIN_USER из docker/.env.")(func)
    func = click.option("--cvat-password", default=None, help="Пароль; по умолчанию ADMIN_PASS из docker/.env.")(func)
    func = click.option("--cvat-org", default=None, help="Слаг организации; по умолчанию ORG_SLUG из docker/.env.")(
        func
    )
    return func


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """Растровые области и библиотечные печати на полосах журнала."""


@main.command("detect")
@click.option(
    "--pack-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Папка пака сканов (внутри — годы, в них выпуски, в них полосы).",
)
@click.option(
    "--db",
    "db_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Файл SQLite. Существующий не переписывается: пак добавляется к уже записанным.",
)
@click.option("--pack-name", default=None, help="Имя пака в базе; по умолчанию — имя папки.")
@click.option(
    "--default-dpi",
    default=None,
    type=int,
    help="DPI для файлов без тега разрешения. Без него такие полосы пропускаются: "
    "молча взять 96 dpi значило бы промахнуться в шесть раз.",
)
@click.option("--only-year", default=None, help="Обработать только этот годовой комплект.")
@click.option("--only-issue", default=None, help="Обработать только этот выпуск.")
@click.option("--limit", default=None, type=int, help="Обработать не больше стольких полос (для проб).")
@click.option(
    "--skip-detected/--no-skip-detected",
    "skip_detected",
    default=False,
    show_default=True,
    help="Пропускать полосы, по которым детекция уже проходила И чей файл не менялся. "
    "«Не менялся» проверяется сначала по размеру и времени правки (без чтения файла), "
    "и лишь при расхождении — по хешу содержимого.",
)
@click.option(
    "--rehash-all/--no-rehash-all",
    "rehash_all",
    default=False,
    show_default=True,
    help="Не верить размеру и времени правки: пересчитать хеш каждой полосы. "
    "Имеет смысл только вместе с --skip-detected и означает перечитывание всего пака.",
)
@click.option(
    "--use-surya-layout/--no-use-surya-layout",
    "use_surya_layout",
    default=True,
    show_default=True,
    help="ПЕРВИЧНЫЙ детектор: Surya предлагает блоки Picture, а пиксельные проверки уточняют "
    "границу, отличают растр от штриха и решают про цвет. Нужен GPU. Без флага работают одни "
    "пиксели — прогон дешевле, но на полосах содержания появляются ложные срабатывания "
    "(строка отточий даёт ту же статистику, что растровая сетка).",
)
@click.option(
    "--first-page-is-cover/--no-first-page-is-cover",
    "first_page_is_cover",
    default=True,
    show_default=True,
    help="Первая полоса каждого выпуска — обложка: одна цветная область во весь кадр без "
    "пиксельной детекции. Верно для всех 123 выпусков пака-1; остальные три обложки "
    "(вторая полоса и две последние) идут обычной детекцией — там бывает и текст, и пара "
    "ч/б фотографий, и штриховой рисунок.",
)
@click.option(
    "--jobs",
    default=8,
    show_default=True,
    type=int,
    help="Процессов на пиксельный этап (чтение, уменьшение, связные компоненты). Surya при "
    "этом работает в родителе и в пул не уезжает, так что флаг совместим с ней.",
)
@click.option("--chroma-thr", default=CHROMA_THR, show_default=True, type=float, help="Порог хроматичности в Lab.")
@click.option(
    "--color-frac-thr",
    default=COLOR_FRAC_THR,
    show_default=True,
    type=float,
    help="Порог ПРОШЛОЙ метрики цвета. В решении больше не участвует, само число пишется "
    "в базу ради сравнения со старой разметкой.",
)
@click.option(
    "--chroma-spread-thr",
    default=CHROMA_SPREAD_THR,
    show_default=True,
    type=float,
    help="Разброс хроматичности hypot(std(a), std(b)) в Lab, с которого область цветная. "
    "Замер по паку-1: ч/б области 2.2..6.5, цветные обложки 9.3..19.6.",
)
@click.option(
    "--chroma-self-frac-thr",
    default=CHROMA_SELF_FRAC_THR,
    show_default=True,
    type=float,
    help="Доля пикселей, отклонившихся от собственной медианы оттенка: второе условие через "
    "«или». По умолчанию ВЫКЛЮЧЕНО — на паке-1 порога, который делит ч/б и цветные по этой "
    "метрике, не нашлось (ч/б доходят до 0.293 при цветных от 0.092).",
)
@click.option(
    "--cell-px",
    default=None,
    type=int,
    help="Сторона клетки детектора точек в пикселях полосы; по умолчанию 128 при 600 dpi, "
    "пересчитанные на её разрешение.",
)
@click.option(
    "--dot-frac-thr",
    default=None,
    type=float,
    help="Доля точечных пятен, с которой клетка считается растровой; по умолчанию 0.88.",
)
@click.option(
    "--min-cells",
    default=None,
    type=int,
    help="Сколько растровых клеток должно быть в области, чтобы она вообще считалась; "
    "по умолчанию 3. Считается по клеткам ДО морфологии.",
)
@click.option(
    "--surya-lineart-p99",
    "lineart_p99",
    default=SURYA_LINEART_P99_PX,
    show_default=True,
    type=int,
    help="p99 площади пятна краски, с которой блок Surya считается ШТРИХОВЫМ рисунком, а не "
    "растром. Замер: фотографии 100..4439, штрих 4761..550783. Проверяется только у блоков, "
    "под которыми не нашлось растровых клеток.",
)
@click.option(
    "--lineart-picture-min-frac",
    default=LINEART_PICTURE_MIN_FRAC,
    show_default=True,
    type=float,
    help="Доля площади полосы, с которой ЦВЕТНОЙ штриховой рисунок считается иллюстрацией. "
    "Мельче — помечается «подозрение на печать»: библиотечная печать это цветной штрих, и "
    "отличить её от рисунка можно только размером (печати 2.1%, рисунок обложки ~100%).",
)
@click.option(
    "--leader-empty-rows-thr",
    default=LEADER_EMPTY_ROWS_THR,
    show_default=True,
    type=float,
    help="Доля строк развёртки без единой точки, с которой область подозревается в отточиях.",
)
@click.option(
    "--leader-periodicity-thr",
    default=LEADER_PERIODICITY_THR,
    show_default=True,
    type=float,
    help="Периодичность строк точек (автокорреляция на сдвигах высоты строки), с которой "
    "область подозревается в отточиях. Замер: отточия 0.595..0.891, настоящие до 0.511.",
)
@click.option(
    "--leader-tone-spread-thr",
    default=LEADER_TONE_SPREAD_THR,
    show_default=True,
    type=float,
    help="Размах яркостей (p95-p5) после уменьшения области примерно до 7.5 dpi, НИЖЕ "
    "которого область подозревается в отточиях. Замер: отточия 9..63, настоящие 101..229. "
    "Работает В ПАРЕ с двумя порогами выше: область выбрасывается, только если провалила и "
    "строение, и тон.",
)
@click.option(
    "--lineart-mid-frac",
    default=LINEART_MID_FRAC_THR,
    show_default=True,
    type=float,
    help="Доля пикселей в средних тонах, НИЖЕ которой область подозревается в штрихе. Замер: "
    "штрих 0.032..0.322, фотографии 0.111..0.568.",
)
@click.option(
    "--lineart-entropy",
    default=LINEART_ENTROPY_THR,
    show_default=True,
    type=float,
    help="Энтропия гистограммы области, НИЖЕ которой она подозревается в штрихе. Замер: "
    "штрих 4.245..7.170, фотографии 5.770..7.721.",
)
@click.option(
    "--lineart-screen-peak",
    default=LINEART_SCREEN_PEAK_THR,
    show_default=True,
    type=float,
    help="Выступ пика растровой сетки в спектре, НИЖЕ которого область подозревается в "
    "штрихе. Замер: штрих 1.100..2.156, фотографии 1.054..49.796. Работает В СВЯЗКЕ с двумя "
    "порогами выше: штрихом область признаётся, только если провалила все три сразу.",
)
@click.option(
    "--lineart-max-dot-frac",
    default=LINEART_MAX_DOT_FRAC,
    show_default=True,
    type=float,
    help="Доля растровых клеток в рамке, ВЫШЕ которой область штрихом не признаётся никогда: "
    "фотография заполняет прямоугольник растром сплошь. Замер: штрих 0.60..0.85, портреты "
    "1975/01 IMG_0048_1L — 0.98..1.00.",
)
@click.option(
    "--stamp-ink-contrast",
    default=STAMP_INK_CONTRAST_THR,
    show_default=True,
    type=float,
    help="Контраст «бумага минус краска», НИЖЕ которого мелкий штрих считается оттиском "
    "библиотечной печати, а не виньеткой рубрики. Замер: оттиски 79..220, виньетки 231..254.",
)
@click.option(
    "--grow-paper-margin",
    default=GROW_PAPER_MARGIN,
    show_default=True,
    type=int,
    help="На сколько уровней ниже бумаги должна быть прилегающая полоска, чтобы страховочная "
    "рамка росла дальше. Ноль отключает рост.",
)
@click.option(
    "--full-page-color-frac",
    default=FULL_PAGE_COLOR_FRAC,
    show_default=True,
    type=float,
    help="Доля полосы, которую должен занимать охватывающий прямоугольник всех найденных "
    "областей, чтобы ЦВЕТНАЯ полоса была помечена одной областью во весь кадр. Второе "
    "условие — разброс цвета по всей полосе выше --chroma-spread-thr: обложка отличается "
    "от двух ч/б фотографий не зазором, а тем, что цветная целиком (33.7 против 4.1).",
)
@click.option(
    "--safety-min-frac",
    default=SAFETY_MIN_FRAC,
    show_default=True,
    type=float,
    help="Доля площади полосы, начиная с которой область остаётся БЕЗ подтверждения Surya. "
    "Страховка от её пропусков: отточия занимают 0.7% полосы и под порог не проходят, а самая "
    "мелкая настоящая неподтверждённая область — 19.8%.",
)
@click.option(
    "--min-region-frac",
    default=MIN_REGION_FRAC,
    show_default=True,
    type=float,
    help="Минимальная доля площади полосы для растровой области.",
)
@click.option(
    "--merge-gap",
    default=None,
    type=int,
    help="Зазор слияния областей в пикселях полосы; по умолчанию 48 при 600 dpi, " "пересчитанные на её разрешение.",
)
@click.option(
    "--full-page-frac",
    default=FULL_PAGE_FRAC,
    show_default=True,
    type=float,
    help="Доля площади, с которой область считается «во всю полосу» (обложка).",
)
@click.option(
    "--debug-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Куда писать оверлеи с найденными областями и подписями kind/разброс/доля точек.",
)
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def detect_command(pack_dir: Path, db_path: Path, pack_name: str | None, log_level: str, **kwargs) -> None:
    """Предварительная детекция растровых областей по оригиналам."""
    _set_log_level(log_level)
    debug_dir = kwargs.pop("debug_dir")
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    params = DetectParams(
        pack_dir=pack_dir, db_path=db_path, pack_name=pack_name or pack_dir.name, debug_dir=debug_dir, **kwargs
    )
    stats = run_detect(params, open_db(db_path))
    click.echo(
        f"Полос обработано: {stats.pages}, пропущено: {stats.skipped}, ошибок: {stats.failed}.\n"
        f"Файлов изменилось с прошлого прогона: {stats.changed}.\n"
        f"Растровых областей: {stats.regions} (цветных {stats.color}, серых {stats.grayscale}, "
        f"во всю полосу {stats.full_page})."
    )


def _recolor_options(func):
    """Общие опции ``recolor`` и ``mark-covers``: обе правят готовую базу."""
    func = click.option("--dry-run", is_flag=True, default=False, help="Только отчёт, ничего не записывать.")(func)
    func = click.option(
        "--pack-dir",
        default=None,
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Папка с оригиналами; по умолчанию берётся из базы.",
    )(func)
    func = click.option("--pack-name", required=True, help="Имя пака в базе.")(func)
    func = click.option("--db", "db_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))(
        func
    )
    return func


@main.command("recolor")
@_recolor_options
@click.option("--jobs", default=8, show_default=True, type=int, help="Процессов на чтение оригиналов.")
@click.option("--chroma-thr", default=CHROMA_THR, show_default=True, type=float)
@click.option("--color-frac-thr", default=COLOR_FRAC_THR, show_default=True, type=float)
@click.option("--chroma-spread-thr", default=CHROMA_SPREAD_THR, show_default=True, type=float)
@click.option("--chroma-self-frac-thr", default=CHROMA_SELF_FRAC_THR, show_default=True, type=float)
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def recolor_command(db_path: Path, pack_name: str, log_level: str, **kwargs) -> None:
    """Пересчитать color/grayscale по уже найденным областям, не перечитывая весь пак.

    Читаются только полосы, у которых области есть: в паке-1 это 693 файла вместо 12 135,
    то есть минуты вместо часов. Координаты не трогаются, ручная разметка из CVAT — тоже.
    """
    _set_log_level(log_level)
    params = RecolorParams(db_path=db_path, pack_name=pack_name, **kwargs)
    stats = run_recolor(params, open_db(db_path))
    click.echo(
        f"Полос прочитано: {stats.pages}, ошибок: {stats.failed}.\n"
        f"Областей пересчитано: {stats.regions}, сменили тип: {stats.changed}."
        + (" (--dry-run, в базу ничего не записано)" if params.dry_run else "")
    )


@main.command("mark-covers")
@_recolor_options
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def mark_covers_command(db_path: Path, pack_name: str, log_level: str, **kwargs) -> None:
    """Пометить первую полосу каждого выпуска обложкой во весь кадр. Пикселей не читает.

    То же, что даёт ``detect --first-page-is-cover``, но по готовой базе и за секунды:
    порядковый номер полосы и размеры кадра там уже есть.
    """
    _set_log_level(log_level)
    params = RecolorParams(db_path=db_path, pack_name=pack_name, **kwargs)
    stats = run_mark_covers(params, open_db(db_path))
    click.echo(
        f"Первых полос: {stats.pages}, помечено заново: {stats.changed}, пропущено с ошибкой: {stats.failed}."
        + (" (--dry-run, в базу ничего не записано)" if params.dry_run else "")
    )


@main.command("validate")
@click.option(
    "--pack-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Папка пака сканов.",
)
@click.option(
    "--cases-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Корень с папками-эталонами: имя папки называет тип дефекта, имена файлов внутри — "
    "это имена оверлеев из --debug-dir.",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Куда писать оверлеи прогона, отчёт report.md и таблицу measurements.csv.",
)
@click.option(
    "--db",
    "db_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="База прошлого прогона. Нужна для двух вещей: узнать порядковые номера полос "
    "(без них не проверить обложку) и набрать контрольную выборку.",
)
@click.option("--pack-name", default=None, help="Имя пака в базе; по умолчанию — имя папки.")
@click.option(
    "--control",
    "control_limit",
    default=40,
    show_default=True,
    type=int,
    help="Сколько случайных полос с уже найденными областями прогнать сверх выборки. "
    "Страховка от обратной ошибки: чинили одно, потеряли другое. 0 — не проверять.",
)
@click.option("--jobs", default=8, show_default=True, type=int, help="Процессов на пиксельный этап.")
@click.option(
    "--use-surya-layout/--no-use-surya-layout",
    "use_surya_layout",
    default=True,
    show_default=True,
    help="Гонять первичный детектор Surya. Без него проверяются одни пиксели.",
)
@click.option("--first-page-is-cover/--no-first-page-is-cover", default=True, show_default=True)
@click.option("--chroma-spread-thr", default=CHROMA_SPREAD_THR, show_default=True, type=float)
@click.option("--chroma-self-frac-thr", default=CHROMA_SELF_FRAC_THR, show_default=True, type=float)
@click.option("--min-region-frac", default=MIN_REGION_FRAC, show_default=True, type=float)
@click.option("--cell-px", default=None, type=int)
@click.option("--dot-frac-thr", default=None, type=float)
@click.option("--min-cells", default=None, type=int)
@click.option("--surya-lineart-p99", "lineart_p99", default=SURYA_LINEART_P99_PX, show_default=True, type=int)
@click.option("--safety-min-frac", default=SAFETY_MIN_FRAC, show_default=True, type=float)
@click.option("--lineart-picture-min-frac", default=LINEART_PICTURE_MIN_FRAC, show_default=True, type=float)
@click.option("--full-page-color-frac", default=FULL_PAGE_COLOR_FRAC, show_default=True, type=float)
@click.option("--leader-empty-rows-thr", default=LEADER_EMPTY_ROWS_THR, show_default=True, type=float)
@click.option("--leader-periodicity-thr", default=LEADER_PERIODICITY_THR, show_default=True, type=float)
@click.option("--leader-tone-spread-thr", default=LEADER_TONE_SPREAD_THR, show_default=True, type=float)
@click.option("--lineart-mid-frac", default=LINEART_MID_FRAC_THR, show_default=True, type=float)
@click.option("--lineart-entropy", default=LINEART_ENTROPY_THR, show_default=True, type=float)
@click.option("--lineart-screen-peak", default=LINEART_SCREEN_PEAK_THR, show_default=True, type=float)
@click.option("--stamp-ink-contrast", default=STAMP_INK_CONTRAST_THR, show_default=True, type=float)
@click.option("--lineart-max-dot-frac", default=LINEART_MAX_DOT_FRAC, show_default=True, type=float)
@click.option("--grow-paper-margin", default=GROW_PAPER_MARGIN, show_default=True, type=int)
@click.option("--log-level", default="WARNING", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def validate_command(
    pack_dir: Path,
    cases_dir: Path,
    out_dir: Path | None,
    db_path: Path | None,
    pack_name: str | None,
    control_limit: int,
    jobs: int,
    use_surya_layout: bool,
    log_level: str,
    **kwargs,
) -> None:
    """Прогнать детекцию по папкам-эталонам и сказать, сколько дефектов осталось.

    Около сотни файлов и меньше минуты — это рабочий цикл настройки порогов. Полный прогон
    по паку идёт часы, и крутить пороги по нему нельзя.
    """
    _set_log_level(log_level)
    options = PageOptions(need_digest=False, **kwargs)
    params = ValidateParams(
        pack_dir=pack_dir,
        cases_root=cases_dir,
        options=options,
        out_dir=out_dir,
        jobs=jobs,
        db_path=db_path,
        pack_name=pack_name or pack_dir.name,
        control_limit=control_limit if db_path is not None else 0,
        use_surya_layout=use_surya_layout,
    )
    report = run_validate(params)
    click.echo("\n".join(console_lines(report)))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(report, out_dir / "measurements.csv")
        write_markdown(report, out_dir / "report.md", out_dir)
        click.echo(f"\nОтчёт: {out_dir / 'report.md'}, измерения: {out_dir / 'measurements.csv'}")


@main.command("fix-pen-marks")
@click.option(
    "--pack-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Папка пака сканов.",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Куда класть починенные полосы; относительный путь внутри пака сохраняется. "
    "ОТДЕЛЬНАЯ папка, а не сам пак: лишний файл рядом с оригиналом стал бы лишней полосой "
    "в выпуске и сдвинул бы порядок остальных.",
)
@click.option(
    "--page",
    "pages",
    multiple=True,
    required=True,
    help="Полоса внутри пака, например 1968/01/IMG_0045_2R.tif. Можно повторять.",
)
@click.option(
    "--weights",
    default=",".join(str(value) for value in DEFAULT_WEIGHTS),
    show_default=True,
    help="Веса линейной комбинации каналов через запятую, в порядке B,G,R. Умолчание найдено "
    "перебором по замеру 1968/01 IMG_0045_2R: клякса уходит с 161 до 217 при бумаге 245, а "
    "текст под ней остаётся на 118. Знак у красного отрицательный — синяя паста поглощает "
    "красный свет и в этом канале чернее печатного текста.",
)
@click.option(
    "--chroma-thr",
    default=PEN_CHROMA_THR,
    show_default=True,
    type=float,
    help="Хроматичность в Lab, с которой пиксель считается пастой. Комбинация подмешивается "
    "ТОЛЬКО внутри найденной по этому порогу маски: на всю полосу она поднимала бы шум ради "
    "пятна в сантиметр.",
)
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def fix_pen_marks_command(
    pack_dir: Path, out_dir: Path, pages: tuple[str, ...], weights: str, chroma_thr: float, log_level: str
) -> None:
    """Погасить следы шариковой ручки линейной комбинацией каналов вокруг самой кляксы.

    Правка ИСХОДНИКА, а не детекции: пометка не растровая иллюстрация и в бинаризованный
    PDF попадёт ровно так же, а текст под ней будет потерян.
    """
    _set_log_level(log_level)
    try:
        parsed = tuple(float(part) for part in weights.split(","))
    except ValueError as exc:
        raise click.BadParameter(f"веса задаются тремя числами через запятую: {exc}", param_hint="--weights")
    if len(parsed) != 3:
        raise click.BadParameter("нужно ровно три числа: B,G,R", param_hint="--weights")

    results = fix_pages(pack_dir, list(pages), out_dir, parsed, chroma_thr)
    for result in results:
        if result.error:
            click.echo(f"{result.rel_path}: {result.error}")
        else:
            click.echo(f"{result.rel_path} -> {result.dst} (пересчитано {result.pen_area_px} px)")
    written = sum(1 for result in results if result.dst is not None)
    click.echo(f"Записано: {written} из {len(results)}.")


@main.command("to-cvat")
@click.option("--db", "db_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--pack-name", required=True, help="Имя пака в базе.")
@click.option(
    "--share-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Каталог, смонтированный в CVAT как /home/django/share. Должен совпадать с IMAGES_DIR в docker/.env.",
)
@click.option(
    "--pack-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Папка с оригиналами; по умолчанию берётся из базы.",
)
@click.option("--only-year", default=None, help="Завести только этот годовой комплект.")
@click.option(
    "--cvat-dpi",
    default=CVAT_DPI,
    show_default=True,
    type=int,
    help="Разрешение уменьшенных копий для разметки: делитель полосы = round(её dpi / этого). "
    "Полосам, уже залитым в CVAT, делитель не меняется — их разметка нарисована в прежнем "
    "масштабе; чтобы перевести и их, задачу-год надо завести заново.",
)
@click.option("--jobs", "workers", default=8, show_default=True, type=int, help="Процессов на уменьшение картинок.")
@click.option(
    "--skip-images/--no-skip-images",
    "skip_images",
    default=False,
    show_default=True,
    help="Не готовить картинки, использовать уже лежащие в share.",
)
@click.option(
    "--force-images/--no-force-images",
    "force_images",
    default=False,
    show_default=True,
    help="Перезаписывать уже готовые уменьшенные копии.",
)
@click.option(
    "--force-annotations/--no-force-annotations",
    "force_annotations",
    default=False,
    show_default=True,
    help="Перезалить предразметку в уже существующую задачу. ОСТОРОЖНО: заливка заменяет "
    "разметку задачи целиком, то есть затирает ручную правку.",
)
@click.option(
    "--recreate-stale/--no-recreate-stale",
    "recreate_stale",
    default=False,
    show_default=True,
    help="Пересоздать задачи-годы, в которых изменились исходники. Ручная разметка со всех "
    "НЕизменившихся полос переносится в новую задачу, изменившиеся получают свежую "
    "автоматическую предразметку. Без флага такие задачи только показываются в отчёте. "
    "CVAT не умеет удалять отдельные джобы, поэтому пересоздаётся вся задача целиком.",
)
@click.option(
    "--backup-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Куда класть JSON-бэкап разметки перед пересозданием задачи; " "по умолчанию — cvat_backup рядом с базой.",
)
@click.option("--annotator", default=None, help="Кому назначить джобы (имя пользователя CVAT).")
@_cvat_options
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def to_cvat_command(db_path: Path, cvat_url, cvat_user, cvat_password, cvat_org, log_level: str, **kwargs) -> None:
    """Уменьшенные копии в share + проект/задачи/джобы CVAT с предразметкой."""
    _set_log_level(log_level)
    params = PublishParams(
        db_path=db_path, settings=CvatSettings(cvat_url, cvat_user, cvat_password, cvat_org), **kwargs
    )
    stats = run_publish(params, open_db(db_path))
    click.echo(
        f"Картинки: готово {stats.images_done}, пропущено {stats.images_skipped}, ошибок {stats.images_failed}.\n"
        f"Задачи: создано {stats.tasks_created}, уже было {stats.tasks_existing}, "
        f"пересоздано {stats.tasks_rebuilt}. Шейпов залито: {stats.shapes}, перенесено: {stats.shapes_carried}."
    )
    if stats.stale_years:
        click.echo(
            f"Разошлись с диском годы: {', '.join(stats.stale_years)} "
            f"(изменившихся полос {stats.pages_changed}, не залитых {stats.pages_unpublished})."
            + ("" if stats.tasks_rebuilt else " Пересоздать их: --recreate-stale.")
        )


@main.command("from-cvat")
@click.option("--db", "db_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out-db",
    "out_db_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Куда писать уточнённую базу. Схема та же; пак в ней пересоздаётся целиком.",
)
@click.option("--pack-name", required=True, help="Имя пака в базе.")
@click.option("--only-year", default=None, help="Выгрузить только этот годовой комплект.")
@click.option("--full-page-frac", default=FULL_PAGE_FRAC, show_default=True, type=float)
@_cvat_options
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def from_cvat_command(
    db_path: Path, out_db_path: Path, cvat_url, cvat_user, cvat_password, cvat_org, log_level: str, **kwargs
) -> None:
    """Уточнённая разметка из CVAT в базу той же схемы."""
    _set_log_level(log_level)
    if out_db_path.resolve() == db_path.resolve():
        raise click.ClickException(
            "--out-db совпадает с --db. Пак в целевой базе пересоздаётся целиком, "
            "и исходная разметка была бы потеряна вместе с параметрами уменьшения."
        )

    params = ExportParams(
        db_path=db_path,
        out_db_path=out_db_path,
        settings=CvatSettings(cvat_url, cvat_user, cvat_password, cvat_org),
        **kwargs,
    )
    stats = run_export(params, open_db(db_path), open_db(out_db_path))
    click.echo(
        f"Полос: {stats.pages}. Растровых областей: {stats.regions} "
        f"(цветных {stats.color}, серых {stats.grayscale}, во всю полосу {stats.full_page}). "
        f"Масок печатей: {stats.masks}.\n"
        f"Шейпов с чужими метками: {stats.unknown_labels}, кадров без полосы: {stats.unmatched_frames}."
    )


if __name__ == "__main__":
    main()
