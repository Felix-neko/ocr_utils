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
from ocr_utils.scan_markup.detection.color_kind import CHROMA_THR, COLOR_FRAC_THR
from ocr_utils.scan_markup.detection.raster import FULL_PAGE_FRAC, MERGE_GAP_PX, MIN_REGION_FRAC
from ocr_utils.scan_markup.detection.run import DetectParams, run_detect
from ocr_utils.scan_markup.geometry import CVAT_DPI

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
    help="Добавлять к найденным полутоновым областям блоки Picture из Surya. "
    "Без флага не нужен GPU, но светлые фотографии будут теряться.",
)
@click.option("--chroma-thr", default=CHROMA_THR, show_default=True, type=float, help="Порог хроматичности в Lab.")
@click.option(
    "--color-frac-thr",
    default=COLOR_FRAC_THR,
    show_default=True,
    type=float,
    help="Доля хроматичных пикселей, с которой область считается цветной.",
)
@click.option(
    "--min-region-frac",
    default=MIN_REGION_FRAC,
    show_default=True,
    type=float,
    help="Минимальная доля площади полосы для растровой области.",
)
@click.option(
    "--merge-gap", default=MERGE_GAP_PX, show_default=True, type=int, help="Зазор слияния областей (копия 1/4)."
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
    help="Куда писать оверлеи с найденными областями и подписями kind/chroma_frac.",
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
