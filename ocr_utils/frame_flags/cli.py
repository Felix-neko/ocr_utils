"""CLI флагов кадра: листы миниатюр и пар для отсмотра, похожесть соседних кадров, запись, удаление и показ сайдкаров."""

import datetime as dt
from pathlib import Path

import click

from ocr_utils.frame_flags.flags import FlagsError, FrameFlag, FrameFlags, read_flags, remove_flags, write_flags
from ocr_utils.frame_flags.neighbors import neighbor_scores, write_scores
from ocr_utils.frame_flags.sheets import contact_sheets, pair_sheet
from ocr_utils.scan_cropping.image_io import collect_images

# Корень живой синхронизации Я.Диска: туда разрешены только сайдкары (`write`), не листы и CSV.
YANDEX_ROOT = Path("/mnt/dump3/yandex_disk_linux_baby_zergling")

# Умолчание пула для `neighbors`: кадры читаются с медленного NTFS-3G, больше воркеров упрутся в диск.
DEFAULT_JOBS = 12


def _outside_yandex(path: Path) -> Path:
    """Проверяет, что путь выхода не под корнем Я.Диска; возвращает его же.

    Raises:
        click.BadParameter: путь под корнем Я.Диска.
    """
    if path.resolve().is_relative_to(YANDEX_ROOT):
        raise click.BadParameter(f"{path}: под Я.Диском пишутся только сайдкары, листы и CSV — в scratchpad/hotstore")
    return path


@click.group()
def main() -> None:
    """Флаги кадров в сайдкарах `<кадр>.flags.json` (схема — ocr_utils/frame_flags/flags.py)."""


@main.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path), help="Куда листы.")
@click.option("--per-sheet", default=20, show_default=True, help="Кадров на лист.")
@click.option("--columns", default=5, show_default=True, help="Столбцов сетки.")
@click.option("--side", default=420, show_default=True, help="Сторона миниатюры, px.")
def sheets(input_dir: Path, out_dir: Path, per_sheet: int, columns: int, side: int) -> None:
    """Листы миниатюр папки с именами кадров и флагами их сайдкаров."""
    paths = collect_images(input_dir, recursive=False)
    for sheet in contact_sheets(paths, _outside_yandex(out_dir), per_sheet, columns, side):
        click.echo(sheet)


@main.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path), help="CSV пар.")
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True, help="Процессов пула.")
def neighbors(input_dir: Path, out: Path, jobs: int) -> None:
    """Похожесть кадров через 1 и 2 по порядку: кандидаты в повторную съёмку разворота."""
    paths = collect_images(input_dir, recursive=False)
    scores = neighbor_scores(paths, jobs)
    write_scores(scores, _outside_yandex(out))
    click.echo(f"{len(scores)} пар → {out}")


@main.command("pair-sheet")
@click.argument("frame_a", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("frame_b", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path), help="JPEG листа.")
@click.option("--width", default=1600, show_default=True, help="Ширина каждого кадра на листе, px.")
def pair_sheet_cmd(frame_a: Path, frame_b: Path, out: Path, width: int) -> None:
    """Два кадра рядом крупно — решить, какая страница поднята или закрыта."""
    click.echo(pair_sheet(frame_a, frame_b, _outside_yandex(out), width))


@main.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--flag", "flag_values", multiple=True, required=True, type=click.Choice([f.value for f in FrameFlag]))
@click.option("--note", default="", help="Почему так помечено.")
@click.option("--pair", default="", help="Имя парного кадра в той же папке.")
@click.option("--replace", is_flag=True, help="Перезаписать существующий сайдкар.")
def write(image: Path, flag_values: tuple[str, ...], note: str, pair: str, replace: bool) -> None:
    """Пишет сайдкар рядом с кадром (единственная разрешённая запись под Я.Диском)."""
    if pair and not (image.parent / pair).is_file():
        raise click.BadParameter(f"нет парного кадра {image.parent / pair}")
    flags = FrameFlags(frozenset(FrameFlag(v) for v in flag_values), note, pair, dt.date.today().isoformat())
    try:
        path = write_flags(image, flags, replace=replace)
    except (FlagsError, FileExistsError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{path}: {read_flags(image).label()}")


@main.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def remove(image: Path) -> None:
    """Удаляет сайдкар кадра (сам кадр не трогается)."""
    click.echo("удалён" if remove_flags(image) else "сайдкара не было")


@main.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--recursive", is_flag=True, help="Обойти подпапки.")
def show(input_dir: Path, recursive: bool) -> None:
    """Таблица сайдкаров папки: кадр, флаги, пара, пояснение; битые сайдкары — с ошибкой."""
    for path in collect_images(input_dir, recursive=recursive):
        try:
            flags = read_flags(path)
        except FlagsError as exc:
            click.echo(f"{path.relative_to(input_dir)}\tОШИБКА\t{exc}")
            continue
        if not flags.empty:
            click.echo(f"{path.relative_to(input_dir)}\t{flags.label()}\t{flags.pair}\t{flags.note}")
