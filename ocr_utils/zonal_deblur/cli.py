"""CLI: устранение зонального смаза на скане."""

from pathlib import Path

import click
import cv2
import numpy as np
from PIL import Image

from ocr_utils.zonal_deblur.deconv import deblur_plane
from ocr_utils.zonal_deblur.psf import MAX_SIGMA, estimate_blur_field, smooth_field

Image.MAX_IMAGE_PIXELS = None

SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Сторона ячейки сетки по умолчанию. Компромисс: мельче ячейка — точнее следует за
# смазом у самого края кадра, но в неё попадает меньше окон и оценка шумит. На
# развороте 3843x2927 перебор сеток дал остаточный смаз 0.19 (4x6), 0.26 (3x6),
# 0.27 (4x5) и 0.28 (5x6) при исходных 0.81 — то есть 4x6, и такая ячейка в него
# попадает. Округление вниз: лишняя строка вредит заметнее, чем недостающая.
TARGET_CELL = 640
MIN_CELLS, MAX_CELLS = 2, 8


def auto_grid(height: int, width: int, target_cell: int = TARGET_CELL) -> tuple[int, int]:
    """Подбирает размер сетки под размер кадра.

    Args:
        height: Высота кадра.
        width: Ширина кадра.
        target_cell: Желаемая сторона ячейки в пикселях.

    Returns:
        Пара (rows, cols).
    """
    rows = int(np.clip(height // target_cell, MIN_CELLS, MAX_CELLS))
    cols = int(np.clip(width // target_cell, MIN_CELLS, MAX_CELLS))
    return rows, cols


def load_image(path: Path) -> tuple[np.ndarray, dict]:
    """Читает изображение и сведения, которые надо сохранить при записи.

    Args:
        path: Путь к файлу.

    Returns:
        Пара (массив float32 в диапазоне 0..1 формы HxWx3 или HxW, метаданные).
    """
    with Image.open(path) as image:
        info = {"dpi": image.info.get("dpi"), "mode": image.mode}
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
            info["mode"] = "RGB"
        array = np.asarray(image)
    return array.astype(np.float32) / 255.0, info


def save_image(path: Path, array: np.ndarray, info: dict) -> None:
    """Записывает результат, сохраняя разрешение и сжатие исходника.

    Args:
        path: Путь для записи.
        array: Массив float32 в диапазоне 0..1.
        info: Метаданные из load_image.
    """
    data = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    image = Image.fromarray(data, mode=info["mode"])
    params: dict = {}
    if info.get("dpi"):
        params["dpi"] = info["dpi"]
    if path.suffix.lower() in (".tif", ".tiff"):
        params["compression"] = "tiff_lzw"
    image.save(path, **params)


def field_table(field) -> str:
    """Собирает текстовую таблицу оценок размытия.

    Args:
        field: Объект BlurField.

    Returns:
        Многострочный текст таблицы.
    """
    lines = ["  ячейка   sigma_max  sigma_min   угол   окон   невязка", "  " + "-" * 54]
    for cell in field.cells:
        lines.append(
            f"  r{cell.row}c{cell.col}     {cell.sigma_major:7.2f}    {cell.sigma_minor:7.2f}"
            f"  {cell.angle_deg:5.0f}°  {cell.windows:5d}   {cell.cost:8.1f}"
        )
    return "\n".join(lines)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), help="Куда записать результат.")
@click.option("--rows", type=int, default=None, help="Число строк сетки; по умолчанию по размеру кадра.")
@click.option("--cols", type=int, default=None, help="Число столбцов сетки; по умолчанию по размеру кадра.")
@click.option(
    "--method", type=click.Choice(["wiener", "rl"]), default="wiener", show_default=True, help="Способ деконволюции."
)
@click.option("--nsr", default=0.01, show_default=True, help="Шум/сигнал для Винера: больше — осторожнее.")
@click.option("--iterations", default=120, show_default=True, help="Итерации Ричардсона—Люси.")
@click.option("--extra-sigma", default=0.0, show_default=True, help="Дополнительная резкость по всему кадру, px.")
@click.option("--smooth", default=0.35, show_default=True, help="Сглаживание оценок по соседним ячейкам, 0..1.")
@click.option("--max-sigma", default=MAX_SIGMA, show_default=True, help="Потолок оценки сигмы.")
@click.option("--dry-run", is_flag=True, help="Только оценить размытие и напечатать таблицу.")
def main(
    source: Path,
    output: Path | None,
    rows: int | None,
    cols: int | None,
    method: str,
    nsr: float,
    iterations: int,
    extra_sigma: float,
    smooth: float,
    max_sigma: float,
    dry_run: bool,
) -> None:
    """Выравнивает резкость по кадру, разворачивая зональный смаз.

    Эталон резкости берётся из самого кадра, поэтому равномерно мягкий скан
    инструмент не исправит — для этого есть --extra-sigma.
    """
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise click.ClickException(f"неподдерживаемый формат: {source.suffix}")

    image, info = load_image(source)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    auto_rows, auto_cols = auto_grid(*gray.shape)
    rows = rows or auto_rows
    cols = cols or auto_cols
    click.echo(f"{source.name}: {gray.shape[1]}x{gray.shape[0]}, сетка {rows}x{cols}")
    field = estimate_blur_field(gray, rows=rows, cols=cols, max_sigma=max_sigma)
    field = smooth_field(field, strength=smooth)
    click.echo(f"эталон собран по {field.reference_windows} окнам")
    click.echo(field_table(field))

    if dry_run:
        return

    target = output or source.with_name(f"{source.stem}_deblur{source.suffix}")
    if image.ndim == 2:
        restored = deblur_plane(image, field, method, nsr, iterations, extra_sigma)
    else:
        # Смаз одинаков во всех каналах, но детали живут в яркости. Разворачивая
        # только её, мы не разводим цветную бахрому по краям букв и экономим втрое.
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = deblur_plane(ycrcb[:, :, 0], field, method, nsr, iterations, extra_sigma)
        restored = np.clip(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)

    save_image(target, restored, info)
    click.echo(f"записано: {target}")


if __name__ == "__main__":
    main()
