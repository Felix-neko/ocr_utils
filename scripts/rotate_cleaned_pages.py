"""Разовый поворот уже готовых копий полос по отметке ``pages.rotate_cw`` из базы.

ЗАЧЕМ ОТДЕЛЬНЫМ СКРИПТОМ. Штатное место поворота — шаг очистки (``scan_cleanup``), он
пишет полосу уже развёрнутой. Но `blurred` и `sharpened` пака-1 собраны прошлыми прогонами,
и перегонять триста гигабайт ради сорока двух полос несоразмерно.

ПОВОРОТ TIFF — без потерь: LZW, декодирование и кодирование обратимы, ``np.rot90`` не
интерполирует.

ПОВОРОТ JPEG — с перекодированием, и это осознанный выбор. Без потерь развернуть JPEG,
размеры которого не кратны MCU, нельзя в принципе: реальные пиксели краевого блока обязаны
лежать в его левом верхнем углу, а поворот их оттуда уводит. ``jpegtran -perfect`` на таких
размерах отказывается, ``-trim`` теряет 7 px содержимого, а обход через дополнение до MCU
даёт лишний столбец. Требование «размер обязан поменяться местами точно» перевешивает.

Цена перекодирования сведена к минимуму ПЕРЕИСПОЛЬЗОВАНИЕМ ТАБЛИЦ КВАНТОВАНИЯ ИСХОДНИКА.
Замер на 1967/01/IMG_0043_2R.jpg (3420x6071, 4:4:4):

    настройка                     отличается пикселей   макс   средн
    q95, 4:2:0 (умолчание PIL)          49.8%            56     0.96
    q95, 4:4:4                          47.6%            27     0.85
    q98, 4:4:4                          37.2%            10     0.57
    таблицы исходника, 4:4:4            18.1%             6     0.24

Умолчание PIL вредно вдвойне: оно роняет цветность в 4:2:0, хотя исходники 4:4:4. Таблицы и
субдискретизация читаются У КАЖДОГО файла: полосы кодировал Capture One, и параметры у
разных снимков могут отличаться. Не прочитались — полоса пропускается с ошибкой, а не
перекодируется вслепую: тихая потеря качества дороже остановки.

ИДЕМПОТЕНТНОСТЬ по размеру: полоса, у которой ширина и высота уже поменяны местами
относительно записанных в базе, считается повёрнутой и пропускается.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click
import numpy as np
from PIL import Image

from ocr_utils.db.repo import iter_pages, require_pack
from ocr_utils.db.session import open_db
from ocr_utils.scan_markup.rotation import ROTATION_NAMES, rotate_cw

Image.MAX_IMAGE_PIXELS = None

#: Расширение -> как сохранять. TIFF пишется LZW, как и весь конвейер.
TIFF_SUFFIXES = {".tif", ".tiff"}


class RotateError(RuntimeError):
    """Полосу повернуть нельзя — и молча пропустить это тоже нельзя."""


def rotate_file(path: Path, degrees: int, expected: "tuple[int, int]") -> str:
    """Поворачивает файл на месте. Возвращает словесный итог для отчёта."""
    with Image.open(path) as image:
        size = image.size
        if size == (expected[1], expected[0]):
            return "уже повёрнут"
        if size != expected:
            raise RotateError(f"размер {size} не совпадает ни с базой {expected}, ни с повёрнутым")
        dpi = image.info.get("dpi")
        qtables = getattr(image, "quantization", None)
        subsampling = _subsampling(image)
        pixels = np.asarray(image)

    turned = rotate_cw(pixels, degrees)
    tmp = path.with_name(f".{path.stem}.part{path.suffix}")
    result = Image.fromarray(turned)
    if path.suffix.lower() in TIFF_SUFFIXES:
        result.save(tmp, format="TIFF", compression="tiff_lzw", **({"dpi": dpi} if dpi else {}))
    else:
        if not qtables:
            raise RotateError("не прочитались таблицы квантования — перекодировать вслепую нельзя")
        result.save(
            tmp, format="JPEG", qtables=qtables, subsampling=subsampling, optimize=True, **({"dpi": dpi} if dpi else {})
        )
    tmp.replace(path)
    return f"{size} -> {Image.open(path).size}"


def _subsampling(image: Image.Image) -> int:
    """Субдискретизация исходника; 0 — 4:4:4. Читается, а не задаётся константой."""
    from PIL import JpegImagePlugin

    if image.format != "JPEG":
        return 0
    try:
        return JpegImagePlugin.get_sampling(image)
    except Exception:  # noqa: BLE001 — не JPEG или экзотический вариант
        return 0


@click.command()
@click.option("--db", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--pack-name", default="пак-1", show_default=True)
@click.option("--blurred-root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sharpened-root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--backup-dir", type=click.Path(file_okay=False, path_type=Path), help="куда сложить копии до правки")
@click.option("--apply", is_flag=True, help="без него — только показать, что будет сделано")
def main(db, pack_name, blurred_root, sharpened_root, backup_dir, apply) -> None:
    """Повернуть готовые копии полос, отмеченных в базе как нуждающиеся в повороте."""
    with open_db(db)() as session:
        pack = require_pack(session, pack_name)
        targets = [
            (page.source_rel_path, page.rotate_cw, int(page.width), int(page.height), page.sharpened_text_pic_rel_path)
            for _y, _i, page in iter_pages(pack)
            if page.rotate_cw
        ]
    click.echo(f"Полос под поворот: {len(targets)}")

    done = skipped = failed = 0
    for rel, degrees, width, height, sharp_rel in targets:
        for root, path in _files(rel, sharp_rel, blurred_root, sharpened_root):
            if not path.exists():
                click.echo(f"  НЕТ ФАЙЛА {path}")
                failed += 1
                continue
            try:
                if not apply:
                    with Image.open(path) as image:
                        state = (
                            "уже повёрнут" if image.size == (height, width) else f"{image.size} -> {(height, width)}"
                        )
                    click.echo(f"  [{root}] {path.relative_to(path.parents[2])} {ROTATION_NAMES[degrees]}: {state}")
                    continue
                if backup_dir is not None:
                    copy = Path(backup_dir) / root / path.name
                    copy.parent.mkdir(parents=True, exist_ok=True)
                    if not copy.exists():
                        shutil.copy2(path, copy)
                state = rotate_file(path, degrees, (width, height))
                click.echo(f"  [{root}] {path.relative_to(path.parents[2])}: {state}")
                skipped += state == "уже повёрнут"
                done += state != "уже повёрнут"
            except (RotateError, OSError) as error:
                click.echo(f"  ОШИБКА [{root}] {path}: {error}")
                failed += 1
    click.echo(f"Повёрнуто: {done}, уже было: {skipped}, ошибок: {failed}.")
    if failed:
        sys.exit(1)


def _files(rel: str, sharp_rel: "str | None", blurred_root, sharpened_root):
    if blurred_root is not None:
        yield "blurred", Path(blurred_root) / rel
    if sharpened_root is not None and sharp_rel:
        yield "sharpened", Path(sharpened_root) / sharp_rel


if __name__ == "__main__":
    main()
