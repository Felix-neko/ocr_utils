"""Готовит уменьшённые копии сканов для разметки в CVAT.

CVAT не умеет масштабировать кадры сам: параметр image_quality — это только
JPEG-качество пережатия, разрешение кадра всегда остаётся 1:1 с исходником
(см. ZipCompressedChunkWriter._compress_image в media_extractors.py — там нет
ни одного resize). Поэтому уменьшаем заранее, своими руками, и заводим задачи
уже из папки-результата.

Каждая картинка делится на свои собственные --divisor: размеры исходников
плавают (кадры обрезаны по-разному, примерно 4400-4900 x 3400-3800), поэтому
гнать всё в один фиксированный размер нельзя.

Перед уменьшением картинка обрезается справа и снизу до размера, кратного
--divisor. Это даёт масштаб ровно 1:divisor: без обрезки, например, 4503 -> 1125
означало бы коэффициент 4.0027, и обратный пересчёт разметки умножением на
divisor промахивался бы тем сильнее, чем правее объект. Цена — потеря не более
divisor-1 пикселя по каждой стороне (для сканов с полями это ничто).

ВАЖНО: маски и прямоугольники, размеченные на уменьшённых кадрах, окажутся в их
координатах. Чтобы применить разметку к оригиналам, координаты надо умножить
обратно на --divisor — после обрезки это точное соответствие.

Запуск:  python3 downscale_for_cvat.py [--divisor 4] [--jobs 16] [--force]
Повторный запуск безопасен: готовые файлы пропускаются, если не указан --force.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Оригиналы лежат внутри синхронизируемой папки Яндекс.Диска, поэтому результат
# кладём наружу неё — иначе 6512 новых файлов уедут в облако.
DEFAULT_SRC = "/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе"
DEFAULT_DST = "/mnt/dump3/mts_downscaled_x4"

# Ужимаем щадяще: поверх этого CVAT наложит своё image_quality=70, а пережимать
# дважды агрессивно — значит без нужды сыпать артефакты на тонкие оттиски печатей.
JPEG_QUALITY = 95


def iter_images(src: Path):
    """Пути всех картинок относительно корня src, в устойчивом порядке."""
    for path in sorted(src.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("."):
            yield path.relative_to(src)


def downscale_one(rel: Path, src: Path, dst: Path, divisor: int, force: bool = False):
    """Уменьшает одну картинку в divisor раз по каждой стороне.

    Сначала обрезает справа и снизу до размера, кратного divisor, и только потом
    уменьшает — чтобы масштаб был ровно 1:divisor, а не дробным.

    Возвращает (rel, статус, текст ошибки).
    """
    target = (dst / rel).with_suffix(".jpg")
    if target.exists() and not force:
        return rel, "skipped", ""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src / rel) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            # Обрезаем хвост справа и снизу (не более divisor-1 пикселя с каждой
            # стороны). Без этого, например, 4503 -> 1125 даёт масштаб 4.0027, и
            # обратный пересчёт разметки умножением на divisor промахивается тем
            # сильнее, чем правее объект. После обрезки масштаб ровно 1:divisor,
            # а начало координат остаётся в левом верхнем углу — поэтому режем
            # именно справа-снизу, а не по центру.
            crop_w = (im.width // divisor) * divisor
            crop_h = (im.height // divisor) * divisor
            if not crop_w or not crop_h:
                return rel, "failed", f"картинка меньше divisor: {im.size}"
            if (crop_w, crop_h) != im.size:
                im = im.crop((0, 0, crop_w, crop_h))
            im = im.resize((crop_w // divisor, crop_h // divisor), Image.Resampling.LANCZOS)
            # Пишем через временный файл и переименовываем: прерванный прогон не
            # оставит обрезанных картинок, которые повторный запуск счёл бы готовыми.
            tmp = target.with_suffix(".jpg.part")
            im.save(tmp, "JPEG", quality=JPEG_QUALITY, subsampling=0)
            tmp.replace(target)
        return rel, "done", ""
    except Exception as exc:  # noqa: BLE001
        return rel, "failed", str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Уменьшает сканы для разметки в CVAT.")
    parser.add_argument("--src", type=Path, default=Path(DEFAULT_SRC), help="папка с оригиналами")
    parser.add_argument("--dst", type=Path, default=Path(DEFAULT_DST), help="куда класть уменьшённые копии")
    parser.add_argument("--divisor", type=int, default=4, help="во сколько раз делить каждую сторону")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="число параллельных процессов")
    parser.add_argument(
        "--force",
        action="store_true",
        help="перезаписывать уже готовые файлы (нужно, если поменялась логика обработки: "
        "размеры на выходе те же, и обычный запуск всё пропустит)",
    )
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"ОШИБКА: нет папки с оригиналами: {args.src}", file=sys.stderr)
        return 1
    if args.divisor < 1:
        print("ОШИБКА: --divisor должен быть >= 1", file=sys.stderr)
        return 1

    rels = list(iter_images(args.src))
    if not rels:
        print(f"ОШИБКА: в {args.src} не найдено картинок", file=sys.stderr)
        return 1
    print(f"Найдено картинок: {len(rels)}; делим каждую сторону на {args.divisor}; процессов: {args.jobs}")
    print(f"Результат: {args.dst}" + (" (перезапись включена)" if args.force else ""))

    counts = {"done": 0, "skipped": 0, "failed": 0}
    worker = partial(downscale_one, src=args.src, dst=args.dst, divisor=args.divisor, force=args.force)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for idx, (rel, status, error) in enumerate(pool.map(worker, rels, chunksize=8), start=1):
            counts[status] += 1
            if status == "failed":
                print(f"  ОШИБКА на {rel}: {error}", file=sys.stderr)
            if idx % 200 == 0 or idx == len(rels):
                print(
                    f"  [{idx}/{len(rels)}] готово={counts['done']} пропущено={counts['skipped']} ошибок={counts['failed']}"
                )

    print(f"\nИтог: готово={counts['done']}, пропущено={counts['skipped']}, ошибок={counts['failed']}.")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
