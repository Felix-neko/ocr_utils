#!/usr/bin/env python3
"""Детектор зонального (частичного) расфокуса в RAF-сканах газетных полос.

ИДЕЯ МЕТОДА
-----------
Оптический расфокус физически убивает высокие пространственные частоты, оставляя
средние. Поэтому для каждого тайла кадра считаем долю ВЧ-энергии:

    HF_ratio = E(высокие частоты) / E(средние частоты)

Эта величина почти не зависит от контраста и содержимого: и чёткий, и размытый
тело-текст имеют схожую средне-частотную энергию (шаг строк/букв), но при расфокусе
ВЧ-составляющая (тонкие штрихи) обрушивается. На карте HF_ratio зона расфокуса
видна как двумерный «провал» — в отличие от одномерных полос ВЧ-провала, которые
дают полутоновые фото и крупные заголовки.

Подробности, обоснование и валидация — в focus_detection_report.md.

ЗАВИСИМОСТИ
-----------
    pip install click opencv-python numpy
    exiftool в PATH (для извлечения встроенного JPEG-превью из RAF)

КАК ЗАПУСТИТЬ
------------
Один файл:

    python detect_zonal_defocus.py "/path/0650.RAF"

Каталог (рекурсивно по *.RAF), показать только подозрительные, в 8 потоков:

    python detect_zonal_defocus.py --only-defocus --workers 8 "/path/to/dir"

Прогон по нужной папке за 1966 год (кавычки обязательны — в пути есть пробелы и кириллица):

    python detect_zonal_defocus.py --only-defocus --workers 8 \
        "/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/ЭГ/в работе/1966/1-3 сфотали что было в Историчке"

Подсказка по всем опциям:

    python detect_zonal_defocus.py --help
"""

import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
import cv2
import numpy as np

# --- Дефолтные параметры метода (откалибровать на размеченной выборке перед массовым прогоном) ---
DEF_GRID_X, DEF_GRID_Y = 12, 8  # сетка тайлов по превью 4416x2944 (~370x370 px)
DEF_HF_ABS = 0.22  # абсолютный порог «обвала» доли ВЧ-энергии
DEF_HF_REL = 0.30  # порог относительно медианы резкости самой полосы
DEF_MIN_SEVERE = 4  # минимум «тяжёлых» тайлов в зоне
DEF_MIN_ROWS, DEF_MIN_COLS = 2, 3  # зона должна быть 2D: >=2 строк и >=3 столбцов


def extract_preview(raf_path):
    """Достаёт встроенное JPEG-превью из RAF (тег PreviewImage) во временный файл.

    Возвращает путь к временному JPEG или None, если превью пустое/отсутствует.
    Полноразмерный RAW для оценки фокуса не нужен — превью 4416x2944 хватает с запасом.
    """
    fd, jpg = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    with open(jpg, "wb") as f:
        subprocess.run(["exiftool", "-b", "-PreviewImage", raf_path], stdout=f, stderr=subprocess.DEVNULL, check=True)
    if os.path.getsize(jpg) == 0:
        os.unlink(jpg)
        return None
    return jpg


def hf_ratio_map(gray, grid_x, grid_y):
    """Карта доли ВЧ-энергии по тайлам и карта средне-частотной энергии.

    Возвращает (R, MID), где R[j, i] = HF_ratio тайла, а MID[j, i] —
    средне-частотная энергия (используется как «есть ли тут контент» для маски).
    """
    H, W = gray.shape
    R = np.zeros((grid_y, grid_x))
    MID = np.zeros((grid_y, grid_x))
    for j in range(grid_y):
        for i in range(grid_x):
            # границы текущего тайла (целочисленное деление — крайние тайлы могут чуть отличаться)
            y0, y1 = j * H // grid_y, (j + 1) * H // grid_y
            x0, x1 = i * W // grid_x, (i + 1) * W // grid_x
            t = gray[y0:y1, x0:x1].astype(np.float64)
            t -= t.mean()  # убираем постоянную составляющую (яркость фона)
            h, w = t.shape
            # окно Хэннинга гасит краевые разрывы тайла (иначе ложная ВЧ-энергия на швах)
            t = t * np.hanning(h)[:, None] * np.hanning(w)[None, :]
            # спектр мощности тайла
            F = np.abs(np.fft.rfft2(t)) ** 2
            fy = np.fft.fftfreq(h)[:, None]
            fx = np.fft.rfftfreq(w)[None, :]
            rad = np.sqrt(fy**2 + fx**2) / 0.5  # радиус в долях Найквиста (1.0 = Найквист)
            # СЧ-кольцо: общий «рисунок» текста (шаг строк/букв) — переживает расфокус
            mid = F[(rad > 0.10) & (rad <= 0.35)].sum()
            # ВЧ-кольцо: тонкие штрихи — первыми гибнут при расфокусе
            hi = F[(rad > 0.35) & (rad <= 0.85)].sum()
            MID[j, i] = mid
            R[j, i] = hi / (mid + 1e-9)
    return R, MID


def detect_array(gray, grid_x, grid_y, hf_abs, hf_rel, min_severe, min_rows, min_cols):
    """Прогоняет метод по уже загруженному grayscale-изображению.

    Возвращает (is_defocus, info_dict). Вынесено отдельно от detect(), чтобы
    метод можно было звать и на готовом массиве (тесты, отладка, другие источники).
    """
    R, MID = hf_ratio_map(gray, grid_x, grid_y)

    # маска контента: тело-текст/детали (высокая СЧ-энергия), без полей, сгиба и пустот
    content = MID > np.percentile(MID, 50)
    ref = float(np.median(R[content]))  # «здоровая» резкость именно этой полосы

    # «тяжёлые» тайлы: одновременно абсолютный обвал ВЧ и обвал относительно своей же полосы
    severe = content & (R < hf_abs) & (R < hf_rel * ref)
    ys, xs = np.where(severe)
    cnt = len(ys)

    rows = cols = 0
    bbox = None
    if cnt >= min_severe:
        rows = int(ys.max() - ys.min() + 1)
        cols = int(xs.max() - xs.min() + 1)
        bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

    # решение: это должна быть двумерная зона, а не одномерная полоса (фото/заголовок)
    is_defocus = cnt >= min_severe and rows >= min_rows and cols >= min_cols
    return is_defocus, dict(ref=ref, severe=cnt, rows=rows, cols=cols, bbox=bbox)


def detect_raf(raf_path, params):
    """Извлекает превью из RAF и прогоняет детектор. Возвращает (raf_path, is_def, info|None).

    info=None означает, что превью не нашлось. params — обычный dict с порогами,
    чтобы объект легко сериализовался для ProcessPoolExecutor.
    """
    jpg = extract_preview(raf_path)
    if jpg is None:
        return raf_path, False, None
    try:
        gray = cv2.imread(jpg, cv2.IMREAD_GRAYSCALE)
        is_def, info = detect_array(gray, **params)
    finally:
        os.unlink(jpg)
    return raf_path, is_def, info


def iter_raf(root):
    """Рекурсивно обходит *.RAF в каталоге (или отдаёт один файл)."""
    if os.path.isfile(root):
        yield root
        return
    for dp, _, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith(".raf"):
                yield os.path.join(dp, name)


def _format_line(raf, is_def, info):
    """Готовит строку отчёта по одному файлу."""
    if info is None:
        return f"[!  нет превью] {raf}"
    if is_def:
        return (
            f"[РАСФОКУС] {raf}  "
            f"ref={info['ref']:.2f} тяжёлых={info['severe']} "
            f"зона={info['rows']}x{info['cols']} тайлов bbox(r0,r1,c0,c1)={info['bbox']}"
        )
    return f"[ ok      ] {raf}  ref={info['ref']:.2f} тяжёлых={info['severe']}"


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("path", type=click.Path(exists=True))
@click.option("--grid-x", default=DEF_GRID_X, show_default=True, help="Число тайлов по горизонтали.")
@click.option("--grid-y", default=DEF_GRID_Y, show_default=True, help="Число тайлов по вертикали.")
@click.option("--hf-abs", default=DEF_HF_ABS, show_default=True, help="Абсолютный порог обвала доли ВЧ.")
@click.option("--hf-rel", default=DEF_HF_REL, show_default=True, help="Порог относительно медианы резкости полосы.")
@click.option("--min-severe", default=DEF_MIN_SEVERE, show_default=True, help="Минимум «тяжёлых» тайлов в зоне.")
@click.option("--min-rows", default=DEF_MIN_ROWS, show_default=True, help="Минимальная высота зоны (тайлов).")
@click.option("--min-cols", default=DEF_MIN_COLS, show_default=True, help="Минимальная ширина зоны (тайлов).")
@click.option("--only-defocus", is_flag=True, help="Печатать только подозрительные файлы.")
@click.option("--workers", default=1, show_default=True, help="Число параллельных процессов (для больших папок).")
def main(path, grid_x, grid_y, hf_abs, hf_rel, min_severe, min_rows, min_cols, only_defocus, workers):
    """Ищет зональный расфокус в RAF-файле или каталоге (рекурсивно по *.RAF).

    PATH — путь к .RAF-файлу или к каталогу со сканами.
    """
    params = dict(
        grid_x=grid_x,
        grid_y=grid_y,
        hf_abs=hf_abs,
        hf_rel=hf_rel,
        min_severe=min_severe,
        min_rows=min_rows,
        min_cols=min_cols,
    )
    files = list(iter_raf(path))
    total = len(files)
    defocus = 0
    no_preview = 0

    if workers <= 1:
        # последовательный режим — проще для отладки и для одиночных файлов
        results = (detect_raf(f, params) for f in files)
        for raf, is_def, info in results:
            if info is None:
                no_preview += 1
            elif is_def:
                defocus += 1
            if is_def or not only_defocus:
                click.echo(_format_line(raf, is_def, info))
    else:
        # параллельный режим — на сотнях файлов даёт кратное ускорение
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(detect_raf, f, params): f for f in files}
            for fut in as_completed(futs):
                raf, is_def, info = fut.result()
                if info is None:
                    no_preview += 1
                elif is_def:
                    defocus += 1
                if is_def or not only_defocus:
                    click.echo(_format_line(raf, is_def, info))

    # итоговая сводка всегда в конце
    click.echo(f"\nИтого: файлов={total}, с расфокусом={defocus}, без превью={no_preview}", err=True)


if __name__ == "__main__":
    main()
