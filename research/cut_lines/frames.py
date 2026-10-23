"""Кадры стенда: пути двух фотосетов РИСК 1991/05, кэш scan_cropping v2, выход на SSD, чтение кадров, воркер пула."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import threadpoolctl

# Основа (1): пак-6, развороты 5687×4268, у многих страниц край строки срезан краем листа (Я.Диск, только чтение).
BASE_DIR = Path(
    "/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-6 от Лены (1989-1992)/РИСК/1991/"
    "05 есть фотографии по страницам"
)
# Донор (2): пак-7, развороты 4622×3584, края целые, у корешка текст убит (Я.Диск, только чтение).
DONOR_DIR = Path("/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-7 от Бори/РИСК/1991/05")
# Кэш прогона scan_cropping v2 по (1): маски книги, бумаги и страниц SAM 3 на ½ кадра.
V2_WORK = Path("/mnt/hotstore/scan_processing/scan_cropping_v2/risk_1991_05/work")
# Выход стенда (SSD).
OUT_ROOT = Path("/mnt/hotstore/scan_processing/cut_lines/risk_1991_05")


def init_worker() -> None:
    """Инициализатор воркера: без hugepage и потоков BLAS/OpenCV (см. ``.claude/rules/gpu_and_pools.md``)."""
    np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def read_bgr(path: Path) -> np.ndarray:
    """Читает картинку в BGR по байтам (путь с кириллицей и пробелами).

    Args:
        path: Путь к картинке.

    Returns:
        Массив ``(H, W, 3)`` uint8.

    Raises:
        ValueError: Картинка не читается.
    """
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не читается: {path}")
    return img


def frames_of(folder: Path) -> list[Path]:
    """Кадры папки в порядке имён (только JPEG верхнего уровня, без скрытых папок).

    Args:
        folder: Папка фотосета.

    Returns:
        Пути кадров.
    """
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
