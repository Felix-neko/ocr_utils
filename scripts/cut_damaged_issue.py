"""Нарезка 16 фото-разворотов МТС 1991/02 на страницы — вход «плохого выпуска» для ``scripts/debug_bad_issue.py``.

Одноразовый скрипт без аргументов: пути и ручные поправки линии сгиба заданы ниже. Развороты
режутся по прямой сгиба через ``research.external_ocr_models.spreads`` (автоподгонка
``find_fold``; там, где она промахивается, линия задаётся руками в ``FOLDS`` по контрольной
картинке ``_линии/<кадр>.jpg``). Сырые файлы на Я.Диске только читаются.

Выход — ``research/external_ocr_models/damaged_issue/1991/02/IMG_<кадр>_<L|R>.jpg``: имена
сортируются в порядке чтения, как того требует ``run_issue``. У обложки берётся только правая
страница (слева — задняя обложка предыдущего выпуска).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PIL import Image

# ``research`` — не установленный пакет, а папка репо: при запуске файлом из PyCharm корень надо добавить руками.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.external_ocr_models.spreads import Fold, find_fold, fold_from_points, preview, split_spread

logger = logging.getLogger(__name__)

RAW_DIR = Path(
    "/mnt/dump3/yandex_disk_linux_baby_zergling/Пробное/Отобранное/МТО/1991/"
    "02 оплачено с.92 битая сравнить с подшивкой из Ленинки попытаться восстановить"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "research" / "external_ocr_models" / "damaged_issue"
ISSUE_DIR = OUT_DIR / "1991" / "02"
PREVIEW_DIR = OUT_DIR / "_линии"  # папки с «_» обход пайплайна пропускает

# Кадры выпуска по порядку чтения: обложка, реклама + «Содержание» (2 полосы), с. 3–9, с. 46–47
# («Евролизинг предлагает услуги», размазанная печать на с. 46), с. 82–100 + реклама.
FRAMES = [622, 624, 625, 626, 627, 628, 648, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677]
# Кадры, у которых нужна только правая страница (обложка: слева задняя обложка предыдущего выпуска).
RIGHT_ONLY = {622}
# Ручная линия сгиба по имени кадра: (x сверху, x снизу) в пикселях исходника, если автоподгонка
# промахнулась — проверяется глазами по _линии/*.jpg.
FOLDS: dict[str, tuple[float, float]] = {
    # Автоподгонка здесь садится на межколонный пробел (x≈3200–3900 вместо ≈2780): у сгиба
    # нет своего минимума краски. Линии — по тёмной полосе тени в центре кадра, проверены глазами.
    "IMG_0622.jpg": (2769, 2765),
    "IMG_0624.jpg": (2742, 2750),
    "IMG_0625.jpg": (2745, 2735),
    "IMG_0626.jpg": (2770, 2775),
    "IMG_0627.jpg": (2737, 2755),
    "IMG_0628.jpg": (2790, 2785),
    "IMG_0648.jpg": (2820, 2815),
    "IMG_0668.jpg": (2752, 2730),
    "IMG_0669.jpg": (2765, 2725),
    "IMG_0670.jpg": (2769, 2751),
    "IMG_0671.jpg": (2771, 2745),
    "IMG_0672.jpg": (2777, 2759),
    "IMG_0673.jpg": (2783, 2765),
    "IMG_0674.jpg": (2760, 2770),
    "IMG_0675.jpg": (2773, 2769),
    "IMG_0676.jpg": (2780, 2795),
    "IMG_0677.jpg": (2786, 2764),
}
QUALITY = 92


def fold_for(path: Path) -> Fold:
    """Линия сгиба кадра: ручная из ``FOLDS``, иначе автоподгонка по тёмному следу между полосами.

    Args:
        path: Файл разворота.

    Returns:
        Прямая сгиба в координатах исходного кадра.
    """
    if path.name in FOLDS:
        with Image.open(path) as image:
            height = image.height
        return fold_from_points(*FOLDS[path.name], height)
    return find_fold(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ISSUE_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for frame in FRAMES:
        path = RAW_DIR / f"IMG_{frame:04d}.jpg"
        fold = fold_for(path)
        with Image.open(path) as image:
            image.load()
            left, right = split_spread(image, fold)
            # Контрольная картинка с линией — проверить глазами, что сгиб не режет текст.
            preview(image, fold).save(PREVIEW_DIR / f"{path.stem}.jpg", quality=80)
            height = image.height
        sides = [("R", right)] if frame in RIGHT_ONLY else [("L", left), ("R", right)]
        for side, page in sides:
            page.save(ISSUE_DIR / f"{path.stem}_{side}.jpg", quality=QUALITY)
        logger.info("%s: сгиб x=%.0f…%.0f, страниц %d", path.name, fold.x_at(0), fold.x_at(height), len(sides))
    logger.info("готово: %s", ISSUE_DIR)


if __name__ == "__main__":
    main()
