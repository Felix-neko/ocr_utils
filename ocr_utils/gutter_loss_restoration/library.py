"""Библиотека литер выпуска: собирается один раз по всей папке.

ПОЧЕМУ ПО ПАПКЕ, А НЕ ПО ПОЛОСЕ. Литера годится в доноры, только если её границы
известны точно, а это бывает лишь когда буква отделена от соседей просветом по всей
высоте строки. В плотном наборе таких букв мало — около полутора десятков на разворот,
и на одной полосе алфавит не наберётся. Зато выпуск набран одной гарнитурой и снят за
один сеанс: полторы сотни разворотов дают по два десятка образцов на букву.

Пробовали добирать слипшиеся буквы, разрезая группу краски по минимуму профиля. Не
работает: ширина очка выходит от 3 до 168 px вместо тридцати, и такой донор превращает
вклейку в чёрную плашку. Лучше меньше литер, но верных.
"""

import json
from pathlib import Path

import numpy as np

import cv2

from ocr_utils.gutter_loss_restoration.glyphs import CUT_DOWN, CUT_UP, Glyph, ink_mask
from ocr_utils.gutter_loss_restoration.layout import align, baseline_in_band, groups, word_groups

# Сколько образцов на букву достаточно, чтобы дальше её не искать.
ENOUGH = 24

# Что вообще может понадобиться при наборе. Латиница и прочий сор распознавания в
# библиотеке не нужны: дорисовывать их всё равно не придётся.
USEFUL = set("абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЬЭЮЯ0123456789-.,;:()»«")


def harvest(
    image: np.ndarray,
    mask: np.ndarray,
    rows,
    pitch: float,
    x_from: int,
    x_to: int,
    samples: dict[str, list],
    line_from: int | None = None,
    line_to: int | None = None,
) -> None:
    """Добавляет в копилку строго разделённые буквы полосы.

    Args:
        image: Кадр BGR.
        mask: Маска краски.
        rows: Список (верх, низ, текст) строк полосы.
        pitch: Шаг строк.
        x_from: Левая граница зоны доноров.
        x_to: Правая граница зоны доноров.
        samples: Копилка, пополняется на месте.
        line_from: Левая граница ВСЕЙ наборной полосы — по ней меряется базовая линия.
        line_to: Правая граница всей наборной полосы.

    Базовая линия обязана меряться по тому же окну, что и при вклейке: оценка зависит
    от того, сколько текста попало в окно, и разные окна дают расхождение в полтора
    десятка пикселей — донор режется от одной линии, а ставится на другую.
    """
    line_from = x_from if line_from is None else line_from
    line_to = x_to if line_to is None else line_to
    for top, bottom, text in rows:
        if not text:
            continue
        paired = align(word_groups(mask, top, bottom, x_from, x_to, pitch), text)
        if not paired:
            continue
        # Базовая линия считается ТЕМ ЖЕ способом, что и при вклейке. Иначе донор режется
        # от одной линии, а ставится на другую, и всё слово садится на четверть строки
        # мимо — замерено.
        baseline = baseline_in_band(mask, top, bottom, line_from, line_to)
        for group, token in paired:
            letters = token.strip()
            runs = groups(mask, top, bottom, group.x0 - 1, group.x1 + 2)
            if not letters or len(runs) != len(letters):
                continue
            y0, y1 = baseline - CUT_UP, baseline + CUT_DOWN
            if y0 < 0 or y1 > image.shape[0]:
                continue
            for char, run in zip(letters, runs):
                if char not in USEFUL or run.width < 3 or len(samples.get(char, ())) >= ENOUGH:
                    continue
                left, right = run.x0, run.x1
                while left > run.x0 - 6 and left > 1 and mask[y0:y1, left - 1].sum() == 0:
                    left -= 1
                while right < run.x1 + 6 and right < image.shape[1] - 2 and mask[y0:y1, right + 1].sum() == 0:
                    right += 1
                patch = image[y0:y1, left : right + 1].astype(np.float32)
                level = _level(image, mask, y0, y1, left, right)
                samples.setdefault(char, []).append(
                    {
                        "ratio": patch / np.maximum(level[None, None, :], 1.0),
                        "ink_w": float(run.width),
                        "left_pad": float(run.x0 - left),
                    }
                )


def _level(image, mask, y0, y1, left, right) -> np.ndarray:
    """Уровень бумаги вокруг литеры."""
    a, b = max(0, left - 40), min(image.shape[1], right + 41)
    context = image[y0:y1, a:b].astype(np.float32)
    distance = cv2.distanceTransform(1 - mask[y0:y1, a:b], cv2.DIST_L2, 3)
    far = distance > (6 if (distance > 6).sum() > 50 else 3)
    if far.sum() < 20:
        return np.array([np.median(context[..., c]) for c in range(3)], np.float32)
    return np.array([np.median(context[..., c][far]) for c in range(3)], np.float32)


def choose(samples: dict[str, list], min_samples: int = 3) -> dict[str, Glyph]:
    """Выбирает по одному образцу на букву — по согласию остальных образцов.

    ПОЧЕМУ НЕ ПРОСТО МЕДИАНА. Сопоставление слова с группами краски изредка сбивается на
    букву, и в копилку попадает чужая литера. Один такой образец, выбранный «медианным
    по числу пикселей», отравляет всё слово: замеряли — буква «о» приехала из «п», и
    «предложения» набралось как «предлпжения». Согласие образцов такую подмену
    отбраковывает: чужая литера непохожа на остальные два десятка и в медианный
    отпечаток не попадает.

    Args:
        samples: Копилка образцов.
        min_samples: Сколько образцов нужно, чтобы букве верить.

    Returns:
        Библиотека литер.
    """
    library = {}
    for char, found in samples.items():
        if len(found) < min_samples:
            continue
        widths = np.array([s["ink_w"] for s in found], float)
        median = float(np.median(widths))
        keep = [s for s, w in zip(found, widths) if abs(w - median) <= max(2.0, 0.15 * median)]
        if len(keep) < min_samples:
            continue
        shape = (28, 20)
        stack = np.stack([cv2.resize(s["ratio"].mean(axis=2), shape[::-1]) for s in keep])
        consensus = np.median(stack, axis=0)
        distance = ((stack - consensus[None]) ** 2).mean(axis=(1, 2))
        pick = keep[int(np.argmin(distance))]
        library[char] = Glyph(ratio=pick["ratio"], ink_w=pick["ink_w"], left_pad=pick["left_pad"])
    return library


def save(path: Path, library: dict[str, Glyph]) -> None:
    """Пишет библиотеку на диск.

    Args:
        path: Куда писать (.npz).
        library: Библиотека литер.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"r_{i}": glyph.ratio for i, glyph in enumerate(library.values())}
    meta = [{"char": char, "ink_w": glyph.ink_w, "left_pad": glyph.left_pad} for char, glyph in library.items()]
    np.savez_compressed(path, meta=json.dumps(meta, ensure_ascii=False), **arrays)


def load(path: Path) -> dict[str, Glyph]:
    """Читает библиотеку с диска.

    Args:
        path: Путь к .npz.

    Returns:
        Библиотека литер.
    """
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return {
        item["char"]: Glyph(ratio=data[f"r_{i}"], ink_w=item["ink_w"], left_pad=item["left_pad"])
        for i, item in enumerate(meta)
    }
