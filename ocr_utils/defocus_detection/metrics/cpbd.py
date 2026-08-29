"""CPBD — Cumulative Probability of Blur Detection.

Narvekar, Karam, «A no-reference perceptual image sharpness metric based on a cumulative
probability of blur detection», QoMEX 2009.

ЧЕМ ЦЕННА ИМЕННО ЗДЕСЬ. Это единственная метрика набора, у которой балл — АБСОЛЮТНАЯ
величина с внятным смыслом: доля переходов, на которых человек размытия не заметил бы.
Ноль — размытие видно на каждом штрихе, единица — ни на одном. Прочие наши метрики дают
либо пиксели (зависят от разрешения и кегля), либо безымянные отношения, и порог на них
приходится подбирать заново под каждую съёмку. На долю — не приходится.

КАК УСТРОЕНО. У каждого перехода меряется ширина w (расстояние между экстремумами) и
контраст C. Психофизика говорит, что порог заметности размытия зависит от контраста: на
контрастном переходе глаз замечает размытие раньше. Эта «едва заметная ширина» w_JNB и
берётся как 5 px при слабом контрасте и 3 px при сильном. Дальше вероятность заметить
размытие на переходе растёт по Вейбуллу от отношения w/w_JNB, и CPBD — это доля переходов,
у которых она не дотянула до порога заметности 0.63.

ОТСТУПЛЕНИЕ ОТ ОРИГИНАЛА, БЕЗ КОТОРОГО МЕТРИКА НЕМОНОТОННА. У авторов граница между
«слабым» и «сильным» контрастом — абсолютные 50 уровней из 255. В их задаче контраст
задан сценой, у нас же его снижает само размытие, и на границе критерий скачком
смягчается: замерено на синтетике, что при σ=2 медианный контраст падает до 43, w_JNB
прыгает с 3 на 5, и БОЛЕЕ размытый кадр получает балл ВЫШЕ (0.135 при σ=1.2 против 0.303
при σ=2.0). Поэтому контраст перехода здесь сравнивается не с абсолютным числом, а с
контрастом самого кадра. Побочная выгода ровно та, что нужна: порог перестаёт зависеть
от освещения, которое в съёмке с рук гуляет от кадра к кадру.

ЧЕСТНАЯ ОГОВОРКА. Константы w_JNB, β и порог 0.63 откалиброваны авторами на натурных
фотографиях и типичной дистанции просмотра, а не на газетном петите в превью 4416×2944.
Поэтому «абсолютность» шкалы здесь — свойство конструкции, а не гарантия того, что порог
0.63 означает для нас то же самое. Проверять переносимость всё равно надо на разметке.

Ширина и контраст переходов берутся у ``edge_width``: там это уже посчитано векторно
на весь кадр за один проход, и заводить второй такой проход незачем.
"""

import numpy as np

from ocr_utils.defocus_detection.metrics import edge_width
from ocr_utils.defocus_detection.metrics.base import Algorithm
from ocr_utils.defocus_detection.tiles import Grid

# Едва заметная ширина края в пикселях: на контрастном переходе глаз чувствительнее,
# и допустимая ширина меньше.
JNB_WIDTH_LOW = 5.0
JNB_WIDTH_HIGH = 3.0
# Граница между «слабым» и «сильным» контрастом — В ДОЛЯХ контраста кадра (p95 амплитуд
# перепадов), а не в абсолютных уровнях. Значение 0.25 воспроизводит оригинальные
# 50 из 255 на кадре с типичной для типографики амплитудой около 200 уровней.
CONTRAST_SPLIT_REL = 0.25

# Показатель Вейбулла из статьи.
BETA = 3.6
# Вероятность, выше которой размытие считается заметным (P_JNB в оригинале).
P_JNB = 0.63

# Минимум переходов в тайле, иначе доля считается по слишком малой выборке.
DEFAULT_MIN_EDGES = 200


def contrast_scale(gray: np.ndarray) -> tuple[float, float]:
    """Контекст кадра: порог амплитуды и масштаб контраста.

    Оба обязаны считаться по ВСЕМУ кадру и передаваться в замеры кусков готовыми: в
    кропе одной строки p95 амплитуд — это разброс внутри букв, а не контраст типографики.

    Args:
        gray: Полутоновый кадр.

    Returns:
        Пара (порог амплитуды, p95 амплитуд перепадов) в уровнях 8 бит.
    """
    img = gray.astype(np.float64)
    probe = img[:: max(1, img.shape[0] // 400)]
    p95 = float(np.percentile(edge_width._row_runs(probe)["amplitude"], 95))
    threshold = max(edge_width.DEFAULT_AMP_MIN, edge_width.DEFAULT_AMP_REL * p95)
    return threshold, max(p95, 1e-6)


def edge_table(gray: np.ndarray, context: tuple[float, float] | None = None) -> dict[str, np.ndarray]:
    """Плоская таблица переходов кадра: координаты, ширина, контраст.

    Args:
        gray: Полутоновый кадр.
        context: Готовая пара (порог амплитуды, масштаб контраста); None — посчитать по кадру.

    Returns:
        Словарь с массивами ``y``, ``x``, ``width``, ``contrast`` и числом ``scale``.
    """
    img = gray.astype(np.float64)
    amp_threshold, scale = contrast_scale(gray) if context is None else context

    runs = edge_width._row_runs(img)
    # Отсечки по длине перехода здесь НЕТ, и это принципиально. ``edge_width`` отбрасывает
    # переходы длиннее двенадцати пикселей как плавные растяжки, и для оценки ширины края
    # это правильно. Но CPBD считает ДОЛЮ незаметно размытых переходов, и выбросить длинные
    # — значит выбросить ровно те, что доказывают размытие: в замере остаются короткие
    # внутрибуквенные, и балл при росте размытия начинает РАСТИ (поймано тестом
    # монотонности: 1.0 → 0.059 → 0.303). Длинный переход — это и есть размытый переход,
    # его вероятность заметности сама получится равной единице.
    keep = runs["start"] & (runs["amplitude"] >= amp_threshold) & np.isfinite(runs["sigma"])
    ys, xs = np.nonzero(keep)
    return dict(y=ys, x=xs, width=runs["run"][keep].astype(np.float64), contrast=runs["amplitude"][keep], scale=scale)


def blur_probability(width: np.ndarray, contrast: np.ndarray, scale: float) -> np.ndarray:
    """Вероятность заметить размытие на каждом переходе.

    Args:
        width: Ширины переходов в пикселях.
        contrast: Перепады яркости на них в уровнях 8 бит.
        scale: Контраст кадра (p95 амплитуд) — относительно него и решается, считать
            переход контрастным или нет.

    Returns:
        Массив вероятностей той же длины.
    """
    jnb = np.where(contrast > CONTRAST_SPLIT_REL * scale, JNB_WIDTH_HIGH, JNB_WIDTH_LOW)
    with np.errstate(over="ignore"):
        return 1.0 - np.exp(-np.power(np.maximum(width, 0.0) / jnb, BETA))


def _tile_sharpness(gray: np.ndarray, grid: Grid) -> np.ndarray:
    """Карта CPBD по тайлам (больше = резче, шкала [0, 1]).

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.

    Returns:
        Массив (ny, nx); NaN там, где переходов не хватило.
    """
    table = edge_table(gray)
    if table["y"].size == 0:
        return np.full((grid.ny, grid.nx), np.nan)

    unblurred = blur_probability(table["width"], table["contrast"], table["scale"]) <= P_JNB
    # Раскладываем переходы по тайлам одним проходом: делить кадр на куски и звать
    # детектор краёв на каждый было бы в разы дороже, а результат тот же.
    iy = np.clip((table["y"] * grid.ny) // grid.height, 0, grid.ny - 1)
    ix = np.clip((table["x"] * grid.nx) // grid.width, 0, grid.nx - 1)
    flat = iy * grid.nx + ix
    size = grid.ny * grid.nx
    total = np.bincount(flat, minlength=size).astype(np.float64)
    good = np.bincount(flat, weights=unblurred.astype(np.float64), minlength=size)

    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total >= DEFAULT_MIN_EDGES, good / np.maximum(total, 1.0), np.nan)
    return share.reshape(grid.ny, grid.nx)


def _region_sharpness(crop: np.ndarray, context: object) -> tuple[float, float]:
    """CPBD одного куска строки и число переходов как вес.

    Args:
        crop: Полутоновый кусок строки.
        context: Пара (порог амплитуды, масштаб контраста) по всему кадру.

    Returns:
        Пара (доля незаметно размытых переходов, их число).
    """
    table = edge_table(crop, context=context)
    count = table["y"].size
    if count == 0:
        return float("nan"), 0.0
    unblurred = blur_probability(table["width"], table["contrast"], table["scale"]) <= P_JNB
    return float(unblurred.mean()), float(count)


ALGORITHM = Algorithm(
    name="cpbd",
    summary="CPBD: доля переходов, на которых размытие незаметно глазу — абсолютная шкала [0,1]",
    tile_sharpness=_tile_sharpness,
    unit="CPBD",
    region_sharpness=_region_sharpness,
    frame_context=contrast_scale,
)
