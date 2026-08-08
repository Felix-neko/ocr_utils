"""Сведение карты тайлов в один балл файла и объединение метрик в сводный ранг.

САМОЕ ВАЖНОЕ РЕШЕНИЕ ЗДЕСЬ — способ агрегации. По умолчанию (``worst``) балл файла —
это резкость его САМЫХ МЯГКИХ 20 % печатных тайлов, а не среднее и не медиана.

Почему не медиана: она заложница вёрстки. Тайлы с крупным заголовком, полутоновым фото
или широким полем дают низкую резкость и при идеальном фокусе, так что медиана меряет
пополам фокус и раскладку полосы.

Почему не «лучшие тайлы»: логика «есть ли на полосе хоть один участок чёткого мелкого
текста» выглядит убедительно, но на размеченной выборке проигрывает — AUC 0.79 против
0.90. Причина в том, что реальные промахи фокуса на съёмке с рук почти всегда
неравномерны по кадру: сначала уплывает часть полосы, и видно это именно по мягкому
краю распределения. Режим ``best`` оставлен как опция.

Почему не «самый худший тайл» (квантиль 0): один тайл — это шум и артефакты вёрстки,
AUC проваливается до 0.59–0.81. Пятая часть тайлов — уже связная область кадра.
"""

import numpy as np

# Доля тайлов, попадающих в агрегацию. Подобрана на размеченной папке 1979 года: плато
# AUC 0.87–0.90 держится при 0.7–0.9, острый провал к 1.0 (один тайл) и к 0.1 (медиана).
DEFAULT_QUANTILE = 0.80

AGGREGATIONS = ("best", "median", "worst")
DEFAULT_AGGREGATION = "worst"


def aggregate(
    tile_map: np.ndarray, printed: np.ndarray, mode: str = DEFAULT_AGGREGATION, quantile: float = DEFAULT_QUANTILE
) -> float:
    """Сводит карту резкости по тайлам в один балл файла.

    Args:
        tile_map: Карта резкости (ny, nx), больше = резче; NaN — тайл не измерен.
        printed: Булева маска тайлов с краской.
        mode: "worst" — квантиль самых мягких тайлов (по умолчанию, ловит и общий промах,
            и неравномерный), "median" — медиана по печатным тайлам,
            "best" — квантиль самых резких тайлов («есть ли на полосе чёткий текст»).
        quantile: Какую долю тайлов брать в режимах "worst"/"best".

    Returns:
        Балл резкости файла (больше = резче) либо NaN, если измеримых тайлов нет.
    """
    values = tile_map[printed & np.isfinite(tile_map)]
    if values.size == 0:
        # Полоса без краски (обложка, пустой лист) — считаем по всем измеримым тайлам,
        # чтобы файл не выпал из отчёта совсем; ранг его всё равно будет условным.
        values = tile_map[np.isfinite(tile_map)]
    if values.size == 0:
        return float("nan")
    if mode == "median":
        return float(np.median(values))
    if mode == "worst":
        return float(np.quantile(values, 1.0 - quantile))
    return float(np.quantile(values, quantile))


def rank_combine(scores_by_metric: dict[str, list[float]]) -> list[float]:
    """Сводит баллы нескольких метрик в один средний ранг (режим ``combo``).

    Шкалы у метрик несопоставимы (пиксели, доли энергии, [0,1]), а порядок файлов —
    вполне, поэтому объединяем именно ранги. Ранг нормирован в [0, 1], где 0 — самый
    мягкий файл выборки, 1 — самый резкий; связки получают средний ранг.

    Args:
        scores_by_metric: Отображение «имя метрики -> список баллов по файлам»
            (одинаковой длины, порядок файлов общий). NaN допустимы.

    Returns:
        Список сводных баллов в [0, 1] той же длины; NaN там, где ни одна метрика
        не смогла оценить файл.
    """
    names = list(scores_by_metric)
    n = len(scores_by_metric[names[0]])
    accumulated = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)

    for name in names:
        values = np.asarray(scores_by_metric[name], dtype=np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        ranks = np.full(n, np.nan)
        # Средний ранг для одинаковых значений: сортируем, затем усредняем позиции связок.
        order = np.argsort(values[finite], kind="mergesort")
        positions = np.empty(order.size, dtype=np.float64)
        positions[order] = np.arange(order.size, dtype=np.float64)
        sorted_values = values[finite][order]
        start = 0
        for end in range(1, order.size + 1):
            if end == order.size or sorted_values[end] != sorted_values[start]:
                mean_pos = (start + end - 1) / 2.0
                positions[order[start:end]] = mean_pos
                start = end
        ranks[finite] = positions / max(order.size - 1, 1)
        accumulated[finite] += ranks[finite]
        weights[finite] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        combined = np.where(weights > 0, accumulated / np.maximum(weights, 1.0), np.nan)
    return [float(v) for v in combined]
