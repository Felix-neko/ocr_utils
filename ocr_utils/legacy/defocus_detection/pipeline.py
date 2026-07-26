"""Оркестрация: прогон выбранного метода по тайлам + нормировка (A), зона (C),
кросс-проверка FFT (E), и сведение двух детекторов в единый вердикт.

Гейт обложек (B) применяется на уровне CLI (по полю edge_density), т.к. это решение
о ВКЛЮЧЕНИИ файла в ранжирование, а не метрика отдельного скана.
"""

import numpy as np

from ocr_utils.legacy.defocus_detection.fft_hf import (
    DEF_GRID_X as FFT_GRID_X,
    DEF_GRID_Y as FFT_GRID_Y,
    DEF_HF_ABS as FFT_HF_ABS,
    DEF_HF_REL as FFT_HF_REL,
    DEF_MIN_COLS as FFT_MIN_COLS,
    DEF_MIN_ROWS as FFT_MIN_ROWS,
    DEF_MIN_SEVERE as FFT_MIN_SEVERE,
    detect_array as fft_detect_array,
)
from ocr_utils.legacy.defocus_detection.laplacian import laplacian_tile_maps
from ocr_utils.legacy.defocus_detection.moire import (
    center_std,
    find_defocus_zone,
    gradient_tile_map,
    moire_tile_maps,
    raster_edge_density,
)

# Уровни уверенности при кросс-проверке (этап E): чем меньше число — тем выше в списке.
VERDICTS = {
    "both": (0, "РАСФОКУС ✓✓ (муар+FFT)"),
    "fft": (1, "расфокус ✓ FFT (муар не увидел)"),
    "moire": (2, "? только муар (возм. цв.декор/край)"),
    "ok": (3, ""),
}


def verdict(res: dict, cross_check: bool) -> str:
    """Сводный вердикт по двум детекторам (этап E): both / fft / moire / ok.

    Args:
        res: Результат analyze().
        cross_check: Учитывать ли голос FFT-детектора.

    Returns:
        Ключ из VERDICTS.
    """
    has_moire = res["zone"] is not None
    has_fft = bool(res.get("fft_defocus")) if cross_check else False
    if has_moire and has_fft:
        return "both"
    if has_fft:
        return "fft"
    if has_moire:
        return "moire"
    return "ok"


def analyze(
    gray: np.ndarray,
    method: str,
    factor: float,
    grid_x: int,
    grid_y: int,
    min_structure: float,
    normalize: str = "structure",
    zone_params: dict | None = None,
    cross_check: bool = False,
) -> dict:
    """Анализирует изображение и возвращает метрики резкости/расфокуса.

    Args:
        gray: Полутоновое изображение.
        method: "moire" или "laplacian".
        factor: Коэффициент уменьшения для метода moire.
        grid_x: Число тайлов по горизонтали.
        grid_y: Число тайлов по вертикали.
        min_structure: Порог локального контраста (std), ниже которого тайл
            считается пустым полем и исключается из статистики.
        normalize: Нормировка метрики moire на «количество резких переходов»
            в тайле (убирает зависимость от количества краски/текста):
            "none" — сырая энергия муара (прежнее поведение);
            "structure" (A1) — делить на std AREA-уменьшения тайла;
            "gradient" (A2) — делить на RMS полноразмерного градиента тайла;
            "global_contrast" (A1+) — structure + дополнительная нормировка на
                центральный std изображения (убирает зависимость от общего
                динамического диапазона/экспозиции).
            Для метода laplacian игнорируется.
        zone_params: Параметры поиска 2D-зоны расфокуса (этап C): ключи margin,
            k_abs, k_rel, g_rel, min_rows, min_cols. Если None или метод не moire
            с нормировкой — зона не ищется.
        cross_check: Запустить второй независимый детектор — FFT HF/MID (fft_hf) —
            как кросс-проверку зоны расфокуса (этап E).

    Returns:
        Словарь с ключами:
        sharpness — медиана метрики по печатным тайлам (выше = резче),
        worst_zone — среднее по 10% самых «мягких» печатных тайлов (для зон),
        edge_density — плотность краёв (этап B, прокси наличия растра),
        zone — найденная 2D-зона расфокуса (dict) или None (этап C),
        inner_median — «здоровый» уровень полосы (медиана ratio внутренних тайлов),
        fft_defocus — вердикт FFT-детектора HF/MID (bool) или None (этап E),
        fft_info — детали FFT-зоны (dict) или None,
        n_printed — число печатных тайлов,
        sharp_map, structure_map, printed_mask — карты для визуализации.
    """
    if method == "moire":
        sharp_map, structure = moire_tile_maps(gray, factor, grid_x, grid_y)
    else:
        sharp_map, structure = laplacian_tile_maps(gray, grid_x, grid_y)

    # A1/A2/A3: нормируем энергию муара на меру «сколько в тайле резких переходов».
    # Так метрика отражает фокус, а не количество краски на полосе. Деление идёт по
    # всем тайлам, но непечатные (пустые поля) всё равно отсекаются маской printed
    # ниже — это и есть гейт A3 (нормируем только там, где есть контент).
    grad = None
    if method == "moire" and normalize != "none":
        grad = gradient_tile_map(gray, grid_x, grid_y)
        denom = structure if normalize in ("structure", "global_contrast") else grad
        sharp_map = sharp_map / np.maximum(denom, 1e-6)

        # A1+: дополнительная нормировка на общий динамический диапазон изображения.
        # Убирает зависимость от экспозиции/контраста: сканы с высоким контрастом
        # дают высокий муар даже при хорошем фокусе, и наоборот.
        if normalize == "global_contrast":
            cstd = center_std(gray)
            # Умножаем на обратный коэффициент: изображения с низким контрастом
            # (часто расфокусные) получают понижающий коэффициент, с высоким — повышающий.
            # Делитель 35.0 подобран эмпирически как типичное значение центрального std.
            sharp_map = sharp_map * (35.0 / np.maximum(cstd, 10.0))

    # B: гейт наличия растра (плотность краёв) — отсев обложек/пустых листов.
    edge_density = raster_edge_density(gray)

    # C: связная 2D-зона расфокуса по нормированному муару (только для moire+нормировки).
    zone, inner_median = None, float("nan")
    if zone_params is not None and method == "moire" and normalize != "none":
        if grad is None:
            grad = gradient_tile_map(gray, grid_x, grid_y)
        zone, inner_median = find_defocus_zone(
            sharp_map,
            structure,
            grad,
            min_structure,
            margin=zone_params["margin"],
            k_abs=zone_params["k_abs"],
            k_rel=zone_params["k_rel"],
            g_rel=zone_params["g_rel"],
            min_rows=zone_params["min_rows"],
            min_cols=zone_params["min_cols"],
        )

    # E: второй независимый детектор (FFT HF/MID) на том же превью — для кросс-проверки.
    fft_defocus, fft_info = None, None
    if cross_check:
        fft_defocus, fft_info = fft_detect_array(
            gray,
            grid_x=FFT_GRID_X,
            grid_y=FFT_GRID_Y,
            hf_abs=FFT_HF_ABS,
            hf_rel=FFT_HF_REL,
            min_severe=FFT_MIN_SEVERE,
            min_rows=FFT_MIN_ROWS,
            min_cols=FFT_MIN_COLS,
        )

    printed = structure > min_structure
    # Если печатных тайлов почти нет (титул, пустой лист) — берём верхние по контрасту,
    # чтобы метрика оставалась осмысленной, а не считалась по шуму.
    if printed.sum() < max(6, grid_x * grid_y // 20):
        printed = structure > np.percentile(structure, 60)

    vals = sharp_map[printed]
    if vals.size == 0:
        return dict(
            sharpness=float("nan"),
            worst_zone=float("nan"),
            edge_density=edge_density,
            zone=zone,
            inner_median=inner_median,
            fft_defocus=fft_defocus,
            fft_info=fft_info,
            n_printed=0,
            sharp_map=sharp_map,
            structure_map=structure,
            printed_mask=printed,
        )

    n_worst = max(1, vals.size // 10)
    worst_zone = float(np.sort(vals)[:n_worst].mean())
    return dict(
        sharpness=float(np.median(vals)),
        worst_zone=worst_zone,
        edge_density=edge_density,
        zone=zone,
        inner_median=inner_median,
        fft_defocus=fft_defocus,
        fft_info=fft_info,
        n_printed=int(printed.sum()),
        sharp_map=sharp_map,
        structure_map=structure,
        printed_mask=printed,
    )
