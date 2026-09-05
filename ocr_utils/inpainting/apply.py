"""Общий цикл закраса: маска → группы → ROI → заливка → вклейка.

Это тот самый кусок, который раньше был зашит внутрь ``GpuModels.inpaint`` и умел
ровно одно: бить маску на связные области и звать LaMa. Здесь он разобран на две
сменные части — ЧЕМ заливать (``fill_roi``) и КАК делить маску на операции
(``groups``), — и потому служит и пальцам, и разметке из CVAT.

Умолчание ``groups=mask_components`` воспроизводит прежнее покомпонентное поведение
бит-в-бит: пальцевый пайплайн от разбора ничего не заметил.
"""

import numpy as np

from ocr_utils.inpainting.roi import DEFAULT_ROI_SCALE, blend_roi, mask_components, roi_bounds


def single_group(mask: np.ndarray) -> "list[np.ndarray]":
    """Стратегия «не делить»: вся маска — одна операция.

    Нужна тому, кто уже сам разложил маску по группам и хочет закрасить одну из них,
    ничего больше не деля (см. ``scan_cleanup.inpaint``).
    """
    return [mask] if np.any(mask) else []


def inpaint_by_groups(
    rgb: np.ndarray,
    mask: np.ndarray,
    fill_roi,
    *,
    groups=mask_components,
    padding: int = 64,
    feather: int = 9,
    roi_scale: float = DEFAULT_ROI_SCALE,
) -> "tuple[np.ndarray, list[tuple[int, int, int, int]]]":
    """Закрашивает маску группами: по одному прогону сети на группу.

    Аргументы:
        rgb: исходная картинка RGB uint8 (H, W, 3);
        mask: что закрасить, uint8 0/255 (H, W); может быть многокомпонентной;
        fill_roi: заливка ОДНОГО ROI — ``fill_roi(roi, roi_mask, bounds) -> roi``
            того же размера. Что за ней стоит (LaMa, Stable Diffusion, заглушка в
            тесте), этому модулю знать не нужно;
        groups: стратегия разбиения маски на операции — ``mask -> list[mask]``;
        padding: контекстное поле вокруг группы, пикс.;
        feather: ширина растушёвки шва, пикс. (уходит ВНУТРЬ маски, см.
            :func:`roi.blend_roi`);
        roi_scale: во сколько раз растянуть ROI от центра после ``padding``.

    Возвращает ``(картинка, список ROI)``. Список нужен debug-оверлею: по нему и
    видно, во что сложилась группировка.

    Пиксели ВНЕ маски совпадают с исходными бит-в-бит — заливка наружу не
    подмешивается вовсе. Пустая маска → входной массив возвращается как есть, сеть
    не запускается.

    Группы обрабатываются последовательно и КАЖДАЯ ВИДИТ РЕЗУЛЬТАТ ПРЕДЫДУЩИХ (ROI
    режется из ``result``, а не из ``rgb``): у двух соседних групп контекстные поля
    перекрываются, и вторая должна опираться на уже закрашенное, а не на объект,
    которого в итоге не останется.
    """
    zones = groups(mask)
    if not zones:
        return rgb, []

    result = rgb.copy()
    bounds_list: "list[tuple[int, int, int, int]]" = []
    for zone in zones:
        bounds = roi_bounds(zone, padding, roi_scale, rgb.shape[:2])
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        roi = result[y1:y2, x1:x2]
        mroi = zone[y1:y2, x1:x2]
        result[y1:y2, x1:x2] = blend_roi(roi, fill_roi(roi, mroi, bounds), mroi, feather)
        bounds_list.append(bounds)
    return result, bounds_list
