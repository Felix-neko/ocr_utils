"""Debug-оверлей: что нашёл детектор и что будет вырезано.

Рисуется поверх кадра ДО удаления пальцев и компенсации уровней — оверлей должен
показывать, что было найдено, а не результат обработки. Пунктиром обозначаются
«черновые»/отброшенные сущности (первичная зона пальца до дилатации, боксы
YOLO-World до SAM, паразитные блоки layout), сплошной линией — итоговые.
"""

from typing import Optional

import cv2
import numpy as np

from ocr_utils.scan_cropping.finger_removal.text_protection import DEFAULT_LAYOUT_PAD_PX, polygons_to_mask
from ocr_utils.scan_cropping.geometry import bbox_corners, ext_with_margins

# Цвета оверлеев в BGR (OpenCV)
COLOR_PAGE = (0, 255, 0)  # ярко-зелёный — криволинейная граница разворота
COLOR_ROT_BBOX = (255, 0, 0)  # ярко-синий — min-area повёрнутый bounding box
COLOR_CROP = (211, 0, 148)  # фиолетовый — финальная crop-зона с припусками
COLOR_FINGER = (0, 0, 255)  # красный — обнаруженная область пальца
COLOR_LAMA_ROI = (0, 255, 255)  # жёлтый — контекстная ROI-рамка, переданная в LaMa
COLOR_COPY_MASK = (0, 165, 255)  # оранжевый — область копирования E2 (маска после доп. эрозии)
COLOR_LAYOUT_BLOCK = (255, 255, 0)  # голубой — блок Surya layout (защищён от закраски)


def _draw_dashed_line(
    img: np.ndarray, pt1: tuple, pt2: tuple, color: tuple, thickness: int, dash_len: int = 20, gap_len: int = 14
) -> None:
    """Пунктирная линия pt1→pt2 (cv2 не умеет рисовать пунктир нативно)."""
    x1, y1 = pt1
    x2, y2 = pt2
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 1:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    draw = True
    while pos < length:
        seg_end = min(pos + (dash_len if draw else gap_len), length)
        if draw:
            p1 = (int(round(x1 + dx * pos)), int(round(y1 + dy * pos)))
            p2 = (int(round(x1 + dx * seg_end)), int(round(y1 + dy * seg_end)))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        pos = seg_end
        draw = not draw


def _draw_dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, thickness: int) -> None:
    """Пунктирный прямоугольник (используется для YOLO-World bbox пальца до SAM)."""
    for p1, p2 in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        _draw_dashed_line(img, p1, p2, color, thickness)


def _draw_dashed_contours(
    img: np.ndarray, mask: np.ndarray, color: tuple, thickness: int, dash_len: int = 24, gap_len: int = 18
) -> None:
    """Пунктирный контур маски — чтобы отличать первичную зону пальца от раздутой.

    Фаза пунктира копится ВДОЛЬ всего контура: точки контура идут почти впритык
    (``CHAIN_APPROX_NONE``), и если сбрасывать фазу на каждом сегменте, каждый
    короткий сегмент рисуется сплошным штрихом — контур выглядит сплошным.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    period = float(dash_len + gap_len)
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(np.float64)
        if len(pts) < 2:
            continue
        travelled = 0.0
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            seg = float(np.hypot(*(p2 - p1)))
            if seg < 1e-6:
                continue
            t = 0.0
            while t < seg:
                phase = (travelled + t) % period
                if phase < dash_len:  # штрих
                    run = min(dash_len - phase, seg - t)
                    a = p1 + (p2 - p1) * (t / seg)
                    b = p1 + (p2 - p1) * ((t + run) / seg)
                    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA)
                else:  # пропуск
                    run = min(period - phase, seg - t)
                t += run
            travelled += seg


def draw_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    geom: Optional[tuple],
    margins: "tuple[int, int, int, int]",
    finger_mask: Optional[np.ndarray] = None,
    lama_roi_bboxes: Optional[list] = None,
    finger_boxes: Optional[np.ndarray] = None,
    copy_mask: Optional[np.ndarray] = None,
    finger_mask_predilate: Optional[np.ndarray] = None,
    layout_polygons: Optional[list] = None,
    parasitic_layout_polygons: Optional[list] = None,
    layout_pad_px: "int | tuple[int, int]" = DEFAULT_LAYOUT_PAD_PX,
    crop_ext: Optional[tuple] = None,
) -> np.ndarray:
    """Кадр с оверлеями: зелёная граница разворота (E1), оранжевая граница области
    копирования (E2, после доп. эрозии), синий min-bbox, фиолетовая crop-зона,
    красная СПЛОШНАЯ граница зоны пальца ПОСЛЕ (асимметричной) дилатации — именно
    её закрашивает LaMa, красный ПУНКТИРНЫЙ тонкий контур первичной зоны пальца
    (после SAM, до дилатации), красный пунктирный bbox от YOLO-World (до SAM),
    жёлтая ROI-рамка контекста для LaMa, голубые тонкие контуры блоков Surya layout
    (``--protect-text-layout``) — они защищены от закраски. Контуры блоков рисуются
    УЖЕ с запасом ``layout_pad_px`` (как в маске защиты), т.е. показывают фактически
    защищённую зону, а не «впритык» очерченный Surya полигон. ``parasitic_layout_polygons``
    (артефакты на пустых страницах, исключённые из crop-зоны) рисуются тем же голубым
    цветом, но ПУНКТИРОМ.

    ``crop_ext`` — финальный ext вырезаемой зоны (с припусками и расширением под
    layout); если не задан, crop-зона рисуется по ``ext`` с ``margins`` (без учёта
    layout). Так фиолетовая рамка на оверлее совпадает с тем, что реально вырежется.

    Рисуется поверх ``bgr`` ДО удаления пальцев и компенсации уровней — оверлей
    должен показывать, что было найдено, а не результат обработки.
    """
    h, w = bgr.shape[:2]
    out = bgr.copy()
    thickness = max(2, int(round(max(h, w) / 500)))
    if int(np.count_nonzero(mask)) > 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_PAGE, thickness, lineType=cv2.LINE_AA)
    if copy_mask is not None and int(np.count_nonzero(copy_mask)) > 0:
        contours, _ = cv2.findContours(copy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_COPY_MASK, thickness, lineType=cv2.LINE_AA)
    if geom is not None:
        cx, cy, angle, ext = geom
        bbox = bbox_corners(cx, cy, angle, ext).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [bbox], True, COLOR_ROT_BBOX, thickness, cv2.LINE_AA)
        crop_zone = crop_ext if crop_ext is not None else ext_with_margins(ext, margins)
        crop = bbox_corners(cx, cy, angle, crop_zone).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [crop], True, COLOR_CROP, thickness, cv2.LINE_AA)
    # Блоки layout — тонкой линией: их много, толстая рамка забила бы кадр.
    # Рисуем контуры маски С padding'ом, чтобы оверлей совпадал с реально
    # защищённой от закраски зоной (Surya обводит блок впритык).
    if layout_polygons:
        layout_mask = polygons_to_mask(out.shape, layout_polygons, layout_pad_px)
        contours, _ = cv2.findContours(layout_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_LAYOUT_BLOCK, max(1, thickness // 2), lineType=cv2.LINE_AA)
    # Паразитные (артефактные) блоки layout — тем же цветом, но ПУНКТИРОМ: они
    # исключены из расчёта crop-зоны (см. classify_parasitic_layouts).
    if parasitic_layout_polygons:
        para_mask = polygons_to_mask(out.shape, parasitic_layout_polygons, layout_pad_px)
        _draw_dashed_contours(out, para_mask, COLOR_LAYOUT_BLOCK, max(1, thickness // 2))
    if lama_roi_bboxes is not None:
        for x1, y1, x2, y2 in lama_roi_bboxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_LAMA_ROI, thickness, cv2.LINE_AA)
    if finger_boxes is not None:
        for bx in finger_boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in bx)
            _draw_dashed_rect(out, x1, y1, x2, y2, COLOR_FINGER, thickness)
    # Первичная зона пальца (после SAM, ДО дилатации) — тонким пунктиром,
    # чтобы было видно, насколько её раздула асимметричная дилатация.
    if finger_mask_predilate is not None and int(np.count_nonzero(finger_mask_predilate)) > 0:
        _draw_dashed_contours(out, finger_mask_predilate, COLOR_FINGER, max(1, thickness // 2))
    # Итоговая (раздутая) зона — сплошной линией: именно она уходит в LaMa.
    if finger_mask is not None and int(np.count_nonzero(finger_mask)) > 0:
        contours, _ = cv2.findContours(finger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_FINGER, thickness, lineType=cv2.LINE_AA)
    return out
