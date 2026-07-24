"""Детекция разворота (YOLO-World + SAM), его правильный поворот и crop.

Пайплайн на каждый кадр:
  1. YOLO-World находит боксы страницы/разворота, SAM строит криволинейный силуэт,
     ``refine_page_mask`` оставляет крупнейшую область и заполняет дыры.
  2. ``min_area_rotated_bbox`` ищет «правильный поворот»: вокруг центра тяжести маски
     перебираются углы ±``ROT_RANGE_DEG`` с шагом ``ROT_STEP_DEG``, выбирается угол с
     минимальной площадью осевого bounding box.
  3. К bbox применяются припуски по каждой стороне отдельно ``--left-margin`` /
     ``--top-margin`` / ``--right-margin`` / ``--bottom-margin`` (пиксели; >0 —
     расширить наружу, <0 — сжать внутрь) → финальная crop-зона.
  4. Исходный кадр поворачивается на найденный угол вокруг центра тяжести, из него
     вырезается crop-зона (выпрямленный прямоугольник) и кладётся в ``--output-dir``
     под тем же именем файла.

Если задана ``--debug-dir`` — туда пишется кадр с оверлеями (всегда JPEG, ДО
удаления пальцев и компенсации уровней): зелёная граница разворота (E1),
оранжевая граница области копирования (E2, после доп. эрозии), синий min-area
bbox, фиолетовая crop-зона, красная граница обнаруженного пальца, жёлтая
ROI-рамка контекста, переданного в LaMa.

Дополнительные опции:
  - ``--output-format`` (png/tiff) — формат файлов в ``--output-dir``; по умолчанию
    как у входного файла.
  - ``--compensate-levels`` — растягивает уровни (по общей интенсивности, не по
    каналам отдельно) по перцентилям внутри маски страницы, эрозированной на
    ``--erosion-px`` (по умолчанию 20).
  - ``--upscale`` — увеличивает выходной холст перед поворотом/кропом (по
    умолчанию не задан — апскейл вообще не считается); сэмплирование всегда
    из исходного кадра.
  - ``--remove-fingers/--no-remove-fingers`` (включено по умолчанию) — перед
    детекцией разворота и кропом детектирует и закрашивает через LaMa палец,
    придерживающий страницу (``ocr_utils.finger_removal``), чтобы он не искажал
    силуэт/bbox страницы и не попадал в финальный кроп.
  - ``--finger-dilate-px`` — дилатация маски пальца перед закраской, пикс.
    (по умолчанию ``FINGER_DILATE_PX``).
  - ``--extra-erosion-px`` — доп. обрезка краёв силуэта книги перед копированием,
    пикс. (E2 = диляция на extra + эрозия на 2*extra от E1); срезает тёмные куски
    обложки в углах crop-зоны (по умолчанию ``EXTRA_EROSION_PX``; 0 — выкл.).

    uv run python -m ocr_utils.detect_and_crop \\
        --input-dir IN --output-dir OUT --debug-dir DBG \\
        --left-margin -150 --top-margin -150 --right-margin -150 --bottom-margin -150
"""

import logging
import timeit
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from ocr_utils.finger_removal.finger_inpaint import lama_inpaint, roi_bounds_list
from ocr_utils.finger_removal.masking import build_finger_mask, keep_border_components, drop_fingers_on_content
from ocr_utils.finger_removal.masking import _suppress_nested_boxes
from ocr_utils.finger_removal.finger_shadow import SHADOW_METHODS, correct_finger_shadow
from ocr_utils.finger_removal.asymmetric_dilation import DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO
from ocr_utils.timing import log_timing
from ocr_utils.finger_removal.text_protection import (
    DEFAULT_LAYOUT_PAD_PX,
    PROTECT_COPY_BACK,
    PROTECT_LIMIT_LAMA,
    PROTECT_MODES,
    copy_back_layout,
    layout_polygons,
    limit_paint_zone,
    polygons_to_mask,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Папка для весов нейромоделей (корень проекта, рядом с finger_models)
MODELS_DIR = Path(__file__).resolve().parents[1] / "finger_models"

# Цвета оверлеев в BGR (OpenCV)
COLOR_PAGE = (0, 255, 0)  # ярко-зелёный — криволинейная граница разворота
COLOR_ROT_BBOX = (255, 0, 0)  # ярко-синий — min-area повёрнутый bounding box
COLOR_CROP = (211, 0, 148)  # фиолетовый — финальная crop-зона с припусками
COLOR_FINGER = (0, 0, 255)  # красный — обнаруженная область пальца
COLOR_LAMA_ROI = (0, 255, 255)  # жёлтый — контекстная ROI-рамка, переданная в LaMa
COLOR_COPY_MASK = (0, 165, 255)  # оранжевый — область копирования E2 (маска после доп. эрозии)
COLOR_LAYOUT_BLOCK = (255, 255, 0)  # голубой — блок Surya layout (защищён от закраски)

# Поиск правильного поворота разворота: перебор углов ± предела с шагом (градусы)
ROT_RANGE_DEG = 35
ROT_STEP_DEG = 1

# Веса по умолчанию (лежат/качаются в finger_models/)
DEFAULT_YOLO_WORLD = "yolov8x-worldv2.pt"
DEFAULT_SAM = "sam_b.pt"

# Классы open-vocabulary детектора, описывающие страницу/разворот книги.
PAGE_CLASSES = ["page", "book page", "open book", "sheet of paper", "paper", "document"]

# Классы фона/подложки — конкурируют с PAGE_CLASSES за боксы, чтобы боксы,
# распознанные как ткань/подложка, не попадали в маску страницы (см. detect_page_mask).
# CLIP путает светлую однотонную бумагу (форзац без текста) с тканью по текстуре
# волокна, независимо от того, что написано в промпте про цвет/яркость — поэтому
# «тёмное/светлое» разделяем не промптом, а напрямую по пикселям (см. ниже).
FABRIC_CLASSES = ["fabric", "cloth", "fabric backdrop", "tablecloth"]

# Настоящая тканевая подложка в этой съёмке — тёмная (чёрный/тёмно-синий стол).
# Если бокс распознан как «ткань», но внутри него в среднем светлее этого порога
# (0-255) — это не подложка, а светлая страница/обложка; возвращаем его в кандидаты.
FABRIC_MAX_MEAN_BRIGHTNESS = 100

# Поддерживаемые форматы входных изображений (без учёта регистра расширения)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Параметры детекции
# Низкий порог нужен для пустых/малоинформативных страниц (форзац без текста
# конкурирует по уверенности с FABRIC_CLASSES и проигрывает даже при CONF=0.05,
# см. IMG_0154.jpg) — дальше отсеиваем контейнментом/размером/яркостью, а не conf.
CONF = 0.01
WORK_SIDE = 2048  # сторона уменьшенной копии для детекции (выше = точнее контур SAM)
MIN_PAGE_FRAC = 0.05  # бокс/маска меньше этой доли кадра — это не страница
MAX_PAGE_FRAC = 1.0  # верхний предел не ставим: страница может занимать весь кадр
# Отсев «мусорных» боксов почти на весь кадр: YOLO-World иногда выдаёт бокс класса
# страницы на 95+% кадра с низкой уверенностью — его SAM-силуэт захватывает фон и
# придерживающую книгу руку по краям, раздувая маску до всего кадра (см. IMG_0012.jpg
# 1975/10 и IMG_0014.jpg 1976/04). Чем крупнее бокс, тем выше должна быть уверенность:
# градуированные ярусы (доля кадра, мин. conf) — бокс площадью ≥ доли отбрасывается,
# если его conf ниже соответствующего порога. По выборке (50 кадров) легитимные крупные
# боксы имеют conf ≥ 0.22 и не превышают ~90% кадра, а мусорные near-full-frame боксы
# встречались при conf 0.013–0.066 — ярусы ложатся в зазор между ними.
LARGE_BOX_CONF_TIERS = ((0.91, 0.05), (0.94, 0.10))
# В refine_page_mask: связная компонента меньше этой доли площади самой крупной
# компоненты считается шумом и отбрасывается; крупнее — это вторая страница
# разворота (см. IMG_0058.jpg), а не шум, и должна остаться в маске.
SECOND_PAGE_MIN_AREA_FRAC = 0.2
# Порог для _suppress_nested_boxes (keep_new_area_frac): бокс, переросший
# локальный якорь сверх growth_factor, всё же оставляем, если он добавляет ≥ этой
# доли НЕ покрытой принятыми боксами площади. YOLO-World иногда не даёт отдельного
# бокса на страницу, целиком занятую фото/иллюстрацией, и покрывает её только
# «широким» боксом на весь разворот; он перерастает бокс соседней (одной) страницы,
# но вносит цельную вторую страницу как новую площадь и должен уцелеть (см.
# IMG_0004.jpg 1972/04, где левая страница-фото иначе теряется).
PAGE_KEEP_NEW_AREA_FRAC = 0.35

# Отбор вложенных per-page под-боксов, дополнительно скармливаемых SAM (см.
# _contained_subboxes / detect_page_mask): под-бокс должен лежать в одном из принятых
# боксов не менее чем на PAGE_SUBBOX_CONTAIN и быть не крупнее PAGE_SUBBOX_MAX_AREA_FRAC
# его площади (иначе это дубль всего разворота, а не под-бокс отдельной страницы).
PAGE_SUBBOX_CONTAIN = 0.85
PAGE_SUBBOX_MAX_AREA_FRAC = 0.9

# Компенсация уровней: перцентили по общей интенсивности внутри маски (минус эрозия)
N_EROSION_PX = 20
LEVELS_LOW_PCT = 1.0
LEVELS_HIGH_PCT = 98.0

# Заливка фона за пределами силуэта книги (перед rotated-crop): эрозия маски
# книги перед расчётом цвета заливки, пикс. — чтобы источник цвета не захватывал
# шумную/смазанную границу силуэта (там же соседствует фон).
BG_FILL_EROSION_PX = 100

# Способы заливки внешней зоны (значения --bg-fill-method). Все, кроме average,
# берут цвет из приграничной полосы страницы и локально продлевают его наружу —
# так воспроизводится и неравномерный свет, и цветная обложка (см.
# background_fill_extrapolation_report.md). Считаем, что на краю страницы текста
# нет, поэтому источник цвета чистый и подавление чернил не нужно.
BG_FILL_AVERAGE = "average"  # один усреднённый цвет по всей странице (старый способ)
BG_FILL_NEAREST = "nearest"  # цвет ближайшего пикселя границы E2 (Вороной, distance transform)
BG_FILL_METHODS = (BG_FILL_AVERAGE, BG_FILL_NEAREST)

# Доп. «обрезка» краёв силуэта книги перед копированием, пикс. Маска страницы на
# тёмном фоне захватывает не только светлые страницы, но и куски сравнительно
# тёмной обложки подшивки у краёв/углов. Просто взять min-area bbox и отступить
# внутрь мало: книга не прямая, и в углах B2 всё равно остаются тёмные фрагменты
# обложки. Поэтому область КОПИРОВАНИЯ (E2) получаем из маски (E1) морфологией
# «диляция на extra + эрозия на 2*extra» — это закрытие мелких вырезов + чистый
# сдвиг края внутрь на extra: периферийные слои обложки срезаются, а то, что в B2
# вне E2, заливается усреднённым светлым цветом страницы. 0 — выключить.
EXTRA_EROSION_PX = 80

# Удаление пальцев (finger_removal) перед детекцией книги/кропом
# Низкий порог нужен для recall (слабые боксы на смазанных/неярких пальцах, см.
# IMG_0028.jpg — лучший бокс conf=0.046, ниже стандартного 0.05); раздутая маска
# была из-за скин-добора и невложенных дублей боксов — то и другое уже устранено
# (skin-добор убран, _suppress_nested_boxes в neural_hand_mask), так что низкий
# conf теперь безопасен.
FINGER_CONF = 0.01
# Дилатация маски пальца (build_finger_mask default=12) — тонкая мягкая тень по
# краю силуэта (полутона на стыке кожа/бумага) иначе не докрашивается.
FINGER_DILATE_PX = 40
# Доля кадра для проверки контакта с рамкой в keep_border_components. Настоящий
# палец физически ОБРЕЗАН рамкой кадра (рука уходит за границу снимка), поэтому
# его маска доходит почти до самого края (~0 px). Узкая полоса надёжнее широкой:
# при 0.12 на 36-Мп сканах полоса ~430 px, и в неё попадают внутренние тёмные
# иллюстрации/фото у верхнего/бокового поля, ошибочно принятые YOLO-World за руку
# (см. IMG_0109.jpg: карта СССР в эмблеме «50 ЛЕТ СОЮЗА ССР» — 408 px от верха,
# 11.4 % высоты — пролезала впритык под 12 %). Настоящий палец здесь на 0 %,
# так что зазор огромный, 4 % чисто разделяет случаи.
FINGER_EDGE_FRAC = 0.04
FINGER_PADDING = 64  # контекст вокруг маски пальца для LaMa, пикс. (как в finger_inpaint.py)
# ROI для LaMa увеличивается в FINGER_ROI_SCALE раз от центра (после padding) —
# без этого LaMa не видит достаточно кромки/фона и заливает дыру доминирующим
# цветом (см. finger_inpaint.py, коммит "Сделали хороший закрас с помощью lama").
FINGER_ROI_SCALE = 1.5
# LaMa заливает область пальца заметно ТЕМНЕЕ окружающей бумаги (проверено на
# нескольких кадрах: разница ~25-35 отн. ед. яркости у самой маски). Поэтому
# перед закраской осветляем зону пальца — плавно, чтобы не было резкой границы:
# полный инкремент внутри самой маски (она уже включает дилатацию на
# FINGER_DILATE_PX), спад до нуля к границе маски + ещё 2×FINGER_DILATE_PX наружу
# (эта кайма — как раз тот контекст, по которому LaMa восстанавливает цвет дыры).
# Значение 20 подобрано по серии кадров из /mnt/system/raw/mts/cropped/1972 —
# заметно снижает остаточное потемнение, не давая цветового ухода в оранжевый
# (при 25-30 на тонированной («состаренной») бумаге появляется через чур тёплый оттенок).
FINGER_ZONE_LIGHT_INCREMENT = 20

_MODEL_CACHE: dict = {}


# ============================================================
# Удаление пальцев (перед детекцией книги/кропом)
# ============================================================


def brighten_finger_zone(
    rgb: np.ndarray, mask: np.ndarray, increment: "float | tuple[float, float]", falloff_px: int
) -> np.ndarray:
    """Осветляет зону пальца перед закраской (см. ``FINGER_ZONE_LIGHT_INCREMENT``).

    Внутри ``mask`` — полный ``increment``; далее вес плавно (линейно по
    расстоянию) спадает до 0 на удалении ``falloff_px`` от границы маски.
    Прибавляется поровну ко всем каналам — контраст-нейтрально (не искажает
    цветовой баланс сам по себе), итоговый цвет заливки всё равно определяет LaMa.

    ``increment`` — одно число (одинаково для всего кадра) либо пара
    ``(слева, справа)``: свет в кадре может падать не симметрично, и тогда
    правая и левая половины разворота требуют разной компенсации (см.
    ``--finger-zone-light-increment``). Компонента маски относится к той
    половине, где лежит центр её масс.
    """
    if int(np.count_nonzero(mask)) == 0:
        return rgb
    left_inc, right_inc = increment if isinstance(increment, tuple) else (increment, increment)
    if left_inc <= 0 and right_inc <= 0:
        return rgb
    h, w = mask.shape[:2]
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    out = rgb.astype(np.float32)
    for i in range(1, num):
        inside = labels == i
        _, xs = np.where(inside)
        inc = left_inc if xs.mean() < w / 2 else right_inc
        if inc <= 0:
            continue
        if falloff_px > 0:
            dist = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5)
            weight = np.clip(1.0 - dist / falloff_px, 0.0, 1.0)
            weight[inside] = 1.0
        else:
            weight = inside.astype(np.float32)
        out += weight[..., None] * inc
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_fingers(
    bgr: np.ndarray,
    device: str,
    conf: float = FINGER_CONF,
    want_boxes: bool = False,
    dilate_px: int = FINGER_DILATE_PX,
    light_increment: "float | tuple[float, float]" = FINGER_ZONE_LIGHT_INCREMENT,
    asymmetric_dilation_ratio: float = DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO,
    protect_text: bool = False,
    protect_mode: str = PROTECT_LIMIT_LAMA,
    layout_pad_px: "int | tuple[int, int]" = DEFAULT_LAYOUT_PAD_PX,
    log_name: str = "",
) -> tuple[np.ndarray, np.ndarray, Optional[list], Optional[np.ndarray], str, Optional[np.ndarray], Optional[list]]:
    """Детектирует и закрашивает пальцы (finger_removal.masking/finger_inpaint) в BGR-кадре.

    Возвращает (bgr, finger_mask, lama_roi_bboxes, yolo_boxes, info, finger_mask_predilate,
    layout_polys) — маска,
    список ROI-боксов LaMa (по одному на компоненту маски) и боксы YOLO-World
    нужны только для debug-оверлея, на итоговый bgr не влияют. ``yolo_boxes``
    берётся из ``build_finger_mask(..., return_boxes=True)`` — та же самая
    детекция, что уже нужна для маски, без повторного прогона YOLO-World
    (раньше эти боксы для debug-оверлея считались отдельным, дублирующим
    вызовом ``finger_yolo_boxes``). Возвращается только при ``want_boxes=True``
    (т.е. когда включён ``--debug-dir``), а не всегда, просто чтобы не тащить
    в debug-неактуальные боксы через весь пайплайн.

    При ``protect_text=True`` кадр (ДО закраски) прогоняется через Surya layout, и
    найденные блоки защищаются от закраски способом ``protect_mode``: урезанием
    зоны закраски (``limit-lama-zone``) либо копированием блоков обратно с
    оригинала уже ПОСЛЕ закраски (``copy-back-layout-zones``) — см.
    ``finger_removal.text_protection``. Возвращаемые ``layout_polys`` нужны только
    для debug-оверлея.

    Палец может исказить детекцию разворота и итоговый кроп, поэтому закраска
    выполняется до ``page_mask``/``crop_rotated``. ``build_finger_mask("auto", ...)``
    не проверяет контакт нейромаски с рамкой кадра — из-за этого крупные ФОТО
    людей/рук на самой странице (в глубине кадра, не с края) иногда ложно
    принимаются за палец. Настоящий палец всегда входит С КРАЯ кадра, поэтому
    дополнительно отсекаем компоненты, не касающиеся рамки, через
    ``keep_border_components``. Перед самой закраской зона пальца осветляется
    (``brighten_finger_zone``) — LaMa иначе заливает дыру заметно темнее
    окружающей бумаги. Если палец не найден — кадр возвращается без
    изменений.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with log_timing("build_finger_mask", log_name):
        mask, info, raw_boxes, mask_predilate = build_finger_mask(
            rgb,
            method="auto",
            device=device,
            conf=conf,
            dilate_px=dilate_px,
            return_boxes=True,
            return_predilate=True,
            asymmetric_dilation_ratio=asymmetric_dilation_ratio,
            log_name=log_name,
        )
    yolo_boxes = raw_boxes if want_boxes else None
    predilate = mask_predilate if want_boxes else None
    if int(np.count_nonzero(mask)) > 0:
        mask = keep_border_components(mask, edge_frac=FINGER_EDGE_FRAC)
        if int(np.count_nonzero(mask)) == 0:
            info = "auto(отсеяно: не у края)"
    if int(np.count_nonzero(mask)) == 0:
        return bgr, mask, None, yolo_boxes, info, predilate, None

    layout_polys: Optional[list] = None
    if protect_text:
        with log_timing("layout_polygons", log_name):
            layout_polys = layout_polygons(rgb)
        info = f"{info}, layout: блоков={len(layout_polys)}"
        if layout_polys:
            layout_mask = polygons_to_mask(mask.shape, layout_polys, layout_pad_px)
            # Ложные «пальцы» на печатном контенте (лица на портретах) убираем из
            # маски ещё ДО закраски — иначе LaMa затрёт сам контент. Работает в
            # обоих режимах защиты (и copy-back, и limit-lama).
            with log_timing("drop_fingers_on_content", log_name):
                mask, mask_predilate, dropped = drop_fingers_on_content(
                    mask, mask_predilate, layout_mask, dilate_px, asymmetric_dilation_ratio, FINGER_EDGE_FRAC
                )
            if dropped:
                info = f"{info}, ложных пальцев на контенте убрано={dropped}"
                if want_boxes:
                    predilate = mask_predilate
                if int(np.count_nonzero(mask)) == 0:
                    return bgr, mask, None, yolo_boxes, info, predilate, layout_polys
            if protect_mode == PROTECT_LIMIT_LAMA:
                before = int(np.count_nonzero(mask))
                mask = limit_paint_zone(mask, mask_predilate, layout_mask)
                after = int(np.count_nonzero(mask))
                info = f"{info}, зона закраски {before}→{after} px"
                if after == 0:
                    return bgr, mask, None, yolo_boxes, info, predilate, layout_polys

    roi_bboxes = roi_bounds_list(mask, padding=FINGER_PADDING, roi_scale=FINGER_ROI_SCALE)
    with log_timing("brighten_finger_zone", log_name):
        rgb_bright = brighten_finger_zone(rgb, mask, light_increment, 2 * dilate_px)
    with log_timing("lama_inpaint", log_name):
        rgb_clean = lama_inpaint(rgb_bright, mask, device=device, padding=FINGER_PADDING, roi_scale=FINGER_ROI_SCALE)

    # Копирование блоков обратно — строго ПОСЛЕ закраски и с ИСХОДНОГО (неосветлённого)
    # кадра: rgb_bright уже подкрашен под LaMa и вернул бы контент со сдвигом яркости.
    if protect_text and protect_mode == PROTECT_COPY_BACK and layout_polys:
        with log_timing("copy_back_layout", log_name):
            rgb_clean, restored = copy_back_layout(rgb, rgb_clean, layout_polys, mask, layout_pad_px)
        info = f"{info}, восстановлено блоков={restored}"

    return cv2.cvtColor(rgb_clean, cv2.COLOR_RGB2BGR), mask, roi_bboxes, yolo_boxes, info, predilate, layout_polys


# ============================================================
# Модели и маска разворота
# ============================================================


def resolve_model_path(name: str) -> str:
    """Путь к весам в finger_models/ (качает ассет ultralytics по имени, если нужно)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODELS_DIR / name)


def load_yolo_world(name: str):
    """Ленивая загрузка YOLO-World с классами страницы + классами ткани/фона.

    Классы фона нужны, чтобы они конкурировали за боксы с классами страницы —
    тогда фон/подложка, ошибочно захваченные в один бокс со страницей, скорее
    получат класс из ``FABRIC_CLASSES`` и будут отфильтрованы в ``detect_page_mask``.
    """
    key = f"world:{name}"
    if key not in _MODEL_CACHE:
        from ultralytics import YOLOWorld

        model = YOLOWorld(resolve_model_path(name))
        model.set_classes(PAGE_CLASSES + FABRIC_CLASSES)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def load_sam(name: str):
    """Ленивая загрузка SAM."""
    key = f"sam:{name}"
    if key not in _MODEL_CACHE:
        from ultralytics import SAM

        _MODEL_CACHE[key] = SAM(resolve_model_path(name))
    return _MODEL_CACHE[key]


def refine_page_mask(mask: np.ndarray) -> np.ndarray:
    """Смыкание разрывов + крупные связные области + заливка дыр.

    Левая и правая страницы разворота часто детектируются ДВУМЯ отдельными
    боксами (левая половина / правая половина или обложка), и у SAM-силуэтов
    между ними остаётся зазор в пару пикселей у корешка — тогда они оказываются
    РАЗНЫМИ связными компонентами. Поэтому смыкание (``MORPH_CLOSE``) нужно
    делать ДО выбора «крупных» компонентов, а не после — иначе одна из половин
    разворота (например, обложка) отбрасывается целиком как «шум».

    Если детектор вместо двух отдельных боксов на страницы выдал ОДИН бокс на
    весь разворот (см. ``_suppress_nested_boxes``), у SAM-силуэта в этом боксе
    зазор у корешка получается намного шире 15px — смыкание его не устраняет, и
    страницы остаются раздельными компонентами. Раньше здесь оставляли только
    САМУЮ БОЛЬШУЮ компоненту — тогда вторая страница (сопоставимая по площади с
    первой) отбрасывалась целиком (см. IMG_0058.jpg: осталась только левая
    страница). Поэтому теперь оставляем ВСЕ компоненты не меньше
    ``SECOND_PAGE_MIN_AREA_FRAC`` от площади самой крупной — мелкий мусор
    (обрывки текста, шум SAM) настолько мельче страницы, что не проходит порог.
    """
    if int(np.count_nonzero(mask)) == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_labels = 1 + np.where(areas >= SECOND_PAGE_MIN_AREA_FRAC * areas.max())[0]
        mask = np.isin(labels, keep_labels).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _contained_subboxes(boxes: np.ndarray, keep: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Вложенные per-page под-боксы, которые дополнительно скармливаем SAM.

    YOLO-World нередко выдаёт и бокс на весь разворот, и отдельные боксы на левую/
    правую страницу; ``_suppress_nested_boxes`` оставляет только самый уверенный
    (обычно широкий бокс разворота), а SAM по ОДНОМУ широкому боксу строит рыхлый
    силуэт — не дотягивается до верха страниц и проваливается у придержанного
    пальцем края. Поэтому вложенные per-page боксы (заведомо ВНУТРИ уже принятой
    области книги, поэтому фон внести не могут) тоже прогоняем через SAM, а их
    силуэты объединяются с основным (``bitwise_or`` в ``detect_page_mask``) —
    страница восстанавливается целиком.

    Берём под-боксы, покрытые одним из принятых (``kept``) не менее чем на
    ``PAGE_SUBBOX_CONTAIN`` и не крупнее ``PAGE_SUBBOX_MAX_AREA_FRAC`` его площади
    (иначе это дубль всего разворота, а не под-бокс отдельной страницы). ``keep`` —
    индексы принятых в ``boxes`` (их самих не дублируем).
    """
    keep_set = set(int(i) for i in keep)
    kept_areas = (kept[:, 2] - kept[:, 0]) * (kept[:, 3] - kept[:, 1])
    extra: list[np.ndarray] = []
    for i in range(len(boxes)):
        if i in keep_set:
            continue
        bi = boxes[i]
        area_i = max(1.0, float((bi[2] - bi[0]) * (bi[3] - bi[1])))
        for kb, akb in zip(kept, kept_areas):
            ix1, iy1 = max(bi[0], kb[0]), max(bi[1], kb[1])
            ix2, iy2 = min(bi[2], kb[2]), min(bi[3], kb[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / area_i >= PAGE_SUBBOX_CONTAIN and area_i <= PAGE_SUBBOX_MAX_AREA_FRAC * akb:
                extra.append(bi)
                break
    return np.array(extra, dtype=boxes.dtype).reshape(-1, 4)


def detect_page_mask(bgr: np.ndarray, device: str) -> np.ndarray:
    """Бинарная маска (uint8 0/255) области страниц: YOLO-World боксы → SAM силуэт.

    Боксы класса из ``FABRIC_CLASSES`` (ткань/подложка) отбрасываются сразу.
    Оставшиеся боксы дополнительно прогоняются через ``_suppress_nested_boxes``
    (та же логика, что и для пальцев в ``masking.py``) — низкоуверенный бокс,
    почти целиком содержащий в себе более уверенный (например, «вся подложка +
    книга» вместо «только книга»), отбрасывается в пользу более точного. Помимо
    принятых боксов SAM получает вложенные per-page под-боксы (``_contained_subboxes``):
    по одному широкому боксу разворота силуэт SAM рыхлый и недобирает края страниц.
    """
    h, w = bgr.shape[:2]
    img_area = h * w

    yolo = load_yolo_world(DEFAULT_YOLO_WORLD)
    det = yolo.predict(bgr, conf=CONF, device=device, verbose=False)
    if not det or det[0].boxes is None or len(det[0].boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    boxes = det[0].boxes.xyxy.cpu().numpy()
    confs = det[0].boxes.conf.cpu().numpy()
    cls = det[0].boxes.cls.cpu().numpy().astype(int)

    # Боксы класса из FABRIC_CLASSES отбрасываем, ЕСЛИ они действительно тёмные
    # (настоящая подложка в кадре — чёрный/тёмно-синий стол). Светлый бокс,
    # который CLIP всё равно назвал «тканью» из-за текстуры волокна бумаги —
    # возвращаем в кандидаты, иначе однотонные страницы без текста (форзац)
    # остаются вообще без детекции.
    is_fabric = cls >= len(PAGE_CLASSES)
    if np.any(is_fabric):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for i in np.where(is_fabric)[0]:
            x1, y1, x2, y2 = boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1 and gray[y1:y2, x1:x2].mean() > FABRIC_MAX_MEAN_BRIGHTNESS:
                is_fabric[i] = False
    boxes, confs = boxes[~is_fabric], confs[~is_fabric]
    if len(boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    area = bw * bh
    size_ok = (area >= MIN_PAGE_FRAC * img_area) & (area <= MAX_PAGE_FRAC * img_area)
    # Плюс: near-full-frame боксы с недостаточной уверенностью — это шум YOLO-World, чей
    # SAM-силуэт сгребает фон и руку по краям (см. LARGE_BOX_CONF_TIERS / IMG_0012, IMG_0014).
    junk_fullframe = np.zeros(len(boxes), dtype=bool)
    for frac_thr, conf_thr in LARGE_BOX_CONF_TIERS:
        junk_fullframe |= (area >= frac_thr * img_area) & (confs < conf_thr)
    keep_size = size_ok & ~junk_fullframe
    boxes, confs = boxes[keep_size], confs[keep_size]
    if len(boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    keep = _suppress_nested_boxes(boxes, confs, keep_new_area_frac=PAGE_KEEP_NEW_AREA_FRAC)
    if len(keep) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    kept = boxes[keep]
    # Помимо принятых боксов SAM получает вложенные per-page под-боксы: по одному
    # широкому боксу разворота силуэт SAM рыхлый и недобирает края (см. _contained_subboxes).
    sam_boxes = np.vstack([kept, _contained_subboxes(boxes, keep, kept)])

    sam = load_sam(DEFAULT_SAM)
    seg = sam.predict(bgr, bboxes=sam_boxes, device=device, verbose=False)
    mask = np.zeros((h, w), dtype=np.uint8)
    if seg and seg[0].masks is not None:
        for m in seg[0].masks.data.cpu().numpy():
            m_bin = (m > 0.5).astype(np.uint8)
            if m_bin.shape != (h, w):
                m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            if MIN_PAGE_FRAC * img_area <= m_bin.sum() <= MAX_PAGE_FRAC * img_area:
                mask = cv2.bitwise_or(mask, m_bin * 255)
    return mask


def page_mask(bgr: np.ndarray, device: str, pad_tb_px: int = 0) -> np.ndarray:
    """Полная маска разворота в разрешении кадра (детекция на уменьшенной копии).

    Результат уже включает ``bridge_component_gaps`` — то есть промежуток между
    отдельными фрагментами (например, корешок между левой и правой страницей)
    заполнен, а не только «крупнейшие компоненты + залитые дыры». Это
    КАНОНИЧЕСКАЯ маска разворота — используется одинаково во всех потребителях
    (debug-оверлей, ``min_area_rotated_bbox``, ``compensate_levels``,
    ``fill_outside_mask``), а не только для одного из них.

    ``pad_tb_px`` > 0 — перед детекцией добавить чёрную рамку сверху и снизу на
    указанное число пикселей, а маску вернуть уже без неё (в координатах исходного
    кадра). Приём для снимков, где книга занимает кадр целиком по вертикали и
    детектится «вся область как разворот»: чёрная рамка даёт SAM явную тёмную
    границу и отодвигает YOLO-боксы от краёв, а рост площади кадра выводит
    near-full-frame боксы из-под порогов LARGE_BOX_CONF_TIERS.
    """
    h, w = bgr.shape[:2]
    if pad_tb_px > 0:
        padded = cv2.copyMakeBorder(bgr, pad_tb_px, pad_tb_px, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = page_mask(padded, device)
        return mask[pad_tb_px : pad_tb_px + h, :]
    scale = WORK_SIDE / max(h, w) if max(h, w) > WORK_SIDE else 1.0
    work = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
    mask = detect_page_mask(work, device)
    if scale != 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = refine_page_mask(mask)
    return bridge_component_gaps(mask)


# ============================================================
# Компенсация уровней
# ============================================================


def compensate_levels(
    bgr: np.ndarray,
    mask: np.ndarray,
    erosion_px: int,
    low_pct: float = LEVELS_LOW_PCT,
    high_pct: float = LEVELS_HIGH_PCT,
    work_side: int = WORK_SIDE,
) -> np.ndarray:
    """Растягивает уровни по общей интенсивности (одинаково для всех каналов).

    Перцентили считаются по пикселям внутри маски страницы, эрозированной на
    ``erosion_px`` (чтобы не захватывать край страницы/фон). Диапазон общий для
    B/G/R — это не независимая цветокоррекция по каналам, а контраст-стретч,
    сохраняющий цветовой баланс.

    Эрозия и ``np.percentile`` считаются на копии, уменьшенной до ``work_side``
    (как и в ``page_mask``) — это лишь ОЦЕНКА перцентилей, полное разрешение ей
    не нужно, а на кадрах 30-48 Мп percentile по маске занимал секунды (см.
    профилирование ``detect_and_crop`` на медленных прогонах). Сам контраст-стретч
    (``rescale_intensity``) применяется к исходному кадру полного разрешения —
    только на нём и формируется итоговый результат.
    """
    h, w = mask.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    if scale < 1.0:
        small_mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        small_bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_erosion_px = max(1, int(round(erosion_px * scale)))
    else:
        small_mask, small_bgr, small_erosion_px = mask, bgr, erosion_px

    eroded = small_mask
    if small_erosion_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_erosion_px * 2 + 1, small_erosion_px * 2 + 1))
        eroded = cv2.erode(small_mask, k)
    sel = eroded > 0
    if not np.any(sel):
        return bgr

    small_bgr_f = small_bgr.astype(np.float32) / 255.0
    lo, hi = np.percentile(small_bgr_f[sel], (low_pct, high_pct))
    if hi <= lo:
        return bgr

    bgr_f = bgr.astype(np.float32) / 255.0
    out = rescale_intensity(bgr_f, in_range=(lo, hi), out_range=(0.0, 1.0))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# Геометрия: правильный поворот, повёрнутый bbox, crop
# ============================================================


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    """Матрица поворота 2×2 на ``angle_deg`` градусов."""
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def min_area_rotated_bbox(mask: np.ndarray) -> Optional[tuple]:
    """Возвращает (cx, cy, angle, (minx, miny, maxx, maxy)) или None.

    Центр тяжести — среднее X и Y по всем пикселям маски. Перебираем углы поворота
    вокруг центра и берём тот, у которого осевой bbox повёрнутых точек минимален по
    площади. ``ext`` — в повёрнутой системе координат (относительно центра).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)  # (N, 2) в (x, y)
    rel = pts - np.array([cx, cy])

    best = None
    for ang in range(-ROT_RANGE_DEG, ROT_RANGE_DEG + 1, ROT_STEP_DEG):
        rot = rel @ _rotation_matrix(ang).T
        mn = rot.min(axis=0)
        mx = rot.max(axis=0)
        area = (mx[0] - mn[0]) * (mx[1] - mn[1])
        if best is None or area < best[0]:
            best = (area, ang, (mn[0], mn[1], mx[0], mx[1]))

    _, angle, ext = best
    return cx, cy, angle, ext


def _ext_with_margins(ext: tuple, margins: "tuple[int, int, int, int]") -> tuple:
    """Применяет припуски к ext (minx, miny, maxx, maxy): >0 расширяет наружу, <0 сжимает внутрь.

    ``margins`` = (left, top, right, bottom) — своя величина на каждую сторону
    crop-зоны (левая двигает minx, верхняя — miny, правая — maxx, нижняя — maxy).
    """
    minx, miny, maxx, maxy = ext
    left, top, right, bottom = margins
    return (minx - left, miny - top, maxx + right, maxy + bottom)


def _bbox_corners(cx: float, cy: float, angle: float, ext: tuple) -> np.ndarray:
    """4 угла повёрнутого bbox в координатах исходного кадра (порядок TL,TR,BR,BL)."""
    minx, miny, maxx, maxy = ext
    corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]], dtype=np.float64)
    # Обратно в исходный кадр: rel = rot @ R(angle), затем + центр
    return (corners @ _rotation_matrix(angle) + np.array([cx, cy])).astype(np.float32)


def _ext_to_mask(shape: "tuple[int, int]", cx: float, cy: float, angle: float, ext: tuple) -> np.ndarray:
    """Бинарная маска (uint8 0/255) залитого повёрнутого bbox ``ext`` в координатах кадра."""
    m = np.zeros(shape[:2], dtype=np.uint8)
    corners = _bbox_corners(cx, cy, angle, ext).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [corners], 255)
    return m


def _layout_ext_bounds(
    cx: float, cy: float, angle: float, layout_mask: Optional[np.ndarray]
) -> Optional["tuple[float, float, float, float]"]:
    """Габариты блоков layout в осях crop-зоны (та же повёрнутая система, что и ``ext``).

    ``layout_mask`` — бинарная маска блоков УЖЕ с padding'ом (см. ``polygons_to_mask``).
    Контурные точки маски переводятся в повёрнутую вокруг ``(cx, cy)`` систему
    координат (как в ``min_area_rotated_bbox``) и по ним берётся осевой bbox.
    Возвращает (minx, miny, maxx, maxy) относительно центра либо None, если маска пуста.
    """
    if layout_mask is None:
        return None
    contours, _ = cv2.findContours(layout_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)
    rot = (pts - np.array([cx, cy])) @ _rotation_matrix(angle).T
    mn = rot.min(axis=0)
    mx = rot.max(axis=0)
    return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))


def crop_ext_with_layout(
    ext: tuple, margins: "tuple[int, int, int, int]", layout_bounds: Optional["tuple[float, float, float, float]"]
) -> tuple:
    """Финальный ext crop-зоны: ext с припусками, дополнительно расширенный под layout.

    Сначала применяются ``margins`` (могут быть и отрицательными — отступ внутрь),
    затем зона расширяется НАРУЖУ ровно настолько, чтобы целиком вместить габариты
    блоков layout (``layout_bounds`` в тех же осях). Если блоки и так внутри —
    ничего не меняется. Так отрицательные припуски не срезают часть обложки/текста,
    которую Surya распознала как контент (см. IMG_0003 с завышенными припусками).
    """
    minx, miny, maxx, maxy = _ext_with_margins(ext, margins)
    if layout_bounds is not None:
        lminx, lminy, lmaxx, lmaxy = layout_bounds
        minx, miny = min(minx, lminx), min(miny, lminy)
        maxx, maxy = max(maxx, lmaxx), max(maxy, lmaxy)
    return (minx, miny, maxx, maxy)


def bridge_component_gaps(mask: np.ndarray, work_side: int = WORK_SIDE) -> np.ndarray:
    """По строкам заполняет промежуток МЕЖДУ первым и последним отрезком маски —
    часть канонической маски разворота (см. ``page_mask``), используется во всех
    потребителях (debug-оверлей, поиск поворота, компенсация уровней, заливка фона).

    SAM иногда рвёт силуэт разворота вдоль корешка (широкий, неравномерный по
    высоте зазор между левой и правой страницей — от десятков до сотен пикселей,
    ``MORPH_CLOSE`` в ``refine_page_mask`` не бриджит его целиком) либо
    фрагментирует силуэт по малоинформативным/пустым участкам страницы (см.
    IMG_0033/0034/0030.jpg) — тогда эта область выпадает из региона интереса:
    не только закрашивается фоном при кропе, но и не учитывается при поиске угла
    поворота, что мешает последующей разбивке разворота на страницы.

    ВАЖНО: строка с ОДНИМ непрерывным отрезком маски не трогается — там разрыв
    может быть только на ВНЕШНЕЙ границе страницы (рваный край, срезанный угол),
    и её закраска фоном в ``fill_outside_mask`` должна остаться как была (полная
    выпуклая оболочка вместо этого «дошивала» бы и такие внешние прорехи тоже —
    затащила бы в регион интереса реальный фон/край стола). Заполняется только
    промежуток МЕЖДУ разными фрагментами в одной строке (например, между
    страницами) — сигнал ≥2 отрезков в строке отличает разрыв «между двумя
    объектами» от вогнутости на краю одного объекта.
    """
    if int(np.count_nonzero(mask)) == 0:
        return mask
    h, w = mask.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    small = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else mask
    sh, sw = small.shape[:2]
    out = small.copy()
    m = small > 0
    for y in range(sh):
        row = m[y]
        if not row.any():
            continue
        diff = np.diff(row.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if row[0]:
            starts = np.concatenate(([0], starts))
        if row[-1]:
            ends = np.concatenate((ends, [sw]))
        if len(starts) >= 2:
            out[y, ends[0] : starts[-1]] = 255
    if scale < 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def trim_cover_fragments(
    mask: np.ndarray, extra_erosion_px: int = EXTRA_EROSION_PX, work_side: int = WORK_SIDE
) -> np.ndarray:
    """E2 из E1: срезает периферийные фрагменты обложки, оставшиеся в маске страницы.

    К маске страницы (``mask`` = E1) применяется диляция на ``extra_erosion_px`` и
    затем эрозия на ``2 * extra_erosion_px``. Это закрытие мелких вырезов/зазубрин
    (диляция+эрозия на ту же величину) плюс чистый сдвиг края внутрь на
    ``extra_erosion_px`` (остаток эрозии): криволинейный край книги отступает
    внутрь, и тонкие слои тёмной обложки у краёв/углов (которые детектор включил в
    маску) отсекаются. Возвращает уменьшенную маску E2 (uint8 0/255) в разрешении
    исходной ``mask``.

    Морфология считается на копии, уменьшенной до ``work_side``: ядро радиусом
    ``2*extra_erosion_px`` (диаметр ~321px при 80) на кадре 30-48 Мп заметно
    тормозит (см. fill_outside_mask/compensate_levels), а для «обрезки» краёв
    точность полного разрешения не нужна — граница потом всё равно у бумажных
    полей, не у текста.
    """
    if extra_erosion_px <= 0:
        return mask
    h, w = mask.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    small = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else mask
    d = max(1, int(round(extra_erosion_px * scale)))
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    k_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (2 * d) + 1, 2 * (2 * d) + 1))
    out = cv2.dilate(small, k_dil, iterations=1)
    out = cv2.erode(out, k_ero, iterations=1)
    if scale < 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def _nearest_edge_fill(small_bgr: np.ndarray, e2_mask: np.ndarray) -> np.ndarray:
    """Заливка «по Вороному»: каждому пикселю — цвет ближайшего пикселя границы E2.

    Ближайший пиксель E2 для точки снаружи всегда лежит на границе E2, поэтому это и
    есть «цвет ближайшей точки границы зоны копирования». ``distance_transform_edt``
    с ``return_indices`` даёт индексы ближайшего известного (нулевого) пикселя — O(N),
    без перебора границы. ``e2_mask`` — маска E2 (0 вне). Возвращает BGR uint8.
    Минус метода — ступеньки на медиальной оси (где ближайшая точка границы
    переключается); лечится размытием заливки (см. ``_distance_weighted_blur``).
    """
    from scipy.ndimage import distance_transform_edt

    iy, ix = distance_transform_edt(e2_mask == 0, return_indices=True, return_distances=False)
    return small_bgr[iy, ix]


def _distance_weighted_blur(img: np.ndarray, e2_mask: np.ndarray, max_sigma: float) -> np.ndarray:
    """Размывает ``img`` тем сильнее, чем дальше пиксель от зоны копирования ``e2_mask``.

    У самой границы E2 размытия нет (вес 0) — так шов остаётся непрерывным, а ядро
    приграничного пикселя почти не залезает в E2 (нет ореола от контента у края).
    Вдали вес растёт до 1 (полное размытие ``max_sigma``) за ``~4·max_sigma`` пикселей
    от границы. Заливка — гладкий цвет без структуры, поэтому линейного бленда
    «резкая ⊕ сильно размытая» достаточно (двоения не даёт), собран на встроенных
    ``cv2.distanceTransform`` + ``cv2.blendLinear``. ``img`` BGR uint8, ``e2_mask`` —
    маска E2 (0 вне). Пиксели E2 вызывающий НЕ переписывает, поэтому E2 нетронута.
    """
    if max_sigma < 0.5:
        return img
    # Расстояние до ближайшего пикселя E2 (0 внутри E2, растёт наружу) → вес размытия.
    d = cv2.distanceTransform((e2_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(d / (4.0 * max_sigma), 0.0, 1.0).astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), float(max_sigma))
    # blendLinear: (w1·s1 + w2·s2)/(w1+w2); w1+w2=1 → попиксельный лерп по alpha.
    return cv2.blendLinear(img, blurred, 1.0 - alpha, alpha)


def fill_outside_mask(
    bgr: np.ndarray,
    mask: np.ndarray,
    erosion_px: int = BG_FILL_EROSION_PX,
    work_side: int = WORK_SIDE,
    method: str = BG_FILL_AVERAGE,
    blur_px: float = 0.0,
) -> np.ndarray:
    """Закрашивает всё вне ``mask`` цветом бумаги/обложки внутри неё.

    Криволинейная маска страницы не идеально совпадает с осевым min-area bbox
    (неровные/загнутые края) — в углы повёрнутого кропа может попасть кусок
    чёрного фона. Заранее закрасив фон, получаем ровный угол вместо чёрного пятна,
    даже если crop-зона чуть шире силуэта.

    ``method`` — способ заливки (см. ``BG_FILL_METHODS``):

    - ``average`` — один усреднённый цвет по всей странице (старый способ): дёшево,
      но не учитывает ни неравномерный свет, ни цветную обложку. Цвет усредняется по
      маске, эрозированной на ``erosion_px`` — чтобы шумный край не сдвигал среднее.
    - ``nearest`` — цвет ближайшей точки границы E2 (см. ``_nearest_edge_fill``): без
      эрозии (цвет нужен у самой границы), учитывает неравномерный свет и цветную
      обложку. Опционально сглаживается размытием (``blur_px``).

    ``blur_px`` (>0, только для локальных методов) — переменное размытие заливки: у
    границы E2 нуль, вдали до σ=``blur_px`` (см. ``_distance_weighted_blur``). Зона
    копирования при этом остаётся нетронутой (переписываем только пиксели вне E2).

    Всё считается на копии, уменьшенной до ``work_side`` (эрозия и экстраполяция на
    кадрах 30-48 Мп заметно тормозят), к полному разрешению применяется только
    подстановка заполненного цвета во внешнюю зону.
    """
    sel = mask > 0
    if not np.any(sel):
        return bgr

    h, w = mask.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    if scale < 1.0:
        small_mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        small_bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_mask, small_bgr = mask, bgr

    if method == BG_FILL_AVERAGE:
        # Источник среднего цвета — маска, эрозированная на erosion_px (без шумного
        # края/подтёкшего фона). Локальным методам эрозия не нужна: они берут цвет у
        # самой границы E2 (уже обрезанной trim_cover_fragments).
        small_erosion_px = max(1, int(round(erosion_px * scale))) if scale < 1.0 else erosion_px
        sample_sel = small_mask
        if small_erosion_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_erosion_px * 2 + 1, small_erosion_px * 2 + 1))
            eroded = cv2.erode(small_mask, k)
            if np.any(eroded > 0):
                sample_sel = eroded
        avg_color = small_bgr[sample_sel > 0].mean(axis=0)
        out = bgr.copy()
        out[~sel] = avg_color.astype(np.uint8)
        return out

    filled_small = _nearest_edge_fill(small_bgr, small_mask)
    if blur_px > 0:
        # Размытие тем сильнее, чем дальше от границы E2 (на downscale).
        filled_small = _distance_weighted_blur(filled_small, small_mask, blur_px * scale)
    # Карта заливки гладкая (без деталей) — обычного билинейного апскейла достаточно.
    filled = cv2.resize(filled_small, (w, h), interpolation=cv2.INTER_LINEAR) if scale < 1.0 else filled_small
    out = bgr.copy()
    out[~sel] = filled[~sel]
    return out


def crop_rotated(
    bgr: np.ndarray, cx: float, cy: float, angle: float, crop_ext: tuple, upscale: Optional[float] = None
) -> np.ndarray:
    """Поворот вокруг центра тяжести + вырез crop-зоны → выпрямленный прямоугольник.

    ``crop_ext`` — финальный ext crop-зоны (уже с припусками и расширением под
    layout, см. ``crop_ext_with_layout``). Берём 4 угла crop-зоны в исходном кадре
    и перспективным преобразованием отображаем их в осевой прямоугольник нужного
    размера (это и есть поворот кадра на найденный угол с одновременным вырезом
    области). ``upscale`` увеличивает только выходной холст (источник сэмплирования —
    всегда исходный полноразмерный кадр), поэтому апскейл получается за один
    интерполяционный проход, без потерь от промежуточного ресайза целого кадра.
    ``None`` — апскейл вообще не считается (экономит время: без умножения размеров и
    без INTER_CUBIC).
    """
    minx, miny, maxx, maxy = crop_ext
    if upscale is None:
        out_w = max(1, int(round(maxx - minx)))
        out_h = max(1, int(round(maxy - miny)))
        flags = cv2.INTER_LINEAR
    else:
        out_w = max(1, int(round((maxx - minx) * upscale)))
        out_h = max(1, int(round((maxy - miny) * upscale)))
        flags = cv2.INTER_CUBIC
    src = _bbox_corners(cx, cy, angle, (minx, miny, maxx, maxy))
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, m, (out_w, out_h), flags=flags)


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
    защищённую зону, а не «впритык» очерченный Surya полигон.

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
        bbox = _bbox_corners(cx, cy, angle, ext).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [bbox], True, COLOR_ROT_BBOX, thickness, cv2.LINE_AA)
        crop_zone = crop_ext if crop_ext is not None else _ext_with_margins(ext, margins)
        crop = _bbox_corners(cx, cy, angle, crop_zone).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [crop], True, COLOR_CROP, thickness, cv2.LINE_AA)
    # Блоки layout — тонкой линией: их много, толстая рамка забила бы кадр.
    # Рисуем контуры маски С padding'ом, чтобы оверлей совпадал с реально
    # защищённой от закраски зоной (Surya обводит блок впритык).
    if layout_polygons:
        layout_mask = polygons_to_mask(out.shape, layout_polygons, layout_pad_px)
        contours, _ = cv2.findContours(layout_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_LAYOUT_BLOCK, max(1, thickness // 2), lineType=cv2.LINE_AA)
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


# ============================================================
# Сбор файлов и сохранение
# ============================================================


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    """Собирает изображения (по расширению, без учёта регистра).

    ``recursive=False`` — только верхний уровень каталога; ``True`` — рекурсивно.
    """
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _imwrite_params(suffix: str) -> list[int]:
    """Параметры cv2.imwrite под формат (качество JPEG / сжатие PNG / сжатие TIFF)."""
    s = suffix.lower()
    if s in (".jpg", ".jpeg"):
        return [cv2.IMWRITE_JPEG_QUALITY, 95]
    if s == ".png":
        return [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if s in (".tif", ".tiff"):
        # LZW — сжатие БЕЗ потерь (в отличие от JPEG-in-TIFF); задаём явно, чтобы
        # не зависеть от дефолта cv2. Код 5 = COMPRESSION_LZW (libtiff).
        return [cv2.IMWRITE_TIFF_COMPRESSION, 5]
    return []


def _write_image(out_path: Path, img: np.ndarray, params: list[int], force_dpi: Optional[int]) -> None:
    """Сохраняет изображение. Без ``force_dpi`` — быстрый ``cv2.imwrite``.

    cv2.imwrite не умеет прописывать разрешение (DPI). Раньше при ``force_dpi`` файл
    после cv2 перечитывался PIL и пересохранялся с тегом dpi — это ВТОРОЙ проход
    кодека (для TIFF-LZW на 30-48 Мп — несколько секунд впустую, всё в один поток).
    Теперь TIFF с DPI пишется ОДНИМ проходом через PIL (LZW без потерь + тег
    разрешения). PNG-ветка (DPI нужен редко) оставлена прежней двухпроходной.
    """
    if force_dpi is None:
        cv2.imwrite(str(out_path), img, params)
        return

    if out_path.suffix.lower() in (".tif", ".tiff"):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        PILImage.fromarray(rgb).save(str(out_path), format="TIFF", compression="tiff_lzw", dpi=(force_dpi, force_dpi))
        return

    cv2.imwrite(str(out_path), img, params)
    with PILImage.open(out_path) as im:
        im.save(out_path, dpi=(force_dpi, force_dpi))


def _parse_light_increment(ctx, param, value: str) -> "tuple[float, float]":
    """Парсит ``--finger-zone-light-increment``: 'N' → (N, N), 'L,R' → (L, R)."""
    parts = [p.strip() for p in str(value).split(",")]
    try:
        if len(parts) == 1:
            v = float(parts[0])
            return (v, v)
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
    except ValueError:
        pass
    raise click.BadParameter("ожидается число ('20') или пара 'слева,справа' ('15,30')")


def _parse_layout_pad(ctx, param, value) -> "tuple[int, int]":
    """Парсит ``--layout-pad-px``: 'N' → (N, N), 'X,Y' → (X, Y) (запас по X и по Y)."""
    parts = [p.strip() for p in str(value).split(",")]
    try:
        if len(parts) == 1:
            v = int(parts[0])
            return (v, v)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        pass
    raise click.BadParameter("ожидается число ('12') или пара 'по_x,по_y' ('12,24')")


def _resolve_output_suffix(orig_suffix: str, output_format: Optional[str]) -> str:
    """Суффикс выходного файла: как у входа, если ``output_format`` не задан."""
    if output_format is None:
        return orig_suffix
    return ".png" if output_format.lower() == "png" else ".tiff"


# ============================================================
# CLI
# ============================================================


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Каталог с исходными изображениями (JPG/PNG)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Куда сохранять повёрнутые и обрезанные развороты (имя файла сохраняется)",
)
@click.option(
    "--debug-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Если задана — сюда кадр с оверлеями (граница/min-bbox/crop-зона)",
)
@click.option("--left-margin", default=0, show_default=True, help="Припуск crop-зоны слева, пикс. (>0 шире, <0 уже)")
@click.option("--top-margin", default=0, show_default=True, help="Припуск crop-зоны сверху, пикс. (>0 шире, <0 уже)")
@click.option("--right-margin", default=0, show_default=True, help="Припуск crop-зоны справа, пикс. (>0 шире, <0 уже)")
@click.option("--bottom-margin", default=0, show_default=True, help="Припуск crop-зоны снизу, пикс. (>0 шире, <0 уже)")
@click.option("--recursive", is_flag=True, default=False, help="Рекурсивно обходить подкаталоги в поисках картинок")
@click.option(
    "--skip-if-exists/--no-skip-if-exists",
    "skip_if_exists",
    default=True,
    show_default=True,
    help="Пропускать файл, если соответствующий результат уже есть в OUTPUT_DIR (докачка "
    "прерванного прогона). Проверяется только output-файл; debug-оверлей при обработке файла "
    "перезаписывается всегда. При --no-skip-if-exists всё пересчитывается заново.",
)
@click.option(
    "--output-format",
    type=click.Choice(["png", "tiff"], case_sensitive=False),
    default=None,
    help="Формат файлов в output-dir (по умолчанию — как у входного файла)",
)
@click.option(
    "--compensate-levels/--no-compensate-levels",
    "do_compensate_levels",
    default=False,
    show_default=True,
    help="Растягивать уровни по перцентилям внутри маски страницы (минус эрозия)",
)
@click.option(
    "--erosion-px",
    default=N_EROSION_PX,
    show_default=True,
    help="Эрозия маски страницы (пикс.) перед расчётом уровней (--compensate-levels)",
)
@click.option(
    "--extra-erosion-px",
    default=EXTRA_EROSION_PX,
    show_default=True,
    help="Доп. обрезка краёв силуэта книги перед копированием, пикс. (диляция на extra + "
    "эрозия на 2*extra → срезает тёмные фрагменты обложки в углах; 0 — выкл.)",
)
@click.option(
    "--upscale",
    default=None,
    type=float,
    show_default=True,
    help="Апскейл выходного изображения перед поворотом/кропом (по умолчанию — без апскейла)",
)
@click.option(
    "--remove-fingers/--no-remove-fingers",
    "do_remove_fingers",
    default=True,
    show_default=True,
    help="Детектировать и закрашивать пальцы (finger_removal) перед детекцией книги/кропом",
)
@click.option(
    "--finger-dilate-px",
    default=FINGER_DILATE_PX,
    show_default=True,
    help="Дилатация маски пальца, пикс. (шире — надёжнее докрашивает полутона на краю силуэта)",
)
@click.option(
    "--finger-zone-light-increment",
    "finger_zone_light_increment",
    default=str(FINGER_ZONE_LIGHT_INCREMENT),
    show_default=True,
    callback=_parse_light_increment,
    help="Осветление зоны пальца перед закраской: одно число (на весь кадр) "
    "либо 'слева,справа' (напр. 15,30) — если свет в кадре падает не симметрично",
)
@click.option(
    "--force-dpi",
    default=None,
    type=int,
    show_default=True,
    help="Принудительно прописать выходным изображениям указанный DPI (по умолчанию — не трогать)",
)
@click.option(
    "--max-asymmetric-dilation-ratio",
    "asymmetric_dilation_ratio",
    default=DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO,
    type=float,
    show_default=True,
    help="Максимальная ДОБАВКА к коэффициенту дилатации маски пальца по «выгодной» оси: "
    "боковой палец растёт по Y в (1 + ratio) раз, верхний/нижний — по X, угловой — в "
    "(1 + ratio/2) по обеим. 0 — прежняя круговая дилатация",
)
@click.option(
    "--protect-text-layout/--no-protect-text-layout",
    "protect_text_layout",
    default=False,
    show_default=True,
    help="Прогонять кадр через Surya layout ДО закраски и защищать найденные блоки (текст, "
    "заголовки, картинки, таблицы) от закраски пальца — позволяет ставить щедрую дилатацию "
    "под тень, не портя контент. Способ защиты — см. --text-protect-mode",
)
@click.option(
    "--text-protect-mode",
    type=click.Choice(PROTECT_MODES, case_sensitive=False),
    default=PROTECT_LIMIT_LAMA,
    show_default=True,
    help="Способ защиты блоков layout (--protect-text-layout): limit-lama-zone — вычесть блоки из "
    "зоны закраски; copy-back-layout-zones — закрасить всё, а потом скопировать пересекающиеся "
    "с зоной закраски блоки обратно с оригинала (лучше вычищает тень между блоками)",
)
@click.option(
    "--layout-pad-px",
    default=str(DEFAULT_LAYOUT_PAD_PX),
    show_default=True,
    callback=_parse_layout_pad,
    help="Запас вокруг блока layout, пикс. (--protect-text-layout): Surya обводит блок впритык, "
    "и без запаса закраска подъедает край строки. Число ('12') — одинаково по обеим осям, пара "
    "'по_x,по_y' ('12,24') — раздельно. Больше запас — целее контент, но хуже вычищается тень у "
    "самого блока; 0 — строго по границе блока",
)
@click.option(
    "--shadow-method",
    type=click.Choice(SHADOW_METHODS, case_sensitive=False),
    default="none",
    show_default=True,
    help="Коррекция теневой зоны вокруг пальца после зарисовки: none | classic | retinex | "
    "docshadow-sd7k | docshadow-kligler | docshadow-jung (нейросеть DocShadow)",
)
@click.option(
    "--bg-fill-method",
    type=click.Choice(BG_FILL_METHODS, case_sensitive=False),
    default=BG_FILL_AVERAGE,
    show_default=True,
    help="Способ заливки внешней зоны (в углы повёрнутого кропа за краем страницы): "
    "average — один усреднённый цвет по всей странице (старый); nearest — цвет ближайшей "
    "точки границы зоны копирования (Вороной): учитывает неравномерный свет и цветную "
    "обложку, можно сгладить через --bg-fill-blur-px",
)
@click.option(
    "--bg-fill-blur-px",
    type=float,
    default=0.0,
    show_default=True,
    help="Размытие зоны заливки (только для локальных методов заливки): 0 — выкл; иначе σ "
    "растёт от 0 у границы зоны копирования до этого значения вдали. Зона копирования не "
    "затрагивается",
)
@click.option(
    "--detect-pad-tb-px",
    type=int,
    default=250,
    show_default=True,
    help="Перед детекцией разворота добавить чёрную рамку сверху и снизу на N пикселей (маска "
    "возвращается в координатах кадра, рамка срезается обратно). Помогает на снимках, где книга "
    "занимает кадр целиком по вертикали и детектится «вся область как разворот»: чёрная рамка "
    "даёт SAM тёмную границу и отодвигает боксы от краёв. 0 — выкл",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="WARNING",
    show_default=True,
    help="Уровень логирования. INFO — печатать тайминги ключевых операций (timeit) по каждому кадру",
)
def main(
    input_dir: Path,
    output_dir: Path,
    debug_dir: Optional[Path],
    left_margin: int,
    top_margin: int,
    right_margin: int,
    bottom_margin: int,
    recursive: bool,
    skip_if_exists: bool,
    output_format: Optional[str],
    do_compensate_levels: bool,
    erosion_px: int,
    extra_erosion_px: int,
    upscale: Optional[float],
    do_remove_fingers: bool,
    finger_dilate_px: int,
    finger_zone_light_increment: "tuple[float, float]",
    force_dpi: Optional[int],
    asymmetric_dilation_ratio: float,
    protect_text_layout: bool,
    text_protect_mode: str,
    layout_pad_px: "tuple[int, int]",
    shadow_method: str,
    bg_fill_method: str,
    bg_fill_blur_px: float,
    detect_pad_tb_px: int,
    log_level: str,
) -> None:
    """Находит разворот, выпрямляет его поворотом и вырезает crop-зону в OUTPUT_DIR."""
    logging.getLogger().setLevel(log_level.upper())
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Припуски crop-зоны: (left, top, right, bottom) — по одному на сторону
    margins = (left_margin, top_margin, right_margin, bottom_margin)

    files = collect_images(input_dir, recursive)
    if not files:
        logger.warning("Изображения не найдены в %s", input_dir)
        return

    logger.info(
        "Файлов: %d | устройство: %s | margins: left=%d top=%d right=%d bottom=%d | recursive: %s | "
        "skip-if-exists: %s | "
        "output-format: %s | compensate-levels: %s (erosion-px=%d) | extra-erosion-px=%d | upscale: %s | "
        "remove-fingers: %s (dilate-px=%d, light-increment=слева=%g,справа=%g) | force-dpi: %s | "
        "max-asymmetric-dilation-ratio: %g | protect-text-layout: %s (mode=%s, pad-px=x=%d,y=%d) | "
        "shadow-method: %s | bg-fill-method: %s (blur-px=%g) | detect-pad-tb-px: %d",
        len(files),
        device,
        left_margin,
        top_margin,
        right_margin,
        bottom_margin,
        recursive,
        skip_if_exists,
        output_format or "как у входа",
        do_compensate_levels,
        erosion_px,
        extra_erosion_px,
        upscale if upscale is not None else "без апскейла",
        do_remove_fingers,
        finger_dilate_px,
        finger_zone_light_increment[0],
        finger_zone_light_increment[1],
        force_dpi if force_dpi is not None else "не трогать",
        asymmetric_dilation_ratio,
        protect_text_layout,
        text_protect_mode,
        layout_pad_px[0],
        layout_pad_px[1],
        shadow_method,
        bg_fill_method,
        bg_fill_blur_px,
        detect_pad_tb_px,
    )

    for path in tqdm(files, desc="Crop", unit="img"):
        _t_frame = timeit.default_timer()
        try:
            # Путь результата (при recursive зеркалим подкаталоги; формат — из
            # --output-format либо как у входа). Считаем его ДО загрузки картинки,
            # чтобы --skip-if-exists мог пропустить файл, не тратя время на imread
            # и модели.
            rel = path.relative_to(input_dir)
            out_suffix = _resolve_output_suffix(path.suffix, output_format)
            out_path = (output_dir / rel).with_suffix(out_suffix)
            # Докачка прерванного прогона: пропускаем файл, только если готов
            # OUTPUT-файл. debug-оверлей при фактической обработке переписывается
            # всегда (его наличие на решение о пропуске не влияет).
            if skip_if_exists and out_path.exists():
                logger.info("Пропуск (результат уже есть): %s", rel)
                continue
            params = _imwrite_params(out_suffix)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with log_timing("imread", path.name):
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                tqdm.write(f"  Не удалось загрузить: {path.name}")
                continue

            bgr_orig = bgr  # для debug-оверлея: без удаления пальцев и без компенсации уровней
            finger_mask: Optional[np.ndarray] = None
            lama_roi_bboxes: Optional[list] = None
            finger_boxes: Optional[np.ndarray] = None
            finger_mask_predilate: Optional[np.ndarray] = None
            layout_polys: Optional[list] = None
            if do_remove_fingers:
                with log_timing("remove_fingers", path.name):
                    (
                        bgr,
                        finger_mask,
                        lama_roi_bboxes,
                        finger_boxes,
                        finger_info,
                        finger_mask_predilate,
                        layout_polys,
                    ) = remove_fingers(
                        bgr,
                        device,
                        want_boxes=debug_dir is not None,
                        dilate_px=finger_dilate_px,
                        light_increment=finger_zone_light_increment,
                        asymmetric_dilation_ratio=asymmetric_dilation_ratio,
                        protect_text=protect_text_layout,
                        protect_mode=text_protect_mode,
                        layout_pad_px=layout_pad_px,
                        log_name=path.name,
                    )
                if int(np.count_nonzero(finger_mask)) > 0:
                    tqdm.write(f"  Пальцы: {finger_info} ({path.name})")

            with log_timing("page_mask", path.name):
                mask = page_mask(
                    bgr, device, pad_tb_px=detect_pad_tb_px
                )  # E1 — силуэт разворота (светлые страницы + куски обложки)

            # Коррекция теневой зоны вокруг пальца (после зарисовки, до кропа/уровней)
            if shadow_method != "none" and finger_mask is not None:
                with log_timing("correct_finger_shadow", path.name):
                    bgr = correct_finger_shadow(bgr, finger_mask, mask, shadow_method, device=device)

            with log_timing("min_area_rotated_bbox", path.name):
                geom = min_area_rotated_bbox(mask)  # B1/B2 строим по E1
            # E2 — область копирования: E1 с обрезанными периферийными фрагментами обложки
            with log_timing("trim_cover_fragments", path.name):
                copy_mask = trim_cover_fragments(mask, extra_erosion_px)
            if do_compensate_levels:
                with log_timing("compensate_levels", path.name):
                    bgr_leveled = compensate_levels(bgr, mask, erosion_px)
            else:
                bgr_leveled = bgr

            # Блоки layout нужны и для расширения crop-зоны, и для debug-оверлея.
            # При удалении пальцев с защитой текста они уже посчитаны в remove_fingers;
            # иначе, при включённой --protect-text-layout, считаем здесь.
            if layout_polys is None and protect_text_layout:
                with log_timing("layout_polygons", path.name):
                    layout_polys = layout_polygons(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            with log_timing("polygons_to_mask", path.name):
                layout_mask = polygons_to_mask(mask.shape, layout_polys, layout_pad_px) if layout_polys else None

            crop_ext: Optional[tuple] = None
            if geom is None:
                # Разворот не найден — кладём оригинал, чтобы не терять файл в пайплайне
                tqdm.write(f"  Разворот не найден, сохраняю оригинал: {rel}")
                with log_timing("write_image", path.name):
                    _write_image(out_path, bgr_leveled, params, force_dpi)
            else:
                with log_timing("crop_geometry", path.name):
                    cx, cy, angle, ext = geom
                    # Финальная crop-зона: ext с припусками, расширенный так, чтобы целиком
                    # вместить блоки layout (иначе отриц. припуски срезают часть обложки).
                    margined = _ext_with_margins(ext, margins)
                    crop_ext = crop_ext_with_layout(ext, margins, _layout_ext_bounds(cx, cy, angle, layout_mask))
                    # Область копирования (E2) расширяем до расширенного crop-bbox: в кольце
                    # между bbox с припусками и расширенным под layout bbox копируем контент
                    # страницы (E1), иначе fill_outside_mask замажет там обложку фоном — ту
                    # самую, ради которой crop и расширяли.
                    if crop_ext != margined:
                        ring = cv2.bitwise_and(
                            _ext_to_mask(mask.shape, cx, cy, angle, crop_ext),
                            cv2.bitwise_not(_ext_to_mask(mask.shape, cx, cy, angle, margined)),
                        )
                        copy_mask = cv2.bitwise_or(copy_mask, cv2.bitwise_and(mask, ring))
                # Копируем только E2 ∩ B2: всё в B2 вне E2 заливаем цветом края (--bg-fill-method)
                with log_timing(f"fill_outside_mask[{bg_fill_method}]", path.name):
                    bgr_for_crop = fill_outside_mask(
                        bgr_leveled, copy_mask, method=bg_fill_method, blur_px=bg_fill_blur_px
                    )
                with log_timing("crop_rotated", path.name):
                    crop = crop_rotated(bgr_for_crop, cx, cy, angle, crop_ext, upscale)
                with log_timing("write_image", path.name):
                    _write_image(out_path, crop, params, force_dpi)

            if debug_dir is not None:
                dbg_path = (debug_dir / rel).with_suffix(".jpg")
                dbg_path.parent.mkdir(parents=True, exist_ok=True)
                with log_timing("draw_overlay", path.name):
                    overlay = draw_overlay(
                        bgr_orig,
                        mask,
                        geom,
                        margins,
                        finger_mask,
                        lama_roi_bboxes,
                        finger_boxes,
                        copy_mask=copy_mask,
                        finger_mask_predilate=finger_mask_predilate,
                        layout_polygons=layout_polys,
                        layout_pad_px=layout_pad_px,
                        crop_ext=crop_ext,
                    )
                with log_timing("write_debug_overlay", path.name):
                    cv2.imwrite(str(dbg_path), overlay, _imwrite_params(".jpg"))

            logger.info("%7.0f мс: ИТОГО кадр (%s)", (timeit.default_timer() - _t_frame) * 1000.0, path.name)

        except Exception as e:
            tqdm.write(f"  Ошибка {path.name}: {e}")
            import traceback

            tqdm.write(traceback.format_exc())

    logger.info("Готово. Crop → %s%s", output_dir, f" | debug → {debug_dir}" if debug_dir else "")


if __name__ == "__main__":
    main()
