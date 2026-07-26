"""Детекция разворота книги в кадре: YOLO-World боксы → SAM силуэт → маска.

Отвечает за КАНОНИЧЕСКУЮ маску разворота (``page_mask``) и всё, что нужно для её
получения: загрузку моделей, отбор боксов и доводку силуэта. Дальнейшее —
поворот, кроп, заливка фона — в ``ocr_utils.detect_and_crop``, который эту маску
потребляет.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ocr_utils.finger_removal.masking import _suppress_nested_boxes


# Папка для весов нейромоделей (корень проекта, рядом с finger_models)
MODELS_DIR = Path(__file__).resolve().parents[1] / "finger_models"

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

_MODEL_CACHE: dict = {}


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


def detect_page_mask(bgr: np.ndarray, device: str, frame_area: Optional[float] = None) -> np.ndarray:
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
    # Доли «near-full-frame» боксов/масок считаются относительно frame_area — площади
    # ИСХОДНОГО кадра БЕЗ добавленного pad_tb_px чёрного поля (см. page_mask). Иначе
    # padding занижает долю, и мусорный full-frame бокс проскакивает под пороги
    # LARGE_BOX_CONF_TIERS (см. IMG_0017). Если не задано — весь переданный кадр.
    frame_area = float(h * w) if frame_area is None else frame_area

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
    size_ok = (area >= MIN_PAGE_FRAC * frame_area) & (area <= MAX_PAGE_FRAC * frame_area)
    # Плюс: near-full-frame боксы с недостаточной уверенностью — это шум YOLO-World, чей
    # SAM-силуэт сгребает фон и руку по краям (см. LARGE_BOX_CONF_TIERS / IMG_0012, IMG_0014).
    junk_fullframe = np.zeros(len(boxes), dtype=bool)
    for frac_thr, conf_thr in LARGE_BOX_CONF_TIERS:
        junk_fullframe |= (area >= frac_thr * frame_area) & (confs < conf_thr)
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
            if MIN_PAGE_FRAC * frame_area <= m_bin.sum() <= MAX_PAGE_FRAC * frame_area:
                mask = cv2.bitwise_or(mask, m_bin * 255)
    return mask


def page_mask(bgr: np.ndarray, device: str, pad_tb_px: int = 0, _unpad_ratio: float = 1.0) -> np.ndarray:
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
    границу и отодвигает YOLO-боксы от краёв.

    ВАЖНО: доли near-full-frame боксов при этом считаются относительно ИСХОДНОГО
    кадра (без добавленной рамки) — иначе рост площади кадра занижал бы долю и
    мусорный full-frame бокс проскакивал под пороги LARGE_BOX_CONF_TIERS (см.
    IMG_0017: с рамкой маска раздувалась до 0.99 кадра). Для этого во внутренний
    вызов пробрасывается ``_unpad_ratio`` = доля исходной высоты в padded-кадре.
    """
    h, w = bgr.shape[:2]
    if pad_tb_px > 0:
        padded = cv2.copyMakeBorder(bgr, pad_tb_px, pad_tb_px, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = page_mask(padded, device, _unpad_ratio=h / float(h + 2 * pad_tb_px))
        return mask[pad_tb_px : pad_tb_px + h, :]
    scale = WORK_SIDE / max(h, w) if max(h, w) > WORK_SIDE else 1.0
    work = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
    # Площадь ИСХОДНОГО кадра (без padding) в координатах work — знаменатель для долей.
    frame_area = float(work.shape[0] * work.shape[1]) * _unpad_ratio
    mask = detect_page_mask(work, device, frame_area=frame_area)
    if scale != 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = refine_page_mask(mask)
    return bridge_component_gaps(mask)


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
