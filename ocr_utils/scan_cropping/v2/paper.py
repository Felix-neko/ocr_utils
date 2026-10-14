"""Бумага внутри книги: SAM 3 по тексту «book page» / «paper page» (1288 px), постобработка ``plain``, цветовая модель бумаги."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.gpu_models import MODELS_DIR
from ocr_utils.scan_cropping.v2.book import disk
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM

# Веса SAM 3 (ultralytics ``SAM3SemanticPredictor``): закрытые веса взяты с зеркала ModelScope со сверкой SHA
# (``research/newspaper_crop/README.md``).
SAM3_WEIGHTS = MODELS_DIR / "sam3" / "sam3.pt"
# Сторона входа SAM 3, px: 1288 (кратно патчу 14) лучше родных 1008 держит кромку страницы (стенд подшивок).
SAM3_IMGSZ = 1288
# Порог уверенности экземпляра SAM 3.
SAM3_CONF = 0.25
# Текстовые подсказки страницы.
SAM3_PROMPTS = ("book page", "paper page")
# Порог доли печати, накрытой экземпляром SAM 3, чтобы взять его в бумагу.
SAM3_MIN_PRINT = 0.05
# Экземпляры двух подсказок с IoU не меньше этого — одна страница.
SAM3_SAME_IOU = 0.8
# Экземпляр без печати (пустой форзац, внутренняя сторона обложки, страница с цветной печатью, которую не ловят оболочки)
# берётся, если не меньше этой доли самого большого экземпляра кадра (из всех, не только с печатью) и уверенность SAM 3
# не ниже SAM3_BLANK_MIN_CONF (решение пользователя 2026-10-15). На выборке 103 кадров пака-6 так добавились только
# настоящие страницы (26 кадров, 0.71–0.96 уверенности), полосы торца и подклеенные листки (4–10 %) отсекаются.
SAM3_BLANK_MIN_SHARE = 0.30
SAM3_BLANK_MIN_CONF = 0.5
# Наклейка, ярлык, вклейка — не отдельная страница: экземпляр меньше SAM3_INSERT_MAX_SHARE самого большого, лежащий
# не меньше чем на SAM3_INSERT_INSIDE в выпуклой оболочке более крупного экземпляра, сливается с ним (МТС 1990/01
# IMG_0828, 1991/07 IMG_0787: наклейка на обложке давала свою трапецию и залив в трапеции обложки).
SAM3_INSERT_MAX_SHARE = 0.30
SAM3_INSERT_INSIDE = 0.90
# Карманы, листки и вклейки на странице, которые SAM 3 не взял в экземпляр страницы (РИСК 1993/03 IMG_0377, 1991/03
# IMG_0201, МТС 1991/04 002_IMG_0675), — ``fill_page_pockets``. Светлая бесцветная не-бумага в книге (L не ниже 5-го
# перцентиля бумаги минус POCKET_L_MARGIN, хрома не выше 95-го плюс POCKET_CHROMA_MARGIN) открывается
# диском POCKET_OPEN_MM: торец стопки и щели у корешка (полосы шириной до ~15 мм) уходят. Остаток минус бумага — кандидаты;
# берутся компактные блоки: площадь ≥ POCKET_MIN_MM2, удлинение повёрнутой рамки ≤ POCKET_MAX_ASPECT, заполнение рамки
# ≥ POCKET_MIN_FILL. На 36 кадрах выборки пака-6 у торца после открытия удлинение 4–10 или заполнение 0.1–0.36, у карманов
# и вклеек — удлинение ≤ 2.3, заполнение ≥ 0.49, площадь ≥ 3 500 мм².
POCKET_L_MARGIN = 10.0
POCKET_CHROMA_MARGIN = 10.0
POCKET_OPEN_MM = 20.0
# Закрытие кандидатов перед открытием, мм: сшивает карман, взятый SAM 3 пятнами.
POCKET_CLOSE_MM = 5.0
# Кромка бумаги у светлых кандидатов, мм, которая идёт в кандидаты (островки кармана, связанные со страницей).
POCKET_NEAR_MM = 5.0
POCKET_MIN_MM2 = 2000.0
POCKET_MAX_ASPECT = 2.5
POCKET_MIN_FILL = 0.45
# Доля блока внутри осевой рамки крупных страниц кадра, не меньше.
POCKET_MIN_INSIDE = 0.8
# Версия кэша шага SAM 3 (``sam3.npz``): с правилами пустых страниц и слияния вложенных экземпляров — 2.
SAM3_CACHE_VERSION = 2
# Затравки бумаги: пиксели внутри оболочек своей печати, не краска (серое выше локального фона минус INK_DROP).
INK_DROP = 25
# Окно локального фона бумаги (серое закрытие), px масштаба сегментации.
PAPER_BG_PX = 15
# Порог расстояния Махаланобиса (в ab Lab) для бумаги и запас по яркости ниже квантиля бумаги (тень изгиба).
MAHAL_THR = 4.0
L_MARGIN = 70
# Сглаживание кадра перед цветовой моделью (медиана), px — гасит краску и шум.
SMOOTH_PX = 7
# Закрытие маски бумаги, px.
CLOSE_PX = 9


def sam3_overrides(weights: str | Path = SAM3_WEIGHTS, imgsz: int = SAM3_IMGSZ) -> dict:
    """Настройки предиктора SAM 3 ultralytics.

    Args:
        weights: Путь к ``sam3.pt``.
        imgsz: Сторона входа сети, px.

    Returns:
        Словарь ``overrides``.
    """
    return dict(conf=SAM3_CONF, task="segment", mode="predict", model=str(weights), half=True, verbose=False,
                save=False, imgsz=imgsz)  # fmt: skip


def load_sam3(weights: str | Path = SAM3_WEIGHTS, imgsz: int = SAM3_IMGSZ):
    """Предиктор SAM 3 по тексту (грузится один раз на прогон).

    Args:
        weights: Путь к ``sam3.pt``.
        imgsz: Сторона входа сети, px.

    Returns:
        ``SAM3SemanticPredictor``.
    """
    from ultralytics.models.sam import SAM3SemanticPredictor

    return SAM3SemanticPredictor(overrides=sam3_overrides(weights, imgsz))


def sam3_pages(predictor, bgr: np.ndarray, hulls: np.ndarray, prompts=SAM3_PROMPTS) -> tuple[np.ndarray, np.ndarray]:
    """Страницы SAM 3 одним проходом: сырое объединение экземпляров и карта меток экземпляров-страниц.

    Отбор экземпляров (``select_instance``): с печатью — накрывает не меньше :data:`SAM3_MIN_PRINT` своей печати;
    без печати — не меньше :data:`SAM3_BLANK_MIN_SHARE` самого большого экземпляра кадра и уверенность не ниже
    :data:`SAM3_BLANK_MIN_CONF` (пустые форзацы). В карте меток экземпляры двух подсказок, совпадающие на
    ≥ :data:`SAM3_SAME_IOU` (IoU), — одна страница; мелкий экземпляр внутри выпуклой оболочки крупного (наклейка) —
    часть крупного (``merge_inserts``); пересечения отдаются экземпляру с большей оценкой.

    Args:
        predictor: ``SAM3SemanticPredictor``.
        bgr: Кадр BGR (масштаб сегментации).
        hulls: Оболочки своей печати (отбор экземпляров).
        prompts: Текстовые подсказки.

    Returns:
        ``(сырая маска bool, карта меток uint8: 0 — ничего, 1..N — страницы)``.
    """
    import torch

    h, w = hulls.shape
    candidates: list[tuple[float, np.ndarray]] = []
    with torch.inference_mode():
        predictor.set_image(bgr)
        for prompt in prompts:
            results = predictor(text=[prompt])
            if not results or results[0].masks is None:
                continue
            scores = results[0].boxes.conf.cpu().numpy()
            for m, sc in zip(results[0].masks.data.cpu().numpy() > 0.5, scores):
                candidates.append(
                    (float(sc), cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
                )
    # Сбросить картинку предиктора и кэш CUDA: иначе на следующем кадре растёт видеопамять.
    if hasattr(predictor, "reset_image"):
        predictor.reset_image()
    torch.cuda.empty_cache()
    raw = np.zeros((h, w), bool)
    found: list[tuple[float, np.ndarray]] = []
    # «Самый большой» — из всех экземпляров кадра: на развороте из двух пустых страниц иначе не от чего отсчитывать.
    largest = max((int(m.sum()) for _, m in candidates), default=0)
    for sc, m in candidates:
        if not select_instance(m, sc, hulls, largest):
            continue
        raw |= m
        # Тот же экземпляр по другой подсказке — склеить, иначе новая страница.
        dup = [k for k, (_, o) in enumerate(found) if (m & o).sum() / max((m | o).sum(), 1) >= SAM3_SAME_IOU]
        if dup:
            k = dup[0]
            found[k] = (max(found[k][0], sc), found[k][1] | m)
        else:
            found.append((sc, m))
    found = merge_inserts(found)
    labels = np.zeros((h, w), np.uint8)
    # Пересечения — экземпляру с большей оценкой: он кладётся последним.
    for k, (_, m) in sorted(enumerate(found), key=lambda t: t[1][0]):
        labels[m] = k + 1
    return raw, labels


def select_instance(mask: np.ndarray, score: float, hulls: np.ndarray, largest: int) -> bool:
    """Берётся ли экземпляр SAM 3 в бумагу: с печатью — всегда, без печати — крупный и уверенный.

    Args:
        mask: Маска экземпляра (bool).
        score: Уверенность SAM 3.
        hulls: Оболочки своей печати (bool).
        largest: Площадь самого большого экземпляра кадра, px.

    Returns:
        True — экземпляр берётся.
    """
    if (mask & hulls).sum() >= SAM3_MIN_PRINT * max(int(hulls.sum()), 1):
        return True
    return mask.sum() >= SAM3_BLANK_MIN_SHARE * largest and score >= SAM3_BLANK_MIN_CONF


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """Выпуклая оболочка маски (bool того же размера)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.fillPoly(out, [cv2.convexHull(np.vstack(contours))], 1)
    return out > 0


def merge_inserts(found: list[tuple[float, np.ndarray]]) -> list[tuple[float, np.ndarray]]:
    """Наклейки и вклейки — к своей странице: мелкий экземпляр внутри выпуклой оболочки крупного сливается с ним.

    Экземпляр меньше :data:`SAM3_INSERT_MAX_SHARE` самого большого, лежащий не меньше чем на :data:`SAM3_INSERT_INSIDE`
    в выпуклой оболочке более крупного, добавляется к его маске (оценка — большего). Выпуклая оболочка, а не сама маска:
    у страницы на месте наклейки вырез, и открытый к краю залив заливка дыр не закрывает. Узкая страница у корешка
    лежит рядом с соседней, а не внутри её оболочки, и не сливается.

    Args:
        found: Экземпляры-страницы ``(уверенность, маска)``.

    Returns:
        Экземпляры после слияния.
    """
    if len(found) < 2:
        return found
    areas = [int(m.sum()) for _, m in found]
    biggest = max(areas)
    order = sorted(range(len(found)), key=lambda k: areas[k])
    merged = {k: (sc, m.copy()) for k, (sc, m) in enumerate(found)}
    for k in order:
        if areas[k] >= SAM3_INSERT_MAX_SHARE * biggest or k not in merged:
            continue
        small = merged[k][1]
        hosts = [j for j in merged if j != k and areas[j] > areas[k]]
        for j in sorted(hosts, key=lambda j: -areas[j]):
            if (small & convex_hull_mask(merged[j][1])).sum() >= SAM3_INSERT_INSIDE * small.sum():
                sc_j, m_j = merged[j]
                merged[j] = (max(sc_j, merged[k][0]), m_j | small)
                del merged[k]
                break
    return [merged[k] for k in sorted(merged)]


def postprocess_plain(raw: np.ndarray, book: np.ndarray, hulls: np.ndarray) -> np.ndarray:
    """Маска бумаги из сырой маски SAM 3 (вариант ``plain``): закрытие, открытие, компоненты с печатью, заливка дыр.

    Args:
        raw: Сырая маска SAM 3.
        book: Маска книги.
        hulls: Оболочки своей печати.

    Returns:
        Маска бумаги.
    """
    paper = raw & book
    paper = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_CLOSE, disk(CLOSE_PX)) > 0
    paper = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_OPEN, disk(3)) > 0
    # Компоненты, задевающие печать, и крупные без печати — пустые страницы (как отбор экземпляров ``select_instance``:
    # иначе пустой форзац, взятый SAM 3, здесь снова выпадал, МТС 1990/06 IMG_1354).
    labels, count = ndimage.label(paper)
    if count:
        sizes = np.bincount(labels.ravel())[1:]
        keep = set(np.unique(labels[hulls & paper]).tolist()) - {0}
        keep |= {k + 1 for k in np.nonzero(sizes >= SAM3_BLANK_MIN_SHARE * sizes.max())[0]}
        paper = np.isin(labels, sorted(keep))
    paper = ndimage.binary_fill_holes(paper)
    return paper & book


def disk_closing(mask: np.ndarray, radius: float) -> np.ndarray:
    """Закрытие маски диском радиуса ``radius`` px через расстояния (быстро на больших радиусах)."""
    dilated = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= radius
    return cv2.distanceTransform(dilated.astype(np.uint8), cv2.DIST_L2, 5) > radius


def disk_opening(mask: np.ndarray, radius: float) -> np.ndarray:
    """Открытие маски диском радиуса ``radius`` px через расстояния (быстро на больших радиусах)."""
    eroded = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) > radius
    return (cv2.distanceTransform((~eroded).astype(np.uint8), cv2.DIST_L2, 5) <= radius) & mask


def fill_page_pockets(
    bgr: np.ndarray, paper: np.ndarray, book: np.ndarray, bg: np.ndarray, hulls: np.ndarray
) -> np.ndarray:
    """Карманы, листки, вклейки на странице, не взятые SAM 3: компактные блоки светлой бесцветной не-бумаги в книге.

    Цвет «бумаги» — относительно бумаги кадра (затравки ``paper_seeds``, без печати — сама бумага SAM 3), поэтому правило
    работает и на пожелтевшей бумаге. Открытие диском :data:`POCKET_OPEN_MM` по «бумага ∪ светлое» отсекает торец стопки
    и щели у корешка, затем берутся компактные блоки (см. константы). Вклейка, лежащая через стык страниц, берётся целиком;
    по страницам её делит ``glue.split_pages`` (ближайший экземпляр SAM 3) — страница под вклейкой не теряет низ.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        paper: Бумага (bool).
        book: Грубая маска книги (bool).
        bg: Фон (bool).
        hulls: Оболочки своей печати (bool).

    Returns:
        Добавка к бумаге (bool), без пикселей ``paper``.
    """
    seeds = paper_seeds(bgr, hulls, book)
    if seeds.sum() < 500:
        seeds = cv2.erode(paper.astype(np.uint8), disk(7)) > 0
    if seeds.sum() < 500:
        return np.zeros_like(paper)
    lab = smooth_lab(bgr)
    light, chroma = lab[..., 0], np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    pale = (light >= np.percentile(light[seeds], 5) - POCKET_L_MARGIN) & (
        chroma <= np.percentile(chroma[seeds], 95) + POCKET_CHROMA_MARGIN
    )
    mm = SEG_PX_PER_MM
    # Кандидаты — светлое вне бумаги, плюс то, что SAM 3 взял из кармана пятнами (РИСК 1991/03 IMG_0201: половина
    # кармана — островки бумаги, часть связана со страницей): светлые островки бумаги, отделённые от крупных страниц, и
    # светлая кромка бумаги у кандидатов шириной POCKET_NEAR_MM. Саму страницу целиком в открытие не включать: полоса
    # торца вплотную к странице вместе с ней пережила бы открытие.
    free = book & ~bg & pale & ~paper
    labels, _ = ndimage.label(paper)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    pages = np.isin(labels, np.nonzero(sizes >= SAM3_BLANK_MIN_SHARE * sizes.max())[0])
    near = cv2.distanceTransform((~free).astype(np.uint8), cv2.DIST_L2, 5) <= POCKET_NEAR_MM * mm
    candidates = free | (paper & pale & (~pages | near))
    joined = disk_closing(candidates, POCKET_CLOSE_MM * mm) & book & ~bg
    opened = disk_opening(joined, POCKET_OPEN_MM * mm)
    opened = cv2.morphologyEx(opened.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Осевая рамка крупных страниц кадра: по ширине — от левого края левой до правого края правой, по высоте — от верха
    # самой высокой до низа самой низкой (страница, у которой SAM 3 не взял низ, не укорачивает рамку).
    spread = np.zeros_like(paper)
    if pages.any():
        ys, xs = np.nonzero(pages)
        spread[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    added = np.zeros_like(paper)
    for k in range(1, count):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < POCKET_MIN_MM2 * mm * mm:
            continue
        comp = labels == k
        rect = cv2.minAreaRect(np.column_stack(np.nonzero(comp))[:, ::-1].astype(np.float32))
        short, long_ = sorted((max(rect[1][0], 1.0), max(rect[1][1], 1.0)))
        if long_ / short > POCKET_MAX_ASPECT or area / (short * long_) < POCKET_MIN_FILL:
            continue
        # Карман и вклейка лежат в пределах разворота; веер торца стопки — за наружным краем страницы (РИСК 1991/01
        # IMG_0390: блок 2 800 мм² за правой страницей).
        if spread[comp].mean() < POCKET_MIN_INSIDE:
            continue
        # Открытие скругляет углы кармана; повёрнутая рамка блока прилегает к его прямым сторонам — светлое внутри неё
        # добирает углы, а торец за краем страницы остаётся снаружи рамки.
        box = np.zeros(paper.shape, np.uint8)
        cv2.fillPoly(box, [np.round(cv2.boxPoints(rect)).astype(np.int32)], 1)
        added |= (comp | ((box > 0) & free)) & ~paper
    return added


def paper_seeds(bgr: np.ndarray, hulls: np.ndarray, book: np.ndarray) -> np.ndarray:
    """Затравки бумаги: внутри оболочек своей печати и книги, без краски.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        hulls: Оболочки своей печати.
        book: Маска книги.

    Returns:
        Булева маска.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    local = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, disk(PAPER_BG_PX))
    not_ink = gray.astype(np.int16) > local.astype(np.int16) - INK_DROP
    inner = cv2.erode((hulls & book).astype(np.uint8), disk(3)) > 0
    return inner & not_ink & (cv2.erode(not_ink.astype(np.uint8), disk(2)) > 0)


class PaperModel:
    """Робастная модель цвета бумаги в Lab: медиана и ковариация (a, b), нижний квантиль L."""

    def __init__(self, lab: np.ndarray, seeds: np.ndarray) -> None:
        """Строит модель по затравкам.

        Args:
            lab: Кадр Lab float32 (сглаженный).
            seeds: Затравки бумаги.
        """
        ab = lab[..., 1:][seeds]
        med = np.median(ab, axis=0)
        # Ядро затравок: 90% ближайших к медиане — выбросы (краска, тень) не портят ковариацию.
        d = np.linalg.norm(ab - med, axis=1)
        core = ab[d <= np.percentile(d, 90)]
        self.mean = core.mean(axis=0)
        self.icov = np.linalg.inv(np.cov(core.T) + np.eye(2) * 1.0)
        self.l_low = float(np.percentile(lab[..., 0][seeds], 2))

    def distance(self, lab: np.ndarray) -> np.ndarray:
        """Расстояние Махаланобиса по (a, b) от модели.

        Args:
            lab: Кадр Lab float32.

        Returns:
            Карта float32.
        """
        d = lab[..., 1:] - self.mean
        return np.sqrt(np.einsum("...i,ij,...j->...", d, self.icov, d)).astype(np.float32)

    def is_paper(self, lab: np.ndarray) -> np.ndarray:
        """Пиксели, похожие на бумагу: близко по оттенку и не темнее нижнего квантиля бумаги минус запас.

        Args:
            lab: Кадр Lab float32.

        Returns:
            Булева маска.
        """
        return (self.distance(lab) < MAHAL_THR) & (lab[..., 0] > self.l_low - L_MARGIN)


def smooth_lab(bgr: np.ndarray) -> np.ndarray:
    """Кадр в Lab float32 после медианы (без краски и шума).

    Args:
        bgr: Кадр BGR.

    Returns:
        Lab float32.
    """
    return cv2.cvtColor(cv2.medianBlur(bgr, SMOOTH_PX), cv2.COLOR_BGR2LAB).astype(np.float32)
