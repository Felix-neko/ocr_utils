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
# Запасные подсказки — только если по основным не взят ни один экземпляр. Цветная обложка, снятая отдельно, для SAM 3
# не «страница бумаги» (пак-6: РИСК 1993/01 IMG_0176, 1993/02 IMG_0054 — 0 px по основным), а «magazine cover»
# берёт её точно по краю (уверенность 0.27–0.57). На крышках закрытых подшивок (ХиП 1991 — 5 кадров, ПХ 1975/01
# IMG_0001) и на разворотах она не срабатывает; «book cover» и «cover» берут и крышки — не годятся (решение
# пользователя 2026-10-25, замер — /mnt/hotstore/scan_processing/single_page/magazine_cover/table.tsv).
SAM3_FALLBACK_PROMPTS = ("magazine cover",)
# Уверенность экземпляра запасной подсказки, не ниже (основной порог SAM 3 — SAM3_CONF 0.25). Прогон по всем 28 кадрам
# пяти кэшей с пустым проходом SAM 3 (2026-10-25, ``/mnt/hotstore/scan_processing/single_page/magazine_cover/all.tsv``):
# подсказка сработала на двух голубых обложках РИСК 1993 (0.27–0.70; у IMG_0054 на кадре 0.35, без краски 0.27) и
# ложно — на закрытой подшивке в синем коленкоре с ярлыком РИСК 1991/02 IMG_0825 (проход по растянутым уровням, 0.25).
SAM3_FALLBACK_MIN_CONF = 0.3
# Запасные подсказки зовутся, если основные накрыли меньше этой доли грубой маски книги (решение пользователя
# 2026-10-25: не только при пустой бумаге — обложку, у которой SAM 3 взял одну плашку, тоже). По 4 739 кадрам пяти
# кэшей (ПХ 1974/01 и 1975, ХиП 1991, пак-5, пак-6) доля у обычных кадров 0.75–0.87 (1-й перцентиль 0.753), ниже 0.35 —
# 30 кадров: 28 с пустым проходом и 2 крышки ХиП 1991 с ярлыком (0.30, 0.33; «magazine cover» на них пуста); между 0.35
# и 0.7 — 8 кадров.
SAM3_FALLBACK_MIN_SHARE = 0.35
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
# Карман — на странице: в кадре есть страница не меньше POCKET_MIN_PAGE_MM2 (страница пака-6 — 64–90 тыс. мм²), блок не
# больше POCKET_MAX_SHARE её площади (карманы и вклейки выборки — до 0.32). Без этого на закрытой книге и обложке без
# бумаги (≤ 250 мм² бумаги) блоком «кармана» становилась светлая крышка переплёта (пак-6: РИСК 1991/03 IMG_0200, 1993/01
# IMG_0176, МТС 1990/07 IMG_0055 и ещё 7 кадров).
POCKET_MIN_PAGE_MM2 = 20000.0
POCKET_MAX_SHARE = 0.6
# Версия кэша шага SAM 3 (``sam3.npz``): с правилами пустых страниц и слияния вложенных экземпляров — 2, с запасными
# подсказками — 3, с порогом уверенности запасных — 4, запасные при доле книги ниже SAM3_FALLBACK_MIN_SHARE — 5. Кэш
# версий 2 и 4, где каждый проход накрыл не меньше SAM3_FALLBACK_MIN_SHARE книги, запасные не меняют
# (``pipeline.sam3_ready``); у версии 3 проход мог дать и запасная подсказка без порога — пересчёт.
SAM3_CACHE_VERSION = 5
SAM3_CACHE_COMPATIBLE = (2, 4)
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


# Растяжение уровней кадра перед SAM 3 (``--sam3-levels``, по умолчанию выключено; стенд ai_slop/stack_edges): бумага —
# пиксели в оболочках своей печати светлее порога Оцу; чёрная точка — SAM3_LEVELS_PCT[0]-й процентиль бумаги минус
# SAM3_LEVELS_MARGIN, белая — SAM3_LEVELS_PCT[1]-й плюс запас; растягивается только светлота L (Lab), усиление не больше
# SAM3_LEVELS_MAX_GAIN (узкий диапазон почти белой бумаги не растягивается в шум).
SAM3_LEVELS_PCT = (1.0, 99.0)
SAM3_LEVELS_MARGIN = 8
SAM3_LEVELS_MAX_GAIN = 4.0
SAM3_LEVELS_MIN_PAPER_PX = 1000


def stretch_paper_levels(bgr: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Кадр с растянутыми по бумаге уровнями светлоты — вход SAM 3.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        region: Где искать бумагу (bool): оболочки своей печати, без них — маска книги.

    Returns:
        ``(растянутый кадр, чёрная точка, белая точка)``; бумаги мало или диапазон вырожден — кадр как есть и (0, 255).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    values = gray[region]
    if values.size < SAM3_LEVELS_MIN_PAPER_PX:
        return bgr, 0.0, 255.0
    # Краска отсекается порогом Оцу по пикселям области: бумага — светлая мода.
    thr, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = values[values > thr]
    if paper.size < SAM3_LEVELS_MIN_PAPER_PX:
        return bgr, 0.0, 255.0
    lo, hi = np.percentile(paper, SAM3_LEVELS_PCT)
    white = min(255.0, float(hi) + SAM3_LEVELS_MARGIN)
    # Чёрная точка опускается так, чтобы диапазон был не уже 255 / SAM3_LEVELS_MAX_GAIN.
    black = max(0.0, min(float(lo) - SAM3_LEVELS_MARGIN, white - 255.0 / SAM3_LEVELS_MAX_GAIN))
    lut = np.clip((np.arange(256) - black) * 255.0 / (white - black), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), black, white


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


def sam3_pages(
    predictor,
    bgr: np.ndarray,
    hulls: np.ndarray,
    prompts=SAM3_PROMPTS,
    fallback_prompts=SAM3_FALLBACK_PROMPTS,
    book: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Страницы SAM 3 одним проходом: сырое объединение экземпляров и карта меток экземпляров-страниц.

    Отбор экземпляров (``select_instance``): с печатью — накрывает не меньше :data:`SAM3_MIN_PRINT` своей печати;
    без печати — не меньше :data:`SAM3_BLANK_MIN_SHARE` самого большого экземпляра кадра и уверенность не ниже
    :data:`SAM3_BLANK_MIN_CONF` (пустые форзацы). В карте меток экземпляры двух подсказок, совпадающие на
    ≥ :data:`SAM3_SAME_IOU` (IoU), — одна страница; мелкий экземпляр внутри выпуклой оболочки крупного (наклейка) —
    часть крупного (``merge_inserts``); пересечения отдаются экземпляру с большей оценкой. Если основные подсказки
    накрыли меньше :data:`SAM3_FALLBACK_MIN_SHARE` грубой маски книги (без маски — если не взят ни один экземпляр),
    к их экземплярам добавляются экземпляры запасных (цветная обложка, снятая отдельно) с уверенностью не ниже
    :data:`SAM3_FALLBACK_MIN_CONF`, и отбор повторяется по всем вместе: плашка, взятая основными, сливается с
    обложкой как вложенный экземпляр.

    Args:
        predictor: ``SAM3SemanticPredictor``.
        bgr: Кадр BGR (масштаб сегментации).
        hulls: Оболочки своей печати (отбор экземпляров).
        prompts: Основные текстовые подсказки.
        fallback_prompts: Запасные подсказки (пустой кортеж — без них).
        book: Грубая маска книги (bool, масштаб сегментации) — от неё доля накрытого основными; None — запасные
            только при пустом отборе.

    Returns:
        ``(сырая маска bool, карта меток uint8: 0 — ничего, 1..N — страницы)``.
    """
    import torch

    with torch.inference_mode():
        predictor.set_image(bgr)
        candidates = sam3_candidates(predictor, prompts, hulls.shape)
        found = choose_instances(candidates, hulls)
        if fallback_prompts and covered_share(found, book) < SAM3_FALLBACK_MIN_SHARE:
            # Запасная подсказка — со своим порогом уверенности: на пороге SAM 3 она берёт и крышку подшивки.
            fallback = [c for c in sam3_candidates(predictor, fallback_prompts, hulls.shape)
                        if c[0] >= SAM3_FALLBACK_MIN_CONF]  # fmt: skip
            if fallback:
                found = choose_instances(candidates + fallback, hulls)
    # Сбросить картинку предиктора и кэш CUDA: иначе на следующем кадре растёт видеопамять.
    if hasattr(predictor, "reset_image"):
        predictor.reset_image()
    torch.cuda.empty_cache()
    raw = np.zeros(hulls.shape, bool)
    labels = np.zeros(hulls.shape, np.uint8)
    # Пересечения — экземпляру с большей оценкой: он кладётся последним.
    for k, (_, m) in sorted(enumerate(found), key=lambda t: t[1][0]):
        raw |= m
        labels[m] = k + 1
    return raw, labels


def covered_share(found: list[tuple[float, np.ndarray]], book: np.ndarray | None) -> float:
    """Доля грубой маски книги, накрытая отобранными экземплярами.

    Args:
        found: Экземпляры ``(уверенность, маска bool)``.
        book: Грубая маска книги или None.

    Returns:
        Доля 0…1; без маски книги (или пустой) — 1, если что-то отобрано, иначе 0.
    """
    if book is None or not book.any():
        return 1.0 if found else 0.0
    covered = np.zeros(book.shape, bool)
    for _, m in found:
        covered |= m
    return float((covered & book).sum() / book.sum())


def sam3_candidates(predictor, prompts, shape: tuple[int, int]) -> list[tuple[float, np.ndarray]]:
    """Все экземпляры SAM 3 по подсказкам для картинки, уже поданной в предиктор (``set_image``).

    Args:
        predictor: ``SAM3SemanticPredictor`` с поданной картинкой.
        prompts: Текстовые подсказки (по одной за вызов).
        shape: ``(h, w)`` масштаба сегментации, к которому приводятся маски.

    Returns:
        Список ``(уверенность, маска bool)`` по всем подсказкам подряд.
    """
    h, w = shape
    candidates: list[tuple[float, np.ndarray]] = []
    for prompt in prompts:
        results = predictor(text=[prompt])
        if not results or results[0].masks is None:
            continue
        scores = results[0].boxes.conf.cpu().numpy()
        for m, sc in zip(results[0].masks.data.cpu().numpy() > 0.5, scores):
            candidates.append((float(sc), cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0))
    return candidates


def choose_instances(candidates: list[tuple[float, np.ndarray]], hulls: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Отбор экземпляров-страниц: правила ``select_instance``, склейка дублей разных подсказок, слияние вложенных.

    Args:
        candidates: ``(уверенность, маска bool)`` по всем подсказкам (``sam3_candidates``).
        hulls: Оболочки своей печати.

    Returns:
        Страницы ``(уверенность, маска bool)``; пустой список — ни один экземпляр не взят.
    """
    found: list[tuple[float, np.ndarray]] = []
    # «Самый большой» — из всех экземпляров кадра: на развороте из двух пустых страниц иначе не от чего отсчитывать.
    largest = max((int(m.sum()) for _, m in candidates), default=0)
    for sc, m in candidates:
        if not select_instance(m, sc, hulls, largest):
            continue
        # Тот же экземпляр по другой подсказке — склеить, иначе новая страница.
        dup = [k for k, (_, o) in enumerate(found) if (m & o).sum() / max((m | o).sum(), 1) >= SAM3_SAME_IOU]
        if dup:
            k = dup[0]
            found[k] = (max(found[k][0], sc), found[k][1] | m)
        else:
            found.append((sc, m))
    return merge_inserts(found)


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
    # Кармана без страницы не бывает: на закрытой книге светлая крышка — не карман.
    page_area = float(sizes.max())
    if page_area < POCKET_MIN_PAGE_MM2 * mm * mm:
        return np.zeros_like(paper)
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
        if area < POCKET_MIN_MM2 * mm * mm or area > POCKET_MAX_SHARE * page_area:
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
