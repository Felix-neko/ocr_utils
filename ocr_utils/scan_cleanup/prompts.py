"""Промпты Stable Diffusion и правило выбора по тому, куда попала зона закраса.

ЗАЧЕМ РАЗНЫЕ ПРОМПТЫ. SD заполняет дыру тем, что описано словами, и одно описание
на все случаи заведомо неверно: библиотечная печать стоит и на пустом поле полосы,
и поверх цветной обложки, и поверх полутоновой фотографии. По паку-1 таких «внутри
растра» — 226 масок из 431, то есть больше половины.

Виды промптов ровно три, по фону под зоной, а не по виду разметки: чем заполнять
дыру, определяет то, что под ней лежит, а не причина, по которой её убирают.
Поэтому у ``other_removal`` то же правило, а собственного «промпта для прочего» нет
— есть лишь добавка «продолжи окружающее», потому что этот вид разнороден и
описать его содержательно нельзя.

ПРЯМОУГОЛЬНИКУ ВО ВСЮ ПОЛОСУ ВЕРИТЬ НЕЛЬЗЯ. У обложки растровая область накрывает
кадр целиком — вместе с белыми полями, на которых как раз и стоит библиотечная
печать. Формально такая зона «внутри растра», а на деле лежит на чистой бумаге, и
промпт про цветную обложку заставляет SD дорисовать на её месте картинку. Проверено
на 1970/02 IMG_0053_2R: SD нарисовала поверх поля цветную иллюстрацию. По паку-1
таких масок на обложках 224 из 431, то есть больше половины.

Поэтому правило двухступенчатое: ТЕСНЫЙ прямоугольник из базы (его человек обвёл
вокруг конкретной иллюстрации) решает сам, а полосный — не решает ничего, и фон
меряется по пикселям вокруг зоны теми же измерителями, что и в детекции разметки:
``processing.has_halftone`` отвечает «есть ли тут печатная картинка вообще», а
``detection.color_kind.classify`` — цветная она или полутоновая.

Промпты по-английски: обе модели обучены на английских подписях, русский текст
им приходится переводить внутри текстового энкодера, и качество от этого падает.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.background_smoothing.processing import has_halftone
from ocr_utils.scan_cleanup.source import PageMarkup, Rect
from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_GRAYSCALE, MASK_OTHER_REMOVAL
from ocr_utils.scan_markup.detection.color_kind import classify, paper_color

# Доля площади РАМКИ ГРУППЫ, которая должна лежать внутри растровой области, чтобы
# считать зону «внутри растра». Именно рамки группы, а не ROI: ROI вдвое больше и у
# зоны на краю иллюстрации почти всегда вылезает на бумагу, так что по нему любая
# приграничная печать считалась бы «на бумаге».
RASTER_INSIDE_FRAC = 0.5

# Доля полосы, начиная с которой прямоугольник считается полосным и перестаёт
# что-либо локализовать: решение по нему передаётся замеру по пикселям.
FULL_PAGE_FRAC = 0.9

# Тональный разброс (p90 - p10 по яркости), ниже которого окрестность считается
# ПЛОСКОЙ, то есть бумагой, а не печатной картинкой. Замер по паку-1:
#   поле текстовой полосы          10
#   тонированная бумага обложки    32   <- порог проходит здесь
#   полутоновые иллюстрации   160-182
#   цветные иллюстрации       105-136
# Одного ``has_halftone`` мало: у обложки бумага серовато-сиреневая, целиком лежит в
# диапазоне «средних тонов», и детектор растра честно отвечает «растр есть». Разброс
# же отличает плоскую подложку от картинки с полутонами независимо от её оттенка.
TONE_SPREAD_MIN = 80.0

PROMPT_PAPER = (
    "plain light aged paper of an old printed magazine page, uniform warm off-white paper tone, "
    "faint paper grain, a few thin straight black printed rules, empty margin, no text, no letters"
)
PROMPT_COLOUR = (
    "flat colour magazine cover artwork, large solid colour polygons, smooth uniform colour fields, "
    "clean geometric shapes, offset print, no text, no letters"
)
PROMPT_HALFTONE = (
    "black and white halftone photograph in an old magazine, smooth continuous grey tones, "
    "offset printing screen, no text, no letters"
)
# Добавка для «прочего под удаление»: вид разнороден (наклейка, помарка, скрепка),
# и единственное осмысленное указание — продолжить то, что вокруг.
PROMPT_OTHER_SUFFIX = "continue the surrounding page content seamlessly, no foreign objects"

NEGATIVE_COMMON = (
    "text, letters, words, numbers, handwriting, signature, stamp, seal, ink blot, smudge, "
    "watermark, logo, ornament, face, blurry, distorted, artifacts, noise, duplicate"
)


@dataclass
class PromptSet:
    """Формулировки промптов; каждая переопределяется опцией CLI."""

    paper: str = PROMPT_PAPER
    colour: str = PROMPT_COLOUR
    halftone: str = PROMPT_HALFTONE
    other_suffix: str = PROMPT_OTHER_SUFFIX
    negative: str = NEGATIVE_COMMON


def overlap_frac(box: "tuple[int, int, int, int]", rect: Rect) -> float:
    """Доля площади ``box``, накрытая прямоугольником ``rect``."""
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area == 0:
        return 0.0
    ix1, iy1 = max(x1, rect.x1), max(y1, rect.y1)
    ix2, iy2 = min(x2, rect.x2), min(y2, rect.y2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1) / area


def raster_kind_at(
    box: "tuple[int, int, int, int]",
    regions: "tuple[Rect, ...]",
    min_frac: float = RASTER_INSIDE_FRAC,
    page_area: "int | None" = None,
    full_page_frac: float = FULL_PAGE_FRAC,
) -> "str | None":
    """Вид ТЕСНОЙ растровой области под зоной или ``None``.

    ``color_text`` в расчёт не идёт: это набор цветной краской, фон под ним — та же
    бумага, и заполнять его надо бумажным промптом.

    Прямоугольники крупнее ``full_page_frac`` полосы отбрасываются: они накрывают и
    поля тоже, то есть о фоне под конкретной зоной не говорят ничего (см. докстринг
    модуля). ``page_area = None`` отключает эту проверку.

    Если зона накрыта несколькими областями, берётся та, что накрывает сильнее:
    печать на стыке фотографии и обложки должна получить описание того фона,
    которого под ней больше.
    """
    best_kind, best = None, min_frac
    for r in regions:
        if r.kind not in (KIND_COLOR, KIND_GRAYSCALE):
            continue
        if page_area and r.area >= full_page_frac * page_area:
            continue
        frac = overlap_frac(box, r)
        if frac >= best:
            best_kind, best = r.kind, frac
    return best_kind


def background_kind(roi_bgr: np.ndarray, tone_spread_min: float = TONE_SPREAD_MIN) -> "str | None":
    """Что за фон в окрестности зоны, по пикселям: ``color``, ``grayscale`` или ``None``.

    ``None`` — плоская подложка (бумага, тонированная бумага обложки, ровная плашка):
    печатной картинки в окрестности нет.

    Три вопроса подряд, и порядок важен:

    1. ПЛОСКАЯ ли окрестность — по разбросу яркости (``TONE_SPREAD_MIN``). Плоская —
       значит бумага, и дальше спрашивать не о чем. Без этого шага серовато-сиреневая
       бумага обложки проходит как «полутоновая фотография»: она целиком лежит в
       диапазоне средних тонов;
    2. есть ли растр — крупные сплошные области средних тонов (``has_halftone``);
    3. и только потом — цветной он или полутоновый. Обратный порядок ошибался бы на
       цветном заголовке посреди белого поля: цвет есть, картинки нет.

    Меряется по ROI целиком, а не по кольцу вокруг зоны: измерения пространственные
    (морфология, разброс по кадру), а зона занимает не больше четверти ROI — он вдвое
    больше её по каждой стороне.
    """
    if roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    p10, p90 = np.percentile(gray, [10, 90])
    if p90 - p10 < tone_spread_min:
        return None
    if not has_halftone(gray):
        return None
    return classify(roi_bgr, paper_color(roi_bgr)).kind


def prompt_for(
    box: "tuple[int, int, int, int]",
    markup: PageMarkup,
    mask_kind: str,
    prompts: "PromptSet | None" = None,
    min_frac: float = RASTER_INSIDE_FRAC,
    roi_bgr: "np.ndarray | None" = None,
) -> "tuple[str, str]":
    """Промпт и негативный промпт для одной зоны закраса.

    ``roi_bgr`` — окрестность зоны; если она передана, а тесного прямоугольника из
    базы под зоной нет, фон определяется по пикселям (см. :func:`background_kind`).
    Без ``roi_bgr`` остаётся только база, и тогда зона на полосной обложке считается
    лежащей на бумаге — что вернее, чем считать её лежащей на картинке.
    """
    prompts = prompts or PromptSet()
    kind = raster_kind_at(box, markup.regions, min_frac, markup.width * markup.height)
    if kind is None and roi_bgr is not None:
        kind = background_kind(roi_bgr)

    if kind == KIND_COLOR:
        prompt = prompts.colour
    elif kind == KIND_GRAYSCALE:
        prompt = prompts.halftone
    else:
        prompt = prompts.paper
    if mask_kind == MASK_OTHER_REMOVAL and prompts.other_suffix:
        prompt = f"{prompt}, {prompts.other_suffix}"
    return prompt, prompts.negative


def prompt_chooser(
    markup: PageMarkup, mask_kind: str, prompts: "PromptSet | None" = None, min_frac: float = RASTER_INSIDE_FRAC
):
    """Колбэк ``(рамка зоны, ROI) -> (промпт, негатив)`` для ``inpainting.backends.SdFiller``.

    Заливщик восстанавливает рамку зоны из её маски внутри ROI и передаёт сюда её, а
    не сам ROI (тот вдвое больше и у приграничной зоны всегда вылезает на бумагу) —
    см. ``backends.zone_box``.

    ROI приходит в RGB, потому что в нём работают сети; измерители фона писались для
    BGR, как весь остальной пиксельный код, — отсюда перевод.
    """

    def choose(box: "tuple[int, int, int, int]", roi_rgb=None) -> "tuple[str, str]":
        roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR) if roi_rgb is not None else None
        return prompt_for(box, markup, mask_kind, prompts, min_frac, roi_bgr)

    return choose
