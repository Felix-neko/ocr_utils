"""Судья dewarp ScanTailor: пара выводов страницы (dewarp выключен / авто) — сходство текста tesseract, прямизна строк, форма модели искажения — и вердикт, брать ли распрямление."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

import cv2
import numpy as np

from ocr_utils.scan_markup import tesseract
from research.page_cutting.scantailor import Distortion
from research.page_cutting.straightness import Straightness

# Порча текста: у «авто» слов меньше этой доли от «выкл» — dewarp съел или перекосил строки до нечитаемости.
# Сходство текстов «авто» и «выкл» не годится: на кривых строках «выкл» сам читается хуже (ХиП 1991/02 IMG_1168 R:
# сходство 0.69 при явно лучшем «авто»).
MIN_WORD_SHARE = 0.9
# Порча текста: средняя уверенность слов tesseract у «авто» ниже, чем у «выкл», больше чем на столько пунктов (0–100).
MAX_CONF_DROP = 5.0
# Прямизна «авто» не хуже «выкл» больше чем на столько (мм прогиба p90 + 0.5·разброс наклонов, °), иначе — «хуже».
MAX_SCORE_LOSS = 0.1
# Модель искажения с расхождением наклонов верхней и нижней кривых больше порога — сильная трапеция, °.
MAX_MODEL_SKEW_DEG = 3.0
# Выигрыш в непрямизне, при котором расхождение кривых модели не считается захватом лишнего.
MIN_SKEW_GAIN = 0.5
# Вес разброса наклонов в общей мере прямизны (мм прогиба + вес·°).
SPREAD_WEIGHT = 0.5


class Verdict(str, Enum):
    """Вердикт по распрямлению строк."""

    # Брать вывод с dewarp.
    DEWARP = "dewarp"
    # Брать вывод без dewarp: распрямление повредило текст.
    DAMAGED = "off_damaged"
    # Брать вывод без dewarp: строки не стали прямее.
    WORSE = "off_worse"
    # Брать вывод без dewarp: модель искажения — сильная трапеция (захват лишнего в контент).
    SKEWED = "off_skewed"
    # Dewarp не отработал (нет вывода) — без dewarp.
    FAILED = "off_failed"


def ocr_words(bgr: np.ndarray) -> tuple[list[str], float]:
    """Слова tesseract (русский, TSV) картинки страницы и их средняя уверенность.

    Args:
        bgr: Картинка страницы.

    Returns:
        ``(слова в нижнем регистре без знаков по краям, длиной от 2 букв; средняя уверенность 0–100 или 0)``.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    words, confs = [], []
    for line in tesseract.tsv_lines(gray, psm=3):
        parts = line.split("\t")
        if len(parts) < 12 or not parts[11].strip():
            continue
        w = parts[11].strip(".,;:!?«»()\"'—-").lower()
        if len(w) >= 2:
            words.append(w)
            confs.append(float(parts[10]))
    return words, float(np.mean(confs)) if confs else 0.0


def text_similarity(a: list[str], b: list[str]) -> float:
    """Сходство двух последовательностей слов (``SequenceMatcher``), 1 — совпадают; оба пусты — 1."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def model_skew_deg(model: Distortion | None) -> float:
    """Расхождение наклонов хорд верхней и нижней кривых модели искажения, ° (0 — параллельны или модели нет)."""
    if model is None or len(model.top) < 2 or len(model.bottom) < 2:
        return 0.0
    a = model.top[-1] - model.top[0]
    b = model.bottom[-1] - model.bottom[0]
    ang = np.degrees(np.arctan2(a[1], a[0]) - np.arctan2(b[1], b[0]))
    return float(abs((ang + 180) % 360 - 180))


def score(s: Straightness) -> float:
    """Общая мера непрямизны: прогиб p90 (мм) + ``SPREAD_WEIGHT`` · разброс наклонов (°); NaN — строк нет."""
    return s.sagitta_p90_mm + SPREAD_WEIGHT * s.slope_spread_deg


@dataclass(frozen=True)
class Judgement:
    """Решение судьи по странице.

    Поля:
        verdict: Вердикт.
        similarity: Сходство слов «авто» с «выкл».
        words_off: Слов в выводе без dewarp.
        words_auto: Слов в выводе с dewarp.
        conf_off: Средняя уверенность слов tesseract без dewarp.
        conf_auto: То же с dewarp.
        score_off: Мера непрямизны без dewarp.
        score_auto: Мера непрямизны с dewarp.
        model_skew_deg: Расхождение наклонов кривых модели искажения, °.
    """

    verdict: Verdict
    similarity: float
    words_off: int
    words_auto: int
    conf_off: float
    conf_auto: float
    score_off: float
    score_auto: float
    model_skew_deg: float


def judge(
    off: np.ndarray, auto: np.ndarray | None, s_off: Straightness, s_auto: Straightness | None, model: Distortion | None
) -> Judgement:
    """Вердикт по паре выводов страницы.

    Порядок проверок: нет вывода → ``FAILED``; модель — сильная трапеция → ``SKEWED``; текст разошёлся → ``DAMAGED``;
    строки не прямее → ``WORSE``; иначе ``DEWARP``.

    Args:
        off: Вывод без dewarp.
        auto: Вывод с dewarp или None.
        s_off: Прямизна строк вывода без dewarp.
        s_auto: Прямизна строк вывода с dewarp или None.
        model: Модель искажения из проекта прогона с dewarp.

    Returns:
        Решение.
    """
    words_off, conf_off = ocr_words(off)
    if auto is None or s_auto is None:
        return Judgement(Verdict.FAILED, 0.0, len(words_off), 0, conf_off, 0.0, score(s_off), float("nan"), 0.0)
    words_auto, conf_auto = ocr_words(auto)
    sim = text_similarity(words_off, words_auto)
    skew = model_skew_deg(model)
    sc_off, sc_auto = score(s_off), score(s_auto)
    gain = sc_off - sc_auto if not (np.isnan(sc_off) or np.isnan(sc_auto)) else float("nan")
    # Слов меньше — порча, только если и уверенность не выросла: на кривых строках «выкл» tesseract дробит слова
    # (ХиП 1991/02 IMG_1090 L: 70 → 61 слово при уверенности 82 → 89 и явно лучшем «авто»).
    fewer = len(words_auto) < MIN_WORD_SHARE * len(words_off) and conf_auto <= conf_off
    if fewer or conf_auto < conf_off - MAX_CONF_DROP:
        verdict = Verdict.DAMAGED
    elif not np.isnan(sc_off) and (np.isnan(sc_auto) or sc_auto > sc_off + MAX_SCORE_LOSS):
        verdict = Verdict.WORSE
    elif skew > MAX_MODEL_SKEW_DEG and not gain >= MIN_SKEW_GAIN:
        # Сильная трапеция модели без заметного выигрыша в прямизне — захват лишнего в контент. С выигрышем — нет:
        # на странице с остатком перспективы (РИСК 1991/03 IMG_0347 R) модель законно расходится на 9°.
        verdict = Verdict.SKEWED
    else:
        verdict = Verdict.DEWARP
    return Judgement(
        verdict,
        round(sim, 3),
        len(words_off),
        len(words_auto),
        round(conf_off, 1),
        round(conf_auto, 1),
        sc_off,
        sc_auto,
        round(skew, 2),
    )
