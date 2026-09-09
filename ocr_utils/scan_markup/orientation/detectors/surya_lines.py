"""Ось текста по строкам, найденным детектором Surya.

Нейросетевой аналог ``ink_axis``: вместо морфологии строки ищет обученная сеть, а ось
считается по форме найденных рамок — вытянутых поперёк страницы или вдоль неё. Ошибки
у него совсем другие: там, где морфология путается в частой штриховке чертежа, сеть
уверенно находит именно текст, а там, где текст набран вразрядку, сеть иногда рвёт строку.

Замер на восьми известных полосах: у боковых доля площади вертикальных строк 0.865-0.990,
у обычных доля горизонтальных 0.994-1.000. Разделяет чисто.

Строка остаётся строкой и вверх ногами, поэтому детектор принципиально ``axis_only``.

Цена — около 0.3 с на полосу на GPU, то есть примерно час на пак; отсюда возможность
исключить его из набора, не трогая остальных. Surya уже используется в проекте
(``background_smoothing/layout.py``), новых зависимостей не появляется.
"""

from __future__ import annotations

from typing import Sequence

from ocr_utils.scan_markup.orientation.detectors.base import Detector, Verdict, unknown

# Ниже этого перевеса одной оси над другой ось считается неразличимой.
AXIS_MARGIN_THR = 0.25

# Меньше этого числа строк — судить не по чему.
MIN_LINES = 3

DEFAULT_BATCH = 8

# Детектор строк один на весь процесс: его просят и surya_lines, и арбитр surya_vote,
# а держать в видеопамяти две копии одной и той же сети незачем.
_SHARED_DETECTION = None


def shared_detection_predictor():
    global _SHARED_DETECTION
    if _SHARED_DETECTION is None:
        from surya.detection import DetectionPredictor

        predictor = DetectionPredictor()
        predictor.disable_tqdm = True  # иначе на каждую пачку рвётся общий прогресс-бар
        _SHARED_DETECTION = predictor
    return _SHARED_DETECTION


def axis_score_of(result) -> tuple[float, int]:
    """Перевес горизонтальных строк над вертикальными и число строк."""
    wide = tall = 0.0
    for box in result.bboxes:
        x0, y0, x1, y1 = box.bbox
        width, height = max(1.0, x1 - x0), max(1.0, y1 - y0)
        if width >= height:
            wide += width * height
        else:
            tall += width * height
    if wide + tall <= 0.0:
        return 0.0, len(result.bboxes)
    return (wide - tall) / (wide + tall), len(result.bboxes)


def surya_available() -> bool:
    try:
        import surya.detection  # noqa: F401
    except Exception:
        return False
    return True


class SuryaAxis:
    def __init__(self, batch_size: int = DEFAULT_BATCH) -> None:
        self._batch_size = batch_size
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            self._predictor = shared_detection_predictor()
        return self._predictor

    def __call__(self, images: Sequence["object"]) -> list[Verdict]:
        if not images:
            return []
        predictor = self._load()
        verdicts: list[Verdict] = []
        for start in range(0, len(images), self._batch_size):
            chunk = list(images[start : start + self._batch_size])
            try:
                results = predictor(chunk)
            except Exception as error:
                verdicts.extend(unknown(f"surya: {error}") for _ in chunk)
                continue
            verdicts.extend(_verdict(result) for result in results)
        return verdicts


def _verdict(result) -> Verdict:
    axis_score, lines = axis_score_of(result)
    if lines < MIN_LINES or axis_score == 0.0:
        return unknown("строк не нашлось")
    metrics = {"axis_score": axis_score, "lines": float(lines)}
    if abs(axis_score) < AXIS_MARGIN_THR:
        return Verdict(0, 0.0, metrics=metrics, note="ось не различается")
    return Verdict(0 if axis_score > 0 else 90, min(1.0, abs(axis_score)), axis_only=True, metrics=metrics)


ALGORITHM = Detector(
    name="surya_lines",
    summary="ось по форме строк, найденных детектором Surya (GPU)",
    stage="gpu",
    make_batch=SuryaAxis,
    available=surya_available,
)
