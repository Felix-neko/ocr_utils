"""Tesseract OSD — штатная детекция ориентации и письменности (``--psm 0``).

Внутри у неё голосование классификатора форм по связным компонентам на всех четырёх
поворотах; это тот самый «канонический» метод, вокруг которого построены все остальные.
Даёт сразу все четыре класса, а не одну ось.

ДВА МАСШТАБА. На разведке по четырём известным боковым полосам 300 dpi узнал три из них,
150 dpi — три ДРУГИЕ, объединение — все четыре. Берётся ответ с большей уверенностью.

ЧЕМ РАЗДЕЛЯЕТ. Не ответом, а уверенностью: на обычных полосах пака-1 OSD говорит
``Rotate: 0`` с уверенностью 27-37 и скриптом Cyrillic, на боковых — 0.7-15 и скриптом
вроде Bengali или Katakana. Поэтому определённый скрипт кладётся в метрики: чужая
письменность на журнале «Материально-техническое снабжение» сама по себе повод не верить.

PGM, а не PNG. Временный файл пишется голым P5: кодировать нечего, libpng не ругается на
профиль ICC, и на 12 тысячах полос это заметная экономия против двух кодирований PNG.
"""

from __future__ import annotations

import numpy as np

from ocr_utils.scan_markup import tesseract
from ocr_utils.page_layout.orientation.detectors.base import Detector, Frame, Verdict, unknown
from ocr_utils.scan_markup.tesseract import TIMEOUT_S, tesseract_available, write_pgm

# Уверенность OSD, начиная с которой считаем ответ полностью надёжным. Замер по обычным
# полосам пака-1: 27-37. Всё, что ниже, приводится пропорционально.
FULL_CONFIDENCE = 30.0

# Имена оставлены для прежних потребителей (``rotated_text.tables.ocr``, legacy-исследования):
# сам вызов tesseract живёт в ``scan_markup.tesseract``.
_write_pgm = write_pgm
_ = TIMEOUT_S


def run_osd(gray: np.ndarray) -> dict[str, str]:
    """Разбор вывода ``tesseract --psm 0``. Пустой словарь, если сказать нечего."""
    out = tesseract.run(gray, ["--psm", "0", "-l", "osd"], prefix="osd_")
    parsed: dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def _verdict_from(parsed: dict[str, str]) -> Verdict | None:
    if "Rotate" not in parsed:
        return None
    try:
        rotate = int(float(parsed["Rotate"])) % 360
        confidence = float(parsed.get("Orientation confidence", 0.0))
    except ValueError:
        return None
    if rotate % 90:
        return None
    metrics = {"osd_confidence": confidence}
    try:
        metrics["script_confidence"] = float(parsed.get("Script confidence", 0.0))
    except ValueError:
        pass
    script = parsed.get("Script", "")
    note = "" if script in ("", "Cyrillic") else f"скрипт {script}"
    # Ключевое соглашение: поле Rotate у tesseract — это уже «на сколько повернуть ПО
    # часовой, чтобы стало прямо», то есть ровно наша валюта. Проверяется командой validate.
    return Verdict(rotate, min(1.0, confidence / FULL_CONFIDENCE), metrics=metrics, note=note)


def detect(frame: Frame) -> Verdict:
    best: Verdict | None = None
    for name, gray in (("300", frame.gray300), ("150", frame.gray150)):
        verdict = _verdict_from(run_osd(gray))
        if verdict is None:
            continue
        verdict = Verdict(
            verdict.rotate_cw, verdict.confidence, metrics={**verdict.metrics, "dpi": float(name)}, note=verdict.note
        )
        if best is None or verdict.confidence > best.confidence:
            best = verdict
    return best if best is not None else unknown("tesseract не определил ориентацию")


ALGORITHM = Detector(
    name="osd",
    summary="tesseract --psm 0 -l osd на 300 и 150 dpi, ответ с большей уверенностью",
    stage="cpu",
    run=detect,
    available=tesseract_available,
)
