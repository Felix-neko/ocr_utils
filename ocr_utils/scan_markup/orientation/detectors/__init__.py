"""Реестр детекторов ориентации.

Порядок в ``DETECTORS`` — порядок колонок в отчётах: сперва свои меры, потом готовые
движки, потом арбитр. Он же порядок предпочтения при сведении: см. ``analysis.combine``.

Здесь НЕТ детектора, который читал бы полосу через Surya на нескольких поворотах, хотя
напрашивается он первым делом. Причина — замер, см. README, раздел «Что пробовали и
отбросили»: распознавание Surya выправляет строку само и на перевёрнутой полосе выдаёт
осмысленный русский текст с уверенностью 1.00, поэтому сторону по его выводу не определить
никаким счётом. Ось Surya различает отлично — этим и занимается ``surya_lines``.
"""

from __future__ import annotations

from ocr_utils.scan_markup.orientation.detectors import (
    doctr_cls,
    ink_axis,
    ocr_vote,
    optional,
    osd,
    profile,
    surya_lines,
)
from ocr_utils.scan_markup.orientation.detectors.base import (
    ROTATION_NAMES,
    ROTATIONS,
    BatchDetector,
    Detector,
    Frame,
    Verdict,
    unknown,
)

DETECTORS: dict[str, Detector] = {
    algorithm.name: algorithm
    for algorithm in (
        ink_axis.ALGORITHM,
        profile.ALGORITHM,
        osd.ALGORITHM,
        doctr_cls.ALGORITHM,
        surya_lines.ALGORITHM,
        optional.ALGORITHM,
        # Арбитр в конце: по нему команда validate отбирает заведомо прямые полосы, и он же
        # называет сторону поворота там, где остальные умеют только ось.
        ocr_vote.ALGORITHM,
    )
}

CHOICES = tuple(DETECTORS)

# Набор по умолчанию: всё, что не требует ни установки, ни отдельного решения. Арбитр в него
# входит — он всё равно запускается только по кандидатам, а без него у боковых полос некому
# определить сторону.
#
# ``doctr`` в умолчание НЕ входит, хотя и доступен. Две причины, обе замерены:
#
# 1. Он течёт. На повторных прогонах ОДНОЙ И ТОЙ ЖЕ пачки из 32 полос память растёт на
#    10.7 МиБ на полосу и не возвращается ни ``gc.collect``, ни ``torch.cuda.empty_cache``,
#    ни удалением самого предиктора, ни ``malloc_trim`` (тот отдаёт 9%). На 12 135 полосах
#    это 130 ГиБ — прогон по паку так просто не доживает до конца.
# 2. Он же и самый слабый по точности: 0.805 на 600 синтетических поворотах против 1.000
#    у ``osd`` и ``ocr_vote``, причём проваливает именно 180° — в 70 случаях из 150 говорит
#    «прямо».
#
# Для коротких прогонов (``validate``, точечная проверка) он по-прежнему годится: там утечка
# ограничена длиной прогона. Включается явно: ``--detectors ...,doctr``.
# ``profile`` в умолчание НЕ входит. Замер по полному прогону, сверенный с ручной разметкой
# 122 кандидатов: он отметил 96 полос, из них верно 17 — точность 0.177, и это 79 из 80 всех
# ложных кандидатов пака. При этом ни одной находки, которой не дали бы другие: убрать его —
# по-прежнему 42 из 42, но работы арбитру втрое меньше (43 полосы вместо 122).
#
# Из реестра он не убран: как мера он другой природы, чем ink_axis (периодичность профиля
# против смыкания глифов), и расхождение между ними — повод посмотреть полосу глазами.
DEFAULT_SET = ("ink_axis", "osd", "surya_lines", "ocr_vote")


def resolve(names: "tuple[str, ...]") -> list[Detector]:
    """Детекторы по именам, с внятной ошибкой на опечатку."""
    resolved = []
    for name in names:
        if name not in DETECTORS:
            raise KeyError(f"нет детектора {name!r}; есть: {', '.join(CHOICES)}")
        resolved.append(DETECTORS[name])
    return resolved


def registry_text() -> str:
    """Список детекторов с пометкой доступности — для ``--detectors list``."""
    lines = []
    for detector in DETECTORS.values():
        marks = [detector.stage]
        if detector.arbiter:
            marks.append("арбитр")
        if not detector.available():
            marks.append("НЕДОСТУПЕН")
        lines.append(f"  {detector.name:<12} [{', '.join(marks)}] {detector.summary}")
    return "\n".join(lines)


__all__ = [
    "DETECTORS",
    "CHOICES",
    "DEFAULT_SET",
    "ROTATIONS",
    "ROTATION_NAMES",
    "BatchDetector",
    "Detector",
    "Frame",
    "Verdict",
    "resolve",
    "registry_text",
    "unknown",
]
