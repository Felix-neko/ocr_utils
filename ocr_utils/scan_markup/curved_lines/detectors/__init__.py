"""Реестр детекторов кривизны строк.

Порядок в ``DETECTORS`` — порядок колонок в отчётах: сперва дешёвая классика на CPU,
потом нейросетевая сегментация на GPU. Добавить детектор — один файл с ``ALGORITHM``
и одна строка здесь.
"""

from __future__ import annotations

from ocr_utils.scan_markup.curved_lines.detectors import line_fit, skew_map
from ocr_utils.scan_markup.curved_lines.detectors.base import BatchDetector, Detector, Frame, GpuPage, Measure, silent

_MODULES = [skew_map, line_fit]
try:
    from ocr_utils.scan_markup.curved_lines.detectors import strip_shift

    _MODULES.append(strip_shift)
except ImportError:  # pragma: no cover — детектор ещё не написан
    pass
try:
    from ocr_utils.scan_markup.curved_lines.detectors import surya_lines

    _MODULES.append(surya_lines)
except ImportError:  # pragma: no cover
    pass
from ocr_utils.scan_markup.curved_lines.detectors import end_curl

_MODULES.append(end_curl)

DETECTORS: dict[str, Detector] = {module.ALGORITHM.name: module.ALGORITHM for module in _MODULES}
CHOICES = tuple(DETECTORS)

# Набор по умолчанию — всё, что есть: задача исследовательская, и каждый детектор
# сравнивается с остальными. Дорогой surya_lines исключается явно: ``--detectors``.
DEFAULT_SET = CHOICES


def resolve(names: tuple[str, ...]) -> list[Detector]:
    resolved = []
    for name in names:
        if name not in DETECTORS:
            raise KeyError(f"нет детектора {name!r}; есть: {', '.join(CHOICES)}")
        resolved.append(DETECTORS[name])
    return resolved


def registry_text() -> str:
    lines = []
    for detector in DETECTORS.values():
        marks = [detector.stage]
        if detector.sufficient:
            marks.append("достаточный")
        if not detector.available():
            marks.append("НЕДОСТУПЕН")
        thresholds = ", ".join(f"{k}≥{v:g}" for k, v in detector.thresholds.items())
        lines.append(f"  {detector.name:<12} [{', '.join(marks)}] {detector.summary}\n      флаг: {thresholds}")
    return "\n".join(lines)


__all__ = [
    "DETECTORS",
    "CHOICES",
    "DEFAULT_SET",
    "BatchDetector",
    "Detector",
    "Frame",
    "GpuPage",
    "Measure",
    "resolve",
    "registry_text",
    "silent",
]
