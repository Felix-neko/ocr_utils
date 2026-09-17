"""Пороги, флаги, score и сводный вердикт.

Единое правило для всех детекторов: у каждого есть флаговые метрики со своими порогами,
``score = max(metric / threshold)`` по ним, ``flag ⇔ score ≥ 1``. Метрики без порога в CSV
попадают, но на флаг не влияют. Молчащая мера (``silent``) не флагуется никогда.

Пороги живут здесь, а не в детекторах, потому что их меняют чаще, чем код: ``--thr
line_fit.sagitta_rel_p90=0.4`` перекрывает умолчание без правки и пересчёта.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from ocr_utils.scan_markup.curved_lines.detectors.base import Detector, Measure

# Имя сводного «детектора» в отчётах и каталогах симлинков.
COMBO = "combo"


class ThresholdError(ValueError):
    """Опечатка в ``--thr``: неизвестный детектор, метрика или не число."""


def parse_override(text: str) -> tuple[str, str, float]:
    """``детектор.метрика=значение`` → кортеж; метрика может содержать подчёркивания."""
    if "=" not in text or "." not in text.split("=", 1)[0]:
        raise ThresholdError(f"ожидалось ДЕТЕКТОР.МЕТРИКА=ЧИСЛО, получено {text!r}")
    key, value = text.split("=", 1)
    name, metric = key.split(".", 1)
    try:
        number = float(value)
    except ValueError as error:
        raise ThresholdError(f"порог {key!r}: не число {value!r}") from error
    return name.strip(), metric.strip(), number


@dataclass(frozen=True)
class Thresholds:
    values: dict[str, dict[str, float]]  # детектор → метрика → порог

    @classmethod
    def from_detectors(cls, detectors: Sequence[Detector], overrides: Sequence[str] = ()) -> "Thresholds":
        values = {detector.name: dict(detector.thresholds) for detector in detectors}
        for text in overrides:
            name, metric, number = parse_override(text)
            if name not in values:
                raise ThresholdError(f"порог {text!r}: детектор {name!r} не в наборе ({', '.join(values)})")
            if metric not in values[name]:
                raise ThresholdError(
                    f"порог {text!r}: у {name} нет флаговой метрики {metric!r} (есть: {', '.join(values[name])})"
                )
            if number <= 0.0:
                raise ThresholdError(f"порог {text!r}: должен быть положительным")
            values[name][metric] = number
        return cls(values)

    def apply(self, name: str, measure: Measure) -> Measure:
        """Проставляет score и флаг по порогам детектора ``name``."""
        if measure.silent:
            return replace(measure, flag=False, score=0.0)
        score = 0.0
        for metric, threshold in self.values.get(name, {}).items():
            value = measure.metrics.get(metric)
            if value is not None and threshold > 0.0:
                score = max(score, float(value) / threshold)
        return replace(measure, flag=score >= 1.0, score=score)

    def table(self) -> str:
        rows = ["| детектор | метрика | порог |", "|---|---|---|"]
        for name, metrics in self.values.items():
            for metric, threshold in metrics.items():
                rows.append(f"| {name} | {metric} | {threshold:g} |")
        return "\n".join(rows)


def combine(
    measures: dict[str, Measure],
    votes: int,
    strong: float,
    solo: Sequence[str] | None = None,
    sufficient: Sequence[str] = (),
) -> Measure:
    """Сводный вердикт: флаг, если проголосовало не меньше ``votes`` детекторов ИЛИ хоть один
    уверен сильнее ``strong`` (score ≥ strong).

    Два условия, а не одно, потому что детекторы ловят разное: локальный поворот блока
    видит только карта углов, а прогиб у корешка — только аппроксимации строк. Требовать
    согласия значило бы терять оба вида; брать любого — собирать все ложные срабатывания.
    Голоса — защита от одиночного выброса, «сильный» голос — от потери того, что видит
    один-единственный детектор. ``solo`` — имена детекторов, которым «сильный» одиночный
    голос разрешён (None — всем).
    """
    spoken = [measure for measure in measures.values() if not measure.silent]
    if not spoken:
        return Measure(note="все детекторы молчат", silent=True)
    flagged = sum(1 for measure in spoken if measure.flag)
    strong_scores = [
        measure.score for name, measure in measures.items() if not measure.silent and (solo is None or name in solo)
    ]
    max_score = max(measure.score for measure in spoken)
    max_solo = max(strong_scores) if strong_scores else 0.0
    mean_score = sum(measure.score for measure in spoken) / len(spoken)
    enough = any(name in sufficient and measure.flag for name, measure in measures.items())
    flag = flagged >= max(1, votes) or max_solo >= strong or enough
    return Measure(
        metrics={"votes": float(flagged), "max_score": max_score, "mean_score": mean_score}, flag=flag, score=max_score
    )
