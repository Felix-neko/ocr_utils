"""Пороги на разности и сводный балл страницы.

Каждая флаговая метрика — разность «стало − было» в шкале «больше — хуже» (или величина,
которая в B по построению нулевая, как остаток поля внутри рисунка). ``score`` страницы —
максимум отношений метрика/порог, ``reason`` — метрика-победитель. Пороги в коде —
стартовые, из замера на эталонных страницах; рабочие подбираются по распределению на паке
и передаются в ``report --thr``.
"""

from __future__ import annotations

from dataclasses import dataclass

# имя метрики → (порог, короткая причина для имени файла)
DEFAULT_THRESHOLDS: dict[str, tuple[float, str]] = {
    "vstroke_dev_max_delta_mm": (0.5, "vtilt"),  # мм; 1967/01 с.38: линейка 50 мм, 0.3° → 1.2° = +0.8 мм
    "hstroke_dev_max_delta_mm": (0.5, "htilt"),  # мм; 1967/01 с.85: дробная черта 13 мм, 0.6° → 6.3° = +1.3 мм
    "parallel_spread_delta_max": (0.6, "parallel"),  # градусы; 1966/01 с.78: полки 0.7 → 2.1
    "field_lineart_weak_frac": (0.3, "lineart"),  # доля; 1967/01 с.80: 61 % тайлов схемы без пары
    "line_dev_max_delta_mm": (1.0, "line"),  # мм; заголовок 100 мм повернулся на 0.8° (1967/01 с.35) = 1.4 мм
    "line_wobble_delta_max": (0.055, "wobble"),  # доли высоты строки; 1967/03 с.11 — разрядка 0.072, шум p99 0.055
    "edge_dev_max_delta_mm": (1.5, "edge"),  # мм; 1967/02 с.57: кромка 120 мм, 0.6° → 1.3° = +1.5 мм
}


@dataclass(frozen=True)
class Verdict:
    score: float
    reason: str
    flags: dict[str, float]  # метрика → её score среди тех, что ≥ 1

    @property
    def flag(self) -> bool:
        return self.score >= 1.0


class Thresholds:
    """Пороги по умолчанию с перекрытиями вида ``имя=число`` из командной строки."""

    def __init__(self, overrides: dict[str, float] | None = None):
        self.values = {name: value for name, (value, _) in DEFAULT_THRESHOLDS.items()}
        for name, value in (overrides or {}).items():
            if name not in self.values:
                raise KeyError(f"неизвестная флаговая метрика {name!r}; есть: {', '.join(self.values)}")
            self.values[name] = float(value)

    @classmethod
    def parse(cls, items: tuple[str, ...]) -> "Thresholds":
        overrides = {}
        for item in items:
            name, _, value = item.partition("=")
            if not value:
                raise ValueError(f"ожидалось имя=число, получено {item!r}")
            overrides[name.strip()] = float(value)
        return cls(overrides)

    def apply(self, metrics: dict[str, float]) -> Verdict:
        best, reason, flags = 0.0, "", {}
        for name, threshold in self.values.items():
            value = float(metrics.get(name, 0.0) or 0.0)
            score = value / threshold if threshold > 0 else 0.0
            if score >= 1.0:
                flags[name] = score
            if score > best:
                best, reason = score, DEFAULT_THRESHOLDS[name][1]
        return Verdict(best, reason, flags)

    def describe(self) -> str:
        return "\n".join(f"{name} = {value:g}" for name, value in self.values.items())
