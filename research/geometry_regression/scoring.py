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
    # Калибровка по паку-1 (12 135 страниц): порог ≈ p98–p99 распределения Δ, проверено по
    # поясам глазами; эталон — 7 из 8 плохих, 0 ложных на 4 хороших.
    "vstroke_dev_max_delta_mm": (0.7, "vtilt"),  # мм; p98.7; 1967/01 с.38: линейка 50 мм 0.3° → 1.2° = +1.0 мм
    "hstroke_dev_max_delta_mm": (
        0.7,
        "htilt",
    ),  # мм; ~p98.5; 1967/01 с.85: дробная черта 13 мм 0.6° → 6.3° = +1.4 мм; 1966/06 с.58 — 0.72
    # градусы; средний по длине |наклон| горизонтальных штрихов страницы: короткие дробные черты
    # формул (5-8 мм) в мм ухода дают мало, а все вместе наклон показывают надёжно (1966/06 с.58:
    # 0.18° → 1.86°); порог перекалибровать по паку.
    "hstroke_tilt_wmean_delta": (0.5, "hmean"),
    "parallel_spread_delta_max": (1.5, "parallel"),
    # мм; сагитта краски линии в A минус в B: ровные линейки 0.1–0.3, погнутые рамки 1966/01
    # с.95 — 1.7, с.78 — 3.5, волна линейки таблицы 1970/06 с.37 — 0.9.
    "stroke_bend_dev_mm": (0.8, "bend"),
    "field_lineart_weak_frac": (
        0.45,
        "lineart",
    ),  # доля; 1967/01 с.80: 59 % тайлов схемы без пары; 1968/02 с.69 (правка законна) — 33 %
    "line_tilt_dev_max_mm": (0.9, "line"),  # мм; заголовок 100 мм под 0.8° (1967/01 с.35) = 1.4 мм; по глифам
    "line_glyph_wobble_max": (0.05, "wobble"),  # доли высоты; волна A − волна B по глифам; перекалибровать по паку
    "edge_dev_max_delta_mm": (2.4, "edge"),  # мм; p99; кромка 120 мм под 1.2° = 2.5 мм
    # мм; клин высоты крупной строки сверх доворота и выпрямления; 1967/03 с.36 — 0.22 мм
    # (пользователь считает порчей), 1968/02 с.92 — 0.25 мм (пользователь порчей не считает):
    # граница различимости, порог на стороне «ловить».
    "line_stretch_mm_max": (0.2, "stretch"),
}

# Порча, которую не искупает никакой выигрыш («так нам точно не надо»): наклонённые черты формул
# (СРЕДНИЙ наклон горизонтальных штрихов при ≥ 2 чертах — все черты разом) и погнутый line art.
# Уход одной черты — обычная порча: на бланке с десятками линеек (1974/02 с.95) одна ушла на
# 0.8 мм при выправленных строках, а одиночное подчёркивание рубрики (1967/01 с.71) — тем более.
# Было «≥ 3», пока LSD давал по два отрезка на кромки одной линии (v9 свёл их в один): те же
# два бара формул (1968/01 с.29: 27 и 9 мм, второй ушёл на 3.9°) раньше считались за четыре.
UNFORGIVABLE = ("hstroke_tilt_wmean_delta", "field_lineart_weak_frac")
UNFORGIVABLE_MIN_HSTROKE_PAIRS = 2

# Порча, которая сама по себе не бывает «грубой»: кромка колонки — метрика шумная (абзацные
# отступы, висячие строки), и 1968/05 с.55 давала 6 мм при выправленных строках; bad только
# при отсутствии выигрыша.
SOFT = ("edge_dev_max_delta_mm",)

# Выигрыш: что коррекция выправила, в тех же единицах. Пороги — «заметно глазом».
GAIN_THRESHOLDS: dict[str, tuple[float, str]] = {
    "text_sag_gain_mm": (0.3, "sag"),  # мм; прогиб строк p90 × высота: 1966/06 с.62 — 0.13 × 3 мм ≈ 0.4
    "text_spread_gain_deg": (0.5, "spread"),  # градусы; разброс наклонов строк: с.62 1.21° → 0.25°
    "line_tilt_gain_mm": (0.9, "line"),  # мм; заголовок выровнялся
    "line_tilt_gain_deg": (0.5, "line_deg"),  # градусы; короткая рубрика выровнялась (1968/02 с.92)
    "hstroke_gain_mm": (0.8, "htilt"),  # мм; линейка легла на ось
    "vstroke_gain_mm": (0.7, "vtilt"),
    "edge_gain_mm": (2.4, "edge"),
}

# Гистерезис (решение пользователя: «если ухудшение минимальное, а выигрыш большой — хрен с
# ним, берём версию FineReader»): порча против выигрыша сравнивается ОТНОСИТЕЛЬНО — `bad`,
# когда порча не меньше ``DEFAULT_RATIO`` выигрыша (1967/01 с.38: 1.41 против 1.41 — bad;
# 1966/06 с.62: 1.06 против 2.23 — mixed; 1970/04 с.26: линейка врезки 3.6 против убранной
# трапеции 8.4 — mixed), либо когда выигрыша нет вовсе (< ``DEFAULT_MIN_GAIN``). Абсолютный
# порог ``DEFAULT_HARD`` остаётся на совсем грубые случаи.
DEFAULT_HARD = 5.0
DEFAULT_MIN_GAIN = 1.0
DEFAULT_RATIO = 0.75


@dataclass(frozen=True)
class Verdict:
    score: float  # порча: max(Δ/порог)
    reason: str
    flags: dict[str, float]  # метрика порчи → её score среди тех, что ≥ 1
    gain: float = 0.0  # выигрыш: max(Δ⁻/порог)
    gain_reason: str = ""
    verdict: str = "ok"  # "bad" | "mixed" | "ok"

    @property
    def flag(self) -> bool:
        return self.verdict == "bad"


class Thresholds:
    """Пороги порчи и выигрыша с перекрытиями вида ``имя=число`` из командной строки."""

    def __init__(
        self,
        overrides: dict[str, float] | None = None,
        hard: float = DEFAULT_HARD,
        min_gain: float = DEFAULT_MIN_GAIN,
        ratio: float = DEFAULT_RATIO,
    ):
        """Пороги порчи и выигрыша.

        Args:
            overrides: перекрытия порогов ``имя → значение`` (порчи или выигрыша).
            hard: порча не ниже — ``bad`` независимо от выигрыша.
            min_gain: выигрыш ниже — порча не прощается.
            ratio: порча не ниже этой доли выигрыша — ``bad``.
        """
        self.values = {name: value for name, (value, _) in DEFAULT_THRESHOLDS.items()}
        self.gains = {name: value for name, (value, _) in GAIN_THRESHOLDS.items()}
        self.hard, self.min_gain, self.ratio = hard, min_gain, ratio
        for name, value in (overrides or {}).items():
            if name in self.values:
                self.values[name] = float(value)
            elif name in self.gains:
                self.gains[name] = float(value)
            else:
                raise KeyError(f"неизвестная метрика {name!r}; есть: {', '.join([*self.values, *self.gains])}")

    @classmethod
    def parse(
        cls,
        items: tuple[str, ...],
        hard: float = DEFAULT_HARD,
        min_gain: float = DEFAULT_MIN_GAIN,
        ratio: float = DEFAULT_RATIO,
    ) -> "Thresholds":
        """Пороги из строк вида ``имя=число`` командной строки плюс параметры гистерезиса."""
        overrides = {}
        for item in items:
            name, _, value = item.partition("=")
            if not value:
                raise ValueError(f"ожидалось имя=число, получено {item!r}")
            overrides[name.strip()] = float(value)
        return cls(overrides, hard, min_gain, ratio)

    def apply(self, metrics: dict[str, float]) -> Verdict:
        best, reason, flags, unforgivable, hard = 0.0, "", {}, 0.0, 0.0
        few_hstrokes = float(metrics.get("hstroke_pairs", 0.0) or 0.0) < UNFORGIVABLE_MIN_HSTROKE_PAIRS
        for name, threshold in self.values.items():
            if name == "hstroke_tilt_wmean_delta" and few_hstrokes:
                continue  # средний наклон по одной-двум чертам в градусах — не мера порчи
            value = float(metrics.get(name, 0.0) or 0.0)
            score = value / threshold if threshold > 0 else 0.0
            if score >= 1.0:
                flags[name] = score
                if name in UNFORGIVABLE and not (name.startswith("hstroke") and few_hstrokes):
                    unforgivable = max(unforgivable, score)
                if name not in SOFT:
                    hard = max(hard, score)
            if score > best:
                best, reason = score, DEFAULT_THRESHOLDS[name][1]
        gain, gain_reason = 0.0, ""
        for name, threshold in self.gains.items():
            value = float(metrics.get(name, 0.0) or 0.0)
            score = value / threshold if threshold > 0 else 0.0
            if score > gain:
                gain, gain_reason = score, GAIN_THRESHOLDS[name][1]
        if best < 1.0:
            verdict = "ok"
        elif hard >= self.hard or unforgivable >= 1.0 or gain < self.min_gain or hard >= self.ratio * gain:
            verdict = "bad"
        else:
            verdict = "mixed"
        return Verdict(best, reason, flags, gain, gain_reason, verdict)

    def describe(self) -> str:
        lines = [f"{name} = {value:g}" for name, value in self.values.items()]
        lines += [f"выигрыш {name} = {value:g}" for name, value in self.gains.items()]
        lines.append(
            f"hard = {self.hard:g}, min_gain = {self.min_gain:g}, ratio = {self.ratio:g}, непрощаемые: {', '.join(UNFORGIVABLE)}"
        )
        return "\n".join(lines)
