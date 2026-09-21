"""Пороги v15: те же правила гистерезиса, что в ядре, плюс условия применимости отдельных метрик.

Отличия от ``ocr_utils.geometry_regression.scoring``:

* новые метрики порчи — перекос по полю (``field_shear_p90_deg``, подтверждается кромкой
  строк ``edge_shear_delta_mm`` ≥ ``SHEAR_CONFIRM_MM`` (0.6 мм: на 28 страницах «равнение» кромка по
  тем же строкам даёт 0.65–5 мм, шир относительно строк занижает уход, когда строки сами
  наклонены), а без измеренных кромок — только при
  запасе ``SHEAR_ALONE_FACTOR``), сдвиг граф таблицы (``vstroke_tilt_wmean_delta``, при ≥ 2
  вертикалях), наклон фото по снимку целиком (``raster_photo_tilt_mm``, не непрощаемый);
* одиночный штрих (единственная пара своей ориентации — подчёркивание рубрики 1968/04 с.90)
  и равномерный уход всех линеек ориентации на один угол (графы таблицы сдвинуты целиком —
  1975/08 с.39) — мягкая порча: bad только без выигрыша (решение пользователя 2026-09-22:
  «баш на баш», mixed);
* новые выигрыши — доворот страницы, подтверждённый текстом (``page_deskew_gain_deg``), и
  выигрыш по строкам внутри таблиц (те же ``text_*_gain``, взятые максимумом);
* пороги выигрыша подняты до «заметно глазом» (прогиб 0.4 мм, разброс 0.8°, наклон отдельной
  строки 1.0°): на страницах «равнение перекосило» выигрыш 0.3 мм / 0.5° пользователь не
  видит (1968/03 с.8); стартовые числа — по p95 пака, калибруются в ``report --thr``.
"""

from __future__ import annotations

from ocr_utils.geometry_regression.scoring import DEFAULT_HARD, DEFAULT_MIN_GAIN, DEFAULT_RATIO, Verdict

# имя метрики → (порог, короткая причина для имени файла)
DEFAULT_THRESHOLDS: dict[str, tuple[float, str]] = {
    "vstroke_dev_max_delta_mm": (0.7, "vtilt"),
    "hstroke_dev_max_delta_mm": (0.7, "htilt"),
    "hstroke_tilt_wmean_delta": (0.5, "hmean"),
    # градусы; прирост среднего |наклона| граф таблицы (1967/07 с.89 — 0.34, 1969/05 с.29 — 0.45;
    # по паку при ≥ 2 вертикалях p95 0.38, p98 0.51) — калибровать по поясам.
    "vstroke_tilt_wmean_delta": (0.35, "vmean"),
    "parallel_spread_delta_max": (1.5, "parallel"),
    "stroke_bend_dev_mm": (0.8, "bend"),
    "field_lineart_weak_frac": (0.45, "lineart"),
    "raster_edge_bend_mm": (0.8, "photo_bend"),
    "raster_photo_tilt_mm": (1.5, "photo_tilt"),
    "line_tilt_dev_max_mm": (0.9, "line"),
    "line_glyph_wobble_max": (0.05, "wobble"),
    "line_stretch_mm_max": (0.2, "stretch"),
    # градусы; p90 |сдвига| по тайлам текста: 28 страниц «равнение» — медиана 0.96, случайные
    # ok — p90 0.67, p95 0.74; калибровать по поясам.
    "field_shear_p90_deg": (0.75, "shear"),
    # мм; уход кромки блока по тем же строкам сверх наклона строк — самостоятельный флаг
    # только при большом уходе, иначе подтверждение перекоса по полю.
    "edge_shear_delta_mm": (2.5, "edge"),
}

UNFORGIVABLE = ("hstroke_tilt_wmean_delta", "field_lineart_weak_frac", "raster_edge_bend_mm")
UNFORGIVABLE_MIN_HSTROKE_PAIRS = 2
# Средние наклоны считаются от стольких пар своей ориентации.
MIN_PAIRS = {"hstroke_tilt_wmean_delta": ("hstroke_pairs", 2), "vstroke_tilt_wmean_delta": ("vstroke_pairs", 2)}
# Уход одного-единственного штриха своей ориентации — мягкая порча.
LONE_PAIRS = {"hstroke_dev_max_delta_mm": "hstroke_pairs", "vstroke_dev_max_delta_mm": "vstroke_pairs"}
# Уход худшей линейки — мягкая порча и тогда, когда все линейки ориентации ушли на один угол
# (равномерность ≥ порога: графы таблицы сдвинуты целиком при выправленных строках — 1975/08
# с.39, 1973/08 с.18: «баш на баш»); сам средний сдвиг граф — всегда мягкий.
UNIFORM = {"hstroke_dev_max_delta_mm": "hstroke_uniform", "vstroke_dev_max_delta_mm": "vstroke_uniform"}
MIN_UNIFORM = 0.7
SOFT = ("vstroke_tilt_wmean_delta",)
# Перекос по полю подтверждается кромкой строк не меньше этого (мм); без кромок — запасом порога.
SHEAR_CONFIRM_MM = 0.6
SHEAR_ALONE_FACTOR = 1.3

GAIN_THRESHOLDS: dict[str, tuple[float, str]] = {
    "text_sag_gain_mm": (0.4, "sag"),
    "text_spread_gain_deg": (0.8, "spread"),
    "line_tilt_gain_mm": (0.9, "line"),
    "line_tilt_gain_deg": (1.0, "line_deg"),
    "hstroke_gain_mm": (0.8, "htilt"),
    "vstroke_gain_mm": (0.7, "vtilt"),
    "edge_shear_gain_mm": (2.4, "edge"),
    "page_deskew_gain_deg": (1.0, "deskew"),
}


class Thresholds15:
    """Пороги порчи и выигрыша v15 с перекрытиями ``имя=число``; интерфейс как у ``Thresholds`` ядра."""

    def __init__(
        self,
        overrides: dict[str, float] | None = None,
        hard: float = DEFAULT_HARD,
        min_gain: float = DEFAULT_MIN_GAIN,
        ratio: float = DEFAULT_RATIO,
    ):
        """Args:
        overrides: перекрытия ``имя → значение`` (порчи или выигрыша).
        hard: порча не ниже — ``bad`` независимо от выигрыша.
        min_gain: выигрыш ниже — порча не прощается.
        ratio: порча не ниже этой доли выигрыша — ``bad``.
        """
        self.values = {name: value for name, (value, _) in DEFAULT_THRESHOLDS.items()}
        self.gains = {name: value for name, (value, _) in GAIN_THRESHOLDS.items()}
        self.reasons = {name: reason for name, (_, reason) in DEFAULT_THRESHOLDS.items()}
        self.unforgivable = UNFORGIVABLE
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
    ) -> "Thresholds15":
        """Пороги из строк ``имя=число`` командной строки плюс параметры гистерезиса."""
        overrides = {}
        for item in items:
            name, _, value = item.partition("=")
            if not value:
                raise ValueError(f"ожидалось имя=число, получено {item!r}")
            overrides[name.strip()] = float(value)
        return cls(overrides, hard, min_gain, ratio)

    def _applicable(self, name: str, metrics: dict[str, float]) -> bool:
        """Есть ли у метрики основание считаться на этой странице."""
        if name in MIN_PAIRS:
            key, minimum = MIN_PAIRS[name]
            return float(metrics.get(key, 0.0) or 0.0) >= minimum
        if name == "field_shear_p90_deg":
            edges = float(metrics.get("edges_matched", 0.0) or 0.0)
            confirmed = float(metrics.get("edge_shear_delta_mm", 0.0) or 0.0) >= SHEAR_CONFIRM_MM
            alone = float(metrics.get(name, 0.0) or 0.0) >= SHEAR_ALONE_FACTOR * self.values[name]
            return confirmed if edges > 0 else alone
        return True

    def _soft(self, name: str, metrics: dict[str, float]) -> bool:
        """Мягкая порча: единственный штрих своей ориентации и метрики из ``SOFT``."""
        if name in SOFT:
            return True
        if name in LONE_PAIRS and float(metrics.get(LONE_PAIRS[name], 0.0) or 0.0) <= 1.0:
            return True
        if name in UNIFORM and float(metrics.get(UNIFORM[name], 0.0) or 0.0) >= MIN_UNIFORM:
            return True
        return False

    def apply(self, metrics: dict[str, float]) -> Verdict:
        best, reason, flags, unforgivable, hard = 0.0, "", {}, 0.0, 0.0
        few_hstrokes = float(metrics.get("hstroke_pairs", 0.0) or 0.0) < UNFORGIVABLE_MIN_HSTROKE_PAIRS
        for name, threshold in self.values.items():
            if not self._applicable(name, metrics):
                continue
            value = float(metrics.get(name, 0.0) or 0.0)
            score = value / threshold if threshold > 0 else 0.0
            if score >= 1.0:
                flags[name] = score
                if name in self.unforgivable and not (name.startswith("hstroke") and few_hstrokes):
                    unforgivable = max(unforgivable, score)
                if not self._soft(name, metrics):
                    hard = max(hard, score)
            if score > best:
                best, reason = score, self.reasons[name]
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
            f"hard = {self.hard:g}, min_gain = {self.min_gain:g}, ratio = {self.ratio:g}, "
            f"непрощаемые: {', '.join(self.unforgivable)}; одиночный штрих — мягкая порча; "
            f"перекос по полю подтверждается кромкой ≥ {SHEAR_CONFIRM_MM:g} мм"
        )
        return "\n".join(lines)


__all__ = ["DEFAULT_THRESHOLDS", "GAIN_THRESHOLDS", "UNFORGIVABLE", "Thresholds15"]
