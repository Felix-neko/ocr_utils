"""Пороги v15: те же правила гистерезиса, что в ядре, плюс условия применимости отдельных метрик.

Отличия от ``ocr_utils.geometry_regression.scoring``:

* новые метрики порчи — перекос выключенного блока (``edge_shear_delta_mm``: уход кромки по
  тем же строкам, считается при сдвиге по полю ``field_shear_p90_deg`` ≥ ``SHEAR_MIN_DEG``;
  без сдвига по полю уход кромки — шум сегментации, без выключенных блоков перекос не
  ставится), сдвиг граф таблицы (``vstroke_tilt_wmean_delta``, при ≥ 2
  вертикалях), наклон фото по снимку целиком (``raster_photo_tilt_mm``, не непрощаемый);
* одиночный штрих (единственная пара своей ориентации — подчёркивание рубрики 1968/04 с.90)
  и равномерный уход всех линеек ориентации на один угол (графы таблицы сдвинуты целиком —
  1975/08 с.39) — мягкая порча: bad только без выигрыша (решение пользователя 2026-09-22:
  «баш на баш», mixed);
* новые выигрыши — доворот страницы, подтверждённый текстом (``page_deskew_gain_deg``), и
  выигрыш по строкам внутри таблиц (те же ``text_*_gain``, взятые максимумом);
* порча рисунков (угол между семействами линий, поворот линии сверх поворота рисунка, разброс
  поворотов, изгиб) — непрощаемая; дробные черты — своё семейство, обычная порча; форма строки —
  относительные меры (кривизна к длине, ступенька/клин/растяжение к высоте букв);
* пороги выигрыша подняты до «заметно глазом» (прогиб 0.4 мм, разброс 0.65°, наклон отдельной
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
    # градусы; дробные черты формул (≥ 2 черты, по следу краски, без веса по длине): средний прирост
    # |наклона| и худшая черта. 1973/08 с.32 — 1.6 / 2.2, с.35 — 2.3 / 2.9, 1974/10 с.52 — 0.6 / 1.7,
    # 1975/06 с.50 — 2.6 / 3.0; калибровать по паку.
    "fraction_tilt_mean_delta_deg": (0.6, "fraction"),
    "fraction_tilt_max_delta_deg": (1.5, "fraction_max"),
    "parallel_spread_delta_max": (1.5, "parallel"),
    "stroke_bend_dev_mm": (0.8, "bend"),
    # мм; излом длинной линии — кусок сдвинут вбок от продолжения прямой без краски между
    # (1974/10 с.30 — 3.9 мм); в B у той же линии излома нет.
    "stroke_jog_dev_mm": (1.5, "jog"),
    "field_lineart_weak_frac": (0.45, "lineart"),
    # Порча рисунков (``lineart.py``), непрощаемая — решение пользователя 2026-09-22: поворот рисунка
    # целиком прощается, пропорции нет. Угол между семействами линий (график 1974/12 с.73 — 1.8°,
    # шкаф 1968/12 с.74 — 1.8°, самосвал 1975/12 с.73 — 1.8°, эталонный шкаф 1966/01 с.78 — 4.3°,
    # стеллаж 1973/03 с.59 — 0.8; законная правка 1968/02 с.69 — 0, выровненная блок-схема 1972/11
    # с.27 — 0.68), поворот линии сверх поворота рисунка (шкаф — 5.4°, самосвал 8.8°, эталонный шкаф
    # 6.4; блок-схема 1972/11 — 2.4, не порча; 1968/02 с.69 — 0.35), рост разброса углов в семействе
    # A − B, изгиб линии от 25 мм.
    "lineart_axis_delta_deg": (0.7, "la_axis"),
    "lineart_rot_max_deg": (3.0, "la_rot"),
    "lineart_spread_delta_deg": (1.0, "la_spread"),
    "lineart_bend_mm": (0.8, "la_bend"),
    "raster_edge_bend_mm": (0.8, "photo_bend"),
    "raster_photo_tilt_mm": (1.5, "photo_tilt"),
    "line_tilt_dev_max_mm": (0.9, "line"),
    # Форма строки относительными мерами (``lines.py``; решение пользователя 2026-09-22): по семи
    # страницам порчи и двум не-порчи. Кривизна к длине: порча +2.6…+7.7·10⁻³, не порча −3.9 и −3.6;
    # ступенька к высоте: порча 0.03–0.08, не порча < 0; клин и неравномерность к высоте: порча
    # 0.05–0.12 (при выпрямлении режутся до 0.04 — 1968/02 с.92). Только заголовки; калибровать по паку.
    "line_bend_ratio": (3.0e-3, "bend_ratio"),
    "line_step_ratio": (0.04, "step"),
    "line_wedge_ratio": (0.05, "wedge"),
    "line_stretch_ratio": (0.05, "stretch"),
    # мм; перекос выключенного блока: уход кромки по краске рядов сверх наклона строк, при
    # подтверждении сдвигом по полю (``field_shear_p90_deg`` ≥ SHEAR_MIN_DEG). На 28 страницах
    # «равнение» уход 0.65–5 мм; по поясам пака глазами (2026-09-22): ≥ 1.2 мм виден сразу,
    # 0.8–1.2 — при внимательном взгляде, 0.6–0.8 — на грани (при 0.6 мм — 629 bad по паку,
    # при 0.8 — 572). Балл — от кромки в мм, чтобы против выигрыша (в мм и градусах) стоял
    # уход, а не угол: снятая трапеция 1975/05 с.61 даёт сдвиг по полю 2.3° при уходе 1.5 мм.
    "edge_shear_delta_mm": (0.8, "shear"),
}
# Балл перекоса: от порога до SHEAR_SCALE_MM балл равен 1 (bad — только при выигрыше ниже 1.33),
# дальше растёт как уход/SHEAR_SCALE_MM: страницы с законной большой правкой (1967/10 с.30 — уход
# 3.8 мм при выпрямленных строках, выигрыш 4.5) остаются mixed, а перекос «в обмен на ничего»
# (1967/07 с.17 — 4.4 мм при выигрыше 1.4) — bad.
SHEAR_SCALE_MM = 1.2

UNFORGIVABLE = (
    "hstroke_tilt_wmean_delta",
    "field_lineart_weak_frac",
    "lineart_axis_delta_deg",
    "lineart_rot_max_deg",
    "lineart_spread_delta_deg",
    "lineart_bend_mm",
    "raster_edge_bend_mm",
)
UNFORGIVABLE_MIN_HSTROKE_PAIRS = 2
# Средние наклоны считаются от стольких пар своей ориентации.
MIN_PAIRS = {
    "hstroke_tilt_wmean_delta": ("hstroke_pairs", 2),
    "vstroke_tilt_wmean_delta": ("vstroke_pairs", 2),
    "fraction_tilt_mean_delta_deg": ("fraction_bars", 2),
    "fraction_tilt_max_delta_deg": ("fraction_bars", 2),
}
# Уход одного-единственного штриха своей ориентации — мягкая порча.
LONE_PAIRS = {"hstroke_dev_max_delta_mm": "hstroke_pairs", "vstroke_dev_max_delta_mm": "vstroke_pairs"}
# Уход худшей линейки — мягкая порча и тогда, когда все линейки ориентации ушли на один угол
# (равномерность ≥ порога: графы таблицы сдвинуты целиком при выправленных строках — 1975/08
# с.39, 1973/08 с.18: «баш на баш»); сам средний сдвиг граф — всегда мягкий.
UNIFORM = {"hstroke_dev_max_delta_mm": "hstroke_uniform", "vstroke_dev_max_delta_mm": "vstroke_uniform"}
MIN_UNIFORM = 0.7
SOFT = ("vstroke_tilt_wmean_delta",)
# Уход кромки считается перекосом только при сдвиге по полю не меньше SHEAR_MIN_DEG (градусы), а
# при сильном уходе кромки (от SHEAR_STRONG_MM) — от SHEAR_MIN_WEAK_DEG: на 28 страницах
# «равнение» p90 сдвига по тайлам 0.5–1.8, и у трёх страниц с уходом кромки 1.8–2.4 мм он лишь
# 0.52–0.54 (1968/12 с.27, 1969/02 с.51, 1969/10 с.49). Без сдвига по полю уход кромки — шум
# сегментации строк; без измеренных кромок (нет выключенных блоков) перекос не ставится.
SHEAR_MIN_DEG = 0.75
SHEAR_MIN_WEAK_DEG = 0.5
SHEAR_STRONG_MM = 1.5

GAIN_THRESHOLDS: dict[str, tuple[float, str]] = {
    "text_sag_gain_mm": (0.4, "sag"),
    "text_spread_gain_deg": (
        0.65,
        "spread",
    ),  # эталонный good 1968/05 с.55: разброс 1.64° против линейки врезки 1.27 мм — mixed
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
        if name == "edge_shear_delta_mm":
            field = float(metrics.get("field_shear_p90_deg", 0.0) or 0.0)
            strong = float(metrics.get(name, 0.0) or 0.0) >= SHEAR_STRONG_MM
            return field >= SHEAR_MIN_DEG or (strong and field >= SHEAR_MIN_WEAK_DEG)
        return True

    def _score(self, name: str, value: float, threshold: float) -> float:
        """Балл метрики: отношение к порогу; у перекоса — пологая шкала выше порога."""
        if threshold <= 0:
            return 0.0
        if name == "edge_shear_delta_mm" and value >= threshold:
            return max(1.0, value / SHEAR_SCALE_MM)
        return value / threshold

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
            score = self._score(name, value, threshold)
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
            f"перекос кромки считается при сдвиге по полю ≥ {SHEAR_MIN_DEG:g}°"
        )
        return "\n".join(lines)


__all__ = ["DEFAULT_THRESHOLDS", "GAIN_THRESHOLDS", "UNFORGIVABLE", "Thresholds15"]
