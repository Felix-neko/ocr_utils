"""Вердикт v17: группы по явлению, индивидуальные жёсткие пороги, откат по совокупности, гистерезис с выигрышем по роду порчи.

ПРАВИЛА (решения пользователя 2026-09-29), по порядку:

1. **ok** — ни одна метрика порчи не дошла до порога (score < 1) и совокупность не сработала.
2. **bad, непрощаемая** — непрощаемая метрика ≥ порога: наклон дробных черт, погнутый line art (в том числе
   разошедшиеся параллели внутри line art), погнутая кромка фото.
3. **bad, жёсткий порог** — любая метрика дошла до СВОЕГО жёсткого порога ``h_i`` (в единицах метрики).
4. **bad, совокупность** — сумма score групп ``Σ ≥ S`` и при этом сумма не меньше ``ratio`` × выигрыш (см. ниже,
   какой выигрыш). У группы score — максимум score её метрик; пола нет.
5. **bad, нет выигрыша** — порча есть (score ≥ 1), а выигрыш ниже ``min_gain``.
6. **bad, гистерезис** — жёсткая (не мягкая) порча ≥ ``ratio`` × выигрыш.
7. иначе **mixed** — порча явно меньше выигрыша: берём FineReader.

ГРУППЫ — по явлению, а не по детектору: один перекос страницы наклоняет и края колонок, и вертикальные линейки
(1970/04 с.26: край 4.7 мм и линейка врезки 2.6 мм — одна и та же порча, эталон good), поэтому они в одной
группе «вертикали»; строки и горизонтальные линейки — «горизонтали».

ВЫИГРЫШ ПО РОДУ ПОРЧИ. Выпрямленная одна строка (заголовок, ``line_quality_gain_mm``) прощает только порчу
текста (строки, края блоков); порчу линеек, line art, дробных черт и фото она не прощает (решение пользователя
2026-09-29, 1967/10 с.74: выпрямленный заголовок над перекорёженной оргсхемой). Для порчи не-текста выигрыш —
максимум остальных выигрышей (выпрямление колонки, край, линейки, доворот).

Мягкие случаи v16 для штрихов сохранены (одиночный штрих своей ориентации, равномерный уход всех линеек,
средний сдвиг граф): они не идут в правила 3, 4 и 6, но сами выводят страницу из ok.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ocr_utils.geometry_regression.scoring import DEFAULT_MIN_GAIN, DEFAULT_RATIO


class Group(str, Enum):
    """Группа метрик одного явления: в сумму «совокупности» группа идёт одним числом."""

    HORIZONTALS = "горизонтали"  # строки и горизонтальные линейки
    VERTICALS = "вертикали"  # края блоков и вертикальные линейки
    STROKES = "линейки"  # параллельность, изгиб и излом линеек вне line art
    FORMULAS = "дробные черты"
    LINEART = "line art"
    PHOTO = "фото"


class Rule(str, Enum):
    """Правило, давшее вердикт."""

    CLEAN = "ниже порогов"
    UNFORGIVABLE = "непрощаемая"
    HARD = "жёсткий порог"
    TOTAL = "совокупность"
    NO_GAIN = "нет выигрыша"
    HYSTERESIS = "порча ≥ ratio × выигрыш"
    MIXED = "выигрыш перевешивает"


class Verdict(str, Enum):
    """Вердикт страницы: ``bad`` — брать страницу без коррекции."""

    OK = "ok"
    MIXED = "mixed"
    BAD = "bad"


@dataclass(frozen=True)
class MetricSpec:
    """Метрика порчи: порог (score = 1), жёсткий порог (bad сразу), группа, короткое имя, непрощаемость, текст ли это."""

    threshold: float
    hard: float
    group: Group
    reason: str
    unforgivable: bool = False
    text: bool = False  # порча текста: её прощает и выпрямленная одна строка


# Жёсткие пороги стартовые: 5 × порог (прежнее правило hard = 5), калибруются по поясам пака.
HARD_FACTOR = 5.0


def _spec(
    threshold: float,
    group: Group,
    reason: str,
    unforgivable: bool = False,
    text: bool = False,
    hard: float | None = None,
) -> MetricSpec:
    return MetricSpec(threshold, HARD_FACTOR * threshold if hard is None else hard, group, reason, unforgivable, text)


# Пороги метрик v16 — как в ``ocr_utils/geometry_regression/v16/scoring.py`` (откалиброваны там); новые — стартовые.
METRICS: dict[str, MetricSpec] = {
    # Горизонтали: единое качество строки (наклон + дуга + волна), форма крупных строк v16, горизонтальные линейки.
    # Порог строки 0.8 мм (было 0.9): заголовок 1967/05 с.26 («растянут и наклонён», эталон bad) дал 0.89.
    "line_quality_mm": _spec(0.8, Group.HORIZONTALS, "line", text=True),
    "line_step_ratio": _spec(0.06, Group.HORIZONTALS, "step", text=True),
    "line_wedge_ratio": _spec(0.05, Group.HORIZONTALS, "wedge", text=True),
    "line_stretch_ratio": _spec(0.05, Group.HORIZONTALS, "stretch", text=True),
    "hstroke_dev_max_delta_mm": _spec(0.7, Group.HORIZONTALS, "htilt"),
    # Вертикали: края выровненных сторон текстовых блоков и вертикальные линейки.
    "edge_quality_mm": _spec(1.0, Group.VERTICALS, "edge", text=True),
    "vstroke_dev_max_delta_mm": _spec(0.7, Group.VERTICALS, "vtilt"),
    "vstroke_tilt_wmean_delta": _spec(0.35, Group.VERTICALS, "vmean"),
    # Линейки вне line art: параллельность (таблицы, формулы), изгиб, излом.
    "parallel_spread_other": _spec(1.5, Group.STROKES, "parallel"),
    "stroke_bend_dev_mm": _spec(0.8, Group.STROKES, "bend"),
    "stroke_jog_dev_mm": _spec(1.5, Group.STROKES, "jog"),
    # Дробные черты формул.
    "hstroke_tilt_wmean_delta": _spec(0.5, Group.FORMULAS, "hmean", unforgivable=True),
    "fraction_tilt_mean_delta_deg": _spec(0.6, Group.FORMULAS, "fraction"),
    "fraction_tilt_max_delta_deg": _spec(1.5, Group.FORMULAS, "fraction_max"),
    # Зона формул по рамкам разбора (``lineart_flow``, приставка ``formula``). Изгиб или перелом черты формулы —
    # размах поперечного смещения вдоль длинной почти горизонтальной черты, мм (1975/08 с.49 — 0.57, эталон «черты
    # наклонились» 0.55–0.97, good — 0.13; по 243 страницам с чертами p95 0.60): непрощаемо, как наклон дробных черт.
    "formula_bar_ptp_mm": _spec(0.5, Group.FORMULAS, "formula_bar", unforgivable=True),
    # Перекос формулы целиком — ухудшение абсолютного наклона рамки формулы к оси изображения, мм ухода её конца
    # (эталон bad 0.79–1.88, good 0.22; по 452 страницам с формулами p90 1.0).
    "formula_skew_mm": _spec(1.0, Group.FORMULAS, "formula_skew"),
    # Line art (непрощаемое целиком): доля тайлов без пары, углы, поворот, разброс, изгиб, разошедшиеся параллели
    # (1967/10 с.74 — оргсхема, 1968/05 с.55 — сетка графика: «такое прощать нельзя»).
    "field_lineart_weak_frac": _spec(0.45, Group.LINEART, "lineart", unforgivable=True),
    "lineart_axis_delta_deg": _spec(0.7, Group.LINEART, "la_axis", unforgivable=True),
    "lineart_rot_max_deg": _spec(3.0, Group.LINEART, "la_rot", unforgivable=True),
    "lineart_spread_delta_deg": _spec(1.5, Group.LINEART, "la_spread", unforgivable=True),
    "lineart_bend_mm": _spec(0.8, Group.LINEART, "la_bend", unforgivable=True),
    "parallel_spread_lineart": _spec(1.5, Group.LINEART, "la_parallel", unforgivable=True),
    # Неригидная деформация рисунка по плотному полю B → A (AAD, ``lineart_flow``), мм: блок-схемы с подписями, которые
    # семейство v16 не мерило (1967/10 с.74 — 0.34, 1974/12 с.42 — 0.38; эталон bad от 0.34 до 0.94, good — до 0.29).
    "lineart_aad_mm": _spec(0.3, Group.LINEART, "la_aad", unforgivable=True),
    # Сдвиг части рисунка относительно соседних параллельных линий (p95 разности поперечных смещений отрезков LSD
    # любой ориентации, ``lineart_flow.line_variants``), мм: перекошенный столбец блоков схемы. Эталон 2026-09-29:
    # bad line art 1.05–7.7 у десяти страниц из тринадцати (1967/10 с.74 — 1.05, 1974/12 с.34 — 2.4), good — до 0.78
    # (волнистый росчерк логотипа 1975/04 с.95).
    "lineart_seg_rel_mm": _spec(1.0, Group.LINEART, "la_shift", unforgivable=True),
    # Форма рисунка сверх деформации соседнего текста (v18, ``lineart_flow.frame_shape``: ``lineart_shape_mm``, поворот
    # частей, изгиб) в вердикт НЕ входит: на отсмотре 13 смен v17 → v18 при пороге 1.5 мм верна одна (1972/10 с.79),
    # остальные — полутоновые фото, которые разбор v6 считает рисунком, и графики, где B и A совпадают (сопоставление
    # участков на растре и штриховке неоднозначно); эталон good 1974/02 с.95 (бланк) — ложная тревога. Меры остаются
    # в кэше и в шапке оверлея для разбора.
    # Фото.
    "raster_edge_bend_mm": _spec(0.8, Group.PHOTO, "photo_bend", unforgivable=True),
    "raster_photo_tilt_mm": _spec(1.5, Group.PHOTO, "photo_tilt"),
}

# Выигрыш: что коррекция выправила. Пороги — «заметно глазом»; строки и края — стартовые.
GAINS: dict[str, tuple[float, str]] = {
    "lines_quality_gain_mm": (0.4, "lines"),  # выпрямление строк колонки (лучшее окно соседних строк)
    "line_quality_gain_mm": (0.9, "line"),  # выпрямленная одна строка (заголовок) — прощает только порчу текста
    "edge_quality_gain_mm": (1.5, "edge"),
    "hstroke_gain_mm": (0.8, "htilt"),
    "vstroke_gain_mm": (0.7, "vtilt"),
    "page_deskew_gain_deg": (1.0, "deskew"),
}
# Выигрыш, который прощает только порчу текста.
TEXT_ONLY_GAINS = ("line_quality_gain_mm",)

# Применимость средних по штрихам (как в v16): от стольких пар своей ориентации.
MIN_PAIRS = {
    "hstroke_tilt_wmean_delta": ("hstroke_pairs", 2),
    "vstroke_tilt_wmean_delta": ("vstroke_pairs", 2),
    "fraction_tilt_mean_delta_deg": ("fraction_bars", 2),
    "fraction_tilt_max_delta_deg": ("fraction_bars", 2),
}
# Мягкие случаи штрихов (v16): одиночный штрих ориентации, равномерный уход всех линеек, средний сдвиг граф.
LONE_PAIRS = {"hstroke_dev_max_delta_mm": "hstroke_pairs", "vstroke_dev_max_delta_mm": "vstroke_pairs"}
UNIFORM = {"hstroke_dev_max_delta_mm": "hstroke_uniform", "vstroke_dev_max_delta_mm": "vstroke_uniform"}
MIN_UNIFORM = 0.7
SOFT = ("vstroke_tilt_wmean_delta",)

# Совокупность: сумма score групп, при которой bad. Стартовое — калибруется по поясам пака.
DEFAULT_TOTAL = 2.5


@dataclass(frozen=True)
class Score:
    """Метрика на странице: значение, score, жёсткий порог, мягкость."""

    name: str
    value: float
    score: float
    spec: MetricSpec
    soft: bool
    applicable: bool


@dataclass(frozen=True)
class Assessment:
    """Итог по странице: вердикт, правило, баллы метрик и групп, сумма, выигрыши.

    Attributes:
        verdict: Вердикт.
        rule: Правило, давшее вердикт.
        culprit: Метрика (или группа), на которой сработало правило.
        scores: Баллы всех метрик порчи.
        groups: Score групп (максимум по жёстким метрикам группы).
        total: Сумма score групп.
        total_other: Сумма score групп только по метрикам не-текста.
        damage: Худшая жёсткая порча (для гистерезиса).
        damage_other: Худшая жёсткая порча не-текста.
        worst: Худшая порча с мягкими (выводит из ok).
        gain: Выигрыш для порчи текста (максимум score всех выигрышей).
        gain_other: Выигрыш для порчи не-текста (без выпрямленной одной строки).
        gain_reason: Какой выигрыш даёт ``gain``.
        gains: Баллы выигрышей.
    """

    verdict: Verdict
    rule: Rule
    culprit: str
    scores: dict[str, Score]
    groups: dict[Group, float]
    total: float
    total_other: float
    damage: float
    damage_other: float
    worst: float
    gain: float
    gain_other: float
    gain_reason: str
    gains: dict[str, float] = field(default_factory=dict)


class Thresholds17:
    """Пороги и правила v17 с перекрытиями из командной строки."""

    def __init__(
        self,
        overrides: dict[str, float] | None = None,
        hard_overrides: dict[str, float] | None = None,
        total: float = DEFAULT_TOTAL,
        min_gain: float = DEFAULT_MIN_GAIN,
        ratio: float = DEFAULT_RATIO,
    ):
        """Пороги.

        Args:
            overrides: ``имя → порог`` для метрик порчи или выигрыша.
            hard_overrides: ``имя → жёсткий порог`` метрик порчи (в единицах метрики).
            total: Порог суммы групп ``S``.
            min_gain: Выигрыш ниже — порча не прощается.
            ratio: Жёсткая порча (и сумма групп) не ниже этой доли выигрыша — bad.
        """
        self.specs = dict(METRICS)
        self.gains = {name: value for name, (value, _) in GAINS.items()}
        for name, value in (overrides or {}).items():
            if name in self.specs:
                spec = self.specs[name]
                self.specs[name] = MetricSpec(
                    float(value), spec.hard, spec.group, spec.reason, spec.unforgivable, spec.text
                )
            elif name in self.gains:
                self.gains[name] = float(value)
            else:
                raise KeyError(f"неизвестная метрика {name!r}")
        for name, value in (hard_overrides or {}).items():
            spec = self.specs[name]
            self.specs[name] = MetricSpec(
                spec.threshold, float(value), spec.group, spec.reason, spec.unforgivable, spec.text
            )
        self.total, self.min_gain, self.ratio = total, min_gain, ratio

    @staticmethod
    def parse_pairs(items: tuple[str, ...]) -> dict[str, float]:
        """Строки ``имя=число`` командной строки → словарь."""
        out = {}
        for item in items:
            name, _, value = item.partition("=")
            if not value:
                raise ValueError(f"ожидалось имя=число, получено {item!r}")
            out[name.strip()] = float(value)
        return out

    def _applicable(self, name: str, metrics: dict[str, float]) -> bool:
        if name in MIN_PAIRS:
            key, minimum = MIN_PAIRS[name]
            return float(metrics.get(key, 0.0) or 0.0) >= minimum
        return True

    def _soft(self, name: str, metrics: dict[str, float]) -> bool:
        if name in SOFT:
            return True
        if name in LONE_PAIRS and float(metrics.get(LONE_PAIRS[name], 0.0) or 0.0) <= 1.0:
            return True
        return name in UNIFORM and float(metrics.get(UNIFORM[name], 0.0) or 0.0) >= MIN_UNIFORM

    @staticmethod
    def _group_sum(items: list[Score]) -> tuple[dict[Group, float], float]:
        """Score групп (максимум по жёстким метрикам группы) и их сумма."""
        groups = {group: 0.0 for group in Group}
        for item in items:
            if not item.soft:
                groups[item.spec.group] = max(groups[item.spec.group], item.score)
        return groups, float(sum(groups.values()))

    def assess(self, metrics: dict[str, float]) -> Assessment:
        """Вердикт страницы по её метрикам (плоский словарь v16 + v17)."""
        scores: dict[str, Score] = {}
        for name, spec in self.specs.items():
            applicable = self._applicable(name, metrics)
            value = float(metrics.get(name, 0.0) or 0.0)
            score = value / spec.threshold if applicable and spec.threshold > 0 else 0.0
            scores[name] = Score(name, value, score, spec, self._soft(name, metrics), applicable)
        groups, total = self._group_sum(list(scores.values()))
        _, total_other = self._group_sum([s for s in scores.values() if not s.spec.text])
        hard_items = [s for s in scores.values() if not s.soft]
        damage = max((s.score for s in hard_items), default=0.0)
        damage_other = max((s.score for s in hard_items if not s.spec.text), default=0.0)
        worst_item = max(scores.values(), key=lambda s: s.score)
        worst = worst_item.score
        gains = {name: float(metrics.get(name, 0.0) or 0.0) / t for name, t in self.gains.items() if t > 0}
        gain_name = max(gains, key=gains.get) if gains else ""
        gain = gains.get(gain_name, 0.0)
        gain_other = max((v for name, v in gains.items() if name not in TEXT_ONLY_GAINS), default=0.0)

        def result(verdict: Verdict, rule: Rule, culprit: str) -> Assessment:
            return Assessment(verdict, rule, culprit, scores, groups, total, total_other, damage, damage_other, worst,
                              gain, gain_other, GAINS[gain_name][1] if gain_name else "", gains)  # fmt: skip

        # Совокупность против выигрыша: вся сумма — против полного выигрыша, сумма не-текста — против выигрыша
        # без одиночной строки.
        crimes = total >= self.total and (total >= self.ratio * gain or total_other >= self.ratio * gain_other)
        if worst < 1.0 and not crimes:
            return result(Verdict.OK, Rule.CLEAN, "")
        unforgivable = [s for s in scores.values() if s.spec.unforgivable and s.score >= 1.0]
        if unforgivable:
            return result(Verdict.BAD, Rule.UNFORGIVABLE, max(unforgivable, key=lambda s: s.score).name)
        over_hard = [s for s in hard_items if s.applicable and s.value >= s.spec.hard]
        if over_hard:
            return result(Verdict.BAD, Rule.HARD, max(over_hard, key=lambda s: s.value / s.spec.hard).name)
        if crimes:
            return result(Verdict.BAD, Rule.TOTAL, max(groups, key=groups.get).value)
        # Выигрыш, положенный худшей порче: одиночная строка прощает только порчу текста.
        if worst_item.score >= 1.0 and (gain if worst_item.spec.text else gain_other) < self.min_gain:
            return result(Verdict.BAD, Rule.NO_GAIN, worst_item.name)
        text_items = [s for s in hard_items if s.spec.text]
        other_items = [s for s in hard_items if not s.spec.text]
        if text_items and max(s.score for s in text_items) >= self.ratio * gain:
            return result(Verdict.BAD, Rule.HYSTERESIS, max(text_items, key=lambda s: s.score).name)
        if other_items and damage_other >= 1.0 and damage_other >= self.ratio * gain_other:
            return result(Verdict.BAD, Rule.HYSTERESIS, max(other_items, key=lambda s: s.score).name)
        return result(Verdict.MIXED, Rule.MIXED, worst_item.name)

    def describe(self) -> str:
        """Пороги одной строкой на метрику — для шапки отчёта."""
        lines = [
            f"{name} = {s.threshold:g} (жёсткий {s.hard:g}, группа «{s.group.value}»"
            f"{', непрощаемая' if s.unforgivable else ''}{', текст' if s.text else ''})"
            for name, s in self.specs.items()
        ]
        lines += [
            f"выигрыш {name} = {value:g}{' (прощает только порчу текста)' if name in TEXT_ONLY_GAINS else ''}"
            for name, value in self.gains.items()
        ]
        lines.append(
            f"совокупность: Σ ≥ S = {self.total:g} и Σ ≥ ratio × выигрыш; min_gain = {self.min_gain:g}, ratio = {self.ratio:g}"
        )
        return "\n".join(lines)


__all__ = [
    "Assessment",
    "GAINS",
    "Group",
    "METRICS",
    "MetricSpec",
    "Rule",
    "Score",
    "TEXT_ONLY_GAINS",
    "Thresholds17",
    "Verdict",
]
