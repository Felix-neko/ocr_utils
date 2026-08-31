"""Абсолютные пороги расфокуса: теги тяжести вместо «покажи худшие N %».

ЗАЧЕМ ЭТО ЕСТЬ. Отбор по доле худших отвечает на вопрос «кто в этой папке хуже
остальных», а спрашивают у детектора другое: «что переснимать». Это разные вопросы, и
расходятся они тем сильнее, чем сильнее гуляет качество съёмки. Замерено на тринадцати
подшивках «Социалистической индустрии»: доля брака по съёмочным сессиям меняется от 0 %
(19.08, целая сессия из 304 кадров без единого плохого) до 10.2 % (17.08). Фиксированные
15 % на первой гонят на просмотр полсотни заведомо хороших кадров, на второй — обрезают
список ровно там, где брак ещё не кончился.

ПОЧЕМУ АБСОЛЮТНЫЙ ПОРОГ ВООБЩЕ ВОЗМОЖЕН. Это не предположение, а измерение
(``defocus_validation_si_report.md``, раздел 5): для каждой из десяти сессий с браком
общий порог 2.970 по ``dom`` даёт РОВНО ту же полноту, что и порог, настроенный по этой
самой сессии, — совпадение до процента во всех десяти. Причина в устройстве метрики: DOM
есть отношение двух сумм, линейных по контрасту, поэтому экспозиция сокращается, и
уровень балла между съёмками не плывёт. У метрик, зависящих от контраста (``laplacian``,
``hf_mid``, ``moire``), такого свойства нет, и вешать на них абсолютный порог нельзя.

ЧТО ПОРОГ НЕ ДЕЛАЕТ. Абсолютным становится УРОВЕНЬ РАЗМЫТИЯ, а не чувствительность к
чужому вкусу. Полнота по сессиям всё равно гуляет от 0 до 100 % — просто потому, что в
разные дни человек помечал брак разной тяжести. Тег отвечает «насколько размыт этот
кадр», а не «забраковали бы вы его в тот день».
"""

from dataclasses import dataclass

import numpy as np

# Теги тяжести. Соответствуют пользовательским суффиксам _defocus / _defocus_light /
# _defocus_ultralight; пустая строка — кадр порогов не превысил.
HEAVY = "defocus_heavy"
MEDIUM = "defocus_medium"
LIGHT = "defocus_light"
ULTRALIGHT = "defocus_ultralight"
NO_TAG = ""

# От тяжёлого к лёгкому — в этом порядке теги перечисляются в отчётах и раскладываются
# по подпапкам симлинков.
TAG_ORDER = (HEAVY, MEDIUM, LIGHT, ULTRALIGHT)

TAG_TITLES = {
    HEAVY: "тяжёлый расфокус",
    MEDIUM: "средний расфокус",
    LIGHT: "лёгкий расфокус",
    ULTRALIGHT: "ультралёгкий расфокус",
}

# Оценка XMP-рейтингом: чем тяжелее брак, тем ниже звёзды. Пригодится, когда дойдут руки
# до сайдкаров; здесь лежит рядом с тегами, чтобы соответствие было в одном месте.
TAG_RATINGS = {HEAVY: 1, MEDIUM: 2, LIGHT: 3, ULTRALIGHT: 4}

# На чём считается порог: "raw" — сам балл метрики, "norm" — балл, нормированный на
# высоту строки (доступен только в режиме по строкам).
BASES = ("raw", "norm")


@dataclass(frozen=True)
class Preset:
    """Готовый набор порогов под конкретную метрику и конкретный режим агрегации.

    Пороги идут ПО ВОЗРАСТАНИЮ балла (то есть по убыванию тяжести): у всех метрик
    пакета шкала однонаправленная — больше значит резче, — поэтому «превысить порог»
    всегда означает «оказаться НИЖЕ него».

    Attributes:
        name: Имя для ``--tag-preset``.
        algorithm: Метрика, на которой пороги откалиброваны.
        aggregation: Режим сведения тайлов, при котором они действительны.
        quantile: Квантиль агрегации.
        basis: "raw" — порог на самом балле, "norm" — на балле, нормированном на кегль.
        heavy: Ниже этого балла — тяжёлый расфокус.
        medium: Ниже этого — средний.
        light: Ниже этого — лёгкий.
        ultralight: Ниже этого — ультралёгкий, самый мягкий уровень: список «посмотреть»,
            а не «переснять». Равенство ``light`` означает, что полосы нет вовсе.
        source: На чём откалибровано. Пишется в отчёт, чтобы через год было понятно,
            чему верить и что перепроверять.
    """

    name: str
    algorithm: str
    aggregation: str
    quantile: float
    basis: str
    heavy: float
    medium: float
    light: float
    ultralight: float
    source: str

    def __post_init__(self) -> None:
        """Проверяет, что пороги идут по возрастанию, а основание известно.

        Raises:
            ValueError: Если порядок порогов нарушен или основание не из ``BASES``.
        """
        if self.basis not in BASES:
            raise ValueError(f"неизвестное основание порога: {self.basis}")
        if not self.heavy <= self.medium <= self.light <= self.ultralight:
            raise ValueError(
                f"пороги пресета {self.name} должны идти по возрастанию балла "
                f"(heavy <= medium <= light <= ultralight), а заданы "
                f"{self.heavy} / {self.medium} / {self.light} / {self.ultralight}"
            )

    def tag(self, score: float) -> str:
        """Присваивает кадру тег тяжести по его баллу.

        Args:
            score: Балл резкости (больше = резче); NaN означает «не измерено».

        Returns:
            Один из ``TAG_ORDER`` либо пустая строка.
        """
        if score is None or not np.isfinite(score):
            # Неизмеренный кадр не тегируем: «нет данных» — это не «всё хорошо», но и
            # не брак. Он и так виден в отчёте отдельной строкой с ошибкой.
            return NO_TAG
        if score < self.heavy:
            return HEAVY
        if score < self.medium:
            return MEDIUM
        if score < self.light:
            return LIGHT
        if score < self.ultralight:
            return ULTRALIGHT
        return NO_TAG

    def describe(self) -> str:
        """Однострочное описание порогов для шапки отчёта.

        Returns:
            Строка вида «dom/best: тяжёлый < 2.92, средний < 2.96, лёгкий < 3».
        """
        levels = ", ".join(
            f"{TAG_TITLES[tag].split()[0]} < {getattr(self, tag.removeprefix('defocus_')):g}" for tag in TAG_ORDER
        )
        return f"{self.algorithm}/{self.aggregation}: {levels}"


PRESETS: dict[str, Preset] = {
    preset.name: preset
    for preset in (
        Preset(
            name="dom-si",
            algorithm="dom",
            aggregation="best",
            quantile=0.80,
            basis="raw",
            heavy=2.92,
            medium=2.96,
            light=3.00,
            ultralight=3.04,
            source=(
                "«Социалистическая индустрия» 1985-1987, 4009 кадров, 176 ручных пометок. "
                "Порог 3.00 ловит 73 % размеченного брака, помечая 7.3 % кадров; проверка на "
                "отсмотренной глазами и чистой подшивке 1988/01-03 — 2 ложных срабатывания из 304. "
                "Уровень ultralight (3.04) добавлен по «Экономической газете» 1982 и стоит особняком: "
                "на той же чистой подшивке он метит 9 кадров из 304 (3 %), то есть его список ЗАВЕДОМО "
                "содержит хорошие кадры и читается как «посмотреть», а не «переснять». Выше поднимать "
                "нельзя: 3.08 ловит 9 из 10 самых лёгких пометок, но метит там же 112 кадров из 304. "
                "Подробности: defocus_validation_si_report.md, разделы 5 и 6"
            ),
        ),
    )
}

DEFAULT_PRESET = "dom-si"


def parse_thresholds(text: str) -> dict[str, float]:
    """Разбирает строку вида ``heavy=2.9,medium=2.95,light=3.0[,ultralight=3.04]``.

    ``ultralight`` необязателен: без него он приравнивается к ``light``, то есть полоса
    самого мягкого уровня пуста и тег не ставится никому. Так уже написанные команды с
    тремя уровнями продолжают работать и означают ровно то же, что раньше.

    Args:
        text: Значение опции ``--tag-thresholds``.

    Returns:
        Словарь с ключами ``heavy``, ``medium``, ``light``, ``ultralight``.

    Raises:
        ValueError: Если формат нарушен, ключ неизвестен или задан не весь
            обязательный набор.
    """
    wanted = ("heavy", "medium", "light", "ultralight")
    required = wanted[:3]
    values: dict[str, float] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"ожидалось «уровень=число», получено «{chunk}»")
        key, _, raw = chunk.partition("=")
        key = key.strip()
        if key not in wanted:
            raise ValueError(f"неизвестный уровень «{key}»; допустимы {', '.join(wanted)}")
        try:
            values[key] = float(raw.strip())
        except ValueError as error:
            raise ValueError(f"«{raw.strip()}» не число") from error
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"не заданы уровни: {', '.join(missing)}")
    values.setdefault("ultralight", values["light"])
    return values


def registry_text() -> str:
    """Печатный список готовых пресетов для ``--tag-preset list``.

    Returns:
        Многострочный текст с порогами и происхождением каждого пресета.
    """
    lines = ["Готовые пресеты порогов (--tag-preset):", ""]
    for preset in PRESETS.values():
        lines.append(f"  {preset.name}")
        lines.append(f"    {preset.describe()} (квантиль {preset.quantile:g}, основание {preset.basis})")
        lines.append(f"    откалиброван: {preset.source}")
        lines.append("")
    return "\n".join(lines)
