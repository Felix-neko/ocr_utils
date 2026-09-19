"""Склейка разорванных переносов («кре-диты» → «кредиты») по словарю морфоанализатора — шаг пост-обработки полосы.

Откуда задача. На полосах с обрезанным левым краем DeepSeek в части ответов оставляет переносы
как напечатано («кре-дитного», 38 из 39 дефисов на с. 47 МТС 1991/02; режим A в
`reports/external_ocr_hyphenation_second_pass.md`), и промптом это не лечится. Прежний опыт
(сессия 17–18.09.2026, в репо не попал) с `pymorphy3`: правило «слитная форма известна словарю,
а первая половина не оканчивается на соединительную «о» перед словарным словом» дало 36/39 на
с. 47 и 9 срабатываний на 348 дефисных словах остальных полос, из них 3 — настоящие составные
(«торгово-экономических»: `known("торгово")` у pymorphy3 ложно, и оговорка не срабатывает).
Здесь то же правило воспроизведено (``JoinRule.C``), рядом — простое ``A``, ``D`` (известная
анализатору дефисная форма не трогается) и боевое ``E``. Сравнение бэкендов и правил на 322
размеченных словах — `scripts/compare_hyphen_join.py` и `run_scripts/experimental/hyphen_labels.csv`
(`reports/external_ocr_damaged_research.md`, разд. 3): pymorphy3 с правилом E — 77/79 переносов
склеено, 36/36 на с. 47, 1 ложное слияние на 234 составных; `mawo-pymorphy3` непригоден — его
``is_known`` истинен почти для любой склейки («фирмыизготовителя», «купляпродажа»), 227/234 ложных.

В боевом прогоне (``ocr.recognize_page``, после доводки структуры) работает :func:`default_morph`
(pymorphy3) с правилом ``E``; склеенные слова уходят в meta (``hyphens_joined``), выключается
``--no-join-hyphens``. Теги внутри слова (``<supplied>``, ``<unclear>``) снимаются на время
проверки и остаются на месте в тексте: заменяется только дефис. ``pymorphy3`` +
``pymorphy3-dicts-ru`` — основные зависимости; ``mawo-pymorphy3`` — только для сравнения (группа
``experimental``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache

# Дефис между двумя кириллическими кусками от двух букв, не часть более длинной цепочки
# («научно-производственно-технический» не трогаем); теги внутри слова допускаются. После дефиса
# может стоять один перевод строки («за-\nдолженность»: модель оставила строку как напечатано) —
# при склейке он уходит вместе с дефисом.
_TAG = r"(?:</?(?:supplied|unclear)>)*"
HYPHENATED = re.compile(
    rf"(?<![\w-])(?P<a>{_TAG}[а-яёА-ЯЁ]{{2,}}(?:{_TAG}[а-яёА-ЯЁ]+)*{_TAG})-\n?(?P<b>{_TAG}[а-яё]{{2,}}(?:{_TAG}[а-яё]+)*{_TAG})(?![\w-])"
)
_STRIP_TAGS = re.compile(r"</?(?:supplied|unclear)>")
# Граница полос: хвост абзаца — половина слова с дефисом на конце (закрывающие теги могут стоять и
# до дефиса, и после: «<supplied>снаб-</supplied>»), голова следующего — строчное продолжение с
# возможными открывающими тегами перед ним.
PAGE_TAIL = re.compile(rf"(?<![\w-])(?P<a>{_TAG}[а-яёА-ЯЁ]{{2,}}(?:{_TAG}[а-яё]+)*{_TAG})-(?P<tail_tags>{_TAG})\s*$")
PAGE_HEAD = re.compile(rf"^(?P<head_tags>{_TAG})(?P<b>[а-яё]{{2,}}(?:{_TAG}[а-яё]+)*{_TAG})(?![\w-])")


class MorphBackend(StrEnum):
    """Какой морфоанализатор отвечает на вопрос «известно ли слово словарю»."""

    PYMORPHY3 = "pymorphy3"  # pymorphy3 + pymorphy3-dicts-ru (OpenCorpora) — боевой
    MAWO = "mawo"  # mawo-pymorphy3: только для сравнения, словарём не служит (см. докстринг модуля)


class JoinRule(StrEnum):
    """Когда дефис между половинами ``a`` и ``b`` считается переносом и убирается.

    ``A`` — слитная форма известна словарю.
    ``C`` — то же, но не тогда, когда обе половины известны и первая кончается на «о» (защита
    составных «торгово-экономический»; прежнее правило опыта 18.09).
    ``D`` — ``A``, но дефисная форма целиком, известная анализатору как составное слово
    («материально-техническое»), не трогается; половины при этом не проверяются.
    ``E`` — ``A``, но первая половина от пяти букв на «о» перед известным словом от пяти букв —
    соединительная гласная составного прилагательного («торгово-экономических», «резино-асбестовые»),
    не перенос («миллио-нов», «эконо-мику» с короткой второй половиной остаются переносами);
    в отличие от ``C`` не требует, чтобы первая половина была известна словарю (у pymorphy3
    «торгово» неизвестно, и ``C`` на этих словах ошибается). Добавлено после разбора трёх ложных
    слияний правила C на корпусе `hyphen_labels.csv`.
    """

    A = "A"
    C = "C"
    D = "D"
    E = "E"


class Morph:
    """Обёртка над анализатором: ``known(слово)`` с кэшем; создаётся один раз на процесс.

    Args:
        backend: Какой пакет использовать.
    """

    def __init__(self, backend: MorphBackend):
        self.backend = MorphBackend(backend)
        if self.backend is MorphBackend.PYMORPHY3:
            import pymorphy3  # noqa: PLC0415 — тяжёлая зависимость группы experimental, грузится по требованию

            self._analyzer = pymorphy3.MorphAnalyzer()
        else:
            from mawo_pymorphy3 import create_analyzer  # noqa: PLC0415

            self._analyzer = create_analyzer()

    @lru_cache(maxsize=65536)
    def known(self, word: str) -> bool:
        """Известно ли слово словарю (хотя бы один разбор не «по догадке»).

        Args:
            word: Слово без тегов, регистр не важен.

        Returns:
            ``True``, если анализатор нашёл слово в словаре (``is_known`` у разбора); у обоих
            бэкендов дефисные составные слова тоже могут быть известны.
        """
        return any(parse.is_known for parse in self._analyzer.parse(word.lower()))


@lru_cache(maxsize=1)
def default_morph() -> Morph:
    """Боевой анализатор — pymorphy3, один на процесс (загрузка ≈ 0.05 с, словарь ≈ 10 МБ в памяти).

    Returns:
        Общий :class:`Morph`; потокобезопасен для чтения, поэтому годится и в пуле потоков прогона.
    """
    return Morph(MorphBackend.PYMORPHY3)


@dataclass
class JoinReport:
    """Что склеено и что оставлено — для стенда и meta."""

    joined: list[str] = field(default_factory=list)  # «кре-диты» → «кредиты», как в тексте до склейки
    kept: list[str] = field(default_factory=list)  # дефисные слова, оставленные как есть


def should_join(a: str, b: str, morph: Morph, rule: JoinRule) -> bool:
    """Решение по правилу для половин без тегов.

    Args:
        a: Часть до дефиса.
        b: Часть после дефиса.
        morph: Анализатор.
        rule: Правило.

    Returns:
        ``True`` — дефис считается переносом, слово склеивается.
    """
    rule = JoinRule(rule)
    whole = a + b
    if rule is JoinRule.D and morph.known(f"{a}-{b}"):
        return False  # анализатор сам знает дефисную форму как слово — это составное, не перенос
    if not morph.known(whole):
        return False
    if rule is JoinRule.C and morph.known(a) and morph.known(b) and a.lower().endswith("о"):
        return False
    if rule is JoinRule.E and len(a) >= 5 and len(b) >= 5 and a.lower().endswith("о") and morph.known(b):
        return False
    return True


class _Replacer:
    """Замена для ``re.sub``: решает по половинам слова и копит отчёт (вместо вложенной функции).

    Args:
        morph: Анализатор.
        rule: Правило склейки.
    """

    def __init__(self, morph: Morph, rule: JoinRule):
        self.morph = morph
        self.rule = JoinRule(rule)
        self.report = JoinReport()

    def __call__(self, match: re.Match) -> str:
        """Слово без дефиса, если это перенос; иначе совпадение как есть."""
        a_raw, b_raw = match.group("a"), match.group("b")
        a, b = _STRIP_TAGS.sub("", a_raw), _STRIP_TAGS.sub("", b_raw)
        if should_join(a, b, self.morph, self.rule):
            self.report.joined.append(f"{a}-{b}")
            return a_raw + b_raw
        self.report.kept.append(f"{a}-{b}")
        return match.group(0)


def join_broken_hyphens(text: str, morph: Morph, rule: JoinRule = JoinRule.E) -> tuple[str, JoinReport]:
    """Убрать дефисы-переносы внутри слов текста по словарю.

    Args:
        text: Тело полосы в markdown (теги повреждений внутри слов допустимы).
        morph: Анализатор (:class:`Morph`).
        rule: Правило склейки.

    Returns:
        ``(текст со склеенными переносами, отчёт)``: в тексте заменён только дефис, теги и
        регистр сохранены; в отчёте — какие слова склеены и какие оставлены.
    """
    replacer = _Replacer(morph, rule)
    return HYPHENATED.sub(replacer, text), replacer.report


@dataclass(frozen=True)
class BoundaryJoin:
    """Итог склейки слова, разорванного границей полос: сшитый абзац и что с дефисом сделано."""

    text: str  # хвост предыдущей полосы + голова следующей одним абзацем
    word: str  # слово как было напечатано, без тегов: «снаб-жения»
    joined: bool  # True — дефис-перенос убран; False — составное слово, дефис оставлен
    head_start: int  # смещение в ``text``, с которого начинается голова (её первый символ или тег)


def join_across_boundary(tail: str, head: str, morph: Morph, rule: JoinRule = JoinRule.E) -> BoundaryJoin | None:
    """Сшить два абзаца по слову, разорванному переносом на границе полос.

    Половины проверяются словарём без тегов (:func:`should_join`), а в тексте убирается только дефис
    (и пробельный хвост абзаца): теги повреждений остаются там, где стояли, поэтому
    ``<supplied>снаб-</supplied>`` + ``жения`` → ``<supplied>снаб</supplied>жения``. Если словарь слитной
    формы не знает (составное «торгово-» + «экономических»), абзацы всё равно сшиваются — без пробела,
    с дефисом.

    Args:
        tail: Последний абзац предыдущей полосы.
        head: Первый абзац следующей полосы.
        morph: Анализатор.
        rule: Правило склейки.

    Returns:
        :class:`BoundaryJoin` или ``None``, если хвост не кончается половиной слова с дефисом либо
        голова не начинается со строчного продолжения.
    """
    tail_match = PAGE_TAIL.search(tail)
    head_match = PAGE_HEAD.match(head)
    if tail_match is None or head_match is None:
        return None
    a_raw, b_raw = tail_match.group("a"), head_match.group("b")
    a, b = _STRIP_TAGS.sub("", a_raw), _STRIP_TAGS.sub("", b_raw)
    joined = should_join(a, b, morph, rule)
    hyphen = "" if joined else "-"
    stem = tail[: tail_match.start()] + a_raw + hyphen + tail_match.group("tail_tags")
    text = stem + head_match.group("head_tags") + b_raw + head[head_match.end() :]
    return BoundaryJoin(text, f"{a}-{b}", joined, len(stem))
