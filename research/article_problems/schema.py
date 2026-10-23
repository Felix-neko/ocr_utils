"""Классы проблем, виды виновников и разбор JSON-ответа модели в случаи и резюме по классам."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum


class ProblemClass(StrEnum):
    """Класс случая; значение — номер класса в постановке пользователя (его же видит модель)."""

    SPARE_PARTS_UNAVAILABLE = "1"  # предприятие не может найти/получить запчасти к своей технике
    WRONG_ASSORTMENT = "2"  # производитель выпускает не то, что нужно потребителям
    SPARE_PARTS_UNDERPRODUCED = "2.1"  # производитель не делает запчасти или делает их явно мало
    COSTLIER_REPLACEMENT_MODEL = "3"  # старую модель сняли, новая дороже сверх роста производительности
    NON_DELIVERY = "4"  # заказанное не поставлено (любой поставщик)
    LATE_DELIVERY = "5"  # поставка не в срок
    DEFECTIVE = "6"  # поставлены брак, некондиция
    OTHER_PROBLEM = "7"  # любые конкретные проблемы; код ставит его на каждый случай
    GOSSNAB_POOR_WORK = "8"  # плохая работа подразделений Госснаба СССР
    WORSE_THAN_FOREIGN = "9"  # советское намного дороже или хуже иностранного аналога
    REFUSES_LISTED_PRODUCT = "10"  # производитель отказывается делать продукцию из своего же ассортимента
    NO_IMPROVEMENT = (
        "11"  # производитель не совершенствует продукцию / не переходит на новые модели (или это невыгодно)
    )
    UNWANTED_PRODUCT = "12"  # производится продукция, которая не нужна никому


# Короткие названия классов для отчётов и промпта.
CLASS_TITLES: dict[ProblemClass, str] = {
    ProblemClass.SPARE_PARTS_UNAVAILABLE: "Предприятие не может найти запчасти для своей техники",
    ProblemClass.WRONG_ASSORTMENT: "Производитель выпускает не то, что нужно потребителям",
    ProblemClass.SPARE_PARTS_UNDERPRODUCED: "Производитель не делает запчасти или делает их явно мало",
    ProblemClass.COSTLIER_REPLACEMENT_MODEL: "Прежняя модель снята, новая дороже сверх роста производительности",
    ProblemClass.NON_DELIVERY: "Непоставка заказанного",
    ProblemClass.LATE_DELIVERY: "Несвоевременная поставка",
    ProblemClass.DEFECTIVE: "Поставка брака и некондиции",
    ProblemClass.OTHER_PROBLEM: "Любые конкретные проблемы и недостатки в работе предприятий",
    ProblemClass.GOSSNAB_POOR_WORK: "Плохая работа подразделений Госснаба СССР",
    ProblemClass.WORSE_THAN_FOREIGN: "Советское оборудование намного дороже или хуже иностранного",
    ProblemClass.REFUSES_LISTED_PRODUCT: "Производитель отказывается выпускать продукцию из своего ассортимента",
    ProblemClass.NO_IMPROVEMENT: "Производитель не совершенствует продукцию и не переходит на новые модели",
    ProblemClass.UNWANTED_PRODUCT: "Производится продукция, которая никому не нужна",
}


class CulpritKind(StrEnum):
    """Кто виноват в случае — чтобы потом отфильтровать, например, только заводы-изготовители."""

    MANUFACTURER = "manufacturer"  # завод-изготовитель, объединение, комбинат
    GOSSNAB_BODY = "gossnab_body"  # подразделение Госснаба: управление, союзглавснабсбыт, база, контора
    MINISTRY_SUPPLY = "ministry_supply"  # министерство, главк, их снабы и сбыты
    CONSTRUCTION = "construction"  # стройка, трест, СМУ
    TRANSPORT = "transport"  # железная дорога, пароходство, автотранспорт
    FARM = "farm"  # колхоз, совхоз, Сельхозтехника
    TRADE = "trade"  # торговля, потребкооперация
    CONSUMER = "consumer"  # сам потребитель (плохо хранит, завышает заявки)
    PLANNING = "planning"  # Госплан, фондирование, нормативы
    OTHER = "other"
    UNKNOWN = "unknown"


class Comparison(StrEnum):
    """В чём советский образец проигрывает иностранному (только для класса 9)."""

    PRICE = "price"
    QUALITY = "quality"
    BOTH = "both"


class ImprovementKind(StrEnum):
    """Вид случая класса 11 (производитель не совершенствует продукцию)."""

    STAGNATION = "stagnation"  # застой: выпускает устаревшее, тянет с новой моделью
    NOT_LAUNCHED = "not_launched"  # разработка готова к производству, но её не запускают
    BARRIER = "barrier"  # названа причина, мешающая совершенствовать (цена не возмещает затрат и т. п.)


class ParseVerdict(StrEnum):
    """Удалось ли разобрать ответ модели."""

    OK = "ok"
    BAD_JSON = "bad_json"  # ответ не разбирается как JSON-объект
    TRUNCATED = "truncated"  # finish_reason length — ответ обрезан потолком токенов


@dataclass
class Case:
    """Один случай, выписанный моделью из статьи.

    Args:
        classes: Классы случая (после нормализации кодом; класс 7 есть всегда).
        victim: Кто пострадал (предприятие, организация, отрасль).
        culprit: Кто виноват; пусто — не назван.
        culprit_kind: Вид виновника.
        product: О какой продукции речь.
        quote: Дословная цитата из статьи, подтверждающая случай.
        explanation: Почему случай отнесён к этим классам (2–4 предложения).
        concrete: Есть ли конкретика — названия, цифры, даты, а не общие слова.
        foreign_analog: Иностранный аналог (класс 9): что и чьё.
        comparison: В чём проигрыш иностранному (класс 9).
        improvement_kind: Вид случая класса 11 (застой, готовая разработка не запущена, причина-барьер);
            ``None`` — не указан.
        quote_found: Нашёл ли код цитату в тексте статьи (заполняется проверкой, не моделью).
        quote_score: Степень совпадения цитаты с текстом, 0–100.
        quote_line: Строка файла выпуска, где начинается цитата; ``None`` — не найдена.
        unknown_classes: Классы из ответа, которых нет в перечне (для отладки промпта).
    """

    classes: list[ProblemClass]
    victim: str = ""
    culprit: str = ""
    culprit_kind: CulpritKind = CulpritKind.UNKNOWN
    product: str = ""
    quote: str = ""
    explanation: str = ""
    concrete: bool = True
    foreign_analog: str = ""
    comparison: Comparison | None = None
    improvement_kind: ImprovementKind | None = None
    quote_found: bool = False
    quote_score: float = 0.0
    quote_line: int | None = None
    unknown_classes: list[str] = field(default_factory=list)


@dataclass
class ArticleAnswer:
    """Разобранный ответ по статье: случаи и резюме по классам.

    Args:
        verdict: Удалось ли разобрать ответ.
        cases: Выписанные случаи.
        class_summaries: Резюме «почему статья отнесена к классу» по каждому классу статьи.
        error: Текст ошибки разбора, если ``verdict`` не ``OK``.
    """

    verdict: ParseVerdict
    cases: list[Case] = field(default_factory=list)
    class_summaries: dict[ProblemClass, str] = field(default_factory=dict)
    error: str = ""

    @property
    def classes(self) -> set[ProblemClass]:
        """Все классы статьи — объединение классов её случаев."""
        return {cls for case in self.cases for cls in case.classes}


def _parse_class(value: object) -> ProblemClass | None:
    """Номер класса из ответа модели → ``ProblemClass``: ``"2.1"``, ``2``, ``"класс 4"`` и т. п.

    Args:
        value: Значение из списка ``classes`` ответа.

    Returns:
        Класс или ``None``, если распознать не удалось.
    """
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if match is None:
        return None
    try:
        return ProblemClass(match.group(0))
    except ValueError:
        return None


def _enum_or(enum_type: type[StrEnum], value: object, default: StrEnum | None) -> StrEnum | None:
    """Значение enum по строке из ответа или запасное, если модель написала что-то своё.

    Args:
        enum_type: Класс enum.
        value: Строка из ответа.
        default: Что вернуть при неизвестном значении.
    """
    try:
        return enum_type(str(value).strip().lower())
    except ValueError:
        return default


# Частая поломка резюме у Space Bunny: ``"class": "7": "текст"`` вместо ``"class": "7", "summary": "текст"``.
BROKEN_SUMMARY_RE = re.compile(r'("class"\s*:\s*"[\d.]+")\s*:\s*"')


def _extract_json(text: str) -> dict:
    """JSON-объект из текста ответа: сам текст или содержимое ```json-блока / от первой ``{`` до последней ``}``.

    Каждый кандидат пробуется как есть, а затем с починкой ``BROKEN_SUMMARY_RE`` (пропущенный ключ ``summary``).

    Args:
        text: Текст ответа модели.

    Returns:
        Разобранный объект; не объект или не JSON — ``ValueError``.
    """
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{") : text.rindex("}") + 1])
    candidates += [BROKEN_SUMMARY_RE.sub(r'\1, "summary": "', candidate) for candidate in candidates]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError("ответ не разбирается как JSON-объект")


def parse_answer(text: str, finish_reason: str | None) -> ArticleAnswer:
    """Ответ модели → ``ArticleAnswer``; нормализует классы, ставит 7 на каждый случай и 8 — при виновнике из Госснаба.

    Args:
        text: Текст ответа модели.
        finish_reason: Причина остановки; ``length`` — ответ обрезан.

    Returns:
        Разобранный ответ; при ошибке — ``verdict`` не ``OK`` и пустые случаи.
    """
    if finish_reason == "length":
        return ArticleAnswer(ParseVerdict.TRUNCATED, error="ответ обрезан потолком max_tokens")
    try:
        data = _extract_json(text)
    except ValueError as error:
        return ArticleAnswer(ParseVerdict.BAD_JSON, error=str(error))
    cases: list[Case] = []
    for raw in data.get("cases") or []:
        if not isinstance(raw, dict):
            continue
        # Классы: распознанные — в enum, нераспознанные — в отладочный список; дубли убираются.
        classes: list[ProblemClass] = []
        unknown: list[str] = []
        for value in raw.get("classes") or []:
            parsed = _parse_class(value)
            if parsed is None:
                unknown.append(str(value))
            elif parsed not in classes:
                classes.append(parsed)
        culprit_kind = _enum_or(CulpritKind, raw.get("culprit_kind"), CulpritKind.UNKNOWN)
        # Виновник — подразделение Госснаба: по определению это класс 8 (без рассуждений модель
        # на пилоте ставила gossnab_body, но класс 8 теряла в 13 статьях из 18).
        if culprit_kind == CulpritKind.GOSSNAB_BODY and ProblemClass.GOSSNAB_POOR_WORK not in classes:
            classes.append(ProblemClass.GOSSNAB_POOR_WORK)
        # Класс 7 — объединение: любой выписанный случай — это конкретная проблема.
        if ProblemClass.OTHER_PROBLEM not in classes:
            classes.append(ProblemClass.OTHER_PROBLEM)
        cases.append(
            Case(
                classes=sorted(classes, key=lambda cls: float(cls.value)),
                victim=str(raw.get("victim") or ""),
                culprit=str(raw.get("culprit") or ""),
                culprit_kind=culprit_kind,
                product=str(raw.get("product") or ""),
                quote=str(raw.get("quote") or ""),
                explanation=str(raw.get("explanation") or ""),
                concrete=bool(raw.get("concrete", True)),
                foreign_analog=str(raw.get("foreign_analog") or ""),
                comparison=_enum_or(Comparison, raw.get("comparison"), None),
                improvement_kind=_enum_or(ImprovementKind, raw.get("improvement_kind"), None),
                unknown_classes=unknown,
            )
        )
    # Резюме по классам: только по классам, которые есть в случаях; остальные — выдумка модели.
    present = {cls for case in cases for cls in case.classes}
    summaries: dict[ProblemClass, str] = {}
    for raw in data.get("class_summaries") or []:
        if not isinstance(raw, dict):
            continue
        parsed = _parse_class(raw.get("class"))
        if parsed is not None and parsed in present and raw.get("summary"):
            summaries[parsed] = str(raw["summary"])
    return ArticleAnswer(ParseVerdict.OK, cases=cases, class_summaries=summaries)
