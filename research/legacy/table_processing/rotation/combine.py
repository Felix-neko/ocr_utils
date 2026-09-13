"""Сведение вердиктов детекторов в один ответ по ячейке и по таблице.

ДВА РАЗНЫХ ВОПРОСА, и они решаются по-разному.

ОСЬ («повёрнута ли») решается голосованием тех детекторов, которые ось меряют. Голоса
взвешены уверенностью, и молчащий детектор голоса не имеет. Замер на 59 размеченных
ячейках: ``glyph_aspect`` 59 из 59, ``surya_lines`` 48 из 50, ``ink_axis`` 54 из 59,
``profile`` 28 из 59. Веса при этом не подкручиваются: разница между детекторами уже
выражена их собственной уверенностью, а подгонка весов под 59 ячеек была бы подгонкой
под шум.

СТОРОНА («влево или вправо») решается иначе — приором таблицы. Дело в том, что сторону
называют только два детектора, и оба недёшевы или неточны, а физика задачи даёт почти
бесплатный ответ: в одной таблице боковые ячейки повёрнуты В ОДНУ СТОРОНУ. Верстальщик не
станет разворачивать соседние графы в разные стороны. Поэтому сторона считается по всей
таблице сразу: побеждает та, за которую уверенно высказались, и она же назначается
ячейкам, где сторона не определилась.

Замер на 19 боковых ячейках четырёх таблиц 1966/01: арбитр на tesseract назвал 90 во всех
19 случаях (отрыв 0.43-1.00), docTR ошибся дважды. Это согласуется с тем, что известно про
пак: широкие заголовки в советских изданиях набирали с поворотом против часовой, чтобы
читались снизу вверх.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from research.legacy.table_processing.rotation.base import Verdict

# Уверенность ниже этой считается молчанием: детектор ответил, но сам себе не верит.
MIN_CONFIDENCE = 0.05

# Перевес голосов, ниже которого ось считается неопределённой и ячейка попадает в отчёт
# как спорная. 0.0 значит «решает простое большинство»; выше ставить нечего — замер
# показал, что расхождения детекторов на этих данных редки и всегда осмысленны.
AXIS_MARGIN = 0.0


@dataclass
class CellDecision:
    """Итог по одной ячейке со следами того, кто как проголосовал."""

    rotate_cw: int
    confidence: float
    axis_votes: dict[str, float] = field(default_factory=dict)
    sign_votes: dict[str, int] = field(default_factory=dict)
    # Вес каждой стороны, сложенный из уверенностей ИМЕННО ТЕХ детекторов, кто её назвал.
    # Отдельным полем, а не выводом из sign_votes: приор таблицы обязан отличать «за 90
    # высказался арбитр с уверенностью 0.83» от «за 270 высказалась мера с 0.20», а по
    # одним только именам детекторов эти два голоса неразличимы.
    sign_weight: dict[int, float] = field(default_factory=dict)
    note: str = ""
    from_prior: bool = False

    @property
    def rotated(self) -> bool:
        return self.rotate_cw != 0


# Детекторы, которым разрешено НАЛОЖИТЬ ВЕТО. Арбитр на распознавании молчит двумя разными
# способами, и различать их обязательно: «повороты читаются одинаково» — это просто
# отсутствие голоса, а «букв не нашлось ни под каким углом» — это утверждение, что читать
# в ячейке нечего. Во втором случае ячейку не трогаем, чем бы её ни назвали дешёвые меры:
# подменять текст, которого мы не прочли, нечем.
#
# Без вето прогон по паку подменял колонки чисел и пустые графы, куда форма букв уверенно
# показывала «боковая ось»: у столбца цифр компоненты действительно шире, чем выше.
VETO_NAMES = ("ocr_vote",)
VETO_NOTE = "букв не нашлось"


def combine_axis(verdicts: dict[str, Verdict], veto_names: "tuple[str, ...]" = VETO_NAMES) -> CellDecision:
    """Ось по взвешенному голосованию; сторона — только если её назвали уверенно."""
    for name in veto_names:
        verdict = verdicts.get(name)
        if verdict is not None and VETO_NOTE in verdict.note:
            return CellDecision(0, 0.0, note=f"{name}: {verdict.note}")
    votes: dict[str, float] = {}
    rotated_weight = upright_weight = 0.0
    sign_weight: dict[int, float] = defaultdict(float)
    sign_votes: dict[str, int] = {}

    for name, verdict in verdicts.items():
        if verdict.confidence < MIN_CONFIDENCE:
            continue
        votes[name] = round(verdict.confidence, 3)
        if verdict.rotated:
            rotated_weight += verdict.confidence
            if not verdict.axis_only:
                sign_weight[verdict.rotate_cw] += verdict.confidence
                sign_votes[name] = verdict.rotate_cw
        else:
            upright_weight += verdict.confidence

    total = rotated_weight + upright_weight
    if total <= 0.0:
        return CellDecision(0, 0.0, votes, sign_votes, note="все детекторы промолчали")

    score = (rotated_weight - upright_weight) / total
    if score <= AXIS_MARGIN:
        return CellDecision(0, min(1.0, abs(score)), votes, sign_votes)

    weights = dict(sign_weight)
    if weights:
        rotation = max(weights.items(), key=lambda item: item[1])[0]
        return CellDecision(rotation, min(1.0, abs(score)), votes, sign_votes, weights)
    # Ось боковая, сторону не назвал никто: 90 как обозначение «повёрнута», сторону
    # доназначит приор таблицы.
    return CellDecision(90, min(1.0, abs(score)), votes, sign_votes, weights, note="сторона от приора")


def table_prior(decisions: list[CellDecision], default: int = 90) -> int:
    """Сторона поворота, общая для всей таблицы: за неё голосуют ячейки со своим знаком."""
    weight: dict[int, float] = defaultdict(float)
    for decision in decisions:
        if not decision.rotated:
            continue
        for rotation, value in decision.sign_weight.items():
            weight[rotation] += value
    if not weight:
        return default
    return max(weight.items(), key=lambda item: item[1])[0]


def apply_prior(decisions: list[CellDecision], default: int = 90) -> tuple[list[CellDecision], int]:
    """Назначить всем боковым ячейкам таблицы одну сторону — ту, за которую голосов больше."""
    prior = table_prior(decisions, default)
    result: list[CellDecision] = []
    for decision in decisions:
        if decision.rotated and decision.rotate_cw != prior:
            result.append(
                CellDecision(
                    prior,
                    decision.confidence,
                    decision.axis_votes,
                    decision.sign_votes,
                    decision.sign_weight,
                    note=(decision.note + "; сторона по приору таблицы").strip("; "),
                    from_prior=True,
                )
            )
        else:
            result.append(decision)
    return result, prior
