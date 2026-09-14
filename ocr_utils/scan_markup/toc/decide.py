"""Решение по выпуску: какие полосы окна — оглавление и какого вида.

Чистые функции без ввода-вывода: на входе признаки полос выпуска (только окна, в порядке
``order_index``), на выходе решение по каждой. Правила — по замеру на паке-1
(``reports/toc_detection.md``):

1. СИЛЬНЫЙ признак: блок ``TableOfContents`` surya, слово «СОДЕРЖАНИЕ» или заголовок
   «Указатель»/«Перечень». Полоса с ним — оглавление всегда.
2. СЛАБЫЙ признак: строки, кончающиеся числом у общего правого края (номера страниц), —
   ``num_tail_lines`` и ``num_tail_ratio`` не ниже порогов. Такая полоса — оглавление, если
   она ПРОДОЛЖАЕТ сильную (идёт сразу за ней; продолжение указателя, хвост оглавления) либо
   сама по себе не похожа на таблицу (нет крупного блока ``Table`` surya). Продолжение — только
   вперёд: заголовок указателя стоит на его первой полосе, а таблица норм отгрузки ПЕРЕД
   указателем (1970/12) продолжением не является.
3. ПАРА «шапка + выходные данные» (с 1970/04 оглавление начинается на шапке журнала и
   кончается на полосе с редколлегией): в начале выпуска сосед помеченной полосы тоже
   оглавление, если на нём стоит слово «РЕДКОЛЛЕГИЯ» (хвост бывает в одну-две строки, и
   tesseract их не всегда читает — 1972/07, 1973/11, 1974/03) либо, без слова, не меньше
   четырёх строк с номером страницы. Пара относительная, а не «полосы 3 и 4»: в 1970/10
   нет второй обложки, и оглавление стоит на полосах 2 и 3.
4. ВИД: полосы-оглавления делятся на отрезки подряд идущих. Отрезок — указатель, если его
   первая полоса несёт заголовок указателя, либо это декабрьский выпуск, отрезок стоит в
   конце и длиннее одной полосы без слова «СОДЕРЖАНИЕ» в начале. Внутри указателя полоса
   со словом «СОДЕРЖАНИЕ» и всё после неё — «Содержание» (1968/12: указатель и содержание
   подряд).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ocr_utils.scan_markup.toc import KIND_CONTENTS, KIND_INDEX, WINDOW_END, WINDOW_START
from ocr_utils.scan_markup.toc.features import PageFeatures

# Декабрьский номер: указатель за год стоит только в нём.
DECEMBER = 12


@dataclass(frozen=True)
class Thresholds:
    """Пороги решения. Значения — по пробному прогону по всему паку-1 (2 091 полоса окна):
    у оглавлений, которые пропустили и surya, и ключевые слова, ``num_tail_ratio`` от 0.23
    при 8-9 строках (вторая полоса 1971/04, шапка 1975/03); у не-оглавлений — до 0.2 при
    3 строках (рекламная полоса 1975/03); таблицы приложений с числами дают 0.3-0.65 и
    отличаются блоком ``Table`` surya площадью от 0.23 (1972/04) до 0.8."""

    # Слабый признак: не меньше стольких строк с номером страницы и не ниже такой их доли.
    weak_min_lines: int = 5
    weak_min_ratio: float = 0.2
    # Полоса «похожа на таблицу»: блоки Table surya занимают не меньше этой доли полосы.
    table_min_area: float = 0.2
    # Сильный признак surya: уверенность блока TableOfContents не ниже.
    surya_min_conf: float = 0.3
    # Ключевое слово считается только при стольких строках с номером страницы: у полосы
    # оглавления они есть всегда (хвост в 1972/07 — две), а слово «СОДЕРЖАНИЕ» капителью
    # бывает и подзаголовком в тексте.
    keyword_min_lines: int = 2
    # Пара «шапка + выходные данные»: сосед помеченной полосы в начале выпуска берётся по
    # слову «РЕДКОЛЛЕГИЯ», а без него — по стольким строкам с номером страницы. Порог высокий
    # нарочно: на второй обложке (фото с подписями) tesseract находит 2-3 такие строки
    # (1973/02, 1974/08, 1975/04), у настоящих выходных данных без слова их не бывало вовсе.
    pair_min_lines: int = 4
    # Окно кандидатов.
    window_start: int = WINDOW_START
    window_end: int = WINDOW_END

    @staticmethod
    def parse(overrides: Sequence[str]) -> "Thresholds":
        """``имя=число`` через запятую или по одному; неизвестное имя — ошибка."""
        values: dict[str, float | int] = {}
        for item in overrides:
            for part in item.split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" not in part:
                    raise ValueError(f"порог задаётся как имя=число, получено {part!r}")
                name, raw = (s.strip() for s in part.split("=", 1))
                if name not in Thresholds.__dataclass_fields__:
                    raise ValueError(f"неизвестный порог {name!r}; бывают {', '.join(Thresholds.__dataclass_fields__)}")
                kind = Thresholds.__dataclass_fields__[name].type
                values[name] = int(raw) if kind == "int" else float(raw)
        return Thresholds(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Decision:
    """Решение по полосе: вид (``None`` — не оглавление), сила и почему."""

    rel_path: str
    order_index: int
    kind: str | None
    score: float
    reason: str

    @property
    def is_toc(self) -> bool:
        return self.kind is not None


def in_window(order_index: int, idx_from_end: int, thresholds: Thresholds) -> bool:
    return order_index <= thresholds.window_start or idx_from_end <= thresholds.window_end


def strong_reason(page: PageFeatures, thresholds: Thresholds) -> str:
    """Почему полоса — оглавление наверняка; пустая строка — сильных признаков нет."""
    reasons = []
    if page.surya_toc_conf >= thresholds.surya_min_conf:
        reasons.append(f"surya {page.surya_toc_conf:.2f}")
    if page.num_tail_lines >= thresholds.keyword_min_lines:
        if page.kw_contents:
            reasons.append("«СОДЕРЖАНИЕ»")
        if page.kw_index:
            reasons.append("заголовок указателя")
    return ", ".join(reasons)


def is_weak(page: PageFeatures, thresholds: Thresholds) -> bool:
    return page.num_tail_lines >= thresholds.weak_min_lines and page.num_tail_ratio >= thresholds.weak_min_ratio


def looks_like_table(page: PageFeatures, thresholds: Thresholds) -> bool:
    return page.surya_table_area >= thresholds.table_min_area


def decide_issue(
    pages: Sequence[PageFeatures], issue_number: int | None, thresholds: Thresholds = Thresholds()
) -> list[Decision]:
    """Решения по полосам окна выпуска. ``pages`` — признаки полос ОКНА по возрастанию
    ``order_index`` (полосы вне окна можно не передавать: они не оглавление по определению)."""
    ordered = sorted(pages, key=lambda p: p.order_index)
    by_index = {p.order_index: p for p in ordered}
    marked: dict[int, tuple[float, str]] = {}  # order_index -> (score, reason)

    # 1. Сильные.
    for page in ordered:
        reason = strong_reason(page, thresholds)
        if reason:
            marked[page.order_index] = (1.0, reason)

    # 2. Слабые: продолжение сильной вперёд — или сами по себе, если не таблица.
    for page in ordered:
        if page.order_index in marked or not is_weak(page, thresholds):
            continue
        if page.order_index - 1 in marked:
            marked[page.order_index] = (page.num_tail_ratio, f"продолжение, {page.num_tail_lines} строк с номером")
        elif not looks_like_table(page, thresholds):
            marked[page.order_index] = (page.num_tail_ratio, f"{page.num_tail_lines} строк с номером, не таблица")
        # Продолжение продолжения: следующий слабый увидит эту полосу в marked на своём шаге.

    # 3. Пара «шапка + выходные данные» — только в начале выпуска.
    for mine in sorted(marked):
        if mine > thresholds.window_start:
            continue
        for other in (mine - 1, mine + 1):
            if other in marked or other not in by_index or other > thresholds.window_start:
                continue
            page = by_index[other]
            if page.kw_imprint:
                marked[other] = (0.5, f"пара к полосе {mine}: выходные данные")
            elif page.num_tail_lines >= thresholds.pair_min_lines:
                marked[other] = (
                    max(page.num_tail_ratio, 0.5),
                    f"пара к полосе {mine}: {page.num_tail_lines} строк с номером",
                )

    # 4. Вид по отрезкам.
    kinds = _assign_kinds(ordered, marked, issue_number, thresholds)

    decisions = []
    for page in ordered:
        if page.order_index in marked:
            score, reason = marked[page.order_index]
            decisions.append(Decision(page.rel_path, page.order_index, kinds[page.order_index], score, reason))
        else:
            decisions.append(Decision(page.rel_path, page.order_index, None, page.num_tail_ratio, ""))
    return decisions


def _runs(indices: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in sorted(indices):
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _assign_kinds(
    ordered: Sequence[PageFeatures], marked: dict[int, tuple[float, str]], issue_number: int | None, thr: Thresholds
) -> dict[int, str]:
    by_index = {p.order_index: p for p in ordered}
    kinds: dict[int, str] = {}
    for run in _runs(list(marked)):
        head = by_index[run[0]]
        at_end = all(by_index[i].idx_from_end <= thr.window_end for i in run)
        index_like = head.kw_index and not head.kw_contents
        # Декабрьский отрезок в конце без заголовка указателя: указатель, пока не встретится
        # «СОДЕРЖАНИЕ» (1968/12 — указатель и содержание подряд). Если «СОДЕРЖАНИЕ» на первой
        # же полосе — это содержание на две полосы, а не указатель.
        december_index = issue_number == DECEMBER and at_end and len(run) > 1 and not head.kw_contents
        kind = KIND_INDEX if index_like or december_index else KIND_CONTENTS
        for index in run:
            if kind == KIND_INDEX and by_index[index].kw_contents:
                kind = KIND_CONTENTS  # с этой полосы указатель кончился и пошло содержание
            kinds[index] = kind
    return kinds


__all__ = ["DECEMBER", "Decision", "Thresholds", "decide_issue", "in_window", "strong_reason"]
