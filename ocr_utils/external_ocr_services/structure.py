"""Пост-обработка markdown полосы: `#` только из оглавления, авторы при своей статье, рубрика перед `#`.

Модель следует правилам промпта нестрого, а будущая сборка выпуска и нарезка по статьям
опираются ровно на эти три вещи: `#` = граница статьи из оглавления, `<author>` сразу после
`#` своей статьи (или под последним абзацем той, что кончается), `<rubric>` перед `#`. Поэтому
после разбора ответа они доводятся кодом — детерминированно и идемпотентно; что изменено,
пишется в .meta.json.

Сопоставление заголовка со списком — нормализация как в ``toc.normalize_title`` плюс
``difflib`` (оглавление сокращает и переставляет слова, регистр и переносы различаются).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from ocr_utils.external_ocr_services.toc import normalize_title

# Порог похожести названия из списка и заголовка на полосе (по нормализованным строкам).
TITLE_MATCH_RATIO = 0.75

_H1 = re.compile(r"^# (.+?)\s*$")
_H2 = re.compile(r"^## (.+?)\s*$")
_AUTHOR = re.compile(r"^(?:- )?<author>\*{0,2}(.+?)\*{0,2}</author>")
_POSITION = re.compile(r"^<position>")
_RUBRIC = re.compile(r"^<rubric>\*{0,2}(.+?)\*{0,2}</rubric>\s*$")
_MARKER = re.compile(r"^<marker>")
_MARKER_TEXT = re.compile(r"^<marker>\*{0,2}(.+?)\*{0,2}</marker>\s*$")


@dataclass
class StructureReport:
    """Что изменила пост-обработка — попадает в .meta.json."""

    demoted_headings: list[str] = field(default_factory=list)
    moved_authors: list[str] = field(default_factory=list)
    dropped_authors: list[str] = field(default_factory=list)
    rubrics_from_headings: list[str] = field(default_factory=list)
    # Рубрики по оглавлению: перенесённые к `#`, вставленные из оглавления, заменённые, и
    # теги `<rubric>` не из оглавления, ставшие `<marker>`; `markers` — сколько `<marker>` в итоге.
    moved_rubrics: list[str] = field(default_factory=list)
    rubrics_from_toc: list[str] = field(default_factory=list)
    rubrics_replaced: list[str] = field(default_factory=list)
    markers_from_rubrics: list[str] = field(default_factory=list)
    markers: int = 0
    # `##` в начале страницы-продолжения с текстом названия из оглавления: на странице его нет,
    # модель повторила название из списка — убирается.
    dropped_headings: list[str] = field(default_factory=list)
    # Формулы в долларах без тега <latex>, обёрнутые кодом.
    wrapped_math: int = 0
    title_in_list: bool | None = None

    def as_dict(self) -> dict:
        return {
            "demoted_headings": self.demoted_headings,
            "moved_authors": self.moved_authors,
            "dropped_authors": self.dropped_authors,
            "rubrics_from_headings": self.rubrics_from_headings,
            "moved_rubrics": self.moved_rubrics,
            "rubrics_from_toc": self.rubrics_from_toc,
            "rubrics_replaced": self.rubrics_replaced,
            "markers_from_rubrics": self.markers_from_rubrics,
            "markers": self.markers,
            "dropped_headings": self.dropped_headings,
            "wrapped_math": self.wrapped_math,
        }


def title_matches(heading: str, titles: list[str]) -> bool:
    """Совпадает ли заголовок с одним из названий: равенство, вхождение или похожесть ≥ порога."""
    key = normalize_title(heading)
    if not key:
        return False
    for title in titles:
        other = normalize_title(title)
        if not other:
            continue
        if key == other or key in other or other in key:
            return True
        if difflib.SequenceMatcher(None, key, other).ratio() >= TITLE_MATCH_RATIO:
            return True
    return False


def _paragraphs(body: str) -> list[str]:
    """Тело как список абзацев (разделитель — пустая строка); переносы внутри абзаца сохраняются."""
    return [block for block in re.split(r"\n\s*\n", body.strip()) if block.strip()]


def _join(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs).rstrip() + "\n"


def demote_unlisted_headings(body: str, titles: list[str], report: StructureReport) -> str:
    """`# …`, не совпавший ни с одним названием из списка, → `## …`. Пустой список — без изменений."""
    if not titles:
        return body
    lines = body.split("\n")
    for index, line in enumerate(lines):
        match = _H1.match(line)
        if match is None:
            continue
        if title_matches(match.group(1), titles):
            report.title_in_list = True
            continue
        lines[index] = "#" + line
        report.demoted_headings.append(match.group(1))
    if report.title_in_list is None and report.demoted_headings:
        report.title_in_list = False
    return "\n".join(lines)


def normalize_name(name: str) -> str:
    """Ключ сопоставления имени автора: без регистра, точек, запятых и пробелов между инициалами."""
    return re.sub(r"[\s.,*]+", "", name.replace("ё", "е")).lower()


def _author_block(paragraphs: list[str], index: int) -> tuple[int, int]:
    """Границы блока автора: сам абзац `<author>` и все `<position>` сразу за ним."""
    end = index + 1
    while end < len(paragraphs) and _POSITION.match(paragraphs[end]):
        end += 1
    return index, end


def place_authors(body: str, authors: list[dict], report: StructureReport) -> str:
    """Расставить блоки `<author>` по статьям.

    * `article == "starts_here"` и блок стоит перед `#` (между ними допускается `<rubric>`) →
      блок переносится сразу после этого `#`;
    * блок автора — первый абзац тела (текста перед ним нет) и сразу за ним (через рубрику)
      идёт `#` → подписью предыдущей статьи он быть не может: переносится после `#`, если
      только модель не пометила его `ends_here` явно (подпись, перетёкшая с прошлой полосы);
    * `article == "continues"` и `printed == "running_header"` → блок удаляется из тела (имя
      остаётся в JSON и в `running_header`);
    * `ends_here` и всё остальное не трогается.
    """
    paragraphs = _paragraphs(body)
    if not paragraphs:
        return body
    by_name: dict[str, dict] = {normalize_name(a.get("name", "")): a for a in authors if a.get("name")}

    def author_of(paragraph: str) -> dict | None:
        match = _AUTHOR.match(paragraph)
        if match is None or paragraph.startswith("- "):
            return None  # пункт списка (оглавление) — не подпись
        key = normalize_name(match.group(1))
        if key in by_name:
            return by_name[key]
        # Несколько имён в одном теге или имя с должностью после запятой.
        for name, author in by_name.items():
            if name and name in key:
                return author
        return {"name": match.group(1), "article": None, "printed": None}

    changed = True
    while changed:
        changed = False
        for index, paragraph in enumerate(paragraphs):
            author = author_of(paragraph)
            if author is None:
                continue
            start, end = _author_block(paragraphs, index)
            block = paragraphs[start:end]
            role, printed = author.get("article"), author.get("printed")
            if role == "continues" and printed == "running_header":
                del paragraphs[start:end]
                report.dropped_authors.append(author["name"])
                changed = True
                break
            # Ближайший `#` после блока, через рубрику; если между ними текст — не «перед #».
            cursor = end
            while cursor < len(paragraphs) and _RUBRIC.match(paragraphs[cursor]):
                cursor += 1
            before_heading = cursor < len(paragraphs) and _H1.match(paragraphs[cursor]) is not None
            first_in_body = start == 0
            if before_heading and (role == "starts_here" or (first_in_body and role != "ends_here")):
                del paragraphs[start:end]
                heading_at = cursor - len(block)
                paragraphs[heading_at + 1 : heading_at + 1] = block
                report.moved_authors.append(author["name"])
                changed = True
                break
    return _join(paragraphs)


def rubric_from_heading(body: str, rubrics: list[str], report: StructureReport) -> str:
    """`## X` непосредственно перед `#`, если X — рубрика из списка, → `<rubric>*X*</rubric>`."""
    if not rubrics:
        return body
    paragraphs = _paragraphs(body)
    keys = {normalize_title(r) for r in rubrics if r}
    for index in range(len(paragraphs) - 1):
        match = _H2.match(paragraphs[index])
        if match is None or _H1.match(paragraphs[index + 1]) is None:
            continue
        text = match.group(1).strip("*_ ")
        if normalize_title(text) in keys or title_matches(text, list(rubrics)):
            paragraphs[index] = f"<rubric>*{text}*</rubric>"
            report.rubrics_from_headings.append(text)
    return _join(paragraphs)


def _rubric_tag(text: str) -> str:
    return f"<rubric>*{text}*</rubric>"


def _marker_tag(text: str) -> str:
    return f"<marker>*{text}*</marker>"


def _expected_rubrics(paragraphs: list[str], articles: list[dict]) -> dict[int, str | None]:
    """Индекс каждого `#`, совпавшего со статьёй списка → её рубрика по оглавлению (или None)."""
    expected: dict[int, str | None] = {}
    for index, paragraph in enumerate(paragraphs):
        match = _H1.match(paragraph)
        if match is None:
            continue
        for article in articles:
            if title_matches(match.group(1), [article["title"]]):
                expected[index] = article.get("rubric") or None
                break
    return expected


def place_rubrics(body: str, articles: list[dict], report: StructureReport, rubrics: list[str] | None = None) -> str:
    """Рубрика — только из оглавления и только перед `#` своей статьи; остальное — `<marker>`.

    Оглавление — источник истины: у статьи из списка рубрика известна (``articles[i]["rubric"]``),
    и перед её `#` должен стоять ровно один `<rubric>` с этой рубрикой, где бы маркер ни был
    напечатан на полосе. Код ничего не удаляет — только переносит и перетегирует:

    * `<rubric>` с текстом не из оглавления → `<marker>` на месте (страховка от промпта);
    * `<rubric>` из оглавления не перед `#` → переносится перед `#` статьи с той же рубрикой на
      этой полосе; такой статьи нет (продолжение) → `<marker>` на месте;
    * перед `#` с известной рубрикой нет тега → вставляется из оглавления; тег есть, но текст
      другой → заменяется рубрикой оглавления, напечатанный вариант остаётся `<marker>`-ом
      сразу после `#`.
    Без списка статей ничего не меняется.
    """
    if not articles:
        return body
    paragraphs = _paragraphs(body)
    if not paragraphs:
        return body
    known = [article["rubric"] for article in articles if article.get("rubric")] + [r for r in (rubrics or []) if r]

    # 1. Теги не из оглавления → маркеры.
    for index, paragraph in enumerate(paragraphs):
        match = _RUBRIC.match(paragraph)
        if match is not None and not title_matches(match.group(1), known):
            paragraphs[index] = _marker_tag(match.group(1))
            report.markers_from_rubrics.append(match.group(1))

    # 2. Рубрики не перед `#` — к своей статье или в маркеры. С конца, чтобы индексы не плыли.
    expected = _expected_rubrics(paragraphs, articles)

    # «Бесхозный» тег: не перед `#` — или перед `#` статьи с другой рубрикой по оглавлению
    # (маркер между двумя статьями относится к той, чья рубрика совпадает).
    def stray(index: int) -> bool:
        match = _RUBRIC.match(paragraphs[index])
        if match is None:
            return False
        following = index + 1
        if following >= len(paragraphs) or _H1.match(paragraphs[following]) is None:
            return True
        expected_here = expected.get(following)
        return bool(expected_here) and not title_matches(match.group(1), [expected_here])

    strays = [index for index in range(len(paragraphs)) if stray(index)]
    for index in reversed(strays):
        text = _RUBRIC.match(paragraphs[index]).group(1)
        targets = [h for h, rubric in expected.items() if rubric and title_matches(text, [rubric])]
        # Заголовок, перед которым рубрики ещё нет; ближайший выше маркера, иначе первый ниже.
        free = [h for h in targets if not (h > 0 and _RUBRIC.match(paragraphs[h - 1]))]
        above = [h for h in free if h < index]
        target = (above[-1] if above else free[0]) if free else None
        if target is None:
            paragraphs[index] = _marker_tag(text)
            report.markers_from_rubrics.append(text)
            continue
        paragraph = paragraphs.pop(index)
        if index < target:
            target -= 1
        paragraphs.insert(target, paragraph)
        report.moved_rubrics.append(text)
        expected = _expected_rubrics(paragraphs, articles)

    # 3. Перед каждым `#` с известной рубрикой — ровно она. Если модель написала маркер с текстом
    # этой рубрики (маркер в углу страницы), он и становится рубрикой — переносится к `#`.
    expected = _expected_rubrics(paragraphs, articles)
    for index in sorted(expected, reverse=True):
        rubric = expected[index]
        if not rubric:
            continue
        previous = paragraphs[index - 1] if index > 0 else ""
        match = _RUBRIC.match(previous)
        if match is None:
            marker_at = next(
                (
                    i
                    for i, paragraph in enumerate(paragraphs)
                    if (m := _MARKER_TEXT.match(paragraph)) and title_matches(m.group(1), [rubric])
                ),
                None,
            )
            if marker_at is not None:
                text = _MARKER_TEXT.match(paragraphs.pop(marker_at)).group(1)
                if marker_at < index:
                    index -= 1
                paragraphs.insert(index, _rubric_tag(text))
                report.moved_rubrics.append(text)
            else:
                paragraphs.insert(index, _rubric_tag(rubric))
                report.rubrics_from_toc.append(rubric)
        elif not title_matches(match.group(1), [rubric]):
            printed = match.group(1)
            paragraphs[index - 1] = _rubric_tag(rubric)
            paragraphs.insert(index + 1, _marker_tag(printed))
            report.rubrics_replaced.append(printed)
    return _join(paragraphs)


def drop_leaked_titles(body: str, titles: list[str], report: StructureReport) -> str:
    """Убрать `##` с названием статьи из списка в начале страницы без `#`.

    На странице-продолжении заголовка статьи нет; если модель начала её с `## <название из
    списка>`, это утечка списка из промпта, а не напечатанный текст (МТС 1991/02, с. 95).
    """
    if not titles:
        return body
    paragraphs = _paragraphs(body)
    if any(_H1.match(p) for p in paragraphs):
        return body
    for index, paragraph in enumerate(paragraphs[:3]):
        match = _H2.match(paragraph)
        if match is not None and title_matches(match.group(1), titles):
            del paragraphs[index]
            report.dropped_headings.append(match.group(1))
            return _join(paragraphs)
        if not (_RUBRIC.match(paragraph) or _MARKER.match(paragraph)):
            break
    return body


_LATEX = re.compile(r"<latex>.*?</latex>", re.S)
# Голая формула в долларах вне тега: сначала $$…$$, потом $…$ без переносов строк внутри.
_BARE_MATH = re.compile(r"\$\$[^$]+?\$\$|\$(?!\$)[^$\n]+?\$")


def wrap_bare_math(body: str, report: StructureReport) -> str:
    """Формулы в долларах без тега `<latex>` — обернуть: модель на части полос ставит одни доллары."""
    pieces: list[str] = []
    cursor = 0
    for tagged in _LATEX.finditer(body):
        pieces.append(_wrap_segment(body[cursor : tagged.start()], report))
        pieces.append(tagged.group(0))
        cursor = tagged.end()
    pieces.append(_wrap_segment(body[cursor:], report))
    return "".join(pieces)


def _wrap_segment(text: str, report: StructureReport) -> str:
    def wrap(match: re.Match) -> str:
        report.wrapped_math += 1
        return f"<latex>{match.group(0)}</latex>"

    return _BARE_MATH.sub(wrap, text)


def apply(body: str, articles: list, rubrics: list[str], authors: list[dict]) -> tuple[str, StructureReport]:
    """Все правки по порядку: рубрики из `##`, понижение `#`, расстановка авторов, рубрики по оглавлению.

    ``articles`` — ``[{"title", "authors", "rubric"}]`` из оглавления (или просто названия строками).
    """
    articles = [{"title": item} if isinstance(item, str) else dict(item) for item in articles]
    titles = [article["title"] for article in articles if article.get("title")]
    report = StructureReport()
    body = rubric_from_heading(body, rubrics, report)
    body = demote_unlisted_headings(body, titles, report)
    body = drop_leaked_titles(body, titles, report)
    body = place_authors(body, authors, report)
    body = place_rubrics(body, articles, report, rubrics)
    body = wrap_bare_math(body, report)
    report.markers = body.count("<marker>")
    return body, report
