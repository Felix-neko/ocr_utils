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

import re
from dataclasses import dataclass, field

from ocr_utils.external_ocr_services.schema import FORMULA_TAG, AuthorArticle, AuthorPrinted, StructureTag
from ocr_utils.external_ocr_services.toc import (  # noqa: F401
    TITLE_MATCH_RATIO,
    best_title_match,
    normalize_title,
    rubric_titles,
    title_matches,
)

# Абзацы, которые различает пост-обработка. Теги допускают курсив/жирный внутри (`<author>**И. Иванов**</author>`).
_H1 = re.compile(r"^# (.+?)\s*$")
_H2 = re.compile(r"^## (.+?)\s*$")
_AUTHOR = re.compile(rf"^(?:- )?<{StructureTag.AUTHOR}>\*{{0,2}}(.+?)\*{{0,2}}</{StructureTag.AUTHOR}>")
_POSITION = re.compile(rf"^<{StructureTag.POSITION}>")
_RUBRIC = re.compile(rf"^<{StructureTag.RUBRIC}>\*{{0,2}}(.+?)\*{{0,2}}</{StructureTag.RUBRIC}>\s*$")
_MARKER = re.compile(rf"^<{StructureTag.MARKER}>")
_MARKER_TEXT = re.compile(rf"^<{StructureTag.MARKER}>\*{{0,2}}(.+?)\*{{0,2}}</{StructureTag.MARKER}>\s*$")


@dataclass
class StructureReport:
    """Что изменила пост-обработка — попадает в .meta.json (``structure``).

    Args:
        demoted_headings: Заголовки ``#``, не найденные в оглавлении и пониженные до ``##``.
        moved_authors: Имена авторов, чьи блоки перенесены после ``#`` своей статьи.
        dropped_authors: Имена из колонтитула на полосе-продолжении, убранные из тела.
        rubrics_from_headings: Тексты ``##`` перед ``#``, оказавшиеся рубриками из списка и ставшие ``<rubric>``.
        moved_rubrics: Рубрики (и маркеры с текстом рубрики), перенесённые к ``#`` своей статьи.
        rubrics_replaced: Напечатанные варианты, заменённые рубрикой из оглавления (сами стали ``<marker>``).
        markers_from_rubrics: Тексты ``<rubric>`` не из оглавления, перетегированные в ``<marker>``.
        header_rubrics_dropped: ``<rubric>``/``<marker>`` с текстом колонтитула полосы, убранные из тела
            (рубрика напечатана только в колонтитуле; она остаётся в поле ``running_header``).
        markers: Сколько ``<marker>`` в итоге (и от модели, и от пост-обработки).
        dropped_headings: ``##`` в начале полосы-продолжения с названием из списка — утечка промпта, убраны.
        headings_restored: `#`, вставленные из ``headings``/``title`` ответа: модель назвала заголовок в
            поле, но в тело не написала (1966/03 с. 69 «Важная служба»).
        wrapped_math: Сколько формул в голых долларах обёрнуто в ``<latex>``.
        heading_ids_model: Сколько ``#`` получили id статьи от модели (id согласован с текстом).
        heading_ids_title: Сколько ``#`` получили id по совпадению названия (модель id не дала).
        heading_ids_wrong: Сколько id модели противоречили тексту ``#`` (заменены по названию).
        rubric_ids_model: То же для ``<rubric>`` — id от модели.
        rubric_ids_title: ``<rubric>`` с id по тексту.
        rubric_ids_wrong: ``<rubric>`` с противоречивым id модели.
        title_in_list: Совпал ли хоть один ``#`` с оглавлением; ``None`` — ``#`` на полосе нет или списка не было.
    """

    demoted_headings: list[str] = field(default_factory=list)
    moved_authors: list[str] = field(default_factory=list)
    dropped_authors: list[str] = field(default_factory=list)
    rubrics_from_headings: list[str] = field(default_factory=list)
    moved_rubrics: list[str] = field(default_factory=list)
    rubrics_replaced: list[str] = field(default_factory=list)
    markers_from_rubrics: list[str] = field(default_factory=list)
    header_rubrics_dropped: list[str] = field(default_factory=list)
    markers: int = 0
    dropped_headings: list[str] = field(default_factory=list)
    headings_restored: list[str] = field(default_factory=list)
    wrapped_math: int = 0
    heading_ids_model: int = 0
    heading_ids_title: int = 0
    heading_ids_wrong: int = 0
    rubric_ids_model: int = 0
    rubric_ids_title: int = 0
    rubric_ids_wrong: int = 0
    title_in_list: bool | None = None

    def as_dict(self) -> dict:
        """Поля для .meta.json — все, кроме ``title_in_list`` (он уходит в ``PageResult``)."""
        return {
            "demoted_headings": self.demoted_headings,
            "moved_authors": self.moved_authors,
            "dropped_authors": self.dropped_authors,
            "rubrics_from_headings": self.rubrics_from_headings,
            "moved_rubrics": self.moved_rubrics,
            "rubrics_replaced": self.rubrics_replaced,
            "markers_from_rubrics": self.markers_from_rubrics,
            "header_rubrics_dropped": self.header_rubrics_dropped,
            "markers": self.markers,
            "dropped_headings": self.dropped_headings,
            "headings_restored": self.headings_restored,
            "wrapped_math": self.wrapped_math,
            "heading_ids": {
                "model": self.heading_ids_model,
                "title": self.heading_ids_title,
                "wrong": self.heading_ids_wrong,
            },
            "rubric_ids": {
                "model": self.rubric_ids_model,
                "title": self.rubric_ids_title,
                "wrong": self.rubric_ids_wrong,
            },
        }


# Строка-ограничитель fenced-блока (иллюстрации): между парой таких строк пустые строки не делят абзацы.
_FENCE_LINE = re.compile(r"^[ \t]*```")


def paragraphs_of(body: str) -> list[str]:
    """Тело как список абзацев (разделитель — пустая строка); переносы внутри абзаца сохраняются.

    Fenced-блок иллюстрации (``…``` … ```…``) — всегда один абзац, даже если модель оставила внутри
    пустые строки: иначе правки ниже могли бы вставить рубрику или автора внутрь блока.

    Args:
        body: Тело полосы в markdown.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    inside_fence = False
    for line in body.strip().split("\n"):
        if _FENCE_LINE.match(line):
            inside_fence = not inside_fence
        if not line.strip() and not inside_fence:
            # Пустая строка вне fenced-блока — граница абзаца.
            if current:
                paragraphs.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append("\n".join(current))
    return [block for block in paragraphs if block.strip()]


def join_paragraphs(paragraphs: list[str]) -> str:
    """Обратно в текст: абзацы через пустую строку, один перевод строки в конце.

    Args:
        paragraphs: Абзацы из ``paragraphs_of`` после правок.
    """
    return "\n\n".join(paragraphs).rstrip() + "\n"


def demote_unlisted_headings(body: str, titles: list[str], report: StructureReport) -> str:
    """`# …`, не совпавший ни с одним названием из списка, → `## …`. Пустой список — без изменений.

    Args:
        body: Тело полосы в markdown.
        titles: Названия статей из оглавления выпуска.
        report: Отчёт пост-обработки — сюда пишутся понижённые заголовки и ``title_in_list``.
    """
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
        lines[index] = "#" + line  # `# ` → `## `
        report.demoted_headings.append(match.group(1))
    # Были `#`, и ни один не совпал — заголовок точно не из списка.
    if report.title_in_list is None and report.demoted_headings:
        report.title_in_list = False
    return "\n".join(lines)


def normalize_name(name: str) -> str:
    """Ключ сопоставления имени автора: без регистра, точек, запятых и пробелов между инициалами.

    Args:
        name: Имя как в теге ``<author>`` или в поле ``authors`` ответа.
    """
    return re.sub(r"[\s.,*]+", "", name.replace("ё", "е")).lower()


def _author_block(paragraphs: list[str], index: int) -> tuple[int, int]:
    """Границы блока автора: сам абзац `<author>` и все `<position>` сразу за ним.

    Args:
        paragraphs: Абзацы тела.
        index: Индекс абзаца с ``<author>``.

    Returns:
        ``(start, end)`` — срез ``paragraphs[start:end]`` с блоком; ``end`` не включается.
    """
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

    Args:
        body: Тело полосы в markdown.
        authors: Поле ``authors`` ответа: ``[{"name", "position", "article", "printed"}]`` — по нему
            определяется, чья подпись и где напечатана.
        report: Отчёт пост-обработки — перенесённые и убранные имена.
    """
    paragraphs = paragraphs_of(body)
    if not paragraphs:
        return body
    # Авторы из ответа по ключу имени: блок в тексте сопоставляется с записью и её полями.
    by_name: dict[str, dict] = {normalize_name(a.get("name", "")): a for a in authors if a.get("name")}

    # Каждое перемещение сдвигает индексы — после него обход начинается заново, пока правок нет.
    changed = True
    while changed:
        changed = False
        for index, paragraph in enumerate(paragraphs):
            author = _author_of(paragraph, by_name)
            if author is None:
                continue
            start, end = _author_block(paragraphs, index)
            block = paragraphs[start:end]
            role, printed = author.get("article"), author.get("printed")
            # Имя из колонтитула на полосе-продолжении — в теле ему не место.
            if role == AuthorArticle.CONTINUES and printed == AuthorPrinted.RUNNING_HEADER:
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
            if before_heading and (
                role == AuthorArticle.STARTS_HERE or (first_in_body and role != AuthorArticle.ENDS_HERE)
            ):
                del paragraphs[start:end]
                heading_at = cursor - len(block)
                paragraphs[heading_at + 1 : heading_at + 1] = block
                report.moved_authors.append(author["name"])
                changed = True
                break
    return join_paragraphs(paragraphs)


def _author_of(paragraph: str, by_name: dict[str, dict]) -> dict | None:
    """Запись автора из ответа для абзаца ``<author>``.

    Args:
        paragraph: Абзац тела.
        by_name: Записи ``authors`` ответа по ключу ``normalize_name``.

    Returns:
        Запись ``{"name", "position", "article", "printed"}`` из ответа; для имени, которого в
        ответе нет, — заглушка без привязки (``article``/``printed`` = ``None``); ``None`` — абзац
        не подпись автора (нет тега или это пункт списка оглавления).
    """
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


def rubric_from_heading(body: str, rubrics: list[str], report: StructureReport) -> str:
    """`## X` непосредственно перед `#`, если X — рубрика из списка, → `<rubric>*X*</rubric>`.

    Args:
        body: Тело полосы в markdown.
        rubrics: Рубрики «Содержания» выпуска.
        report: Отчёт пост-обработки — тексты превращённых заголовков.
    """
    if not rubrics:
        return body
    paragraphs = paragraphs_of(body)
    keys = {normalize_title(r) for r in rubrics if r}
    for index in range(len(paragraphs) - 1):
        match = _H2.match(paragraphs[index])
        if match is None or _H1.match(paragraphs[index + 1]) is None:
            continue
        text = match.group(1).strip("*_ ")
        if normalize_title(text) in keys or title_matches(text, list(rubrics)):
            paragraphs[index] = f"<rubric>*{text}*</rubric>"
            report.rubrics_from_headings.append(text)
    return join_paragraphs(paragraphs)


def _rubric_tag(text: str) -> str:
    """Абзац рубрики в принятой разметке: тег с курсивом внутри."""
    return f"<{StructureTag.RUBRIC}>*{text}*</{StructureTag.RUBRIC}>"


def _marker_tag(text: str) -> str:
    """Абзац маркера (текст вне потока статьи) в принятой разметке."""
    return f"<{StructureTag.MARKER}>*{text}*</{StructureTag.MARKER}>"


def _expected_rubrics(paragraphs: list[str], articles: list[dict]) -> dict[int, str | None]:
    """Индекс каждого `#`, совпавшего со статьёй списка → её рубрика по оглавлению (или None).

    Args:
        paragraphs: Абзацы тела.
        articles: Статьи оглавления ``[{"title", "rubric"}]``.

    Returns:
        ``{индекс абзаца с "#": рубрика}``; ``None`` — статья в оглавлении без рубрики. `#`, не
        совпавших ни с одной статьёй, в словаре нет.
    """
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


def _is_stray_rubric(paragraphs: list[str], expected: dict[int, str | None], index: int) -> bool:
    """«Бесхозный» ли тег ``<rubric>``: не перед `#` — или перед `#` статьи с другой рубрикой.

    Маркер между двумя статьями относится к той, чья рубрика по оглавлению совпадает с его текстом.

    Args:
        paragraphs: Абзацы тела.
        expected: Рубрики по оглавлению для каждого `#` (``_expected_rubrics``).
        index: Индекс проверяемого абзаца.

    Returns:
        ``True`` — тег надо переносить к своей статье или превращать в ``<marker>``; ``False`` —
        абзац не рубрика или рубрика стоит на своём месте.
    """
    match = _RUBRIC.match(paragraphs[index])
    if match is None:
        return False
    following = index + 1
    if following >= len(paragraphs) or _H1.match(paragraphs[following]) is None:
        return True
    expected_here = expected.get(following)
    return bool(expected_here) and not title_matches(match.group(1), [expected_here])


def place_rubrics(body: str, articles: list[dict], report: StructureReport, rubrics: list[str] | None = None) -> str:
    """Рубрика — только из оглавления и только перед `#` своей статьи; остальное — `<marker>`.

    Оглавление — источник истины: у статьи из списка рубрика известна (``articles[i]["rubric"]``),
    и перед её `#` должен стоять ровно один `<rubric>` с этой рубрикой, где бы маркер ни был
    напечатан на полосе. Код ничего не удаляет — только переносит и перетегирует:

    * `<rubric>` с текстом не из оглавления → `<marker>` на месте (страховка от промпта);
    * `<rubric>` из оглавления не перед `#` → переносится перед `#` статьи с той же рубрикой на
      этой полосе; такой статьи нет (продолжение) → `<marker>` на месте;
    * перед `#` с известной рубрикой тег есть, но текст другой → заменяется рубрикой оглавления,
      напечатанный вариант остаётся `<marker>`-ом сразу после `#`; тега и маркера нет — рубрика
      не напечатана, ничего не вставляется (рубрика статьи известна из toc.json).
    Без списка статей ничего не меняется.

    Args:
        body: Тело полосы в markdown.
        articles: Статьи оглавления ``[{"title", "authors", "rubric"}]`` — источник истины о рубриках.
        report: Отчёт пост-обработки — что перенесено, вставлено, заменено, перетегировано.
        rubrics: Список рубрик выпуска целиком (в дополнение к рубрикам статей) — чтобы тег с
            рубрикой, у которой на этой полосе нет статьи, не считался «не из оглавления».
    """
    if not articles:
        return body
    paragraphs = paragraphs_of(body)
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
    strays = [index for index in range(len(paragraphs)) if _is_stray_rubric(paragraphs, expected, index)]
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
            # Ни тега, ни маркера с текстом рубрики на полосе нет — рубрика не напечатана, ничего не
            # вставляется: до 19.09.2026 сюда подставлялась рубрика из оглавления, и `<rubric>`
            # вставал над каждой статьёй (0120_1L 1976/12); рубрика статьи и так есть в toc.json.
        elif not title_matches(match.group(1), [rubric]):
            printed = match.group(1)
            paragraphs[index - 1] = _rubric_tag(rubric)
            paragraphs.insert(index + 1, _marker_tag(printed))
            report.rubrics_replaced.append(printed)
    return join_paragraphs(paragraphs)


# Абзацы, над которыми восстановленный `#` не ставится: ведущие рубрика/маркер (идут перед `#`),
# иллюстрация (fenced-блок) и сноска — плавающие блоки в начале полосы.
_ABOVE_HEADING = re.compile(rf"^(?:<{StructureTag.RUBRIC}>|<{StructureTag.MARKER}>|```|<footnote>|\[\^\w+\]:)")


def restore_heading_from_refs(body: str, model_headings: list[dict], titles: list[str], report: StructureReport) -> str:
    """`#` из ``headings`` (до v20 — ``title``) ответа, если модель назвала заголовок, но в тело его не написала.

    Модель изредка перечисляет заголовок в поле и опускает строку `#` в теле (1966/03 с. 69: в
    ``headings`` «Важная служба» с верным id, автор ``starts_here`` над названием, а тело начинается
    с первого абзаца). Сверка на сборке видит только `#` в теле, поэтому строка вставляется здесь —
    первым содержательным абзацем: после ведущих рубрики, маркера, иллюстрации, сноски; блок
    ``<author>`` не пропускается — ``place_authors`` переставит его под `#`. При списке статей
    текст должен совпасть с названием из списка (иначе это мог быть подзаголовок), без списка
    берётся как есть. Ничего не делается, если в теле уже есть хоть один `#`.

    Args:
        body: Тело полосы после понижения чужих `#`.
        model_headings: Записи ``headings`` ответа ``[{"text", "article_id"}]`` (из ``title`` до v20).
        titles: Названия статей из оглавления; пусто — списка не было.
        report: Отчёт — вставленные заголовки в ``headings_restored``.
    """
    paragraphs = paragraphs_of(body)
    if not paragraphs or any(_H1.match(paragraph) for paragraph in paragraphs):
        return body
    text = next(
        (
            str(ref.get("text")).strip()
            for ref in model_headings
            if ref.get("text") and (not titles or title_matches(str(ref["text"]), titles))
        ),
        None,
    )
    if text is None:
        return body
    at = 0
    while at < len(paragraphs) and _ABOVE_HEADING.match(paragraphs[at]):
        at += 1
    paragraphs.insert(at, f"# {text}")
    report.headings_restored.append(text)
    if titles:
        report.title_in_list = True  # вставлен заголовок из списка — понижение до него `#` не видело
    return join_paragraphs(paragraphs)


def drop_leaked_titles(body: str, titles: list[str], report: StructureReport) -> str:
    """Убрать `##` с названием статьи из списка в начале страницы без `#`.

    На странице-продолжении заголовка статьи нет; если модель начала её с `## <название из
    списка>`, это утечка списка из промпта, а не напечатанный текст (МТС 1991/02, с. 95).

    Args:
        body: Тело полосы в markdown.
        titles: Названия статей из оглавления.
        report: Отчёт пост-обработки — убранные заголовки.
    """
    if not titles:
        return body
    paragraphs = paragraphs_of(body)
    if any(_H1.match(p) for p in paragraphs):  # есть `#` — полоса не продолжение, `##` законны
        return body
    # Смотрим только первые три абзаца: перед утёкшим названием могут стоять рубрика и маркер.
    for index, paragraph in enumerate(paragraphs[:3]):
        match = _H2.match(paragraph)
        if match is not None and title_matches(match.group(1), titles):
            del paragraphs[index]
            report.dropped_headings.append(match.group(1))
            return join_paragraphs(paragraphs)
        if not (_RUBRIC.match(paragraph) or _MARKER.match(paragraph)):
            break
    return body


_LATEX = re.compile(rf"<{FORMULA_TAG}>.*?</{FORMULA_TAG}>", re.S)
# Голая формула в долларах вне тега: сначала $$…$$, потом $…$ без переносов строк внутри.
_BARE_MATH = re.compile(r"\$\$[^$]+?\$\$|\$(?!\$)[^$\n]+?\$")


def wrap_bare_math(body: str, report: StructureReport) -> str:
    """Формулы в долларах без тега `<latex>` — обернуть: модель на части полос ставит одни доллары.

    Args:
        body: Тело полосы в markdown.
        report: Отчёт пост-обработки — число обёрнутых формул.
    """
    # Текст режется на куски «между уже обёрнутыми формулами»: внутри <latex> доллары не трогаем.
    pieces: list[str] = []
    cursor = 0
    for tagged in _LATEX.finditer(body):
        pieces.append(_wrap_segment(body[cursor : tagged.start()], report))
        pieces.append(tagged.group(0))
        cursor = tagged.end()
    pieces.append(_wrap_segment(body[cursor:], report))
    return "".join(pieces)


def _wrap_segment(text: str, report: StructureReport) -> str:
    """Обернуть все голые формулы в куске текста без тегов ``<latex>``.

    Args:
        text: Кусок тела между уже обёрнутыми формулами.
        report: Отчёт — счётчик ``wrapped_math``.

    Returns:
        Тот же кусок, где каждая ``$…$`` / ``$$…$$`` обёрнута в ``<latex>…</latex>``.
    """
    formulas = _BARE_MATH.findall(text)
    report.wrapped_math += len(formulas)
    return _BARE_MATH.sub(rf"<{FORMULA_TAG}>\g<0></{FORMULA_TAG}>", text)


def drop_header_rubrics(body: str, headers: list[str | None], report: StructureReport) -> str:
    """Убрать `<rubric>`/`<marker>` с текстом колонтитула: рубрика напечатана только в колонтитуле.

    В МТС рубрика на полосе-начале статьи часто стоит только курсивом в колонтитуле («Проблемы,
    суждения, поиск» над заголовком, 0190_1L 1976/12); модель кладёт её и в ``running_header``, и
    тегом в тело — тогда `<rubric>` встаёт над статьёй без напечатанной рубрики. Настоящая шапка
    рубрики (крупная, с буквицей, 0070_2R) в колонтитул не попадает, её тег остаётся.

    Args:
        body: Тело полосы.
        headers: ``running_header`` и ``running_footer`` ответа (``None`` — нет).
        report: Отчёт — убранные тексты в ``header_rubrics_dropped``.
    """
    names = [header_key(header) for header in headers if header]
    if not names:
        return body
    kept = []
    for paragraph in paragraphs_of(body):
        match = _RUBRIC.match(paragraph) or _MARKER_TEXT.match(paragraph)
        if match is not None and header_key(match.group(1)) in names:
            report.header_rubrics_dropped.append(match.group(1))
            continue
        kept.append(paragraph)
    return join_paragraphs(kept)


_DIGITS = re.compile(r"\d+")


def header_key(text: str) -> str:
    """Ключ сравнения текста с колонтитулом: как ``normalize_title``, но без цифр — модель пишет
    колонтитул вместе с номером страницы («Резервы — на службу пятилетке! 65», «46 Опыт работы…»).

    Args:
        text: Колонтитул из ответа или текст тега/заголовка из тела.
    """
    return normalize_title(_DIGITS.sub(" ", text))


def _assign_refs(
    texts: list[str], model_refs: list[dict], entries: list[dict], id_key: str
) -> tuple[list[dict], int, int, int]:
    """Id для каждого текста тела (заголовка или рубрики): от модели, если согласован с текстом, иначе по названию.

    Args:
        texts: Тексты `#` (или `<rubric>`) в порядке тела после доводки.
        model_refs: Записи модели ``[{"text", id_key}]`` — id принимается, если запись с тем же
            (или похожим) текстом есть и её id указывает на статью/рубрику, чьё название подходит тексту.
        entries: Статьи или рубрики списка ``[{"id", "title"}]``.
        id_key: ``article_id`` или ``rubric_id``.

    Returns:
        ``(записи [{"text", id_key}] по текстам тела, от модели, по названию, противоречивых)``.
    """
    by_id = {entry["id"]: entry["title"] for entry in entries if entry.get("id")}
    titles = [entry["title"] for entry in entries]
    refs: list[dict] = []
    from_model = from_title = wrong = 0
    for text in texts:
        model_id = None
        for ref in model_refs:
            if ref.get(id_key) and title_matches(text, [ref["text"]]):
                model_id = ref[id_key]
                break
        if model_id is not None and model_id in by_id and title_matches(text, [by_id[model_id]]):
            refs.append({"text": text, id_key: model_id})
            from_model += 1
            continue
        if model_id is not None:
            wrong += 1
        index = best_title_match(text, titles)
        found = entries[index].get("id") if index is not None else None
        refs.append({"text": text, id_key: found})
        if found is not None:
            from_title += 1
    return refs, from_model, from_title, wrong


def heading_refs(body: str, articles: list[dict], model_headings: list[dict], report: StructureReport) -> list[dict]:
    """``headings`` полосы по строкам тела: каждый `#` с id статьи (от модели или по названию).

    Args:
        body: Тело после доводки.
        articles: Статьи списка ``[{"id", "title", …}]``.
        model_headings: Поле ``headings`` ответа модели.
        report: Отчёт — счётчики источников id.
    """
    texts = [m.group(1) for p in paragraphs_of(body) if (m := _H1.match(p))]
    refs, report.heading_ids_model, report.heading_ids_title, report.heading_ids_wrong = _assign_refs(
        texts, model_headings, [a for a in articles if a.get("title")], "article_id"
    )
    return refs


def rubric_refs(body: str, rubrics: list, model_rubrics: list[dict], report: StructureReport) -> list[dict]:
    """``rubrics`` полосы по строкам тела: каждый `<rubric>` с id рубрики (от модели или по названию).

    Args:
        body: Тело после доводки.
        rubrics: Рубрики списка ``[{"id", "title"}]`` (или строки — тогда id нет).
        model_rubrics: Поле ``rubrics`` ответа модели.
        report: Отчёт — счётчики источников id.
    """
    texts = [m.group(1) for p in paragraphs_of(body) if (m := _RUBRIC.match(p))]
    entries = [r if isinstance(r, dict) else {"id": None, "title": str(r)} for r in rubrics if r]
    refs, report.rubric_ids_model, report.rubric_ids_title, report.rubric_ids_wrong = _assign_refs(
        texts, model_rubrics, entries, "rubric_id"
    )
    return refs


def apply(
    body: str,
    articles: list,
    rubrics: list,
    authors: list[dict],
    running_header: str | None = None,
    running_footer: str | None = None,
    model_headings: list[dict] | None = None,
) -> tuple[str, StructureReport]:
    """Все правки по порядку: рубрики из `##`, понижение `#`, восстановление `#` из поля, авторы, рубрики по оглавлению.

    Порядок шагов важен: каждый следующий опирается на то, что предыдущий уже привёл в норму.

    Args:
        body: Тело полосы в markdown из ответа модели (после ``tags_from_edge_words``).
        articles: Статьи оглавления ``[{"id", "title", "authors", "rubric", "rubric_id"}]`` или просто названия строками.
        rubrics: Рубрики «Содержания» выпуска — ``[{"id", "title"}]`` или строки.
        authors: Поле ``authors`` ответа модели — привязка подписей к статьям и место печати.
        running_header: Колонтитул сверху из ответа модели — тег с его текстом из тела убирается.
        running_footer: Колонтитул снизу — то же.
        model_headings: Записи ``headings`` ответа модели (``title`` до v20) — заголовок, названный в
            поле, но пропущенный в теле, вставляется; ``None`` — не восстанавливать.

    Returns:
        ``(тело после всех правок, отчёт)``; отчёт уходит в ``meta["structure"]``, а его
        ``title_in_list`` — в ``PageResult``.
    """
    # Единый вид списка статей: голые строки (тесты, старые списки) → словари с одним «title».
    articles = [{"title": item} if isinstance(item, str) else dict(item) for item in articles]
    rubrics = rubric_titles(rubrics)
    # Названия из оглавления — эталон для `#`: заголовок на полосе сверяется с ними по title_matches.
    titles = [article["title"] for article in articles if article.get("title")]
    # Отчёт о каждой правке — уходит в meta.structure, чтобы проверять доводку глазами.
    report = StructureReport()
    # 1. `## <рубрика из списка>` прямо перед `#` → `<rubric>`: до понижения заголовков, иначе шаг 2
    #    принял бы такую строку за обычный `##`, а шаг 5 — за чужой текст и сделал бы `<marker>`.
    body = rubric_from_heading(body, rubrics, report)
    # 2. `#` только для названий из оглавления: не совпавшие `#` → `##` (подзаголовок внутри статьи).
    body = demote_unlisted_headings(body, titles, report)
    #    Заголовок, названный моделью в поле headings/title, но не написанный в теле, — вставить: после
    #    понижения (чтобы «`#` нет» считалось по итогу), до авторов и рубрик (они встанут к нему).
    body = restore_heading_from_refs(body, model_headings or [], titles, report)
    # 3. Полоса-продолжение без `#`, начатая с `## <название из списка>`, — утечка списка из промпта,
    #    а не напечатанный текст: такой `##` снимается. После шага 2, чтобы видеть уже понятые `#`.
    body = drop_leaked_titles(body, titles, report)
    # 4. Блоки `<author>` — сразу после `#` своей статьи (по полям article/printed из ответа);
    #    имя из колонтитула на продолжении из тела убирается. До рубрик: рубрика встаёт перед `#`,
    #    автор после, и ей нужно видеть уже окончательное место заголовка.
    body = place_authors(body, authors, report)
    # 5. Рубрики по оглавлению: маркер, напечатанный где угодно на полосе, подтягивается к `#`
    #    своей статьи; недостающая вставляется из списка; рубрика не из списка → `<marker>`.
    body = drop_header_rubrics(body, [running_header, running_footer], report)
    body = place_rubrics(body, articles, report, rubrics)
    # 6. Формулы в голых долларах → `<latex>…</latex>`; последним, чтобы не трогать теги,
    #    расставленные выше, и не ловить доллары внутри уже обёрнутых формул.
    body = wrap_bare_math(body, report)
    # Итог по всем шагам: сколько `<marker>` на полосе (и от модели, и от шага 5).
    report.markers = body.count(f"<{StructureTag.MARKER}>")
    return body, report
