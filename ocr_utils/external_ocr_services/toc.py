"""Оглавление выпуска: слияние полос, рубрики-продолжения, списки для промпта и их отпечаток.

Полосы оглавления распознаются по одной (указатель на 3–7 полос одним запросом упёрся бы в
потолок выходных токенов), поэтому структуру выпуска собирает этот модуль: секции полос идут
подряд, а первая секция полосы-продолжения без рубрики приклеивается к последней секции
предыдущей — рубрика, начатая на одной полосе, продолжается на следующей без повторного
заголовка. Статьи с одинаковым названием (перекрытие тайлов, повтор строки) схлопываются.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

from ocr_utils.external_ocr_services.schema import (
    TOC_KINDS_WITH_TOC,
    BlockTag,
    StructureTag,
    TocArticle,
    TocKind,
    TocPage,
    TocSection,
)

# Виды оглавления, которые сливаются порознь; порядок — порядок в toc.json / toc.md.
KINDS = TOC_KINDS_WITH_TOC

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


# Порог похожести названия из списка и заголовка на полосе (по нормализованным строкам).
TITLE_MATCH_RATIO = 0.75


def normalize_title(title: str) -> str:
    """Ключ дедупликации: без регистра, пунктуации и лишних пробелов.

    Args:
        title: Название статьи или рубрики как напечатано.
    """
    return _NON_WORD.sub(" ", title.replace("ё", "е").replace("Ё", "Е")).strip().lower()


def title_matches(heading: str, titles: list[str]) -> bool:
    """Совпадает ли заголовок с одним из названий: равенство, вхождение или похожесть ≥ порога.

    Args:
        heading: Заголовок (или рубрика) как напечатан на полосе.
        titles: Названия из оглавления, с которыми сверяем.

    Returns:
        ``True``, если нормализованный заголовок равен одному из названий, входит в него (или
        оно в заголовок) либо похож на него по ``difflib`` не меньше чем на ``TITLE_MATCH_RATIO``.
    """
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


@dataclass
class IssueToc:
    """Слитое оглавление одного вида (contents или index) по всем его полосам выпуска.

    Args:
        kind: Вид — ``CONTENTS`` или ``INDEX``.
        sections: Секции по рубрикам в порядке полос, уже слитые (продолжения приклеены).
        pages: Относительные пути полос оглавления в порядке выпуска.
        continuations: Сколько раз первая секция полосы была приклеена к предыдущей — для лога и отладки.
    """

    kind: TocKind
    sections: list[TocSection] = field(default_factory=list)
    pages: list[str] = field(default_factory=list)
    continuations: int = 0

    @property
    def articles(self) -> list[TocArticle]:
        """Все статьи выпуска подряд, без секций."""
        return [article for section in self.sections for article in section.articles]

    @property
    def rubrics(self) -> list[str]:
        """Рубрики выпуска без повторов, в порядке первого появления."""
        seen: dict[str, None] = {}
        for section in self.sections:
            if section.rubric:
                seen.setdefault(section.rubric.strip(), None)
        return list(seen)


def merge_pages(kind: TocKind, pages: list[tuple[str, TocPage]]) -> IssueToc:
    """Полосы одного вида в порядке выпуска -> одно оглавление.

    Первая секция полосы без рубрики при непустом накопленном списке считается продолжением
    последней секции (флаг ``continues_previous`` модели учитывается, но не обязателен: модель
    его забывает чаще, чем ставит зря).

    Args:
        kind: Вид оглавления, к которому относятся все ``pages``.
        pages: ``[(относительный путь, TocPage), ...]`` уже отсортированные по положению в выпуске.

    Returns:
        ``IssueToc`` со слитыми секциями, списком полос и числом приклеенных продолжений.
    """
    toc = IssueToc(kind=TocKind(kind))
    seen_titles: set[str] = set()  # ключи уже взятых названий — перекрытие тайлов даёт повторы
    for rel, page in pages:
        toc.pages.append(rel)
        for index, section in enumerate(page.sections):
            # Статьи секции без повторов по нормализованному названию.
            articles = []
            for article in section.articles:
                key = normalize_title(article.title)
                if not key or key in seen_titles:
                    continue
                seen_titles.add(key)
                articles.append(TocArticle(article.title, list(article.authors), article.page, article.issue))
            # Первая секция полосы без рубрики — продолжение рубрики с предыдущей полосы.
            if index == 0 and toc.sections and section.rubric is None:
                toc.sections[-1].articles.extend(articles)
                toc.continuations += 1
                continue
            # Пустая секция без рубрики — ничего не несёт (все статьи были повторами).
            if not articles and section.rubric is None:
                continue
            toc.sections.append(TocSection(rubric=section.rubric, articles=articles))
    return toc


def prompt_lists(toc: IssueToc | None) -> tuple[list[str], list[dict]]:
    """Рубрики и статьи (``[{"title", "authors": [имена], "rubric": рубрика секции | None}]``) для промпта.

    Рубрика статьи берётся из оглавления — оно источник истины: по ней пост-обработка ставит
    ``<rubric>`` перед ``#`` статьи, где бы маркер рубрики ни был напечатан на полосе.

    Args:
        toc: Слитое «Содержание» выпуска; ``None`` (нет полос оглавления) — пустые списки.

    Returns:
        ``(рубрики, статьи)``: рубрики без повторов в порядке появления; статьи —
        ``[{"title", "authors": [имена], "rubric"}]`` в порядке оглавления.
    """
    if toc is None:
        return [], []
    articles = [
        {
            "title": article.title,
            "authors": [author["name"] for author in article.authors if author.get("name")],
            "rubric": section.rubric.strip() if section.rubric else None,
        }
        for section in toc.sections
        for article in section.articles
    ]
    return toc.rubrics, articles


def toc_hash(rubrics: list[str], articles: list[dict]) -> str:
    """Отпечаток списков, с которыми распознавалась полоса: сменился список — полоса устарела.

    Args:
        rubrics: Список рубрик из ``prompt_lists``.
        articles: Список статей из ``prompt_lists``.

    Returns:
        12 hex-знаков SHA-1 от JSON списков; пустые списки тоже дают устойчивый отпечаток.
    """
    # sort_keys — чтобы порядок ключей в словарях статей не менял отпечаток; 12 hex-знаков хватает.
    payload = json.dumps({"rubrics": rubrics, "articles": articles}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def to_dict(tocs: dict[TocKind, IssueToc]) -> dict:
    """Содержимое ``toc.json`` выпуска: по виду — секции, полосы, счётчики.

    Args:
        tocs: Оглавления выпуска по видам (из ``pipeline.build_issue_toc``).
    """
    return {
        TocKind(kind).value: {
            "pages": toc.pages,
            "continuations": toc.continuations,
            "sections": [asdict(section) for section in toc.sections],
        }
        for kind, toc in tocs.items()
    }


def to_markdown(tocs: dict[TocKind, IssueToc]) -> str:
    """Читаемое оглавление выпуска: заголовок по виду, рубрики, «Автор. Название — страница».

    Args:
        tocs: Оглавления выпуска по видам.
    """
    titles = {TocKind.CONTENTS: "Содержание выпуска", TocKind.INDEX: "Указатель статей за год"}
    lines: list[str] = []
    for kind, toc in tocs.items():
        lines.append(f"# {titles.get(kind, kind)}")
        lines.append("")
        lines.append(f"Полосы: {', '.join(toc.pages)}")
        lines.append("")
        for section in toc.sections:
            if section.rubric:
                lines.append(f"## {section.rubric}")
                lines.append("")
            for article in section.articles:
                authors = ", ".join(author["name"] for author in article.authors if author.get("name"))
                # «№ 3, 12» в указателе, «12» в содержании, ничего — если номера нет.
                where = ", ".join(
                    part for part in (f"№ {article.issue}" if article.issue else "", article.page or "") if part
                )
                prefix = f"{authors.rstrip('.')}. " if authors else ""
                lines.append(f"- {prefix}{article.title}{' — ' + where if where else ''}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def from_dict(payload: dict) -> dict[TocKind, IssueToc]:
    """Обратное чтение ``toc.json`` (для повторного прогона без запроса полос оглавления).

    Args:
        payload: Разобранный ``toc.json`` — то, что вернул ``to_dict``.
    """
    tocs: dict[TocKind, IssueToc] = {}
    for kind, raw in payload.items():
        sections = [
            TocSection(
                rubric=section.get("rubric"),
                articles=[
                    TocArticle(item["title"], list(item.get("authors") or []), item.get("page"), item.get("issue"))
                    for item in section.get("articles") or []
                ],
            )
            for section in raw.get("sections") or []
        ]
        tocs[TocKind(kind)] = IssueToc(
            TocKind(kind), sections, list(raw.get("pages") or []), int(raw.get("continuations") or 0)
        )
    return tocs


# Абзац-элемент оглавления: пункт списка или абзац с автором/названием и номером страницы в конце
# («… — 5», «… — № 3, 12»), либо рубрика внутри оглавления.
_TOC_ENTRY = re.compile(
    rf"^(- )?(<{StructureTag.AUTHOR}>.*?</{StructureTag.AUTHOR}>\.?\s*)?.+ — (№ ?\d+[^\d]*)?\d+[.,;\s]*$|"
    rf"^<{StructureTag.RUBRIC_IN_TOC}>.*</{StructureTag.RUBRIC_IN_TOC}>\s*$",
    re.S,
)
_TOC_OPEN = f"<{BlockTag.TOC}>"
_TOC_CLOSE = f"</{BlockTag.TOC}>"


def ensure_toc_block(body: str) -> tuple[str, bool]:
    """Обернуть список оглавления в ``<toc>…</toc>``, если модель забыла тег.

    Границы — первый и последний абзац, похожий на элемент оглавления (пункт «Автор. Название —
    страница» или ``<rubric_in_toc>``); всё между ними, включая абзацы другого вида, попадает
    внутрь. Тег уже есть — текст не меняется.

    Args:
        body: Тело полосы этапа toc в markdown.

    Returns:
        ``(тело, обёрнуто ли)``: ``True`` — тег добавлен кодом.
    """
    if _TOC_OPEN in body:
        return body, False
    # Открывающего тега нет, а закрывающий модель написать успела — он теперь лишний.
    body = body.replace(_TOC_CLOSE, "")
    paragraphs = [block for block in re.split(r"\n\s*\n", body.strip()) if block.strip()]
    # Абзац считается элементом списка, если хоть одна его строка — пункт или рубрика (пункты
    # часто идут подряд без пустых строк, а модель может приклеить к ним `</toc>`).
    hits = [
        index
        for index, paragraph in enumerate(paragraphs)
        if any(_TOC_ENTRY.match(line.strip()) for line in paragraph.split("\n"))
    ]
    if not hits:
        return body, False
    first, last = hits[0], hits[-1]
    wrapped = paragraphs[:first] + [_TOC_OPEN] + paragraphs[first : last + 1] + [_TOC_CLOSE] + paragraphs[last + 1 :]
    return "\n\n".join(wrapped).rstrip() + "\n", True


# Пункт списка оглавления: «[- ][<author>**Имена**</author>. ]Название — [№ выпуск, ]страница».
_TOC_ARTICLE_LINE = re.compile(
    rf"^(?:- )?(?:<{StructureTag.AUTHOR}>\*{{0,2}}(?P<authors>.+?)\*{{0,2}}</{StructureTag.AUTHOR}>\.?\s*)?"
    r"(?P<title>.+?)\s+—\s+(?:№\s?(?P<issue>\d+)[,\s]+)?(?P<page>\d+)[.,;\s]*$",
    re.S,
)
_TOC_RUBRIC_LINE = re.compile(
    rf"^<{StructureTag.RUBRIC_IN_TOC}>\*{{0,2}}(?P<rubric>.+?)\*{{0,2}}</{StructureTag.RUBRIC_IN_TOC}>\s*$", re.S
)


def render_toc_block(page: TocPage) -> str:
    """Блок ``<toc>…</toc>`` по структуре ``toc`` одной полосы — в той же разметке, что просит промпт.

    Args:
        page: Структурированное оглавление полосы из ответа модели.

    Returns:
        Текст блока: рубрики в ``<rubric_in_toc>``, статьи пунктами «Автор. Название — страница»
        (в указателе — «№ выпуск, страница»), авторы в ``<author>``.
    """
    lines = [_TOC_OPEN, ""]
    for section in page.sections:
        if section.rubric:
            lines += [f"<{StructureTag.RUBRIC_IN_TOC}>*{section.rubric.strip()}*</{StructureTag.RUBRIC_IN_TOC}>", ""]
        for article in section.articles:
            names = ", ".join(author["name"] for author in article.authors if author.get("name"))
            prefix = f"<{StructureTag.AUTHOR}>**{names}**</{StructureTag.AUTHOR}>. " if names else ""
            where = ", ".join(
                part for part in (f"№ {article.issue}" if article.issue else "", article.page or "") if part
            )
            lines.append(f"- {prefix}{article.title}{' — ' + where if where else ''}")
        if section.articles:
            lines.append("")
    lines.append(_TOC_CLOSE)
    return "\n".join(lines)


def parse_toc_block(body: str) -> list[TocSection] | None:
    """Разобрать блок ``<toc>…</toc>`` тела в секции: рубрики и пункты «Автор. Название — страница».

    Args:
        body: Тело полосы этапа toc в markdown.

    Returns:
        Секции в порядке чтения (рубрика ``None`` до первой ``<rubric_in_toc>``); ``None`` — блока
        ``<toc>`` в теле нет. Абзацы, не похожие ни на пункт, ни на рубрику, пропускаются.
    """
    if _TOC_OPEN not in body or _TOC_CLOSE not in body:
        return None
    inside = body[body.index(_TOC_OPEN) + len(_TOC_OPEN) : body.index(_TOC_CLOSE)]
    sections: list[TocSection] = [TocSection(None)]
    for block in re.split(r"\n\s*\n", inside):
        # Пункты могут идти подряд без пустой строки — режем абзац на строки, каждая — кандидат.
        for line in (item.strip() for item in block.split("\n")):
            if not line:
                continue
            rubric = _TOC_RUBRIC_LINE.match(line)
            if rubric is not None:
                sections.append(TocSection(rubric.group("rubric").strip()))
                continue
            entry = _TOC_ARTICLE_LINE.match(line)
            if entry is None:
                continue
            names = [name.strip() for name in (entry.group("authors") or "").split(",") if name.strip()]
            sections[-1].articles.append(
                TocArticle(
                    entry.group("title").strip(),
                    [{"name": name, "position": None} for name in names],
                    entry.group("page"),
                    entry.group("issue"),
                )
            )
    return [section for section in sections if section.rubric is not None or section.articles]


@dataclass
class TocReconcile:
    """Итог сверки тела полосы оглавления с объектом ``toc``.

    Args:
        missing_in_body: Названия статей из ``toc``, которых в теле не было (тело достроено).
        missing_in_toc: Названия статей из тела, которых в ``toc`` не было (объект достроен).
        rebuilt: Блок ``<toc>`` тела построен заново по достроенному объекту.
    """

    missing_in_body: list[str] = field(default_factory=list)
    missing_in_toc: list[str] = field(default_factory=list)
    rebuilt: bool = False

    def message(self) -> str | None:
        """Одна строка для лога и поля ``messages`` полосы; ``None`` — расхождений не было.

        Returns:
            Текст вида «оглавление: в теле не было N статей из toc (…); в toc добавлено M статей из
            тела (…); блок <toc> построен заново» или ``None``.
        """
        if not self.missing_in_body and not self.missing_in_toc:
            return None
        parts = []
        if self.missing_in_body:
            parts.append(f"в теле не было {len(self.missing_in_body)} статей из toc ({_few(self.missing_in_body)})")
        if self.missing_in_toc:
            parts.append(f"в toc добавлено {len(self.missing_in_toc)} статей из тела ({_few(self.missing_in_toc)})")
        if self.rebuilt:
            parts.append("блок <toc> построен заново по toc")
        return "оглавление: " + "; ".join(parts)

    def as_dict(self) -> dict:
        """Плоский словарь для ``meta["toc_check"]``."""
        return {"missing_in_body": self.missing_in_body, "missing_in_toc": self.missing_in_toc, "rebuilt": self.rebuilt}


def _few(titles: list[str], limit: int = 3) -> str:
    """Первые названия списком через запятую, остальное — «и ещё N».

    Args:
        titles: Названия статей.
        limit: Сколько показывать целиком.
    """
    shown = ", ".join(f"«{title}»" for title in titles[:limit])
    rest = len(titles) - limit
    return shown + (f" и ещё {rest}" if rest > 0 else "")


def _section_for(page: TocPage, rubric: str | None) -> TocSection:
    """Секция объекта ``toc`` для статьи из тела: по рубрике, иначе без рубрики, иначе новая в конце.

    Args:
        page: Объект ``toc`` полосы (достраивается на месте).
        rubric: Рубрика контекста статьи в теле; ``None`` — до первой рубрики.
    """
    if rubric is not None:
        for section in page.sections:
            if section.rubric and title_matches(rubric, [section.rubric]):
                return section
    else:
        for section in page.sections:
            if section.rubric is None:
                return section
    section = TocSection(rubric)
    page.sections.append(section)
    return section


def reconcile_toc(body: str, page: TocPage) -> tuple[str, TocPage, TocReconcile]:
    """Сверить пункты блока ``<toc>`` тела с объектом ``toc`` в обе стороны и достроить оба.

    Модель изредка выдаёт в теле только рубрики, а статьи кладёт лишь в объект (1966/03, с. 93:
    0 пунктов при 20 статьях), или наоборот теряет статью в объекте. Объект и тело — две
    транскрипции одной полосы, поэтому расхождения закрываются друг из друга: статьи только из
    тела добавляются в секцию объекта с той же рубрикой, после чего блок ``<toc>`` строится
    заново по объекту — тело и объект совпадают ровно.

    Args:
        body: Тело полосы после ``ensure_toc_block`` (тег ``<toc>`` на месте).
        page: Объект ``toc`` из ответа модели; достраивается на месте.

    Returns:
        ``(тело, объект, итог сверки)``; без расхождений тело и объект возвращаются как есть.
    """
    check = TocReconcile()
    parsed = parse_toc_block(body)
    if parsed is None or not page.sections:
        return body, page, check
    body_titles = [article.title for section in parsed for article in section.articles]
    toc_titles = [article.title for section in page.sections for article in section.articles]
    check.missing_in_body = [title for title in toc_titles if not title_matches(title, body_titles)]
    for section in parsed:
        for article in section.articles:
            if title_matches(article.title, toc_titles):
                continue
            check.missing_in_toc.append(article.title)
            _section_for(page, section.rubric).articles.append(article)
    if not check.missing_in_body and not check.missing_in_toc:
        return body, page, check
    start, end = body.index(_TOC_OPEN), body.index(_TOC_CLOSE) + len(_TOC_CLOSE)
    check.rebuilt = True
    return body[:start] + render_toc_block(page) + body[end:], page, check
