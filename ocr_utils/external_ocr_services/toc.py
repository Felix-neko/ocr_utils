"""Оглавление выпуска: слияние полос, рубрики-продолжения, списки для промпта и их отпечаток.

Полосы оглавления распознаются по одной (указатель на 3–7 полос одним запросом упёрся бы в
потолок выходных токенов), поэтому структуру выпуска собирает этот модуль: секции полос идут
подряд, а первая секция полосы-продолжения без рубрики приклеивается к последней секции
предыдущей — рубрика, начатая на одной полосе, продолжается на следующей без повторного
заголовка. Статьи с одинаковым названием (перекрытие тайлов, повтор строки) схлопываются.
"""

from __future__ import annotations

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


def normalize_title(title: str) -> str:
    """Ключ дедупликации: без регистра, пунктуации и лишних пробелов.

    Args:
        title: Название статьи или рубрики как напечатано.
    """
    return _NON_WORD.sub(" ", title.replace("ё", "е").replace("Ё", "Е")).strip().lower()


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
    paragraphs = [block for block in re.split(r"\n\s*\n", body.strip()) if block.strip()]
    hits = [index for index, paragraph in enumerate(paragraphs) if _TOC_ENTRY.match(paragraph.strip())]
    if not hits:
        return body, False
    first, last = hits[0], hits[-1]
    wrapped = paragraphs[:first] + [_TOC_OPEN] + paragraphs[first : last + 1] + [_TOC_CLOSE] + paragraphs[last + 1 :]
    return "\n\n".join(wrapped).rstrip() + "\n", True


# Пункт списка с номером страницы в конце — то, что должно стоять в <toc> на каждую статью.
_TOC_ARTICLE_LINE = re.compile(r"^(- )?.+ — (№ ?\d+[^\d]*)?\d+[.,;\s]*$", re.S)
# Доля статей объекта ``toc``, ниже которой список в теле считается потерянным и строится заново.
TOC_BODY_MIN_SHARE = 0.5


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


def ensure_toc_entries(body: str, page: TocPage) -> tuple[str, bool]:
    """Если в теле осталось меньше половины статей из ``toc`` — блок ``<toc>`` строится заново по ``toc``.

    Модель изредка выдаёт в теле только рубрики, а статьи кладёт лишь в объект ``toc`` (1966/03,
    с. 93: 0 пунктов при 20 статьях). Объект — та же транскрипция той же полосы, только
    структурная, поэтому список из него полный и в той же разметке.

    Args:
        body: Тело полосы после ``ensure_toc_block`` (тег ``<toc>`` уже есть).
        page: Структурированное оглавление полосы.

    Returns:
        ``(тело, перестроен ли блок)``.
    """
    expected = sum(len(section.articles) for section in page.sections)
    if not expected or _TOC_OPEN not in body or _TOC_CLOSE not in body:
        return body, False
    start, end = body.index(_TOC_OPEN), body.index(_TOC_CLOSE) + len(_TOC_CLOSE)
    inside = body[start:end]
    found = sum(1 for block in re.split(r"\n\s*\n", inside) if _TOC_ARTICLE_LINE.match(block.strip()))
    if found >= expected * TOC_BODY_MIN_SHARE:
        return body, False
    return body[:start] + render_toc_block(page) + body[end:], True
