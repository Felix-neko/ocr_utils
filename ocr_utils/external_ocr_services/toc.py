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

from ocr_utils.external_ocr_services.schema import TocArticle, TocPage, TocSection

KIND_CONTENTS = "contents"
KIND_INDEX = "index"
KINDS = (KIND_CONTENTS, KIND_INDEX)

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def normalize_title(title: str) -> str:
    """Ключ дедупликации: без регистра, пунктуации и лишних пробелов."""
    return _NON_WORD.sub(" ", title.replace("ё", "е").replace("Ё", "Е")).strip().lower()


@dataclass
class IssueToc:
    """Слитое оглавление одного вида (contents или index) по всем его полосам выпуска."""

    kind: str
    sections: list[TocSection] = field(default_factory=list)
    pages: list[str] = field(default_factory=list)  # относительные пути полос в порядке выпуска
    # Сколько раз первая секция полосы была приклеена к предыдущей — для лога и отладки.
    continuations: int = 0

    @property
    def articles(self) -> list[TocArticle]:
        return [article for section in self.sections for article in section.articles]

    @property
    def rubrics(self) -> list[str]:
        seen: dict[str, None] = {}
        for section in self.sections:
            if section.rubric:
                seen.setdefault(section.rubric.strip(), None)
        return list(seen)


def merge_pages(kind: str, pages: list[tuple[str, TocPage]]) -> IssueToc:
    """Полосы одного вида в порядке выпуска -> одно оглавление.

    ``pages`` — ``[(относительный путь, TocPage), ...]`` уже отсортированные по положению в
    выпуске. Первая секция полосы без рубрики при непустом накопленном списке считается
    продолжением последней секции (флаг ``continues_previous`` модели учитывается, но не
    обязателен: модель его забывает чаще, чем ставит зря).
    """
    toc = IssueToc(kind=kind)
    seen_titles: set[str] = set()
    for rel, page in pages:
        toc.pages.append(rel)
        for index, section in enumerate(page.sections):
            articles = []
            for article in section.articles:
                key = normalize_title(article.title)
                if not key or key in seen_titles:
                    continue
                seen_titles.add(key)
                articles.append(TocArticle(article.title, list(article.authors), article.page, article.issue))
            if index == 0 and toc.sections and section.rubric is None:
                toc.sections[-1].articles.extend(articles)
                toc.continuations += 1
                continue
            if not articles and section.rubric is None:
                continue
            toc.sections.append(TocSection(rubric=section.rubric, articles=articles))
    return toc


def prompt_lists(toc: IssueToc | None) -> tuple[list[str], list[dict]]:
    """Рубрики и статьи (``[{"title", "authors": [имена], "rubric": рубрика секции | None}]``) для промпта.

    Рубрика статьи берётся из оглавления — оно источник истины: по ней пост-обработка ставит
    ``<rubric>`` перед ``#`` статьи, где бы маркер рубрики ни был напечатан на полосе.
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
    """Отпечаток списков, с которыми распознавалась полоса: сменился список — полоса устарела."""
    payload = json.dumps({"rubrics": rubrics, "articles": articles}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def to_dict(tocs: dict[str, IssueToc]) -> dict:
    """Содержимое ``toc.json`` выпуска: по виду — секции, полосы, счётчики."""
    return {
        kind: {
            "pages": toc.pages,
            "continuations": toc.continuations,
            "sections": [asdict(section) for section in toc.sections],
        }
        for kind, toc in tocs.items()
    }


def to_markdown(tocs: dict[str, IssueToc]) -> str:
    """Читаемое оглавление выпуска: заголовок по виду, рубрики, «Автор. Название — страница»."""
    titles = {KIND_CONTENTS: "Содержание выпуска", KIND_INDEX: "Указатель статей за год"}
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
                where = ", ".join(
                    part for part in (f"№ {article.issue}" if article.issue else "", article.page or "") if part
                )
                prefix = f"{authors.rstrip('.')}. " if authors else ""
                lines.append(f"- {prefix}{article.title}{' — ' + where if where else ''}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def from_dict(payload: dict) -> dict[str, IssueToc]:
    """Обратное чтение ``toc.json`` (для повторного прогона без запроса полос оглавления)."""
    tocs: dict[str, IssueToc] = {}
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
        tocs[kind] = IssueToc(kind, sections, list(raw.get("pages") or []), int(raw.get("continuations") or 0))
    return tocs
