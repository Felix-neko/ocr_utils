"""Сводка прогона из кэша: разбор ответов, проверка цитат, results.jsonl, cases.csv, листы по классам и summary.md."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ocr_utils.external_ocr_services.cache import RequestCache
from research.article_problems.articles import Article
from research.article_problems.payload import Variant, cache_entry
from research.article_problems.quotes import find_quote
from research.article_problems.schema import CLASS_TITLES, ArticleAnswer, Case, ParseVerdict, ProblemClass, parse_answer


@dataclass
class ArticleResult:
    """Статья и то, что по ней ответила модель (или что ответа нет).

    Args:
        article: Статья.
        answer: Разобранный ответ с проверенными цитатами; ``None`` — ответа в кэше нет.
        cost_usd: Цена ответа.
        prompt_tokens: Токенов входа.
        completion_tokens: Токенов выхода.
        reasoning_tokens: Из них — рассуждений.
        provider: Кто обслужил запрос.
        answered_by: Метка варианта, чей ответ взят (основной или запасной).
    """

    article: Article
    answer: ArticleAnswer | None
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    provider: str | None = None
    answered_by: str = ""


def verify_cases(article: Article, cases: list[Case]) -> list[Case]:
    """Проверить цитаты случаев по тексту статьи и вернуть случаи с заполненными ``quote_*``.

    Args:
        article: Статья.
        cases: Случаи из ответа модели.

    Returns:
        Новые объекты случаев с ``quote_found``, ``quote_score``, ``quote_line``.
    """
    result = []
    for case in cases:
        match = find_quote(article, case.quote)
        result.append(
            Case(**{**case.__dict__, "quote_found": match.found, "quote_score": match.score, "quote_line": match.line})
        )
    return result


def _result_from(article: Article, variant: Variant, cache: RequestCache, cache_root: Path) -> ArticleResult | None:
    """Ответ одного варианта по статье из кэша: разбор, проверка цитат, токены и цена.

    Args:
        article: Статья.
        variant: Вариант, чей ответ читать.
        cache: Кэш запросов.
        cache_root: Его корень.

    Returns:
        Результат или ``None``, если этим вариантом статью не спрашивали.
    """
    cached = cache.lookup(cache_entry(cache_root, article, variant))
    if cached is None:
        return None
    response = cached.response
    answer = parse_answer(response.text, response.finish_reason)
    answer = ArticleAnswer(answer.verdict, verify_cases(article, answer.cases), answer.class_summaries, answer.error)
    return ArticleResult(
        article,
        answer,
        cost_usd=response.cost_usd or 0.0,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        reasoning_tokens=response.reasoning_tokens,
        provider=response.provider,
        answered_by=variant.label,
    )


def load_results(
    articles: list[Article],
    variant: Variant,
    cache: RequestCache,
    cache_root: Path,
    fallbacks: tuple[Variant, ...] = (),
) -> list[ArticleResult]:
    """Ответы по статьям из кэша; если основной ответ не годен — первый годный запасной.

    Args:
        articles: Статьи прогона.
        variant: Основной вариант.
        cache: Кэш запросов.
        cache_root: Его корень.
        fallbacks: Запасные варианты по порядку (например, синхронный повтор статей, чей ответ из
            пачки обрезан потолком токенов).

    Returns:
        По результату на статью в том же порядке; без ответа — ``answer=None``.
    """
    results = []
    for article in articles:
        found = [_result_from(article, option, cache, cache_root) for option in (variant, *fallbacks)]
        found = [result for result in found if result is not None]
        good = [result for result in found if result.answer and result.answer.verdict == ParseVerdict.OK]
        # Годный ответ основного варианта, иначе годный запасной, иначе — хоть какой (с вердиктом ошибки).
        results.append(good[0] if good else found[0] if found else ArticleResult(article, None))
    return results


def _case_dict(case: Case) -> dict:
    """Случай в JSON-сериализуемом виде (enum → строки).

    Args:
        case: Случай.
    """
    data = asdict(case)
    data["classes"] = [cls.value for cls in case.classes]
    data["culprit_kind"] = case.culprit_kind.value
    data["comparison"] = case.comparison.value if case.comparison else None
    data["improvement_kind"] = case.improvement_kind.value if case.improvement_kind else None
    return data


def source_ref(article: Article, line: int | None) -> str:
    """Ссылка ``файл:строка`` на место в выпуске (строка цитаты или начало статьи).

    Args:
        article: Статья.
        line: Строка цитаты; ``None`` — начало статьи.
    """
    return f"{article.source_path}:{line or article.line_start}"


def write_results_jsonl(results: list[ArticleResult], path: Path) -> None:
    """По строке JSON на статью: паспорт, вердикт разбора, классы, резюме, случаи, токены.

    Args:
        results: Результаты по статьям.
        path: Файл ``results.jsonl``.
    """
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            article, answer = result.article, result.answer
            row = {
                "article": article.key,
                "title": article.title,
                "rubric": article.rubric,
                "authors": list(article.authors),
                "source": source_ref(article, None),
                "n_chars": article.n_chars,
                "verdict": answer.verdict.value if answer else "missing",
                "classes": sorted((cls.value for cls in answer.classes), key=float) if answer else [],
                "class_summaries": {cls.value: text for cls, text in answer.class_summaries.items()} if answer else {},
                "cases": [_case_dict(case) for case in answer.cases] if answer else [],
                "cost_usd": result.cost_usd,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "provider": result.provider,
                "answered_by": result.answered_by,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_cases_csv(results: list[ArticleResult], path: Path) -> int:
    """Строка на пару «случай × класс» — для фильтров в таблице.

    Args:
        results: Результаты по статьям.
        path: Файл ``cases.csv``.

    Returns:
        Число строк.
    """
    columns = [
        "class",
        "article",
        "title",
        "rubric",
        "case_no",
        "victim",
        "culprit",
        "culprit_kind",
        "product",
        "concrete",
        "quote_found",
        "quote_score",
        "quote",
        "explanation",
        "foreign_analog",
        "comparison",
        "improvement_kind",
        "source",
    ]
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            if result.answer is None:
                continue
            for number, case in enumerate(result.answer.cases, start=1):
                for cls in case.classes:
                    writer.writerow(
                        {
                            "class": cls.value,
                            "article": result.article.key,
                            "title": result.article.title,
                            "rubric": result.article.rubric or "",
                            "case_no": number,
                            "victim": case.victim,
                            "culprit": case.culprit,
                            "culprit_kind": case.culprit_kind.value,
                            "product": case.product,
                            "concrete": case.concrete,
                            "quote_found": case.quote_found,
                            "quote_score": round(case.quote_score, 1),
                            "quote": case.quote,
                            "explanation": case.explanation,
                            "foreign_analog": case.foreign_analog,
                            "comparison": case.comparison.value if case.comparison else "",
                            "improvement_kind": case.improvement_kind.value if case.improvement_kind else "",
                            "source": source_ref(result.article, case.quote_line),
                        }
                    )
                    rows += 1
    return rows


def class_sheet(cls: ProblemClass, results: list[ArticleResult]) -> str:
    """Markdown-лист класса: статьи класса с резюме, случаями и цитатами.

    Args:
        cls: Класс.
        results: Результаты по статьям.

    Returns:
        Текст листа.
    """
    lines = [f"# Класс {cls.value}. {CLASS_TITLES[cls]}", ""]
    matched = [result for result in results if result.answer and cls in result.answer.classes]
    lines.append(f"Статей: {len(matched)}.")
    lines.append("")
    for result in matched:
        article, answer = result.article, result.answer
        assert answer is not None
        lines.append(f"## {article.key} — {article.title or 'без заголовка'}")
        lines.append("")
        lines.append(f"Рубрика: {article.rubric or '—'}. Источник: `{source_ref(article, None)}`")
        lines.append("")
        summary = answer.class_summaries.get(cls)
        if summary:
            lines.append(f"**Почему в классе:** {summary}")
            lines.append("")
        for case in answer.cases:
            if cls not in case.classes:
                continue
            classes = ", ".join(value.value for value in case.classes)
            mark = "" if case.quote_found else " ⚠ цитата не найдена в тексте"
            lines.append(f"- **[{classes}]** {case.victim} ← {case.culprit or 'виновник не назван'} ({case.product})")
            lines.append(f"  - {case.explanation}")
            lines.append(f"  - > {case.quote}{mark}")
            lines.append(f"  - `{source_ref(article, case.quote_line)}`")
        lines.append("")
    return "\n".join(lines)


def summary_markdown(results: list[ArticleResult], variant: Variant) -> str:
    """Сводка прогона: покрытие, классы по статьям и случаям, цитаты, токены, цена, провайдеры.

    Args:
        results: Результаты по статьям.
        variant: Вариант прогона.

    Returns:
        Текст ``summary.md``.
    """
    answered = [result for result in results if result.answer is not None]
    verdicts = Counter(result.answer.verdict.value for result in answered if result.answer)
    articles_by_class: Counter[ProblemClass] = Counter()
    cases_by_class: Counter[ProblemClass] = Counter()
    by_year: dict[str, Counter[ProblemClass]] = defaultdict(Counter)
    quotes_total = quotes_found = 0
    for result in answered:
        assert result.answer is not None
        for cls in result.answer.classes:
            articles_by_class[cls] += 1
            by_year[result.article.year][cls] += 1
        for case in result.answer.cases:
            quotes_total += 1
            quotes_found += case.quote_found
            for cls in case.classes:
                cases_by_class[cls] += 1
    with_cases = sum(1 for result in answered if result.answer and result.answer.cases)
    lines = [
        f"# Сводка прогона {variant.label}",
        "",
        f"Модель `{variant.model_slug}`, рассуждения `{variant.reasoning.value}`, промпт v{variant.prompt_version}.",
        "",
        f"- Статей: {len(results)}, с ответом: {len(answered)}, без ответа: {len(results) - len(answered)}.",
        f"- Разбор ответов: {dict(verdicts)}.",
        f"- Статей хотя бы с одним случаем: {with_cases}.",
        f"- Цитат найдено в тексте: {quotes_found} из {quotes_total}.",
        f"- Токены: вход {sum(r.prompt_tokens for r in answered):,}, выход {sum(r.completion_tokens for r in answered):,}"
        f" (рассуждений {sum(r.reasoning_tokens for r in answered):,}).",
        f"- Цена по usage.cost: ${sum(r.cost_usd for r in answered):.4f}.",
        f"- Провайдеры: {dict(Counter(r.provider for r in answered))}.",
        "",
        "| Класс | Название | Статей | Случаев |",
        "|---|---|---|---|",
    ]
    for cls in ProblemClass:
        lines.append(f"| {cls.value} | {CLASS_TITLES[cls]} | {articles_by_class[cls]} | {cases_by_class[cls]} |")
    lines += ["", "## Статей по годам", "", "| Год | " + " | ".join(cls.value for cls in ProblemClass) + " |"]
    lines.append("|---" * (len(ProblemClass) + 1) + "|")
    for year in sorted(by_year):
        lines.append(f"| {year} | " + " | ".join(str(by_year[year][cls]) for cls in ProblemClass) + " |")
    return "\n".join(lines) + "\n"


def write_report(results: list[ArticleResult], variant: Variant, out_dir: Path) -> dict[str, int]:
    """Записать все выходы прогона в папку.

    Args:
        results: Результаты по статьям.
        variant: Вариант прогона.
        out_dir: Папка прогона (``runs/{метка}``).

    Returns:
        Числа для лога: статей, строк CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    write_results_jsonl(results, out_dir / "results.jsonl")
    rows = write_cases_csv(results, out_dir / "cases.csv")
    sheets = out_dir / "by_class"
    sheets.mkdir(exist_ok=True)
    for cls in ProblemClass:
        (sheets / f"class_{cls.value}.md").write_text(class_sheet(cls, results), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(results, variant), encoding="utf-8")
    return {"articles": len(results), "csv_rows": rows}
