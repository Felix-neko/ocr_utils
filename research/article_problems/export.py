"""Экспорт найденных статей по классам: папка на класс, файл на статью с шапкой «кто нашёл и почему» и полным текстом."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from research.article_problems.articles import Article
from research.article_problems.schema import CLASS_TITLES, ProblemClass

# Короткие имена папок классов (номер впереди, чтобы папки шли по порядку).
CLASS_DIRS: dict[ProblemClass, str] = {
    ProblemClass.SPARE_PARTS_UNAVAILABLE: "01_нет_запчастей",
    ProblemClass.WRONG_ASSORTMENT: "02_не_то_что_нужно",
    ProblemClass.SPARE_PARTS_UNDERPRODUCED: "02.1_запчасти_не_производят",
    ProblemClass.COSTLIER_REPLACEMENT_MODEL: "03_замена_дорогой_моделью",
    ProblemClass.NON_DELIVERY: "04_непоставка",
    ProblemClass.LATE_DELIVERY: "05_несвоевременная_поставка",
    ProblemClass.DEFECTIVE: "06_брак_некондиция",
    ProblemClass.OTHER_PROBLEM: "07_любые_проблемы",
    ProblemClass.GOSSNAB_POOR_WORK: "08_плохая_работа_Госснаба",
    ProblemClass.WORSE_THAN_FOREIGN: "09_хуже_или_дороже_иностранного",
    ProblemClass.REFUSES_LISTED_PRODUCT: "10_отказ_от_своего_ассортимента",
    ProblemClass.NO_IMPROVEMENT: "11_нет_совершенствования_продукции",
    ProblemClass.UNWANTED_PRODUCT: "12_никому_не_нужная_продукция",
}
# Подписи видов случаев класса 11 в шапке файла.
IMPROVEMENT_TITLES = {
    "stagnation": "застой — выпускает устаревшее, не переходит на новое",
    "not_launched": "разработка готова к производству, но её не запускают",
    "barrier": "названа причина, мешающая совершенствовать продукцию",
}
# Символы, недопустимые или неудобные в именах файлов.
UNSAFE_RE = re.compile(r"[^\w\-]+", re.UNICODE)
TITLE_IN_NAME = 60


@dataclass(frozen=True)
class ModelRun:
    """Прогон одной модели для экспорта.

    Args:
        name: Имя модели в шапке («DeepSeek V4.1 Flash»).
        rows: Строки ``results.jsonl`` прогона по ключу статьи.
    """

    name: str
    rows: dict[str, dict]


def file_name(article: Article) -> str:
    """Имя файла статьи: ``1968_09_A15_ПРИЧИНА_НЕРИТМИЧНОСТИ.md`` (заголовок обрезан до 60 знаков).

    Args:
        article: Статья.
    """
    title = UNSAFE_RE.sub("_", article.title).strip("_")[:TITLE_IN_NAME].rstrip("_")
    stem = f"{article.year}_{article.issue}_{article.article_id}"
    return f"{stem}_{title}.md" if title else f"{stem}.md"


def _model_section(run: ModelRun, row: dict, cls: ProblemClass) -> list[str]:
    """Раздел шапки от одной модели: её резюме по классу и случаи этого класса с цитатами.

    Args:
        run: Прогон модели.
        row: Строка ``results.jsonl`` этой модели по статье.
        cls: Класс папки.

    Returns:
        Строки markdown.
    """
    lines = [f"### {run.name}", ""]
    summary = row["class_summaries"].get(cls.value)
    if summary:
        lines += [f"**Почему в классе:** {summary}", ""]
    for number, case in enumerate((case for case in row["cases"] if cls.value in case["classes"]), start=1):
        mark = "" if case["quote_found"] else " *(цитата не найдена в тексте дословно)*"
        lines.append(
            f"{number}. **Классы случая:** {', '.join(case['classes'])}. **Кто пострадал:** {case['victim'] or '—'}. "
            f"**Виновник:** {case['culprit'] or 'не назван'}. **Продукция:** {case['product'] or '—'}."
        )
        lines.append(f"   - {case['explanation']}")
        lines.append(f"   - > {case['quote']}{mark}")
        if case.get("improvement_kind") and cls == ProblemClass.NO_IMPROVEMENT:
            lines.append(
                f"   - Вид случая: {IMPROVEMENT_TITLES.get(case['improvement_kind'], case['improvement_kind'])}"
            )
        if case.get("foreign_analog"):
            lines.append(f"   - Иностранный аналог: {case['foreign_analog']} ({case.get('comparison') or '—'})")
    lines.append("")
    return lines


def article_document(article: Article, cls: ProblemClass, runs: list[ModelRun]) -> str:
    """Файл статьи для папки класса: шапка (кто нашёл, почему) и полный текст.

    Args:
        article: Статья.
        cls: Класс папки.
        runs: Прогоны моделей; в шапку попадают те, кто отнёс статью к классу.

    Returns:
        Текст markdown-файла.
    """
    found = [run for run in runs if article.key in run.rows and cls.value in run.rows[article.key]["classes"]]
    lines = [
        f"# {article.title or 'Без заголовка'}",
        "",
        f"- **Статья:** {article.key} — «Материально-техническое снабжение», {article.year}, № {article.issue}",
        f"- **Рубрика:** {article.rubric or '—'}",
        f"- **Авторы:** {'; '.join(article.authors) or '—'}",
        f"- **Источник:** `{article.source_path}:{article.line_start}`",
        f"- **Класс {cls.value}:** {CLASS_TITLES[cls]}",
        f"- **Нашли:** {', '.join(run.name for run in found)}",
    ]
    # Все классы статьи у каждой модели — чтобы видеть, в каких ещё папках она лежит.
    for run in runs:
        row = run.rows.get(article.key)
        classes = ", ".join(row["classes"]) if row else "нет ответа"
        lines.append(f"- **Все классы статьи у {run.name}:** {classes or '—'}")
    lines += ["", "## Почему статья отнесена к классу", ""]
    for run in found:
        lines += _model_section(run, run.rows[article.key], cls)
    lines += ["---", "", "## Текст статьи", "", article.text, ""]
    return "\n".join(lines)


def export_by_class(articles: list[Article], runs: list[ModelRun], out_dir: Path) -> dict[ProblemClass, int]:
    """Разложить статьи по папкам классов: статья попадает в папку, если хоть одна модель дала ей класс.

    Args:
        articles: Все статьи пака.
        runs: Прогоны моделей.
        out_dir: Корень экспорта (папки классов создаются внутри).

    Returns:
        Сколько статей записано в каждую папку.
    """
    counts = {cls: 0 for cls in ProblemClass}
    for cls in ProblemClass:
        (out_dir / CLASS_DIRS[cls]).mkdir(parents=True, exist_ok=True)
    for article in articles:
        # Объединение классов обеих моделей.
        classes = {value for run in runs if article.key in run.rows for value in run.rows[article.key]["classes"]}
        for cls in ProblemClass:
            if cls.value not in classes:
                continue
            path = out_dir / CLASS_DIRS[cls] / file_name(article)
            path.write_text(article_document(article, cls, runs), encoding="utf-8")
            counts[cls] += 1
    return counts
