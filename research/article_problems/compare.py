"""Сравнение двух прогонов по results.jsonl: совпадение статей по классам и листы «нашёл только один прогон»."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from research.article_problems.schema import CLASS_TITLES, ProblemClass


@dataclass(frozen=True)
class ClassOverlap:
    """Совпадение двух прогонов по одному классу на уровне статей.

    Args:
        cls: Класс.
        first: Статей с классом в первом прогоне.
        second: Во втором.
        both: В обоих.
        only_first: Только в первом.
        only_second: Только во втором.
    """

    cls: ProblemClass
    first: int
    second: int
    both: int
    only_first: int
    only_second: int

    @property
    def jaccard(self) -> float:
        """Доля общих статей среди найденных хоть одним прогоном (0 — ничего общего, 1 — полное совпадение)."""
        union = self.both + self.only_first + self.only_second
        return self.both / union if union else 1.0


def load_rows(results_path: Path) -> dict[str, dict]:
    """Строки ``results.jsonl`` по ключу статьи.

    Args:
        results_path: Файл ``results.jsonl`` прогона.
    """
    rows = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["article"]] = row
    return rows


def overlaps(first: dict[str, dict], second: dict[str, dict]) -> list[ClassOverlap]:
    """Совпадение по каждому классу на общих статьях двух прогонов.

    Args:
        first: Строки первого прогона.
        second: Строки второго.

    Returns:
        По записи на класс в порядке ``ProblemClass``.
    """
    keys = first.keys() & second.keys()
    result = []
    for cls in ProblemClass:
        in_first = {key for key in keys if cls.value in first[key]["classes"]}
        in_second = {key for key in keys if cls.value in second[key]["classes"]}
        result.append(
            ClassOverlap(
                cls,
                len(in_first),
                len(in_second),
                len(in_first & in_second),
                len(in_first - in_second),
                len(in_second - in_first),
            )
        )
    return result


def only_sheet(cls: ProblemClass, owner: dict[str, dict], other: dict[str, dict], owner_name: str) -> str:
    """Markdown-лист статей, где класс нашёл только ``owner``: резюме по классу и случаи с цитатами.

    Args:
        cls: Класс.
        owner: Строки прогона, который класс нашёл.
        other: Строки прогона, который его не нашёл.
        owner_name: Имя прогона-владельца для заголовка.

    Returns:
        Текст листа.
    """
    keys = sorted(
        key
        for key in owner.keys() & other.keys()
        if cls.value in owner[key]["classes"] and cls.value not in other[key]["classes"]
    )
    lines = [f"# Класс {cls.value} только у {owner_name}: {CLASS_TITLES[cls]}", "", f"Статей: {len(keys)}.", ""]
    for key in keys:
        row = owner[key]
        lines += [
            f"## {key} — {row['title'] or 'без заголовка'}",
            "",
            f"Рубрика: {row['rubric'] or '—'}. `{row['source']}`",
            "",
        ]
        summary = row["class_summaries"].get(cls.value)
        if summary:
            lines += [f"**Почему в классе:** {summary}", ""]
        for case in row["cases"]:
            if cls.value in case["classes"]:
                mark = "" if case["quote_found"] else " ⚠ цитата не найдена"
                lines.append(f"- **[{', '.join(case['classes'])}]** {case['victim']} ← {case['culprit'] or '—'}")
                lines.append(f"  - {case['explanation']}")
                lines.append(f"  - > {case['quote']}{mark}")
        lines.append("")
    return "\n".join(lines)


def write_comparison(
    first_path: Path, second_path: Path, first_name: str, second_name: str, out_dir: Path
) -> list[ClassOverlap]:
    """Записать сравнение двух прогонов: ``summary.md`` и листы ``only_{имя}/class_*.md``.

    Args:
        first_path: ``results.jsonl`` первого прогона.
        second_path: ``results.jsonl`` второго прогона.
        first_name: Короткое имя первого (в заголовках и именах папок).
        second_name: Короткое имя второго.
        out_dir: Папка сравнения.

    Returns:
        Совпадения по классам.
    """
    first, second = load_rows(first_path), load_rows(second_path)
    table = overlaps(first, second)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Сравнение прогонов: {first_name} и {second_name}",
        "",
        f"Общих статей: {len(first.keys() & second.keys())}.",
        "",
        f"| Класс | {first_name} | {second_name} | Обе | Только {first_name} | Только {second_name} | Жаккар |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in table:
        lines.append(
            f"| {item.cls.value} | {item.first} | {item.second} | {item.both} | {item.only_first} | {item.only_second} | {item.jaccard:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for owner, other, name in ((first, second, first_name), (second, first, second_name)):
        sheets = out_dir / f"only_{name}"
        sheets.mkdir(exist_ok=True)
        for cls in ProblemClass:
            (sheets / f"class_{cls.value}.md").write_text(only_sheet(cls, owner, other, name), encoding="utf-8")
    return table
