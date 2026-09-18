"""Сравнение бэкендов и правил склейки переносов (ocr_utils.external_ocr_services.hyphen_join) на размеченном корпусе.

Корпус — `run_scripts/experimental/hyphen_labels.csv`: дефисные слова из выходов внешнего OCR
(1991/02, 1966/03, 1976/12, мини-набор) и все 39 с с. 47 МТС 1991/02, каждое с меткой
«перенос» (разорванный перенос, склеить), «составное» (настоящее слово с дефисом, не трогать)
или «спорно» (имена собственные, опечатки OCR — в счёт не идут). Для каждого бэкенда
(pymorphy3, mawo-pymorphy3) и правила (A, C, D) считается полнота на переносах, ложные
слияния на составных, время загрузки и скорость. Ничего не пишет, кроме таблицы в терминал
(``--md`` — в файл).

Запуск: ``uv run python scripts/compare_hyphen_join.py [--md reports/…]`` (группа зависимостей experimental).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_utils.external_ocr_services.hyphen_join import JoinRule, Morph, MorphBackend, should_join  # noqa: E402

LABELS = Path(__file__).resolve().parents[1] / "run_scripts" / "experimental" / "hyphen_labels.csv"


def load_labels(path: Path) -> list[dict]:
    """Строки разметки: ``word``, ``label``, ``src``."""
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def evaluate(morph: Morph, rule: JoinRule, rows: list[dict]) -> dict:
    """Метрики одного сочетания бэкенд × правило на размеченных словах.

    Args:
        morph: Анализатор.
        rule: Правило.
        rows: Разметка.

    Returns:
        Словарь: ``joined_breaks``/``breaks`` (полнота на переносах), ``joined_compounds``/``compounds``
        (ложные слияния), ``p47`` (склеено из 36 переносов с. 47), списки ошибок, секунды на слово.
    """
    started = time.perf_counter()
    stats = {"breaks": 0, "joined_breaks": 0, "compounds": 0, "joined_compounds": 0, "p47": 0, "p47_total": 0}
    missed: list[str] = []
    false: list[str] = []
    for row in rows:
        a, b = row["word"].split("-", 1)
        decision = should_join(a, b, morph, rule)
        if row["label"] == "перенос":
            stats["breaks"] += 1
            stats["joined_breaks"] += decision
            if row["src"] == "p47":
                stats["p47_total"] += 1
                stats["p47"] += decision
            if not decision:
                missed.append(row["word"])
        elif row["label"] == "составное":
            stats["compounds"] += 1
            stats["joined_compounds"] += decision
            if decision:
                false.append(row["word"])
    stats["seconds_per_word"] = (time.perf_counter() - started) / max(1, len(rows))
    stats["missed"] = missed
    stats["false"] = false
    return stats


@click.command()
@click.option("--labels", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=LABELS)
@click.option("--md", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Куда записать таблицу.")
def main(labels: Path, md: Path | None) -> None:
    """Таблица «бэкенд × правило»: полнота на переносах, ложные слияния, скорость."""
    rows = load_labels(labels)
    lines = [
        f"Корпус: {len(rows)} слов — переносов {sum(r['label'] == 'перенос' for r in rows)}, "
        f"составных {sum(r['label'] == 'составное' for r in rows)}, спорных {sum(r['label'] == 'спорно' for r in rows)}.",
        "",
        "| бэкенд | загрузка, с | правило | переносы склеены | с. 47 | составные слиты (ложно) | мкс/слово | пропущено | ложные |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for backend in MorphBackend:
        started = time.perf_counter()
        morph = Morph(backend)
        load_s = time.perf_counter() - started
        for rule in JoinRule:
            s = evaluate(morph, rule, rows)
            lines.append(
                f"| {backend.value} | {load_s:.2f} | {rule.value} | {s['joined_breaks']}/{s['breaks']} "
                f"({s['joined_breaks'] / max(1, s['breaks']):.2f}) | {s['p47']}/{s['p47_total']} | "
                f"{s['joined_compounds']}/{s['compounds']} ({s['joined_compounds'] / max(1, s['compounds']):.3f}) | "
                f"{s['seconds_per_word'] * 1e6:.0f} | {', '.join(s['missed'][:12])}{'…' if len(s['missed']) > 12 else ''} | "
                f"{', '.join(s['false'])} |"
            )
    text = "\n".join(lines) + "\n"
    click.echo(text)
    if md is not None:
        md.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
