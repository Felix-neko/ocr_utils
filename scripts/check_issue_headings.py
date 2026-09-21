"""Проверка собранных выпусков external_ocr_services: `#` в теле против статей оглавления, итог сверки из sidecar.

Зачем: лишний `#` в md выпуска режет статью пополам при нарезке для LCIR (разбор 20.09.2026: 12
фантомов в 6 выпусках из 15). После сверки на сборке (``reconcile``) в каждом выпуске должно быть
ровно столько `#`, сколько заголовков списков на полосах оглавления плюс найденных статей; скрипт
считает это по всем выпускам папки выхода и печатает, что сверка удалила, понизила и восстановила.

    uv run python scripts/check_issue_headings.py --out-dir /mnt/system/raw/mts/pack1_external_ocr_services/out [--verbose]

Читает ``{год}/{выпуск}.md``, ``{выпуск}.pages.json`` и ``{выпуск}/toc.json``; ничего не пишет.
Кроме счёта `#` проверяет структуру: дубли id в комментариях, порядок статей по оглавлению,
место `#` относительно страницы из оглавления (расхождение ≥ 2), рубрика перед первой статьёй своего
раздела. Код возврата 1, если хоть в одном выпуске `#` больше, чем положено, или есть замечания по
структуре (инверсии порядка бывают законными — оглавление перечисляет заметки не по печати; смотреть глазами).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click

_H1 = re.compile(r"^# (.+)$", re.M)
_LIST_HEADING = re.compile(r"содержание|оглавление|указатель|перечень", re.IGNORECASE)


def check_issue(md: Path, verbose: bool) -> bool:
    """Один выпуск: посчитать `#`, сравнить с оглавлением и sidecar, напечатать строку итога.

    Args:
        md: Файл выпуска ``{год}/{выпуск}.md``.
        verbose: Печатать ли каждый фантом, восстановление и статью без заголовка.

    Returns:
        ``True``, если лишних `#` нет и замечаний по структуре нет.
    """
    key = f"{md.parent.name}/{md.stem}"
    text = md.read_text(encoding="utf-8")
    side = json.loads(md.with_name(md.stem + ".pages.json").read_text(encoding="utf-8"))
    toc_path = md.parent / md.stem / "toc.json"
    toc = json.loads(toc_path.read_text(encoding="utf-8")) if toc_path.is_file() else {}
    articles_in_toc = sum(len(s["articles"]) for s in toc.get("contents", {}).get("sections", []))
    headings = _H1.findall(text)
    list_headings = [h for h in headings if _LIST_HEADING.search(h)]
    found = [a for a in side.get("articles", []) if a.get("status") in ("found", "restored")]
    # Заголовок годового указателя — и заголовок списка, и статья «Содержания»: считается один раз.
    index_articles = sum(1 for a in found if _LIST_HEADING.search(a.get("title") or ""))
    expected = len(list_headings) + len(found) - index_articles
    ok = len(headings) <= expected
    comments = re.findall(r"^<!-- article (A\d+) -->$", text, re.M)
    problems = _structure_problems(text, side, toc)
    print(
        f"{key}: `#` {len(headings)} (заголовков списков {len(list_headings)}), статей в оглавлении {articles_in_toc}, "
        f"найдено {len(found)} (восстановлено {sum(1 for a in found if a['status'] == 'restored')}), "
        f"без заголовка {sum(1 for a in side.get('articles', []) if a.get('status') == 'missing')}, "
        f"комментариев article {len(comments)}, фантомов `#` {len(side.get('phantom_headings', []))}, "
        f"лишних <rubric> {len(side.get('phantom_rubrics', []))}, правок на полосах оглавления "
        f"{len(side.get('toc_page_headings', []))}, номеров выведено "
        f"{sum(1 for p in side['pages'] if p.get('page_number_source') == 'suggested')}"
        + ("" if ok else "  ← ЛИШНИЕ `#`")
        + ("" if not problems else f"  ← замечаний по структуре: {len(problems)}")
    )
    if verbose:
        for problem in problems:
            print(f"    структура: {problem}")
        for item in side.get("phantom_headings", []):
            print(f"    фантом `#` {item['action']}: {item['page']} «{item['text'][:70]}»")
        for item in side.get("phantom_rubrics", []):
            print(f"    лишний <rubric> {item['action']}: {item['page']} «{item['text'][:70]}»")
        for item in side.get("toc_page_headings", []):
            print(f"    полоса оглавления {item['action']}: {item['page']} «{item['text'][:70]}»")
        for entry in side.get("articles", []):
            if entry.get("status") != "found":
                extra = entry.get("source") or entry.get("reason") or ""
                print(
                    f"    статья {entry['status']} {extra}: {entry['id']} «{entry['title'][:70]}» (с. {entry.get('toc_page')})"
                )
    return ok and not problems


def _structure_problems(text: str, side: dict, toc: dict) -> list[str]:
    """Замечания по структуре выпуска: дубли id, порядок статей, место `#` относительно оглавления, рубрика не перед своим разделом.

    Args:
        text: md выпуска.
        side: sidecar ``.pages.json``.
        toc: ``toc.json`` выпуска.

    Returns:
        Строки-замечания (пусто — всё в порядке).
    """
    problems: list[str] = []
    sections = toc.get("contents", {}).get("sections", [])
    marks = re.findall(r"^<!-- (article|rubric) ([AR]\d+) -->$", text, re.M)
    articles = [ref for kind, ref in marks if kind == "article"]
    for ref in {ref for ref in articles if articles.count(ref) > 1}:
        problems.append(f"статья {ref} помечена {articles.count(ref)} раза")
    rubrics = [ref for kind, ref in marks if kind == "rubric"]
    for ref in set(rubrics):
        allowed = sum(1 for section in sections if section.get("id") == ref)
        if rubrics.count(ref) > allowed:
            problems.append(f"рубрика {ref} помечена {rubrics.count(ref)} раза при {allowed} разделах")
    # Порядок статей по id (порядок оглавления): инверсии — либо оглавление перечисляет заметки не по
    # печати (законно), либо не тот `#` выбран основным; смотреть глазами.
    numbers = [int(ref[1:]) for ref in articles]
    for i in range(len(numbers) - 1):
        if numbers[i] > numbers[i + 1]:
            problems.append(f"порядок: {articles[i]} перед {articles[i + 1]}")
    # Место `#`: номер полосы против страницы из оглавления (±1 — норма: опечатки номеров).
    for entry in side.get("articles", []):
        toc_page, number = entry.get("toc_page"), entry.get("page_number")
        if entry.get("status") in ("found", "restored") and toc_page is not None and number is not None:
            if abs(number - toc_page) >= 2:
                problems.append(
                    f"{entry['id']} «{entry['title'][:40]}»: `#` на с. {number}, в оглавлении с. {toc_page}"
                )
    # Рубрика — непосредственно перед `#` первой статьи своего раздела.
    for index, (kind, ref) in enumerate(marks):
        if kind != "rubric":
            continue
        following = next((r for k, r in marks[index + 1 :] if k == "article"), None)
        expected = [s["articles"][0]["id"] for s in sections if s.get("id") == ref and s["articles"]]
        if following not in expected:
            problems.append(f"рубрика {ref} стоит перед {following}, ожидались {expected}")
    return problems


@click.command()
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--verbose", is_flag=True, help="Печатать фантомы, восстановления и статьи без заголовка.")
def main(out_dir: Path, verbose: bool) -> None:
    """Проверить все собранные выпуски под ``--out-dir``."""
    issues = sorted(out_dir.glob("*/[0-9][0-9].md"))
    if not issues:
        raise click.ClickException(f"в {out_dir} нет файлов выпусков")
    bad = [md for md in issues if not check_issue(md, verbose)]
    print(f"выпусков {len(issues)}, с лишними `#` или замечаниями по структуре — {len(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
