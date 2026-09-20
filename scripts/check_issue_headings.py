"""Проверка собранных выпусков external_ocr_services: `#` в теле против статей оглавления, итог сверки из sidecar.

Зачем: лишний `#` в md выпуска режет статью пополам при нарезке для LCIR (разбор 20.09.2026: 12
фантомов в 6 выпусках из 15). После сверки на сборке (``reconcile``) в каждом выпуске должно быть
ровно столько `#`, сколько заголовков списков на полосах оглавления плюс найденных статей; скрипт
считает это по всем выпускам папки выхода и печатает, что сверка удалила, понизила и восстановила.

    uv run python scripts/check_issue_headings.py --out-dir /mnt/system/raw/mts/pack1_external_ocr_services/out [--verbose]

Читает ``{год}/{выпуск}.md``, ``{выпуск}.pages.json`` и ``{выпуск}/toc.json``; ничего не пишет.
Код возврата 1, если хоть в одном выпуске `#` больше, чем положено.
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
        ``True``, если лишних `#` нет.
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
    print(
        f"{key}: `#` {len(headings)} (заголовков списков {len(list_headings)}), статей в оглавлении {articles_in_toc}, "
        f"найдено {len(found)} (восстановлено {sum(1 for a in found if a['status'] == 'restored')}), "
        f"без заголовка {sum(1 for a in side.get('articles', []) if a.get('status') == 'missing')}, "
        f"комментариев article {len(comments)}, фантомов `#` {len(side.get('phantom_headings', []))}, "
        f"лишних <rubric> {len(side.get('phantom_rubrics', []))}, правок на полосах оглавления "
        f"{len(side.get('toc_page_headings', []))}, номеров выведено "
        f"{sum(1 for p in side['pages'] if p.get('page_number_source') == 'suggested')}"
        + ("" if ok else "  ← ЛИШНИЕ `#`")
    )
    if verbose:
        for item in side.get("phantom_headings", []):
            print(f"    фантом `#` {item['action']}: {item['page']} «{item['text'][:70]}»")
        for item in side.get("phantom_rubrics", []):
            print(f"    лишний <rubric> {item['action']}: {item['page']} «{item['text'][:70]}»")
        for item in side.get("toc_page_headings", []):
            print(f"    полоса оглавления {item['action']}: {item['page']} «{item['text'][:70]}»")
        for entry in side.get("articles", []):
            if entry.get("status") != "found":
                print(
                    f"    статья {entry['status']}: {entry['id']} «{entry['title'][:70]}» (с. {entry.get('toc_page')})"
                )
    return ok


@click.command()
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--verbose", is_flag=True, help="Печатать фантомы, восстановления и статьи без заголовка.")
def main(out_dir: Path, verbose: bool) -> None:
    """Проверить все собранные выпуски под ``--out-dir``."""
    issues = sorted(out_dir.glob("*/[0-9][0-9].md"))
    if not issues:
        raise click.ClickException(f"в {out_dir} нет файлов выпусков")
    bad = [md for md in issues if not check_issue(md, verbose)]
    print(f"выпусков {len(issues)}, с лишними `#` — {len(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
