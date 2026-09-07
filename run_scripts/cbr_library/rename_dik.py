"""Переименование скачанных выпусков «Деньги и кредит» в схему пака Сафронова.

Целевая структура — <корень>/<год>/<NN>.pdf, у сдвоенных номеров <NN>-<MM>.pdf.
Номер выпуска берём не из имени файла с сервера (там разнобой), а из заголовка
в индексе: «Выпуск N 3/4 (1938)» -> 1938/03-04.pdf.
"""

import argparse
import json
import os
import re
import sys

RE_NUMBER = re.compile(r"\bN\s*([\d/]+)")


def target_name(title: str) -> str:
    """«Выпуск Т.85 N 2 (2026)» -> «02.pdf», «Выпуск N 3/4 (1938)» -> «03-04.pdf»."""
    # у томов заголовок вида «Т.85 N 2» — номер журнала это последнее N
    matches = RE_NUMBER.findall(title)
    if not matches:
        raise ValueError(f"не разобрать номер выпуска: {title!r}")
    parts = [int(p) for p in matches[-1].split("/") if p]
    return "-".join(f"{p:02d}" for p in parts) + ".pdf"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True, help="JSON от scrape_dik_index.py")
    ap.add_argument("--root", required=True, help="корень с папками по годам")
    ap.add_argument("--name", default="Полный текст", help="какие файлы переименовывать")
    ap.add_argument("--apply", action="store_true", help="без флага — только показать план")
    args = ap.parse_args()

    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)

    plan, missing, clashes = [], [], {}
    for edition in index:
        for entry in edition.get("files", []):
            if entry.get("name") != args.name or not entry.get("path"):
                continue
            year = str(edition.get("year") or 0)
            src = os.path.join(args.root, year, os.path.basename(entry["path"]))
            dst = os.path.join(args.root, year, target_name(edition["title"]))
            if not os.path.exists(src):
                # файл мог быть переименован в прошлый прогон — это не ошибка
                if not os.path.exists(dst):
                    missing.append((edition["title"], src))
                continue
            clashes.setdefault(dst, []).append(edition["title"])
            plan.append((src, dst))

    for dst, titles in clashes.items():
        if len(titles) > 1:
            print(f"КОНФЛИКТ: {dst} <- {titles}", file=sys.stderr)
            sys.exit(1)

    for title, src in missing:
        print(f"НЕТ ФАЙЛА: {title} ({src})", file=sys.stderr)

    for src, dst in plan:
        if src == dst:
            continue
        print(f"{os.path.relpath(src, args.root)} -> {os.path.relpath(dst, args.root)}")
        if args.apply:
            if os.path.exists(dst):
                raise FileExistsError(dst)
            os.rename(src, dst)

    print(f"{'переименовано' if args.apply else 'к переименованию'}: {len(plan)}", file=sys.stderr)


if __name__ == "__main__":
    main()
