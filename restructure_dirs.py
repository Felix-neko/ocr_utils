#!/usr/bin/env python3
"""
Реструктуризация директорий журнала "Плановое хозяйство".

Убирает промежуточный годовой уровень и переименовывает директории выпусков
в унифицированный формат "ПХ-YYYY-NN" (или "ПХ-YYYY-NN-MM" для сдвоенных).
Пустые "Новая папка" и прочий мусор удаляются.
"""

import re
import shutil
import sys
from pathlib import Path

ROOT = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989) [pics_only]")

# Паттерн извлекает номер(а) выпуска и год из имени директории:
# "Плановое хозяйство 2-3-1931.page_pics"  → groups: ("2-3", "1931")
# "Плановое хозяйство № 10-1966.page_pics" → groups: ("10", "1966")
# "План хоз № 5-1974.page_pics"            → groups: ("5", "1974")
ISSUE_RE = re.compile(r"(\d+(?:-\d+)?)-(\d{4})\.page_pics$")


def make_new_name(issue_part: str, year: str) -> str:
    """Формирует имя вида ПХ-YYYY-NN или ПХ-YYYY-NN-MM."""
    nums = issue_part.split("-")
    padded = "-".join(n.zfill(2) for n in nums)
    return f"ПХ-{year}-{padded}"


def main(dry_run: bool = False) -> None:
    moves: list[tuple[Path, Path]] = []
    to_delete: list[Path] = []

    for year_dir in sorted(ROOT.iterdir()):
        if not year_dir.is_dir():
            continue
        if not year_dir.name.isdigit():
            print(f"[WARN] Неожиданная директория верхнего уровня: {year_dir.name}")
            continue

        for issue_dir in sorted(year_dir.iterdir()):
            if not issue_dir.is_dir():
                continue

            m = ISSUE_RE.search(issue_dir.name)
            if not m:
                # Пустые "Новая папка" и прочий мусор — помечаем на удаление
                contents = list(issue_dir.rglob("*"))
                if contents:
                    print(f"[WARN] Непустая нераспознанная директория, пропускаем: {issue_dir.relative_to(ROOT)}")
                else:
                    to_delete.append(issue_dir)
                continue

            new_name = make_new_name(m.group(1), m.group(2))
            dst = ROOT / new_name

            if dst.exists():
                print(f"[ERROR] Цель уже существует: {dst}")
                sys.exit(1)

            moves.append((issue_dir, dst))

    # Показываем план перемещений
    print(f"Переносов: {len(moves)}")
    for src, dst in moves:
        print(f"  {src.relative_to(ROOT)}  →  {dst.name}")

    # Показываем удаления
    if to_delete:
        print(f"\nПустых директорий под удаление: {len(to_delete)}")
        for d in to_delete:
            print(f"  {d.relative_to(ROOT)}")

    if dry_run:
        print("\n[dry-run] Реальных изменений не производилось.")
        return

    # Выполняем перемещения
    for src, dst in moves:
        src.rename(dst)

    # Удаляем пустые "Новая папка" и т.п.
    for d in to_delete:
        if d.exists():
            shutil.rmtree(d)
            print(f"Удалена: {d.relative_to(ROOT)}")

    # Удаляем опустевшие годовые директории
    for year_dir in sorted(ROOT.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            remaining = list(year_dir.iterdir())
            if not remaining:
                year_dir.rmdir()
                print(f"Удалена годовая директория: {year_dir.name}")
            else:
                print(f"[WARN] В {year_dir.name} остались файлы: {[r.name for r in remaining]}")

    print("\nГотово.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv or "-n" in sys.argv
    main(dry_run=dry)