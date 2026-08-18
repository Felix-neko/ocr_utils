#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "tqdm",
# ]
# ///
"""Раскладывает группы дубликатов из отчёта select_best_raws по отдельным папкам.

На вход подаётся текстовый отчёт, полученный командой `select_best_raws.py --mode report`,
и папка с исходными RAF-файлами. Для каждой группы дубликатов создаётся своя подпапка,
куда КОПИРУЮТСЯ (не переносятся) все кадры группы. Дальше кадрам вручную проставляются
постфиксы _defocus / _focus_okay и папка используется как разметка для валидации
алгоритмов детектирования расфокуса.
"""

import re
import shutil
from pathlib import Path

import click
from tqdm import tqdm

# "Группа 7 / 32 (2 файлов):"
GROUP_RE = re.compile(r"^Группа\s+(\d+)\s*/\s*\d+\s*\((\d+)\s+файлов\):")

# "    DSCF0050.RAF: резкость=1877.7" или то же самое с "  <-- лучший" на конце
FILE_RE = re.compile(r"^\s+(?P<name>\S+):\s*резкость=(?P<sharpness>[\d.]+)(?P<best>\s+<--\s+лучший)?\s*$")


def parse_report(report_path: Path) -> list[list[str]]:
    """Разбирает отчёт select_best_raws и возвращает группы дубликатов.

    Отчёт пишется через `tee`, поэтому в нём остаются полосы прогресса tqdm с "\\r"
    и лог первого прохода вида "DSCF0013.RAF: sharpness=1735.3" (латиницей). Строки файлов
    ищем только внутри уже открытой группы, а к именам применяем шаблон с русским
    "резкость=" — этого достаточно, чтобы не подцепить строки первого прохода.

    Args:
        report_path: Путь к текстовому отчёту.

    Returns:
        Список групп; каждая группа — список имён RAF-файлов в порядке из отчёта.
    """
    groups: list[list[str]] = []
    current: list[str] | None = None

    for raw_line in report_path.read_text(encoding="utf-8").splitlines():
        # tqdm перезаписывает строку через "\r" — берём последний фрагмент строки.
        line = raw_line.split("\r")[-1]

        if GROUP_RE.match(line):
            current = []
            groups.append(current)
            continue

        if current is None:
            continue

        m = FILE_RE.match(line)
        if m:
            current.append(m.group("name"))
        elif line.strip():
            # Любая другая непустая строка означает, что перечисление группы кончилось.
            current = None

    return groups


@click.command()
@click.argument("report", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Только показать, что будет скопировано")
def main(report: Path, input_dir: Path, output_dir: Path, dry_run: bool) -> None:
    """Копирует кадры каждой группы дубликатов в отдельную подпапку OUTPUT_DIR.

    REPORT — отчёт `select_best_raws.py --mode report`,
    INPUT_DIR — папка с исходными RAF,
    OUTPUT_DIR — куда раскладывать группы.
    """
    groups = parse_report(report)
    if not groups:
        raise click.ClickException(f"В отчёте {report} не найдено ни одной группы дубликатов")

    total_files = sum(len(g) for g in groups)
    click.echo(f"Групп: {len(groups)}, файлов в них: {total_files}")
    click.echo(f"  input  = {input_dir}")
    click.echo(f"  output = {output_dir}")

    missing: list[str] = []
    copied = 0
    width = len(str(len(groups)))

    for idx, group in enumerate(tqdm(groups, desc="Копируем группы", disable=dry_run), 1):
        # Имя папки включает первый кадр группы — так проще ориентироваться в выходной папке.
        group_dir = output_dir / f"group_{idx:0{width}d}_{Path(group[0]).stem}"
        if dry_run:
            click.echo(f"\n{group_dir} ({len(group)} файлов):")

        for name in group:
            src = input_dir / name
            if not src.exists():
                missing.append(name)
                continue
            if dry_run:
                click.echo(f"    {src} -> {group_dir / name}")
                continue
            group_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, group_dir / name)
            copied += 1

    if missing:
        click.echo(f"\nНе найдено в {input_dir}: {len(missing)} файлов")
        for name in missing:
            click.echo(f"    {name}")

    if not dry_run:
        click.echo(f"\nСкопировано файлов: {copied} в {len(groups)} папок")


if __name__ == "__main__":
    main()
