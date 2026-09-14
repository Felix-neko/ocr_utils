"""Карта модулей и классов пакета → docs/modules.md.

Обходит файлы `ocr_utils/` и `research/`, отслеживаемые git, и собирает из докстрингов первую
строку по каждому модулю и каждому классу верхнего уровня. Функции не включаются: их больше
тысячи, и такая карта была бы дороже, чем `grep`.

Зачем: агенту (и человеку) нужен способ понять, где что лежит, не читая 300 файлов. Карта
собирается из докстрингов, а не пишется руками — иначе она протухает за месяц.

Запуск::

    uv run python scripts/gen_module_map.py          # перегенерировать docs/modules.md
    uv run python scripts/gen_module_map.py --check  # код 1, если файл устарел

Модули без докстринга печатаются в stderr — их стоит дописать.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "modules.md"
PACKAGES = ("ocr_utils", "research")

HEADER = """# Карта модулей

Файл сгенерирован `scripts/gen_module_map.py` из докстрингов — **не править руками**,
перегенерировать после добавления модуля или класса. Первая строка докстринга модуля и каждого
класса верхнего уровня; функции не включены — их ищите через `grep -n "^def "`.

"""


def first_line(node: ast.AST) -> str:
    """Первая непустая строка докстринга узла или пустая строка."""
    doc = ast.get_docstring(node) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line.replace("|", "\\|")
    return ""


def tracked_modules() -> list[Path]:
    """Python-файлы пакетов: отслеживаемые git плюс неотслеживаемые, но не игнорируемые.

    Через git, а не обходом ФС: в корне репо лежат гигабайты данных, а ещё не закоммиченный
    живой код (новый подпакет до первого коммита) карте тоже нужен.
    """
    cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *PACKAGES]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / line for line in out.stdout.splitlines() if line.endswith(".py")]


def base_names(cls: ast.ClassDef) -> str:
    """Имена базовых классов через запятую (для наследников и датаклассов важно)."""
    names = []
    for base in cls.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return ", ".join(names)


def render() -> tuple[str, list[str]]:
    """Собрать markdown карты; вторым значением — модули без докстринга."""
    by_package: dict[str, list[tuple[str, str, list[str]]]] = defaultdict(list)
    missing: list[str] = []
    for path in sorted(tracked_modules()):
        rel = path.relative_to(ROOT)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"пропущен {rel}: {exc}", file=sys.stderr)
            continue
        doc = first_line(tree)
        if not doc and path.name != "__init__.py":
            missing.append(str(rel))
        classes = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = base_names(node)
                label = f"`{node.name}`" + (f" ({bases})" if bases else "")
                cdoc = first_line(node)
                classes.append(f"{label} — {cdoc}" if cdoc else label)
        # Подпакет — первые два уровня пути: ocr_utils/scan_markup, research/external_ocr_models.
        # research/legacy/table_processing — три уровня, чтобы не смешивать с другими legacy.
        parts = rel.parts
        depth = 3 if parts[:2] == ("research", "legacy") else 2
        package = "/".join(parts[:depth]) if len(parts) > depth else str(rel.parent)
        by_package[package].append((str(rel), doc, classes))

    lines = [HEADER]
    for package in sorted(by_package):
        modules = by_package[package]
        lines.append(f"## {package}\n")
        init = next((doc for rel, doc, _ in modules if rel.endswith("__init__.py")), "")
        if init:
            lines.append(f"{init}\n")
        lines.append("| Модуль | Назначение | Классы |")
        lines.append("|---|---|---|")
        for rel, doc, classes in modules:
            if rel.endswith("__init__.py") and not classes:
                continue
            name = rel[len(package) + 1 :] if rel.startswith(package + "/") else rel
            lines.append(f"| `{name}` | {doc} | {'<br>'.join(classes)} |")
        lines.append("")
    return "\n".join(lines), missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Карта модулей и классов → docs/modules.md")
    parser.add_argument("--check", action="store_true", help="не писать, а проверить, что файл актуален")
    args = parser.parse_args()

    text, missing = render()
    if missing:
        print("модули без докстринга:", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)

    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} устарел: uv run python scripts/gen_module_map.py", file=sys.stderr)
            return 1
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"записано {OUT.relative_to(ROOT)}: {text.count(chr(10))} строк")
    return 0


if __name__ == "__main__":
    sys.exit(main())
