"""Поиск имён файлов, из-за которых порядок страниц может разъехаться между Windows и Linux.

ЗАЧЕМ. Порядок страниц в сканах задаётся ТОЛЬКО именем файла, а сортировка по имени в
Windows и в Linux разная: проводник Windows сортирует «естественно» и без учёта регистра
(``StrCmpLogicalW``: числовые группы сравниваются как числа, ``a`` и ``A`` — одно и то же),
а Linux (``ls``, ``glob``, ``sorted``) сравнивает байты. Пока имена однотипные
(``IMG_0001.jpg`` … ``IMG_0123.jpg``), обе сортировки дают один и тот же ряд. Ломается это
на «лишних» именах: досканах вида ``IMG_0124_1.jpg``, копиях ``IMG_0124 (1).jpg``, именах,
различающихся только регистром, и номерах разной разрядности. Такой файл встаёт в ряд
по-разному в разных ОС — и разворот уезжает не на своё место, причём молча.

Скрипт обходит папку и ищет шесть классов таких имён В ПРЕДЕЛАХ ОДНОЙ ПАПКИ (сортировка
идёт внутри папки, поэтому одноимённые файлы в соседних папках безопасны):

    суффикс _N          — ``{имя}.jpg`` и ``{имя}_1.jpg``; классический доскан;
    копия               — ``{имя} (1).jpg``, ``{имя} - копия.jpg``, ``{имя} copy.jpg``;
    регистр расширения  — ``{имя}.jpg`` и ``{имя}.JPG``: для Windows это ОДИН файл,
                          при копировании на NTFS/в архив один из двух пропадёт;
    регистр имени       — имена различаются только регистром букв, риск тот же;
    разрядность номера  — ``IMG_5.jpg`` и ``IMG_0005.jpg`` в одной папке: Windows поставит
                          их рядом по числу, Linux — далеко друг от друга по байтам;
    редкая схема имени  — имя выбивается из схемы, по которой названа вся остальная папка.
                          Ловит то, чего не ловят классы выше: ``IMG_0005_1.jpg``, у
                          которого базовый ``IMG_0005.jpg`` уже удалён (пары нет, а файл
                          всё равно чужой в ряду), и любые схемы, не предусмотренные
                          списком шаблонов, — вплоть до чужого префикса и другого
                          регистра букв в нём.

Результат — таблица в Markdown (плюс сводка по классам), при желании ещё и CSV.

Запуск:
    uv run python scripts/find_risky_duplicate_names.py "/mnt/dump3/.../МТС/в работе"
    uv run python scripts/find_risky_duplicate_names.py <папка> --csv risky_names.csv
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Расширения, по которым идёт поиск. Регистр не важен — он как раз один из проверяемых
# рисков, поэтому сравнение везде ведётся по приведённому к нижнему регистру имени.
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# Классы находок в порядке убывания опасности: первые два меняют порядок страниц,
# следующие два способны вообще потерять файл при переносе на Windows, последний —
# страховочный: ловит выбросы, под которые не написан отдельный шаблон.
CLASS_SUFFIX_N = "суффикс _N"
CLASS_COPY = "копия"
CLASS_EXT_CASE = "регистр расширения"
CLASS_NAME_CASE = "регистр имени"
CLASS_PADDING = "разрядность номера"
CLASS_RARE_SCHEME = "редкая схема имени"

CLASS_ORDER = (CLASS_SUFFIX_N, CLASS_COPY, CLASS_EXT_CASE, CLASS_NAME_CASE, CLASS_PADDING, CLASS_RARE_SCHEME)

# Доля файлов, которую должна занимать схема имени, чтобы считаться в папке основной.
# Ниже этого порога папка признаётся разнородной по построению (обложки, вкладки,
# служебные кадры вперемешку с полосами) и выбросы в ней не ищутся вовсе.
MIN_MAIN_SCHEME_SHARE = 0.8

# «Имя_N»: квантификатор жадный намеренно — у ``IMG_0053_2`` базой должен стать
# ``IMG_0053``, а не ``IMG``. Ложные срабатывания на обычных ``IMG_0001`` отсекаются
# дальше проверкой того, что базовый файл реально лежит в этой же папке.
RE_SUFFIX_N = re.compile(r"^(.*)_(\d+)$")

# Хвосты, которые дописывают проводник, Яндекс.Диск и копирование «рядом».
RE_COPY_SUFFIXES = (
    re.compile(r"^(.*?)\s*\(\d+\)$"),  # IMG_0124 (1)
    re.compile(r"^(.*?)\s*[-–—]\s*копия(?:\s*\(\d+\))?$", re.IGNORECASE),  # IMG_0124 - копия (2)
    re.compile(r"^(.*?)\s*[-–—_]?\s*copy(?:\s*\(\d+\))?$", re.IGNORECASE),  # IMG_0124 - copy
    re.compile(r"^(.*?)\s*[(_]копия\)?$", re.IGNORECASE),  # IMG_0124_копия
)

# Имя, оканчивающееся номером: префикс (может быть пустым) плюс цифровой хвост.
RE_TRAILING_NUMBER = re.compile(r"^(.*?)(\d+)$")

# Для сведения имени к схеме: латиница и кириллица считаются буквами наравне.
RE_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
RE_DIGIT = re.compile(r"\d")


@dataclass
class Finding:
    """Одна находка: пара имён в одной папке, которую сортировки могут развести.

    Attributes:
        directory: Папка относительно корня обхода.
        base: Имя файла, считающегося «основным» (или первое из пары). Пустая строка,
            если пары нет — так бывает у одиночного выброса по схеме имени.
        other: Имя парного/конфликтующего файла.
        kind: Класс находки, одна из констант ``CLASS_*``.
        base_size: Размер основного файла в байтах, 0 при отсутствии пары.
        other_size: Размер парного файла в байтах.
    """

    directory: str
    base: str
    other: str
    kind: str
    base_size: int
    other_size: int

    @property
    def same_size(self) -> bool:
        """True, если размеры совпали — скорее всего это буквальная копия, а не другой кадр."""
        return bool(self.base) and self.base_size == self.other_size


def collect_by_directory(root: Path, extensions: tuple[str, ...]) -> dict[str, list[str]]:
    """Собирает имена подходящих файлов, сгруппированные по папке.

    Args:
        root: Корень обхода.
        extensions: Расширения в нижнем регистре, включая точку.

    Returns:
        Словарь «путь папки относительно корня» -> отсортированный список имён файлов.
    """
    by_dir: dict[str, list[str]] = defaultdict(list)
    for dirpath, _dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        for name in filenames:
            if os.path.splitext(name)[1].lower() in extensions:
                by_dir[rel_dir].append(name)
    return {d: sorted(names) for d, names in by_dir.items() if names}


def find_suffix_pairs(
    names: list[str], lower_index: dict[str, str], extensions: tuple[str, ...]
) -> list[tuple[str, str, str]]:
    """Ищет пары ``{имя}.jpg`` + ``{имя}_N.jpg`` и хвосты копий вида ``{имя} (1).jpg``.

    Args:
        names: Имена файлов одной папки.
        lower_index: Индекс «имя в нижнем регистре» -> реальное имя.
        extensions: Расширения, среди которых искать базовый файл.

    Returns:
        Список троек (базовое имя, парное имя, класс находки).
    """
    found: list[tuple[str, str, str]] = []
    for name in names:
        stem, _ext = os.path.splitext(name)

        candidates: list[tuple[str, str]] = []
        match = RE_SUFFIX_N.match(stem)
        if match and match.group(1):
            candidates.append((match.group(1), CLASS_SUFFIX_N))
        for pattern in RE_COPY_SUFFIXES:
            match = pattern.match(stem)
            if match and match.group(1):
                candidates.append((match.group(1), CLASS_COPY))
                break

        for base_stem, kind in candidates:
            # Базовый файл ищется с ЛЮБЫМ из расширений и в любом регистре: доскан вполне
            # может лежать рядом с исходником, сохранённым в другом формате.
            base_name = next(
                (lower_index[base_stem.lower() + e] for e in extensions if base_stem.lower() + e in lower_index), None
            )
            if base_name is not None:
                found.append((base_name, name, kind))
                break
    return found


def find_case_collisions(names: list[str]) -> list[tuple[str, str, str]]:
    """Ищет имена, различающиеся только регистром: для Windows это один и тот же файл.

    Args:
        names: Имена файлов одной папки.

    Returns:
        Список троек (первое имя, второе имя, класс находки).
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[name.lower()].append(name)

    found: list[tuple[str, str, str]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        stems = {os.path.splitext(n)[0] for n in group}
        # Если сами имена (без расширения) совпали побайтово, различие сидит в расширении.
        kind = CLASS_EXT_CASE if len(stems) == 1 else CLASS_NAME_CASE
        first = group[0]
        for other in group[1:]:
            found.append((first, other, kind))
    return found


def name_scheme(stem: str) -> str:
    """Сводит имя к его схеме: буквы становятся ``A``, цифры ``9``.

    ``IMG_0124`` и ``IMG_0007`` дают одну схему ``AAA_9999``, а доскан ``IMG_0124_1`` —
    другую, ``AAA_9999_9``. Сравнение схем и есть способ найти файл, названный не так,
    как вся остальная папка, не перечисляя заранее все возможные «не так».

    Args:
        stem: Имя файла без расширения.

    Returns:
        Строку-схему.
    """
    return RE_DIGIT.sub("9", RE_LETTER.sub("A", stem))


def find_scheme_outliers(names: list[str], already_reported: set[str]) -> list[tuple[str, str, str]]:
    """Ищет имена, выбивающиеся из схемы, по которой названа вся остальная папка.

    Страховка на случаи, под которые не написан отдельный шаблон. Главный из них —
    доскан, у которого базовый кадр уже удалён: пары нет, класс «суффикс _N» такой файл
    пропускает, а в ряду страниц он всё равно чужой.

    Args:
        names: Имена файлов одной папки.
        already_reported: Имена, уже попавшие в другие классы, — их пропускаем, чтобы
            один и тот же файл не занял две строки отчёта.

    Returns:
        Список троек (пустая строка вместо базового имени, имя-выброс, класс находки).
    """
    schemes: dict[str, int] = defaultdict(int)
    for name in names:
        schemes[name_scheme(os.path.splitext(name)[0])] += 1
    if len(schemes) < 2:
        return []

    main_scheme, main_count = max(schemes.items(), key=lambda item: item[1])
    if main_count < len(names) * MIN_MAIN_SCHEME_SHARE:
        return []

    found: list[tuple[str, str, str]] = []
    for name in names:
        if name in already_reported:
            continue
        if name_scheme(os.path.splitext(name)[0]) != main_scheme:
            found.append(("", name, CLASS_RARE_SCHEME))
    return found


def find_padding_conflicts(names: list[str]) -> list[tuple[str, str, str]]:
    """Ищет номера разной разрядности при одном префиксе: ``IMG_5.jpg`` против ``IMG_0005.jpg``.

    Windows сравнивает такие имена как числа и ставит их рядом, Linux — как байты, и
    ``IMG_0005`` уезжает в начало ряда, а ``IMG_5`` в конец.

    Args:
        names: Имена файлов одной папки.

    Returns:
        Список троек (короткий номер, длинный номер, класс находки) — по одному примеру
        на каждую пару разрядностей внутри префикса.
    """
    # Префикс (без учёта регистра) -> разрядность номера -> самое раннее имя с ней.
    by_prefix: dict[str, dict[int, str]] = defaultdict(dict)
    for name in names:
        stem, _ext = os.path.splitext(name)
        match = RE_TRAILING_NUMBER.match(stem)
        if not match:
            continue
        prefix, digits = match.group(1), match.group(2)
        by_prefix[prefix.lower()].setdefault(len(digits), name)

    found: list[tuple[str, str, str]] = []
    for widths in by_prefix.values():
        if len(widths) < 2:
            continue
        ordered = [widths[w] for w in sorted(widths)]
        for other in ordered[1:]:
            found.append((ordered[0], other, CLASS_PADDING))
    return found


def scan(root: Path, extensions: tuple[str, ...]) -> tuple[list[Finding], int, int]:
    """Обходит дерево и собирает все находки.

    Args:
        root: Корень обхода.
        extensions: Расширения в нижнем регистре, включая точку.

    Returns:
        Тройка (находки, число просмотренных файлов, число просмотренных папок).
    """
    by_dir = collect_by_directory(root, extensions)
    findings: list[Finding] = []

    for rel_dir, names in by_dir.items():
        # При коллизии по регистру за «базовый» берётся первое имя по сортировке —
        # сам факт коллизии всё равно попадёт в отчёт отдельной строкой.
        lower_index: dict[str, str] = {}
        for name in names:
            lower_index.setdefault(name.lower(), name)
        raw = find_suffix_pairs(names, lower_index, extensions)
        raw += find_case_collisions(names)
        raw += find_padding_conflicts(names)
        # Выбросы по схеме ищутся последними: всё, что уже нашли шаблоны, им не нужно
        # дублировать.
        raw += find_scheme_outliers(names, {n for pair in raw for n in pair[:2]})

        for base, other, kind in raw:
            base_size = (root / rel_dir / base).stat().st_size if base else 0
            other_size = (root / rel_dir / other).stat().st_size
            findings.append(Finding(rel_dir, base, other, kind, base_size, other_size))

    findings.sort(key=lambda f: (CLASS_ORDER.index(f.kind), f.directory, f.base, f.other))
    return findings, sum(len(n) for n in by_dir.values()), len(by_dir)


def print_report(findings: list[Finding], files_seen: int, dirs_seen: int) -> None:
    """Печатает таблицу находок в Markdown и сводку по классам.

    Args:
        findings: Находки, уже отсортированные.
        files_seen: Сколько файлов просмотрено.
        dirs_seen: Сколько папок просмотрено.
    """
    print(f"Просмотрено файлов: {files_seen}, папок: {dirs_seen}\n")

    if not findings:
        print("Опасных имён не найдено.")
        return

    print("| Папка | Файл | Парный файл | Класс | Размеры, байт |")
    print("|---|---|---|---|---|")
    for f in findings:
        if f.base:
            sizes = f"{f.base_size} / {f.other_size}"
            if f.same_size:
                sizes += " (совпали)"
            base_cell = f"`{f.base}`"
        else:
            # Пары нет — во второй колонке ставится прочерк, размер печатается один.
            sizes = str(f.other_size)
            base_cell = "—"
        print(f"| `{f.directory}` | {base_cell} | `{f.other}` | {f.kind} | {sizes} |")

    by_kind: dict[str, int] = defaultdict(int)
    for f in findings:
        by_kind[f.kind] += 1

    print("\nПо классам:")
    for kind in CLASS_ORDER:
        print(f"    {kind:<20} {by_kind.get(kind, 0)}")
    print(f"\nВсего находок: {len(findings)}, папок с проблемой: {len({f.directory for f in findings})}")


def write_csv(findings: list[Finding], path: Path) -> None:
    """Сохраняет находки в CSV.

    Args:
        findings: Находки.
        path: Куда писать.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["папка", "файл", "парный файл", "класс", "размер", "размер парного"])
        for f in findings:
            writer.writerow([f.directory, f.base, f.other, f.kind, f.base_size, f.other_size])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path, help="Папка со сканами, обходится рекурсивно")
    parser.add_argument(
        "--ext",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help=f"Расширения файлов, по умолчанию: {' '.join(DEFAULT_EXTENSIONS)}",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Дополнительно выгрузить находки в CSV")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"Не папка: {args.root}", file=sys.stderr)
        return 1

    extensions = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext)
    findings, files_seen, dirs_seen = scan(args.root, extensions)
    print_report(findings, files_seen, dirs_seen)

    if args.csv is not None:
        write_csv(findings, args.csv)
        print(f"\nCSV: {args.csv}")

    # Ненулевой код возврата, чтобы находки было видно в прогонах из шелла.
    return 2 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
