"""Сравнение алгоритмов детекции расфокуса по ручной разметке.

ЗАДАЧА. Прогон `run_scripts/defocus_detection/run_compare_algorithms_186_fuji.sh` кладёт
рядом CSV со ВСЕМИ файлами выпуска — по одному на алгоритм. Отдельно есть папка, где те
же кадры разложены по группам и часть из них помечена суффиксом в имени: `_defocus`,
`_defocus_light`, `_defocus_ultralight`, `_good_focus`, `_cover`. Скрипт связывает одно
с другим и отвечает на вопрос «какой алгоритм раньше поднимает наверх то, что человек
пометил как расфокус».

ЧТО СЧИТАЕТСЯ И ПОЧЕМУ ИМЕННО ТАК.

1. **Позиция, а не балл.** Шкалы у метрик несопоставимы (пиксели ширины края, доля
   энергии спектра, [0,1]-мера), поэтому единственная общая валюта — место файла в
   списке, отсортированном от худшего к лучшему. Инструмент и задуман как ранжирующий:
   человек смотрит верхушку списка.

2. **Три класса разметки, а не два.** `_defocus` и `_defocus_light` обязаны попасть в
   подозрительные — это цель. `_defocus_ultralight` попасть хорошо бы, но не обязан, и
   считать его пропуск ошибкой нельзя: иначе метрика штрафует за то, чего от алгоритма
   не требуют. Поэтому ultralight вынесен в отдельный столбец и в основной AUC не входит.
   `_good_focus` — то, что подниматься наверх не должно; именно на нём меряются ложные
   срабатывания.

3. **AUC против good_focus, а не против всей папки.** В папке 215 файлов, размечено 38;
   про неразмеченные неизвестно ничего — среди них наверняка есть и мягкие кадры.
   Считать их отрицательными значит подмешивать в знаменатель шум. Разметка человека
   сделана внутри групп дублей (один и тот же разворот пересняли несколько раз), то есть
   good_focus и defocus в одной группе сняты подряд с одной вёрсткой — это самое честное
   попарное сравнение, какое здесь возможно.

4. **Порог «чтобы поймать всех».** Практический вопрос не «какой AUC», а «сколько кадров
   придётся просмотреть глазами, чтобы не пропустить ни одного явного расфокуса».
   Это позиция самого невезучего файла из must-класса, она же — размер списка на просмотр.

ВТОРОЙ ОТЧЁТ (ЗОНАЛЬНЫЙ). Он считается через ширину края всегда, какой бы алгоритм ни
выбрали для общего балла (`zonal.py` импортирует `edge_width.edge_stats` напрямую).
Поэтому зональные колонки во всех CSV обязаны совпасть; скрипт проверяет это явно и,
если совпадение есть, разбирает зональный рейтинг один раз как отдельный «алгоритм».

Запуск:

    uv run python compare_defocus_algorithms.py \
        --labels-dir "/mnt/SYSTEM/raw/SI/focus_groups_186_FUJI_СИ_1985_07-09_чётная" \
        --reports-dir defocus_compare_186_fuji \
        --out defocus_algorithms_comparison_186_fuji.md
"""

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import click

# Суффиксы разметки в порядке убывания тяжести. Проверяются именно в этом порядке:
# "_defocus_light" содержит "_defocus" как префикс, и наивный поиск подстроки отнёс бы
# лёгкий расфокус к тяжёлым.
LABEL_SUFFIXES = ("_defocus_ultralight", "_defocus_light", "_defocus", "_good_focus", "_cover")

# Как классы разметки участвуют в оценке.
#   must  — обязаны попасть в подозрительные (цель алгоритма);
#   nice  — попадут, и хорошо, но не обязаны; в AUC не входят;
#   never — попадать не должны, на них меряются ложные срабатывания;
#   skip  — не участвуют вовсе (обложка: на ней нет ни текста, ни растра,
#           метрики на ней по построению меряют неизвестно что).
CLASS_ROLE = {
    "defocus": "must",
    "defocus_light": "must",
    "defocus_ultralight": "nice",
    "good_focus": "never",
    "cover": "skip",
}

# Порядок вывода классов в таблицах — от самого тяжёлого расфокуса к хорошему фокусу.
CLASS_ORDER = ("defocus", "defocus_light", "defocus_ultralight", "good_focus", "cover")

# Доли выпуска, на которых меряется полнота. Пять процентов — типовой рабочий отбор
# (столько кадров и приходится переснимать), остальные показывают, как быстро растёт
# улов, если смотреть список глубже.
RECALL_PERCENTS = (5.0, 10.0, 20.0)

# Ключ файла — номер кадра: "DSCF0017_defocus.RAF" и "DSCF0017.RAF" это один и тот же
# снимок, лежащий в двух папках. Регулярка заодно переживает опечатки в суффиксе
# (в разметке встретилось "DSCF0025_)good_focus.RAF").
FRAME_KEY = re.compile(r"^(DSCF\d+)", re.IGNORECASE)


@dataclass
class Ranking:
    """Один список файлов, отсортированный от самого подозрительного к самому чистому.

    Attributes:
        name: Имя алгоритма (или "zonal" для зонального отчёта).
        rank: Позиция файла в списке, начиная с 1; ключ — номер кадра.
        value: Балл файла в естественных единицах алгоритма (для справки в таблицах).
        total: Сколько файлов в списке.
        missing: Кадры выпуска, оставшиеся без оценки (для зонального — мало текста).
    """

    name: str
    rank: dict[str, int] = field(default_factory=dict)
    value: dict[str, float] = field(default_factory=dict)
    total: int = 0
    missing: list[str] = field(default_factory=list)


def frame_key(name: str) -> str | None:
    """Выделяет номер кадра из имени файла.

    Args:
        name: Имя файла, например "DSCF0017_defocus.RAF".

    Returns:
        Ключ вида "DSCF0017" либо None, если имя не похоже на снимок.
    """
    match = FRAME_KEY.match(name)
    return match.group(1).upper() if match else None


def parse_label(name: str) -> str | None:
    """Определяет класс разметки по суффиксу имени файла.

    Args:
        name: Имя файла с расширением.

    Returns:
        Имя класса без ведущего подчёркивания либо None, если файл не размечен.
    """
    stem = Path(name).stem
    for suffix in LABEL_SUFFIXES:
        # Суффикс ищется не строгим endswith, а вхождением в хвост: в разметке попадаются
        # лишние символы между номером и суффиксом ("DSCF0025_)good_focus").
        if stem.lower().endswith(suffix) or stem.lower().endswith(suffix.replace("_", "_)", 1)):
            return suffix.lstrip("_")
    return None


def collect_labels(labels_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Собирает ручную разметку из папки с группами дублей.

    Args:
        labels_dir: Папка, обходится рекурсивно (внутри — подпапки group_NN_*).

    Returns:
        Кортеж из двух словарей по ключу кадра: класс разметки и имя группы
        (имя подпапки, в которой лежит кадр; для файлов в корне — "—").
    """
    labels: dict[str, str] = {}
    groups: dict[str, str] = {}
    for path in sorted(labels_dir.rglob("*.RAF")):
        key = frame_key(path.name)
        if key is None:
            continue
        label = parse_label(path.name)
        if label is None:
            continue
        labels[key] = label
        groups[key] = path.parent.name if path.parent != labels_dir else "—"
    return labels, groups


def read_overall(csv_path: Path) -> Ranking:
    """Читает CSV прогона и строит рейтинг по общему качеству фокуса.

    Колонка ``rank`` в CSV уже проставлена: файл пишется отсортированным от худшего
    к лучшему, так что пересортировывать ничего не нужно.

    Args:
        csv_path: Путь к файлу <алгоритм>_scores.csv.

    Returns:
        Рейтинг по общему баллу.
    """
    ranking = Ranking(name=csv_path.stem.replace("_scores", ""))
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = frame_key(Path(row["file"]).name)
            if key is None:
                continue
            score = float(row["score"]) if row["score"] else float("nan")
            if row["error"] or not math.isfinite(score):
                ranking.missing.append(key)
                continue
            ranking.rank[key] = int(row["rank"])
            ranking.value[key] = score
    ranking.total = len(ranking.rank)
    return ranking


def read_zonal(csv_path: Path) -> Ranking:
    """Читает CSV прогона и строит рейтинг по зональному расфокусу.

    Позиции здесь в CSV нет: файл отсортирован по общему баллу, а зональный отчёт
    сортируется по убыванию перепада (``zonal_drop``) — его и воспроизводим.

    Args:
        csv_path: Путь к файлу <алгоритм>_scores.csv.

    Returns:
        Рейтинг по перепаду резкости внутри кадра; в ``missing`` — кадры, которым
        зональной оценки не досталось (не хватило однородных тайлов).
    """
    ranking = Ranking(name="zonal")
    drops: list[tuple[str, float]] = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = frame_key(Path(row["file"]).name)
            if key is None:
                continue
            if not row["zonal_drop"]:
                ranking.missing.append(key)
                continue
            drops.append((key, float(row["zonal_drop"])))

    drops.sort(key=lambda item: -item[1])
    for position, (key, drop) in enumerate(drops, 1):
        ranking.rank[key] = position
        ranking.value[key] = drop
    ranking.total = len(drops)
    return ranking


def zonal_columns_identical(reports: dict[str, Path]) -> bool:
    """Проверяет, что зональные колонки одинаковы во всех прогонах.

    Зональный расфокус считается через ширину края независимо от выбранного алгоритма,
    поэтому расхождение означало бы ошибку — либо в прогоне, либо в понимании кода.

    Args:
        reports: Словарь «алгоритм → путь к его CSV».

    Returns:
        True, если у всех алгоритмов совпали и состав кадров с оценкой, и сами перепады.
    """
    reference: dict[str, float] | None = None
    for path in reports.values():
        ranking = read_zonal(path)
        current = {key: round(value, 9) for key, value in ranking.value.items()}
        if reference is None:
            reference = current
        elif current != reference:
            return False
    return True


def auc(positive: list[int], negative: list[int]) -> float:
    """Считает AUC ранжирования: вероятность, что расфокус стоит выше хорошего кадра.

    Позиции уникальны (это места в списке), поэтому поправка на связи не нужна.

    Args:
        positive: Позиции файлов, которые обязаны быть наверху (1 — самый верх).
        negative: Позиции файлов, которые наверху быть не должны.

    Returns:
        Значение в [0, 1]; 0.5 — случайное ранжирование, NaN — если один из классов пуст.
    """
    if not positive or not negative:
        return float("nan")
    wins = sum(1 for p in positive for n in negative if p < n)
    return wins / (len(positive) * len(negative))


@dataclass
class Score:
    """Сводка качества одного рейтинга по ручной разметке.

    Attributes:
        name: Имя алгоритма.
        auc_must: AUC «defocus + defocus_light против good_focus» — основная цифра.
        auc_wide: AUC «весь размеченный расфокус, включая ultralight, против good_focus».
        recall: Доля must-файлов, попавших в худшие N % выпуска; ключ — процент.
        catch_all: Позиция самого невезучего must-файла: столько кадров надо просмотреть,
            чтобы не пропустить ни одного явного расфокуса.
        first_false: Позиция самого высокого good_focus — первое ложное срабатывание.
        median: Медианная позиция по каждому классу разметки.
        must_total: Сколько must-файлов получило оценку — знаменатель полноты.
        rated: Сколько размеченных файлов вообще получили оценку.
    """

    name: str
    auc_must: float
    auc_wide: float
    recall: dict[float, tuple[int, int]]
    catch_all: int | None
    first_false: int | None
    median: dict[str, float]
    must_total: int
    rated: int


def evaluate(ranking: Ranking, labels: dict[str, str]) -> Score:
    """Сводит один рейтинг и разметку в набор сравнимых чисел.

    Args:
        ranking: Рейтинг файлов от самого подозрительного к самому чистому.
        labels: Класс разметки по ключу кадра.

    Returns:
        Сводка качества этого рейтинга.
    """
    by_class: dict[str, list[int]] = defaultdict(list)
    for key, label in labels.items():
        position = ranking.rank.get(key)
        if position is not None:
            by_class[label].append(position)

    must = [p for label, positions in by_class.items() if CLASS_ROLE[label] == "must" for p in positions]
    nice = [p for label, positions in by_class.items() if CLASS_ROLE[label] == "nice" for p in positions]
    never = [p for label, positions in by_class.items() if CLASS_ROLE[label] == "never" for p in positions]

    recall: dict[float, tuple[int, int]] = {}
    for percent in RECALL_PERCENTS:
        cutoff = max(1, math.ceil(ranking.total * percent / 100.0))
        recall[percent] = (sum(1 for p in must if p <= cutoff), cutoff)

    median = {}
    for label in CLASS_ORDER:
        positions = sorted(by_class.get(label, []))
        if positions:
            middle = len(positions) // 2
            median[label] = (
                float(positions[middle]) if len(positions) % 2 else (positions[middle - 1] + positions[middle]) / 2.0
            )

    return Score(
        name=ranking.name,
        auc_must=auc(must, never),
        auc_wide=auc(must + nice, never),
        recall=recall,
        catch_all=max(must) if must else None,
        first_false=min(never) if never else None,
        median=median,
        must_total=len(must),
        rated=sum(len(positions) for positions in by_class.values()),
    )


def _cell(value: float | int | None, digits: int = 3) -> str:
    """Форматирует число для markdown-таблицы.

    Args:
        value: Значение или None.
        digits: Сколько знаков после запятой для дробных.

    Returns:
        Строка со значением либо "—".
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return str(value) if isinstance(value, int) else f"{value:.{digits}f}"


def summary_table(scores: list[Score], total: int) -> list[str]:
    """Строит сводную таблицу «алгоритм × качество».

    Args:
        scores: Сводки по каждому рейтингу.
        total: Сколько файлов в выпуске.

    Returns:
        Строки markdown.
    """
    recall_headers = " | ".join(f"в худших {p:g}%" for p in RECALL_PERCENTS)
    lines = [
        f"| алгоритм | AUC (must) | AUC (+ultralight) | {recall_headers} | поймать всех | 1-е ложное |",
        "|---|--:|--:|" + "--:|" * len(RECALL_PERCENTS) + "--:|--:|",
    ]
    for score in scores:
        cells = [score.name, _cell(score.auc_must), _cell(score.auc_wide)]
        for percent in RECALL_PERCENTS:
            hits, cutoff = score.recall[percent]
            cells.append(f"{hits}/{score.must_total} (топ-{cutoff})")
        cells.append(f"{_cell(score.catch_all)} из {total}")
        cells.append(_cell(score.first_false))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def build_report(
    labels: dict[str, str],
    groups: dict[str, str],
    rankings: list[Ranking],
    labels_dir: Path,
    reports_dir: Path,
    zonal_same: bool,
) -> str:
    """Собирает markdown-отчёт сравнения.

    Args:
        labels: Класс разметки по ключу кадра.
        groups: Имя группы дублей по ключу кадра.
        rankings: Рейтинги (алгоритмы по общему баллу плюс зональный).
        labels_dir: Папка с разметкой — для шапки отчёта.
        reports_dir: Папка с CSV прогонов — для шапки отчёта.
        zonal_same: Совпали ли зональные колонки у всех алгоритмов.

    Returns:
        Текст markdown-документа.
    """
    scores = [evaluate(ranking, labels) for ranking in rankings]
    total = max(ranking.total for ranking in rankings)

    counts = defaultdict(int)
    for label in labels.values():
        counts[label] += 1

    lines = [
        "# Сравнение алгоритмов детекции расфокуса",
        "",
        f"Выпуск: `{reports_dir.resolve()}` (прогон по {total} файлам).",
        f"Разметка: `{labels_dir.resolve()}`.",
        "",
        "Размечено файлов: " + ", ".join(f"**{label}** — {counts[label]}" for label in CLASS_ORDER if counts[label]),
        "",
        "Роли классов: `defocus` и `defocus_light` — **должны** попасть в подозрительные "
        "(на них считается AUC и полнота); `defocus_ultralight` — хорошо бы, но не обязан; "
        "`good_focus` — попадать **не должен**, на нём меряются ложные срабатывания; "
        "`cover` — из оценки исключена, на обложке нет ни текста, ни растра.",
        "",
    ]

    if zonal_same:
        lines += [
            "> **Зональный отчёт одинаков у всех алгоритмов.** `zonal.py` считает перепад "
            "через ширину края независимо от того, чем считается общий балл, — проверено "
            "по CSV всех прогонов, значения совпали. Поэтому ниже он идёт одной строкой "
            "`zonal`, а не шестью.",
            "",
        ]
    else:
        lines += ["> **Внимание:** зональные колонки в прогонах разошлись — это неожиданно, стоит разобраться.", ""]

    lines += ["## 1. Сводка", ""]
    lines += summary_table(scores, total)
    lines += [
        "",
        "- **AUC (must)** — вероятность, что размеченный расфокус стоит в списке выше "
        "случайного `good_focus`. 0.5 — монетка, 1.0 — идеальное разделение.",
        "- **в худших N %** — сколько must-файлов попало в верхние N % списка (в скобках — " "сколько это файлов).",
        "- **поймать всех** — позиция самого невезучего must-файла: столько кадров надо "
        "просмотреть глазами, чтобы не пропустить ни одного явного расфокуса.",
        "- **1-е ложное** — позиция самого высокого `good_focus`: с какого места начинается "
        "просмотр заведомо хороших кадров.",
        "",
        "## 2. Медианная позиция по классам",
        "",
        "Чем ниже число, тем выше класс в списке подозрительных. Разметка расставлена внутри "
        "групп дублей, поэтому осмысленно именно сравнение классов между собой, а не с 1.",
        "",
        "| алгоритм | " + " | ".join(CLASS_ORDER) + " |",
        "|---|" + "--:|" * len(CLASS_ORDER),
    ]
    for score in scores:
        cells = [score.name] + [_cell(score.median.get(label), 1) for label in CLASS_ORDER]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## 3. Позиции размеченных файлов", ""]
    lines += [
        f"Позиция в списке подозрительных из {total}; 1 — самый подозрительный. "
        "Прочерк — файл остался без оценки (для зонального это «мало однородного текста»).",
        "",
    ]
    header = ["кадр", "класс", "группа", *(r.name for r in rankings)]
    lines += ["| " + " | ".join(header) + " |", "|---|---|---|" + "--:|" * len(rankings)]

    def sort_key(item: tuple[str, str]) -> tuple[int, str]:
        return (CLASS_ORDER.index(item[1]), item[0])

    for key, label in sorted(labels.items(), key=sort_key):
        cells = [key, label, groups.get(key, "—")]
        cells += [_cell(ranking.rank.get(key)) for ranking in rankings]
        lines.append("| " + " | ".join(cells) + " |")

    missing_zonal = [r for r in rankings if r.name == "zonal"]
    if missing_zonal and missing_zonal[0].missing:
        skipped = sorted(set(missing_zonal[0].missing) & set(labels))
        lines += [
            "",
            f"Без зональной оценки осталось {len(missing_zonal[0].missing)} кадров выпуска"
            + (f", из размеченных: {', '.join(skipped)}." if skipped else "."),
        ]

    lines.append("")
    return "\n".join(lines)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--labels-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Папка с ручной разметкой (обходится рекурсивно).",
)
@click.option(
    "--reports-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Папка с CSV прогонов вида <алгоритм>_scores.csv.",
)
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Куда записать markdown-отчёт.")
def main(labels_dir: Path, reports_dir: Path, out: Path | None) -> None:
    """Сопоставляет ручную разметку расфокуса с отчётами всех алгоритмов."""
    labels, groups = collect_labels(labels_dir)
    if not labels:
        raise click.ClickException(f"В {labels_dir} не нашлось файлов с суффиксами разметки.")

    csv_paths = {path.stem.replace("_scores", ""): path for path in sorted(reports_dir.glob("*_scores.csv"))}
    if not csv_paths:
        raise click.ClickException(f"В {reports_dir} не нашлось файлов *_scores.csv.")

    click.echo(f"Разметка: {len(labels)} файлов; прогонов: {len(csv_paths)} ({', '.join(csv_paths)})", err=True)

    zonal_same = zonal_columns_identical(csv_paths)
    rankings = [read_overall(path) for path in csv_paths.values()]
    if zonal_same:
        rankings.append(read_zonal(next(iter(csv_paths.values()))))
    else:
        for name, path in csv_paths.items():
            zonal = read_zonal(path)
            zonal.name = f"zonal:{name}"
            rankings.append(zonal)

    unmatched = sorted(key for key in labels if not any(key in r.rank for r in rankings))
    if unmatched:
        click.echo(f"Размеченных кадров нет в прогоне: {', '.join(unmatched)}", err=True)

    report = build_report(labels, groups, rankings, labels_dir, reports_dir, zonal_same)
    click.echo(report)
    if out is not None:
        out.write_text(report, encoding="utf-8")
        click.echo(f"Отчёт: {out}", err=True)


if __name__ == "__main__":
    main()
