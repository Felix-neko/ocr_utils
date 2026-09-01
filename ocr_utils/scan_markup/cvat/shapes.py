"""Счёт шейпов по задачам CVAT: снимок до операции и сверка после.

ЗАЧЕМ. Задачу-год приходится ПЕРЕСОЗДАВАТЬ всякий раз, когда состав выпуска изменился:
кадр в существующую задачу CVAT не дописать и не подменить, границы джобов задаются один раз
при создании (см. ``scan_markup.cvat.publish``). Пересоздание переносит разметку само, но
перенос — это сопоставление кадров по именам, и он может промолчать: полоса, переехавшая в
другой выпуск, меняет имя кадра, и её разметка однажды так и потерялась — 82 шейпа из 83
переехали, а один пропал молча, и заметить это удалось только пересчётом.

Отсюда правило: перед пересозданием снять снимок, после — сверить. Сверка ПОКАДРОВАЯ, а не
по итоговой сумме: перенос мог одновременно потерять один шейп и приобрести другой, и сумма это
скроет.

УБЫЛЬ НЕ ВСЕГДА ОШИБКА. У полосы, которую заменили на диске, разметку переносить нельзя —
обведённое относилось к старому файлу, — и она законно исчезает. Поэтому сверка не решает за
человека: она печатает, где именно стало меньше, и возвращает код 2. Что с этим делать,
решает тот, кто менял полосы.
"""

import json
import logging
from collections import Counter
from pathlib import Path

from sqlalchemy.orm import Session

from ocr_utils.scan_markup.cvat.client import CvatSettings, make_cvat_client
from ocr_utils.scan_markup.cvat.project import frame_index_by_name
from ocr_utils.scan_markup.db.repo import require_pack

logger = logging.getLogger(__name__)

# Код возврата «шейпов стало меньше». Не 1: единица у click значит «команда не отработала», а
# здесь она отработала и как раз сообщает результат.
EXIT_LOST = 2


def count_task_shapes(task) -> dict[str, int]:
    """``{имя кадра: сколько на нём шейпов}`` по одной задаче.

    По ИМЕНАМ, а не по номерам кадров: пересозданная задача нумерует кадры заново, и номер
    старой задачи в новой означает уже другую полосу.
    """
    names = {index: name for name, index in frame_index_by_name(task).items()}
    counted = Counter(names[shape.frame] for shape in task.get_annotations().shapes if shape.frame in names)
    return dict(counted)


def snapshot(session: Session, pack_name: str, years: list[str], settings: CvatSettings) -> dict[str, dict[str, int]]:
    """Снимок ``{год: {имя кадра: шейпов}}`` по задачам указанных лет."""
    pack = require_pack(session, pack_name)
    wanted = {year.name: year.cvat_task_id for year in pack.year_packages if not years or year.name in years}

    result: dict[str, dict[str, int]] = {}
    with make_cvat_client(settings) as client:
        for name, task_id in sorted(wanted.items()):
            if task_id is None:
                logger.warning("Год %s: задача в базе не записана, пропускаю", name)
                continue
            result[name] = count_task_shapes(client.tasks.retrieve(task_id))
    return result


def compare(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]) -> list[tuple[str, str, int, int]]:
    """Расхождения снимков: ``[(год, имя кадра, было, стало), ...]``.

    Сверяются ТОЛЬКО годы, попавшие в ``after``. Снимок обычно снимают сразу по нескольким
    годам, а сверяют один; без этого условия непроверенный год выглядел бы потерянным целиком
    — и сверка кричала бы об убыли там, где её никто не измерял.

    Кадр, переехавший в другой выпуск, ищется по имени файла — так же, как при переносе
    разметки: полный путь у него другой, а имя файла то же.
    """
    changes = []
    for year, new in sorted(after.items()):
        old = before.get(year)
        if old is None:
            continue  # год появился после снимка — сравнивать не с чем
        by_basename: dict[str, str] = {}
        for name in new:
            by_basename.setdefault(name.rsplit("/", 1)[-1], name)
        for name, count in sorted(old.items()):
            found = new.get(name)
            if found is None:
                moved = by_basename.get(name.rsplit("/", 1)[-1])
                found = new.get(moved, 0) if moved else 0
            if found != count:
                changes.append((year, name, count, found))
    return changes


def report(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]) -> tuple[list[str], bool]:
    """Строки отчёта и признак «где-то стало меньше». Сверяются годы из ``after``."""
    lines = [f"{'год':6s} {'было':>6s} {'стало':>6s}"]
    lost = False
    for year in sorted(after):
        if year not in before:
            lines.append(f"{year:6s} {'—':>6s} {sum(after[year].values()):6d}   <- в снимке нет, сверять не с чем")
            continue
        old_total, new_total = sum(before[year].values()), sum(after[year].values())
        mark = "" if new_total >= old_total else "   <- УБЫЛО"
        lost = lost or new_total < old_total
        lines.append(f"{year:6s} {old_total:6d} {new_total:6d}{mark}")
    skipped = sorted(set(before) - set(after))
    if skipped:
        lines.append(f"Не проверялись (в снимке есть, сейчас не измерялись): {', '.join(skipped)}")

    changes = compare(before, after)
    if changes:
        lines.append("")
        lines.append("Кадры, где число шейпов изменилось:")
        for year, name, old, new in changes:
            lines.append(f"  {year} {name}: было {old}, стало {new}")
            lost = lost or new < old
    else:
        lines.append("")
        lines.append("Покадрово всё сошлось: ни один шейп не потерян.")
    return lines, lost


def write_snapshot(data: dict[str, dict[str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def read_snapshot(path: Path) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text(encoding="utf-8"))
