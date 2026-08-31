"""Валидационная выборка: папки с примерами дефектов -> что должно получиться.

ОТКУДА БЕРЁТСЯ ЭТАЛОН. Прогон ``detect --debug-dir`` кладёт по оверлею на каждую полосу с
находками. Разобрав их глазами, примеры неправильной разметки раскладывают по папкам —
имя папки называет ТИП дефекта. Разбор пака-1 дал шесть папок, они и перечислены ниже.

Имя файла оверлея разбирается обратно в ``rel_path`` функцией
``detection.overlay.overlay_to_rel_path`` — сборка и разбор лежат рядом намеренно, иначе
они разъедутся молча и выборка станет пустой без единой ошибки.

Папку заводит не только разбор оверлеев руками: ``dot_leaders`` добавлена по ложному
срабатыванию, найденному при отладке (строка отточий в оглавлении даёт ту же статистику
мелких круглых пятен, что и растровая сетка). Такие находки надо класть в выборку сразу,
иначе следующая правка порогов их молча вернёт.

ЧЕГО ЗДЕСЬ НЕТ. Дефект «картинка была, но не обнаружена» проверяется отдельно: по одним
оверлеям его не увидеть, оверлея у такой полосы просто нет. Частичную страховку от него
даёт режим ``--pages-from-db`` (см. ``validation.run``) — он показывает, у скольких полос
с уже найденными областями находки пропали.
"""

from dataclasses import dataclass
from pathlib import Path

from ocr_utils.scan_markup.detection.overlay import overlay_to_rel_path

# Идентификаторы дефектов.
COLOR_ON_GRAY = "color_on_gray"
LINEART = "lineart"
FALSE_POSITIVE = "false_positive"
MERGED = "merged"
SPLIT = "split"
BROKEN_SOURCE = "broken_source"
DOT_LEADERS = "dot_leaders"


@dataclass(frozen=True)
class Defect:
    """Тип дефекта: как называется папка, что от детекции ждут и как это проверить."""

    key: str
    folder: str
    expectation: str
    scored: bool = True


DEFECTS = (
    Defect(
        COLOR_ON_GRAY, "grayscale детектирован как цветное изображение", "все найденные области должны быть grayscale"
    ),
    Defect(LINEART, "line art детектирован как растр", "областей быть не должно"),
    Defect(
        FALSE_POSITIVE,
        "тут картинок не было но их детектировали (возможно из-за артефактов но лучше не надо)",
        "областей быть не должно",
    ),
    Defect(MERGED, "две картинки детектированы как одна большая", "областей должно быть не меньше двух"),
    Defect(SPLIT, "растровая картинка детектирована как несколько маленьких", "область должна быть ровно одна"),
    Defect(DOT_LEADERS, "отточия в оглавлении приняты за растр", "областей быть не должно"),
    Defect(
        BROKEN_SOURCE,
        "а тут надо будет просто починить картинку-исходник",
        "чинится правкой исходника, а не детектором",
        scored=False,
    ),
)

BY_FOLDER = {defect.folder: defect for defect in DEFECTS}


@dataclass(frozen=True)
class Case:
    """Одна полоса выборки: путь в паке, тип дефекта и имя исходного оверлея."""

    rel_path: str
    path: Path
    defect: Defect
    overlay: str


def collect_cases(cases_root: Path, pack_dir: Path) -> tuple[list[Case], list[str]]:
    """Собирает выборку из папок ``cases_root``; вторым значением — список замечаний.

    Незнакомая папка и отсутствующий в паке файл не роняют прогон, а попадают в замечания:
    выборку пополняют руками, и опечатка в имени не должна выглядеть как «дефект починен».
    """
    cases: list[Case] = []
    notes: list[str] = []
    for folder in sorted(p for p in cases_root.iterdir() if p.is_dir()):
        defect = BY_FOLDER.get(folder.name)
        if defect is None:
            notes.append(f"папка {folder.name!r} не описана в DEFECTS — пропущена")
            continue
        for entry in sorted(folder.iterdir()):
            if entry.suffix.lower() != ".jpg":
                continue
            rel_path = overlay_to_rel_path(entry.name)
            path = pack_dir / rel_path
            if not path.exists():
                notes.append(f"{folder.name}: {rel_path} нет в паке")
                continue
            cases.append(Case(rel_path, path, defect, entry.name))
    return cases, notes
