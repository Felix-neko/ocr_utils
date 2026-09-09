"""Отчёты по прогону: таблицы, CSV и каталог симлинков.

Главный артефакт — КАТАЛОГ СИМЛИНКОВ, а не таблица: находок ожидаются сотни, и решать по
ним всё равно придётся глазами, открыв картинку. Таблицы нужны для другого — чтобы видеть,
где детекторы разошлись, и понимать, кому из них верить.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

import cv2
import numpy as np

from ocr_utils.scan_cropping.image_io import IMAGE_EXTS
from ocr_utils.scan_markup.orientation.analysis import (
    CANDIDATE_CONFIDENCE,
    PageResult,
    rotation_counts,
    triggering_verdict,
)
from ocr_utils.scan_markup.orientation.detectors import ROTATION_NAMES, ROTATIONS
from ocr_utils.scan_markup.orientation.detectors.base import rotate_cw
from ocr_utils.scan_markup.orientation.pdf_pages import PdfPage, lookup

logger = logging.getLogger(__name__)

# Подкаталог для полос, у которых детекторы разошлись в стороне поворота. Отдельно, а не
# вперемешку с уверенными находками: разногласие — это то, что человек должен увидеть как
# разногласие, а не как ответ.
DISPUTED_DIR = "спорные"

# Сводный вердикт лежит в своём подкаталоге под этим именем.
COMBO_DIR = "combo"

# Все полосы, которые хоть один детектор счёл повёрнутыми, — вместе с теми, которые арбитр
# потом отверг. Отдельный каталог нужен как раз ради отвергнутых: по ним видно, не потерялось
# ли что-то настоящее, а по одному ``combo/`` этого не увидеть никогда.
CANDIDATES_DIR = "кандидаты"


class LinkDirError(Exception):
    """В каталоге симлинков нашлось что-то, кроме симлинков."""


def _slug(text: str) -> str:
    """Имя, годное для файла: пробелы и разделители — в подчёркивание."""
    return re.sub(r"[\s/\\]+", "_", text).strip("_")


def link_name(
    result: PageResult, pdf: PdfPage | None, rotate_cw: int, ordinal: int | None = None, suffix: str | None = None
) -> str:
    """Имя симлинка: год, выпуск, имя картинки, номер страницы, тип поворота.

    Порядок именно такой, и он не случаен: год и выпуск впереди собирают находки одного
    номера рядом, а номер страницы стоит после имени файла, потому что внутри выпуска они
    и так растут вместе — ``order_index`` берётся из сортировки имён.
    """
    parts = Path(result.rel_path).parts
    origin = "_".join(_slug(part) for part in parts[:-1])
    stem = Path(result.rel_path).stem
    number = pdf.page_number if pdf is not None else ordinal
    page = f"p{number:03d}" if number is not None else "pXXX"
    suffix = suffix if suffix is not None else Path(result.rel_path).suffix
    pieces = [piece for piece in (origin, _slug(stem), page, ROTATION_NAMES[rotate_cw]) if piece]
    return "_".join(pieces) + suffix


def write_link_dir(
    root: Path,
    results: Sequence[PageResult],
    detector_names: Sequence[str],
    pdf_pages: dict[str, PdfPage],
    link_root: Path | None = None,
) -> dict[str, int]:
    """Симлинки на найденные полосы: подкаталог на детектор, плюс combo и спорные.

    Повторный прогон чистит ТОЛЬКО свои симлинки. Наткнувшись на обычный файл, отказывается
    удалять что-либо: каталог мог оказаться не тем, а восстанавливать удалённое неоткуда.
    """
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name in [*detector_names, COMBO_DIR, DISPUTED_DIR, CANDIDATES_DIR]:
        counts[name] = _write_one(root / name, results, name, pdf_pages, link_root)
    return counts


def _target(result: PageResult, link_root: Path | None) -> Path:
    """Куда целить симлинк.

    ``--link-root`` нужен ровно для одного случая: разбираются заострённые JPEG, а смотреть
    глазами удобнее оригинальный TIFF. Раскладка каталогов у них одна, а РАСШИРЕНИЕ разное,
    поэтому подставить путь напрямую нельзя — имя ищется по основе. Не нашли ничего — целим
    в разобранный файл: симлинк в никуда хуже симлинка не туда, куда просили.
    """
    if link_root is None:
        return result.path.resolve()
    direct = link_root / result.rel_path
    if direct.exists():
        return direct.resolve()
    relative = Path(result.rel_path)
    siblings = sorted(
        path for path in (link_root / relative.parent).glob(f"{relative.stem}.*") if path.suffix.lower() in IMAGE_EXTS
    )
    return siblings[0].resolve() if siblings else result.path.resolve()


def _write_one(
    directory: Path, results: Sequence[PageResult], name: str, pdf_pages: dict[str, PdfPage], link_root: Path | None
) -> int:
    if directory.exists():
        for entry in directory.iterdir():
            if not entry.is_symlink():
                raise LinkDirError(f"в {directory} лежит не симлинк ({entry.name}) — каталог не тронут")
        for entry in directory.iterdir():
            entry.unlink()
    directory.mkdir(parents=True, exist_ok=True)

    written = 0
    for ordinal, result in enumerate(results, start=1):
        rotate = _rotation_for(result, name)
        if rotate is None:
            continue
        pdf = lookup(pdf_pages, result.rel_path)
        target = _target(result, link_root)
        link = directory / link_name(result, pdf, rotate, ordinal, suffix=target.suffix)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        written += 1
    return written


def _rotation_for(result: PageResult, name: str) -> int | None:
    """Какой поворот показывает этот подкаталог для этой полосы, или None — не показывает."""
    if name == CANDIDATES_DIR:
        if not result.candidate:
            return None
        # У отвергнутой полосы сводного поворота нет вовсе, поэтому в имя идёт тот, из-за
        # которого она в кандидаты и попала.
        if result.combo is not None and result.combo.rotate_cw != 0:
            return result.combo.rotate_cw
        triggering = triggering_verdict(result)
        return triggering[1].rotate_cw if triggering else None
    if name == DISPUTED_DIR:
        return result.combo.rotate_cw if result.disputed and result.combo is not None else None
    if name == COMBO_DIR:
        if result.disputed or result.combo is None or result.combo.rotate_cw == 0:
            return None
        return result.combo.rotate_cw
    verdict = result.verdicts.get(name)
    if verdict is None or verdict.rotate_cw == 0 or verdict.confidence <= 0.0:
        return None
    return verdict.rotate_cw


def metric_columns(results: Sequence[PageResult], detector_names: Sequence[str]) -> dict[str, list[str]]:
    """Какие метрики каждый детектор вообще выдал — по ним строятся колонки CSV."""
    columns: dict[str, set[str]] = {name: set() for name in detector_names}
    for result in results:
        for name, verdict in result.verdicts.items():
            if name in columns:
                columns[name].update(verdict.metrics)
    return {name: sorted(keys) for name, keys in columns.items()}


def write_csv(
    path: Path, results: Sequence[PageResult], detector_names: Sequence[str], pdf_pages: dict[str, PdfPage]
) -> None:
    """Все полосы и все метрики. Именно все: по этому файлу калибруются пороги."""
    metrics = metric_columns(results, detector_names)
    header = ["полоса", "pdf", "стр", "ширина", "высота", "combo", "combo_conf", "сторону_дал", "спорная"]
    for name in detector_names:
        header += [f"{name}", f"{name}_conf", f"{name}_только_ось", f"{name}_замечание"]
        header += [f"{name}_{key}" for key in metrics[name]]
    header.append("ошибка")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for result in results:
            pdf = lookup(pdf_pages, result.rel_path)
            combo = result.combo
            row = [
                result.rel_path,
                pdf.pdf_name if pdf else "",
                pdf.page_number if pdf else "",
                result.width,
                result.height,
                ROTATION_NAMES[combo.rotate_cw] if combo else "",
                f"{combo.confidence:.3f}" if combo else "",
                result.sign_from,
                "да" if result.disputed else "",
            ]
            for name in detector_names:
                verdict = result.verdicts.get(name)
                if verdict is None:
                    row += ["", "", "", ""] + [""] * len(metrics[name])
                    continue
                row += [
                    ROTATION_NAMES[verdict.rotate_cw],
                    f"{verdict.confidence:.3f}",
                    "да" if verdict.axis_only else "",
                    verdict.note,
                ]
                row += [f"{verdict.metrics[key]:.4f}" if key in verdict.metrics else "" for key in metrics[name]]
            row.append(result.error)
            writer.writerow(row)
    logger.info("CSV: %s", path)


def _render(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Markdown-таблица."""
    line = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([line, rule, *body])


def counts_table(results: Sequence[PageResult], detector_names: Sequence[str]) -> str:
    headers = ["детектор", *(ROTATION_NAMES[rotation] for rotation in ROTATIONS), "молчит"]
    rows = []
    for name in [*detector_names, "combo"]:
        counts = rotation_counts(results, name)
        silent = len(results) - sum(counts.values())
        rows.append([name, *(counts[rotation] for rotation in ROTATIONS), silent])
    return _render(headers, rows)


def agreement_table(results: Sequence[PageResult], detector_names: Sequence[str]) -> str:
    """Доля совпадений попарно — только по полосам, где ОБА высказались.

    Молчание — не мнение: считать его согласием значило бы наградить детектор, который
    молчит чаще других, а он как раз и менее полезен.
    """
    headers = ["", *detector_names]
    rows = []
    for left in detector_names:
        row = [left]
        for right in detector_names:
            if left == right:
                row.append("—")
                continue
            same = total = 0
            for result in results:
                a, b = result.verdicts.get(left), result.verdicts.get(right)
                if a is None or b is None or a.confidence <= 0.0 or b.confidence <= 0.0:
                    continue
                total += 1
                # Детектор, различающий только ось, сравнивается по оси: требовать от него
                # совпадения по стороне бессмысленно, он о ней и не высказывался.
                if a.axis_only or b.axis_only:
                    same += (a.rotate_cw % 180) == (b.rotate_cw % 180)
                else:
                    same += a.rotate_cw == b.rotate_cw
            row.append(f"{same / total:.3f}" if total else "—")
        rows.append(row)
    return _render(headers, rows)


def _link(result: PageResult) -> str:
    return f"[{result.rel_path}](file://{quote(str(result.path.resolve()))})"


def findings_table(
    results: Sequence[PageResult],
    pdf_pages: dict[str, PdfPage],
    limit: int,
    detector_names: Sequence[str],
    skip_disputed: bool = False,
) -> str:
    headers = ["полоса", "pdf", "стр", "поворот", "увер.", "сторону дал", *detector_names]
    rows = []
    for result in sorted(results, key=lambda item: -(item.combo.confidence if item.combo else 0.0)):
        if result.combo is None or result.combo.rotate_cw == 0:
            continue
        if skip_disputed and result.disputed:
            continue
        pdf = lookup(pdf_pages, result.rel_path)
        row = [
            _link(result),
            pdf.pdf_name if pdf else "",
            pdf.page_number if pdf else "",
            ROTATION_NAMES[result.combo.rotate_cw] + (" ?" if result.disputed else ""),
            f"{result.combo.confidence:.2f}",
            result.sign_from or "—",
        ]
        for name in detector_names:
            verdict = result.verdicts.get(name)
            row.append("—" if verdict is None else f"{ROTATION_NAMES[verdict.rotate_cw]} {verdict.confidence:.2f}")
        rows.append(row)
        if len(rows) >= limit:
            break
    return _render(headers, rows)


def markdown_report(
    results: Sequence[PageResult],
    detector_names: Sequence[str],
    pdf_pages: dict[str, PdfPage],
    root: Path,
    link_dir: Path | None,
    link_counts: dict[str, int],
    limit: int = 400,
    arbiters: Sequence[str] = (),
) -> str:
    errors = [result for result in results if result.error]
    found = [result for result in results if result.rotated and not result.disputed]
    disputed = [result for result in results if result.disputed]
    blocks = [
        f"# Полосы под поворот: {root}",
        "",
        f"Полос: {len(results)}. Найдено под поворот: {len(found)}. Спорных: {len(disputed)}. "
        f"Ошибок чтения: {len(errors)}.",
        "",
    ]
    if link_dir is not None:
        blocks += [f"Симлинки: `{link_dir}`", "", _render(["каталог", "полос"], sorted(link_counts.items())), ""]
    blocks += [
        "## 1. Что нашёл каждый детектор",
        "",
        counts_table(results, detector_names),
        "",
        "«Молчит» — полосы, где детектор не набрал уверенности: мало текста, чужой скрипт, "
        "не различил ось. Это не ошибка, а честный отказ, и сравнивать детекторы надо с "
        "оглядкой на него."
        + (
            f" У арбитров ({', '.join('`' + name + '`' for name in arbiters)}) молчание значит "
            "другое: они и не смотрели полосу, потому что быстрые детекторы не сочли её кандидатом."
            if arbiters
            else ""
        ),
        "",
        "## 2. Как отбирались кандидаты",
        "",
        f"Кандидатом полоса становится, если ХОТЬ ОДИН быстрый детектор назвал её повёрнутой "
        f"с уверенностью не ниже {CANDIDATE_CONFIDENCE:.2f}. Согласия не требуется намеренно: "
        "пропущенная на этом шаге полоса не всплывёт уже никогда, а лишний кандидат стоит "
        "одного прогона арбитра, то есть секунд.",
        "",
        "Дальше по кандидатам идёт арбитр — он читает полосу на всех четырёх поворотах, и он "
        "же ОТМЕНЯЕТ находку, если поворот не нужен. Поэтому находок заметно меньше, чем "
        "кандидатов, и разница между ними — не потери, а отсев.",
        "",
        f"Все кандидаты разложены симлинками в `{CANDIDATES_DIR}/`, принятые — в `{COMBO_DIR}/`. "
        f"Разность этих двух каталогов и есть отвергнутое: по ней видно, не потерялось ли "
        "что-то настоящее, а по одному `combo/` этого не увидеть.",
        "",
        _render(
            ["", "полос"],
            [
                ["кандидатов", sum(1 for r in results if r.candidate)],
                ["из них признано повёрнутыми", sum(1 for r in results if r.rotated and not r.disputed)],
                ["спорных", sum(1 for r in results if r.disputed)],
                ["отвергнуто арбитром", sum(1 for r in results if r.candidate and not r.rotated)],
            ],
        ),
        "",
        "## 3. Попарное согласие",
        "",
        agreement_table(results, detector_names),
        "",
        "Доля совпавших ответов среди полос, где ВЫСКАЗАЛИСЬ ОБА. Детекторы, различающие "
        "только ось, сравниваются по оси.",
        "",
        f"## 4. Находки (до {limit}); спорные вынесены отдельно",
        "",
        findings_table(results, pdf_pages, limit, detector_names, skip_disputed=True),
    ]
    section = 5
    if disputed:
        blocks += [
            "",
            f"## {section}. Спорные: ось видна, сторона названа по-разному",
            "",
            findings_table(disputed, pdf_pages, limit, detector_names),
        ]
        section += 1
    if errors:
        blocks += [
            "",
            f"## {section}. Не прочитались",
            "",
            _render(["полоса", "ошибка"], [[result.rel_path, result.error] for result in errors[:limit]]),
        ]
    return "\n".join(blocks) + "\n"


def validation_report(trials, detector_names: Sequence[str], root: Path, pages: int, arbiter: str | None = None) -> str:
    """Отчёт синтетической проверки: точность, матрицы ошибок, цена одной полосы."""
    from ocr_utils.scan_markup.orientation.validation import accuracy, confusion

    blocks = [
        f"# Проверка детекторов ориентации: {root}",
        "",
        f"Эталон: {pages} полос. Каждая подана во всех четырёх поворотах — {len(trials)} попыток.",
        "",
        (
            f"Прямизну эталонных полос подтвердил `{arbiter}`: он читает сам текст, и его мнение "
            "не зависит от остальных. Отбирать эталон общим согласием было бы нельзя — так из "
            "него выпали бы ровно те полосы, на которых детекторы ошибаются, и точность каждого "
            "оказалась бы завышена тем сильнее, чем чаще он ошибается."
            if arbiter
            else "ВНИМАНИЕ: арбитра в наборе не было, эталон отобран общим согласием детекторов. "
            "Это завышает их точность — сравнивать можно только с оглядкой на это."
        ),
        "",
        "ЧЕГО ЭТА ПРОВЕРКА НЕ ПОКАЗЫВАЕТ: как детектор ведёт себя на настоящей боковой полосе, "
        "где текста одна подпись, а всё остальное — чертёж. Повёрнутая текстовая полоса и полоса "
        "с боком напечатанным генпланом — разные задачи, и первая заметно легче.",
        "",
        "## 1. Точность",
        "",
    ]
    rows = []
    for name in detector_names:
        share, spoke, total = accuracy(trials, name)
        exact, _, _ = accuracy(trials, name, strict=True)
        seconds = [trial.seconds[name] for trial in trials if name in trial.seconds]
        median = sorted(seconds)[len(seconds) // 2] if seconds else float("nan")
        rows.append(
            [
                name,
                "—" if exact != exact else f"{exact:.3f}",
                "—" if share != share else f"{share:.3f}",
                f"{spoke}/{total}",
                "—" if median != median else f"{median:.2f} с",
            ]
        )
    blocks += [
        _render(["детектор", "точно", "по своему обещанию", "высказался", "медиана времени"], rows),
        "",
        "«Точно» — ответ совпал с ожидаемым один в один. «По своему обещанию» — детектору, "
        "поднявшему `axis_only`, зачтён ответ, совпавший по ОСИ: он и не брался называть "
        "сторону, и спрашивать с него за неё нечестно. Расхождение этих двух колонок и есть "
        "мера того, сколько работы детектор оставляет другим; у `profile` и `surya_lines` она "
        "ровно вдвое, потому что сторону они не различают принципиально.",
        "",
        "Столбец «высказался» отдельно, потому что детектор, который молчит на трёх полосах из "
        "четырёх, но не ошибается на четвёртой, и детектор, который всегда отвечает и почти не "
        "ошибается, — разные инструменты, и одной точностью они не различаются.",
        "",
        "## 2. Матрицы ошибок",
        "",
    ]
    for name in detector_names:
        table = confusion(trials, name)
        headers = ["ждали \\ ответил", *(ROTATION_NAMES[r] for r in ROTATIONS), "молчит"]
        body = [[ROTATION_NAMES[expected], *(row[key] for key in headers[1:])] for expected, row in table.items()]
        blocks += [f"### {name}", "", _render(headers, body), ""]
    return "\n".join(blocks) + "\n"


# --- Контактный лист --------------------------------------------------------
# Миниатюры находок ОДНИМ листом. Открывать сотни симлинков по одному никто не станет,
# а на листе видно сразу всё. Ключевая деталь: миниатюра показывается УЖЕ ПОВЁРНУТОЙ на
# предложенный угол — тогда проверка сводится к «читается или нет», а не к разглядыванию
# боковой полосы с попыткой представить её развёрнутой.
SHEET_TILE_PX = 300
SHEET_COLUMNS = 6
SHEET_CAPTION_PX = 22

# Плиток на лист. Больше — и лист перестаёт открываться просмотрщиком: 60 плиток по 300 px
# это уже 1800x3000, а находок бывают сотни.
SHEET_PER_PAGE = 60

# Подпись рисуется шрифтом OpenCV, который кириллицу не умеет вовсе — вместо букв рисует
# знаки вопроса. Поэтому в подпись идёт только латиница и цифры, а имена полос пака-1
# как раз такие: 1967_01_IMG_0043_2R.
CAPTION_SCALE = 0.38


def _tile(image: np.ndarray, rotate: int, caption: str) -> np.ndarray:
    """Одна плитка: полоса, повёрнутая на предложенный угол, с подписью сверху."""
    turned = rotate_cw(image, rotate)
    height, width = turned.shape[:2]
    scale = SHEET_TILE_PX / max(height, width)
    turned = cv2.resize(
        turned, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )
    if turned.ndim == 2:
        turned = cv2.cvtColor(turned, cv2.COLOR_GRAY2BGR)
    pad_x = SHEET_TILE_PX - turned.shape[1]
    canvas = cv2.copyMakeBorder(
        turned,
        SHEET_CAPTION_PX,
        SHEET_TILE_PX - turned.shape[0],
        pad_x // 2,
        pad_x - pad_x // 2,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    cv2.putText(canvas, caption[:40], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, CAPTION_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(160, 160, 160))


def _caption(result: PageResult, pdf_pages: dict[str, PdfPage]) -> str:
    pdf = lookup(pdf_pages, result.rel_path)
    page = f"p{pdf.page_number:03d}" if pdf else "p???"
    mark = "?" if result.disputed else ""
    rotation = ROTATION_NAMES[result.combo.rotate_cw]
    return f"{Path(result.rel_path).stem} {page} {rotation}{mark} {result.combo.confidence:.2f}"


def _grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    """Плитки в сетку; неполный последний ряд добивается белым до ширины остальных."""
    blank = np.full_like(tiles[0], 255)
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def contact_sheet(
    base_path: Path,
    results: Sequence[PageResult],
    pdf_pages: dict[str, PdfPage],
    columns: int = SHEET_COLUMNS,
    per_page: int = SHEET_PER_PAGE,
) -> list[Path]:
    """Листы миниатюр находок, уже повёрнутых на предложенный угол."""
    chosen = [result for result in results if result.rotated]
    written: list[Path] = []
    for sheet_index in range(0, len(chosen), per_page):
        tiles = []
        for result in chosen[sheet_index : sheet_index + per_page]:
            image = cv2.imread(str(result.path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            tiles.append(_tile(image, result.combo.rotate_cw, _caption(result, pdf_pages)))
        if not tiles:
            continue
        path = base_path.with_name(f"{base_path.stem}_{sheet_index // per_page + 1:02d}{base_path.suffix}")
        cv2.imwrite(str(path), _grid(tiles, columns))
        written.append(path)
    logger.info("Контактных листов: %d", len(written))
    return written
