"""Команды стенда «текст разворотов»: ``ocr`` — чтение левой и правой полос каждого кадра, ``match`` — пары кадров двух съёмок по тексту."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
import threadpoolctl

from research.spread_ocr.match import best_page, sequence_offset, shingles, similarity_matrix
from research.spread_ocr.ocr import read_number, read_page
from research.spread_ocr.split import FOLD_MARGIN, NUMBER_FOLD_REACH, SpreadSplit, deskew, find_split, page_crops

# Сколько первых и последних строк полосы идёт в pages.csv.
EDGE_LINES = 3
# Страниц в выпуске (ISSN-строка обложки: «1—112»): номер вне 1..MAX_PAGE — мусор OCR.
MAX_PAGE = 112
# Пара по тексту голосует за нумерацию донора номером базы при сходстве не ниже этого.
MIN_NUMBER_VOTE_SIMILARITY = 0.5
# Две полосы донора с таким сходством считаются одной полосой (пересъёмка того же разворота).
SAME_CONTENT = 0.5


class Side(str, Enum):
    """Сторона разворота."""

    LEFT = "left"
    RIGHT = "right"


def _init_worker() -> None:
    """Инициализатор воркера: без hugepage и потоков BLAS/OpenCV (см. ``.claude/rules/gpu_and_pools.md``)."""
    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def page_json(out_dir: Path, set_name: str, frame: str, side: Side) -> Path:
    """Путь JSON-кэша прочитанной полосы.

    Args:
        out_dir: Корень выхода стенда.
        set_name: Имя съёмки.
        frame: Имя кадра без расширения.
        side: Сторона разворота.

    Returns:
        ``<out>/json/<съёмка>/<кадр>_<сторона>.json`` (папка не создаётся).
    """
    return out_dir / "json" / set_name / f"{frame}_{side.value}.json"


def number_band(gray: np.ndarray, split: SpreadSplit, side: Side, header_box: tuple[int, int] | None) -> np.ndarray:
    """Вырезать из полного кадра горизонтальную полосу колонтитула для дочитывания номера.

    Args:
        gray: Полный кадр в полутоне.
        split: Геометрия разворота.
        side: Сторона разворота; у левой полосы берётся запас ``NUMBER_FOLD_REACH`` за сгиб.
        header_box: (верх, высота) опоры колонтитула в px полосы; ``None`` — верхние 15 % бумаги.

    Returns:
        Полутоновая полоса кадра.
    """
    if header_box is None:
        top, bottom = split.top, split.top + int(0.15 * (split.bottom - split.top))
    else:
        # Полоса выровнена поворотом вокруг центра, поэтому запас по высоте берётся щедрый: ±3–4 высоты строки.
        y, height = header_box
        top, bottom = split.top + y - 3 * height, split.top + y + 4 * height
    if side == Side.LEFT:
        left, right = split.left, split.fold_x + NUMBER_FOLD_REACH
    else:
        left, right = split.fold_x - FOLD_MARGIN, split.right + 1
    return gray[max(top, 0) : max(bottom, top + 1), max(left, 0) : right]


def valid_number(value: int | None, side: Side) -> int | None:
    """Номер страницы, если он правдоподобен для стороны: левая — чётная, правая — нечётная, в пределах выпуска.

    Args:
        value: Прочитанный номер или ``None``.
        side: Сторона разворота.

    Returns:
        Тот же номер или ``None``.
    """
    if value is None or not 1 <= value <= MAX_PAGE:
        return None
    return value if (value % 2 == 0) == (side == Side.LEFT) else None


def ocr_frame(path: Path, set_name: str, out_dir: Path) -> list[dict]:
    """Прочитать обе полосы кадра и записать JSON-кэш и полный текст.

    Args:
        path: Путь JPG кадра (только чтение).
        set_name: Имя съёмки (подпапка выхода).
        out_dir: Корень выхода стенда.

    Returns:
        Две записи (левая, правая полоса) с номером, колонтитулом, строками, уверенностью, углом выравнивания
        и абсциссой сгиба.
    """
    frame = path.stem
    sides = (Side.LEFT, Side.RIGHT)
    cached = [page_json(out_dir, set_name, frame, side) for side in sides]
    # Идемпотентность: кадр с готовыми JSON обеих полос не перечитывается.
    if all(p.exists() for p in cached):
        return [json.loads(p.read_text()) for p in cached]
    bgr = cv2.imread(str(path))
    split = find_split(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    records = []
    for side, crop, json_path in zip(sides, page_crops(bgr, split), cached):
        upright, angle = deskew(crop)
        page = read_page(upright)
        # Номер: сначала из полного прохода, если он правдоподобен, иначе — отдельным проходом по полосе колонтитула.
        number = valid_number(page.page_number, side)
        source = "page" if number is not None else ""
        if number is None:
            number = valid_number(read_number(number_band(gray, split, side, page.header_box)), side)
            source = "band" if number is not None else ""
        record = {
            "set": set_name,
            "frame": frame,
            "side": side.value,
            "page_number": number,
            "page_number_source": source,
            "page_number_raw": page.page_number,
            "header": page.header,
            "lines": page.lines,
            "ocr_confidence": round(page.confidence, 1),
            "skew_deg": angle,
            "fold_x": split.fold_x,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
        # Полный текст полосы — отдельным файлом для чтения глазами.
        text_path = out_dir / "text" / set_name / f"{frame}_{side.value}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("\n".join(page.lines) + "\n")
        records.append(record)
    return records


def infer_spread_numbers(records: list[dict]) -> list[dict]:
    """Согласовать номера полос кадра: правая = левая + 1.

    Если прочитан номер только одной полосы — второй выводится из него. Если прочитаны оба, но не согласованы,
    верится правой полосе: у журнала номер левой стоит у сгиба и часто обрезан («16» → «6»), а номер правой —
    у наружного края; исключение — правый дочитан проходом по колонтитулу, а левый взят из полного прохода.

    Args:
        records: Записи полос (как из ``ocr_frame``), в любом порядке.

    Returns:
        Новые записи; у выведенного номера ``page_number_source = "spread"``.
    """
    by_frame: dict[tuple[str, str], dict[str, dict]] = {}
    for record in records:
        by_frame.setdefault((record["set"], record["frame"]), {})[record["side"]] = record
    # Согласованные номера (левая, правая) и источник правой/левой по кадру.
    resolved: dict[tuple[str, str], dict[str, tuple[int | None, str]]] = {}
    for key, pair in by_frame.items():
        left = pair.get(Side.LEFT.value, {})
        right = pair.get(Side.RIGHT.value, {})
        left_number, right_number = left.get("page_number"), right.get("page_number")
        left_source, right_source = left.get("page_number_source", ""), right.get("page_number_source", "")
        if left_number is not None and right_number is not None and right_number != left_number + 1:
            # Несогласованные номера: правой веры больше, кроме «правая из колонтитула, левая из полного прохода».
            if right_source == "band" and left_source == "page":
                right_number, right_source = left_number + 1, "spread"
            else:
                left_number, left_source = right_number - 1, "spread"
        elif left_number is None and right_number is not None:
            left_number, left_source = right_number - 1, "spread"
        elif right_number is None and left_number is not None:
            right_number, right_source = left_number + 1, "spread"
        resolved[key] = {Side.LEFT.value: (left_number, left_source), Side.RIGHT.value: (right_number, right_source)}
    result = []
    for record in records:
        number, source = resolved[(record["set"], record["frame"])][record["side"]]
        result.append({**record, "page_number": number, "page_number_source": source})
    return result


@click.group()
def main() -> None:
    """Стенд «текст разворотов»: OCR полос двух съёмок и сопоставление кадров по тексту."""


@main.command()
@click.option(
    "--set", "sets", multiple=True, required=True, help="Съёмка в виде ИМЯ=ПАПКА (кадры *.jpg); можно несколько раз."
)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда.")
@click.option("--jobs", default=12, show_default=True, help="Воркеров tesseract (по кадру на воркер).")
def ocr(sets: tuple[str, ...], out_dir: Path, jobs: int) -> None:
    """Прочитать левую и правую полосы каждого кадра; записать text/, json/ и pages.csv."""
    tasks = []
    for spec in sets:
        name, folder = spec.split("=", 1)
        for path in sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.JPG")):
            tasks.append((path, name))
    click.echo(f"Кадров: {len(tasks)}, воркеров: {jobs}")
    started = time.time()
    records = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        futures = {pool.submit(ocr_frame, path, name, out_dir): path for path, name in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            records.extend(future.result())
            if done % 20 == 0 or done == len(tasks):
                click.echo(f"  {done}/{len(tasks)} кадров, {time.time() - started:.0f} с")
    rows = []
    for record in infer_spread_numbers(records):
        lines = record["lines"]
        rows.append(
            {
                "set": record["set"],
                "frame": record["frame"],
                "side": record["side"],
                "page_number": record["page_number"],
                "page_number_source": record.get("page_number_source", ""),
                "header": record["header"],
                "n_lines": len(lines),
                "first_lines": " | ".join(lines[:EDGE_LINES]),
                "last_lines": " | ".join(lines[-EDGE_LINES:]),
                "ocr_confidence": record["ocr_confidence"],
            }
        )
    table = pd.DataFrame(rows).sort_values(["set", "frame", "side"])
    table["page_number"] = table["page_number"].astype("Int64")
    table.to_csv(out_dir / "pages.csv", index=False)
    click.echo(f"Записано {out_dir / 'pages.csv'}: {len(table)} полос")


def _load_pages(out_dir: Path, set_name: str) -> list[dict]:
    """Прочитать JSON-кэш всех полос съёмки, по порядку кадров и сторон."""
    paths = sorted((out_dir / "json" / set_name).glob("*.json"))
    return infer_spread_numbers([json.loads(p.read_text()) for p in paths])


@main.command()
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда.")
@click.option("--base", "base_set", required=True, help="Имя съёмки-базы (для каждой её полосы ищется пара).")
@click.option("--donor", "donor_set", required=True, help="Имя съёмки-донора.")
@click.option(
    "--akaze-pairs",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="CSV пар по ключевым точкам (base,donor,similarity) для сверки.",
)
def match(out_dir: Path, base_set: str, donor_set: str, akaze_pairs: Path | None) -> None:
    """Сопоставить полосы и кадры базы с донором по тексту; записать pairs_by_text.csv и pairs_compare.csv."""
    base = _load_pages(out_dir, base_set)
    donor = _load_pages(out_dir, donor_set)
    base_sh = [shingles("\n".join(p["lines"])) for p in base]
    donor_sh = [shingles("\n".join(p["lines"])) for p in donor]
    matrix = similarity_matrix(base_sh, donor_sh)
    donor_self = similarity_matrix(donor_sh, donor_sh)
    np.save(out_dir / "similarity.npy", matrix)

    # 1. Постранично: лучшая полоса донора для каждой полосы базы (любой кадр, любая сторона).
    found_pages = [best_page(matrix[i], donor_self, SAME_CONTENT) for i in range(len(base))]

    # 2. Нумерация по порядку кадров донора: у донора каждый разворот снят один раз, подряд, поэтому
    #    номер левой полосы = 2·k + сдвиг. Голосуют прочитанные номера донора и базы (через пару по тексту).
    donor_frames = sorted({p["frame"] for p in donor})
    frame_rank = {frame: k for k, frame in enumerate(donor_frames)}
    votes: list[tuple[int, int]] = []
    frame_votes: dict[str, list[int]] = {}
    read_pages = [(p, p) for p in donor] + [
        (page, donor[found.donor_index])
        for page, found in zip(base, found_pages)
        if found.donor_index >= 0 and found.similarity >= MIN_NUMBER_VOTE_SIMILARITY
    ]
    for page, donor_page in read_pages:
        # Голосуют только прочитанные (не выведенные из соседней полосы) номера.
        if page["page_number"] is None or page.get("page_number_source") not in ("page", "band"):
            continue
        left_number = page["page_number"] - (1 if page["side"] == Side.RIGHT.value else 0)
        votes.append((frame_rank[donor_page["frame"]], left_number))
        frame_votes.setdefault(donor_page["frame"], []).append(left_number - 2 * frame_rank[donor_page["frame"]])
    offset, support = sequence_offset(votes)
    click.echo(f"Нумерация донора: левая = 2·k {offset:+d} (голосов за сдвиг {support} из {len(votes)})")
    sequence_rows = []
    for frame, k in frame_rank.items():
        local = frame_votes.get(frame, [])
        sequence_rows.append(
            {
                "donor_frame": frame,
                "rank": k,
                "left_page_seq": 2 * k + offset,
                "right_page_seq": 2 * k + offset + 1,
                "votes": len(local),
                "votes_agree": sum(1 for value in local if value == offset),
            }
        )
    sequence = pd.DataFrame(sequence_rows)
    sequence.to_csv(out_dir / "donor_sequence.csv", index=False)
    seq_number = {
        (r.donor_frame, side.value): (r.left_page_seq if side == Side.LEFT else r.right_page_seq)
        for r in sequence.itertuples()
        for side in Side
    }

    page_rows = []
    for i, page in enumerate(base):
        found = found_pages[i]
        partner = donor[found.donor_index] if found.donor_index >= 0 else None
        page_rows.append(
            {
                "base_frame": page["frame"],
                "base_side": page["side"],
                "donor_frame": partner["frame"] if partner else "",
                "donor_side": partner["side"] if partner else "",
                "page_number_base": page["page_number"],
                "page_number_donor": partner["page_number"] if partner else None,
                "page_number_seq": seq_number[(partner["frame"], partner["side"])] if partner else None,
                "text_similarity": round(found.similarity, 3),
                "runner_up": round(found.runner_up, 3),
                "confidence": round(found.confidence, 3),
                "base_shingles": len(base_sh[i]),
            }
        )
    pages = pd.DataFrame(page_rows)
    for column in ("page_number_base", "page_number_donor", "page_number_seq"):
        pages[column] = pages[column].astype("Int64")
    pages.to_csv(out_dir / "pairs_by_text.csv", index=False)

    # 3. Покадрово: кадр донора с наибольшей суммой сходства «левая↔левая + правая↔правая».
    base_frames = sorted({p["frame"] for p in base})
    base_index = {(p["frame"], p["side"]): i for i, p in enumerate(base)}
    donor_index = {(p["frame"], p["side"]): j for j, p in enumerate(donor)}
    frame_score = np.zeros((len(base_frames), len(donor_frames)), dtype=np.float32)
    for a, bf in enumerate(base_frames):
        for b, df in enumerate(donor_frames):
            for side in Side:
                i, j = base_index.get((bf, side.value)), donor_index.get((df, side.value))
                if i is not None and j is not None:
                    frame_score[a, b] += matrix[i, j]
    # Сходство кадров донора между собой — чтобы пересъёмки одного разворота не считались соперниками.
    donor_frame_self = np.zeros((len(donor_frames), len(donor_frames)), dtype=np.float32)
    for a, fa in enumerate(donor_frames):
        for b, fb in enumerate(donor_frames):
            values = [
                donor_self[donor_index[(fa, s.value)], donor_index[(fb, s.value)]]
                for s in Side
                if (fa, s.value) in donor_index and (fb, s.value) in donor_index
            ]
            donor_frame_self[a, b] = max(values) if values else 0.0
    akaze = {}
    if akaze_pairs is not None:
        table = pd.read_csv(akaze_pairs)
        akaze = {Path(r.base).stem: (Path(r.donor).stem, float(r.similarity)) for r in table.itertuples()}
    by_page = {(r.base_frame, r.base_side): r for r in pages.itertuples()}
    compare_rows = []
    for a, bf in enumerate(base_frames):
        row = frame_score[a]
        best = int(np.argmax(row))
        score = float(row[best])
        other = donor_frame_self[best] < SAME_CONTENT
        other[best] = False
        runner_up = float(row[other].max()) if other.any() else 0.0
        text_pair = donor_frames[best] if score > 0 else ""
        left, right = by_page.get((bf, Side.LEFT.value)), by_page.get((bf, Side.RIGHT.value))
        akaze_pair = akaze.get(bf, ("", float("nan")))
        compare_rows.append(
            {
                "base_frame": bf,
                "akaze_pair": akaze_pair[0],
                "akaze_similarity": akaze_pair[1],
                "text_pair": text_pair,
                "text_frame_score": round(score, 3),
                "text_frame_confidence": round(max(score - runner_up, 0.0), 3),
                "agree": bool(akaze_pair[0] and akaze_pair[0] == text_pair),
                "akaze_pair_also_same_spread": bool(
                    akaze_pair[0]
                    and text_pair
                    and akaze_pair[0] in donor_frames
                    and donor_frame_self[donor_frames.index(akaze_pair[0]), best] >= SAME_CONTENT
                ),
                "left_best": f"{left.donor_frame}:{left.donor_side}:{left.text_similarity}" if left else "",
                "right_best": f"{right.donor_frame}:{right.donor_side}:{right.text_similarity}" if right else "",
                "page_left": left.page_number_base if left else None,
                "page_right": right.page_number_base if right else None,
                "pages_seq": (
                    f"{seq_number[(text_pair, Side.LEFT.value)]}-{seq_number[(text_pair, Side.RIGHT.value)]}"
                    if text_pair
                    else ""
                ),
            }
        )
    compare = pd.DataFrame(compare_rows)
    compare.to_csv(out_dir / "pairs_compare.csv", index=False)
    click.echo(
        f"Полос базы: {len(base)}, донора: {len(donor)}; кадров базы: {len(base_frames)}; "
        f"согласий с ключевыми точками: {int(compare['agree'].sum())}"
    )
