"""CLI стенда «в кадре одна страница»: признаки по кэшам v2 (``features``), листы миниатюр по поясам (``sheets``), сверка правила с разметкой (``evaluate``)."""

from __future__ import annotations

import csv
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from ocr_utils.page_layout.overlay_frame import FONT_PATH, header_strip
from ocr_utils.scan_cropping.v2.pipeline import init_worker
from research.single_page.features import frame_features, frame_thumb
from research.single_page.labels import Label, read_labels
from research.single_page.rule import RuleParams, Verdict, typical_widths, verdict_of

log = logging.getLogger(__name__)

# Воркеров по умолчанию: кадр — три npz по ~5 Мпикс и PNG сегментации; упор в диск SSD, не в счёт.
DEFAULT_JOBS = 12
# Лист отсмотра: миниатюр в строке и строк на листе.
SHEET_COLS = 6
SHEET_ROWS = 5
# Рамка миниатюры на листе, px.
TILE_W = 360
TILE_H = 270
# Кегль подписи миниатюры, px.
CAPTION_FONT = 12


def parse_cache(spec: str) -> tuple[str, Path]:
    """Разбор опции ``--cache имя=путь``.

    Args:
        spec: Строка ``имя=путь``.

    Returns:
        ``(имя набора, корень кэша)``.

    Raises:
        click.BadParameter: Нет ``=``.
    """
    if "=" not in spec:
        raise click.BadParameter(f"ожидается имя=путь: {spec}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def cache_keys(cache: Path) -> list[str]:
    """Ключи готовых кадров кэша: папки с ``pages.json``, ``paper.npz`` и ``sam3.npz`` (на любой глубине).

    Args:
        cache: Корень кэша прогона v2.

    Returns:
        Относительные пути папок кадров, по порядку.
    """
    keys = []
    for pages in cache.rglob("pages.json"):
        root = pages.parent
        if (root / "paper.npz").exists() and (root / "sam3.npz").exists() and (root / "seg.png").exists():
            keys.append(root.relative_to(cache).as_posix())
    return sorted(keys)


def feature_task(args: tuple[str, Path, str, Path]) -> dict | None:
    """Признаки кадра и его миниатюра (для пула).

    Args:
        args: ``(набор, корень кэша, ключ кадра, папка миниатюр набора)``.

    Returns:
        Строка CSV или None (ошибка — в лог).
    """
    dataset, cache, key, thumbs = args
    try:
        row = frame_features(dataset, cache, key).row()
        path = thumbs / f"{key}.jpg"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame_thumb(cache, key), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return row
    except Exception:  # noqa: BLE001 — кадр с ошибкой не роняет прогон
        log.exception("%s/%s: ошибка", dataset, key)
        return None


def write_csv(path: Path, rows: list[dict]) -> None:
    """CSV по строкам (колонки — объединение ключей по порядку появления).

    Args:
        path: Файл.
        rows: Строки.
    """
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    """Строки CSV признаков с числами вместо строк, где это числа.

    Args:
        path: ``features.csv``.

    Returns:
        Список словарей.
    """
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            row: dict = {}
            for k, v in r.items():
                try:
                    row[k] = float(v) if k not in ("dataset", "key", "source", "narrow_pose", "verdict") else v
                except ValueError:
                    row[k] = v
            out.append(row)
    return out


@click.group()
def main() -> None:
    """Стенд «в кадре одна страница» по кэшам scan_cropping v2 (только CPU, кэши только читаются)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--cache", "caches", multiple=True, required=True, help="Набор: имя=корень кэша v2 (можно несколько).")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Папка выхода стенда.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def features(caches: tuple[str, ...], out: Path, jobs: int) -> None:
    """Признаки всех кадров наборов → ``features.csv`` (с вердиктом правила) и миниатюры ``thumbs/<набор>/``."""
    tasks = []
    for spec in caches:
        name, cache = parse_cache(spec)
        keys = cache_keys(cache)
        log.info("%s: кадров %d (%s)", name, len(keys), cache)
        tasks += [(name, cache, k, out / "thumbs" / name) for k in keys]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for r in pool.map(feature_task, tasks, chunksize=4) if r is not None]
    rows = with_verdicts(rows, RuleParams())
    write_csv(out / "features.csv", rows)
    log.info("записано %d строк → %s", len(rows), out / "features.csv")


def with_verdicts(rows: list[dict], params: RuleParams) -> list[dict]:
    """Строки с типичной шириной страницы набора, относительными ширинами и вердиктом правила.

    Args:
        rows: Строки признаков.
        params: Пороги правила.

    Returns:
        Новые строки с колонками ``typical_w_mm``, ``wide_rel``, ``narrow_rel``, ``verdict``.
    """
    typical = typical_widths(rows, params)
    out = []
    for r in rows:
        v = verdict_of(r, typical[r["dataset"]], params)
        wide = max(r["w_left_mm"], r["w_right_mm"])
        narrow = min(r["w_left_mm"], r["w_right_mm"]) if r["n_pages"] >= 2 else 0.0
        out.append({**r, "typical_w_mm": round(typical[r["dataset"]], 1),
                    "wide_rel": round(wide / typical[r["dataset"]], 3),
                    "narrow_rel": round(narrow / typical[r["dataset"]], 3), "verdict": v.value})  # fmt: skip
    return out


def caption(r: dict) -> list[str]:
    """Две строки подписи миниатюры: ключ и главные признаки.

    Args:
        r: Строка признаков с вердиктом.

    Returns:
        Строки подписи.
    """
    return [f"{r['dataset']}/{r['key']}"[-52:],
            f"p{int(r['n_pages'])} i{int(r['n_inst'])} wr={r['width_ratio']:.2f} W={r['wide_rel']:.2f} "
            f"ps={r['print_spine']:.2f} a={r['paper_aspect']:.2f} {r['verdict']}"]  # fmt: skip


def letterbox(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Картинка, вписанная в рамку ``width × height`` без искажения пропорций (поля — серые).

    Args:
        img: Картинка BGR.
        width: Ширина рамки, px.
        height: Высота рамки, px.

    Returns:
        Картинка BGR размера рамки.
    """
    k = min(width / img.shape[1], height / img.shape[0])
    small = cv2.resize(
        img, (max(int(img.shape[1] * k), 1), max(int(img.shape[0] * k), 1)), interpolation=cv2.INTER_AREA
    )
    out = np.full((height, width, 3), 60, np.uint8)
    y0, x0 = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
    out[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
    return out


def caption_strip(lines: list[str], width: int) -> np.ndarray:
    """Подпись миниатюры мелким шрифтом DejaVu (кириллица): белая полоса по строке на элемент.

    Args:
        lines: Строки подписи.
        width: Ширина полосы, px.

    Returns:
        Полоса BGR.
    """
    font = ImageFont.truetype(FONT_PATH, CAPTION_FONT)
    picture = Image.new("RGB", (width, 4 + (CAPTION_FONT + 4) * len(lines)), (255, 255, 255))
    draw = ImageDraw.Draw(picture)
    for k, line in enumerate(lines):
        draw.text((3, 2 + (CAPTION_FONT + 4) * k), line, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)


def draw_sheet(rows: list[dict], thumbs: Path, title: str) -> np.ndarray:
    """Лист миниатюр с подписями.

    Args:
        rows: Строки признаков (не больше ``SHEET_COLS × SHEET_ROWS``).
        thumbs: Корень миниатюр (``thumbs/<набор>/<ключ>.jpg``).
        title: Заголовок листа.

    Returns:
        Картинка BGR.
    """
    tiles = []
    for r in rows:
        img = cv2.imread(str(thumbs / r["dataset"] / f"{r['key']}.jpg"))
        if img is None:
            continue
        img = letterbox(img, TILE_W, TILE_H)
        cap = caption_strip(caption(r), img.shape[1])
        tiles.append(cv2.copyMakeBorder(np.vstack([img, cap]), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(80, 80, 80)))
    if not tiles:
        return np.full((60, 600, 3), 255, np.uint8)
    blank = np.full_like(tiles[0], 255)
    while len(tiles) % SHEET_COLS:
        tiles.append(blank)
    grid = np.vstack([np.hstack(tiles[i : i + SHEET_COLS]) for i in range(0, len(tiles), SHEET_COLS)])
    return np.vstack([header_strip([title], grid.shape[1]), grid])


def band_query(rows: list[dict], expr: str) -> list[dict]:
    """Строки, для которых выражение по колонкам истинно (``eval`` в пространстве имён строки).

    Args:
        rows: Строки признаков.
        expr: Выражение Python по именам колонок (например ``width_ratio < 0.3 and n_inst <= 1``).

    Returns:
        Отобранные строки.
    """
    out = []
    for r in rows:
        env = {k.replace("-", "_"): v for k, v in r.items()}
        env["isnan"] = math.isnan
        if eval(expr, {"__builtins__": {}}, env):  # noqa: S307 — выражение пишет сам пользователь стенда
            out.append(r)
    return out


def issue_of(key: str) -> str:
    """Выпуск кадра по ключу кэша: ключ без последней части (части — через ``/`` или ``__``).

    Args:
        key: Ключ кадра (``1975/01__IMG_0001``, ``МТС__1990__01__IMG_0828``, ``01 оплачено/IMG_0925``).

    Returns:
        Ключ выпуска.
    """
    parts = key.replace("__", "/").split("/")
    return "/".join(parts[:-1])


def frames_at_issue_edges(rows: list[dict], count: int) -> list[dict]:
    """Первые и последние ``count`` кадров каждого выпуска (обложки — главные кандидаты в одну страницу).

    Args:
        rows: Строки признаков.
        count: Кадров с каждого края выпуска.

    Returns:
        Отобранные строки в порядке набора, выпуска и кадра.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["dataset"], issue_of(r["key"])), []).append(r)
    out = []
    for gk in sorted(groups):
        frames = sorted(groups[gk], key=lambda r: r["key"])
        picked = frames if len(frames) <= 2 * count else frames[:count] + frames[-count:]
        out += picked
    return out


@main.command()
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Папка стенда (features.csv, thumbs/).")
@click.option("--name", required=True, help="Имя пояса: листы ляжут в sheets/<имя>_NN.jpg.")
@click.option("--where", "expr", default="True", show_default=True, help="Отбор строк выражением по колонкам.")
@click.option("--sort", "sort_key", default="width_ratio", show_default=True,
              help="Колонка сортировки; key — порядок кадров.")  # fmt: skip
@click.option("--issue-edges", type=int, default=0, show_default=True,
              help="Только первые и последние N кадров каждого выпуска (0 — все).")  # fmt: skip
@click.option("--limit", type=int, default=600, show_default=True, help="Не больше стольких кадров.")
def sheets(out: Path, name: str, expr: str, sort_key: str, issue_edges: int, limit: int) -> None:
    """Листы миниатюр пояса признаков на отсмотр глазами (+ список ключей пояса ``sheets/<имя>.txt``)."""
    rows = band_query(read_rows(out / "features.csv"), expr)
    if issue_edges:
        rows = frames_at_issue_edges(rows, issue_edges)
    order = (
        (lambda r: (r["dataset"], r["key"])) if sort_key == "key" else (lambda r: (r[sort_key], r["dataset"], r["key"]))
    )
    chosen = sorted(rows, key=order)[:limit]
    folder = out / "sheets"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.txt").write_text("".join(f"{r['dataset']}/{r['key']}\n" for r in chosen))
    per = SHEET_COLS * SHEET_ROWS
    for n in range(0, len(chosen), per):
        sheet = draw_sheet(chosen[n : n + per], out / "thumbs", f"{name}: {expr} ({n + 1}-{n + per} of {len(chosen)})")
        cv2.imwrite(str(folder / f"{name}_{n // per:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    log.info("%s: %d кадров, %d листов → %s", name, len(chosen), math.ceil(len(chosen) / per), folder)


@main.command()
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Папка стенда (features.csv).")
@click.option("--labels", "label_files", type=click.Path(path_type=Path, exists=True), multiple=True, required=True,
              help="Файлы разметки sets/*.txt.")  # fmt: skip
def evaluate(out: Path, label_files: tuple[Path, ...]) -> None:
    """Сверка правила с разметкой: таблица «разметка × вердикт», промахи и срабатывания вне разметки (→ evaluate.txt)."""
    rows = with_verdicts(read_rows(out / "features.csv"), RuleParams())
    write_csv(out / "features.csv", rows)
    labels: dict[str, Label] = {}
    for f in label_files:
        labels |= read_labels(f)
    lines = evaluation_lines(rows, labels)
    (out / "evaluate.txt").write_text("\n".join(lines) + "\n")
    click.echo("\n".join(lines))


def evaluation_lines(rows: list[dict], labels: dict[str, Label]) -> list[str]:
    """Строки сверки: таблица «разметка × вердикт», промахи по разметке, срабатывания на неразмеченных (развороты).

    Неразмеченный кадр считается разворотом (разметка перечисляет все неразвороты наборов).

    Args:
        rows: Строки признаков с вердиктом.
        labels: Ключ ``набор/кадр`` → метка.

    Returns:
        Строки отчёта.
    """
    by_key = {f"{r['dataset']}/{r['key']}": r for r in rows}
    missing = [k for k in labels if k not in by_key]
    counts: dict[tuple[str, str], int] = {}
    misses = []
    for key, r in sorted(by_key.items()):
        label = labels.get(key, Label.SPREAD)
        counts[(label.value, r["verdict"])] = counts.get((label.value, r["verdict"]), 0) + 1
        if label is not Label.MAYBE and label.value != r["verdict"]:
            misses.append(f"  ✗ {key}: разметка {label.value}, правило {r['verdict']}")
    verdicts = [v.value for v in Verdict]
    lines = [f"кадров {len(by_key)}, размечено {len(labels)} (нет в признаках: {len(missing)})",
             "разметка \\ вердикт  " + " ".join(f"{v:>8s}" for v in verdicts)]  # fmt: skip
    for lab in [lb.value for lb in Label]:
        cells = [counts.get((lab, v), 0) for v in verdicts]
        if any(cells):
            lines.append(f"{lab:20s}" + " ".join(f"{c:8d}" for c in cells))
    lines.append(f"промахов: {len(misses)}")
    return lines + misses
