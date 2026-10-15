"""CLI стенда «цветные страницы и фон»: фон на коврике без печати (``background``), сегментация цветных страниц без потери контента на краях (``segment``)."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.pipeline import init_worker
from research.colour_pages.background import book_change
from research.colour_pages.classify import Colour
from research.colour_pages.segment import PageStep, Rule, colour_region, content_outside, page_step

log = logging.getLogger(__name__)

# Кэш прогона пака-6 по умолчанию.
PACK6_CACHE = Path("/mnt/hotstore/scan_processing/scan_cropping_v2/pack6_v4_shadow/cache")
# Цвета листа «книга было | стало» (BGR): только было — красный, только стало — синий, обе — серый.
COLOUR_ONLY_BEFORE = (0, 0, 220)
COLOUR_ONLY_AFTER = (220, 90, 20)
COLOUR_BOTH = (150, 150, 150)
# Ширина картинки кадра на листе, px.
SHEET_FRAME_W = 900
# Наборы стенда по уровням цветности (ключи кэша).
SETS_DIR = Path(__file__).parent / "sets"
SET_NAMES = ("colour_full", "colour_partial", "colour_trace", "control")
# Цвета листа сегментации (BGR): бумага — зелёный, рост по палитре — красный, защита контента — жёлтый, контент за
# бумагой — синий.
COLOUR_PAPER = (0, 190, 0)
COLOUR_PALETTE = (0, 0, 230)
COLOUR_PROTECT = (0, 210, 230)
COLOUR_LOST = (230, 60, 0)


@click.group()
def main() -> None:
    """Стенд «цветные страницы и фон»."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def background_one(cache_dir: Path) -> tuple[str, float, float, float, float, np.ndarray | None]:
    """Книга кадра до и после правила фона (для пула): доли рамки и площади и картинка, если книга изменилась."""
    cv2.setNumThreads(1)
    try:
        change = book_change(cache_dir)
    except Exception:  # кадр с ошибкой не роняет стенд
        log.exception("%s: ошибка", cache_dir.name)
        return cache_dir.name, 0.0, 0.0, 0.0, 0.0, None
    if change is None:
        return cache_dir.name, 0.0, 0.0, 0.0, 0.0, None
    picture = None
    if (change.before != change.after).any():
        img = cv2.imread(str(cache_dir / "seg.png"))
        layer = img.copy()
        layer[change.before & change.after] = COLOUR_BOTH
        layer[change.before & ~change.after] = COLOUR_ONLY_BEFORE
        layer[~change.before & change.after] = COLOUR_ONLY_AFTER
        picture = cv2.addWeighted(layer, 0.5, img, 0.5, 0)
        picture = cv2.resize(picture, (SHEET_FRAME_W, int(SHEET_FRAME_W * img.shape[0] / img.shape[1])))
        cv2.putText(picture, cache_dir.name[-45:], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return (cache_dir.name, change.border_before, change.border_after, float(change.before.mean()),
            float(change.after.mean()), picture)  # fmt: skip


@main.command("background")
@click.option(
    "--cache",
    "cache_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=PACK6_CACHE,
    show_default=True,
    help="--cache прогона v2.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Куда писать background.csv и картинки изменившихся кадров.",
)
@click.option("--jobs", default=12, show_default=True, help="Процессов.")
def background_cmd(cache_root: Path, out_dir: Path, jobs: int) -> None:
    """Маска книги с правилом фона на коврике без печати: где изменилась, доля рамки и площади до и после."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [d for d in sorted(cache_root.iterdir()) if d.is_dir()]
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for key, b0, b1, a0, a1, picture in pool.map(background_one, dirs, chunksize=4):
            rows.append({"key": key, "border_before": b0, "border_after": b1, "area_before": a0, "area_after": a1,
                         "changed": picture is not None})  # fmt: skip
            if picture is not None:
                cv2.imwrite(str(out_dir / f"{key}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "background.csv", index=False)
    changed = table[table.changed]
    log.info("кадров %d, книга изменилась у %d", len(table), len(changed))
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        log.info("\n%s", changed.drop(columns="changed").round(3).to_string(index=False))


def read_set(path: Path) -> list[str]:
    """Ключи кэша из файла набора (строки с ``#`` — комментарии)."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def step_picture(step: PageStep) -> np.ndarray:
    """Картинка шага: бумага, рост по палитре, защита контента, контент за бумагой."""
    layer = step.img.copy()
    layer[step.paper] = COLOUR_PAPER
    layer[step.palette_added] = COLOUR_PALETTE
    layer[step.protected] = COLOUR_PROTECT
    layer[step.content & step.book & ~step.bg & ~(step.paper | step.protected)] = COLOUR_LOST
    picture = cv2.addWeighted(layer, 0.45, step.img, 0.55, 0)
    return cv2.resize(picture, (SHEET_FRAME_W, int(SHEET_FRAME_W * step.img.shape[0] / step.img.shape[1])))


def segment_one(args: tuple[Path, str, Path | None]) -> dict | None:
    """Кадр (для пула): шаг без правил и со всеми правилами, меры и лист «было | стало», если что-то изменилось."""
    cache_dir, set_name, out_dir = args
    cv2.setNumThreads(1)
    try:
        base = page_step(cache_dir, frozenset())
        variant = page_step(cache_dir, frozenset(Rule))
    except Exception:  # кадр с ошибкой не роняет стенд
        log.exception("%s: ошибка", cache_dir.name)
        return None
    if base is None or variant is None:
        return None
    full = colour_region([base, variant], frozenset({Colour.FULL}))
    every = colour_region([base, variant], frozenset(Colour))
    row = {
        "key": cache_dir.name,
        "set": set_name,
        "levels": "/".join(level.value for level in variant.levels),
        "poses_before": "/".join(p.pose.value for p in base.pages),
        "poses_after": "/".join(p.pose.value for p in variant.pages),
        "lost_full_before": content_outside(base, full),
        "lost_full_after": content_outside(variant, full),
        "lost_all_before": content_outside(base, every),
        "lost_all_after": content_outside(variant, every),
        "palette_mm2": float(variant.palette_added.sum() / 6.9**2),
        "paper_diff_px": int(((base.paper | base.protected) ^ (variant.paper | variant.protected)).sum()),
    }
    if out_dir is not None and row["paper_diff_px"] > 0:
        pair = np.hstack([step_picture(base), np.full((step_picture(base).shape[0], 10, 3), 255, np.uint8),
                          step_picture(variant)])  # fmt: skip
        cv2.putText(pair, cache_dir.name[-45:], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / set_name / f"{cache_dir.name}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return row


@main.command("segment")
@click.option(
    "--cache",
    "cache_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=PACK6_CACHE,
    show_default=True,
    help="--cache прогона v2.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Куда писать segment.csv и листы «было | стало» по наборам.",
)
@click.option(
    "--sets",
    "set_names",
    default=",".join(SET_NAMES),
    show_default=True,
    help="Наборы из sets/ через запятую; all — весь кэш.",
)
@click.option("--jobs", default=12, show_default=True, help="Процессов.")
def segment_cmd(cache_root: Path, out_dir: Path, set_names: str, jobs: int) -> None:
    """Шаг «бумага и страницы» без правил и со всеми правилами для цветных страниц: контент за бумагой, рост, позы."""
    import pandas as pd

    tasks = []
    for name in set_names.split(","):
        keys = (
            [d.name for d in sorted(cache_root.iterdir()) if d.is_dir()]
            if name == "all"
            else read_set(SETS_DIR / f"{name}.txt")
        )
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        tasks += [(cache_root / k, name, out_dir) for k in keys]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for r in pool.map(segment_one, tasks, chunksize=2) if r]
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "segment.csv", index=False)
    summary = table.groupby("set").agg(
        frames=("key", "size"),
        changed=("paper_diff_px", lambda v: int((v > 0).sum())),
        lost_full_before=("lost_full_before", "sum"),
        lost_full_after=("lost_full_after", "sum"),
        lost_all_before=("lost_all_before", "sum"),
        lost_all_after=("lost_all_after", "sum"),
        palette_mm2=("palette_mm2", "sum"),
        poses_changed=("poses_before", lambda v: int((v != table.loc[v.index, "poses_after"]).sum())),
    )
    with pd.option_context("display.width", 220):
        log.info("контент за бумагой (мм²) и рост по палитре по наборам:\n%s", summary.round(0))
