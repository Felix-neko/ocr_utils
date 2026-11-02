"""Команды стенда псевдоглифов: ``features`` — кандидаты и меры по вырезкам, ``collect`` — общий CSV."""

from __future__ import annotations

import time
from concurrent.futures import as_completed
from pathlib import Path

import click

from research.pseudo_glyphs.crops import HP1991_CANVAS, OUT_ROOT, list_crops
from research.pseudo_glyphs.pool import DEFAULT_JOBS, make_pool


@click.group()
def main() -> None:
    """Стенд «псевдоглифы» для разбора текстовых блоков."""


@main.command()
@click.option("--canvas-root", type=click.Path(path_type=Path, exists=True), default=HP1991_CANVAS, show_default=True,
              help="Корень вырезок холста (<выпуск>/<кадр>_<L|R>.png + .json).")  # fmt: skip
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_ROOT / "hp1991", show_default=True,
              help="Корень выхода стенда.")  # fmt: skip
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
@click.option("--limit", type=int, default=0, help="Не больше стольких страниц (0 — все).")
@click.option("--force/--no-force", default=False, help="Пересчитать страницы, у которых выход уже есть.")
def features(canvas_root: Path, out_dir: Path, jobs: int, limit: int, force: bool) -> None:
    """Кандидаты в псевдоглифы с признаками и меры сторон (с правилом 0.3 мм и без) по всем вырезкам."""
    from research.pseudo_glyphs.crops import page_key
    from research.pseudo_glyphs.features import collect, page_paths, process_page

    paths = list_crops(canvas_root)
    if not force:
        paths = [p for p in paths if not page_paths(out_dir, page_key(p))[1].exists()]
    if limit:
        paths = paths[:limit]
    click.echo(f"страниц к разбору: {len(paths)}, воркеров {jobs}")
    started = time.monotonic()
    with make_pool(jobs) as pool:
        futures = {pool.submit(process_page, p, out_dir): p for p in paths}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                key, count, seconds = future.result()
            except Exception as error:  # noqa: BLE001 — одна битая страница не роняет прогон
                click.echo(f"ОШИБКА {futures[future]}: {error!r}")
                continue
            if done % 50 == 0 or done == len(paths):
                click.echo(f"{done}/{len(paths)} {key}: кандидатов {count}, {seconds:.1f} с; "
                           f"всего {time.monotonic() - started:.0f} с")  # fmt: skip
    frame = collect(out_dir)
    click.echo(f"кандидатов всего: {len(frame)} → {out_dir / 'candidates.csv'}")


@main.command("collect")
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_ROOT / "hp1991", show_default=True,
              help="Корень выхода стенда.")  # fmt: skip
def collect_cmd(out_dir: Path) -> None:
    """Собрать CSV кандидатов страниц в один."""
    from research.pseudo_glyphs.features import collect

    frame = collect(out_dir)
    click.echo(f"кандидатов всего: {len(frame)}")


def map_key(key: str) -> str:
    """Ключ страницы для кэша карт (без подпапок): ``07/IMG_0099_R`` → ``07__IMG_0099_R``."""
    return key.replace("/", "__")


@main.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_ROOT / "hp1991", show_default=True,
              help="Корень выхода стенда (серый 300 dpi — gray300/, карты — maps/<детектор>/).")  # fmt: skip
@click.option("--engine", "engines", multiple=True, default=("craft", "pero"), show_default=True,
              help="Детекторы символов (по одному процессу на GPU, строго по очереди).")  # fmt: skip
def maps(out_dir: Path, engines: tuple[str, ...]) -> None:
    """Карты CRAFT и pero ParseNet по серому 300 dpi всех страниц (кэш ``glyph_maps``)."""
    from ocr_utils.page_layout.text_blocks.edge_guard import GlyphEngine
    from ocr_utils.page_layout.text_blocks.glyph_maps import MapJob, compute_maps

    pngs = sorted((out_dir / "gray300").glob("*/*.png"))
    jobs = [MapJob(map_key(f"{p.parent.name}/{p.stem}"), png=p) for p in pngs]
    started = time.monotonic()
    for engine in engines:
        compute_maps(jobs, out_dir / "maps", (GlyphEngine(engine),))
        click.echo(f"{engine}: {len(jobs)} страниц, {time.monotonic() - started:.0f} с")


def parse_raw_labels(text: str) -> dict[int, tuple[str, str]]:
    """Разобрать разметку раунда: строки ``<лист> default <класс>`` и ``<номер> <класс> [заметка]``.

    ``default`` задаёт класс всем плиткам листа, не названным явно (лист — номер листа в описи ``index.csv``).
    Пустые строки и строки с ``#`` в начале пропускаются.

    Args:
        text: Содержимое файла разметки.

    Returns:
        ``номер плитки → (класс, заметка)``; ключ ``-лист`` хранит класс по умолчанию листа.
    """
    out: dict[int, tuple[str, str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=2)
        if parts[1] == "default":
            out[-int(parts[0])] = (parts[2].strip(), "")
        else:
            out[int(parts[0])] = (parts[1], parts[2] if len(parts) > 2 else "")
    return out


@main.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_ROOT / "hp1991", show_default=True,
              help="Корень выхода стенда (листы раундов — sheets/<раунд>/).")  # fmt: skip
@click.option("--round-name", required=True, help="Раунд разметки.")
def labels(out_dir: Path, round_name: str) -> None:
    """Перенести разметку раунда (sets/raw/<раунд>.txt) в sets/labels.csv по описи листов."""
    import pandas as pd

    from research.pseudo_glyphs.select import Label

    folder = out_dir / "sheets" / round_name
    index = pd.read_csv(folder / "index.csv")
    picked = pd.read_csv(folder / "picked.csv").set_index("cid")
    raw = parse_raw_labels((SETS_DIR / "raw" / f"{round_name}.txt").read_text())
    rows = []
    for item in index.itertuples():
        label, note = raw.get(item.number, raw.get(-item.sheet, ("", "")))
        if not label:
            continue
        Label(label)  # проверка: класс из перечня
        cand = picked.loc[item.cid]
        rows.append({"key": cand.key, "comp": int(cand.comp), "label": label, "band": cand.band, "note": note,
                     "round": round_name, "cid": item.cid})  # fmt: skip
    new = pd.DataFrame(rows)
    if LABELS_CSV.exists():
        old = pd.read_csv(LABELS_CSV)
        old = old[old["round"] != round_name]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(LABELS_CSV, index=False)
    click.echo(f"меток раунда {len(rows)}, всего {len(new)}: " + str(new.label.value_counts().to_dict()))


@main.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_ROOT / "hp1991", show_default=True,
              help="Корень выхода стенда (candidates.csv, scores.csv; сводки — в eval/).")  # fmt: skip
def evaluate(out_dir: Path) -> None:
    """Полнота и вред правил по разметке, с CRAFT и без, и точность сработавших по паку (пояс fired)."""
    from dataclasses import replace

    import numpy as np
    import pandas as pd

    from research.pseudo_glyphs.evaluate import by_class, craft_sweep, load, summary
    from research.pseudo_glyphs.rules import RuleParams, reasons

    frame = load(out_dir, LABELS_CSV)
    target = out_dir / "eval"
    target.mkdir(parents=True, exist_ok=True)
    params = RuleParams()
    rows = []
    for name, p in (("правила", params), ("правила + CRAFT 0.3", replace(params, craft_doubt=0.3)),
                    ("правила + CRAFT 0.4", replace(params, craft_doubt=0.4))):  # fmt: skip
        reason = reasons(frame, p)
        fired = reason != ""
        rows.append({"вариант": name, **summary(frame, fired)})
        table = by_class(frame, fired, reason)
        table.to_csv(target / f"by_class_{name.replace(' ', '_').replace('+', '')}.csv", index=False)
        click.echo(f"\n{name}\n{table.to_string(index=False)}")
    # Один CRAFT без правил: всё спорное (у края или далеко) с CRAFT ниже порога.
    for t in (0.2, 0.3, 0.4):
        doubt = (frame.edge_mm <= params.near_mm) | ((frame.gap_rel.fillna(0) > 1) & (frame.gap_mm.fillna(0) >= 2))
        rows.append({"вариант": f"только CRAFT {t} (у края или далеко)", **summary(frame, doubt & (frame.craft < t))})
    result = pd.DataFrame(rows)
    result.to_csv(target / "summary.csv", index=False)
    click.echo(
        "\n"
        + result[
            ["вариант", "pseudo", "recall", "true", "harm", "harm_share", "risk_true", "risk_harm_share", "random_harm"]
        ].to_string()
    )
    sweep = craft_sweep(frame, params, np.arange(0.1, 0.65, 0.05))
    sweep.to_csv(target / "craft_sweep.csv", index=False)
    click.echo("\n" + sweep[["craft", "recall", "harm", "harm_share"]].round(3).to_string())
    # Точность по паку: размеченные из пояса «сработало» (раунд r3, выборка поровну по причинам).
    fired_band = frame[frame.band == "fired"].assign(reason=lambda d: reasons(d, params))
    precision = fired_band.groupby("reason").group.apply(lambda g: f"{(g == 'pseudo').sum()}/{len(g)}")
    click.echo("\nточность сработавших по причинам (пояс fired):\n" + precision.to_string())


def _scores_page(args: tuple):
    """Воркер: оценки CRAFT и pero кандидатов одной страницы."""
    from research.pseudo_glyphs.scores import page_scores

    return page_scores(*args)


@main.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_ROOT / "hp1991", show_default=True,
              help="Корень выхода стенда (candidates.csv, gray300/, maps/).")  # fmt: skip
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def scores(out_dir: Path, jobs: int) -> None:
    """Оценки CRAFT и pero у всех кандидатов (по готовым картам) → scores.csv."""
    import pandas as pd

    frame = pd.read_csv(out_dir / "candidates.csv")
    tasks = [(out_dir / "gray300" / f"{key}.png", out_dir / "maps", map_key(key), part)
             for key, part in frame.groupby("key")]  # fmt: skip
    with make_pool(jobs) as pool:
        parts = list(pool.map(_scores_page, tasks, chunksize=8))
    result = pd.concat(parts, ignore_index=True)
    result.to_csv(out_dir / "scores.csv", index=False)
    click.echo(f"оценок {len(result)} → {out_dir / 'scores.csv'}")


def _compare_page(args: tuple):
    """Воркер: «было | стало» одной вырезки."""
    from research.pseudo_glyphs.compare import compare_page

    return compare_page(*args)


@main.command()
@click.option("--canvas-root", type=click.Path(path_type=Path, exists=True), default=HP1991_CANVAS, show_default=True,
              help="Корень вырезок холста.")  # fmt: skip
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_ROOT / "hp1991" / "compare", show_default=True,
              help="Папка прогона сравнения.")  # fmt: skip
@click.option("--craft-dir", type=click.Path(path_type=Path), default=OUT_ROOT / "hp1991" / "maps" / "craft",
              show_default=True, help="Карты CRAFT (для режима правила + CRAFT).")  # fmt: skip
@click.option("--craft-doubt", type=float, default=None, help="Порог CRAFT для спорных; не задан — без режима с CRAFT.")
@click.option("--keys", default="", help="Только эти страницы (через запятую, <выпуск>/<кадр>_<часть>).")
@click.option("--overlay/--no-overlay", default=True, help="Оверлеи «было | стало» страниц, где что-то закрашено.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def compare(canvas_root: Path, out_dir: Path, craft_dir: Path, craft_doubt: float | None, keys: str, overlay: bool,
            jobs: int) -> None:  # fmt: skip
    """Меры сторон блоков «было | стало» с фильтром псевдоглифов по всем вырезкам → summary.csv."""
    from research.pseudo_glyphs.compare_report import summarize
    from research.pseudo_glyphs.crops import page_key
    from research.pseudo_glyphs.rules import RuleParams

    paths = list_crops(canvas_root)
    if keys:
        wanted = set(keys.split(","))
        paths = [p for p in paths if page_key(p) in wanted]
    params = RuleParams(craft_doubt=craft_doubt)
    click.echo(f"страниц {len(paths)}, воркеров {jobs}")
    started = time.monotonic()
    tasks = [(p, out_dir, params, craft_dir if craft_doubt is not None else None, overlay) for p in paths]
    with make_pool(jobs) as pool:
        futures = {pool.submit(_compare_page, t): t[0] for t in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001
                click.echo(f"ОШИБКА {futures[future]}: {error!r}")
            if done % 100 == 0:
                click.echo(f"{done}/{len(paths)}, {time.monotonic() - started:.0f} с")
    table = summarize(out_dir)
    click.echo(table.to_string())


def _pack1_page(args: tuple[str, int]):
    """Воркер: кандидаты одной полосы пака-1."""
    from research.pseudo_glyphs.packcheck import page_frame

    return page_frame(*args)


@main.command("pack1-features")
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_ROOT / "pack1_sets", show_default=True,
              help="Куда положить кандидаты размеченных полос пака-1 (candidates.csv, items.csv).")  # fmt: skip
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def pack1_features(out_dir: Path, jobs: int) -> None:
    """Кандидаты и признаки на полосах размеченных наборов edge_marks и text_block_specks (регрессия «знаки не теряются»)."""
    import pandas as pd

    from research.pseudo_glyphs.packcheck import labelled_items

    items = labelled_items()
    pages = sorted({(r.pdf, int(r.page)) for r in items.itertuples()})
    click.echo(f"элементов {len(items)}, полос {len(pages)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    with make_pool(jobs) as pool:
        for frame in pool.map(_pack1_page, pages):
            frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(out_dir / "candidates.csv", index=False)
    items.to_csv(out_dir / "items.csv", index=False)
    click.echo(f"→ {out_dir}")


@main.command("pack1-check")
@click.option("--out-dir", type=click.Path(path_type=Path, exists=True), default=OUT_ROOT / "pack1_sets",
              show_default=True, help="Папка pack1-features (candidates.csv, items.csv).")  # fmt: skip
def pack1_check(out_dir: Path) -> None:
    """Какие размеченные элементы наборов пака-1 задевают правила: вред на знаках (L, P), полнота на соре (S, M)."""
    import pandas as pd

    from research.pseudo_glyphs.packcheck import hits
    from research.pseudo_glyphs.rules import RuleParams, relative_height

    frame = pd.read_csv(out_dir / "candidates.csv")
    frame["key"] = frame.pdf + "#" + frame.page.astype(str)
    frame["h_rel"] = relative_height(frame)
    items = pd.read_csv(out_dir / "items.csv")
    result = hits(items, frame, RuleParams())
    result.to_csv(out_dir / "hits.csv", index=False)
    table = result.groupby(["set", "label"]).agg(n=("id", "size"), candidate=("candidate", "sum"),
                                                 fired=("reason", lambda r: int((r != "").sum())))  # fmt: skip
    click.echo(table.to_string())
    signs = result[result.truth == "sign"]
    click.echo(f"знаков {len(signs)}, задето {int((signs.reason != '').sum())}; "
               f"сора {int((result.truth == 'junk').sum())}, снято {int((result[result.truth == 'junk'].reason != '').sum())}")  # fmt: skip
    click.echo(result[(result.truth == "sign") & (result.reason != "")].to_string())


# Разметка стенда в git.
SETS_DIR = Path(__file__).resolve().parent / "sets"
LABELS_CSV = SETS_DIR / "labels.csv"


def _parse_bands(spec: str) -> dict:
    """``row_edge=60,far=40`` → ``{Band: число}``."""
    from research.pseudo_glyphs.select import Band

    out = {}
    for part in spec.split(","):
        name, count = part.split("=")
        out[Band(name.strip())] = int(count)
    return out


@main.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_ROOT / "hp1991", show_default=True,
              help="Корень выхода стенда (candidates.csv; листы — в sheets/<раунд>).")  # fmt: skip
@click.option("--round-name", required=True, help="Имя раунда разметки (папка листов).")
@click.option("--bands", default="row_edge=60,far=50,tall=50,grey=50,thin=40,edge_only=30,random=120",
              show_default=True, help="Сколько кандидатов из каждого пояса.")  # fmt: skip
@click.option("--per-sheet", type=int, default=40, show_default=True, help="Плиток на лист.")
def sheets(out_dir: Path, round_name: str, bands: str, per_sheet: int) -> None:
    """Листы на разметку по поясам признаков; уже размеченные компоненты (sets/labels.csv) пропускаются."""
    import pandas as pd

    from research.pseudo_glyphs.select import sample_bands
    from research.pseudo_glyphs.sheets import write_sheets

    from research.pseudo_glyphs.rules import reasons

    frame = pd.read_csv(out_dir / "candidates.csv")
    scores_path = out_dir / "scores.csv"
    if scores_path.exists():
        frame = frame.merge(pd.read_csv(scores_path), on="cid", how="left")
    # Причины правил по умолчанию — для поясов «сработало» и «спорно».
    frame["reason"] = reasons(frame)
    done = set()
    if LABELS_CSV.exists():
        labels = pd.read_csv(LABELS_CSV)
        done = set(labels.key + "#" + labels.comp.astype(str))
    frame["uid"] = frame.key + "#" + frame.comp.astype(str)
    picked = sample_bands(frame[~frame.uid.isin(done)], _parse_bands(bands))
    picked = picked.set_index("cid", drop=False)
    target = out_dir / "sheets" / round_name
    paths = write_sheets(picked, target, f"Псевдоглифы ХиП 1991, раунд {round_name}", per_sheet=per_sheet)
    picked.to_csv(target / "picked.csv", index=False)
    click.echo(f"листов {len(paths)}, кандидатов {len(picked)} → {target}")
