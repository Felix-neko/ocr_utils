"""Подкладывает пересъёмку отдельных страниц в копию подшивки вместо исходных снимков (сопоставление по AKAZE)."""

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np

from merge_best_shoots import Frame, load_features, similarity_pairs

# Подпапка копии, куда уходят заменённые оригиналы (скрытая: обходы папок её пропускают)
DEBUG_SUBDIR = ".gutter_loss_for_debugging"


@dataclass
class Match:
    """Сопоставление снимка пересъёмки с оригиналом подшивки."""

    # Снимок пересъёмки
    reshoot: Path
    # Лучший оригинал (None — сходство ниже порога, пары нет)
    original: Path | None
    # Сходство с лучшим и вторым по сходству оригиналом (доля RANSAC-инлайеров)
    best: float
    second: float
    # Второй по сходству оригинал — для разбора спорных случаев
    runner_up: Path | None


def raf_files(folder: Path, recursive: bool) -> list[Path]:
    """RAF-файлы папки по порядку имён; скрытые подпапки (имя с точки) не обходятся.

    Args:
        folder: Папка.
        recursive: Брать ли RAF из вложенных (не скрытых) подпапок.

    Returns:
        Отсортированный по пути список RAF.
    """
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in folder.glob(pattern)
        if p.is_file()
        and p.suffix.upper() == ".RAF"
        and not any(part.startswith(".") for part in p.relative_to(folder).parts[:-1])
    )


def best_matches(reshoots: list[Path], originals: list[Path], similarity: np.ndarray, min_sim: float) -> list[Match]:
    """Для каждого снимка пересъёмки — самый похожий оригинал и отрыв от второго.

    Args:
        reshoots: Снимки пересъёмки (строки матрицы).
        originals: Оригиналы подшивки (столбцы матрицы).
        similarity: Матрица сходства (len(reshoots), len(originals)).
        min_sim: Порог сходства, ниже которого пара не засчитывается.

    Returns:
        Список Match в порядке reshoots.
    """
    matches = []
    for i, reshoot in enumerate(reshoots):
        # Два самых похожих оригинала по убыванию сходства
        order = np.argsort(-similarity[i])
        best, second = similarity[i, order[0]], similarity[i, order[1]] if len(order) > 1 else 0.0
        matches.append(
            Match(
                reshoot=reshoot,
                original=originals[order[0]] if best >= min_sim else None,
                best=float(best),
                second=float(second),
                runner_up=originals[order[1]] if len(order) > 1 else None,
            )
        )
    return matches


def replacement_name(original: Path, reshoot: Path, gutter_light: bool) -> str:
    """Имя подложенного кадра: имя оригинала (с его пометками), номер пересъёмки и пометки замены.

    Args:
        original: Заменяемый оригинал, например DSCF1023.RAF или DSCF1257_defocus.RAF.
        reshoot: Снимок пересъёмки, например DSCF0015.RAF.
        gutter_light: Пометить ли дополнительно ``.gutter_loss.light`` (текст у корешка всё ещё теряется).

    Returns:
        Например «DSCF1023.DSCF0015.replacement.RAF» или «DSCF1023.DSCF0015.replacement.gutter_loss.light.RAF».
    """
    tail = ".replacement.gutter_loss.light" if gutter_light else ".replacement"
    return f"{original.stem}.{reshoot.stem}{tail}{reshoot.suffix}"


@click.command()
@click.argument("orig_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reshoot_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Кэш признаков")
@click.option(
    "--gutter-light",
    multiple=True,
    help="Путь снимка пересъёмки относительно reshoot_dir, который пометить .gutter_loss.light. Можно повторять",
)
@click.option("--min-sim", default=0.05, show_default=True, help="Порог сходства «та же страница»")
@click.option("--max-scale", default=2.0, show_default=True, help="Допуск масштаба гомографии (другое расстояние)")
@click.option("--width", default=1536, show_default=True, help="Ширина кадра для AKAZE")
@click.option("--max-points", default=4000, show_default=True, help="Предел AKAZE-точек на кадр")
@click.option("--jobs", default=8, show_default=True, help="Процессов; чтение RAF упирается в NTFS, 8 хватает")
@click.option(
    "--apply", "do_apply", is_flag=True, help="Скопировать подшивку и подложить кадры (иначе — только mapping.csv)"
)
@click.option("--overwrite", is_flag=True, help="Разрешить запись в непустую выходную папку")
def main(
    orig_dir: Path,
    reshoot_dir: Path,
    out_dir: Path,
    cache_dir: Path,
    gutter_light: tuple[str, ...],
    min_sim: float,
    max_scale: float,
    width: int,
    max_points: int,
    jobs: int,
    do_apply: bool,
    overwrite: bool,
) -> None:
    """Сопоставляет пересъёмку с оригиналами подшивки и собирает копию подшивки с подменой.

    Оригинальная папка только читается. В копии каждый заменяемый оригинал уходит в
    ``.gutter_loss_for_debugging/``, а на его место ложится снимок пересъёмки с именем
    ``<оригинал>.<пересъёмка>.replacement[.gutter_loss.light].RAF``.

    Args:
        orig_dir: Папка подшивки (оригиналы; скрытые подпапки не обходятся).
        reshoot_dir: Папка пересъёмки (RAF во вложенных подпапках тоже берутся).
        out_dir: Куда собрать копию подшивки; ``mapping.csv`` пишется сюда всегда.
        cache_dir: Кэш AKAZE-признаков.
        gutter_light: Снимки пересъёмки (пути относительно reshoot_dir) для пометки ``.gutter_loss.light``.
        min_sim: Порог сходства пары.
        max_scale: Допуск масштаба гомографии.
        width: Ширина кадра для AKAZE.
        max_points: Предел точек на кадр.
        jobs: Число процессов пула.
        do_apply: Собрать копию и подложить кадры; без флага — только сопоставление и mapping.csv.
        overwrite: Разрешить запись в непустую папку.

    Returns:
        None.
    """
    originals = raf_files(orig_dir, recursive=False)
    reshoots = raf_files(reshoot_dir, recursive=True)
    light = {reshoot_dir / rel for rel in gutter_light}
    unknown = light - set(reshoots)
    if unknown:
        raise click.ClickException(f"--gutter-light: нет таких снимков пересъёмки: {sorted(map(str, unknown))}")
    click.echo(f"Оригиналов: {len(originals)}, снимков пересъёмки: {len(reshoots)}")

    # Шаг 1: признаки (кэш на SSD) и сходство всех пар «пересъёмка × оригинал»
    orig_frames = [
        Frame(p, pts, d, None)
        for p, (pts, d) in zip(originals, load_features(originals, cache_dir, width, max_points, jobs))
    ]
    new_frames = [
        Frame(p, pts, d, None)
        for p, (pts, d) in zip(reshoots, load_features(reshoots, cache_dir, width, max_points, jobs))
    ]
    pairs = [(i, j) for i in range(len(new_frames)) for j in range(len(orig_frames))]
    sims = similarity_pairs(pairs, new_frames, orig_frames, max_scale, jobs, "Пересъёмка × оригиналы")
    similarity = np.zeros((len(new_frames), len(orig_frames)))
    for (i, j), s in sims.items():
        similarity[i, j] = s

    # Шаг 2: лучший оригинал для каждой пересъёмки; порядок съёмки как проверка
    matches = best_matches(reshoots, originals, similarity, min_sim)
    position = {p: k for k, p in enumerate(originals)}
    # Пересъёмка снята «в порядке номеров»: по имени снимка (без подпапки) оригиналы должны идти по возрастанию
    by_shot = sorted((m for m in matches if m.original is not None), key=lambda m: m.reshoot.name)
    order_breaks = [
        (a.reshoot.name, a.original.name, b.reshoot.name, b.original.name)
        for a, b in zip(by_shot, by_shot[1:])
        if position[b.original] < position[a.original]
    ]
    targets: dict[Path, list[Match]] = {}
    for m in matches:
        if m.original is not None:
            targets.setdefault(m.original, []).append(m)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "reshoot": str(m.reshoot.relative_to(reshoot_dir)),
            "original": m.original.name if m.original else "",
            "similarity": f"{m.best:.3f}",
            "second": f"{m.second:.3f}",
            "runner_up": m.runner_up.name if m.runner_up else "",
            "gutter_light": m.reshoot in light,
            "new_name": replacement_name(m.original, m.reshoot, m.reshoot in light) if m.original else "",
        }
        for m in sorted(matches, key=lambda m: position.get(m.original, len(originals)))
    ]
    with (out_dir / "mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Сводка для решения глазами
    matched = [m for m in matches if m.original is not None]
    best = np.array([m.best for m in matched]) if matched else np.zeros(1)
    click.echo(
        f"Сопоставлено {len(matched)} из {len(matches)}; заменяемых оригиналов {len(targets)}; "
        f"сходство min {best.min():.3f}, p10 {np.quantile(best, 0.1):.3f}, медиана {np.median(best):.3f}"
    )
    for m in matches:
        if m.original is None:
            click.echo(f"  без пары: {m.reshoot.relative_to(reshoot_dir)} (лучшее сходство {m.best:.3f})")
    for original, ms in targets.items():
        if len(ms) > 1:
            click.echo(f"  на {original.name} несколько снимков: {[m.reshoot.name for m in ms]}")
    for a_new, a_old, b_new, b_old in order_breaks:
        click.echo(f"  порядок нарушен: {a_new}→{a_old}, затем {b_new}→{b_old}")
    click.echo(f"Таблица сопоставления: {out_dir / 'mapping.csv'}")

    if not do_apply:
        click.echo("Без --apply: подшивка не копировалась, кадры не подкладывались")
        return

    # Шаг 3: копия подшивки (с её скрытыми подпапками) и подмена
    if any(p.suffix.upper() == ".RAF" for p in out_dir.iterdir()) and not overwrite:
        raise click.ClickException(f"В {out_dir} уже есть RAF; повторно — с --overwrite")
    click.echo(f"Копирую {orig_dir} → {out_dir}")
    subprocess.run(["rsync", "-a", f"{orig_dir}/", f"{out_dir}/"], check=True)
    debug_dir = out_dir / DEBUG_SUBDIR
    debug_dir.mkdir(exist_ok=True)
    for original, ms in targets.items():
        # Оригинал в копии — в отладочную подпапку (rsync мог положить его заново при повторном прогоне)
        copy = out_dir / original.name
        if copy.exists():
            copy.replace(debug_dir / original.name)
        for m in ms:
            shutil.copy2(m.reshoot, out_dir / replacement_name(original, m.reshoot, m.reshoot in light))
    n_raf = sum(1 for p in out_dir.iterdir() if p.suffix.upper() == ".RAF")
    click.echo(f"Готово: RAF в копии {n_raf}, заменено оригиналов {len(targets)}, подложено снимков {len(matched)}")


if __name__ == "__main__":
    main()
