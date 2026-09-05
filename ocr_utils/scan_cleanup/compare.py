"""Инструменты сравнения: выбрать параметры глазами, а не на веру.

Две команды, обе кладут результат в подпапки по относительному пути полосы:

* :func:`run_compare_masks` — защитная маска и результат размытия всеми
  сочетаниями метод × k × припуск × размытие;
* :func:`run_compare_inpaint` — закрас всеми бэкендами и вариантами группировки.

ГЛАВНОЕ В ОБОИХ — РЯД ОДИНАКОВЫХ ВРЕЗОК 1:1. Судить надо о бледных перемычках
букв и о фактуре заливки, а на уменьшенной странице в 6000 px не видно ни того,
ни другого. Поэтому по каждой полосе выбираются одни и те же окна, и все варианты
выкладываются в ряд именно по ним; уменьшенные страницы остаются лишь для общего
впечатления.

Выборка полос детерминированная: перезапуск обязан сравнивать тот же материал.
"""

import itertools
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.background_smoothing.processing import METHOD_SAUVOLA
from ocr_utils.paper import AUTO_INK_LEVEL, INK_LEVEL
from ocr_utils.inpainting.backends import BACKEND_SD, SdParams, make_filler
from ocr_utils.inpainting.grouping import group_masks
from ocr_utils.scan_cleanup.inpaint import (
    DEFAULT_LAMA_HOLE_MAX_PX,
    FEATHER_PX,
    GROUP_ROI_SCALE,
    GROUP_MIN_DILATE_PX,
    ROI_PADDING,
    page_masks,
    zone_kinds,
)
from ocr_utils.scan_cleanup.overlay import downscale, draw_page_overlay, label
from ocr_utils.scan_cleanup.prompts import PromptSet, prompt_chooser
from ocr_utils.scan_cleanup.protect import analysis_roi, build_protect
from ocr_utils.scan_cleanup.smoothing import SmoothOptions, smooth_page
from ocr_utils.scan_cleanup.source import PageMarkup, load_markup
from ocr_utils.scan_markup.db.models import MASK_KINDS

logger = logging.getLogger(__name__)

PREVIEW_SIDE = 2000
JPEG_QUALITY = 90
SAMPLE_SEED = 17


@dataclass
class CompareMasksParams:
    db_path: Path
    pack_name: str
    pack_dir: Path
    out_dir: Path
    only_year: "str | None" = None
    only_issue: "str | None" = None
    pages_file: "Path | None" = None
    explicit_pages: "tuple[str, ...]" = ()
    limit: "int | None" = None
    sample: int = 0
    methods: "tuple[str, ...]" = ("otsu", "sauvola")
    sauvola_ks: "tuple[float, ...]" = (0.10,)
    dilate_pxs: "tuple[float, ...]" = (15.0, 25.0, 35.0)
    blur_pxs: "tuple[float, ...]" = (60.0, 120.0, 240.0)
    ink_levels: "tuple[float | None, ...]" = (INK_LEVEL,)
    show_removed: bool = False
    crop_side: int = 1200
    crops: int = 3
    montage_cols: int = 4
    jobs: int = 8


@dataclass
class CompareInpaintParams:
    db_path: Path
    pack_name: str
    pack_dir: Path
    out_dir: Path
    only_year: "str | None" = None
    only_issue: "str | None" = None
    pages_file: "Path | None" = None
    explicit_pages: "tuple[str, ...]" = ()
    limit: "int | None" = None
    sample: int = 30
    backends: "tuple[str, ...]" = ("lama", "sd")
    kinds: "tuple[str, ...]" = ()
    group_dilate_fracs: "tuple[float, ...]" = (1.0 / 3.0,)
    roi_scales: "tuple[float, ...]" = (GROUP_ROI_SCALE,)
    lama_holes: "tuple[int, ...]" = (DEFAULT_LAMA_HOLE_MAX_PX,)
    group_min_dilate_px: int = GROUP_MIN_DILATE_PX
    montage_cols: int = 4
    sd: SdParams = field(default_factory=SdParams)
    prompts: PromptSet = field(default_factory=PromptSet)


# ----------------------------------------------------------------------
# Общее
# ----------------------------------------------------------------------


def _write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def _row(images: "list[np.ndarray]", gap: int = 10) -> np.ndarray:
    """Картинки в ряд, выровненные по высоте белым полем."""
    height = max(im.shape[0] for im in images)
    padded = [np.pad(im, ((0, height - im.shape[0]), (0, 0), (0, 0)), constant_values=255) for im in images]
    spacer = np.full((height, gap, 3), 255, np.uint8)
    row: "list[np.ndarray]" = []
    for im in padded:
        row.extend([im, spacer])
    return np.hstack(row[:-1])


def _grid(images: "list[np.ndarray]", cols: int, gap: int = 10) -> np.ndarray:
    """Картинки сеткой в ``cols`` столбцов.

    Сеткой, а не одним рядом: вариантов бывает под два десятка, и ряд врезок по
    1200 px вырастает в двадцать три тысячи пикселей — такую полосу нельзя ни
    открыть целиком, ни сравнить глазами. При ``cols <= 0`` возвращается ряд.
    """
    if cols <= 0 or len(images) <= cols:
        return _row(images, gap)
    rows = [_row(images[i : i + cols], gap) for i in range(0, len(images), cols)]
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    spacer = np.full((gap, width, 3), 255, np.uint8)
    stacked: "list[np.ndarray]" = []
    for r in rows:
        stacked.extend([r, spacer])
    return np.vstack(stacked[:-1])


def pick_crops(mask: np.ndarray, side: int, count: int) -> "list[tuple[int, int]]":
    """Окна с наибольшей плотностью краски, разнесённые друг от друга.

    Детерминированно и без случайности: одна и та же полоса даёт одни и те же
    окна при каждом прогоне, иначе ряд вариантов было бы не с чем сравнивать.
    Плотность считается по сетке из окон в полстороны шагом — этого хватает,
    чтобы поймать текстовый блок, и стоит копейки.
    """
    h, w = mask.shape
    side = min(side, h, w)
    step = max(1, side // 2)
    scores = []
    for y in range(0, max(1, h - side + 1), step):
        for x in range(0, max(1, w - side + 1), step):
            scores.append((float((mask[y : y + side, x : x + side] > 0).mean()), x, y))
    scores.sort(reverse=True)

    chosen: "list[tuple[int, int]]" = []
    for _score, x, y in scores:
        if all(abs(x - cx) >= side or abs(y - cy) >= side for cx, cy in chosen):
            chosen.append((x, y))
        if len(chosen) >= count:
            break
    return chosen


def select_pages(
    db_path: Path,
    pack_name: str,
    *,
    only_year=None,
    only_issue=None,
    pages_file=None,
    explicit_pages=(),
    limit=None,
    sample=0,
    with_masks=False,
) -> "list[PageMarkup]":
    """Полосы для сравнения. ``sample`` берёт случайные, но с фиксированным зерном."""
    from ocr_utils.scan_cleanup.runner import wanted_rel

    pages = load_markup(
        db_path, pack_name, only_year=only_year, only_issue=only_issue, only_rel=wanted_rel(pages_file, explicit_pages)
    )
    if with_masks:
        pages = [p for p in pages if p.needs_inpaint]
    if sample and len(pages) > sample:
        pages = sorted(random.Random(SAMPLE_SEED).sample(pages, sample), key=lambda p: p.rel_path)
    if limit is not None:
        pages = pages[:limit]
    return pages


# ----------------------------------------------------------------------
# Сравнение защитных масок
# ----------------------------------------------------------------------


def mask_variants(params: CompareMasksParams) -> "list[tuple[str, SmoothOptions]]":
    """Декартово произведение повторяемых опций: (имя варианта, настройки)."""
    out = []
    for method, k, dilate, blur, ink in itertools.product(
        params.methods, params.sauvola_ks, params.dilate_pxs, params.blur_pxs, params.ink_levels
    ):
        # Для Оцу параметр k не значит ничего — не плодим одинаковые варианты.
        if method != METHOD_SAUVOLA and k != params.sauvola_ks[0]:
            continue
        ink_tag = "off" if ink is None else ("auto" if ink == AUTO_INK_LEVEL else f"{ink:g}")
        name = (
            f"{method}" + (f"_k{k:g}" if method == METHOD_SAUVOLA else "") + f"_dil{dilate:g}_blur{blur:g}_ink{ink_tag}"
        )
        out.append((name, SmoothOptions(method=method, sauvola_k=k, dilate_px=dilate, blur_px=blur, ink_level=ink)))
    return out


def diff_tiles(gray: np.ndarray, base: np.ndarray, new: np.ndarray, side: int, count: int) -> "list[np.ndarray]":
    """Врезки 1:1 вокруг того, что новое правило УБРАЛО из маски.

    Красным обведено убранное поверх исходных пикселей — по такой врезке сразу видно,
    складки это и пыль или бледные буквы. Тем же приёмом проверялось 1966/01, где
    убранным оказались волокна бумаги.

    Окна выбираются по маске разницы, а не по маске краски: показывать надо самые
    плотные места потерь.
    """
    removed = ((base > 0) & ~(new > 0)).astype(np.uint8)
    if not removed.any():
        return []
    tiles = []
    for x, y in pick_crops(removed, side, count):
        crop = cv2.cvtColor(gray[y : y + side, x : x + side], cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(removed[y : y + side, x : x + side], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, (0, 0, 255), 1)
        tiles.append(crop)
    return tiles


def compare_masks_page(markup: PageMarkup, params: CompareMasksParams) -> str:
    """Считает все варианты по одной полосе и раскладывает их файлами."""
    src = cv2.imread(str(markup.source_path(params.pack_dir)), cv2.IMREAD_COLOR)
    if src is None:
        return f"{markup.rel_path}: не читается"

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    protect, rects = build_protect(gray.shape, markup, None)
    roi = analysis_roi(gray.shape, markup)
    page_dir = params.out_dir / Path(markup.rel_path).with_suffix("")
    _write(page_dir / "source.jpg", downscale(src, PREVIEW_SIDE))

    variants = mask_variants(params)
    results = []
    for name, opts in variants:
        res = smooth_page(src, gray, protect if rects else None, roi, opts)
        results.append((name, res))
        _write(page_dir / name / "result.jpg", downscale(res.image, PREVIEW_SIDE))
        _write(
            page_dir / name / "overlay.jpg",
            downscale(draw_page_overlay(src, res.m_primary, res.m_dilated, rects), PREVIEW_SIDE),
        )

    # Врезки выбираются по ПЕРВОМУ варианту и одинаковы для всех: сравнивать надо
    # одно и то же место, иначе разница вариантов путается с разницей мест.
    reference = results[0][1].m_primary
    windows = pick_crops(reference, params.crop_side, params.crops)
    side = min(params.crop_side, *gray.shape)
    for i, (x, y) in enumerate(windows):
        # Врезки идут 1:1: ради них всё и затевалось, уменьшать их нельзя.
        tiles = [label(src[y : y + side, x : x + side], "source")]
        tiles += [label(res.image[y : y + side, x : x + side], name) for name, res in results]
        _write(page_dir / "crops" / f"crop_{i}.jpg", _grid(tiles, params.montage_cols))

    if params.show_removed and len(results) > 1:
        # Разница считается от ПЕРВОГО варианта: он и есть точка отсчёта в сетке.
        base = results[0][1].m_primary
        for name, res in results[1:]:
            for i, tile in enumerate(diff_tiles(gray, base, res.m_primary, params.crop_side, params.crops)):
                _write(page_dir / name / f"removed_{i}.jpg", tile)

    covered = ", ".join(f"{name} {100 * (res.m_dilated > 0).mean():.0f}%" for name, res in results)
    return f"{markup.rel_path}: вариантов {len(results)}, врезок {len(windows)}; под защитой {covered}"


def run_compare_masks(params: CompareMasksParams) -> str:
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    pages = select_pages(
        params.db_path,
        params.pack_name,
        only_year=params.only_year,
        only_issue=params.only_issue,
        pages_file=params.pages_file,
        explicit_pages=params.explicit_pages,
        limit=params.limit,
        sample=params.sample,
    )
    if not pages:
        return "Полос не выбрано: проверьте --page / --pages-file / --only-year"

    lines: "list[str]" = []
    jobs = max(1, params.jobs)
    if jobs == 1:
        for markup in tqdm(pages, desc="compare-masks", unit="полоса"):
            lines.append(compare_masks_page(markup, params))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            tasks = [(markup, params) for markup in pages]
            for line in tqdm(pool.map(_masks_worker, tasks), total=len(tasks), desc="compare-masks", unit="полоса"):
                lines.append(line)
    lines.append(f"Готово: {params.out_dir}")
    return "\n".join(lines)


def _masks_worker(args) -> str:
    markup, params = args
    try:
        return compare_masks_page(markup, params)
    except Exception as e:
        logger.exception("Ошибка на %s", markup.rel_path)
        return f"{markup.rel_path}: ошибка — {e}"


# ----------------------------------------------------------------------
# Сравнение закраса
# ----------------------------------------------------------------------


def inpaint_variants(params: CompareInpaintParams) -> "list[tuple[str, str, float, float, int]]":
    """(имя, бэкенд, доля группировки, масштаб ROI, предел дыры LaMa)."""
    out = []
    for backend, frac, scale, hole in itertools.product(
        params.backends, params.group_dilate_fracs, params.roi_scales, params.lama_holes
    ):
        # Предел дыры — только у LaMa; у SD своя опция размера.
        if backend != "lama" and hole != params.lama_holes[0]:
            continue
        name = f"{backend}_grp{frac:g}_roi{scale:g}" + (f"_hole{hole}" if backend == "lama" else "")
        out.append((name, backend, frac, scale, hole))
    return out


def run_compare_inpaint(params: CompareInpaintParams) -> str:
    """Закрашивает выборку всеми вариантами. Последовательно: работа идёт на GPU.

    Зоны строятся ровно как в боевом прогоне — по всем видам разметки сразу.
    Инструмент сравнения, воспроизводящий не то, что делает прогон, хуже, чем
    никакой: по нему выбирают параметры.
    """
    from functools import partial

    from tqdm import tqdm

    from ocr_utils.inpainting.apply import inpaint_by_groups
    from ocr_utils.inpainting.roi import roi_bounds
    from ocr_utils.scan_cropping.gpu_models import GpuModels

    pages = select_pages(
        params.db_path,
        params.pack_name,
        only_year=params.only_year,
        only_issue=params.only_issue,
        pages_file=params.pages_file,
        explicit_pages=params.explicit_pages,
        limit=params.limit,
        sample=params.sample,
        with_masks=True,
    )
    if not pages:
        return "Полос с масками не выбрано"

    kinds = params.kinds or MASK_KINDS
    variants = inpaint_variants(params)
    need_sd = any(v[1] == BACKEND_SD for v in variants)
    lines = [f"Полос: {len(pages)}, вариантов на зону: {len(variants)}"]

    with GpuModels(with_detection=False, with_lama=True, sd_model=params.sd.model if need_sd else None) as models:
        for markup in tqdm(pages, desc="compare-inpaint", unit="полоса"):
            src = cv2.imread(str(markup.source_path(params.pack_dir)), cv2.IMREAD_COLOR)
            if src is None:
                lines.append(f"{markup.rel_path}: не читается")
                continue
            rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            page_dir = params.out_dir / Path(markup.rel_path).with_suffix("")

            masks = page_masks(markup, kinds)
            if not masks:
                continue
            union = np.zeros((markup.height, markup.width), np.uint8)
            for mask in masks.values():
                union = cv2.bitwise_or(union, mask)

            for i, zone in enumerate(
                group_masks(union, params.group_dilate_fracs[0], min_dilate_px=params.group_min_dilate_px)
            ):
                kinds_here = zone_kinds(zone, masks)
                zone_label = "+".join(kinds_here) or "zone"
                main_kind = kinds_here[0] if kinds_here else ""
                chosen: "list[str]" = []
                per_variant: "list[tuple[str, np.ndarray]]" = []

                for name, backend, frac, scale, hole in variants:
                    filler = make_filler(
                        backend,
                        models,
                        hole_max_px=hole,
                        prompts=prompt_chooser(markup, main_kind, params.prompts),
                        sd=params.sd,
                        on_prompt=lambda _box, prompt: chosen.append(prompt),
                    )
                    # Доля группировки остаётся осью сравнения: при 0 зона распадается
                    # обратно на отдельные связные области, и видно, что даёт объединение.
                    out_rgb, _rois = inpaint_by_groups(
                        rgb,
                        zone,
                        filler,
                        groups=partial(group_masks, dilate_frac=frac, min_dilate_px=params.group_min_dilate_px),
                        padding=ROI_PADDING,
                        feather=FEATHER_PX,
                        roi_scale=scale,
                    )
                    per_variant.append((name, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)))

                # Врезка режется по ОБЩЕМУ ROI зоны, а не по ROI варианта: у вариантов
                # с разной группировкой ROI разные, а сравнивать надо одно и то же место.
                x1, y1, x2, y2 = roi_bounds(zone, ROI_PADDING, params.roi_scales[0], src.shape[:2])
                tiles = [label(src[y1:y2, x1:x2], f"before {zone_label}")]
                tiles += [label(img[y1:y2, x1:x2], name) for name, img in per_variant]
                _write(page_dir / f"zone_{i}_{zone_label}.jpg", _grid(tiles, params.montage_cols))

                if chosen:
                    prompt_file = page_dir / f"zone_{i}_{zone_label}.prompt.txt"
                    prompt_file.parent.mkdir(parents=True, exist_ok=True)
                    # dict.fromkeys — уникальные с сохранением порядка; перевод строки в
                    # конце обязателен, иначе несколько файлов, слитых `cat`, дают одну строку.
                    prompt_file.write_text("\n".join(dict.fromkeys(chosen)) + "\n", encoding="utf-8")
                lines.append(f"{markup.rel_path} [{zone_label}]: вариантов {len(per_variant)}")

    lines.append(f"Готово: {params.out_dir}")
    return "\n".join(lines)
