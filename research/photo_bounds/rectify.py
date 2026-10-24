"""Демонстрация на кадрах стенда page_rectify: поиск кадров с фото среди выпрямленных итогов (``scan``) и прогон второго преобразования тремя вариантами — как есть, область фото четырёхугольником, четырёхугольник плюс стороны фото в опорах поля (подмены в процессе стенда, боевой код не меняется)."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.units import MM_PER_INCH, px_to_mm
from research.photo_bounds.bounds import PhotoBound, Side, find_photos
from research.photo_bounds.supports import PhotoRegion, photo_samples, region_of

log = logging.getLogger(__name__)

# Разрешение копии для поиска кадров с фото (детектор растра page_layout без surya). Не 150: детектор держит свои
# копии 150 dpi и отпускает полный кадр, а у кадра в 150 dpi копия совпадает с ним и пропадает вместе с ним.
SCAN_DPI = 300
# Фото меньше этой доли страницы при поиске кадров не считаются (виньетки, мелкие портреты), доля площади кадра.
SCAN_MIN_FRAC = 0.01


class Variant(StrEnum):
    """Вариант второго преобразования."""

    BASE = "base"  # как в бою: рамка растра — преграда и зона отсева, опор в фото нет
    QUAD = "quad"  # область фото — кривой четырёхугольник (или уточнённая рамка): преграда и отсев по нему
    SIDES = "sides"  # QUAD плюс стороны фото в опорах поля


def scan_frame(args: tuple[Path, float]) -> dict:
    """Растр на выпрямленном итоге (детектор ``page_layout`` без surya, копия 300 dpi): число фото и их доля кадра.

    Args:
        args: ``(итог выпрямления TIFF, dpi бумаги)``.

    Returns:
        Строка: путь, число растровых областей, их рамки (px итога) и суммарная доля площади.
    """
    from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
    from ocr_utils.page_layout.image import PageImage, Variant as ImageVariant

    path, dpi = args
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"path": str(path), "status": "unreadable"}
    f = SCAN_DPI / dpi
    small = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    page = PageImage.from_array(small, dpi=SCAN_DPI, variant=ImageVariant.SCAN, cache_name="photo_bounds_scan")
    layout = PageLayout(page, {Find.RASTER}, LayoutOptions(use_surya=False)).process(None)
    h, w = small.shape[:2]
    boxes = [r.box for r in layout.raster_pics if not r.full_page]
    boxes = [b for b in boxes if (b.x1 - b.x0) * (b.y1 - b.y0) >= SCAN_MIN_FRAC * w * h]
    frac = sum((b.x1 - b.x0) * (b.y1 - b.y0) for b in boxes) / float(w * h)
    return {
        "path": str(path),
        "status": "ok",
        "photos": len(boxes),
        "frac": round(frac, 4),
        "boxes": ";".join(",".join(str(int(round(v / f))) for v in (b.x0, b.y0, b.x1, b.y1)) for b in boxes),
    }


@dataclass
class Patches:
    """Подмены второго преобразования на время одного кадра.

    Поля:
        variant: Вариант.
        dpi: Разрешение холста.
        regions_log: Найденные области фото по вызовам ``page_regions`` (для листов и мер).
        photos_by_page: Фото, найденные на текущем холсте, по ``id`` разбора страницы (для выборок прохода).
        sample_log: Сколько выборок сторон фото добавлено (по проходам).
    """

    variant: Variant
    dpi: float
    regions_log: list = None
    photos_by_page: dict = None
    sample_log: list = None

    def __post_init__(self) -> None:
        self.regions_log, self.photos_by_page, self.sample_log = [], {}, []


def photo_regions_on(gray: np.ndarray, regions: tuple, dpi: float) -> tuple:
    """Растровые области страницы → области фото (многоугольники по найденным сторонам); прочие — как есть.

    Args:
        gray: Серый холст.
        regions: Области ``page_regions``.
        dpi: Разрешение холста.

    Returns:
        Области, где растр заменён на :class:`PhotoRegion`.
    """
    from ocr_utils.scan_cropping.v2.rectify.text_field import RegionKindR

    out = []
    for region in regions:
        if region.kind is not RegionKindR.RASTER or isinstance(region, PhotoRegion):
            out.append(region)
            continue
        seed = tuple(int(round(v)) for v in region.box)
        photos = [p for p in find_photos(gray, [seed], dpi) if p.sides]
        if not photos:
            out.append(region)
            continue
        out.extend(region_of(p) for p in photos)
    return tuple(out)


@contextlib.contextmanager
def patched(patches: Patches):
    """Подменяет в ``text_run`` поиск областей, разбор страницы и выборки линеек на время кадра.

    ``BASE`` — без подмен. ``QUAD`` — растр ``page_regions`` заменяется областями фото (:func:`photo_regions_on`).
    ``SIDES`` — ещё и разбор страницы ищет фото на текущем холсте каждого прохода (поле сдвигает их), а ``rule_samples``
    добавляет выборки сторон фото (:func:`supports.photo_samples`) к выборкам линеек: горизонтальные стороны считаются
    в число опор, как горизонтальные линейки.

    Args:
        patches: Состояние подмен.
    """
    from ocr_utils.scan_cropping.v2.rectify import text_run

    if patches.variant is Variant.BASE:
        yield patches
        return
    original = (text_run.page_regions, text_run.analyze_page, text_run.rule_samples)

    def page_regions(canvas, box, dpi):
        regions = original[0](canvas, box, dpi)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        replaced = photo_regions_on(gray, regions, dpi)
        patches.regions_log.append((box, regions, replaced))
        return replaced

    def analyze_page(canvas, box, dpi, rect, regions=()):
        page = original[1](canvas, box, dpi, rect, regions)
        if patches.variant is Variant.SIDES:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            photos = []
            for region in regions:
                if isinstance(region, PhotoRegion) and region.photo is not None and region.photo.corners is not None:
                    seed = tuple(int(round(v)) for v in region.box)
                    found = [p for p in find_photos(gray, [seed], dpi) if p.corners is not None]
                    photos.extend(found[:1])
            patches.photos_by_page[id(page)] = photos
        return page

    def rule_samples(page, dpi):
        h, v, count = original[2](page, dpi)
        added = 0
        for photo in patches.photos_by_page.get(id(page), []):
            ph, pv, pc = photo_samples(photo, dpi)
            h, v, count = h + ph, v + pv, count + pc
            added += len(ph) + len(pv)
        patches.sample_log.append(added)
        return h, v, count

    text_run.page_regions, text_run.analyze_page = page_regions, analyze_page
    if patches.variant is Variant.SIDES:
        text_run.rule_samples = rule_samples
    try:
        yield patches
    finally:
        text_run.page_regions, text_run.analyze_page, text_run.rule_samples = original


@dataclass(frozen=True)
class SideStraightness:
    """Прямизна стороны фото на итоге.

    Поля:
        side: Сторона.
        tilt_deg: Наклон хорды от оси, градусы (модуль).
        ptp_mm: Размах точек края от хорды (без 5 % выбросов с каждой стороны), мм.
    """

    side: Side
    tilt_deg: float
    ptp_mm: float


def straightness(photo: PhotoBound) -> list[SideStraightness]:
    """Прямизна чётких сторон фото: наклон прямой МНК по годным точкам края и размах остатка от неё (p95 − p5).

    Args:
        photo: Фото (на итоге выпрямления).

    Returns:
        Меры по чётким сторонам.
    """
    out = []
    for fit, sharp in zip(photo.sides, photo.sharp_sides):
        if not sharp or fit.inlier.sum() < 6:
            continue
        a, c = fit.along[fit.inlier], fit.across[fit.inlier]
        slope, icept = np.polyfit(a, c, 1)
        r = c - (slope * a + icept)
        out.append(SideStraightness(fit.side, float(np.degrees(np.arctan(abs(slope)))),
                                    px_to_mm(float(np.percentile(r, 95) - np.percentile(r, 5)), photo.dpi)))  # fmt: skip
    return out


def photos_on(image: np.ndarray, seeds: list[tuple[int, int, int, int]], dpi: float) -> list[PhotoBound]:
    """Фото на картинке по затравкам (для мер итога).

    Args:
        image: BGR.
        seeds: Затравки px.
        dpi: Разрешение.

    Returns:
        Фото с чёткими сторонами.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [p for p in find_photos(gray, seeds, dpi) if p.sides]


def dpi_of(px_per_mm: float) -> float:
    """px/мм → dpi."""
    return px_per_mm * MM_PER_INCH


@dataclass(frozen=True)
class DemoJob:
    """Кадр демонстрации.

    Поля:
        label: Подпись набора.
        src: Исходный кадр прогона v2.
        input_dir: Папка исходных кадров набора.
        cache: Кэш v2 набора.
        v2_output: Итоги v2 набора.
        px_per_mm: Плотность бумаги, px/мм.
        out_dir: Папка выхода демонстрации.
    """

    label: str
    src: Path
    input_dir: Path
    cache: Path
    v2_output: Path
    px_per_mm: float
    out_dir: Path


# Параметры второго преобразования — как у боевого прогона v4 (run_page_rectify_text.sh): шаг поля 3 мм, у корешка 3 мм.
DEMO_FIELD_STEP_MM = 3.0
DEMO_GUTTER_SMOOTH_MM = 3.0


def line_points(page, dpi: float) -> np.ndarray:
    """Точки выборок строк страницы (``line_samples``, положение до выпрямления) — для листов и счёта.

    Args:
        page: Разбор страницы (``PageText``) или None.
        dpi: Разрешение холста.

    Returns:
        Точки (N, 2) px холста.
    """
    from ocr_utils.scan_cropping.v2.rectify.text_field import line_samples

    if page is None:
        return np.zeros((0, 2))
    samples, _ = line_samples(page, dpi)
    return samples.source


def run_demo(job: DemoJob) -> dict:
    """Кадр тремя вариантами (:class:`Variant`): итоги, области фото, выборки, меры прямизны сторон фото и текста.

    Args:
        job: Кадр.

    Returns:
        Сводка кадра (JSON-совместимая); картинки и ``.npz`` — в ``out_dir/<кадр>/``.
    """
    from ocr_utils.scan_cropping.v2.pipeline import FrameJob
    from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod
    from ocr_utils.scan_cropping.v2.rectify.coons import TANGENT_MM
    from ocr_utils.scan_cropping.v2.rectify.text_field import ACROSS_SMOOTH_MM, FIELD_SMOOTH_MM
    from ocr_utils.scan_cropping.v2.rectify.text_run import TextParams, text_frame
    from research.page_rectify.cli import V2_SUFFIX, RunConfig, Status, load_frame

    frame = FrameJob(job.src, job.src.relative_to(job.input_dir))
    out = job.out_dir / job.label / frame.rel.stem
    out.mkdir(parents=True, exist_ok=True)
    # Вето фона — если прогон v2 шёл с ним (в кэше кадра есть ``book_veto.npz``; пак-6 v6 — без вето, ХиП 1991 — с ним).
    bg_veto = (job.cache / frame.key / "book_veto.npz").exists()
    cfg = RunConfig(job.input_dir, job.cache, job.v2_output, out / "copy", None, bg_veto, SpineMethod.AUTO, TANGENT_MM,
                    False, job.label, job.px_per_mm)  # fmt: skip
    v2_path = job.v2_output / frame.rel.with_suffix(V2_SUFFIX)
    loaded = load_frame(frame, cfg, v2_path, out / "copy.tiff", {})
    if isinstance(loaded, Status):
        return {"frame": str(frame.rel), "status": loaded.value}
    image, dpi, geometry = loaded
    params = TextParams(TANGENT_MM, DEMO_FIELD_STEP_MM, FIELD_SMOOTH_MM, DEMO_GUTTER_SMOOTH_MM, ACROSS_SMOOTH_MM)
    summary = {"frame": str(frame.rel), "label": job.label, "dpi": dpi, "variants": {}}
    seeds = None
    for variant in Variant:
        state = Patches(variant, dpi)
        with patched(state):
            result = text_frame(image, geometry, dpi, params)
        if variant is Variant.QUAD:
            # Затравки фото для мер итога — найденные на холсте после Кунса области фото.
            seeds = [tuple(int(round(v)) for v in r.box) for _, _, reg in state.regions_log for r in reg
                     if isinstance(r, PhotoRegion)]  # fmt: skip
            regions_dump = [(list(box), [list(r.box) for r in before if r.kind.value == "raster"],
                             [r.polygon.tolist() for r in after if isinstance(r, PhotoRegion)])
                            for box, before, after in state.regions_log]  # fmt: skip
            summary["regions"] = regions_dump
        pages0 = result.analyses[0]
        points = [line_points(p, dpi) for p in pages0]
        summary["variants"][variant.value] = {
            "lines": [int(p.lines) for p in result.pages],
            "modes": [p.mode.value for p in result.pages],
            "line_points": [len(p) for p in points],
            "photo_side_samples": state.sample_log[:2],
            **result.before.row("before_"),
            **result.after.row("after_"),
        }
        np.savez_compressed(out / f"{variant.value}.npz", points_l=points[0], points_r=points[1])
        # Подписи: «°» на листах не печатается шрифтом подписей side_by_side — меры в шапке пишутся словами.
        cv2.imwrite(str(out / f"{variant.value}_first.jpg"), result.first, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out / f"{variant.value}_final.jpg"), result.final, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # Прямизна сторон фото: на холсте после Кунса (общий для вариантов) и на итоге каждого варианта.
    first = cv2.imread(str(out / "base_first.jpg"))
    measures = {"first": [[s.__dict__ | {"side": s.side.value} for s in straightness(p)]
                          for p in photos_on(first, seeds or [], dpi)]}  # fmt: skip
    for variant in Variant:
        final = cv2.imread(str(out / f"{variant.value}_final.jpg"))
        measures[variant.value] = [[s.__dict__ | {"side": s.side.value} for s in straightness(p)]
                                   for p in photos_on(final, seeds or [], dpi)]  # fmt: skip
    summary["photo_sides"] = measures
    summary["seeds"] = [list(s) for s in seeds or []]
    summary["status"] = "ok"
    (out / "summary.json").write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=1, default=float))
    return summary


# Поле вырезки вокруг фото на листе, мм; ширина панели, px.
SHEET_MARGIN_MM = 12.0
SHEET_PANEL_W = 700
COLOUR_SIDE = (200, 60, 200)  # подогнанная сторона фото — фиолетовый, как четырёхугольник на листах пака-1
COLOUR_CHORD = (0, 165, 255)  # прямая через концы стороны — оранжевый, справочная
COLOUR_RECT = (0, 165, 255)  # рамка растра детектора
COLOUR_KEPT = (0, 150, 0)  # выборка строки в опорах
COLOUR_BACK = (220, 90, 20)  # выборка строки, вернувшаяся из зоны отсева


def draw_photo(view: np.ndarray, photo: PhotoBound, offset: np.ndarray, scale: float) -> None:
    """Подогнанные чёткие стороны фото и прямые через их концы.

    Args:
        view: Панель BGR (рисуется на месте).
        photo: Фото.
        offset: Левый верх вырезки, px.
        scale: Увеличение вырезки.
    """
    for fit, sharp in zip(photo.sides, photo.sharp_sides):
        if not sharp or fit.coef is None:
            continue
        pts = (fit.curve(*fit.span) - offset) * scale
        cv2.polylines(view, [np.round(pts).astype(np.int32)], False, COLOUR_SIDE, 2, cv2.LINE_AA)
        a, b = (int(round(v)) for v in pts[0]), (int(round(v)) for v in pts[-1])
        cv2.line(view, tuple(a), tuple(b), COLOUR_CHORD, 1, cv2.LINE_AA)


def panel(image: np.ndarray, box, photo: PhotoBound | None, dpi: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Вырезка вокруг рамки с полем :data:`SHEET_MARGIN_MM`, ужатая до :data:`SHEET_PANEL_W`, со сторонами фото.

    Args:
        image: BGR.
        box: Рамка фото px.
        photo: Фото на этой картинке или None.
        dpi: Разрешение.

    Returns:
        ``(панель, левый верх вырезки, увеличение)``.
    """
    h, w = image.shape[:2]
    m = SHEET_MARGIN_MM * dpi / 25.4
    x0, y0 = int(max(box[0] - m, 0)), int(max(box[1] - m, 0))
    x1, y1 = int(min(box[2] + m, w)), int(min(box[3] + m, h))
    scale = SHEET_PANEL_W / (x1 - x0)
    view = cv2.resize(image[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if photo is not None:
        draw_photo(view, photo, np.array([x0, y0], float), scale)
    return view, np.array([x0, y0], float), scale


def demo_sheets(frame_dir: Path) -> list[Path]:
    """Листы кадра демонстрации: (1) «как есть | область фото | плюс стороны в опорах» — итог у каждого фото со
    сторонами и мерами; (2) отсев выборок строк у фото: рамка растра против области фото на холсте после Кунса.

    Args:
        frame_dir: Папка кадра демонстрации (``summary.json``, ``*_final.jpg``, ``*.npz``).

    Returns:
        Пути листов.
    """
    import json

    from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
    from ocr_utils.page_layout.text_blocks.sides_overlay import side_by_side

    summary = json.loads((frame_dir / "summary.json").read_text())
    dpi = float(summary["dpi"])
    seeds = [tuple(s) for s in summary["seeds"]]
    paths = []
    finals = {v: cv2.imread(str(frame_dir / f"{v.value}_final.jpg")) for v in Variant}
    legend = [LegendEntry("подогнанная сторона фото (чёткая)", COLOUR_SIDE),
              LegendEntry("прямая через концы стороны", COLOUR_CHORD)]  # fmt: skip
    for i, seed in enumerate(seeds):
        views, titles = [], []
        for variant in Variant:
            photos = photos_on(finals[variant], [seed], dpi)
            photo = photos[0] if photos else None
            view, _, _ = panel(finals[variant], seed, photo, dpi)
            stats = straightness(photo) if photo is not None else []
            text = ", ".join(f"{s.side.value[0]} {s.tilt_deg:.2f}/{s.ptp_mm:.2f}" for s in stats)
            info = summary["variants"][variant.value]
            views.append(view)
            titles.append(f"{variant.value}: {text}; строки p95 {info['after_line_p95_mm']} мм")
        head = [
            f"{summary['label']}/{summary['frame']} фото {i + 1} — итог выпрямления тремя вариантами",
            "base — как в бою; quad — область фото четырёхугольником; sides — плюс стороны фото в опорах поля",
            "у каждой чёткой стороны (t верх, b низ, l левая, r правая): наклон, градусы / размах точек края от прямой, мм (p95−p5)",
        ]
        picture = framed(side_by_side(views, titles), head, legend)
        path = frame_dir / f"sheet_photo{i + 1}.jpg"
        cv2.imwrite(str(path), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    # Отсев выборок строк: точки выборок до (рамка) и после (область фото) на холсте после Кунса.
    first = cv2.imread(str(frame_dir / "base_first.jpg"))
    base, quad = np.load(frame_dir / "base.npz"), np.load(frame_dir / "quad.npz")
    views, titles = [], []
    for page, (box, rects, polys) in enumerate(summary.get("regions", [])):
        if not rects:
            continue
        x0, y0, x1, y1 = box
        side = "points_l" if page == 0 else "points_r"
        before, after = base[side], quad[side]
        crop = first[y0:y1, x0:x1].copy()
        scale = SHEET_PANEL_W / crop.shape[1]
        for name, kept, regions in (("рамка растра", before, rects), ("область фото", after, polys)):
            view = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            off = np.array([x0, y0], float)
            if name == "рамка растра":
                for r in regions:
                    a = np.round((np.array(r[:2]) - off) * scale).astype(int)
                    b = np.round((np.array(r[2:]) - off) * scale).astype(int)
                    cv2.rectangle(view, tuple(a), tuple(b), COLOUR_RECT, 2)
            else:
                for poly in regions:
                    pts = np.round((np.array(poly) - off) * scale).astype(np.int32)
                    cv2.polylines(view, [pts], True, COLOUR_SIDE, 2, cv2.LINE_AA)
            old = {tuple(np.round(p).astype(int)) for p in before}
            for p in kept:
                back = name == "область фото" and tuple(np.round(p).astype(int)) not in old
                q = tuple(np.round((p - off) * scale).astype(int))
                cv2.circle(view, q, 3 if back else 2, COLOUR_BACK if back else COLOUR_KEPT, -1)
            views.append(view)
            titles.append(f"стр. {page + 1}, {name}: выборок строк {len(kept)}")
    if views:
        legend2 = [LegendEntry("рамка растра детектора (было)", COLOUR_RECT),
                   LegendEntry("область фото (стало)", COLOUR_SIDE),
                   LegendEntry("выборка строки в опорах", COLOUR_KEPT),
                   LegendEntry("выборка строки, вернувшаяся с областью фото", COLOUR_BACK)]  # fmt: skip
        head = [f"{summary['label']}/{summary['frame']} — отсев выборок строк у фото (холст после Кунса)"]
        picture = framed(side_by_side(views, titles), head, legend2)
        path = frame_dir / "sheet_supports.jpg"
        cv2.imwrite(str(path), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    return paths
