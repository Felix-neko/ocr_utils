"""Конвейер v2 по папке кадров: пальцы (GPU) → фон и книга (пул) → SAM 3 (GPU) → бумага и страницы (пул) → surya по выпрямленному кадру (GPU) → защита, copy back, закрас до прямоугольника (пул)."""

from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import threadpoolctl

from ocr_utils.scan_cropping.image_io import imwrite_params, resolve_output_suffix, write_image
from ocr_utils.scan_cropping.levels import LEVELS_MAX_BLACK, compensate_levels
from ocr_utils.scan_cropping.v2.background import BG_SCALE, bg_flood, hand_mask
from ocr_utils.scan_cropping.v2.book import book_close_open, content_hulls
from ocr_utils.scan_cropping.v2.colour_pages import colour_pages_flat, colour_pages_growth
from ocr_utils.scan_cropping.v2.content_guard import protect_content
from ocr_utils.scan_cropping.v2.copy_back import CORE_GROW_PX, LAYOUT_PAD_PX, copy_back_blocks
from ocr_utils.scan_cropping.v2.edge import SNAP_BG_REACH, snap_outward
from ocr_utils.scan_cropping.v2.border_fill import SEAM_MM
from ocr_utils.scan_cropping.v2.fill import (
    FillMethod,
    ReplicateParams,
    fill_then_overlay,
    fill_to_rectangle,
    paper_bbox,
)
from ocr_utils.scan_cropping.v2.finger_shadow import SHADOW_SCALE, estimate_finger_shadow, lighten_finger_shadow
from ocr_utils.scan_cropping.v2.fingers import INK_CORE_BAND_PX, LAMA_HOLE_MAX_PX, LIGHT_INCREMENT, TIGHT_DILATE_PX
from ocr_utils.scan_cropping.v2.glue import glue_hull_bridge
from ocr_utils.scan_cropping.v2.glue_refine import REFINE_DEPTH, refine_colour
from ocr_utils.scan_cropping.v2.glyphs import content_mask
from ocr_utils.scan_cropping.v2.ink_fill import fill_bold_ink, touching_ink
from ocr_utils.scan_cropping.v2.keystone import plan_warp, unwarp_polygons, warp_for_layout
from ocr_utils.scan_cropping.v2.page_pose import PageQuad, page_quads
from ocr_utils.scan_cropping.v2.paper import SAM3_CACHE_VERSION, fill_page_pockets, postprocess_plain
from ocr_utils.scan_cropping.v2.scale import REF_PX_PER_MM, Scale
from ocr_utils.scan_cropping.cropping import CROP_FILL_BLUR_PX
from ocr_utils.scan_cropping.v2.side_fit import regularize_sides, snap_sheet_edges

log = logging.getLogger(__name__)

# Умолчания опций CLI в мм бумаги: те же припуски, что были заданы в px калибровочного кадра (13.8 px/мм, подшивки
# МТО 1990) — при этой плотности они дают ровно прежние пиксели, при любой другой (450 dpi = 17.7 px/мм) — те же мм.
FINGER_DILATE_MM = round(TIGHT_DILATE_PX / REF_PX_PER_MM, 4)  # 40 px → 2.8986 мм
LAYOUT_PAD_MM = (round(LAYOUT_PAD_PX[0] / REF_PX_PER_MM, 4), round(LAYOUT_PAD_PX[1] / REF_PX_PER_MM, 4))  # 18, 36 px
CORE_GROW_MM = round(CORE_GROW_PX / REF_PX_PER_MM, 4)  # 4 px → 0.2899 мм
REFINE_DEPTH_MM = round(REFINE_DEPTH / (REF_PX_PER_MM * BG_SCALE), 4)  # 35 px сегментации (6.9 px/мм) → 5.0725 мм
CROP_FILL_BLUR_MM = round(CROP_FILL_BLUR_PX / REF_PX_PER_MM, 6)  # replicate: 48 px → 3.478261 мм

# Число CPU-воркеров по умолчанию: 16 физических ядер; на кадр в воркере держится несколько копий полного кадра
# (~75 МБ каждая), так что память не ограничивает.
DEFAULT_JOBS = 16
# Версия формата ``layout.json``: 4 — блоки surya после бумаги и трапеций, по выпрямленному кадру, с типами блоков
# (``{"version": 4, "keystone": bool, "blocks": [...], "labels": [...]}``). Версия 3 (без типов) читается — блоки
# годны для защиты и copy back, но шаг surya пересчитывается, чтобы на оверлеях были типы; прежние форматы
# (surya до SAM 3) не читаются.
LAYOUT_VERSION = 4
# Версии ``layout.json``, блоки которых годны в дело.
LAYOUT_READABLE = (3, 4)
# Подпапка ``--debug`` для картинок «вход surya: выпрямленный кадр с рамками».
SURYA_INPUT_DIR = "surya_input"


@dataclass(frozen=True)
class RunParams:
    """Параметры прогона v2.

    Поля:
        input_dir: Папка входных кадров (относительные пути выхода считаются от неё).
        output_dir: Папка итоговых прямоугольников.
        work_dir: Папка промежуточных данных по кадрам (кэш шагов: повторный прогон берёт готовое).
        debug_dir: Папка оверлеев или None.
        scale: Масштабы по плотности кадра.
        fill: Способ закраса до прямоугольника.
        fill_seam_mm: Ширина мягкого шва от каймы листа к закрасу, мм (для ``fill=pull_push_seam``).
        replicate: Параметры прежней заливки v1 (для ``fill=replicate``).
        hole_max_px: Предел рамки дыры для LaMa, px; None — прежнее ужатие ROI до 512.
        finger_dilate_mm: Припуск зоны закраса пальца, мм бумаги.
        layout_pad_mm: Припуск блоков surya, мм бумаги (X, Y) — один для copy back и защиты контента.
        light_increment: Высветление зоны пальца перед LaMa, уровни (слева, справа).
        keystone_layout: surya layout по кадру, выпрямленному по трапециям страниц (``keystone``); блоки
            возвращаются обратным преобразованием наклонными четырёхугольниками.
        fit_sides: Устранять выступы и ямы на внешних боковых сторонах плоских страниц (``side_fit``).
        snap_sides: При выравнивании подтягивать сторону к настоящему краю листа, где SAM 3 колеблется.
        copy_back_over_fingers: Copy back блоков surya и поверх ядра пальца и рук (весь «не фон»): контент важнее
            вида — при ложном срабатывании детектора пальцев текст не теряется, при верном возвращается кусок пальца.
        fill_before_copy_back: Закрашивать до прямоугольника от точного края бумаги (без защиты контента и copy back),
            а защиту и copy back класть поверх готового закраса: печать, край обложки и кусок пальца в блоке surya у
            края не становятся опорой Telea и не тянутся шлейфом (МТС 1990/03 IMG_0973).
        ink_fill: Второй проход SAM 3 по кадру без крупной краски (``ink_fill``), бумага — объединение двух проходов.
        refine_depth_mm: Глубина уточнения склейки от фона, мм бумаги (в px масштаба сегментации — ``Scale.seg_px``).
        compensate_levels: Растянуть уровни итога.
        levels_max_black: Потолок чёрной точки растяжки, уровни; None — без потолка.
        output_format: Формат итога (``png`` / ``tiff``) или None — как у входа.
        force_dpi: Записать в итог это разрешение или None.
        skip_if_exists: Пропускать кадры, итог которых уже есть.
        redo: Пересчитать кэш шагов в ``work_dir``.
        jobs: Число CPU-воркеров.
        save_fingers: Оставлять в кэше кадр без пальцев (``clean.png``, ~18 МБ); по умолчанию он удаляется после
            финала кадра.
        finger_shadow: После copy back и растяжки уровней осветлять тень пальца по карте (``finger_shadow``): закрас
            LaMa продолжает тень пальца с бумаги у края зоны, copy back возвращает её с оригинала.
        save_before_shadow: Сохранять в кэш кадра итог до осветления тени и маски входа карты (``before_shadow.png``,
            ``shadow_masks.npz``, ~40 МБ на кадр) — для стенда ``research/finger_repaint edge``.
        copy_back_core_grow_mm: Припуск ядра пальца при copy back, мм бумаги (``copy_back.CORE_GROW_PX`` = 4 px
            калибровочного кадра): кайма кожи у края ядра не возвращается с оригинала; 0 — без припуска.
        colour_pages: Правила для целиком цветных страниц (``colour_pages``): рост бумаги у края по палитре страницы
            вверх и вниз, защита контента при любой позе.
    """

    input_dir: Path
    output_dir: Path
    work_dir: Path
    debug_dir: Path | None = None
    scale: Scale = field(default_factory=Scale)
    fill: FillMethod = FillMethod.PULL_PUSH_SEAM
    fill_seam_mm: float = SEAM_MM
    replicate: ReplicateParams = field(default_factory=ReplicateParams)
    hole_max_px: int | None = LAMA_HOLE_MAX_PX
    finger_dilate_mm: float = FINGER_DILATE_MM
    layout_pad_mm: tuple[float, float] = LAYOUT_PAD_MM
    light_increment: tuple[float, float] = LIGHT_INCREMENT
    ink_fill: bool = True
    fit_sides: bool = True
    keystone_layout: bool = True
    snap_sides: bool = False
    copy_back_over_fingers: bool = True
    fill_before_copy_back: bool = True
    refine_depth_mm: float = REFINE_DEPTH_MM
    compensate_levels: bool = False
    levels_max_black: int | None = LEVELS_MAX_BLACK
    output_format: str | None = None
    force_dpi: int | None = None
    skip_if_exists: bool = False
    redo: bool = False
    jobs: int = DEFAULT_JOBS
    save_fingers: bool = False
    finger_shadow: bool = False
    save_before_shadow: bool = False
    copy_back_core_grow_mm: float = CORE_GROW_MM
    colour_pages: bool = True


@dataclass(frozen=True)
class FrameJob:
    """Кадр прогона.

    Поля:
        src: Путь входного кадра.
        rel: Путь относительно ``input_dir``.
    """

    src: Path
    rel: Path

    @property
    def key(self) -> str:
        """Плоский ключ кадра для папки кэша: относительный путь без расширения, ``/`` → ``__``."""
        return "__".join(self.rel.with_suffix("").parts)


@dataclass(frozen=True)
class WorkPaths:
    """Файлы кэша одного кадра в ``work_dir/<ключ>/``."""

    root: Path

    @property
    def clean(self) -> Path:
        """Кадр без пальцев (PNG без потерь)."""
        return self.root / "clean.png"

    @property
    def fingers(self) -> Path:
        """Маски закраса пальцев: зона и ядро (npz)."""
        return self.root / "fingers.npz"

    @property
    def layout(self) -> Path:
        """Полигоны блоков surya в координатах кадра (json)."""
        return self.root / "layout.json"

    @property
    def seg(self) -> Path:
        """Кадр без пальцев на масштабе сегментации (PNG)."""
        return self.root / "seg.png"

    @property
    def before_shadow(self) -> Path:
        """Итог до осветления тени пальца (PNG без потерь) — вход стенда ``research/finger_repaint edge``."""
        return self.root / "before_shadow.png"

    @property
    def shadow_masks(self) -> Path:
        """Маски входа карты тени в координатах итога: ``paint``, ``core``, ``book``, ``original`` (npz)."""
        return self.root / "shadow_masks.npz"

    @property
    def seg_noink(self) -> Path:
        """Кадр сегментации без крупной краски — вход второго прохода SAM 3 (PNG)."""
        return self.root / "seg_noink.png"

    @property
    def book(self) -> Path:
        """Маски фона, книги, печати и оболочек на масштабе сегментации (npz)."""
        return self.root / "book.npz"

    @property
    def sam3(self) -> Path:
        """Сырая маска SAM 3 и карта экземпляров-страниц (npz)."""
        return self.root / "sam3.npz"

    @property
    def pages(self) -> Path:
        """Страницы кадра: трапеции, признаки и класс положения (json)."""
        return self.root / "pages.json"

    @property
    def masks(self) -> Path:
        """Маски бумаги по шагам на масштабе сегментации (npz)."""
        return self.root / "paper.npz"


@dataclass(frozen=True)
class FrameStats:
    """Строка сводки по кадру.

    Поля:
        key: Ключ кадра.
        status: ``ok`` / ``no_paper`` / ``error``.
        finger_px: Площадь зоны закраса пальцев, px кадра.
        blocks_back: Сколько блоков surya задето закрасом (возвращены внутри книги).
        back_px: Сколько px зоны закраса возвращено с оригинала.
        glue_added: Добавлено склейкой и оставлено уточнением, доля маски бумаги.
        refine_cut: Срезано уточнением, доля маски бумаги.
        ink_added: Добавлено вторым проходом SAM 3 (кадр без крупной краски), доля маски бумаги.
        side_added: Залито выравниванием боковых сторон, доля маски бумаги.
        side_cut: Срезано выравниванием боковых сторон, доля маски бумаги.
        protected: Добавлено защитой контента (блоки surya плоских страниц), доля маски бумаги.
        poses: Классы положения страниц слева направо через ``/`` (``flat`` / ``raised``).
        fill_share: Доля прямоугольника вне бумаги (закрашено).
        width: Ширина итога, px.
        height: Высота итога, px.
        seconds: Время CPU-части финала, с.
        shadow_mm2: Площадь зоны поиска тени пальца, мм² (0 — шаг выключен или пальца на бумаге нет).
        shadow_mean: Средняя прибавка осветления тени в зоне поиска, уровни.
        message: Текст ошибки или пусто.
    """

    key: str
    status: str
    finger_px: int = 0
    blocks_back: int = 0
    back_px: int = 0
    glue_added: float = 0.0
    refine_cut: float = 0.0
    ink_added: float = 0.0
    side_added: float = 0.0
    side_cut: float = 0.0
    protected: float = 0.0
    poses: str = ""
    fill_share: float = 0.0
    width: int = 0
    height: int = 0
    seconds: float = 0.0
    shadow_mm2: float = 0.0
    shadow_mean: float = 0.0
    message: str = ""


def init_worker() -> None:
    """Инициализатор воркера: без hugepage и потоков BLAS/OpenCV (см. ``.claude/rules/gpu_and_pools.md``)."""
    np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def read_bgr(path: Path) -> np.ndarray:
    """Читает картинку в BGR по байтам (путь с кириллицей и пробелами).

    Args:
        path: Путь к картинке.

    Returns:
        Массив ``(H, W, 3)`` uint8.

    Raises:
        ValueError: Картинка не читается.
    """
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не читается: {path}")
    return img


def write_png(path: Path, img: np.ndarray) -> None:
    """Пишет PNG с быстрым сжатием (кэш шагов), создавая папку.

    Args:
        path: Путь.
        img: Картинка.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def output_path(job: FrameJob, params: RunParams) -> Path:
    """Путь итога кадра.

    Args:
        job: Кадр.
        params: Параметры прогона.

    Returns:
        Путь в ``output_dir`` с расширением по ``output_format``.
    """
    return params.output_dir / job.rel.with_suffix(resolve_output_suffix(job.rel.suffix, params.output_format))


def seg_shape(shape: tuple[int, int], scale: Scale) -> tuple[int, int]:
    """Размер масштаба сегментации ``(H, W)`` для кадра размера ``shape``.

    Args:
        shape: ``(H, W)`` кадра.
        scale: Масштабы прогона.

    Returns:
        ``(H, W)`` на масштабе сегментации (как ``cv2.resize`` с ``fx = fy = scale.seg``).
    """
    h, w = shape
    return int(round(h * scale.seg)), int(round(w * scale.seg))


def to_frame(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Маска масштаба сегментации → размер кадра: билинейно и порог 128 (край не ступенчатый).

    Args:
        mask: Булева маска.
        shape: ``(H, W)`` кадра.

    Returns:
        Булева маска размера кадра.
    """
    m = cv2.resize(mask.astype(np.uint8) * 255, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return m >= 128


# --- Шаг 1, GPU: пальцы и блоки surya ---------------------------------------------------------------------------


def gpu_fingers(job: FrameJob, models, params: RunParams) -> None:
    """Закрас пальцев (inkback) одного кадра → кэш.

    Args:
        job: Кадр.
        models: ``GpuModels`` с детекцией и LaMa.
        params: Параметры прогона.
    """
    from ocr_utils.scan_cropping.v2.fingers import remove_fingers_inkback

    work = WorkPaths(params.work_dir / job.key)
    bgr = read_bgr(job.src)
    t0 = time.perf_counter()
    result = remove_fingers_inkback(
        bgr,
        models,
        dilate_px=params.scale.mm_px(params.finger_dilate_mm),
        hole_max_px=params.hole_max_px,
        core_band_px=params.scale.frame_px(INK_CORE_BAND_PX),
        glyph_scale=params.scale.glyph,
        light_increment=params.light_increment,
        log_name=job.key,
        px_per_mm=params.scale.px_per_mm,
    )
    write_png(work.clean, result.clean)
    np.savez_compressed(work.fingers, paint=result.paint > 0, core=result.core > 0)
    log.info("%s: пальцы %.1f с, %s", job.key, time.perf_counter() - t0, result.info)


# --- Шаг 2, CPU: фон и грубая маска книги ------------------------------------------------------------------------


def cpu_book(args: tuple[FrameJob, RunParams]) -> str:
    """Своя печать, фон, грубая маска книги и оболочки печати одного кадра (для пула) → кэш.

    Args:
        args: ``(кадр, параметры)``.

    Returns:
        Ключ кадра.
    """
    job, params = args
    work = WorkPaths(params.work_dir / job.key)
    if book_ready(work, params):
        return job.key
    clean = read_bgr(work.clean)
    h, w = seg_shape(clean.shape[:2], params.scale)
    img = cv2.resize(clean, (w, h), interpolation=cv2.INTER_AREA)
    # Своя печать — защита глифов с узким зазором по кадру без пальцев.
    content = content_mask(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB), (h, w), params.scale.glyph)
    bg = bg_flood(img, content)
    # Оболочки печати для книги — внутри «не фон»; для бумаги — внутри уже готовой книги.
    book = book_close_open(bg, content_hulls(content, ~bg))
    hulls = content_hulls(content, book)
    write_png(work.seg, img)
    if params.ink_fill:
        # Кадр для второго прохода SAM 3: напечатанные рамки и плашки закрашены цветом бумаги, иначе SAM 3 принимает
        # толстую рамку за край листа (МТС 1991/05 IMG_0250).
        noink, ink = fill_bold_ink(img, book, bg, hulls)
        write_png(work.seg_noink, noink)
        np.savez_compressed(work.book, content=content, bg=bg, book=book, hulls=hulls, ink=ink)
    else:
        np.savez_compressed(work.book, content=content, bg=bg, book=book, hulls=hulls)
    return job.key


def book_ready(work: WorkPaths, params: RunParams) -> bool:
    """Готов ли в кэше шаг «фон и книга» (с кадром без краски, если нужен второй проход SAM 3).

    Args:
        work: Файлы кэша кадра.
        params: Параметры прогона.

    Returns:
        True — шаг можно не считать.
    """
    if params.redo or not (work.book.exists() and work.seg.exists()):
        return False
    if not params.ink_fill:
        return True
    if not work.seg_noink.exists():
        return False
    with np.load(work.book) as data:
        return "ink" in data.files


def sam3_ready(work: WorkPaths, params: RunParams) -> bool:
    """Готов ли в кэше шаг SAM 3 (с вторым проходом по кадру без краски, если он нужен).

    Args:
        work: Файлы кэша кадра.
        params: Параметры прогона.

    Returns:
        True — шаг можно не считать.
    """
    if params.redo or not work.sam3.exists():
        return False
    with np.load(work.sam3) as data:
        # Кэш прежней версии (без правил пустых страниц и слияния вложенных экземпляров) считается заново.
        if "version" not in data.files or int(data["version"]) != SAM3_CACHE_VERSION:
            return False
        return not params.ink_fill or "raw_noink" in data.files


# --- Шаг 3, GPU: SAM 3 --------------------------------------------------------------------------------------------


def gpu_sam3(job: FrameJob, predictor, params: RunParams) -> None:
    """Страницы SAM 3 одного кадра (и второй проход по кадру без крупной краски) → кэш.

    Args:
        job: Кадр.
        predictor: ``SAM3SemanticPredictor`` (``paper.load_sam3``).
        params: Параметры прогона.
    """
    from ocr_utils.scan_cropping.v2.paper import sam3_pages

    work = WorkPaths(params.work_dir / job.key)
    img = read_bgr(work.seg)
    hulls = np.load(work.book)["hulls"]
    t0 = time.perf_counter()
    raw, inst = sam3_pages(predictor, img, hulls)
    out = {"raw": raw, "inst": inst, "version": np.int32(SAM3_CACHE_VERSION)}
    if params.ink_fill:
        # Второй проход — по кадру без крупной краски.
        out["raw_noink"], out["inst_noink"] = sam3_pages(predictor, read_bgr(work.seg_noink), hulls)
    np.savez_compressed(work.sam3, **out)
    # Страницы и их трапеции после нового SAM 3 могут быть другими, а surya считается по кадру, выпрямленному по ним.
    work.layout.unlink(missing_ok=True)
    log.info("%s: SAM 3 %.1f с, страниц %d", job.key, time.perf_counter() - t0, int(inst.max()))


# --- Шаг 4, CPU: бумага, copy back, закрас до прямоугольника ------------------------------------------------------


@dataclass(frozen=True)
class PaperMasks:
    """Маски бумаги по шагам на масштабе сегментации.

    Поля:
        plain: SAM 3 после постобработки ``plain``.
        snapped: После доводки края до фона (``snap_bg``).
        glued: После склейки страниц.
        paper: Итог после уточнения склейки (с объединением второго прохода SAM 3, см. :func:`frame_paper`).
        cut: Срезанное уточнением.
        ink_added: Добавлено вторым проходом SAM 3 по кадру без крупной краски.
        side_added: Залито выравниванием боковых сторон (ямы).
        side_cut: Срезано выравниванием боковых сторон (выбросы).
        pocket_added: Добавлено как карман, листок или вклейка на странице (``paper.fill_page_pockets``).
        palette_added: Добавлено у края цветной страницы по её палитре (``colour_pages.colour_pages_growth``).
    """

    plain: np.ndarray
    snapped: np.ndarray
    glued: np.ndarray
    paper: np.ndarray
    cut: np.ndarray
    ink_added: np.ndarray
    side_added: np.ndarray
    side_cut: np.ndarray
    pocket_added: np.ndarray | None = None
    palette_added: np.ndarray | None = None


def paper_masks(img: np.ndarray, book: dict, sam3: dict, refine_depth: int) -> PaperMasks:
    """Маска бумаги: SAM 3 ``plain`` → ``snap_bg`` → склейка ``hull_bridge`` без фильтра → уточнение ``colour``.

    Args:
        img: Кадр без пальцев на масштабе сегментации.
        book: Маски шага 2 (``content``, ``bg``, ``book``, ``hulls``).
        sam3: Выход SAM 3 (``raw``, ``inst``).
        refine_depth: Глубина уточнения от фона, px.

    Returns:
        Маски по шагам.
    """
    plain = postprocess_plain(sam3["raw"], book["book"], book["hulls"])
    snapped = snap_outward(img, plain, book["book"], book["bg"], bg_reach=SNAP_BG_REACH)
    glued = glue_hull_bridge(snapped, book["book"], book["content"], sam3["inst"]).paper
    refined = refine_colour(img, book["book"], snapped, glued, book["hulls"], refine_depth)
    # Карманы, листки, вклейки на странице, которые SAM 3 не взял в экземпляр страницы.
    pocket = fill_page_pockets(img, refined.paper, book["book"], book["bg"], book["hulls"])
    empty = np.zeros_like(plain)
    return PaperMasks(plain, snapped, glued, refined.paper | pocket, refined.cut, empty, empty, empty, pocket, empty)


def frame_paper(img: np.ndarray, book: dict, sam3: dict, refine_depth: int) -> PaperMasks:
    """Бумага кадра: маски по SAM 3 по кадру ∪ итог по второму проходу SAM 3 (кадр без крупной краски), если он есть.

    Из добавки второго прохода берутся только куски, касающиеся закрашенной краски (``ink_fill.touching_ink``).
    Объединение, а не замена: по кадру без краски SAM 3 местами берёт лист хуже (стенд ``research/sam3_ink_fill``:
    IMG_0663 без объединения потерял бы 29 тыс. px), а где напечатанная рамка останавливала SAM 3, добавляется поле до
    края листа (IMG_0250: +366 тыс. px сегментации).

    Args:
        img: Кадр без пальцев на масштабе сегментации.
        book: Маски шага 2.
        sam3: Выход SAM 3 (``raw``, ``inst`` и, если был второй проход, ``raw_noink``, ``inst_noink``).
        refine_depth: Глубина уточнения от фона, px.

    Returns:
        Маски по шагам первого прохода с объединённым итогом ``paper`` и добавкой ``ink_added``.
    """
    masks = paper_masks(img, book, sam3, refine_depth)
    if "raw_noink" not in sam3:
        return masks
    alt = paper_masks(img, book, {"raw": sam3["raw_noink"], "inst": sam3["inst_noink"]}, refine_depth).paper
    # Принимаются только куски добавки, касающиеся закрашенной краски: остальное — колебания SAM 3.
    added = touching_ink(alt & ~masks.paper, book["ink"]) if "ink" in book else np.zeros_like(masks.paper)
    return replace(masks, paper=masks.paper | added, ink_added=added)


def layout_data(path: Path) -> dict | None:
    """Содержимое ``layout.json`` кадра, если формат годен в дело (``LAYOUT_READABLE``).

    Args:
        path: ``layout.json`` кадра.

    Returns:
        Словарь файла или None, если файла нет или он в прежнем формате (surya только на кадрах с пальцем).
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("version") not in LAYOUT_READABLE:
        return None
    return data


def read_layout(path: Path) -> list[np.ndarray] | None:
    """Полигоны блоков surya из кэша.

    Args:
        path: ``layout.json`` кадра.

    Returns:
        Полигоны (px кадра) или None, если файла нет или он в прежнем формате.
    """
    data = layout_data(path)
    if data is None:
        return None
    return [np.asarray(p, np.float32) for p in data["blocks"]]


def read_layout_labels(path: Path) -> list[str] | None:
    """Типы блоков surya из кэша (``Text``, ``SectionHeader``, ``Picture``...), по порядку ``read_layout``.

    Args:
        path: ``layout.json`` кадра.

    Returns:
        Типы или None, если файла нет или в нём нет типов (версия 3).
    """
    data = layout_data(path)
    if data is None or "labels" not in data:
        return None
    return [str(label) for label in data["labels"]]


def layout_keystone(path: Path) -> bool | None:
    """Был ли surya по выпрямленному кадру (флаг ``keystone`` в ``layout.json``); None — нет годного файла."""
    data = layout_data(path)
    if data is None:
        return None
    return bool(data.get("keystone"))


def layout_ready(work: WorkPaths, params: RunParams) -> bool:
    """Готов ли в кэше шаг surya: файл нужной версии, посчитанный с тем же ключом выпрямления.

    Args:
        work: Файлы кэша кадра.
        params: Параметры прогона.

    Returns:
        True — шаг можно не считать.
    """
    data = layout_data(work.layout)
    if params.redo or data is None or data["version"] != LAYOUT_VERSION:
        # Нет файла или он без типов блоков (версия 3) — шаг считается заново.
        return False
    return bool(data.get("keystone")) or not params.keystone_layout


def read_page_quads(path: Path) -> list[np.ndarray]:
    """Трапеции страниц слева направо из ``pages.json`` (px сегментации), без вырожденных."""
    if not path.exists():
        return []
    return [np.asarray(row["quad"], np.float64) for row in json.loads(path.read_text()) if row.get("quad")]


def paint_on_seg(fingers, shape: tuple[int, int]) -> np.ndarray:
    """Зона закраса пальцев на масштабе сегментации (bool)."""
    return cv2.resize(fingers["paint"].astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0


def frame_pages(img: np.ndarray, book: dict, sam3: dict, paint_seg: np.ndarray, masks: PaperMasks) -> list[PageQuad]:
    """Страницы кадра по итоговой бумаге: трапеции, признаки, класс положения."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return page_quads(masks.paper, book["book"], book["hulls"], sam3["inst"], gray, paint_seg)


# --- Шаг 4, CPU: бумага и страницы --------------------------------------------------------------------------------


def cpu_pages(args: tuple[FrameJob, RunParams]) -> str:
    """Бумага кадра (два прохода SAM 3, выравнивание сторон) и страницы с трапециями (для пула) → кэш.

    Args:
        args: ``(кадр, параметры)``.

    Returns:
        Ключ кадра.
    """
    job, params = args
    work = WorkPaths(params.work_dir / job.key)
    img = read_bgr(work.seg)
    book = dict(np.load(work.book))
    sam3 = dict(np.load(work.sam3))
    if not params.ink_fill:
        sam3.pop("raw_noink", None)
    masks = frame_paper(img, book, sam3, params.scale.seg_px(params.refine_depth_mm))
    paint_seg = paint_on_seg(np.load(work.fingers), img.shape[:2])
    pages = frame_pages(img, book, sam3, paint_seg, masks)
    if params.fit_sides:
        # Поле страницы до кромки листа, где SAM 3 взял его рваными языками (цветная плашка с белым полем, МТС 1990/04
        # IMG_1113); страницы пересчитываются — рваная маска делала страницу «приподнятой».
        edge_added = snap_sheet_edges(masks.paper, pages, book["bg"], book["book"], img)
        if edge_added.any():
            masks = replace(masks, paper=masks.paper | edge_added)
            pages = frame_pages(img, book, sam3, paint_seg, masks)
        # Цветные страницы: поле и край — по палитре уверенного центра страницы, вверх и вниз (верхнее поле МТС 1990/04
        # IMG_1113 — тень кромки нашлась, но край маски не «рваный»); страницы пересчитываются.
        if params.colour_pages:
            palette_added = colour_pages_growth(img, pages, masks.paper, book["book"], book["bg"])
            if palette_added.any():
                masks = replace(masks, paper=masks.paper | palette_added, palette_added=palette_added)
                pages = frame_pages(img, book, sam3, paint_seg, masks)
        # Выравнивание внешних боковых сторон плоских страниц: выбросы SAM 3 (заливка LaMa на торце, IMG_0240),
        # ямы, колебания края в тени у кромки (IMG_0538); затем страницы пересчитываются по выровненной бумаге.
        side = regularize_sides(masks.paper, pages, book["bg"], img, params.snap_sides)
        masks = replace(masks, paper=side.paper, side_added=side.added | edge_added, side_cut=side.cut)
        pages = frame_pages(img, book, sam3, paint_seg, masks)
    np.savez_compressed(work.masks, **asdict(masks))
    # Трапеции изменились (новая бумага) — разметка surya по кадру, выпрямленному по старым трапециям, устарела.
    if work.pages.exists() and not same_quads(
        read_page_quads(work.pages), [p.quad for p in pages if p.quad is not None]
    ):
        work.layout.unlink(missing_ok=True)
    write_pages(work.pages, pages)
    return job.key


def same_quads(old: list[np.ndarray], new: list[np.ndarray], tol: float = 2.0) -> bool:
    """Совпадают ли трапеции страниц (те же по числу, вершины не дальше ``tol`` px сегментации)."""
    return len(old) == len(new) and all(np.abs(np.asarray(a) - np.asarray(b)).max() <= tol for a, b in zip(old, new))


# --- Шаг 5, GPU: surya layout по выпрямленному кадру ---------------------------------------------------------------


def gpu_layout(job: FrameJob, models, params: RunParams) -> None:
    """Блоки surya одного кадра → кэш: по кадру, выпрямленному по трапециям страниц, с обратным переносом блоков.

    Кадр — исходный (с пальцами), как в v1: блок, задетый пальцем, потом вернётся с оригинала. Без трапеций или при
    ``--no-keystone-layout`` — по кадру как есть.

    Args:
        job: Кадр.
        models: ``GpuModels`` с surya layout.
        params: Параметры прогона.
    """
    from ocr_utils.scan_cropping.finger_removal.text_protection import layout_blocks

    work = WorkPaths(params.work_dir / job.key)
    bgr = read_bgr(job.src)
    t0 = time.perf_counter()
    # Трапеции — в px сегментации; размер сегментации тот же, что у шага «фон и книга» (``seg_shape``).
    ratio = bgr.shape[1] / seg_shape(bgr.shape[:2], params.scale)[1]
    warp = None
    if params.keystone_layout:
        warp = plan_warp([q * ratio for q in read_page_quads(work.pages)])
    if warp is None:
        polys, labels = layout_blocks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), models)
    else:
        canvas = warp_for_layout(bgr, warp)
        raw, labels = layout_blocks(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), models)
        polys = unwarp_polygons(raw, warp)
        if params.debug_dir is not None:
            write_layout_debug(job, params, canvas, raw, labels)
    work.layout.write_text(json.dumps({"version": LAYOUT_VERSION, "keystone": warp is not None,
                                       "blocks": [np.asarray(p).tolist() for p in polys],
                                       "labels": labels}))  # fmt: skip
    log.info("%s: surya %.1f с, блоков %d, %s", job.key, time.perf_counter() - t0, len(polys),
             "по выпрямленному кадру" if warp is not None else "по кадру")  # fmt: skip


def write_layout_debug(
    job: FrameJob, params: RunParams, canvas: np.ndarray, polys: list[np.ndarray], labels: list[str]
) -> None:
    """Выпрямленный кадр с рамками surya и их типами (как их видит сеть) → ``debug/surya_input/<кадр>.jpg``, ширина
    1600 px.

    Args:
        job: Кадр.
        params: Параметры прогона.
        canvas: Выпрямленный кадр.
        polys: Рамки surya на нём.
        labels: Типы блоков surya, по порядку ``polys``.
    """
    from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
    from ocr_utils.scan_cropping.v2.overlay import COLOUR_SURYA, COLOUR_SURYA_LABEL, LABEL_ALPHA, block_labels

    s = 1600 / canvas.shape[1]
    small = cv2.resize(canvas, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    scaled = [np.asarray(p, np.float32) * s for p in polys]
    pts = [np.round(p).astype(np.int32).reshape(-1, 1, 2) for p in scaled]
    cv2.polylines(small, pts, True, COLOUR_SURYA, 2, cv2.LINE_AA)
    small = block_labels(small, scaled, labels)
    picture = framed(small, [f"{job.rel} — вход surya: кадр, выпрямленный по трапециям страниц"],
                     [LegendEntry("блок surya на выпрямленном кадре", COLOUR_SURYA),
                      LegendEntry("тип блока surya (Text — текст, SectionHeader — заголовок, Picture — картинка…)",
                                  COLOUR_SURYA_LABEL, alpha=LABEL_ALPHA)])  # fmt: skip
    # Отдельная папка рядом с оверлеями, структура подпапок входа повторяется.
    path = params.debug_dir / SURYA_INPUT_DIR / job.rel.with_suffix(".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])


def book_zone(original: np.ndarray, core: np.ndarray, book: np.ndarray) -> np.ndarray:
    """«Не фон» без рук на масштабе сегментации: грубая маска книги минус кожа, связная с пальцем или рамкой.

    Args:
        original: Исходный кадр BGR (с пальцами).
        core: Ядро пальца (bool) размера кадра.
        book: Грубая маска книги на масштабе сегментации.

    Returns:
        Булева маска масштаба сегментации.
    """
    h, w = book.shape
    small = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
    core_small = cv2.resize(core.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return book & ~hand_mask(small, core_small)


def copy_back_zone(original: np.ndarray, core: np.ndarray, book: np.ndarray) -> np.ndarray:
    """Где copy back разрешён: грубая маска книги без рук, в размере кадра.

    Кожа не фон, поэтому палец на странице входит в «не фон»; без вычета рук блок surya под пальцем вернул бы с
    оригинала и кусок пальца (РИСК 1991/05 IMG_0645). Руки — кожа по цвету, связная с ядром пальца или рамкой кадра
    (``background.hand_mask``), как в исходном определении книги стенда «не фон − руки».

    Args:
        original: Исходный кадр BGR (с пальцами).
        core: Ядро пальца (bool) размера кадра.
        book: Грубая маска книги на масштабе сегментации.

    Returns:
        Булева маска размера кадра.
    """
    return to_frame(book_zone(original, core, book), original.shape[:2])


def write_pages(path: Path, pages: list[PageQuad]) -> None:
    """Страницы кадра в кэш: трапеция (px сегментации), признаки и класс положения.

    Args:
        path: ``pages.json`` кадра.
        pages: Страницы слева направо.
    """
    rows = [
        {"quad": None if p.quad is None else p.quad.round(1).tolist(), "pose": p.pose.value, **asdict(p.features)}
        for p in pages
    ]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def spine_on_frame(pages: list[PageQuad], seg_to_frame: float) -> float | None:
    """Абсцисса стыка страниц, px кадра: середина между правой стороной левой трапеции и левой стороной правой.

    Args:
        pages: Страницы слева направо (трапеции в px сегментации).
        seg_to_frame: Множитель px сегментации → px кадра.

    Returns:
        Абсцисса или None (страниц с трапецией не две).
    """
    quads = [p.quad for p in pages if p.quad is not None]
    if len(quads) != 2:
        return None
    left = np.sort(quads[0][:, 0])[2:].mean()
    right = np.sort(quads[1][:, 0])[:2].mean()
    return float((left + right) / 2 * seg_to_frame)


def cpu_finish(args: tuple[FrameJob, RunParams]) -> FrameStats:
    """Финал кадра (для пула): маска бумаги, copy back блоков surya в книге, закрас до прямоугольника, запись, оверлей.

    Args:
        args: ``(кадр, параметры)``.

    Returns:
        Строка сводки.
    """
    job, params = args
    t0 = time.perf_counter()
    try:
        return finish_frame(job, params, t0)
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка финала", job.key)
        return FrameStats(job.key, "error", seconds=time.perf_counter() - t0, message=repr(exc))


def finish_frame(job: FrameJob, params: RunParams, t0: float) -> FrameStats:
    """Тело :func:`cpu_finish`.

    Args:
        job: Кадр.
        params: Параметры прогона.
        t0: Время начала, с (``perf_counter``).

    Returns:
        Строка сводки.
    """
    from ocr_utils.scan_cropping.v2.overlay import OverlayLayers, draw_finger_shadow, draw_overlay

    work = WorkPaths(params.work_dir / job.key)
    img = read_bgr(work.seg)
    book = dict(np.load(work.book))
    sam3 = dict(np.load(work.sam3))
    # Маски бумаги — с шага «бумага и страницы».
    with np.load(work.masks) as data:
        masks = PaperMasks(**{name: data[name] for name in PaperMasks.__dataclass_fields__ if name in data.files})
    clean = read_bgr(work.clean)
    shape = clean.shape[:2]
    fingers = np.load(work.fingers)
    paint = fingers["paint"]
    polys = read_layout(work.layout) or []
    # Типы блоков — только для подписей на оверлее; в кэше версии 3 их нет.
    labels = read_layout_labels(work.layout)
    # Оригинал нужен для copy back, «не фона без рук» и оверлея.
    original = read_bgr(job.src)
    zone_seg = book_zone(original, fingers["core"], book["book"])
    # Страницы: трапеции и класс «плоско / приподнята» по итоговой бумаге (дёшево, детерминированно — как в шаге 4);
    # защита контента — только блоки плоских страниц.
    pages = frame_pages(img, book, sam3, paint_on_seg(fingers, img.shape[:2]), masks)
    # Припуск блоков — один для защиты контента и copy back: на оверлее рисуется ровно возвращаемая зона.
    pad = (params.scale.mm_px(params.layout_pad_mm[0]), params.scale.mm_px(params.layout_pad_mm[1]))
    # Цветные страницы — защита контента при любой позе: рваная маска делала их «приподнятыми» (IMG_1113).
    guard_pages = colour_pages_flat(img, pages) if params.colour_pages else pages
    protected = protect_content(masks.paper, polys, guard_pages, zone_seg, img.shape[0] / shape[0], pad)
    np.savez_compressed(work.masks, **asdict(masks), protected=protected)
    # Copy back блоков surya в зоне закраса — только внутри грубой маски книги без рук.
    # Где copy back разрешён: «не фон» целиком (с пальцами и руками) — детектор пальцев ошибается, и лучше вернуть
    # с оригинала кусок пальца, чем потерять текст под ложным срабатыванием (решение пользователя 2026-10-13); или,
    # по ``--no-copy-back-over-fingers``, «не фон» без рук.
    allowed = book["book"] if params.copy_back_over_fingers else zone_seg
    filled_from = clean
    # Picture и Figure — целиком (ложные пальцы на портретах), остальные блоки — только зона дилатации пальца.
    # Ядро — с припуском: кайма кожи у края ядра SAM 3 не возвращается с оригинала (IMG_1197).
    core_grow = params.scale.mm_px(params.copy_back_core_grow_mm, minimum=0)
    clean, returned, blocks = copy_back_blocks(
        original, clean, polys, paint, to_frame(allowed, shape), pad, labels, fingers["core"], core_grow
    )
    paper = to_frame(masks.paper | protected, shape)
    box = paper_bbox(paper)
    # Стык страниц на кадре: закрас pull-push у продолжения стыка смешивает страницы с растушёвкой.
    spine = spine_on_frame(pages, shape[0] / img.shape[0])
    if params.fill_before_copy_back:
        # Закрас — от точного края бумаги по кадру без copy back; защита и copy back — поверх готового закраса.
        # Рамка вырезки — прежняя (бумага ∪ защита). Возвращённое с оригинала вне «бумага ∪ защита» в итог не попадает,
        # как и при прежнем порядке.
        out = fill_then_overlay(
            filled_from,
            clean,
            to_frame(masks.paper, shape),
            to_frame(protected, shape),
            params.fill,
            params.replicate,
            params.scale.px_per_mm,
            spine,
            params.fill_seam_mm,
        )
    else:
        out = fill_to_rectangle(
            clean, paper, params.fill, params.replicate, None, params.scale.px_per_mm, spine, params.fill_seam_mm
        )
    status = "ok"
    if out is None:
        # Бумага не найдена — кадр пишется без вырезки, как в v1.
        out, status = clean, "no_paper"
    if params.compensate_levels:
        out = compensate_levels(out, max_black=params.levels_max_black)
    shadow, before_shadow = None, out
    if params.finger_shadow and box is not None and status == "ok":
        # Тень пальца — на итоге (после закраса, copy back и уровней), в его координатах: маски — в рамке вырезки.
        # Пиксели оригинала — вне зоны закраса и возвращённое copy back: только там может быть кожа и печать.
        x0, y0, x1, y1 = box
        cut = (slice(y0, y1), slice(x0, x1))
        if out.shape[:2] == (y1 - y0, x1 - x0):
            shadow_in = dict(
                paint=paint[cut], core=fingers["core"][cut], book=paper[cut], original=(~paint | returned)[cut]
            )
            if params.save_before_shadow:
                # Точный вход карты — для стенда: итог до осветления и те же маски.
                cv2.imwrite(str(work.before_shadow), out)
                np.savez_compressed(work.shadow_masks, **shadow_in)
            shadow = estimate_finger_shadow(
                out, shadow_in["paint"], shadow_in["core"], shadow_in["book"], shadow_in["original"],
                params.scale.px_per_mm,
            )  # fmt: skip
            if shadow.gaps:
                log.info("%s: тень пальца, расхождение цвета опоры и поверхности по зонам: %s", job.key,
                         ", ".join(f"{g:.1f}" for g in shadow.gaps))  # fmt: skip
            out = lighten_finger_shadow(out, shadow)
        else:
            log.warning("%s: итог %s не совпал с рамкой вырезки — тень не осветляется", job.key, out.shape[:2])
    out_path = output_path(job, params)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(out_path, out, imwrite_params(out_path.suffix), params.force_dpi)
    if params.debug_dir is not None:
        dbg = params.debug_dir / job.rel.with_suffix(".jpg")
        dbg.parent.mkdir(parents=True, exist_ok=True)
        shadow_area = None if shadow is None else shadow_on_frame(shadow.area, box, shape)
        layers = OverlayLayers(masks, protected, fingers, returned, polys, pages, box, pad, labels, shadow_area)
        surya = "по выпрямленному кадру" if layout_keystone(work.layout) else "по кадру"
        picture = draw_overlay(original, book, layers, out,
                               f"{job.rel} — v2, закрас до прямоугольника: {params.fill.value}, surya {surya}")  # fmt: skip
        cv2.imwrite(str(dbg), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if shadow is not None:
            x0, y0, x1, y1 = box
            cut = (slice(y0, y1), slice(x0, x1))
            roi = draw_finger_shadow(before_shadow, shadow, paint[cut], fingers["core"][cut], params.scale.px_per_mm,
                                     f"{job.rel} — тень пальца: зона поиска и карта осветления")  # fmt: skip
            if roi is not None:
                cv2.imwrite(str(dbg.with_name(f"{dbg.stem}_shadow.jpg")), roi, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not params.save_fingers:
        # Кадр без пальцев нужен только шагам этого кадра; в кэше не храним (по умолчанию).
        work.clean.unlink(missing_ok=True)
    area = max(int(masks.paper.sum()), 1)
    fill_share = 0.0
    if box is not None:
        x0, y0, x1, y1 = box
        fill_share = float((~paper[y0:y1, x0:x1]).mean())
    return FrameStats(
        job.key,
        status,
        finger_px=int(paint.sum()),
        blocks_back=blocks,
        back_px=int(returned.sum()),
        glue_added=float((masks.paper & ~masks.snapped).sum() / area),
        refine_cut=float(masks.cut.sum() / area),
        ink_added=float(masks.ink_added.sum() / area),
        side_added=float(masks.side_added.sum() / area),
        side_cut=float(masks.side_cut.sum() / area),
        protected=float(protected.sum() / area),
        poses="/".join(p.pose.value for p in pages),
        fill_share=fill_share,
        width=int(out.shape[1]),
        height=int(out.shape[0]),
        seconds=time.perf_counter() - t0,
        **shadow_stats(shadow, params.scale.px_per_mm),
    )


def shadow_on_frame(area: np.ndarray, box: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    """Зона поиска тени (масштаб карты, координаты итога) → маска размера кадра для оверлея.

    Args:
        area: Зона поиска на масштабе ``finger_shadow.SHADOW_SCALE`` в координатах итога (bool).
        box: Рамка вырезки ``(x0, y0, x1, y1)``, px кадра.
        shape: Размер кадра ``(H, W)``.

    Returns:
        Булева маска размера кадра.
    """
    x0, y0, x1, y1 = box
    out = np.zeros(shape, bool)
    out[y0:y1, x0:x1] = cv2.resize(area.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return out


def shadow_stats(shadow, px_per_mm: float) -> dict[str, float]:
    """Поля сводки по тени пальца: площадь зоны поиска, мм², и средняя прибавка в ней, уровни (нули — шага не было).

    Args:
        shadow: ``finger_shadow.FingerShadow`` или None.
        px_per_mm: Плотность итога, px/мм.

    Returns:
        ``{"shadow_mm2": …, "shadow_mean": …}``.
    """
    if shadow is None or not shadow.area.any():
        return {"shadow_mm2": 0.0, "shadow_mean": 0.0}
    mm = px_per_mm * SHADOW_SCALE
    return {"shadow_mm2": float(shadow.area.sum() / mm**2), "shadow_mean": float(shadow.delta[shadow.area].mean())}


# --- Прогон по папке ----------------------------------------------------------------------------------------------


def visible_frames(files: list[Path], root: Path) -> list[Path]:
    """Кадры без скрытых папок: ``.covers``, ``.дубликаты к удалению``, ``.replaced`` и т. п. — не входные кадры.

    Args:
        files: Кадры (``image_io.collect_images``).
        root: Корень входа.

    Returns:
        Кадры, ни одна часть относительного пути которых не начинается с точки.
    """
    return [f for f in files if not any(part.startswith(".") for part in f.relative_to(root).parts)]


def frame_ready(job: FrameJob, params: RunParams) -> bool:
    """Готов ли в кэше шаг «пальцы» кадра (кадр без пальцев и маски закраса, которые читают следующие шаги).

    Args:
        job: Кадр.
        params: Параметры прогона.

    Returns:
        True — шаг можно не считать.
    """
    work = WorkPaths(params.work_dir / job.key)
    return not params.redo and work.clean.exists() and work.fingers.exists()


def write_summary_row(path: Path, row: FrameStats) -> None:
    """Дописывает строку сводки (заголовок — если файла ещё нет).

    Args:
        path: ``summary.csv`` в кэше.
        row: Строка.
    """
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(row).keys()))
        if new:
            writer.writeheader()
        writer.writerow(asdict(row))


class FrameStream:
    """Кадры в работе: фоны, страницы и финалы в пуле, SAM 3 и surya в главном процессе, сводка по мере готовности."""

    def __init__(self, pool: ProcessPoolExecutor, models, predictor, params: RunParams) -> None:
        """Пустой поток.

        Args:
            pool: Пул CPU-шагов.
            models: ``GpuModels`` с surya layout.
            predictor: SAM 3 (``paper.load_sam3``).
            params: Параметры прогона.
        """
        self.pool = pool
        self.models = models
        self.predictor = predictor
        self.params = params
        self.summary = params.work_dir / "summary.csv"
        self.books: dict[Future, FrameJob] = {}
        self.pages: dict[Future, FrameJob] = {}
        self.finishes: dict[Future, FrameJob] = {}
        self.stats: list[FrameStats] = []

    @property
    def busy(self) -> int:
        """Сколько кадров в работе."""
        return len(self.books) + len(self.pages) + len(self.finishes)

    def submit(self, job: FrameJob) -> None:
        """Отдать в пул фон и книгу кадра (после его закраса пальцев)."""
        self.books[self.pool.submit(cpu_book, (job, self.params))] = job

    def record(self, row: FrameStats) -> None:
        """Строка сводки: в список, в ``summary.csv`` и в лог."""
        self.stats.append(row)
        write_summary_row(self.summary, row)
        log.info("%s: %s, %d×%d, закрашено вне бумаги %.3f, copy back блоков %d", row.key, row.status, row.width,
                 row.height, row.fill_share, row.blocks_back)  # fmt: skip

    def drain(self, block: bool) -> None:
        """Разобрать готовое: фоны → SAM 3 (GPU) → страницы в пул; страницы → surya (GPU) → финал в пул; финалы → сводка.

        Args:
            block: Ждать хотя бы один готовый шаг (иначе — только уже готовые).
        """
        pending = list(self.books) + list(self.pages) + list(self.finishes)
        if not pending:
            return
        ready, _ = wait(pending, timeout=None if block else 0, return_when=FIRST_COMPLETED)
        for fut in ready:
            if fut in self.books:
                job = self.books.pop(fut)
                try:
                    fut.result()
                    if not sam3_ready(WorkPaths(self.params.work_dir / job.key), self.params):
                        gpu_sam3(job, self.predictor, self.params)
                except Exception as exc:
                    log.exception("%s: ошибка фона или SAM 3", job.key)
                    self.record(FrameStats(job.key, "error", message=repr(exc)))
                    continue
                self.pages[self.pool.submit(cpu_pages, (job, self.params))] = job
            elif fut in self.pages:
                job = self.pages.pop(fut)
                try:
                    fut.result()
                    if not layout_ready(WorkPaths(self.params.work_dir / job.key), self.params):
                        gpu_layout(job, self.models, self.params)
                except Exception as exc:
                    log.exception("%s: ошибка страниц или surya", job.key)
                    self.record(FrameStats(job.key, "error", message=repr(exc)))
                    continue
                self.finishes[self.pool.submit(cpu_finish, (job, self.params))] = job
            else:
                self.finishes.pop(fut)
                self.record(fut.result())


def run_batch(files: list[Path], params: RunParams) -> list[FrameStats]:
    """Прогон v2 по списку кадров потоком: кадр проходит все шаги, итог и оверлей появляются по мере готовности.

    Все модели (``GpuModels`` и SAM 3) грузятся один раз и живут весь прогон: пик видеопамяти ~7.4 ГБ (замер
    2026-10-12 на RTX 5060 Ti 16 ГБ). GPU-шаги идут в главном процессе по кадрам: пальцы кадра *i*, SAM 3 для
    кадров, у которых пул уже посчитал фон, surya для кадров, у которых готовы бумага и трапеции; CPU-шаги (фон и
    книга, бумага и страницы, финал) — в пуле, пока GPU занят следующим кадром. В работе не больше ``2 × jobs`` кадров. Готовые шаги берутся из кэша (кроме ``redo``).

    Args:
        files: Входные кадры.
        params: Параметры прогона.

    Returns:
        Строки сводки по кадрам (дописываются и в ``work_dir/summary.csv`` по мере готовности).
    """
    from ocr_utils.scan_cropping.gpu_models import GpuModels
    from ocr_utils.scan_cropping.v2.paper import load_sam3

    jobs = [FrameJob(f, f.relative_to(params.input_dir)) for f in files]
    if params.skip_if_exists:
        before = len(jobs)
        jobs = [j for j in jobs if not output_path(j, params).exists()]
        log.info("пропущено готовых: %d", before - len(jobs))
    params.work_dir.mkdir(parents=True, exist_ok=True)
    in_flight = max(2, 2 * params.jobs)
    ctx = get_context("forkserver")
    with (
        ProcessPoolExecutor(params.jobs, mp_context=ctx, initializer=init_worker) as pool,
        GpuModels(with_lama=True, with_layout=True) as models,
    ):
        stream = FrameStream(pool, models, load_sam3(), params)
        for job in jobs:
            while stream.busy >= in_flight:
                stream.drain(block=True)
            if not frame_ready(job, params):
                try:
                    gpu_fingers(job, models, params)
                except Exception as exc:
                    log.exception("%s: ошибка закраса пальцев", job.key)
                    stream.record(FrameStats(job.key, "error", message=repr(exc)))
                    continue
            stream.submit(job)
            stream.drain(block=False)
        while stream.busy:
            stream.drain(block=True)
    return sorted(stream.stats, key=lambda r: r.key)
