"""Конвейер v2 по папке кадров: пальцы (потоки, GPU под общей блокировкой) → фон и книга (пул) → SAM 3 и SAM 2 (GPU) → вето фона, бумага, страницы и подгоночные области без текста (пул) → surya по холсту Кунса этих областей с откатом на трапеции (GPU) → защита, copy back, закрас до прямоугольника (пул)."""

from __future__ import annotations

import csv
import functools
import json
import logging
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import threadpoolctl

from ocr_utils.frame_flags.flags import FlagsError, FrameFlags, read_flags
from ocr_utils.scan_cropping.image_io import imwrite_params, resolve_output_suffix, write_image
from ocr_utils.scan_cropping.levels import LEVELS_MAX_BLACK, compensate_levels
from ocr_utils.scan_cropping.v2.background import BG_SCALE, bg_flood, hand_mask
from ocr_utils.scan_cropping.v2.bg_veto import SAM2_CACHE_VERSION, veto_book
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
from ocr_utils.scan_cropping.v2.coons_layout import (
    CANVAS_MARGIN_MM,
    FramesReason,
    LayoutWarp,
    canvas_record,
    frames_on_frame,
    from_surya,
    plan_coons,
    unwarp_coons,
    warp_coons,
)
from ocr_utils.scan_cropping.v2.keystone import plan_warp, unwarp_polygons, warp_for_layout
from ocr_utils.scan_cropping.v2.overlay import OverlayTag, overlay_path
from ocr_utils.scan_cropping.v2.page_boundary import FitKind, PageBoundary, fit_boundary
from ocr_utils.scan_cropping.v2.edge_zones import merge_levels, outer_zones, union_levels_edges
from ocr_utils.scan_cropping.v2.ragged_edge import RAGGED_TORTUOSITY, is_ragged, measure_far_side, page_raw_mask
from ocr_utils.scan_cropping.v2.gutter_strip import Strip, find_strip, spine_line, strip_paper
from ocr_utils.scan_cropping.v2.page_frames import (
    PageFrame,
    frames_digest,
    load_frames,
    save_frames,
    single_frame,
    spread_frames,
)
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PageQuad, page_quads
from ocr_utils.scan_cropping.v2.single_page import FrameKind
from ocr_utils.scan_cropping.v2.paper import (
    SAM3_CACHE_COMPATIBLE,
    SAM3_CACHE_VERSION,
    SAM3_FALLBACK_MIN_SHARE,
    covered_share,
    fill_page_pockets,
    postprocess_plain,
)
from ocr_utils.scan_cropping.v2.scale import REF_PX_PER_MM, Scale
from ocr_utils.scan_cropping.cropping import CROP_FILL_BLUR_PX
from ocr_utils.scan_cropping.v2.side_fit import regularize_sides, snap_sheet_edges
from ocr_utils.scan_cropping.v2.spread_lines import (
    Line,
    SpineMethod,
    SpreadLines,
    finger_hands,
    auto_source,
    spine_candidates,
    spread_lines,
)

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
# Потоков закраса пальцев по умолчанию: GPU в шаге — ~10 % времени (YOLO-World, SAM, LaMa), остальное — CPU OpenCV и
# NumPy без GIL (возврат краски, Telea контекста, запись clean.png; профиль 2026-10-23, ПХ 1975: 21 с на кадр при
# занятом CPU, из них GPU ~2 с). Три потока перекрывают CPU одних кадров с GPU других.
DEFAULT_FINGER_THREADS = 3
# Версия формата ``layout.json``: 5 — блоки surya по холсту выбранного выпрямления (``--layout-warp``) с типами и
# сырыми блоками холста: ``{"version": 5, "mode": запрошенный LayoutWarp, "warp": применённый LayoutWarp, "reason":
# FramesReason отката, "frames": отпечаток областей surya (``page_frames.frames_digest``), "canvas": параметры холста,
# "canvas_blocks": [...], "blocks": [...], "labels": [...]}``; блоки — многоугольники в px кадра (у coons — кривые,
# десятки вершин). Версия 4 (``{"keystone": bool, ...}``) годна для режимов keystone и none; версия 3 (без типов)
# читается — блоки годны для защиты и copy back, но шаг surya пересчитывается; прежние форматы не читаются.
LAYOUT_VERSION = 5
# Версии ``layout.json``, блоки которых годны в дело.
LAYOUT_READABLE = (3, 4, 5)


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
        layout_warp: Чем выпрямлять кадр перед surya layout (``coons_layout.LayoutWarp``): ``coons`` — холст Кунса по
            подгоночным областям страниц без текста (``layout_frames.npz`` шага страниц), блоки возвращаются на кадр
            кривыми многоугольниками; не построился — откат на трапеции; ``keystone`` — гомографии по трапециям
            страниц, блоки — наклонные четырёхугольники; ``none`` — кадр как есть.
        layout_canvas_margin_mm: Поле холста Кунса вокруг прямоугольников страниц, мм бумаги.
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
        finger_threads: Потоков закраса пальцев в главном процессе (вызовы сетей — под общей блокировкой GPU); 1 —
            по кадру, как раньше.
        save_fingers: Оставлять в кэше кадр без пальцев (``clean.png``, ~18 МБ); по умолчанию он удаляется после
            финала кадра.
        finger_shadow: После copy back и растяжки уровней осветлять тень пальца по карте (``finger_shadow``): закрас
            LaMa продолжает тень пальца с бумаги у края зоны, copy back возвращает её с оригинала.
        save_before_shadow: Сохранять в кэш кадра итог до осветления тени и маски входа карты (``before_shadow.png``,
            ``shadow_masks.npz``, ~40 МБ на кадр) — для стенда ``research/finger_repaint edge``.
        debug_steps: При ``debug_dir`` рисовать ещё картинки шагов сегментации бумаги, каждую в свою подпапку
            (``overlay.draw_steps``: ``01_book`` … ``07_lines``).
        spine_method: Способ прямой-разделителя страниц на оверлеях (``spread_lines.SpineMethod``, кроме ``FOLD``);
            прямые разворота — только диагностика, итог кадра от них не зависит.
        copy_back_core_grow_mm: Припуск ядра пальца при copy back, мм бумаги (``copy_back.CORE_GROW_PX`` = 4 px
            калибровочного кадра): кайма кожи у края ядра не возвращается с оригинала; 0 — без припуска.
        colour_pages: Правила для целиком цветных страниц (``colour_pages``): рост бумаги у края по палитре страницы
            вверх и вниз, защита контента при любой позе.
        gutter_strip: Полоса-вклейка у корешка (``gutter_strip``) — в бумагу кадра: склейка страниц закрывает щель
            только до ~11.6 мм, белая часть полосы шире закрашивалась до прямоугольника.
        bg_veto: Вето фона (``bg_veto``): фон ``flood`` отдаётся книге, где SAM 3 видит страницу, а SAM 2 по точкам
            коврика — не фон (рукописный номер, логотип, линейка у края листа не уводят фон на страницу).
        sam3_levels: Растягивать уровни светлоты кадра по бумаге перед SAM 3 (``paper.stretch_paper_levels``);
            проба против рваных краёв, по умолчанию выключено. Кэш SAM 3 этим флагом не помечается: для сравнения
            прогонять в отдельный ``work_dir``.
        dual_levels: Двойная грубая сегментация (``edge_zones``; по умолчанию включена с 2026-10-24): база SAM 3 без
            растяжения, а в зонах у внешних
            кромок книги — второй проход по кадру с растянутыми уровнями; явные промахи прохода — зона на базе.
        levels_edges: При ``dual_levels`` — ещё и объединение базы с растянутым проходом у верхних и нижних краёв
            страниц (``edge_zones.union_levels_edges``: выемки края у корешка, дыры у кромки).
        refine_ragged: Тонкое уточнение рваных внешних сторон страниц разворота (``ragged_edge`` — детектор по сырой
            маске SAM 3, ``edge_refine`` — сегментация ленты вдоль стороны в полном разрешении).
        ragged_threshold: Порог извилистости сырой маски SAM 3 для детектора рваной стороны.
        output_pages: Папка выхода страниц (``rectify``): итог кадра выпрямляется (Кунс + поле по тексту), режется по
            кривой линии разреза на страницы ``<кадр>_L`` / ``<кадр>_R`` (``_LR`` — без разреза), каждая
            закрашивается до прямоугольника; None — выхода страниц нет.
        use_flags: Читать сайдкары флагов кадра ``<кадр>.flags.json`` (``frame_flags``): ``ignore-both`` — кадр не
            обрабатывается; остальные флаги — для выхода страниц.
        rectify_passes: Проходов «разбор текста → поле» выпрямления; 0 — только Кунс по подгоночным областям.
        rectify_illumination: Выравнивать освещение страниц в выходе страниц (не-цветных и не запрещённых флагом).
        gutter_decompression: Растягивать сжатые у корешка буквы разворота в выходе страниц (``gutter_decompression``:
            модель по ширине букв, страница шире со стороны корешка).
        output_pages_meta: Папка паспортов страниц (``rectify.passport``: исходный кадр, часть, флаги, dpi, цветность,
            контуры закраса пальцев) и масок закраса ``<кадр>_L.fingers.png`` — отдельно от страниц; None — не писать.
        pages_dpi: Тег разрешения страниц или None — настоящая плотность кадра (``scale.px_per_mm`` × 25.4).
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
    layout_warp: LayoutWarp = LayoutWarp.COONS
    layout_canvas_margin_mm: float = CANVAS_MARGIN_MM
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
    finger_threads: int = DEFAULT_FINGER_THREADS
    save_fingers: bool = False
    finger_shadow: bool = False
    save_before_shadow: bool = False
    debug_steps: bool = False
    spine_method: SpineMethod = SpineMethod.AUTO
    copy_back_core_grow_mm: float = CORE_GROW_MM
    colour_pages: bool = True
    bg_veto: bool = True
    gutter_strip: bool = True
    sam3_levels: bool = False
    dual_levels: bool = True
    levels_edges: bool = True
    refine_ragged: bool = False
    ragged_threshold: float = RAGGED_TORTUOSITY
    output_pages: Path | None = None
    use_flags: bool = True
    rectify_passes: int = 3
    rectify_illumination: bool = True
    gutter_decompression: bool = True
    output_pages_meta: Path | None = None
    pages_dpi: float | None = None


@dataclass(frozen=True)
class FrameJob:
    """Кадр прогона.

    Поля:
        src: Путь входного кадра.
        rel: Путь относительно ``input_dir``.
        flags: Флаги кадра из сайдкара (пусто — обычный кадр или ``--no-flags``).
    """

    src: Path
    rel: Path
    flags: FrameFlags = field(default_factory=FrameFlags)

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
    def layout_frames(self) -> Path:
        """Подгоночные области страниц без текста — вход холста Кунса для surya (npz, ``page_frames.save_frames``)."""
        return self.root / "layout_frames.npz"

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
    def dual(self) -> Path:
        """Итог двойной грубой сегментации по зонам у кромок книги (json, ``--dual-levels``)."""
        return self.root / "dual.json"

    @property
    def ragged(self) -> Path:
        """Меры рваности внешних сторон страниц по сырой маске SAM 3 и флаги детектора (json, ``--refine-ragged``)."""
        return self.root / "ragged.json"

    @property
    def refine_in(self) -> Path:
        """Вход уточнения рваных сторон: трапеции и маски страниц, номера рваных (npz); нет — уточнять нечего."""
        return self.root / "refine_in.npz"

    @property
    def refine_out(self) -> Path:
        """Выбранные края уточнённых сторон по строкам ленты (npz)."""
        return self.root / "refine_out.npz"

    @property
    def refine_report(self) -> Path:
        """Итог уточнения по сторонам и способам: выбранный способ, меры, промахи (json)."""
        return self.root / "refine.json"

    @property
    def sam2_bg(self) -> Path:
        """Фон SAM 2 по точкам коврика на масштабе сегментации (npz) — вход вето фона."""
        return self.root / "sam2_bg.npz"

    @property
    def book_veto(self) -> Path:
        """Маски шага «фон и книга», пересобранные по фону-вето (npz, те же ключи, что ``book.npz``)."""
        return self.root / "book_veto.npz"

    @property
    def masks(self) -> Path:
        """Маски бумаги по шагам на масштабе сегментации (npz)."""
        return self.root / "paper.npz"


@dataclass(frozen=True)
class FrameStats:
    """Строка сводки по кадру.

    Поля:
        key: Ключ кадра.
        status: ``ok`` / ``no_paper`` / ``error`` / ``ignored`` (флаг ``ignore-both``) / ``bad_flags`` (сайдкар не
            прошёл проверку).
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
        frame_kind: Сколько страниц в кадре (``single_page.FrameKind``: разворот, одна страница, страница с полоской
            соседней, страницы нет).
        fill_share: Доля прямоугольника вне бумаги (закрашено).
        width: Ширина итога, px.
        height: Высота итога, px.
        seconds: Время CPU-части финала, с.
        shadow_mm2: Площадь зоны поиска тени пальца, мм² (0 — шаг выключен или пальца на бумаге нет).
        shadow_mean: Средняя прибавка осветления тени в зоне поиска, уровни.
        flags: Флаги кадра из сайдкара через ``+`` или пусто.
        pages: Записанные части выхода страниц через ``+`` (``L``, ``R``, ``LR``, ``P``) или пусто.
        pages_status: Исход выхода страниц (``rectify.pages.PagesStatus``) или пусто — выхода нет.
        pages_tint: Доли тонированного фона левой и правой страницы через ``/`` (``colour_pages.tinted_share``).
        pages_finger_mm2: Площадь закраса пальцев на записанных страницах через ``/``, мм².
        pages_text: Меры прямизны текста страниц (``rectify.measure.TextMeasures``) после Кунса (``before_*``) и после
            поля (``after_*``) — JSON одной строкой, или пусто.
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
    frame_kind: str = ""
    fill_share: float = 0.0
    width: int = 0
    height: int = 0
    seconds: float = 0.0
    shadow_mm2: float = 0.0
    shadow_mean: float = 0.0
    flags: str = ""
    pages: str = ""
    pages_status: str = ""
    pages_tint: str = ""
    pages_finger_mm2: str = ""
    pages_text: str = ""
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


def sam2_ready(work: WorkPaths, params: RunParams) -> bool:
    """Готов ли в кэше фон SAM 2 нужной версии (шаг нужен только при вето фона).

    Args:
        work: Файлы кэша кадра.
        params: Параметры прогона.

    Returns:
        True — шаг можно не считать.
    """
    if not params.bg_veto:
        return True
    if params.redo or not work.sam2_bg.exists():
        return False
    with np.load(work.sam2_bg) as data:
        return "version" in data.files and int(data["version"]) == SAM2_CACHE_VERSION


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
        version = int(data["version"]) if "version" in data.files else 0
        if version in SAM3_CACHE_COMPATIBLE:
            # Прежняя версия годна, если каждый проход накрыл не меньше SAM3_FALLBACK_MIN_SHARE книги: запасные
            # подсказки зовутся только ниже этой доли.
            if not work.book.exists():
                return False
            with np.load(work.book) as book_data:
                book = book_data["book"]
            raws = [data[k] for k in data.files if k.startswith("raw")]
            if any(covered_share([(1.0, raw)], book) < SAM3_FALLBACK_MIN_SHARE for raw in raws):
                return False
        elif version != SAM3_CACHE_VERSION:
            return False
        if params.dual_levels and "raw_lv" not in data.files:
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
    from ocr_utils.scan_cropping.v2.paper import sam3_pages, stretch_paper_levels

    work = WorkPaths(params.work_dir / job.key)
    img = read_bgr(work.seg)
    book = np.load(work.book)
    hulls = book["hulls"]
    noink = read_bgr(work.seg_noink) if params.ink_fill else None
    if params.sam3_levels:
        # Растяжение уровней по бумаге: точки — по кадру, то же преобразование и для кадра без краски.
        region = hulls if hulls.any() else book["book"]
        img, black, white = stretch_paper_levels(img, region)
        if noink is not None:
            noink = stretch_paper_levels(noink, region)[0]
        log.info("%s: уровни перед SAM 3 — чёрная %.0f, белая %.0f", job.key, black, white)
    t0 = time.perf_counter()
    raw, inst = sam3_pages(predictor, img, hulls, book=book["book"])
    out = {"raw": raw, "inst": inst, "version": np.int32(SAM3_CACHE_VERSION)}
    if noink is not None:
        # Второй проход — по кадру без крупной краски.
        out["raw_noink"], out["inst_noink"] = sam3_pages(predictor, noink, hulls, book=book["book"])
    if params.dual_levels:
        # Проход по кадру с растянутыми уровнями — для зон у кромок книги (edge_zones.merge_levels в cpu_pages).
        region = hulls if hulls.any() else book["book"]
        stretched, black, white = stretch_paper_levels(read_bgr(work.seg), region)
        out["raw_lv"], out["inst_lv"] = sam3_pages(predictor, stretched, hulls, book=book["book"])
        log.info("%s: растянутый проход SAM 3 — уровни %.0f…%.0f, страниц %d", job.key, black, white,
                 int(out["inst_lv"].max()))  # fmt: skip
    np.savez_compressed(work.sam3, **out)
    # Страницы и их трапеции после нового SAM 3 могут быть другими, а surya считается по кадру, выпрямленному по ним.
    work.layout.unlink(missing_ok=True)
    log.info("%s: SAM 3 %.1f с, страниц %d", job.key, time.perf_counter() - t0, int(inst.max()))


def gpu_sam2_bg(job: FrameJob, sam2, params: RunParams) -> None:
    """Фон SAM 2 по точкам коврика одного кадра → кэш (вход вето фона).

    Args:
        job: Кадр.
        sam2: ``bg_veto.Sam2Background``.
        params: Параметры прогона.
    """
    work = WorkPaths(params.work_dir / job.key)
    img = read_bgr(work.seg)
    content = np.load(work.book)["content"]
    t0 = time.perf_counter()
    bg = sam2(img, content)
    np.savez_compressed(work.sam2_bg, bg=bg, version=np.int32(SAM2_CACHE_VERSION))
    log.info("%s: SAM 2 фон %.1f с, доля кадра %.3f", job.key, time.perf_counter() - t0, bg.mean())


def frame_book(work: WorkPaths, params: RunParams, img: np.ndarray, sam3: dict) -> dict:
    """Маски шага «фон и книга» для бумаги: по фону-вето (пересобираются и кладутся в ``book_veto.npz``) или как есть.

    Args:
        work: Файлы кэша кадра.
        params: Параметры прогона.
        img: Кадр масштаба сегментации.
        sam3: Выход SAM 3 (нужна сырая маска ``raw``).

    Returns:
        Словарь масок ``content``, ``bg``, ``book``, ``hulls`` (и ``ink``).
    """
    book = dict(np.load(work.book))
    if not params.bg_veto:
        return book
    vetoed = veto_book(img, book, sam3["raw"], np.load(work.sam2_bg)["bg"])
    np.savez_compressed(work.book_veto, **vetoed)
    log.info("%s: вето фона вернуло книге %.1f мм²", work.root.name,
             (vetoed["book"] & ~book["book"]).sum() / (REF_PX_PER_MM * BG_SCALE) ** 2)  # fmt: skip
    return vetoed


def read_book(work: WorkPaths, params: RunParams) -> dict:
    """Маски шага «фон и книга», которыми считалась бумага кадра (с вето фона — из ``book_veto.npz``).

    Args:
        work: Файлы кэша кадра.
        params: Параметры прогона.

    Returns:
        Словарь масок.
    """
    return dict(np.load(work.book_veto if params.bg_veto else work.book))


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
        strip_added: Добавлено полосой-вклейкой у корешка (``gutter_strip.strip_paper``).
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
    strip_added: np.ndarray | None = None


def masks_arrays(masks: PaperMasks) -> dict[str, np.ndarray]:
    """Поля масок бумаги для ``np.savez``: незаданные (None) пропускаются — иначе в npz ложится object-массив, который
    без ``allow_pickle`` не читается (при загрузке пропущенное поле получает умолчание None).

    Args:
        masks: Маски бумаги кадра.

    Returns:
        Имя поля → массив.
    """
    return {name: value for name, value in asdict(masks).items() if value is not None}


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


def layout_warp(path: Path) -> LayoutWarp | None:
    """Чем был выпрямлен кадр перед surya (поле ``warp`` версии 5, флаг ``keystone`` версий 3–4); None — нет файла.

    Args:
        path: ``layout.json`` кадра.

    Returns:
        Применённое выпрямление или None.
    """
    data = layout_data(path)
    if data is None:
        return None
    if "warp" in data:
        return LayoutWarp(data["warp"])
    return LayoutWarp.KEYSTONE if data.get("keystone") else LayoutWarp.NONE


def layout_ready(work: WorkPaths, params: RunParams) -> bool:
    """Готов ли в кэше шаг surya: файл с типами блоков, посчитанный тем же режимом выпрямления по тем же областям.

    Режимы ``none`` и ``keystone``: годен файл версии 4 с тем же выпрямлением или версии 5 с тем же запрошенным
    режимом. Режим ``coons``: годен файл версии 5 режима ``coons``, у которого отпечаток областей совпал с текущими
    ``layout_frames.npz`` (в том числе откат на трапеции при отпечатке ``none``): области пересчитываются шагом
    страниц при каждом прогоне, и сменившиеся области сбрасывают surya без удаления файла.

    Args:
        work: Файлы кэша кадра.
        params: Параметры прогона.

    Returns:
        True — шаг можно не считать.
    """
    data = layout_data(work.layout)
    if params.redo or data is None or data["version"] == 3:
        # Нет файла или он без типов блоков (версия 3) — шаг считается заново.
        return False
    mode = params.layout_warp
    if data["version"] == 4:
        return mode is not LayoutWarp.COONS and layout_warp(work.layout) is mode
    if data.get("mode") != mode.value:
        return False
    if mode is not LayoutWarp.COONS:
        return True
    return data.get("frames") == frames_digest(load_frames(work.layout_frames)[0])


def read_page_quads(path: Path) -> list[np.ndarray]:
    """Трапеции страниц слева направо из ``pages.json`` (px сегментации), без вырожденных."""
    if not path.exists():
        return []
    return [np.asarray(row["quad"], np.float64) for row in json.loads(path.read_text()) if row.get("quad")]


def paint_on_seg(fingers, shape: tuple[int, int]) -> np.ndarray:
    """Зона закраса пальцев на масштабе сегментации (bool)."""
    return cv2.resize(fingers["paint"].astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0


def forced_kind(flags: FrameFlags) -> FrameKind | None:
    """Вид кадра по его флагам: ``single-page`` → ``SINGLE``, ``spread`` → ``SPREAD``, иначе None (по правилу).

    Args:
        flags: Флаги кадра.

    Returns:
        Вид кадра или None.
    """
    if flags.single_page:
        return FrameKind.SINGLE
    return FrameKind.SPREAD if flags.spread else None


def frame_pages(
    img: np.ndarray, book: dict, sam3: dict, paint_seg: np.ndarray, masks: PaperMasks, flags: FrameFlags | None = None
) -> list[PageQuad]:
    """Страницы кадра по итоговой бумаге: трапеции, признаки, класс положения, сколько страниц в кадре.

    Args:
        img: Кадр масштаба сегментации, BGR.
        book: Маски шага «фон и книга».
        sam3: Кэш SAM 3 кадра (``inst`` — экземпляры-страницы).
        paint_seg: Зона закраса пальцев на масштабе сегментации.
        masks: Маски бумаги.
        flags: Флаги кадра (``single-page`` / ``spread`` перекрывают правило ``single_page``); None — без флагов
            (стенды по кэшу).

    Returns:
        Страницы слева направо.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    forced = None if flags is None else forced_kind(flags)
    return page_quads(masks.paper, book["book"], book["hulls"], sam3["inst"], gray, paint_seg, forced)


# --- Шаг 4, CPU: бумага и страницы --------------------------------------------------------------------------------


def dual_sam3(work: WorkPaths, sam3: dict, book: np.ndarray, paint_seg: np.ndarray, levels_edges: bool = False) -> dict:
    """SAM 3 кадра после двойной грубой сегментации: база, в зонах у кромок книги — растянутый проход (``edge_zones``).

    Итог зон пишется в ``dual.json`` кэша кадра и в лог.

    Args:
        work: Файлы кэша кадра.
        sam3: Выход SAM 3 (``raw``, ``inst``, ``raw_lv``, ``inst_lv`` и, может быть, второй проход без краски).
        book: Грубая маска книги (после вето фона).
        paint_seg: Зона закраса пальцев на масштабе сегментации (строки у неё не идут в прямую кромки).
        levels_edges: Ещё и объединение с растянутым проходом у верхних и нижних краёв страниц
            (``edge_zones.union_levels_edges``).

    Returns:
        Новый словарь SAM 3 с заменёнными ``raw`` и ``inst``.
    """
    zones = outer_zones(book, paint_seg)
    merged = merge_levels(sam3["raw"].astype(bool), sam3["inst"], sam3["raw_lv"].astype(bool), sam3["inst_lv"], zones)
    report = {"frame": merged.frame_status, "sides": {str(k): str(v) for k, v in merged.status.items()},
              "zone_mm2": {str(k): round(float(z.sum()) / SEG_PX_PER_MM**2) for k, z in merged.zones.items()},
              "changed_mm2": round(float((merged.raw ^ sam3["raw"].astype(bool)).sum()) / SEG_PX_PER_MM**2)}  # fmt: skip
    out = {**sam3, "raw": merged.raw, "inst": merged.inst}
    if levels_edges:
        # У верхних и нижних краёв страниц — объединение с растянутым проходом (выемки края у корешка).
        union = union_levels_edges(out["raw"], out["inst"], sam3["raw_lv"].astype(bool), sam3["inst_lv"], book,
                                   sam3.get("raw_noink"))  # fmt: skip
        report["edges_added_mm2"] = {"top": round(float(union.added_top.sum()) / SEG_PX_PER_MM**2, 1),
                                     "bottom": round(float(union.added_bottom.sum()) / SEG_PX_PER_MM**2, 1)}  # fmt: skip
        out = {**out, "raw": union.raw, "inst": union.inst}
    work.dual.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    log.info("%s: двойная сегментация — %s", work.root.name, merged.frame_status or report["sides"])
    return out


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
    sam3 = dict(np.load(work.sam3))
    book = frame_book(work, params, img, sam3)
    if not params.ink_fill:
        sam3.pop("raw_noink", None)
    paint_seg = paint_on_seg(np.load(work.fingers), img.shape[:2])
    if params.dual_levels and "raw_lv" in sam3:
        sam3 = dual_sam3(work, sam3, book["book"], paint_seg, params.levels_edges)
    masks = frame_paper(img, book, sam3, params.scale.seg_px(params.refine_depth_mm))
    pages = frame_pages(img, book, sam3, paint_seg, masks, job.flags)
    if params.gutter_strip:
        # Полоса-вклейка у корешка — в бумагу: склейка закрывает щель только до ~11.6 мм (ПХ 1975/05 IMG_0021).
        strip_added = strip_paper(img, sam3["inst"], book["book"], pages, masks.paper)
        if strip_added.any():
            masks = replace(masks, paper=masks.paper | strip_added, strip_added=strip_added)
            pages = frame_pages(img, book, sam3, paint_seg, masks, job.flags)
    if params.fit_sides:
        # Поле страницы до кромки листа, где SAM 3 взял его рваными языками (цветная плашка с белым полем, МТС 1990/04
        # IMG_1113); страницы пересчитываются — рваная маска делала страницу «приподнятой».
        edge_added = snap_sheet_edges(masks.paper, pages, book["bg"], book["book"], img)
        if edge_added.any():
            masks = replace(masks, paper=masks.paper | edge_added)
            pages = frame_pages(img, book, sam3, paint_seg, masks, job.flags)
        # Цветные страницы: поле и край — по палитре уверенного центра страницы, вверх и вниз (верхнее поле МТС 1990/04
        # IMG_1113 — тень кромки нашлась, но край маски не «рваный»); страницы пересчитываются.
        if params.colour_pages:
            palette_added = colour_pages_growth(img, pages, masks.paper, book["book"], book["bg"])
            if palette_added.any():
                masks = replace(masks, paper=masks.paper | palette_added, palette_added=palette_added)
                pages = frame_pages(img, book, sam3, paint_seg, masks, job.flags)
        # Выравнивание внешних боковых сторон плоских страниц: выбросы SAM 3 (заливка LaMa на торце, IMG_0240),
        # ямы, колебания края в тени у кромки (IMG_0538); затем страницы пересчитываются по выровненной бумаге.
        side = regularize_sides(masks.paper, pages, book["bg"], img, params.snap_sides)
        masks = replace(masks, paper=side.paper, side_added=side.added | edge_added, side_cut=side.cut)
        pages = frame_pages(img, book, sam3, paint_seg, masks, job.flags)
    if params.refine_ragged:
        mark_ragged(work, sam3["raw"], pages, paint_seg, params.ragged_threshold)
    save_pages(work, masks, pages)
    save_layout_frames(work, params, img, book, masks, paint_seg, job.flags)
    return job.key


def save_pages(work: WorkPaths, masks: PaperMasks, pages: list[PageQuad]) -> None:
    """Бумага по шагам и страницы кадра → кэш; разметка surya сбрасывается, если трапеции изменились.

    Args:
        work: Файлы кэша кадра.
        masks: Маски бумаги.
        pages: Страницы кадра.
    """
    np.savez_compressed(work.masks, **masks_arrays(masks))
    # Трапеции изменились (новая бумага) — разметка surya по кадру, выпрямленному по старым трапециям, устарела.
    if work.pages.exists() and not same_quads(
        read_page_quads(work.pages), [p.quad for p in pages if p.quad is not None]
    ):
        work.layout.unlink(missing_ok=True)
    write_pages(work.pages, pages)


def mark_ragged(
    work: WorkPaths, raw: np.ndarray, pages: list[PageQuad], paint_seg: np.ndarray, threshold: float
) -> list[int]:
    """Детектор рваных внешних сторон разворота по сырой маске SAM 3 → ``ragged.json`` и вход уточнения.

    Мера (``ragged_edge.measure_far_side``) — по сырой маске SAM 3 страницы (``page_raw_mask``): дальше по конвейеру
    рваный край сглаживается, оставаясь не на месте. Только развороты из двух страниц с трапециями.

    Args:
        work: Файлы кэша кадра.
        raw: Сырое объединение экземпляров SAM 3 (после двойной сегментации, если она была).
        pages: Страницы кадра.
        paint_seg: Зона закраса пальцев (точки там не меряются).
        threshold: Порог извилистости.

    Returns:
        Номера рваных страниц (0 — левая, 1 — правая).
    """
    for path in (work.refine_in, work.refine_out, work.refine_report):
        path.unlink(missing_ok=True)
    two = [p for p in pages if p.quad is not None]
    report: dict = {"sides": []}
    flagged = []
    if len(two) == 2:
        for i, name in ((0, "L"), (1, "R")):
            mask = page_raw_mask(raw, two[i].mask, two[1 - i].mask, SEG_PX_PER_MM)
            m = measure_far_side(mask, two[i].quad, i == 0, paint_seg, SEG_PX_PER_MM)
            ragged = is_ragged(m, threshold)
            report["sides"].append({"page": name, "ragged": ragged, **({k: round(float(v), 4)
                                    for k, v in asdict(m).items()} if m else {})})  # fmt: skip
            if ragged:
                flagged.append(i)
    work.ragged.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    if flagged:
        np.savez_compressed(work.refine_in, flagged=np.array(flagged), quad0=two[0].quad, quad1=two[1].quad,
                            mask0=two[0].mask, mask1=two[1].mask)  # fmt: skip
    return flagged


# --- Шаг 4б, GPU и CPU: тонкое уточнение рваных сторон -------------------------------------------------------------


def gpu_refine(job: FrameJob, refiner, params: RunParams) -> None:
    """Уточнение рваных сторон кадра (``edge_refine.refine_side``) → ``refine_out.npz`` и ``refine.json``.

    Args:
        job: Кадр.
        refiner: ``edge_refine.Refiner`` (SAM 3 и SAM 2.1 прогона).
        params: Параметры прогона.
    """
    from ocr_utils.scan_cropping.v2.edge_refine import refine_side

    work = WorkPaths(params.work_dir / job.key)
    data = dict(np.load(work.refine_in))
    clean = read_bgr(work.clean)
    seg_shape_hw = data["mask0"].shape
    hulls = np.load(work.book)["hulls"]
    core = cv2.resize(np.load(work.fingers)["core"].astype(np.uint8), seg_shape_hw[::-1],
                      interpolation=cv2.INTER_NEAREST) > 0  # fmt: skip
    out, report = {}, {"sides": []}
    t0 = time.perf_counter()
    for i in data["flagged"].tolist():
        res = refine_side(refiner, clean, data[f"quad{i}"], i == 0, data[f"mask{i}"], hulls, core, SEG_PX_PER_MM)
        side = {"page": "LR"[i], "chosen": res.chosen, "full_set": res.full_set, "status": res.status,
                "finger_share": round(res.finger_share, 3),
                "methods": {n: {"cover": round(r.cover, 3), "rough_mm": round(r.rough_mm, 3), "step_mm": round(r.step_mm, 2),
                                "ink_out_mm2": round(r.ink_out_mm2, 2), "missed_chunks": r.missed_chunks,
                                "miss": r.miss, "accepted": r.accepted} for n, r in res.results.items()}}  # fmt: skip
        report["sides"].append(side)
        if res.edge is not None:
            out[f"edge{i}"] = res.edge
        log.info("%s: уточнение %s — %s", job.key, "LR"[i], res.chosen or res.status)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    # Полное разрешение / масштаб сегментации — для ленты при замене бумаги (без чтения clean.png в пуле).
    np.savez_compressed(work.refine_out, ratio=np.float64(clean.shape[1] / seg_shape_hw[1]), **out)
    work.refine_report.write_text(json.dumps(report, ensure_ascii=False, indent=1))


def cpu_refine(args: tuple[FrameJob, RunParams]) -> str:
    """Замена бумаги в лентах уточнённых сторон и пересчёт страниц (для пула) → кэш.

    Args:
        args: ``(кадр, параметры)``.

    Returns:
        Ключ кадра.
    """
    from ocr_utils.scan_cropping.v2.edge_refine import Band, apply_edge

    job, params = args
    work = WorkPaths(params.work_dir / job.key)
    edges = dict(np.load(work.refine_out))
    ratio = float(edges.pop("ratio"))
    if not edges:
        return job.key
    data = dict(np.load(work.refine_in))
    img = read_bgr(work.seg)
    sam3 = dict(np.load(work.sam3))
    book = frame_book(work, params, img, sam3)
    paint_seg = paint_on_seg(np.load(work.fingers), img.shape[:2])
    if params.dual_levels and "raw_lv" in sam3:
        sam3 = dual_sam3(work, sam3, book["book"], paint_seg, params.levels_edges)
    with np.load(work.masks) as d:
        masks = PaperMasks(**{n: d[n] if n in d.files else None for n in PaperMasks.__dataclass_fields__})
    paper = masks.paper
    for i in data["flagged"].tolist():
        if f"edge{i}" in edges:
            paper = apply_edge(paper, Band(data[f"quad{i}"], i == 0, SEG_PX_PER_MM, ratio), edges[f"edge{i}"])
    masks = replace(masks, paper=paper)
    pages = frame_pages(img, book, sam3, paint_seg, masks, job.flags)
    save_pages(work, masks, pages)
    save_layout_frames(work, params, img, book, masks, paint_seg, job.flags)
    return job.key


def surya_frames(
    img: np.ndarray,
    book: dict,
    inst: np.ndarray,
    pages: list[PageQuad],
    paint_seg: np.ndarray,
    core_seg: np.ndarray,
    method: SpineMethod,
) -> tuple[list[PageFrame] | None, FramesReason]:
    """Подгоночные области страниц без текста — вход холста Кунса для surya (двух у разворота, одной у кадра с одной страницей).

    Те же области, что строит финал кадра (:func:`spread_fit`), но без вершин блоков surya: текст только сдвигает
    сторону наружу, а кривизну и положение задают маска SAM 3, трапеция, линия корешка, полоса-вклейка и зоны пальца.
    Руки для прямых разворота не нужны: линия корешка от них не зависит (``spine_candidates`` смотрит на страницы),
    поэтому кадр с NTFS не читается.

    Args:
        img: Кадр масштаба сегментации, BGR.
        book: Маски шага «фон и книга» (``book`` — грубая маска книги).
        inst: Экземпляры-страницы SAM 3 (из ``sam3.npz``, как в финале).
        pages: Страницы кадра слева направо.
        paint_seg: Зона закраса пальцев на масштабе сегментации.
        core_seg: Ядро пальца на масштабе сегментации (bool).
        method: Способ линии-разделителя страниц (``--spine-method``).

    Returns:
        ``(области или None, причина)``: областей не построить (нет страниц с трапецией, разворот без двух) —
        ``NOT_TWO_PAGES``, подгонка упала — ``FIT_FAILED``.
    """
    single = frame_kind_of(pages)[0] in (FrameKind.SINGLE, FrameKind.SLIVER)
    lines = None if single else frame_spread_lines(pages, book["book"], book["book"], paint_seg, method)
    if lines is None and not single:
        return None, FramesReason.NOT_TWO_PAGES
    try:
        if single:
            # Одна страница: та же область, что у выхода страниц (``single_fit``), без вершин текста.
            fit = single_fit(pages, [], paint_seg, core_seg, 1.0)
        else:
            fit = spread_fit(img, inst, book["book"], pages, lines, [], paint_seg, core_seg, 1.0)
    except Exception:  # noqa: BLE001 — подгонка не роняет кадр: surya откатывается на трапеции
        log.exception("подгоночные области для surya не построились")
        return None, FramesReason.FIT_FAILED
    if fit is None:
        return None, FramesReason.NOT_TWO_PAGES
    return fit.frames, FramesReason.OK


def save_layout_frames(
    work: WorkPaths,
    params: RunParams,
    img: np.ndarray,
    book: dict,
    masks: PaperMasks,
    paint_seg: np.ndarray,
    flags: FrameFlags,
) -> None:
    """Области страниц без текста → ``layout_frames.npz`` (только при ``--layout-warp coons``).

    Страницы считаются так же, как в финале кадра (по итоговой бумаге и экземплярам SAM 3 из файла кэша), чтобы
    области surya отличались от итоговых только сдвигом текстом.

    Args:
        work: Файлы кэша кадра.
        params: Параметры прогона.
        img: Кадр масштаба сегментации, BGR.
        book: Маски шага «фон и книга», которыми считалась бумага.
        masks: Итоговые маски бумаги кадра.
        paint_seg: Зона закраса пальцев на масштабе сегментации.
        flags: Флаги кадра (вид кадра — ``frame_pages``).
    """
    if params.layout_warp is not LayoutWarp.COONS:
        return
    sam3 = dict(np.load(work.sam3))
    pages = frame_pages(img, book, sam3, paint_seg, masks, flags)
    core = np.load(work.fingers)["core"]
    core_seg = cv2.resize(core.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    frames, reason = surya_frames(img, book, sam3["inst"], pages, paint_seg, core_seg, params.spine_method)
    if reason is not FramesReason.OK:
        log.info("%s: области для surya — нет (%s)", work.root.name, reason.value)
    save_frames(work.layout_frames, frames, reason.value)


def same_quads(old: list[np.ndarray], new: list[np.ndarray], tol: float = 2.0) -> bool:
    """Совпадают ли трапеции страниц (те же по числу, вершины не дальше ``tol`` px сегментации)."""
    return len(old) == len(new) and all(np.abs(np.asarray(a) - np.asarray(b)).max() <= tol for a, b in zip(old, new))


# --- Шаг 5, GPU: surya layout по выпрямленному кадру ---------------------------------------------------------------


def gpu_layout(job: FrameJob, models, params: RunParams) -> None:
    """Блоки surya одного кадра → кэш: по холсту выбранного выпрямления с обратным переносом блоков на кадр.

    Кадр — исходный (с пальцами), как в v1: блок, задетый пальцем, потом вернётся с оригинала. Режим ``coons``: холст
    Кунса по подгоночным областям из ``layout_frames.npz`` (блоки — кривые многоугольники); областей нет или
    поверхность вырождена — откат на трапеции, нет и трапеций — кадр как есть. Режим ``keystone`` — трапеции с
    откатом на кадр, ``none`` — кадр как есть. Файл пишется через временный и ``os.replace``: запись не идёт по
    жёсткой ссылке в чужой кэш (копия ``cp -al``).

    Args:
        job: Кадр.
        models: ``GpuModels`` с surya layout.
        params: Параметры прогона.
    """
    from ocr_utils.scan_cropping.finger_removal.text_protection import layout_blocks

    work = WorkPaths(params.work_dir / job.key)
    bgr = read_bgr(job.src)
    t0 = time.perf_counter()
    # Области и трапеции — в px сегментации; размер сегментации тот же, что у шага «фон и книга» (``seg_shape``).
    seg_hw = seg_shape(bgr.shape[:2], params.scale)
    seg_scale = (seg_hw[1] / bgr.shape[1], seg_hw[0] / bgr.shape[0])
    mode = params.layout_warp
    reason, digest, frames = FramesReason.OK, "", None
    coons = keystone = None
    if mode is LayoutWarp.COONS:
        frames, stored = load_frames(work.layout_frames)
        digest = frames_digest(frames)
        coons, reason = plan_coons(frames, seg_scale, bgr.shape[:2], params.scale.px_per_mm,
                                   params.layout_canvas_margin_mm) if frames else (None, FramesReason(stored))  # fmt: skip
    if coons is None and mode is not LayoutWarp.NONE:
        ratio = bgr.shape[1] / seg_hw[1]
        keystone = plan_warp([q * ratio for q in read_page_quads(work.pages)])
    record: dict = {"version": LAYOUT_VERSION, "mode": mode.value, "reason": reason.value, "frames": digest}
    if coons is not None:
        canvas = warp_coons(bgr, coons)
        raw, labels = layout_blocks(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), models)
        canvas_blocks = from_surya(raw, coons)
        polys = unwarp_coons(canvas_blocks, coons, params.scale.px_per_mm)
        warp = LayoutWarp.COONS
        record.update(canvas=canvas_record(coons), canvas_blocks=[np.round(p, 2).tolist() for p in canvas_blocks])
    elif keystone is not None:
        canvas = warp_for_layout(bgr, keystone)
        raw, labels = layout_blocks(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), models)
        polys = unwarp_polygons(raw, keystone)
        warp = LayoutWarp.KEYSTONE
        record.update(canvas={"width": keystone.size[0], "height": keystone.size[1], "spine_x": keystone.spine_x},
                      canvas_blocks=[np.round(np.asarray(p, np.float64), 2).tolist() for p in raw])  # fmt: skip
    else:
        canvas, raw = None, []
        polys, labels = layout_blocks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), models)
        warp = LayoutWarp.NONE
    record.update(warp=warp.value, blocks=[np.round(np.asarray(p, np.float64), 2).tolist() for p in polys],
                  labels=labels)  # fmt: skip
    seconds = time.perf_counter() - t0
    tmp = work.layout.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record))
    os.replace(tmp, work.layout)
    fallback = f", откат: {reason.value}" if mode is LayoutWarp.COONS and warp is not LayoutWarp.COONS else ""
    log.info("%s: surya %.1f с, блоков %d, %s%s", job.key, seconds, len(polys), warp_caption(warp), fallback)
    if params.debug_dir is not None and canvas is not None:
        write_layout_debug(job, params, canvas, raw, labels, bgr, polys, warp, reason,
                           frames_on_frame(frames, seg_scale), coons, keystone)  # fmt: skip


def warp_caption(warp: LayoutWarp | None) -> str:
    """Подпись выпрямления входа surya для логов и оверлеев.

    Args:
        warp: Применённое выпрямление или None (нет файла).

    Returns:
        Строка вида «по Кунсу областей».
    """
    return {
        LayoutWarp.COONS: "по холсту Кунса подгоночных областей",
        LayoutWarp.KEYSTONE: "по кадру, выпрямленному по трапециям",
        LayoutWarp.NONE: "по кадру",
    }.get(warp, "нет")


def write_layout_debug(
    job: FrameJob,
    params: RunParams,
    canvas: np.ndarray,
    raw: list[np.ndarray],
    labels: list[str],
    frame: np.ndarray,
    polys: list[np.ndarray],
    warp: LayoutWarp,
    reason: FramesReason,
    outlines: list[np.ndarray],
    coons,
    keystone,
) -> None:
    """Лист «вход surya» → ``debug/surya/<кадр>_surya.jpg``: холст с блоками сети | кадр с перенесёнными блоками.

    Args:
        job: Кадр.
        params: Параметры прогона.
        canvas: Холст, который видела сеть (малый холст Кунса или холст трапеций).
        raw: Блоки surya на холсте, px ``canvas``.
        labels: Типы блоков по порядку ``raw``.
        frame: Исходный кадр BGR.
        polys: Блоки, перенесённые на кадр, px кадра.
        warp: Применённое выпрямление.
        reason: Причина отката на трапеции (``FramesReason.OK`` — отката нет).
        outlines: Контуры подгоночных областей surya в px кадра (пусто — областей нет).
        coons: ``CoonsWarp`` или None.
        keystone: ``KeystoneWarp`` или None.
    """
    from ocr_utils.scan_cropping.v2.overlay import SuryaInput, draw_surya_input

    page_rects = []
    spine_x = None
    if coons is not None:
        # Прямоугольники страниц и линия корешка — в px малого холста.
        s = coons.scale
        page_rects = [
            ((r.x0 + 0.5) * s, (r.y0 + 0.5) * s, (r.x1 + 0.5) * s, (r.y1 + 0.5) * s) for r in coons.layout.rects
        ]
        spine_x = None if coons.layout.single else (coons.layout.spine_x + 0.5) * s
    elif keystone is not None:
        spine_x = keystone.spine_x
    if warp is LayoutWarp.COONS:
        header = (f"{job.rel} — вход surya: холст Кунса по подгоночным областям страниц (поле "
                  f"{params.layout_canvas_margin_mm:g} мм, масштаб входа сети {coons.scale:.3f})")  # fmt: skip
    else:
        fallback = f"откат: трапеции ({reason.value}); " if params.layout_warp is LayoutWarp.COONS else ""
        header = f"{job.rel} — вход surya: {fallback}кадр, выпрямленный по трапециям страниц"
    inputs = SuryaInput(canvas, raw, labels, frame, polys, outlines, page_rects, spine_x)
    picture = draw_surya_input(inputs, header)
    # Своя подпапка типа картинки, структура подпапок входа повторяется.
    path = overlay_path(params.debug_dir, job.rel, OverlayTag.SURYA)
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
        {"quad": None if p.quad is None else p.quad.round(1).tolist(), "pose": p.pose.value, **asdict(p.features),
         "frame_kind": p.frame_kind.value, "spread_index": p.spread_index}
        for p in pages
    ]  # fmt: skip
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def frame_kind_of(pages: list[PageQuad]) -> tuple[FrameKind, int | None]:
    """Вид кадра и место страницы кадра ``SLIVER`` в развороте по его страницам.

    Args:
        pages: Страницы кадра (у всех один ``frame_kind``).

    Returns:
        ``(вид, место широкой страницы)``; страниц нет — ``NO_PAGE``.
    """
    if not pages:
        return FrameKind.NO_PAGE, None
    return pages[0].frame_kind, pages[0].spread_index


def read_frame_kind(path: Path) -> tuple[FrameKind, int | None]:
    """Вид кадра из ``pages.json`` (кэш прежней версии без поля — разворот).

    Args:
        path: ``pages.json`` кадра.

    Returns:
        ``(вид, место широкой страницы)``; файла нет или страниц нет — ``NO_PAGE``.
    """
    if not path.exists():
        return FrameKind.NO_PAGE, None
    rows = json.loads(path.read_text())
    if not rows:
        return FrameKind.NO_PAGE, None
    return FrameKind(rows[0].get("frame_kind", FrameKind.SPREAD.value)), rows[0].get("spread_index")


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
    from ocr_utils.scan_cropping.v2.overlay import (
        OverlayLayers,
        StepInputs,
        draw_finger_shadow,
        draw_overlay,
        draw_steps,
    )

    work = WorkPaths(params.work_dir / job.key)
    img = read_bgr(work.seg)
    book = read_book(work, params)
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
    # Страницы: трапеции и класс «плоско / приподнята / опущена» по итоговой бумаге (дёшево, детерминированно — как в шаге 4);
    # защита контента — только блоки плоских страниц.
    pages = frame_pages(img, book, sam3, paint_on_seg(fingers, img.shape[:2]), masks, job.flags)
    # Припуск блоков — один для защиты контента и copy back: на оверлее рисуется ровно возвращаемая зона.
    pad = (params.scale.mm_px(params.layout_pad_mm[0]), params.scale.mm_px(params.layout_pad_mm[1]))
    # Цветные страницы — защита контента при любой позе: рваная маска делала их «приподнятыми» (IMG_1113).
    guard_pages = colour_pages_flat(img, pages) if params.colour_pages else pages
    protected = protect_content(masks.paper, polys, guard_pages, zone_seg, img.shape[0] / shape[0], pad)
    np.savez_compressed(work.masks, **masks_arrays(masks), protected=protected)
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
    # Прямые разворота и подгоночные области страниц — для оверлеев и выхода страниц; считаются один раз.
    want_fit = params.output_pages is not None or (params.debug_dir is not None and params.debug_steps)
    lines, fit = None, None
    if params.debug_dir is not None or want_fit:
        lines = frame_spread_lines(
            pages, book["book"], zone_seg, paint_on_seg(fingers, img.shape[:2]), params.spine_method
        )
    single = frame_kind_of(pages)[0] in (FrameKind.SINGLE, FrameKind.SLIVER)
    if want_fit and (lines is not None or single):
        try:
            core_seg = cv2.resize(fingers["core"].astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            paint_seg = paint_on_seg(fingers, img.shape[:2])
            if single:
                fit = single_fit(pages, polys, paint_seg, core_seg > 0, img.shape[0] / shape[0])
            else:
                fit = spread_fit(img, sam3["inst"], book["book"], pages, lines, polys, paint_seg, core_seg > 0,
                                 img.shape[0] / shape[0])  # fmt: skip
        except Exception:  # noqa: BLE001 — подгонка не роняет кадр: без неё нет только страниц и листа 07
            log.exception("%s: подгоночные области страниц не построились", job.key)
    pages_out = PagesOutcome()
    if params.output_pages is not None:
        seg_scale = (img.shape[1] / shape[1], img.shape[0] / shape[0])
        # Закрас пальцев, который остался в итоге (без возвращённого copy back), — в координатах итога.
        finger_out = None
        if box is not None and status == "ok":
            x0, y0, x1, y1 = box
            finger_out = (paint & ~returned)[y0:y1, x0:x1]
        pages_out = write_rectified_pages(job, params, out, status, box, fit, seg_scale, polys, labels, finger_out,
                                          frame_kind_of(pages))  # fmt: skip
    if params.debug_dir is not None:
        dbg = overlay_path(params.debug_dir, job.rel, OverlayTag.MAIN)
        dbg.parent.mkdir(parents=True, exist_ok=True)
        shadow_area = None if shadow is None else shadow_on_frame(shadow.area, box, shape)
        layers = OverlayLayers(masks, protected, fingers, returned, polys, pages, box, pad, labels, shadow_area, lines)
        surya = warp_caption(layout_warp(work.layout))
        picture = draw_overlay(original, book, layers, out,
                               f"{job.rel} — v2, закрас до прямоугольника: {params.fill.value}, surya {surya}")  # fmt: skip
        cv2.imwrite(str(dbg), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if shadow is not None:
            x0, y0, x1, y1 = box
            cut = (slice(y0, y1), slice(x0, x1))
            roi = draw_finger_shadow(before_shadow, shadow, paint[cut], fingers["core"][cut], params.scale.px_per_mm,
                                     f"{job.rel} — тень пальца: зона поиска и карта осветления")  # fmt: skip
            if roi is not None:
                shadow_path = overlay_path(params.debug_dir, job.rel, OverlayTag.SHADOW)
                shadow_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(shadow_path), roi, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if params.debug_steps:
            # Шаги сегментации бумаги — по маскам кэша; маски книги до вето — из ``book.npz``.
            frames0 = load_frames(work.layout_frames)[0] if params.layout_warp is LayoutWarp.COONS else None
            steps = StepInputs(img, dict(np.load(work.book)), book, sam3, masks, protected, pages, lines, fit, frames0)
            for tag, step in draw_steps(steps, str(job.rel)).items():
                step_path = overlay_path(params.debug_dir, job.rel, tag)
                step_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(step_path), step, [cv2.IMWRITE_JPEG_QUALITY, 88])
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
        frame_kind=frame_kind_of(pages)[0].value,
        fill_share=fill_share,
        width=int(out.shape[1]),
        height=int(out.shape[0]),
        seconds=time.perf_counter() - t0,
        **shadow_stats(shadow, params.scale.px_per_mm),
        flags=job.flags.label(),
        pages=pages_out.label,
        pages_status=pages_out.status,
        pages_tint=pages_out.tint,
        pages_finger_mm2=pages_out.finger_mm2,
        pages_text=pages_out.text,
    )


# Припуск к ядру пальца, где точки верха и низа сплайна с узлами на изгибах негодны, мм (``research/page_cutting``
# ``bends.CORE_MARGIN_MM``): шов закраса LaMa у ядра; дальше в зоне закраса край листа настоящий (ХиП 1991/01 IMG_1052).
BEND_CORE_MARGIN_MM = 3.0


@dataclass(frozen=True)
class SpreadFit:
    """Подгонка разворота: оверлей шага ``07_lines`` и выход страниц (основной итог кадра от неё не зависит).

    Поля:
        old: Прежние рамки страниц сплайном ``page_boundary`` (для сравнения), по странице слева направо.
        frames: Подгоночные области страниц (``page_frames.spread_frames``; у одной страницы — одна,
            ``page_frames.single_frame``).
        strip: Полоса-вклейка или None.
        spine: Линия корешка или None (одна страница).
        spine_source: Откуда линия корешка (подпись).
        gap_mm: Медианная щель между экземплярами SAM 3 у корешка, мм (признак полосы).
    """

    old: list[PageBoundary]
    frames: list[PageFrame]
    strip: Strip | None
    spine: Line | None
    spine_source: str
    gap_mm: float


def spread_fit(
    img: np.ndarray,
    inst: np.ndarray,
    book: np.ndarray,
    pages: list[PageQuad],
    lines: SpreadLines,
    polys: list[np.ndarray],
    paint: np.ndarray,
    core: np.ndarray,
    seg: float,
) -> SpreadFit | None:
    """Полоса-вклейка, линия корешка и подгоночные области двух страниц разворота.

    Args:
        img: Кадр масштаба сегментации, BGR.
        inst: Экземпляры-страницы SAM 3.
        book: Грубая маска книги.
        pages: Страницы кадра слева направо.
        lines: Прямые разворота v2 (разделитель ``auto``).
        polys: Блоки surya на кадре (px кадра).
        paint: Зона закраса пальцев на масштабе сегментации.
        core: Ядро пальца на масштабе сегментации.
        seg: Масштаб сегментации относительно кадра.

    Returns:
        Подгонка или None (страниц с трапецией не две).
    """
    two = [p for p in pages if p.quad is not None]
    if len(two) != 2:
        return None
    strip, sam3 = find_strip(img, inst, book, lines.spine)
    spine, source = spine_line(strip, lines.spine, lines.method.value)
    # Вершины блоков surya на масштабе сегментации — к странице, в маске которой центр блока.
    blocks = [np.asarray(poly, np.float64).reshape(-1, 2) * seg for poly in polys]
    texts = []
    for page in two:
        h, w = page.mask.shape
        own = [b for b in blocks if page.mask[min(int(b[:, 1].mean()), h - 1), min(int(b[:, 0].mean()), w - 1)]]
        texts.append(np.vstack(own) if own else np.zeros((0, 2)))
    r = max(int(round(BEND_CORE_MARGIN_MM * SEG_PX_PER_MM)), 1)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core_zone = cv2.dilate(core.astype(np.uint8), disk) > 0
    old = [fit_boundary(p.mask, p.quad, right, FitKind.SPLINE, SEG_PX_PER_MM, paint, text)
           for p, right, text in ((two[0], True, texts[0]), (two[1], False, texts[1]))]  # fmt: skip
    frames = spread_frames(two, spine, strip, paint, core_zone, texts)
    return SpreadFit(old, frames, strip, spine, source, float(sam3.extra.get("gap_mm", 0.0)))


def single_fit(
    pages: list[PageQuad], polys: list[np.ndarray], paint: np.ndarray, core: np.ndarray, seg: float
) -> SpreadFit | None:
    """Подгоночная область одной страницы кадра (``SINGLE`` или ``SLIVER``): без корешка и полосы-вклейки.

    Args:
        pages: Страницы кадра (одна).
        polys: Блоки surya на кадре (px кадра).
        paint: Зона закраса пальцев на масштабе сегментации.
        core: Ядро пальца на масштабе сегментации.
        seg: Масштаб сегментации относительно кадра.

    Returns:
        Подгонка с одной областью или None (страница без трапеции).
    """
    one = [p for p in pages if p.quad is not None]
    if len(one) != 1:
        return None
    page = one[0]
    h, w = page.mask.shape
    blocks = [np.asarray(poly, np.float64).reshape(-1, 2) * seg for poly in polys]
    own = [b for b in blocks if page.mask[min(int(b[:, 1].mean()), h - 1), min(int(b[:, 0].mean()), w - 1)]]
    text = np.vstack(own) if own else np.zeros((0, 2))
    r = max(int(round(BEND_CORE_MARGIN_MM * SEG_PX_PER_MM)), 1)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core_zone = cv2.dilate(core.astype(np.uint8), disk) > 0
    frame = single_frame(page, paint, core_zone, text)
    return SpreadFit([], [frame], None, None, "одна страница", 0.0)


def frame_spread_lines(
    pages: list[PageQuad], book: np.ndarray, zone: np.ndarray, paint: np.ndarray, method: SpineMethod
) -> SpreadLines | None:
    """Прямые разворота кадра для оверлеев: разделитель способом ``method`` и ограничители внешних краёв страниц.

    Args:
        pages: Страницы кадра слева направо.
        book: Грубая маска книги масштаба сегментации.
        zone: Книга без рук (``book_zone``) того же масштаба: кожа — ``book & ~zone``, руки — её компоненты у пальцев.
        paint: Зона закраса пальцев на масштабе сегментации.
        method: Способ разделителя; не построился — через середины трапеций (``QUAD_MID``).

    Returns:
        Прямые или None (страниц с трапецией не две).
    """
    two = [p for p in pages if p.quad is not None]
    if len(two) != 2:
        return None
    candidates = spine_candidates(two)
    chosen = method if method in candidates else SpineMethod.QUAD_MID
    # У AUTO в подписи — способ, который за ним стоит на этом кадре.
    label = auto_source(candidates) if chosen is SpineMethod.AUTO else chosen
    return spread_lines(two, candidates[chosen], label, book, finger_hands(book & ~zone, paint))


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


@dataclass(frozen=True)
class PagesOutcome:
    """Исход выхода страниц кадра для сводки.

    Поля:
        label: Записанные части через ``+`` (``L``, ``R``, ``LR``).
        status: Исход (``rectify.pages.PagesStatus``); пусто — выхода страниц нет.
        tint: Доли тонированного фона левой и правой страницы через ``/``.
        finger_mm2: Площадь закраса пальцев на записанных страницах через ``/``, мм².
        text: Меры прямизны текста до и после поля (JSON одной строкой) или пусто.
    """

    label: str = ""
    status: str = ""
    tint: str = ""
    finger_mm2: str = ""
    text: str = ""


def pages_path(job: FrameJob, params: RunParams, suffix: str) -> Path:
    """Путь части разворота в выходе страниц: ``<output_pages>/<подпапки>/<кадр><суффикс>.<расширение итога>``.

    Args:
        job: Кадр.
        params: Параметры прогона (``output_pages`` задан).
        suffix: Суффикс части (``rectify.pages.PagePart``: ``_L``, ``_R``, ``_LR``).

    Returns:
        Путь файла страницы.
    """
    ext = resolve_output_suffix(job.rel.suffix, params.output_format)
    return params.output_pages / job.rel.parent / f"{job.rel.stem}{suffix}{ext}"


def pages_done(job: FrameJob, params: RunParams) -> bool:
    """Записаны ли уже все части выхода страниц, которые кадру положены по флагам.

    Args:
        job: Кадр.
        params: Параметры прогона.

    Returns:
        True — выхода страниц нет или все положенные части есть на диске.
    """
    if params.output_pages is None:
        return True
    from ocr_utils.scan_cropping.v2.rectify.pages import wanted_parts

    cached = WorkPaths(params.work_dir / job.key).pages
    if not cached.exists():
        # Вид кадра ещё не известен (кэша страниц нет) — части не проверить, кадр считается заново.
        return False
    kind, index = read_frame_kind(cached)
    return all(pages_path(job, params, part.value).exists() for part in wanted_parts(job.flags, kind, index))


def write_rectified_pages(
    job: FrameJob,
    params: RunParams,
    out: np.ndarray,
    status: str,
    box: tuple[int, int, int, int] | None,
    fit: SpreadFit | None,
    seg_scale: tuple[float, float],
    polys: list[np.ndarray],
    labels: list[str] | None,
    fingers: np.ndarray | None = None,
    kind: tuple[FrameKind, int | None] = (FrameKind.SPREAD, None),
) -> PagesOutcome:
    """Выход страниц кадра: выпрямление итога, разрез на страницы по флагам, закрас до прямоугольника, запись.

    Ошибка выпрямления не роняет кадр: основной итог уже записан, в сводку идёт исход ``error``.

    Args:
        job: Кадр (флаги — ``job.flags``).
        params: Параметры прогона (``output_pages`` задан).
        out: Итог кадра v2 BGR (после закраса, уровней и осветления тени).
        status: Исход основного итога (``ok`` — вырезан по рамке ``box``).
        box: Рамка вырезки итога в px кадра.
        fit: Подгонка разворота (области двух страниц) или None.
        seg_scale: Масштаб сегментации относительно кадра по X и Y.
        polys: Блоки surya в px кадра (формулы — опоры-преграды разбора текста).
        labels: Типы блоков surya или None.
        fingers: Зона закраса пальцев, не возвращённая copy back, в координатах итога (bool, размер ``out``) или
            None; переносится на страницы и пишется в паспорт (``--output-pages-meta``).
        kind: Вид кадра и место широкой страницы кадра ``SLIVER`` (``frame_kind_of``): одна страница пишется
            частью ``_P`` (у ``SLIVER`` — частью своей стороны), без страницы — ничего.

    Returns:
        Исход выхода страниц для сводки.
    """
    from ocr_utils.scan_cropping.v2.rectify.geometry import SpreadGeometry, equations_of
    from ocr_utils.scan_cropping.v2.rectify.overlay import pages_overlay
    from ocr_utils.scan_cropping.v2.rectify.pages import PagePart, PagesStatus, cut_pages, page_mask_of, wanted_parts
    from ocr_utils.scan_cropping.v2.rectify.passport import PagePassport, finger_area_mm2, write_passport, zone_contours
    from ocr_utils.scan_cropping.v2.rectify.sheets import sheet_result
    from ocr_utils.scan_cropping.v2.rectify.text_run import PageControl, TextParams, text_frame
    from ocr_utils.scan_cropping.v2.gutter_decompression.sheet import gutter_sheet, summary_fields
    from ocr_utils.frame_flags.flags import Side
    from ocr_utils.units import MM_PER_INCH

    frame_kind, spread_index = kind
    if job.flags.skip_frame:
        return PagesOutcome(status=PagesStatus.OK.value)
    if status != "ok" or box is None:
        return PagesOutcome(status=PagesStatus.NO_PAPER.value)
    if frame_kind is FrameKind.NO_PAGE:
        log.info("%s: в кадре нет страницы (закрытая крышка, ярлык) — выход страниц пропущен", job.key)
        return PagesOutcome(status=PagesStatus.NO_PAGE.value)
    parts = wanted_parts(job.flags, frame_kind, spread_index)
    if not parts:
        return PagesOutcome(status=PagesStatus.OK.value)
    if fit is None:
        log.warning("%s: нет подгоночных областей страниц — выход страниц пропущен", job.key)
        return PagesOutcome(status=PagesStatus.NO_FRAMES.value)
    # Размеры выпрямления — в мм бумаги, поэтому dpi — настоящая плотность кадра, а не тег --force-dpi (у пака-4 тег 450
    # при 13.8 px/мм ≈ 350 dpi: поле считалось бы в 1.29 раза мельче задуманного). Тег страниц — --pages-dpi или она же.
    dpi = params.scale.px_per_mm * MM_PER_INCH
    tag_dpi = int(round(params.pages_dpi or dpi))
    geometry = SpreadGeometry(fit.frames, box, seg_scale, equations_of(polys, labels or []))
    # Флаги кадра → запреты страницам: без освещения — по ``no-illumination-*``; без поля по строкам — игнорируемая (её
    # текст не тянет поле) и помеченная ``no-text-rectify-*`` (передняя обложка, страница-картинка). Флаг обложки сам
    # ничего не выключает: задней обложке с анонсом книг выпрямление строк нужно; цветные и тонированные страницы
    # автоматика (``text_run``) и так оставляет без освещения и поля.
    # Одна страница кадра — один запрет по флагам её сторон: у ``SINGLE`` — обеих (флаг любой стороны выключает), у
    # ``SLIVER`` — стороны широкой страницы. Игнорирование одной стороны одну страницу не выключает: её пишут
    # (``wanted_parts``), значит, и выпрямляют.
    page_sides = [(Side.LEFT,), (Side.RIGHT,)]
    if frame_kind is FrameKind.SINGLE:
        page_sides = [(Side.LEFT, Side.RIGHT)]
    elif frame_kind is FrameKind.SLIVER:
        page_sides = [(Side.LEFT,) if spread_index == 0 else (Side.RIGHT,)]
    single = len(page_sides) == 1
    controls = tuple(PageControl(equalize=all(job.flags.equalize(side) for side in sides),
                                 field=all((single or not job.flags.ignored(side)) and job.flags.text_rectify(side)
                                           for side in sides))
                     for sides in page_sides)  # fmt: skip
    text = TextParams(illumination=params.rectify_illumination, passes=params.rectify_passes,
                      gutter_decompression=params.gutter_decompression)  # fmt: skip
    try:
        result = text_frame(out, geometry, dpi, text, controls)
        images = cut_pages(result, parts, dpi, params.fill, params.fill_seam_mm)
    except Exception:  # noqa: BLE001 — выход страниц не роняет кадр: основной итог уже записан
        log.exception("%s: выход страниц не построился", job.key)
        return PagesOutcome(status=PagesStatus.ERROR.value)
    # Страницы итога выпрямления (``result.pages``) по частям: у разворота — слева направо, у одной страницы — одна.
    sides_of = {PagePart.LEFT: (0,), PagePart.RIGHT: (1,), PagePart.SPREAD: (0, 1)}
    if single:
        sides_of = {part: (0,) for part in parts}
    finger_mm2 = []
    for page in images:
        path = pages_path(job, params, page.part.value)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_image(path, page.image, imwrite_params(path.suffix), tag_dpi)
        sides = sides_of[page.part]
        zone = None if fingers is None else page_mask_of(result, fingers, page)
        finger_mm2.append(0.0 if zone is None else finger_area_mm2(zone, dpi))
        passport = PagePassport(
            source=job.rel.as_posix(),
            part=page.part.value.lstrip("_"),
            flags=[f.value for f in job.flags.flags],
            cover=any(job.flags.cover(side) for i in sides for side in page_sides[i]),
            dpi=round(dpi, 2),
            width=int(page.image.shape[1]),
            height=int(page.image.shape[0]),
            tint=round(max(result.pages[i].tint for i in sides), 4),
            plain=all(result.pages[i].plain for i in sides),
            finger_mm2=0.0 if zone is None else round(finger_area_mm2(zone, dpi), 1),
            finger_zones=[] if zone is None else zone_contours(zone, dpi),
        )
        if params.output_pages_meta is not None:
            meta = params.output_pages_meta / job.rel.parent / f"{job.rel.stem}{page.part.value}"
            write_passport(meta, passport, zone)
    if params.debug_dir is not None and params.debug_steps:
        header = [f"{job.rel} — выход страниц: Кунс + поле по тексту ({params.rectify_passes} прох.), "
                  f"{dpi:.0f} dpi, флаги: {job.flags.label() or 'нет'}"]  # fmt: skip
        # Триптих стенда page_rectify (лист 04): итог v2 с областями | Кунс | итог, оси строк и линия разреза.
        sheets = {OverlayTag.RECTIFY: sheet_result(result, header),
                  OverlayTag.RECTIFIED_PAGES: pages_overlay(result, parts, header)}  # fmt: skip
        if result.unstretched is not None and result.gutter is not None:
            # Растяжение у корешка применено хотя бы к одной странице: «было | стало» у корешка.
            sheets[OverlayTag.GUTTER] = gutter_sheet(result.unstretched, result.final, result.layout, result.canvas,
                                                     result.gutter, dpi / MM_PER_INCH, header)  # fmt: skip
        for tag, sheet in sheets.items():
            sheet_path = overlay_path(params.debug_dir, job.rel, tag)
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return PagesOutcome(
        "+".join(page.part.value.lstrip("_") for page in images),
        PagesStatus.OK.value,
        "/".join(f"{p.tint:.2f}" for p in result.pages),
        "/".join(f"{v:.0f}" for v in finger_mm2),
        json.dumps({**result.before.row("before_"), **result.after.row("after_"), **summary_fields(result.gutter)}),
    )


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
    names = list(asdict(row).keys())
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        if header != names:
            # Сводка прежней версии с другими колонками: дописывать в неё — разъехать столбцы. Откладываем рядом.
            path.rename(path.with_name(f"{path.stem}.{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"))
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        if new:
            writer.writeheader()
        writer.writerow(asdict(row))


class LockedModels:
    """``GpuModels`` для потоков закраса пальцев: каждый вызов метода моделей — под общей блокировкой GPU прогона."""

    def __init__(self, models, lock: threading.RLock) -> None:
        """Args:
        models: ``GpuModels``.
        lock: Блокировка GPU, общая с шагами SAM 3, SAM 2, уточнения и surya главного потока.
        """
        self._models = models
        self._lock = lock

    def __getattr__(self, name: str):
        """Атрибут моделей; метод — обёрнутый блокировкой."""
        attr = getattr(self._models, name)
        return functools.partial(self._locked, attr) if callable(attr) else attr

    def _locked(self, fn, *args, **kwargs):
        """Вызов ``fn`` под блокировкой GPU."""
        with self._lock:
            return fn(*args, **kwargs)


def finger_task(job: FrameJob, models: LockedModels, params: RunParams) -> None:
    """Закрас пальцев кадра в потоке (``gpu_fingers`` с моделями под блокировкой GPU)."""
    gpu_fingers(job, models, params)


class FrameStream:
    """Кадры в работе: фоны, страницы и финалы в пуле, SAM 3 и surya в главном процессе, сводка по мере готовности."""

    def __init__(
        self,
        pool: ProcessPoolExecutor,
        models,
        predictor,
        sam2,
        params: RunParams,
        refiner=None,
        threads: ThreadPoolExecutor | None = None,
        gpu_lock: threading.RLock | None = None,
    ) -> None:
        """Пустой поток.

        Args:
            pool: Пул CPU-шагов.
            models: ``GpuModels`` с surya layout.
            predictor: SAM 3 (``paper.load_sam3``).
            sam2: Фон SAM 2 для вето (``bg_veto.Sam2Background``) или None без вето.
            params: Параметры прогона.
            refiner: ``edge_refine.Refiner`` для уточнения рваных сторон или None.
            threads: Потоки закраса пальцев или None — пальцы считает вызывающий.
            gpu_lock: Блокировка GPU, общая с потоками пальцев (None — без потоков).
        """
        self.pool = pool
        self.threads = threads
        self.gpu_lock = gpu_lock if gpu_lock is not None else threading.RLock()
        self.fingers: dict[Future, FrameJob] = {}
        self.models = models
        self.predictor = predictor
        self.sam2 = sam2
        self.refiner = refiner
        self.params = params
        self.refines: dict[Future, FrameJob] = {}
        self.summary = params.work_dir / "summary.csv"
        self.books: dict[Future, FrameJob] = {}
        self.pages: dict[Future, FrameJob] = {}
        self.finishes: dict[Future, FrameJob] = {}
        self.stats: list[FrameStats] = []

    @property
    def busy(self) -> int:
        """Сколько кадров в работе."""
        return len(self.fingers) + len(self.books) + len(self.pages) + len(self.refines) + len(self.finishes)

    def submit_fingers(self, job: FrameJob) -> None:
        """Закрас пальцев кадра в поток (готовый в кэше — сразу фон и книга в пул)."""
        if frame_ready(job, self.params):
            self.submit(job)
            return
        locked = LockedModels(self.models, self.gpu_lock)
        self.fingers[self.threads.submit(finger_task, job, locked, self.params)] = job

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
        """Разобрать готовое: фоны → SAM 3 и SAM 2 (GPU) → страницы в пул; страницы → surya (GPU) → финал в пул; финалы → сводка.

        Args:
            block: Ждать хотя бы один готовый шаг (иначе — только уже готовые).
        """
        pending = list(self.fingers) + list(self.books) + list(self.pages) + list(self.refines) + list(self.finishes)
        if not pending:
            return
        ready, _ = wait(pending, timeout=None if block else 0, return_when=FIRST_COMPLETED)
        for fut in ready:
            if fut in self.fingers:
                job = self.fingers.pop(fut)
                try:
                    fut.result()
                except Exception as exc:
                    log.exception("%s: ошибка закраса пальцев", job.key)
                    self.record(FrameStats(job.key, "error", message=repr(exc)))
                    continue
                self.submit(job)
            elif fut in self.books:
                job = self.books.pop(fut)
                try:
                    fut.result()
                    work = WorkPaths(self.params.work_dir / job.key)
                    with self.gpu_lock:
                        if not sam3_ready(work, self.params):
                            gpu_sam3(job, self.predictor, self.params)
                        if not sam2_ready(work, self.params):
                            gpu_sam2_bg(job, self.sam2, self.params)
                except Exception as exc:
                    log.exception("%s: ошибка фона, SAM 3 или SAM 2", job.key)
                    self.record(FrameStats(job.key, "error", message=repr(exc)))
                    continue
                self.pages[self.pool.submit(cpu_pages, (job, self.params))] = job
            elif fut in self.pages or fut in self.refines:
                refined = fut in self.refines
                job = (self.refines if refined else self.pages).pop(fut)
                work = WorkPaths(self.params.work_dir / job.key)
                try:
                    fut.result()
                    if not refined and self.refiner is not None and work.refine_in.exists():
                        # Рваные стороны: уточнение на GPU, замена бумаги и пересчёт страниц — в пуле, потом surya.
                        with self.gpu_lock:
                            gpu_refine(job, self.refiner, self.params)
                        self.refines[self.pool.submit(cpu_refine, (job, self.params))] = job
                        continue
                    if not layout_ready(work, self.params):
                        with self.gpu_lock:
                            gpu_layout(job, self.models, self.params)
                except Exception as exc:
                    log.exception("%s: ошибка страниц, уточнения или surya", job.key)
                    self.record(FrameStats(job.key, "error", message=repr(exc)))
                    continue
                self.finishes[self.pool.submit(cpu_finish, (job, self.params))] = job
            else:
                self.finishes.pop(fut)
                self.record(fut.result())


def run_batch(files: list[Path], params: RunParams) -> list[FrameStats]:
    """Прогон v2 по списку кадров потоком: кадр проходит все шаги, итог и оверлей появляются по мере готовности.

    Все модели (``GpuModels``, SAM 3 и при вето фона SAM 2) грузятся один раз и живут весь прогон: пик видеопамяти
    ~7.4 ГБ без SAM 2 (замер 2026-10-12 на RTX 5060 Ti 16 ГБ). GPU-шаги идут в главном процессе: закрас пальцев — в
    ``finger_threads`` потоках (его CPU-часть перекрывается с GPU других кадров, вызовы сетей — под общей блокировкой
    GPU), SAM 3 и SAM 2 для кадров, у которых пул уже посчитал фон, уточнение рваных сторон и surya — в главном потоке
    под той же блокировкой; CPU-шаги (фон и книга, бумага и страницы, финал) — в пуле. В работе не больше ``2 × jobs``
    кадров. Готовые шаги берутся из кэша (кроме ``redo``).

    Args:
        files: Входные кадры.
        params: Параметры прогона.

    Returns:
        Строки сводки по кадрам (дописываются и в ``work_dir/summary.csv`` по мере готовности).
    """
    from ocr_utils.scan_cropping.gpu_models import GpuModels
    from ocr_utils.scan_cropping.v2.bg_veto import Sam2Background
    from ocr_utils.scan_cropping.v2.edge_refine import Refiner
    from ocr_utils.scan_cropping.v2.paper import load_sam3

    params.work_dir.mkdir(parents=True, exist_ok=True)
    jobs, skipped = frame_jobs(files, params)
    for row in skipped:
        write_summary_row(params.work_dir / "summary.csv", row)
    if params.skip_if_exists:
        before = len(jobs)
        jobs = [j for j in jobs if not (output_path(j, params).exists() and pages_done(j, params))]
        log.info("пропущено готовых: %d", before - len(jobs))
    in_flight = max(2, 2 * params.jobs)
    ctx = get_context("forkserver")
    gpu_lock = threading.RLock()
    with (
        ProcessPoolExecutor(params.jobs, mp_context=ctx, initializer=init_worker) as pool,
        ThreadPoolExecutor(max(1, params.finger_threads), thread_name_prefix="fingers") as threads,
        GpuModels(with_lama=True, with_layout=True) as models,
    ):
        # SAM 2.1 Hiera-L для вето фона: +~1 ГБ видеопамяти к ~7.4 ГБ остальных моделей.
        sam2 = Sam2Background() if params.bg_veto or params.refine_ragged else None
        predictor = load_sam3()
        # Уточнение рваных сторон — теми же SAM 3 и SAM 2.1: новых весов в видеопамяти нет.
        refiner = Refiner(predictor, sam2) if params.refine_ragged else None
        stream = FrameStream(pool, models, predictor, sam2 if params.bg_veto else None, params, refiner, threads,
                             gpu_lock)  # fmt: skip
        for job in jobs:
            # В работе не больше in_flight кадров, в потоках пальцев — не больше потоков (следующий кадр ждёт в очереди).
            while stream.busy >= in_flight or len(stream.fingers) >= max(1, params.finger_threads):
                stream.drain(block=True)
            stream.submit_fingers(job)
            stream.drain(block=False)
        while stream.busy:
            stream.drain(block=True)
    return sorted(stream.stats + skipped, key=lambda r: r.key)


def frame_jobs(files: list[Path], params: RunParams) -> tuple[list[FrameJob], list[FrameStats]]:
    """Кадры прогона с флагами из сайдкаров; кадры с ``ignore-both`` и с битым сайдкаром в прогон не идут.

    Args:
        files: Входные кадры.
        params: Параметры прогона (``use_flags`` — читать сайдкары).

    Returns:
        Кадры в работу и строки сводки отсеянных (``ignored`` / ``bad_flags``).
    """
    jobs, skipped = [], []
    for src in files:
        job = FrameJob(src, src.relative_to(params.input_dir))
        if params.use_flags:
            try:
                job = replace(job, flags=read_flags(src))
            except FlagsError as exc:
                # Битый сайдкар — кадр не обрабатывается: молча обработать его как обычный значило бы потерять флаг.
                log.error("%s: сайдкар не прошёл проверку, кадр пропущен: %s", job.key, exc)
                skipped.append(FrameStats(job.key, "bad_flags", message=str(exc)))
                continue
        if job.flags.skip_frame:
            log.info("%s: ignore-both — кадр не обрабатывается (%s)", job.key, job.flags.note or "без пояснения")
            skipped.append(FrameStats(job.key, "ignored", flags=job.flags.label()))
            continue
        jobs.append(job)
    if params.use_flags:
        flagged = sum(not j.flags.empty for j in jobs)
        log.info("флаги кадров: с флагами %d, отсеяно %d (ignore-both и битые сайдкары)", flagged, len(skipped))
    return jobs, skipped
