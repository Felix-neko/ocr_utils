"""Защитная маска контента внутри области (бумага или книга): вырезки ~300 dpi, разбор разными движками, маски, меры."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from research.bound_volumes.frames import OUT_ROOT, Frame, read_bgr

log = logging.getLogger(__name__)

CONTENT_ROOT = OUT_ROOT / "runs" / "content"
# Вырезка для движков: кадр ~350 dpi → ~300 dpi.
CROP_SCALE = 0.85
# Масштаб масок и оверлеев относительно кадра (как у масок книги и бумаги).
MASK_SCALE = 0.5
# Зазор защитной маски вокруг объектов движка, px ½ кадра (~1.5 мм).
GUARD_PX = 10
# Блок surya шире этой доли вырезки по обеим сторонам — «вся страница», в защиту не идёт.
FULL_PAGE_FRAC = 0.85


class Area(str, Enum):
    """Область анализа."""

    # Бумага после сегментации бумаги.
    PAPER = "paper"
    # Книга после постобработки, до сегментации бумаги.
    BOOK = "book"


class Engine(str, Enum):
    """Движок разбора страницы."""

    SURYA_LAYOUT = "surya_layout"
    SURYA_LINES = "surya_lines"
    PERO = "pero"
    PAGE_LAYOUT = "page_layout"
    DEEPSEEK = "deepseek"
    PAGE_LAYOUT_DEEPSEEK = "page_layout+deepseek"
    GLYPHS = "glyphs"


@dataclass(frozen=True)
class Item:
    """Объект движка: многоугольник в px ½ кадра и вид (метка движка)."""

    kind: str
    polygon: list[list[float]]


@dataclass(frozen=True)
class CropMeta:
    """Вырезка области: смещение в px кадра и масштаб к кадру."""

    x0: int
    y0: int
    scale: float


def area_mask(frame: Frame, area: Area, paper_method: str) -> np.ndarray:
    """Маска области в ½ кадра.

    Args:
        frame: Кадр.
        area: Область.
        paper_method: Способ сегментации бумаги (для области PAPER).

    Returns:
        Булева маска.
    """
    root = OUT_ROOT / "runs"
    path = root / "paper" / paper_method if area == Area.PAPER else root / "book_post" / "close_open"
    return cv2.imread(str(path / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE) > 0


def crop_dir(area: Area) -> Path:
    """Папка вырезок области.

    Args:
        area: Область.

    Returns:
        Путь.
    """
    return CONTENT_ROOT / area.value / "crops"


def make_crop(frame: Frame, area: Area, paper_method: str) -> tuple[Path, CropMeta]:
    """Вырезка области из кадра после закраса пальцев: вне области — ровный цвет бумаги, масштаб ~300 dpi.

    Args:
        frame: Кадр.
        area: Область.
        paper_method: Способ бумаги.

    Returns:
        ``(путь PNG, метаданные)``.
    """
    from research.bound_volumes.fingers import Guard, out_paths

    full = read_bgr(out_paths(Guard.INK_BACK, frame)[0])
    h, w = full.shape[:2]
    mask = (
        cv2.resize(area_mask(frame, area, paper_method).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    )
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    sub = full[y0:y1, x0:x1].copy()
    inside = mask[y0:y1, x0:x1]
    # Цвет заливки — светлая половина пикселей области (бумага, без краски).
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    light = inside & (gray >= np.median(gray[inside]))
    sub[~inside] = np.median(sub[light], axis=0).astype(np.uint8)
    crop = cv2.resize(sub, None, fx=CROP_SCALE, fy=CROP_SCALE, interpolation=cv2.INTER_AREA)
    path = crop_dir(area) / f"{frame.flat_key}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    meta = CropMeta(int(x0), int(y0), CROP_SCALE)
    path.with_suffix(".json").write_text(json.dumps(asdict(meta)))
    return path, meta


def read_meta(path: Path) -> CropMeta:
    """Метаданные вырезки.

    Args:
        path: Путь PNG вырезки.

    Returns:
        Метаданные.
    """
    return CropMeta(**json.loads(path.with_suffix(".json").read_text()))


def to_half(points, meta: CropMeta) -> list[list[float]]:
    """Точки вырезки (px вырезки) → px ½ кадра.

    Args:
        points: Точки ``(N, 2)``.
        meta: Метаданные вырезки.

    Returns:
        Список ``[x, y]``.
    """
    p = np.asarray(points, np.float64).reshape(-1, 2)
    p = (p / meta.scale + [meta.x0, meta.y0]) * MASK_SCALE
    return p.round(1).tolist()


def box_poly(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """Прямоугольник как многоугольник.

    Args:
        x0: Лево.
        y0: Верх.
        x1: Право.
        y1: Низ.

    Returns:
        Четыре вершины.
    """
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def items_path(area: Area, engine: str, frame: Frame) -> Path:
    """Файл объектов движка по кадру.

    Args:
        area: Область.
        engine: Имя движка.
        frame: Кадр.

    Returns:
        Путь JSON.
    """
    return CONTENT_ROOT / area.value / engine / f"{frame.flat_key}.json"


def save_items(path: Path, items: list[Item], seconds: float) -> None:
    """Пишет объекты движка.

    Args:
        path: Путь JSON.
        items: Объекты.
        seconds: Время движка на кадр.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(seconds=seconds, items=[asdict(i) for i in items]), ensure_ascii=False))


def load_items(path: Path) -> tuple[list[Item], float]:
    """Читает объекты движка.

    Args:
        path: Путь JSON.

    Returns:
        ``(объекты, секунды)``.
    """
    data = json.loads(path.read_text())
    return [Item(**i) for i in data["items"]], float(data["seconds"])


def surya_items(rgb: np.ndarray, model, meta: CropMeta) -> list[Item]:
    """Блоки surya layout вырезки (вырезка уменьшается до 2048 по длинной стороне).

    Args:
        rgb: Вырезка RGB.
        model: ``SuryaLayoutModel``.
        meta: Метаданные вырезки.

    Returns:
        Объекты (без блоков «вся страница» и с уверенностью ниже 0.3).
    """
    h, w = rgb.shape[:2]
    s = min(1.0, 2048 / max(h, w))
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    out = []
    for b in model.predict_one(small).blocks:
        if b.confidence < 0.3:
            continue
        bw, bh = (b.box.x1 - b.box.x0) / s, (b.box.y1 - b.box.y0) / s
        if bw > FULL_PAGE_FRAC * w and bh > FULL_PAGE_FRAC * h:
            continue
        poly = np.asarray(b.polygon if b.polygon is not None else box_poly(b.box.x0, b.box.y0, b.box.x1, b.box.y1)) / s
        out.append(Item(b.label, to_half(poly, meta)))
    return out


def surya_line_items(rgb: np.ndarray, predictor, meta: CropMeta) -> list[Item]:
    """Строки детектора текста surya.

    Args:
        rgb: Вырезка RGB.
        predictor: ``surya.detection.DetectionPredictor``.
        meta: Метаданные.

    Returns:
        Объекты вида ``line``.
    """
    from PIL import Image

    res = predictor([Image.fromarray(rgb)])[0]
    return [Item("line", to_half(b.polygon, meta)) for b in res.bboxes if b.confidence >= 0.3]


def pero_items(bgr: np.ndarray, meta: CropMeta) -> list[Item]:
    """Строки и регионы pero (подпроцесс в своём окружении).

    Args:
        bgr: Вырезка BGR (~300 dpi).
        meta: Метаданные.

    Returns:
        Объекты вида ``line`` и ``region``.
    """
    from ocr_utils.page_layout.text_blocks.engines.pero import PeroEngine

    r = PeroEngine().segment(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), dpi=300)
    out = [Item("line", to_half(l.polygon, meta)) for l in r.lines if l.polygon is not None and len(l.polygon) >= 3]
    return out + [Item("region", to_half(p, meta)) for p in r.regions if len(p) >= 3]


def page_layout_items(path: Path, model, meta: CropMeta) -> list[Item]:
    """Наш разбор страницы: блоки surya + растр, таблицы, line art, повёрнутый текст.

    Args:
        path: PNG вырезки (~300 dpi). Страница открывается из файла: разбор отпускает полный кадр после первого
            этапа и перечитывает его через загрузчик, у страницы из массива загрузчика нет.
        model: ``SuryaLayoutModel``.
        meta: Метаданные.

    Returns:
        Объекты: блоки surya по меткам и области детекторов по виду.
    """
    from ocr_utils.page_layout.analysis import LayoutOptions, PageLayout
    from ocr_utils.page_layout.image import PageImage, Variant
    from ocr_utils.page_layout.surya.source import SuryaSource

    image = PageImage.from_file(path, variant=Variant.CAMERA, default_dpi=300)
    bgr = read_bgr(path)
    lay = PageLayout(image, options=LayoutOptions(orientation_detectors=())).process(SuryaSource(None, model))
    h, w = bgr.shape[:2]
    out = []
    if lay.raw_surya_content is not None:
        for b in lay.raw_surya_content.blocks:
            if b.confidence < 0.3 or (
                (b.box.x1 - b.box.x0) > FULL_PAGE_FRAC * w and (b.box.y1 - b.box.y0) > FULL_PAGE_FRAC * h
            ):
                continue
            out.append(Item(f"surya:{b.label}", to_half(box_poly(b.box.x0, b.box.y0, b.box.x1, b.box.y1), meta)))
    for r in lay.regions:
        if r.full_page:
            continue
        out.append(
            Item(
                str(r.kind.value if hasattr(r.kind, "value") else r.kind),
                to_half(box_poly(r.box.x0, r.box.y0, r.box.x1, r.box.y1), meta),
            )
        )
    return out


def deepseek_items(record: dict, meta: CropMeta) -> list[Item]:
    """Элементы DeepSeek-OCR-2 (промпт markdown) из записи воркера.

    Args:
        record: Запись ``markdown.jsonl`` (``elements`` с рамками в px поданной вырезки).
        meta: Метаданные.

    Returns:
        Объекты по метке элемента.
    """
    out = []
    for e in record.get("elements", []):
        out.append(Item(str(e.get("label", "?"))[:24], to_half(box_poly(e["x0"], e["y0"], e["x1"], e["y1"]), meta)))
    return out


def protection_mask(items: list[Item], shape: tuple[int, int], area: np.ndarray) -> np.ndarray:
    """Защитная маска: объединение объектов, сомкнутое в колонки и раздутое на :data:`GUARD_PX`, внутри области.

    Args:
        items: Объекты движка.
        shape: ``(H, W)`` ½ кадра.
        area: Маска области.

    Returns:
        Булева маска.
    """
    from research.newspaper_crop.text_guard import guard_mask

    own = np.zeros(shape, np.uint8)
    for it in items:
        if len(it.polygon) >= 3:
            cv2.fillPoly(own, [np.asarray(it.polygon).round().astype(np.int32)], 1)
    return guard_mask(own > 0, GUARD_PX) & area


# Метки surya, которые в «криволинейном» разборе заменяются огибающими text_blocks (текстовые блоки).
SURYA_TEXT_LABELS = ("Text", "SectionHeader", "PageHeader", "PageFooter", "Caption", "ListItem", "Footnote",
                     "TextInlineMath", "Handwriting")  # fmt: skip


def text_blocks_items(path: Path, meta: CropMeta) -> list[Item]:
    """Криволинейные границы текстовых блоков ``page_layout.text_blocks`` (огибающая по осям строк).

    Args:
        path: PNG вырезки (~300 dpi; детектор считает пороги в мм бумаги при 300 dpi).
        meta: Метаданные вырезки.

    Returns:
        Объекты вида ``text_block`` — многоугольники огибающих.
    """
    from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
    from ocr_utils.page_layout.text_blocks.page import analyse_gray

    gray = cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2GRAY)
    analysis = analyse_gray(gray, InkEngine())
    # Разбор отдаёт координаты рабочей копии; к пикселям вырезки — отношением ширин.
    scale = gray.shape[1] / analysis.width
    return [Item("text_block", to_half(np.asarray(b.envelope.polygon) * scale, meta)) for b in analysis.blocks
            if b.envelope is not None and len(b.envelope.polygon) >= 3]  # fmt: skip


def non_text_layout(items: list[Item]) -> list[Item]:
    """Нетекстовые находки ``page_layout``: блоки surya не из :data:`SURYA_TEXT_LABELS` и области детекторов.

    Args:
        items: Объекты движка ``page_layout``.

    Returns:
        Отфильтрованные объекты.
    """
    return [it for it in items if not (it.kind.startswith("surya:") and it.kind.split(":", 1)[1] in SURYA_TEXT_LABELS)]
