"""Стадии разбора пака: CPU-части по полосам и по кандидатам (для пула), без GPU и без vLLM.

Порядок разбора полосы (решение пользователя 2026-09-27):

0. surya layout — из кэша (набит заранее, ``prefill-surya``);
1. ориентация — CPU-детекторы и :func:`orientation.analysis.combine`; уверенный неспорный поворот
   применяется: кадр поворачивается, surya для повёрнутого кадра считается в родителе под своим
   ключом кэша (``<полоса>/rot<угол>``);
2. растр → 3. таблицы (подсказки line art и линейки-сироты наружу) → 4. line art: кандидаты
   классического детектора (с подсказками surya и таблиц), формулы surya сразу, кандидаты —
   вырезками в DeepSeek (первый проход; при надобности — второй по залитой вырезке);
5. текстовые блоки — с запретами: растр, печати, таблицы, line art, формулы; барьеры — рамки и
   линейки-сироты.

Стадии обмениваются файлами в рабочей папке прогона (``work/``): полоса — JSON, кандидат —
PNG-вырезка и запись в ``jobs_*.jsonl``; вывод DeepSeek — ``deepseek/pass{1,2}/<промпт>.jsonl``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.line_art.deepseek.decide import Decision, decide, needs_pass2
from ocr_utils.page_layout.line_art.deepseek.pass2 import fill_words
from ocr_utils.page_layout.line_art.deepseek.rules import Crop, crop_region
from ocr_utils.page_layout.orientation.analysis import combine
from ocr_utils.page_layout.pack_analysis.raster_db import DB_SOURCE
from ocr_utils.page_layout.regions import Region, RegionKind
from ocr_utils.page_layout.surya.cache import SuryaCache
from ocr_utils.page_layout.surya.source import SuryaSource
from ocr_utils.scan_markup.rotation import rotate_cw

logger = logging.getLogger(__name__)

# CPU-детекторы ориентации: ось — ink_axis и profile, сторона — osd (tesseract). GPU-детекторы и
# арбитр ocr_vote не гоняются: на заострённых копиях пака-1 почти все полосы уже прямые.
ORIENTATION_DETECTORS = ("ink_axis", "profile", "osd")

# Поворот применяется только при такой сводной уверенности и без спора детекторов о стороне.
ROTATE_MIN_CONFIDENCE = 0.6

# Разрешение сканов пака-1 без тега.
DEFAULT_DPI = 600


@dataclass(frozen=True)
class PageTask:
    """Полоса пака.

    Attributes:
        name: Имя «год/выпуск/полоса» (оно же ключ кэша surya).
        path: Путь к заострённой копии.
    """

    name: str
    path: Path


def page_key(name: str) -> str:
    """Имя полосы для файлов: «1973/06/IMG_0103_2R» → «1973_06_IMG_0103_2R»."""
    return name.replace("/", "_")


def rotated_name(name: str, degrees: int) -> str:
    """Ключ кэша surya для повёрнутого кадра."""
    return f"{name}/rot{degrees}"


def load_image(task: PageTask, rotate: int = 0) -> PageImage:
    """Полоса как ``PageImage``; при ``rotate`` ≠ 0 — повёрнутый кадр под своим ключом кэша.

    Args:
        task: Полоса.
        rotate: Поворот по часовой, градусы (0, 90, 180, 270).

    Returns:
        Страница.
    """
    image = PageImage.from_file(task.path, Variant.SHARPENED, task.name, default_dpi=DEFAULT_DPI)
    if rotate == 0:
        return image
    turned = rotate_cw(image.bgr_at(image.dpi), rotate)
    return PageImage.from_array(turned, image.dpi, Variant.SHARPENED, rotated_name(task.name, rotate))


def orient_page(task: PageTask) -> dict:
    """Стадия 1: вердикт ориентации полосы CPU-детекторами.

    Args:
        task: Полоса.

    Returns:
        ``{"page", "rotate_cw", "confidence", "disputed", "axis_only", "note", "apply"}``;
        ``apply`` — поворот уверенный, неспорный и ненулевой.
    """
    image = load_image(task)
    options = LayoutOptions(use_surya=False, orientation_detectors=ORIENTATION_DETECTORS)
    layout = PageLayout(image, {Find.ORIENTATION}, options).process(None)
    verdict, _source, disputed = combine(layout.orientation_verdicts)
    apply = (
        verdict.rotate_cw != 0
        and not disputed
        and not verdict.axis_only
        and verdict.confidence >= ROTATE_MIN_CONFIDENCE
    )
    return {
        "page": task.name,
        "rotate_cw": int(verdict.rotate_cw),
        "confidence": round(float(verdict.confidence), 3),
        "disputed": bool(disputed or verdict.axis_only),
        "axis_only": bool(verdict.axis_only),
        "note": verdict.note,
        "apply": bool(apply),
    }


def _region_json(region) -> dict:
    """Область ``page_layout`` → словарь для JSON (рамка в пикселях полосы)."""
    return {
        "kind": region.kind.value,
        "box": list(region.box.as_tuple()),
        "confidence": region.confidence,
        "info": region.info,
    }


def candidates_page(
    task: PageTask, rotate: int, cache_root: Path, work: Path, raster: list[dict] | None = None
) -> dict:
    """Стадия 2: растр, таблицы, кандидаты line art, формулы surya, повёрнутый текст; вырезки кандидатов.

    Args:
        task: Полоса.
        rotate: Применённый поворот (из стадии 1), градусы.
        cache_root: Корень кэша surya.
        work: Рабочая папка прогона.
        raster: Готовые растровые области полосы (из базы разметки, :func:`raster_db.load_raster`), в
            пикселях кадра полосы. ``None`` — растр ищет детектор; список (в том числе пустой) —
            детектор растра не запускается, области идут в разбор известными исключениями: line art
            и повёрнутый текст ищутся вне них, таблицы, накрытые ими, отбрасываются.

    Returns:
        Описание полосы (то же пишется в ``work/pages/<ключ>.json``), с ``candidates`` — кандидаты
        line art: ``{"id", "crop": Crop, "info"}``.
    """
    image = load_image(task, rotate)
    surya = SuryaSource(SuryaCache(cache_root, readonly=True), None)
    finds = {Find.TABLES, Find.LINE_ART, Find.ROTATED_TEXT}
    known = None
    if raster is None:
        finds.add(Find.RASTER)
    else:
        # Выверенный растр — известное семейство: ``PageLayout`` берёт его в исключения, как на ``detect``.
        known = {Find.RASTER: [Region(Box(*r["box"]), RegionKind(r["kind"]), None, DB_SOURCE) for r in raster]}
    layout = PageLayout(image, finds, LayoutOptions(), known=known).process(surya)
    gray = image.gray
    crops_dir = work / "crops" / "pass1"
    crops_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for index, region in enumerate(layout.line_arts):
        crop_gray, crop = crop_region(gray, region.box, image.dpi)
        candidate_id = f"{page_key(task.name)}_{index}"
        cv2.imwrite(str(crops_dir / f"{candidate_id}.png"), crop_gray)
        candidates.append({"id": candidate_id, "crop": crop.to_json(), "info": region.info})
    record = {
        "page": task.name,
        "rotate_cw": rotate,
        "size": [image.width, image.height],
        "dpi": image.dpi,
        "surya_used": layout.surya_used,
        "raster": (
            raster if raster is not None else [_region_json(r) for r in layout.raster_pics + layout.stamp_suspects]
        ),
        "tables": [_region_json(r) for r in layout.tables],
        "formulas": [_region_json(r) for r in layout.formulas],
        "rotated_text": [_region_json(r) for r in layout.rotated_text_not_in_tables_regions],
        "loose_rules": [rule.to_json() for rule in layout.loose_rules],
        "candidates": candidates,
    }
    write_json(work / "pages" / f"{page_key(task.name)}.json", record)
    return record


def pass2_candidate(candidate: dict, markdown: list[dict], words: list[dict], work: Path) -> bool:
    """Стадия 4: нужен ли второй проход; если да — залитая вырезка в ``crops/pass2``.

    Args:
        candidate: Кандидат из JSON полосы.
        markdown: Блоки ``markdown`` первого прохода.
        words: Слова ``ocr`` первого прохода.
        work: Рабочая папка.

    Returns:
        ``True``, если вырезка второго прохода записана.
    """
    crop = Crop.from_json(candidate["crop"])
    if not needs_pass2(crop, markdown):
        return False
    gray = cv2.imread(str(work / "crops" / "pass1" / f"{candidate['id']}.png"), cv2.IMREAD_GRAYSCALE)
    binary, _ = fill_words(gray, crop.inner, words)
    target = work / "crops" / "pass2"
    target.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target / f"{candidate['id']}.png"), binary)
    return True


def decide_candidates(record: dict, image: PageImage, deepseek: dict, work: Path) -> list[tuple[dict, Decision]]:
    """Стадия 6, часть line art: решение по каждому кандидату полосы.

    Args:
        record: JSON полосы.
        image: Страница (повёрнутая, если надо) — краска для достройки рамок.
        deepseek: ``{"pass1_markdown": {id: элементы}, "pass2_markdown": {id: элементы}}``.
        work: Рабочая папка.

    Returns:
        Пары (кандидат, решение).
    """
    ink = image.bitonal_at(image.dpi) == 0
    barriers = [Box(*r["box"]) for r in record["raster"] + record["tables"]]
    decisions = []
    for candidate in record["candidates"]:
        crop = Crop.from_json(candidate["crop"])
        gray = cv2.imread(str(work / "crops" / "pass1" / f"{candidate['id']}.png"), cv2.IMREAD_GRAYSCALE)
        markdown = deepseek["pass1_markdown"].get(candidate["id"], [])
        second = deepseek["pass2_markdown"].get(candidate["id"])
        binary = None
        if second is not None:
            binary = cv2.imread(str(work / "crops" / "pass2" / f"{candidate['id']}.png"), cv2.IMREAD_GRAYSCALE)
        decisions.append((candidate, decide(crop, gray, markdown, ink, barriers, second, binary)))
    return decisions


def write_json(path: Path, payload: dict) -> None:
    """Записать JSON, создав папку."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False))


def read_jsonl_map(path: Path, field: str = "elements") -> dict[str, list]:
    """``{id: поле}`` из JSONL вывода воркера DeepSeek (пустой словарь, если файла нет)."""
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            result[record["id"]] = record.get(field, [])
    return result


def init_worker() -> None:
    """Инициализатор воркера пула: без hugepage, один поток OpenCV и BLAS (``.claude/rules/gpu_and_pools.md``)."""
    np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001 — без threadpoolctl остаётся только окружение
        pass


__all__ = [
    "DEFAULT_DPI",
    "ORIENTATION_DETECTORS",
    "PageTask",
    "RegionKind",
    "candidates_page",
    "decide_candidates",
    "init_worker",
    "load_image",
    "orient_page",
    "page_key",
    "pass2_candidate",
    "read_jsonl_map",
    "rotated_name",
    "write_json",
]
