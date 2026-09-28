"""Кэш входа блоковой стадии: аргументы ``blocks_of`` полосы, перехваченные при разборе ровно как в паке v3, в pickle на полосу."""

from __future__ import annotations

import json
import pickle
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.pack_analysis.final import text_blocks
from ocr_utils.page_layout.pack_analysis.stages import PageTask, load_image, page_key
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks import page as page_module
from ocr_utils.page_layout.text_blocks.blocks import TextBlock
from ocr_utils.page_layout.text_blocks.page import AxisKind

# Качество JPEG подложки рабочей копии (150 dpi, серый): только для оверлеев.
GRAY_JPEG_QUALITY = 90


@dataclass
class BlockInput:
    """Всё, что получает ``blocks_of`` на полосе, плюс подложка для оверлеев и прежний результат.

    Attributes:
        key: Полоса ``год/выпуск/полоса``.
        axes: Оси строк (``lines.LineAxis``) — уже со второй осью, колонкой и флагом ``cross``.
        zones: Зоны вёрстки (``columns.Zone``).
        gutters: Локальные межколонники (``columns.Gutter``).
        width: Ширина рабочей копии, пиксели.
        height: Высота рабочей копии, пиксели.
        ink: Краска текста рендера ``RENDER_DPI`` (булев массив).
        rules: Сплошные черты страницы (``segment.Rule``).
        dpi: Разрешение рабочей копии.
        smooth_pitches, coarse_factor, dilate: Параметры огибающей, с которыми звали ``blocks_of``.
        leaders: Отточия (``leaders.Leader``).
        barriers: Линейки-барьеры (``barriers.BarrierLines``) или ``None``.
        gray: Серая рабочая копия ``WORK_DPI`` (подложка оверлеев).
        legacy: Блоки, которые боевой ``blocks_of`` вернул на этом входе.
    """

    key: str
    axes: list
    zones: list
    gutters: list
    width: int
    height: int
    ink: np.ndarray
    rules: list
    dpi: float
    smooth_pitches: float
    coarse_factor: float
    dilate: float
    leaders: list
    barriers: object
    gray: np.ndarray
    legacy: list[TextBlock]


class BlocksSpy:
    """Подмена ``page.blocks_of`` на время разбора (контекстный менеджер): запоминает аргументы и результат.

    Разбор полосы идёт одним вызовом ``blocks_of`` (на полосах «только текст» нет повёрнутых зон,
    значит нет и разбора по областям). Если вызовов оказалось больше одного, полоса отмечается
    флагом ``calls`` и в кэш не идёт: вход тогда не один.

    Attributes:
        calls: Сколько раз звали ``blocks_of``.
        args: Позиционные аргументы последнего вызова.
        result: Что вернул последний вызов.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.args: tuple = ()
        self.result: list[TextBlock] = []
        self._original = page_module.blocks_of

    def blocks_of(self, *args):
        """Вызвать настоящий ``blocks_of`` и запомнить аргументы с результатом (подпись — как у оригинала)."""
        self.calls += 1
        self.args = args
        self.result = self._original(*args)
        return self.result

    def __enter__(self) -> "BlocksSpy":
        page_module.blocks_of = self.blocks_of
        return self

    def __exit__(self, *exc) -> None:
        page_module.blocks_of = self._original


def capture_page(key: str, pack_dir: Path, sharpened_dir: Path, records_dir: Path | None = None) -> BlockInput:
    """Разобрать полосу тем же входом, что в разборе пака v3, и перехватить вход блоковой стадии.

    Args:
        key: Полоса ``год/выпуск/полоса``.
        pack_dir: Выход разбора пака (``pack1_page_analysis_v3``): ``work/pages`` и ``pages``.
        sharpened_dir: Заострённые сканы.
        records_dir: Папка записей кандидатов полос; ``None`` — ``<pack_dir>/work/pages``. Нужна, когда
            ``work/pages`` пересобирается, а разбор v3 строился на прежних записях
            (``work/pages_before_rules_v2``).

    Returns:
        :class:`BlockInput` полосы.

    Raises:
        RuntimeError: ``blocks_of`` звали не один раз (разбор по областям).
    """
    name = page_key(key)
    records_dir = records_dir or pack_dir / "work" / "pages"
    record = json.loads((records_dir / f"{name}.json").read_text(encoding="utf-8"))
    final = json.loads((pack_dir / "pages" / f"{name}.json").read_text(encoding="utf-8"))
    image = load_image(PageTask(key, sharpened_dir / f"{key}.jpg"), record["rotate_cw"])
    with BlocksSpy() as spy:
        analysis, _ = text_blocks(image, record, final["objects"], AxisKind.BODY)
    if spy.calls != 1:
        raise RuntimeError(f"{key}: blocks_of вызван {spy.calls} раз — полоса разбиралась по областям")
    (axes, zones, gutters, width, ink, rules, dpi, smooth, coarse, dilate, leaders, barriers) = spy.args
    gray300 = image.gray_at(RENDER_DPI)
    gray = cv2.resize(gray300, (analysis.width, analysis.height), interpolation=cv2.INTER_AREA)
    return BlockInput(
        key=key,
        axes=list(axes),
        zones=list(zones),
        gutters=list(gutters),
        width=int(width),
        height=int(analysis.height),
        ink=np.asarray(ink) > 0,
        rules=list(rules or []),
        dpi=float(dpi),
        smooth_pitches=float(smooth),
        coarse_factor=float(coarse),
        dilate=float(dilate),
        leaders=list(leaders or []),
        barriers=barriers,
        gray=gray,
        legacy=list(spy.result),
    )


def save(item: BlockInput, path: Path) -> None:
    """Записать вход полосы: краска — упакованными битами, подложка — JPEG, всё вместе — pickle со сжатием.

    Args:
        item: Вход полосы.
        path: Куда писать (``<cache>/<ключ>.pkl.z``).
    """
    ok, jpeg = cv2.imencode(".jpg", item.gray, [cv2.IMWRITE_JPEG_QUALITY, GRAY_JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"{item.key}: не удалось сжать подложку")
    payload = dict(item.__dict__)
    payload["ink"] = (item.ink.shape, np.packbits(item.ink))
    payload["gray"] = jpeg.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), 3))
    tmp.replace(path)


def load(path: Path) -> BlockInput:
    """Прочитать вход полосы, записанный :func:`save`.

    Args:
        path: Файл кэша.

    Returns:
        :class:`BlockInput` с распакованной краской и подложкой.
    """
    payload = pickle.loads(zlib.decompress(path.read_bytes()))
    shape, bits = payload["ink"]
    payload["ink"] = np.unpackbits(bits, count=int(np.prod(shape))).reshape(shape).astype(bool)
    payload["gray"] = cv2.imdecode(np.frombuffer(payload["gray"], np.uint8), cv2.IMREAD_GRAYSCALE)
    return BlockInput(**payload)


def cache_path(cache_dir: Path, key: str) -> Path:
    """Файл кэша полосы ``key`` в папке ``cache_dir``."""
    return cache_dir / f"{page_key(key)}.pkl.z"


__all__ = ["BlockInput", "BlocksSpy", "WORK_DPI", "cache_path", "capture_page", "load", "save"]
