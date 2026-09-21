"""Кэш разметки surya layout на диске: pickle на полосу, та же раскладка папок, что у пака.

ЗАЧЕМ. Surya layout стоит около секунды GPU на полосу — три с половиной часа на пак-1, — а
нужна она двум потребителям сразу: растровому детектору (блоки Picture) и детектору таблиц
(Table, Figure, Form, Text). Считать её дважды незачем, а перечитывать пак ради повторного
прогона одного из детекторов — тем более. Поэтому ответ модели кладётся на диск, и любой
следующий прогон берёт его оттуда.

ПРАВИЛО КЭША: разбор полосы уже есть — грузим с диска и модель не зовём; файла нет, он не
читается или битый (в том числе не тот ``scan_rel_path`` внутри) — разбираем заново и
ПЕРЕЗАПИСЫВАЕМ. Любое исключение при чтении — это промах, а не ошибка прогона.

ФОРМАТ совместим с кэшем исследований (``research/legacy/table_processing``, команда
``layout-pack``): ``<каталог>/{год}/{выпуск}/{основа}.pkl`` со словарём
``{"scan_rel_path", "dpi", "layout": PageLayout.to_json(), "surya": сырой LayoutResult}``.
Именно им размечен пак-1 по сырым TIFF «Готовое» (``layout_surya_готовое``).

СЫРОЙ ОТВЕТ SURYA ПРИ ЧТЕНИИ НЕ ВОССТАНАВЛИВАЕТСЯ. Он лежит в pickle ради других
исследований, а конвейеру нужны только блоки из ``layout``. Восстановить его значило бы
импортировать surya (а с ней torch) в каждом из шестнадцати воркеров пула — секунды на
старт и сотни мегабайт на процесс ни за что. Поэтому читает ``_LenientUnpickler``: классы
чужих пакетов подменяются заглушкой, и pickle разбирается без них.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

from ocr_utils.page_layout.geometry import Box
from ocr_utils.scan_markup.table_detection.layout import PageLayout

logger = logging.getLogger(__name__)

# Блоки surya, которые растровый детектор берёт как затравки иллюстраций — те же, что в
# ``background_smoothing.layout.PICTURE_LABELS``.
PICTURE_LABELS = ("Picture",)

# Модули, чьи классы при чтении кэша восстанавливаются по-настоящему. Всё остальное (surya,
# pydantic, torch) — заглушка.
_TRUSTED_MODULES = ("builtins", "collections", "copyreg", "datetime", "numpy", "_codecs", "pathlib")


@dataclass(frozen=True)
class CachedLayout:
    """Разметка полосы в пикселях картинки ``layout.width`` x ``layout.height``.

    ``raw`` — сырой ответ surya, есть только у свежепосчитанной разметки (для записи в кэш);
    у прочитанной из кэша он ``None``.
    """

    layout: PageLayout
    dpi: int = 0
    raw: object | None = None


class _Stub:
    """Заглушка вместо класса чужого пакета: принимает любое состояние и ничего не делает."""

    def __new__(cls, *args, **kwargs):  # noqa: ARG003 — сигнатура задаётся pickle
        return object.__new__(cls)

    def __setstate__(self, state) -> None:
        self.state = state


class _LenientUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.split(".")[0] in _TRUSTED_MODULES:
            return super().find_class(module, name)
        return _Stub


def cache_path(cache_dir: Path, rel_path: str) -> Path:
    """Файл кэша полосы: путь полосы внутри пака с расширением ``.pkl``."""
    return Path(cache_dir) / Path(rel_path).with_suffix(".pkl")


def load(cache_dir: "Path | None", rel_path: str) -> "CachedLayout | None":
    """Разметка полосы из кэша или ``None`` — «нет, битый или чужой; считать заново»."""
    if cache_dir is None:
        return None
    path = cache_path(cache_dir, rel_path)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            payload = _LenientUnpickler(handle).load()
        if not isinstance(payload, dict):
            raise ValueError(f"в файле не словарь, а {type(payload).__name__}")
        if payload.get("scan_rel_path") != rel_path:
            raise ValueError(f"внутри записан путь {payload.get('scan_rel_path')!r}")
        layout = PageLayout.from_json(payload["layout"])
        if layout.width <= 0 or layout.height <= 0:
            raise ValueError(f"размер картинки {layout.width}x{layout.height}")
        return CachedLayout(layout, int(payload.get("dpi") or 0))
    except Exception as error:  # noqa: BLE001 — промах кэша, а не ошибка прогона
        logger.warning(
            "%s: кэш разметки не читается (%s: %s), полоса будет разобрана заново", path, type(error).__name__, error
        )
        return None


def save(cache_dir: Path, rel_path: str, cached: CachedLayout) -> Path:
    """Записать разметку атомарно: ``.part`` и ``replace``, чтобы прерванный прогон не оставил обрезка."""
    path = cache_path(cache_dir, rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scan_rel_path": rel_path, "dpi": cached.dpi, "layout": cached.layout.to_json(), "surya": cached.raw}
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def scaled_to(cached: CachedLayout, width: int, height: int) -> PageLayout:
    """Разметка в пикселях картинки ``width`` x ``height`` (копия 1/4 может отличаться на пиксель)."""
    layout = cached.layout
    if (layout.width, layout.height) == (width, height):
        return layout
    return layout.scaled(width / max(1, layout.width))


def picture_boxes(cached: CachedLayout, width: int, height: int, labels=PICTURE_LABELS) -> list[Box]:
    """Рамки блоков-иллюстраций в пикселях картинки ``width`` x ``height``."""
    layout = scaled_to(cached, width, height)
    return [block.box.clipped(width, height) for block in layout.blocks if block.label in labels]
