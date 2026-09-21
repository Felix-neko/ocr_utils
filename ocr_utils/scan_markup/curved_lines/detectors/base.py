"""Общий контракт детекторов кривизны строк.

Детектор получает подготовленный кадр (CPU) или пачку картинок (GPU) и отдаёт
:class:`Measure` — набор непрерывных метрик. Флаг и score детектор НЕ ставит: это делает
``flags.Thresholds`` по порогам из командной строки, чтобы перекалибровка не требовала
пересчёта. Всё, что знает отчёт, — имя детектора и его метрики.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from ocr_utils.scan_markup.curved_lines import CURVED_LINES_VERSION
from ocr_utils.page_layout.orientation.detectors.base import Frame  # noqa: F401 (реэкспорт: тот же кадр)


@dataclass(frozen=True)
class Measure:
    """Мнение одного детектора об одной полосе.

    ``metrics`` — непрерывные величины, ВСЕ в шкале «больше — кривее»; сравнивать их между
    детекторами напрямую нельзя, только внутри одного, для порога.
    ``score`` — max(metric / threshold) по флаговым метрикам; ``flag`` ⇔ score ≥ 1. Ставятся
    снаружи, в ``flags.Thresholds.apply``.
    ``silent`` — мерить было не по чему (мало строк, ошибка движка); такая полоса не
    флагуется и в статистику не входит, а причина лежит в ``note``.
    ``raw`` — сырые измерения (тайлы с углами, центр-линии строк) для оверлея и кэша; в CSV
    не идёт, в памяти прогона по паку не держится.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    note: str = ""
    flag: bool = False
    score: float = 0.0
    silent: bool = False
    raw: dict | None = None

    def stripped(self) -> "Measure":
        """Та же мера без сырых данных — то, что остаётся в памяти на прогоне по паку."""
        return self if self.raw is None else Measure(self.metrics, self.note, self.flag, self.score, self.silent, None)


def silent(note: str) -> Measure:
    """Мера «сказать нечего» с причиной словами."""
    return Measure(note=note, silent=True)


@dataclass(frozen=True)
class GpuPage:
    """Полоса на GPU-этап: картинка уже уменьшена воркером и разжата в родителе."""

    path: Path
    rel_path: str
    image: object  # PIL.Image.Image


class BatchDetector(Protocol):
    """GPU-детектор: модель грузится лениво, вход — пачка полос."""

    def __call__(self, pages: Sequence[GpuPage], keep_raw: bool) -> list[Measure]: ...


def _always() -> bool:
    return True


@dataclass(frozen=True)
class Detector:
    """Запись в реестре детекторов.

    ``thresholds`` — флаговые метрики и их пороги по умолчанию; остальные метрики детектора
    в CSV тоже попадают, но флага не ставят.
    ``version`` — поднимать при любой правке алгоритма или агрегации: она входит в ключ
    кэша, и старые измерения перестают подхватываться.
    ``draw`` — рисует сырые данные на оверлее 150 dpi: ``draw(canvas_bgr, raw, scale)``,
    где ``scale`` переводит координаты ``raw`` в пиксели холста.
    ``solo`` — может ли детектор в одиночку, одним «сильным» голосом, дать сводный флаг.
    У детектора с одиночными выбросами на ровном тексте (``strip_shift``) — нет: его голос
    учитывается только вместе с чужим.
    ``sufficient`` — обычного флага этого детектора ДОСТАТОЧНО для сводного, без голосов
    и без «сильного» score. Так подключается вспомогательный детектор, который ловит то,
    чего остальные не видят вовсе (``end_curl``): требовать от него согласия с другими
    значило бы отменить сам смысл его добавления.
    """

    name: str
    summary: str
    stage: str  # "cpu" | "gpu"
    thresholds: dict[str, float]
    version: int = 1
    run: Callable[[Frame, bool], Measure] | None = None
    make_batch: Callable[..., BatchDetector] | None = None
    draw: Callable[[np.ndarray, dict, float], None] | None = None
    available: Callable[[], bool] = _always
    solo: bool = True
    sufficient: bool = False

    def __post_init__(self) -> None:
        if self.stage not in ("cpu", "gpu"):
            raise ValueError(f"stage={self.stage!r}, а бывает 'cpu' или 'gpu'")
        if self.stage == "cpu" and self.run is None:
            raise ValueError(f"{self.name}: CPU-детектору нужен run")
        if self.stage == "gpu" and self.make_batch is None:
            raise ValueError(f"{self.name}: GPU-детектору нужен make_batch")

    @property
    def cache_key(self) -> str:
        return f"v{CURVED_LINES_VERSION}.{self.version}"
