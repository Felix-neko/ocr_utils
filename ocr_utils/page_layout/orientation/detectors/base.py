"""Общий контракт детекторов ориентации.

Детекторы взаимозаменяемы: каждый получает подготовленный кадр (или пачку картинок, если
живёт на GPU) и отдаёт :class:`Verdict` в одной и той же шкале. Всё, что знает отчёт, —
это имя детектора и вердикт; как именно детектор считает, отчёту неизвестно.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

# Обозначения углов — из общего модуля: их делят с базой, CVAT и очисткой, и второй
# источник истины по соседству рано или поздно разошёлся бы с первым.
from ocr_utils.scan_markup.rotation import ROTATION_NAMES, ROTATIONS, rotate_cw  # noqa: E402,F401


@dataclass(frozen=True)
class Verdict:
    """Мнение одного детектора об одной полосе.

    ``rotate_cw`` — на сколько повернуть ПО ЧАСОВОЙ, чтобы стало прямо.
    ``confidence`` — приведённая к 0..1 уверенность; у разных детекторов она получается из
    разных величин (уверенность OSD, softmax сети, отрыв от второго места), и сравнивать её
    между детекторами напрямую нельзя — только внутри одного, для порога.
    ``axis_only`` — детектор различает лишь ОСЬ текста (книжная против боковой), а знак
    поворота выдумывать не берётся. Тогда ``rotate_cw`` равен 0 или 90, и 90 читается как
    «повёрнута, сторона неизвестна».
    """

    rotate_cw: int
    confidence: float
    axis_only: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.rotate_cw not in ROTATIONS:
            raise ValueError(f"rotate_cw={self.rotate_cw}, а бывает только {ROTATIONS}")

    @property
    def rotated(self) -> bool:
        return self.rotate_cw != 0


def unknown(note: str) -> Verdict:
    """Вердикт «сказать нечего»: нулевая уверенность и причина словами.

    Отдельная функция, а не константа: причина у каждого случая своя, и в отчёте она нужна
    («мало текста» и «tesseract не смог» — разные поводы посмотреть на полосу глазами).
    """
    return Verdict(rotate_cw=0, confidence=0.0, note=note)


@dataclass
class Frame:
    """Подготовленная полоса: один раз разжата, дальше её делят все CPU-детекторы.

    Два масштаба нужны обоим tesseract-детекторам: на разведке 300 dpi и 150 dpi ловили
    РАЗНЫЕ полосы из четырёх известных, и объединение масштабов дало все четыре.
    """

    rel_path: str
    path: Path
    width: int  # размеры исходника, не рабочей копии
    height: int
    dpi: int
    gray150: np.ndarray
    gray300: np.ndarray

    # Какие повороты вообще рассматривать. Сужать этот набор — самый дешёвый способ поднять
    # точность: неверный ответ, которого нет в наборе, не может быть дан в принципе. Для
    # пака-1 хватает (0, 90) — все 42 размеченные боковые полосы требуют поворота по часовой,
    # что и понятно: широкие рисунки в советских изданиях ставили одинаково. Заодно вдвое
    # дешевле обходится арбитр: два прогона распознавания вместо четырёх.
    allowed: tuple[int, ...] = ROTATIONS


class BatchDetector(Protocol):
    """GPU-детектор: модель грузится лениво, вход — пачка картинок."""

    def __call__(self, images: Sequence["object"]) -> list[Verdict]: ...


def _always() -> bool:
    return True


@dataclass(frozen=True)
class Detector:
    """Запись в реестре детекторов.

    ``stage`` разводит два способа исполнения, и это не украшательство: CPU-детекторы едут
    в пул процессов, а GPU-детекторы обязаны остаться в родителе — видеопамять одна на всех
    (см. CLAUDE.md и ``scan_markup/detection/run.py``).

    ``arbiter`` — детектор слишком дорог для всего пака и запускается только по кандидатам,
    которых отметили быстрые.
    """

    name: str
    summary: str
    stage: str  # "cpu" | "gpu"
    run: Callable[[Frame], Verdict] | None = None
    make_batch: Callable[[], BatchDetector] | None = None
    arbiter: bool = False
    available: Callable[[], bool] = _always

    def __post_init__(self) -> None:
        if self.stage not in ("cpu", "gpu"):
            raise ValueError(f"stage={self.stage!r}, а бывает 'cpu' или 'gpu'")
        if self.stage == "cpu" and self.run is None:
            raise ValueError(f"{self.name}: CPU-детектору нужен run")
        if self.stage == "gpu" and self.make_batch is None:
            raise ValueError(f"{self.name}: GPU-детектору нужен make_batch")
