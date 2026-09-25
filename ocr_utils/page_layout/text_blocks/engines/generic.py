"""Общий адаптер чужих сегментаторов строк с единым JSON воркера (Orli, PaddleOCR, Surya, CRAFT и др.)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks.engines.base import EngineLine, EngineResult
from ocr_utils.page_layout.text_blocks.engines.external import centre_from_polygon, polygon_height, polyline, run_worker


@dataclass(frozen=True)
class WorkerSpec:
    """Как запускать воркер одного движка: интерпретатор его окружения, файл воркера и аргументы.

    Attributes:
        name: Имя движка в CLI и в именах файлов прогона.
        python: Интерпретатор окружения движка.
        worker: Имя файла воркера в ``engines/workers/``.
        extra: Дополнительные аргументы воркера после ``<png> <out.json>`` (пути моделей и т. п.).
        timeout: Предел времени на страницу, секунды.
    """

    name: str
    python: Path
    worker: str
    extra: tuple[str, ...] = field(default_factory=tuple)
    timeout: int = 900


def line_of(item: dict, scale: float) -> EngineLine | None:
    """Строка движка из записи единого JSON воркера.

    Что считать осью, решается по тому, что движок дал сам, в порядке доверия:
    готовая центр-линия (``centre``) → базовая линия (``baseline``, поднимается на полвысоты в
    ``lines.axis_of``) → середина залитого полигона (``boundary``). Высота — из ``height`` воркера,
    иначе из полигона.

    Args:
        item: Запись строки: ключи ``centre``, ``baseline``, ``boundary`` (списки ``[x, y]`` в пикселях
            рендера 300 dpi), ``height`` (в тех же пикселях) и ``confidence`` — все необязательны.
        scale: Множитель перевода пикселей рендера в пиксели рабочей копии.

    Returns:
        :class:`EngineLine` в пикселях рабочей копии или ``None``, если ни одной годной ломаной нет.
    """
    # Полигон строки нужен и для оси (если другого нет), и для высоты.
    boundary = polyline(item["boundary"], scale) if len(item.get("boundary") or []) >= 3 else None
    height = float(item.get("height") or 0.0) * scale
    if height <= 0.0 and boundary is not None:
        height = polygon_height(boundary)
    confidence = float(item["confidence"]) if item.get("confidence") is not None else float("nan")
    # Готовая центр-линия движка — ось как есть.
    if len(item.get("centre") or []) >= 2:
        points, is_baseline = polyline(item["centre"], scale), False
    # Базовая линия: ось получится подъёмом на полвысоты.
    elif len(item.get("baseline") or []) >= 2:
        points, is_baseline = polyline(item["baseline"], scale), True
    # Только контур: ось — середина его толщины по столбцам.
    elif boundary is not None:
        points, is_baseline = centre_from_polygon(boundary), False
    else:
        return None
    if len(points) < 2:
        return None
    return EngineLine(points=points, polygon=boundary, height=height, baseline=is_baseline, confidence=confidence)


def fill_heights(lines: list[EngineLine]) -> list[EngineLine]:
    """Подставить высоту строкам, у которых её нет, медианой по странице.

    Базовая линия без полигона и без высоты (Laypa) иначе не поднялась бы к центру строки.

    Args:
        lines: Строки страницы.

    Returns:
        Те же строки; нулевая высота заменена медианой известных высот, если они есть.
    """
    known = [line.height for line in lines if line.height > 0.0]
    if not known:
        return lines
    median = float(np.median(known))
    return [
        (
            line
            if line.height > 0.0
            else EngineLine(line.points, line.polygon, median, line.baseline, line.confidence, line.mark_spans)
        )
        for line in lines
    ]


class WorkerEngine:
    """Поставщик строк через воркер в чужом окружении с единым JSON (см. :func:`line_of`)."""

    def __init__(self, spec: WorkerSpec) -> None:
        """Args:
        spec: Как запускать воркер движка.
        """
        self.spec = spec
        self.name = spec.name

    def segment(self, gray300: np.ndarray, dpi: float = WORK_DPI) -> EngineResult:
        """Строки страницы в пикселях рабочей копии ``dpi``.

        Args:
            gray300: Серый рендер страницы 300 dpi.
            dpi: Разрешение рабочей копии.

        Returns:
            :class:`EngineResult` со строками и регионами движка; в ``note`` — число строк и
            что дал воркер в ``meta`` (модель, устройство).
        """
        started = time.monotonic()
        data = run_worker(self.spec.python, self.spec.worker, gray300, list(self.spec.extra), self.spec.timeout)
        scale = dpi / RENDER_DPI
        # Строки без годной ломаной отбрасываются здесь же.
        lines = fill_heights([line for line in (line_of(item, scale) for item in data["lines"]) if line is not None])
        regions = [polyline(region, scale) for region in data.get("regions", []) if len(region) >= 3]
        meta = data.get("meta") or {}
        return EngineResult(
            lines=lines,
            regions=regions,
            engine=self.name,
            seconds=time.monotonic() - started,
            note=f"строк {len(lines)}; {meta.get('model', '')} {meta.get('device', '')}".strip(),
        )


__all__ = ["WorkerEngine", "WorkerSpec", "fill_heights", "line_of"]
