"""Настройки режима «фокус по строкам» одним объектом.

Вынесено отдельно, чтобы ``analysis.analyze_file`` не пришлось раздувать пятью
дополнительными позиционными аргументами: набор целиком уезжает в процесс-воркер, а
значит должен быть простым и пиклящимся.
"""

from dataclasses import dataclass
from pathlib import Path

from ocr_utils.defocus_detection.lines.measure import DEFAULT_HEIGHT_CORRIDOR
from ocr_utils.defocus_detection.lines.regions import DEFAULT_CHUNK_ASPECT
from ocr_utils.defocus_detection.lines.zonal_tiles import DEFAULT_MIN_LINES, DEFAULT_TILE_SIDE


@dataclass(frozen=True)
class LineOptions:
    """Как мерить резкость по областям строк.

    Attributes:
        n_tiles: Сторона зональной сетки (3 — сетка 3x3).
        height_corridor: Коридор перцентилей высоты строки внутри тайла — «мелкий текст».
        chunk_aspect: Ширина куска строки в её высотах.
        min_lines: Минимум измеренных строк в тайле зональной сетки.
        debug_dir: Куда складывать отладочные наложения; None — не складывать.
    """

    n_tiles: int = DEFAULT_TILE_SIDE
    height_corridor: tuple[float, float] = DEFAULT_HEIGHT_CORRIDOR
    chunk_aspect: float = DEFAULT_CHUNK_ASPECT
    min_lines: int = DEFAULT_MIN_LINES
    debug_dir: Path | None = None
