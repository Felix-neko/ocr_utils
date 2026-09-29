"""Карты детекторов символов для защиты сторон блоков (CRAFT, pero ParseNet): расчёт воркерами в их окружениях одним GPU-процессом на детектор, кэш PNG по ключу полосы."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks.edge_guard import GlyphEngine
from ocr_utils.page_layout.text_blocks.engines.catalog import LINE_ENGINES_ROOT
from ocr_utils.page_layout.text_blocks.engines.external import ENGINES_ROOT, WORKERS

# Интерпретаторы окружений и модель pero по умолчанию (те же, что у движков строк ``craft`` и ``pero``).
CRAFT_PYTHON = LINE_ENGINES_ROOT / "craft" / "bin" / "python"
PERO_PYTHON = ENGINES_ROOT / "pero" / "bin" / "python"
PERO_MODEL_DIR = ENGINES_ROOT / "pero_model" / "pero_eu_cz_print_newspapers_2022-09-26"
# Детекторы второго прохода по умолчанию: голосование CRAFT + pero (reports/edge_guard.md).
DEFAULT_ENGINES = (GlyphEngine.CRAFT, GlyphEngine.PERO)
# Предел времени одного воркера на пачку полос, с.
TIMEOUT_S = 3600


@dataclass(frozen=True)
class MapJob:
    """Полоса для карт: ключ кэша (имя файла карты) и серый рендер ``RENDER_DPI`` — массивом или готовым PNG."""

    key: str
    gray300: np.ndarray | None = None
    png: Path | None = None


def _command(engine: GlyphEngine, jobs_file: Path, out_dir: Path) -> list[str]:
    """Команда воркера детектора.

    Args:
        engine: Детектор.
        jobs_file: JSON ``[{"key", "png"}]``.
        out_dir: Папка карт детектора.

    Returns:
        Аргументы подпроцесса.
    """
    if engine is GlyphEngine.CRAFT:
        return [str(CRAFT_PYTHON), str(WORKERS / "craft_maps_worker.py"), "--jobs", str(jobs_file), "--out-dir",
                str(out_dir), "--fp16"]  # fmt: skip
    return [str(PERO_PYTHON), str(WORKERS / "pero_maps_worker.py"), "--jobs", str(jobs_file), "--out-dir", str(out_dir),
            "--model-dir", str(PERO_MODEL_DIR)]  # fmt: skip


def compute_maps(jobs: list[MapJob], maps_dir: Path, engines: tuple[GlyphEngine, ...] = DEFAULT_ENGINES) -> None:
    """Досчитать карты детекторов для полос, которых нет в кэше ``<maps_dir>/<детектор>/<ключ>.png``.

    Детекторы идут строго по очереди — видеопамять одна; каждый — одним процессом на все полосы (модель грузится
    один раз). Полосы передаются воркерам PNG во временной папке.

    Args:
        jobs: Полосы.
        maps_dir: Корень кэша карт.
        engines: Детекторы.

    Raises:
        RuntimeError: Воркер завершился с ошибкой.
    """
    with tempfile.TemporaryDirectory(prefix="glyph_maps_") as folder:
        written: dict[str, str] = {}
        for engine in engines:
            todo = [job for job in jobs if not (maps_dir / engine.value / f"{job.key}.png").exists()]
            if not todo:
                continue
            for job in todo:
                if job.key in written:
                    continue
                if job.png is not None:
                    written[job.key] = str(job.png)
                else:
                    path = Path(folder) / f"{job.key}.png"
                    cv2.imwrite(str(path), job.gray300)
                    written[job.key] = str(path)
            jobs_file = Path(folder) / f"{engine.value}.json"
            jobs_file.write_text(json.dumps([{"key": job.key, "png": written[job.key]} for job in todo]))
            result = subprocess.run(_command(engine, jobs_file, maps_dir / engine.value), capture_output=True,
                                    text=True, timeout=TIMEOUT_S)  # fmt: skip
            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
                raise RuntimeError(f"карты {engine.value}: код {result.returncode}; " + " | ".join(tail))


def load_maps(
    maps_dir: Path, key: str, engines: tuple[GlyphEngine, ...] = DEFAULT_ENGINES
) -> dict[GlyphEngine, np.ndarray] | None:
    """Карты полосы из кэша, 0…1; ``None``, если хотя бы одной нет.

    Args:
        maps_dir: Корень кэша.
        key: Ключ полосы.
        engines: Детекторы.

    Returns:
        ``детектор → карта`` или ``None``.
    """
    out = {}
    for engine in engines:
        path = maps_dir / engine.value / f"{key}.png"
        if not path.exists():
            return None
        out[engine] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return out


__all__ = ["CRAFT_PYTHON", "DEFAULT_ENGINES", "MapJob", "PERO_MODEL_DIR", "PERO_PYTHON", "compute_maps", "load_maps"]
