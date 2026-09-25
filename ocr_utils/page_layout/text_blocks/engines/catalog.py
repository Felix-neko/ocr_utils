"""Каталог чужих движков строк с единым воркером: где их окружения и модели и с какими аргументами звать.

Окружения лежат не в репозитории, а в ``LINE_ENGINES_ROOT`` (корень ``/``, рядом с кэшем uv — пакеты
ставятся жёсткими ссылками и места почти не занимают); установка каждого — в README подпакета,
сравнение — ``reports/line_axis_models.md``.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

from ocr_utils.page_layout.text_blocks.engines.generic import WorkerSpec

# Корень окружений и моделей движков из сравнения 2026-09-25.
LINE_ENGINES_ROOT = Path.home() / "Projects" / "mts_markup" / "line_axis_engines"
MODELS = LINE_ENGINES_ROOT / "models"


class WorkerEngineName(str, Enum):
    """Движки, которые подключаются общим адаптером :class:`~.generic.WorkerEngine`."""

    SURYA = "surya"
    ORLI = "orli"
    PADDLE = "paddle"
    CHRONICLING = "chronicling"
    LAYPA = "laypa"
    CRAFT = "craft"
    DOCUFCN = "docufcn"
    TEXTSNAKE = "textsnake"


def _venv_python(name: str) -> Path:
    """Интерпретатор окружения движка ``name`` в ``LINE_ENGINES_ROOT``."""
    return LINE_ENGINES_ROOT / name / "bin" / "python"


def spec_of(name: WorkerEngineName, python: Path | None = None) -> WorkerSpec:
    """Как запускать движок ``name``.

    Args:
        name: Движок из каталога.
        python: Свой интерпретатор вместо окружения по умолчанию (``None`` — по умолчанию).

    Returns:
        :class:`WorkerSpec` с интерпретатором, файлом воркера, аргументами и пределом времени.
    """
    name = WorkerEngineName(name)
    # Surya стоит в основном окружении проекта: воркер зовётся тем же питоном отдельным процессом.
    default_python = Path(sys.executable) if name is WorkerEngineName.SURYA else _venv_python(name.value)
    extra: tuple[str, ...] = ()
    timeout = 900
    if name is WorkerEngineName.ORLI:
        extra = (str(MODELS / "orli"),)
    elif name is WorkerEngineName.PADDLE:
        extra = (str(MODELS / "paddle"),)
    elif name is WorkerEngineName.CHRONICLING:
        extra = (str(MODELS / "chronicling"),)
    elif name is WorkerEngineName.LAYPA:
        extra = (str(MODELS / "laypa"),)
        timeout = 1800
    elif name is WorkerEngineName.CRAFT:
        extra = (str(MODELS / "craft"),)
    elif name is WorkerEngineName.DOCUFCN:
        extra = (str(MODELS / "docufcn"),)
    elif name is WorkerEngineName.TEXTSNAKE:
        extra = (str(MODELS / "textsnake"),)
        timeout = 1800
    return WorkerSpec(
        name=name.value,
        python=Path(python) if python else default_python,
        worker=f"{name.value}_worker.py",
        extra=extra,
        timeout=timeout,
    )


__all__ = ["LINE_ENGINES_ROOT", "MODELS", "WorkerEngineName", "spec_of"]
