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
    PADDLE6 = "paddle6"
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
    extra: tuple[str, ...] = ()
    timeout = 900
    system_python: Path | None = None
    # Окружение и воркер обычно названы по движку; PP-OCRv6 живёт в окружении и воркере PaddleOCR.
    home = name.value
    if name is WorkerEngineName.ORLI:
        # Orli отдаёт одни базовые линии без высоты; полигоны строк (kraken calculate_polygonal_environment,
        # ~24 с на полосу) нужны ради высоты — без неё ось не поднять к центру строки. Модель — по умолчанию.
        extra = ("--polygonize",)
    elif name is WorkerEngineName.PADDLE:
        # Кэш весов PaddleX (``MODELS/paddle``) воркер задаёт сам; полная страница — предел стороны 4000.
        extra = ("--model", "PP-OCRv5_server_det")
    elif name is WorkerEngineName.PADDLE6:
        home = WorkerEngineName.PADDLE.value
        extra = ("--model", "PP-OCRv6_medium_det")
    elif name is WorkerEngineName.CHRONICLING:
        # Веса вёрстки и базовых линий воркер берёт по умолчанию из MODELS/chronicling.
        extra = ()
    elif name is WorkerEngineName.LAYPA:
        # Laypa идёт в docker (loghi/docker.laypa + loghi-tooling), на CPU: Rancher Desktop GPU не
        # пробрасывает. Воркер на одной стандартной библиотеке — хватает системного python3.
        extra = ()
        system_python = Path("/usr/bin/python3")
        timeout = 1800
    elif name is WorkerEngineName.CRAFT:
        # Веса и исходники CRAFT воркер берёт по умолчанию из LINE_ENGINES_ROOT; fp16 — вдвое меньше видеопамяти
        # при тех же строках (577 слов на 1973/08 с.85 в обоих режимах).
        extra = ("--fp16",)
    elif name is WorkerEngineName.DOCUFCN:
        # generic-historical-line на родном входе 768 px: крупнее — строки слипаются или дробятся на слова.
        extra = ()
    elif name is WorkerEngineName.TEXTSNAKE:
        # CTW1500; длинная сторона 2000 вместо родных ~1150: слипаний через межколонник на 1973/08 12 → 2.
        extra = ("--long-side", "2000")
        timeout = 1800
    # Surya стоит в основном окружении проекта: воркер зовётся тем же питоном отдельным процессом.
    default_python = Path(sys.executable) if name is WorkerEngineName.SURYA else _venv_python(home)
    default_python = system_python or default_python
    return WorkerSpec(
        name=name.value,
        python=Path(python) if python else default_python,
        worker=f"{home}_worker.py",
        extra=extra,
        timeout=timeout,
    )


__all__ = ["LINE_ENGINES_ROOT", "MODELS", "WorkerEngineName", "spec_of"]
