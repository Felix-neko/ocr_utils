"""Источники стенда: боксы surya Equation из кэша page_layout, выносные формулы DeepSeek, заострённые JPEG, слои выборки."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

# Кадр кэша surya — 150 dpi (``page_layout.surya``): одна единица кадра — 25.4 / 150 мм.
FRAME_DPI = 150.0
MM_PER_UNIT = 25.4 / FRAME_DPI

# Выносная формула DeepSeek — ``$$…$$`` (промпт внешнего OCR: «LaTeX в долларах внутри <latex>»),
# строчная — одиночные ``$…$``. Строчные ищутся после вырезания выносных, иначе ``$$`` даёт пары ``$``.
DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)([^$]+?)\$(?!\$)")


class Stratum(str, Enum):
    """Слой выборки: что о полосе говорят surya и DeepSeek (эталон размечен по слоям, пак — сумма слоёв)."""

    BOTH = "both"  # есть и бокс Equation, и ``$$`` DeepSeek
    DEEPSEEK_ONLY = "deepseek_only"  # ``$$`` есть, боксов Equation нет — кандидаты в пропуски surya
    SURYA_ONLY = "surya_only"  # боксы Equation есть, ``$$`` нет — кандидаты в ложные
    INLINE_FRAC_ONLY = "inline_frac_only"  # ни того, ни другого, но у DeepSeek строчные ``\frac``
    INLINE_ONLY = "inline_only"  # только строчные ``$…$`` без дробей (в выборку не брался)
    NO_SIGNAL = "no_signal"  # ни одного признака формулы


@dataclass(frozen=True)
class EquationBox:
    """Бокс surya ``Equation`` в кадре кэша (150 dpi) и его уверенность."""

    box: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class PageSignals:
    """Признаки формул на одной полосе: боксы surya и счёт формул DeepSeek."""

    page: str  # «год/выпуск/полоса», как в кэше и в страницах внешнего OCR
    frame: tuple[int, int]  # ширина и высота кадра кэша
    equations: tuple[EquationBox, ...]
    display: int  # выносных ``$$…$$`` у DeepSeek
    inline: int  # строчных ``$…$``
    inline_frac: int  # строчных с ``\frac``

    @property
    def stratum(self) -> Stratum:
        """Слой выборки по признакам (см. :class:`Stratum`)."""
        if self.display and self.equations:
            return Stratum.BOTH
        if self.display:
            return Stratum.DEEPSEEK_ONLY
        if self.equations:
            return Stratum.SURYA_ONLY
        if self.inline_frac:
            return Stratum.INLINE_FRAC_ONLY
        if self.inline:
            return Stratum.INLINE_ONLY
        return Stratum.NO_SIGNAL


def count_deepseek(markdown: str) -> tuple[int, int, int]:
    """Счёт формул в markdown полосы от DeepSeek.

    Args:
        markdown: текст полосы (``pages/<год>/<выпуск>/<полоса>.md``).

    Returns:
        ``(выносных, строчных, строчных с \\frac)``.
    """
    display = DISPLAY_RE.findall(markdown)
    inline = INLINE_RE.findall(DISPLAY_RE.sub("", markdown))
    return len(display), len(inline), sum("\\frac" in formula for formula in inline)


def read_equations(cache_file: Path) -> tuple[tuple[int, int], tuple[EquationBox, ...]]:
    """Кадр и боксы ``Equation`` из JSON-кэша surya одного варианта.

    Args:
        cache_file: ``<кэш>/<вариант>/<полоса>.json`` (формат ``page_layout.surya.SuryaCache``).

    Returns:
        ``((ширина, высота) кадра, боксы Equation в порядке кэша)``.
    """
    payload = json.loads(cache_file.read_text())
    frame = (int(payload["frame"]["width"]), int(payload["frame"]["height"]))
    boxes = tuple(
        EquationBox(tuple(float(v) for v in block["box"]), float(block["confidence"]))
        for block in payload["blocks"]
        if block["label"] == "Equation"
    )
    return frame, boxes


def collect_page(layout_dir: Path, pages_dir: Path, page: str) -> PageSignals:
    """Признаки формул одной полосы.

    Args:
        layout_dir: вариант кэша surya (``LAYOUT_CACHE_DIR/sharpened``).
        pages_dir: страницы внешнего OCR (``EXTERNAL_OCR_SERVICES_PAGES``).
        page: «год/выпуск/полоса».

    Returns:
        :class:`PageSignals`; полоса без markdown считается полосой без формул DeepSeek.
    """
    frame, equations = read_equations(layout_dir / f"{page}.json")
    markdown_file = pages_dir / f"{page}.md"
    counts = count_deepseek(markdown_file.read_text()) if markdown_file.exists() else (0, 0, 0)
    return PageSignals(page, frame, equations, *counts)


def load_gray(sharpened_dir: Path, page: str) -> np.ndarray:
    """Серый заострённый JPEG полосы (600 dpi для пака-1) — та же картинка, что видели surya и DeepSeek.

    Args:
        sharpened_dir: ``SHARPENED_DIR``.
        page: «год/выпуск/полоса».

    Returns:
        Массив ``uint8`` высота × ширина.
    """
    return np.asarray(Image.open(sharpened_dir / f"{page}.jpg").convert("L"))
