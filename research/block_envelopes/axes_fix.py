"""Замена осей-выбросов крупного набора на входе блоковой стадии стенда: ядро — ``ocr_utils.page_layout.text_blocks.axes_fix``, здесь — применение к кэшу входа."""

from __future__ import annotations

from ocr_utils.page_layout.text_blocks.axes_fix import AxisFix, body_reference, fixed_axes, slope_of  # noqa: F401
from research.block_envelopes.capture import BlockInput


def fixed_input(inp: BlockInput) -> tuple[BlockInput, list[AxisFix]]:
    """Вход блоковой стадии с заменёнными осями-выбросами (:func:`fixed_axes`).

    Args:
        inp: Вход полосы.

    Returns:
        ``(новый вход, протокол)``.
    """
    axes, log = fixed_axes(inp.axes, inp.dpi)
    if not log:
        return inp, log
    return BlockInput(**{**inp.__dict__, "axes": axes}), log


__all__ = ["AxisFix", "body_reference", "fixed_axes", "fixed_input", "slope_of"]
