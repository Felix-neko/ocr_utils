"""Перевод сырых ответов судей в вердикты: разностные (текст с кандидатом и без), рамки символов/слов (накрывают ли кандидата) и оценки (вероятность символа)."""

from __future__ import annotations

from research.edge_marks.judges.common import Verdict, overlap_share
from research.text_block_specks.judge import Verdict as DiffVerdict
from research.text_block_specks.judge import diff_verdict

# Рамка символа/слова накрывает кандидата не меньше чем на эту долю — кандидат признан частью текста.
COVER_SHARE = 0.5
# Оценка «это символ» (0…1): не ниже — знак, не выше нижнего порога — сор, между — неясно.
SIGN_SCORE, JUNK_SCORE = 0.5, 0.2


def from_diff(with_text: str, erased: str, side: str) -> tuple[Verdict, float, str]:
    """Разностный вердикт: текст не изменился — сор; у края добавился знак — знак (``text_block_specks.judge.diff_verdict``).

    Returns:
        ``(вердикт, оценка «сор» 0…1, пояснение)``.
    """
    verdict, extra = diff_verdict(with_text, erased, side)
    if verdict is DiffVerdict.NOISE:
        return Verdict.JUNK, 1.0, ""
    if verdict is DiffVerdict.SIGN:
        return Verdict.SIGN, 0.0, extra
    return Verdict.UNSURE, 0.5, f"{verdict.value}:{extra}"


def from_boxes(box: tuple, boxes: list) -> tuple[Verdict, float, str]:
    """Вердикт по рамкам символов или слов движка: кандидат накрыт рамкой — знак, иначе сор.

    Args:
        box: Рамка кандидата.
        boxes: Рамки движка ``[x0, y0, x1, y1, …]`` в тех же пикселях.

    Returns:
        ``(вердикт, оценка «сор», пояснение)``.
    """
    best = max((overlap_share(box, tuple(item[:4])) for item in boxes), default=0.0)
    if best >= COVER_SHARE:
        return Verdict.SIGN, 1.0 - best, f"накрыт {best:.2f}"
    return Verdict.JUNK, 1.0 - best, f"накрыт {best:.2f}"


def from_score(score: float) -> tuple[Verdict, float, str]:
    """Вердикт по оценке «это символ» 0…1."""
    if score >= SIGN_SCORE:
        return Verdict.SIGN, 1.0 - score, f"{score:.2f}"
    if score <= JUNK_SCORE:
        return Verdict.JUNK, 1.0 - score, f"{score:.2f}"
    return Verdict.UNSURE, 1.0 - score, f"{score:.2f}"


__all__ = ["from_boxes", "from_diff", "from_score"]
