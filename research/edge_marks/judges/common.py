"""Общий формат выхода судей: строка JSONL на кандидата и масштаб, плюс служебная строка со временем загрузки модели."""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path


class Verdict(str, Enum):
    """Вердикт судьи по кандидату."""

    JUNK = "junk"  # не символ: сор, пометка
    SIGN = "sign"  # настоящий знак
    UNSURE = "unsure"


class Scale(str, Enum):
    """На чём судил судья."""

    COMPONENT = "a"  # вырезка «кандидат + 1.5 x_h»
    LINE = "b"  # строка целиком
    PAGE = "c"  # полоса целиком


class Writer:
    """Запись результатов судьи в JSONL: ``{"id", "scale", "verdict", "score", "detail", "seconds"}``.

    Первая строка — служебная ``{"meta": {"load_seconds": …}}``, её пишет :meth:`loaded`.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self.started = time.monotonic()

    def loaded(self, **meta) -> None:
        """Отметить конец загрузки модели (и прочие сведения о прогоне)."""
        self.handle.write(json.dumps({"meta": {"load_seconds": time.monotonic() - self.started, **meta}}) + "\n")
        self.handle.flush()

    def write(self, cid: str, scale: Scale, verdict: Verdict, score: float, seconds: float, detail: str = "") -> None:
        """Строка результата по кандидату ``cid`` на масштабе ``scale``."""
        record = {"id": cid, "scale": Scale(scale).value, "verdict": Verdict(verdict).value, "score": float(score)}
        record.update({"seconds": round(float(seconds), 4), "detail": detail})
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def overlap_share(box: tuple, other: tuple) -> float:
    """Доля площади ``box``, накрытая ``other`` (рамки ``x0, y0, x1, y1``)."""
    w = min(box[2], other[2]) - max(box[0], other[0])
    h = min(box[3], other[3]) - max(box[1], other[1])
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return max(0.0, w) * max(0.0, h) / area


__all__ = ["Scale", "Verdict", "Writer", "overlap_share"]
