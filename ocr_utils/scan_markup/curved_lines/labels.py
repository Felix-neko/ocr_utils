"""Разметка «кривая / прямая» и таблица разделения по метрикам.

Файл меток — CSV ``rel_path,label[,note]``; путь без расширения, потому что меряются
заострённые ``.jpg``, а размечались оригиналы ``.tif``. Метки нужны для двух вещей:
раздела «Разметка» в отчёте (какая полоса как измерилась) и таблицы разделения — по
каждой метрике видно, разводит ли она классы вообще и где лежит зазор.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

CURVED = "curved"
STRAIGHT = "straight"
LABELS = (CURVED, STRAIGHT)


@dataclass(frozen=True)
class Label:
    label: str
    note: str = ""


def _key(rel_path: str) -> str:
    return str(Path(rel_path).with_suffix("")).replace("\\", "/")


def load_labels(path: Path) -> dict[str, Label]:
    """Метки по ключу «путь без расширения». Строки с ``#`` и пустые пропускаются."""
    labels: dict[str, Label] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if row[0].strip() == "rel_path":
                continue  # заголовок
            if len(row) < 2 or row[1].strip() not in LABELS:
                raise ValueError(f"метки: ожидалось «путь,{'|'.join(LABELS)}[,заметка]», получено {row!r}")
            labels[_key(row[0].strip())] = Label(row[1].strip(), row[2].strip() if len(row) > 2 else "")
    return labels


def lookup(labels: dict[str, Label], rel_path: str) -> Label | None:
    return labels.get(_key(rel_path))


@dataclass(frozen=True)
class Separation:
    """Насколько одна метрика разводит классы."""

    detector: str
    metric: str
    threshold: float | None
    curved: np.ndarray
    straight: np.ndarray

    @property
    def gap(self) -> float:
        """min по кривым минус max по прямым: положительный — разделяет чисто."""
        if self.curved.size == 0 or self.straight.size == 0:
            return float("nan")
        return float(self.curved.min() - self.straight.max())

    @property
    def suggested(self) -> float:
        """Середина зазора; при перекрытии — середина между медианами."""
        if self.curved.size == 0 or self.straight.size == 0:
            return float("nan")
        if self.gap > 0:
            return float((self.curved.min() + self.straight.max()) / 2.0)
        return float((np.median(self.curved) + np.median(self.straight)) / 2.0)

    def confusion(self) -> tuple[int, int, int, int]:
        """(TP, FN, FP, TN) при текущем пороге; без порога — нули."""
        if self.threshold is None:
            return 0, 0, 0, 0
        tp = int((self.curved >= self.threshold).sum())
        fp = int((self.straight >= self.threshold).sum())
        return tp, int(self.curved.size - tp), fp, int(self.straight.size - fp)


def separation(
    rows: Sequence[tuple[str, dict[str, dict[str, float]]]],
    thresholds: dict[str, dict[str, float]],
    detector_names: Sequence[str],
) -> list[Separation]:
    """По каждой (детектор, метрика) — значения у кривых и у прямых.

    ``rows`` — ``(label, {детектор: {метрика: значение}})`` по размеченным полосам.
    """
    result: list[Separation] = []
    for name in detector_names:
        metrics: list[str] = []
        for _, per_detector in rows:
            for metric in per_detector.get(name, {}):
                if metric not in metrics:
                    metrics.append(metric)
        for metric in metrics:
            curved = [r[name][metric] for label, r in rows if label == CURVED and metric in r.get(name, {})]
            straight = [r[name][metric] for label, r in rows if label == STRAIGHT and metric in r.get(name, {})]
            result.append(
                Separation(
                    name,
                    metric,
                    thresholds.get(name, {}).get(metric),
                    np.array(curved, dtype=np.float64),
                    np.array(straight, dtype=np.float64),
                )
            )
    return result


def separation_table(items: Sequence[Separation]) -> str:
    def fmt(value: float) -> str:
        return "—" if value != value else f"{value:.3f}"

    lines = [
        "| детектор | метрика | порог | кривые: min / p50 / max | прямые: min / p50 / max | зазор | предложить | TP/FN/FP/TN |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in items:
        c, s = item.curved, item.straight
        c_text = f"{fmt(c.min())} / {fmt(np.median(c))} / {fmt(c.max())}" if c.size else "—"
        s_text = f"{fmt(s.min())} / {fmt(np.median(s))} / {fmt(s.max())}" if s.size else "—"
        confusion = "/".join(str(v) for v in item.confusion()) if item.threshold is not None else "—"
        threshold = f"{item.threshold:g}" if item.threshold is not None else "—"
        lines.append(
            f"| {item.detector} | {item.metric} | {threshold} | {c_text} | {s_text} | {fmt(item.gap)} | "
            f"{fmt(item.suggested)} | {confusion} |"
        )
    return "\n".join(lines)
