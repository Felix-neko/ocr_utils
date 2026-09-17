"""CSV с метриками по страницам и markdown-сводка прогона."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from research.geometry_regression.scoring import DEFAULT_THRESHOLDS, Thresholds, Verdict

FIXED_FIELDS = ("pdf", "year", "page", "score", "reason", "flags", "error")

# Пояса score для сводки и выборки глазами.
BELTS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.3), (1.3, 2.0), (2.0, float("inf")))


@dataclass
class PageRow:
    """Одна страница в CSV: метрики плюс вердикт."""

    pdf: str
    page: int  # с единицы, как в PDF-читалке
    metrics: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    reason: str = ""
    flags: dict[str, float] = field(default_factory=dict)
    error: str = ""

    @property
    def year(self) -> str:
        parts = self.pdf.split("_")
        return parts[1] if len(parts) > 1 else self.pdf

    @property
    def key(self) -> tuple[str, int]:
        return (self.pdf, self.page)

    def apply(self, verdict: Verdict) -> None:
        self.score, self.reason, self.flags = verdict.score, verdict.reason, dict(verdict.flags)


def metric_names(rows: list[PageRow]) -> list[str]:
    names: dict[str, None] = {}
    for row in rows:
        for name in row.metrics:
            names.setdefault(name, None)
    return list(names)


def write_csv(path: Path, rows: list[PageRow]) -> None:
    names = metric_names(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*FIXED_FIELDS, *names])
        for row in sorted(rows, key=lambda r: r.key):
            flags = ";".join(f"{name}:{value:.2f}" for name, value in row.flags.items())
            writer.writerow(
                [row.pdf, row.year, row.page, f"{row.score:.3f}", row.reason, flags, row.error]
                + [_fmt(row.metrics.get(name)) for name in names]
            )


def _fmt(value) -> str:
    if value is None:
        return ""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def read_csv(path: Path) -> list[PageRow]:
    rows: list[PageRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            metrics = {
                name: float(value)
                for name, value in record.items()
                if name not in FIXED_FIELDS and value not in (None, "")
            }
            flags = {}
            for item in filter(None, record.get("flags", "").split(";")):
                name, _, value = item.partition(":")
                flags[name] = float(value)
            rows.append(
                PageRow(
                    record["pdf"],
                    int(record["page"]),
                    metrics,
                    float(record["score"] or 0.0),
                    record.get("reason", ""),
                    flags,
                    record.get("error", ""),
                )
            )
    return rows


def reflag(rows: list[PageRow], thresholds: Thresholds) -> None:
    for row in rows:
        if not row.error:
            row.apply(thresholds.apply(row.metrics))


def load_labels(path: Path) -> dict[tuple[str, int], tuple[str, str]]:
    """Эталон ``pdf,page,label,note`` → {(pdf, page): (label, note)}."""
    labels = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            labels[(record["pdf"], int(record["page"]))] = (record["label"], record.get("note", ""))
    return labels


def pair_name(row: PageRow) -> str:
    return f"{row.pdf}_p{row.page:03d}_s{row.score:.2f}_{row.reason or 'none'}.jpg"


def markdown_report(
    rows: list[PageRow], thresholds: Thresholds, labels: dict | None, out_dir: Path, top: int = 25
) -> str:
    good = [r for r in rows if not r.error]
    flagged = [r for r in good if r.score >= 1.0]
    lines = [
        "# Страницы, где коррекция геометрии FineReader сделала хуже",
        "",
        f"Страниц: {len(rows)}, с ошибкой: {len(rows) - len(good)}, флаг (score ≥ 1): {len(flagged)}"
        f" ({100.0 * len(flagged) / max(1, len(good)):.1f} %). Выход: `{out_dir}`.",
        "",
        "## Пороги",
        "",
        "| метрика | порог | причина |",
        "|---|---|---|",
        *(f"| `{name}` | {thresholds.values[name]:g} | {DEFAULT_THRESHOLDS[name][1]} |" for name in thresholds.values),
        "",
        "## Пояса score",
        "",
        "| пояс | страниц | доля |",
        "|---|---|---|",
    ]
    for low, high in BELTS:
        count = sum(1 for r in good if low <= r.score < high)
        label = f"≥ {low:g}" if high == float("inf") else f"{low:g}–{high:g}"
        lines.append(f"| {label} | {count} | {100.0 * count / max(1, len(good)):.1f} % |")
    lines += ["", "## Причины флага", "", "| причина | страниц |", "|---|---|"]
    for reason, count in Counter(r.reason for r in flagged).most_common():
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "## Распределение флаговых метрик по паку",
        "",
        "| метрика | p50 | p90 | p95 | p97 | p99 | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in thresholds.values:
        values = np.array([r.metrics.get(name, 0.0) for r in good], dtype=np.float64)
        if values.size:
            q = np.percentile(values, [50, 90, 95, 97, 99])
            lines.append(f"| `{name}` | " + " | ".join(f"{v:.3f}" for v in q) + f" | {values.max():.3f} |")
    lines += ["", "## Флаги по годам", "", "| год | страниц | флагов | доля |", "|---|---|---|---|"]
    by_year: dict[str, list[PageRow]] = {}
    for r in good:
        by_year.setdefault(r.year, []).append(r)
    for year in sorted(by_year):
        n = len(by_year[year])
        f = sum(1 for r in by_year[year] if r.score >= 1.0)
        lines.append(f"| {year} | {n} | {f} | {100.0 * f / max(1, n):.1f} % |")
    if labels:
        lines += ["", "## Эталон", "", "| страница | метка | score | причина | вердикт |", "|---|---|---|---|---|"]
        by_key = {r.key: r for r in rows}
        for key, (label, note) in sorted(labels.items()):
            row = by_key.get(key)
            if row is None:
                lines.append(f"| {key[0]} с.{key[1]} | {label} | — | — | нет в прогоне |")
                continue
            hit = (row.score >= 1.0) == (label == "bad")
            lines.append(
                f"| {key[0]} с.{key[1]} | {label} | {row.score:.2f} | {row.reason} | {'верно' if hit else '**мимо**'} {note} |"
            )
    lines += ["", f"## Топ находок (первые {top})", "", "| страница | score | причина | флаги |", "|---|---|---|---|"]
    for r in sorted(flagged, key=lambda r: -r.score)[:top]:
        flags = ", ".join(f"{k}={v:.1f}" for k, v in r.flags.items())
        lines.append(f"| {r.pdf} с.{r.page} | {r.score:.2f} | {r.reason} | {flags} |")
    errors = [r for r in rows if r.error]
    if errors:
        lines += ["", "## Ошибки", "", *(f"* {r.pdf} с.{r.page}: {r.error}" for r in errors[:50])]
    return "\n".join(lines) + "\n"
