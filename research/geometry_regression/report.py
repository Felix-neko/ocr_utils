"""CSV с метриками по страницам и markdown-сводка прогона."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ocr_utils.geometry_regression.scoring import DEFAULT_THRESHOLDS, UNFORGIVABLE, Thresholds, Verdict

FIXED_FIELDS = ("pdf", "year", "page", "verdict", "score", "reason", "gain", "gain_reason", "flags", "error")

# Пояса score для сводки и выборки глазами.
BELTS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.3), (1.3, 2.0), (2.0, float("inf")))


@dataclass
class PageRow:
    """Одна страница в CSV: метрики плюс вердикт."""

    pdf: str
    page: int  # с единицы, как в PDF-читалке
    metrics: dict[str, float] = field(default_factory=dict)
    score: float = 0.0  # порча
    reason: str = ""
    flags: dict[str, float] = field(default_factory=dict)
    error: str = ""
    gain: float = 0.0
    gain_reason: str = ""
    verdict: str = "ok"

    @property
    def year(self) -> str:
        parts = self.pdf.split("_")
        return parts[1] if len(parts) > 1 else self.pdf

    @property
    def key(self) -> tuple[str, int]:
        return (self.pdf, self.page)

    def apply(self, verdict: Verdict) -> None:
        self.score, self.reason, self.flags = verdict.score, verdict.reason, dict(verdict.flags)
        self.gain, self.gain_reason, self.verdict = verdict.gain, verdict.gain_reason, verdict.verdict


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
                [
                    row.pdf,
                    row.year,
                    row.page,
                    row.verdict,
                    f"{row.score:.3f}",
                    row.reason,
                    f"{row.gain:.3f}",
                    row.gain_reason,
                    flags,
                    row.error,
                ]
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
                    float(record.get("gain") or 0.0),
                    record.get("gain_reason", ""),
                    record.get("verdict", "ok"),
                )
            )
    return rows


def reflag(rows: list[PageRow], thresholds: Thresholds) -> None:
    for row in rows:
        if not row.error:
            row.apply(thresholds.apply(row.metrics))


# Файлы эталона в папке валидации: имя → метка страниц в нём.
LABEL_FILES = {"fr_correction_bad.csv": "bad", "fr_correction_good.csv": "good"}


def load_labels(path: Path) -> dict[tuple[str, int], tuple[str, str]]:
    """Эталон → ``{(pdf, page): (label, note)}``.

    ``path`` — папка валидации (``research/geometry_regression/validation/pack1``) с TSV
    ``fr_correction_bad.csv`` и ``fr_correction_good.csv`` вида ``название<TAB>путь<TAB>аннотация``,
    где название — ``full_ГГГГ_НН с.N``. Для совместимости принимается и старый CSV
    ``pdf,page,label,note`` одним файлом.
    """
    labels: dict[tuple[str, int], tuple[str, str]] = {}
    if path.is_dir():
        for name, label in LABEL_FILES.items():
            file = path / name
            if not file.is_file():
                continue
            with file.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle, delimiter="\t")
                next(reader, None)
                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    pdf, _, page = row[0].partition(" с.")
                    labels[(pdf.strip(), int(page))] = (label, row[2].strip() if len(row) > 2 else "")
        return labels
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            labels[(record["pdf"], int(record["page"]))] = (record["label"], record.get("note", ""))
    return labels


def pair_name(row: PageRow) -> str:
    return f"{row.pdf}_p{row.page:03d}_d{row.score:.2f}_g{row.gain:.2f}_{row.reason or 'none'}.jpg"


def markdown_report(
    rows: list[PageRow], thresholds: Thresholds, labels: dict | None, out_dir: Path, top: int = 25
) -> str:
    good = [r for r in rows if not r.error]
    reasons = getattr(thresholds, "reasons", {name: DEFAULT_THRESHOLDS[name][1] for name in DEFAULT_THRESHOLDS})
    unforgivable = getattr(thresholds, "unforgivable", UNFORGIVABLE)
    flagged = [r for r in good if r.score >= 1.0]
    bad = [r for r in good if r.verdict == "bad"]
    mixed = [r for r in good if r.verdict == "mixed"]
    lines = [
        "# Страницы, где коррекция геометрии FineReader сделала хуже",
        "",
        f"Страниц: {len(rows)}, с ошибкой: {len(rows) - len(good)}; порча ≥ 1: {len(flagged)}"
        f" ({100.0 * len(flagged) / max(1, len(good)):.1f} %), из них **bad** {len(bad)}"
        f" ({100.0 * len(bad) / max(1, len(good)):.1f} %) и **mixed** {len(mixed)} (порча < {thresholds.ratio:g} выигрыша"
        f" при выигрыше ≥ {thresholds.min_gain:g} — берём версию с коррекцией). Выход: `{out_dir}`.",
        "",
        "## Пороги",
        "",
        "| метрика | порог | причина | непрощаемая |",
        "|---|---|---|---|",
        *(
            f"| `{name}` | {thresholds.values[name]:g} | {reasons.get(name, '')} | {'да' if name in unforgivable else ''} |"
            for name in thresholds.values
        ),
        "",
        "Выигрыш: " + ", ".join(f"`{name}` ≥ {value:g}" for name, value in thresholds.gains.items()) + ".",
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
    lines += ["", "## Причины порчи (bad / mixed)", "", "| причина | bad | mixed |", "|---|---|---|"]
    counts_bad, counts_mixed = Counter(r.reason for r in bad), Counter(r.reason for r in mixed)
    for reason, _ in Counter(r.reason for r in flagged).most_common():
        lines.append(f"| {reason} | {counts_bad.get(reason, 0)} | {counts_mixed.get(reason, 0)} |")
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
    lines += ["", "## По годам", "", "| год | страниц | bad | mixed | доля bad |", "|---|---|---|---|---|"]
    by_year: dict[str, list[PageRow]] = {}
    for r in good:
        by_year.setdefault(r.year, []).append(r)
    for year in sorted(by_year):
        n = len(by_year[year])
        nb = sum(1 for r in by_year[year] if r.verdict == "bad")
        nm = sum(1 for r in by_year[year] if r.verdict == "mixed")
        lines.append(f"| {year} | {n} | {nb} | {nm} | {100.0 * nb / max(1, n):.1f} % |")
    if labels:
        lines += [
            "",
            "## Эталон",
            "",
            "| страница | метка | порча | выигрыш | вердикт | итог |",
            "|---|---|---|---|---|---|",
        ]
        by_key = {r.key: r for r in rows}
        for key, (label, note) in sorted(labels.items()):
            row = by_key.get(key)
            if row is None:
                lines.append(f"| {key[0]} с.{key[1]} | {label} | — | — | — | нет в прогоне |")
                continue
            hit = (row.verdict == "bad") == (label == "bad")
            lines.append(
                f"| {key[0]} с.{key[1]} | {label} | {row.score:.2f} {row.reason} | {row.gain:.2f} {row.gain_reason} | {row.verdict} | {'верно' if hit else '**мимо**'} {note} |"
            )
    lines += ["", f"## Топ bad (первые {top})", "", "| страница | порча | выигрыш | флаги |", "|---|---|---|---|"]
    for r in sorted(bad, key=lambda r: -r.score)[:top]:
        flags = ", ".join(f"{k}={v:.1f}" for k, v in r.flags.items())
        lines.append(f"| {r.pdf} с.{r.page} | {r.score:.2f} {r.reason} | {r.gain:.2f} {r.gain_reason} | {flags} |")
    errors = [r for r in rows if r.error]
    if errors:
        lines += ["", "## Ошибки", "", *(f"* {r.pdf} с.{r.page}: {r.error}" for r in errors[:50])]
    return "\n".join(lines) + "\n"
