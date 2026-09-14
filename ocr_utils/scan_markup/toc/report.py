"""Проверка детектора по эталону: матрицы ошибок по признакам и по решению, списки промахов."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ocr_utils.scan_markup.toc.decide import Thresholds, is_weak, strong_reason
from ocr_utils.scan_markup.toc.features import PageFeatures
from ocr_utils.scan_markup.toc.labels import Label, lookup
from ocr_utils.scan_markup.toc.run import IssueResult


@dataclass
class Confusion:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    missed: list[str] = field(default_factory=list)
    false: list[str] = field(default_factory=list)

    def add(self, rel_path: str, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted:
            self.fp += 1
            self.false.append(rel_path)
        elif actual:
            self.fn += 1
            self.missed.append(rel_path)
        else:
            self.tn += 1

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    def row(self, name: str) -> str:
        return f"| {name} | {self.tp} | {self.fn} | {self.fp} | {self.tn} | {self.recall:.3f} | {self.precision:.3f} |"


def evaluate(results: Sequence[IssueResult], labels: dict[str, Label], thresholds: Thresholds) -> str:
    """Markdown-отчёт по размеченным полосам окна."""
    signals: dict[str, Confusion] = {
        "surya TableOfContents": Confusion(),
        "«СОДЕРЖАНИЕ»": Confusion(),
        "заголовок указателя": Confusion(),
        "строки с номером (слабый)": Confusion(),
        "любой сильный": Confusion(),
        "РЕШЕНИЕ": Confusion(),
    }
    kind_errors: list[str] = []
    labelled = 0
    for result in results:
        by_index = {d.order_index: d for d in result.decisions}
        for feature in result.features:
            label = lookup(labels, feature.rel_path)
            if label is None:
                continue
            labelled += 1
            decision = by_index[feature.order_index]
            actual = label.is_toc
            signals["surya TableOfContents"].add(
                feature.rel_path, feature.surya_toc_conf >= thresholds.surya_min_conf, actual
            )
            signals["«СОДЕРЖАНИЕ»"].add(feature.rel_path, feature.kw_contents, actual)
            signals["заголовок указателя"].add(feature.rel_path, feature.kw_index, actual)
            signals["строки с номером (слабый)"].add(feature.rel_path, is_weak(feature, thresholds), actual)
            signals["любой сильный"].add(feature.rel_path, bool(strong_reason(feature, thresholds)), actual)
            signals["РЕШЕНИЕ"].add(feature.rel_path, decision.is_toc, actual)
            if actual and decision.is_toc and decision.kind != label.label:
                kind_errors.append(f"{feature.rel_path}: детектор {decision.kind}, эталон {label.label}")

    lines = [
        f"Размеченных полос окна: {labelled} (в эталоне {len(labels)}).",
        "",
        "| признак | TP | FN | FP | TN | полнота | точность |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [confusion.row(name) for name, confusion in signals.items()]
    final = signals["РЕШЕНИЕ"]
    if final.missed:
        lines += ["", "Пропущено решением:"] + [f"* {item}" for item in final.missed]
    if final.false:
        lines += ["", "Ложно решением:"] + [f"* {item}" for item in final.false]
    if kind_errors:
        lines += ["", "Вид перепутан:"] + [f"* {item}" for item in kind_errors]
    return "\n".join(lines)


__all__ = ["Confusion", "evaluate"]
