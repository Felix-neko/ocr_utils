"""Сравнение выпрямителей: стало ли ровнее и не испортилось ли изображение.

ДВЕ ГРУППЫ МЕР, и вторая не менее важна первой. Первая говорит, стало ли ровнее: остаточная
сагитта, разброс углов, регулярность решётки. Вторая сторожит, чтобы выпрямление не съело
картинку: любая переинтерполяция размывает штрих, а слишком смелое поле смещений способно
утащить часть краски за край кадра. Поэтому меряется масса краски и резкость до и после, и
выпрямитель, уронивший их заметно, считается вредным независимо от того, что он сделал с
геометрией.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from research.legacy.table_processing.detection.ruling import binarize
from research.legacy.table_processing.warping.engines import Warper
from research.legacy.table_processing.warping.metrics import Geometry, measure

# Допустимая потеря краски и резкости. Замер интерполяции: бикубический ``remap`` на нулевом
# поле смещений оставляет 100% краски и 97-99% резкости, поэтому 2% и 10% — это запас на
# саму интерполяцию, а не на порчу.
MAX_INK_LOSS = 0.02
MAX_SHARPNESS_LOSS = 0.10


@dataclass(frozen=True)
class Outcome:
    """Что вышло у одного выпрямителя на одной таблице."""

    warper: str
    before: Geometry
    after: Geometry
    ink_ratio: float
    sharpness_ratio: float
    seconds: float
    note: str

    @property
    def safe(self) -> bool:
        return self.ink_ratio >= 1.0 - MAX_INK_LOSS and self.sharpness_ratio >= 1.0 - MAX_SHARPNESS_LOSS

    @property
    def sagitta_ratio(self) -> float:
        if self.before.sagitta_max_mm <= 0:
            return 1.0
        return self.after.sagitta_max_mm / self.before.sagitta_max_mm

    def as_row(self) -> dict[str, object]:
        return {
            "warper": self.warper,
            "sagitta_before_mm": round(self.before.sagitta_max_mm, 3),
            "sagitta_after_mm": round(self.after.sagitta_max_mm, 3),
            "sagitta_ratio": round(self.sagitta_ratio, 3),
            "spread_before_deg": round(self.before.angle_spread_deg, 3),
            "spread_after_deg": round(self.after.angle_spread_deg, 3),
            "lattice_before_mm": round(self.before.lattice_rms_mm, 3),
            "lattice_after_mm": round(self.after.lattice_rms_mm, 3),
            "ink_ratio": round(self.ink_ratio, 3),
            "sharpness_ratio": round(self.sharpness_ratio, 3),
            "safe": int(self.safe),
            "seconds": round(self.seconds, 2),
            "note": self.note,
        }


HEADER = (
    "warper",
    "sagitta_before_mm",
    "sagitta_after_mm",
    "sagitta_ratio",
    "spread_before_deg",
    "spread_after_deg",
    "lattice_before_mm",
    "lattice_after_mm",
    "ink_ratio",
    "sharpness_ratio",
    "safe",
    "seconds",
    "note",
)


def ink_mass(gray: np.ndarray) -> float:
    return float(np.count_nonzero(binarize(gray)))


def sharpness(gray: np.ndarray) -> float:
    """Энергия градиента: грубая, зато не требует эталона и чувствительна к размытию."""
    return float(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)).mean())


def apply(gray: np.ndarray, dpi: int, warper: Warper, before: "Geometry | None" = None) -> tuple[np.ndarray, Outcome]:
    """Применить один выпрямитель и померить, что вышло."""
    reference = before if before is not None else measure(gray, dpi)
    started = time.time()
    result = warper.run(gray, dpi)
    elapsed = time.time() - started

    after = measure(result.image, dpi) if result.changed else reference
    ink_before = max(1.0, ink_mass(gray))
    sharp_before = max(1e-6, sharpness(gray))
    outcome = Outcome(
        warper=warper.name,
        before=reference,
        after=after,
        ink_ratio=ink_mass(result.image) / ink_before if result.changed else 1.0,
        sharpness_ratio=sharpness(result.image) / sharp_before if result.changed else 1.0,
        seconds=elapsed,
        note=result.note,
    )
    return result.image, outcome


def best(outcomes: list[Outcome]) -> "Outcome | None":
    """Лучший из безопасных: тот, кто сильнее всех снизил сагитту."""
    safe = [outcome for outcome in outcomes if outcome.safe and outcome.sagitta_ratio < 1.0]
    return min(safe, key=lambda item: item.sagitta_ratio) if safe else None
