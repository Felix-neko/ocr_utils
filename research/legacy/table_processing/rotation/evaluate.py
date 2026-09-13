"""Сравнение детекторов поворота ячейки по ручной разметке.

МЕРА — доля верных ответов ПО ОСИ, отдельно по повёрнутым и по прямым ячейкам. Одной общей
доли мало: прямых ячеек в таблице кратно больше, и детектор, который всегда говорит
«прямая», набрал бы 80% и выглядел бы приличным.

МОЛЧАНИЕ НЕ ОШИБКА. Детектор, честно сказавший «судить не по чему» (уверенность 0), в
знаменатель точности не идёт, но считается отдельной колонкой: детектор, который молчит
на половине ячеек, бесполезен, даже если на второй половине безупречен.
"""

from __future__ import annotations

from dataclasses import dataclass

from research.legacy.table_processing.rotation.base import Verdict


@dataclass
class Score:
    """Счёт одного детектора."""

    name: str
    rotated_ok: int = 0
    rotated_total: int = 0
    upright_ok: int = 0
    upright_total: int = 0
    silent: int = 0
    sign_ok: int = 0
    sign_total: int = 0
    seconds: float = 0.0

    @property
    def rotated_rate(self) -> float:
        return self.rotated_ok / self.rotated_total if self.rotated_total else 0.0

    @property
    def upright_rate(self) -> float:
        return self.upright_ok / self.upright_total if self.upright_total else 0.0

    @property
    def sign_rate(self) -> float:
        return self.sign_ok / self.sign_total if self.sign_total else 0.0

    def add(self, verdict: Verdict, truth_rotated: bool, truth_sign: int = 90) -> None:
        if verdict.confidence <= 0.0:
            self.silent += 1
            return
        if truth_rotated:
            self.rotated_total += 1
            self.rotated_ok += int(verdict.rotated)
            if verdict.rotated and not verdict.axis_only:
                self.sign_total += 1
                self.sign_ok += int(verdict.rotate_cw == truth_sign)
        else:
            self.upright_total += 1
            self.upright_ok += int(not verdict.rotated)

    def as_row(self) -> list[object]:
        return [
            self.name,
            f"{self.rotated_ok}/{self.rotated_total}",
            f"{self.rotated_rate:.2f}",
            f"{self.upright_ok}/{self.upright_total}",
            f"{self.upright_rate:.2f}",
            self.silent,
            f"{self.sign_ok}/{self.sign_total}" if self.sign_total else "-",
            f"{self.seconds * 1000:.0f}",
        ]


HEADER = ("детектор", "боковые", "доля", "прямые", "доля", "молчал", "сторона", "мс/ячейка")
