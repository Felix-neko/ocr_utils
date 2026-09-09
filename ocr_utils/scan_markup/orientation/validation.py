"""Проверка детекторов на синтетических поворотах заведомо прямых полос.

ЗАЧЕМ ИМЕННО ТАК. Размеченного набора «эта полоса повёрнута» нет и делать его руками по
12 тысячам полос никто не станет. Зато есть тысячи полос, про которые ВСЕ детекторы
согласны, что они прямые, — и вот их можно крутить самим, зная правильный ответ заранее.
Материал при этом остаётся настоящим: та же бумага, тот же кегль, тот же растр.

Побочный и не менее важный смысл: проверка ПРИШИВАЕТ ЗНАКОВЫЕ СОГЛАШЕНИЯ. Каждый чужой
движок считает угол по-своему (docTR — против часовой, tesseract — по часовой), и ошибка
в знаке даёт детектор, который уверенно и стабильно поворачивает не в ту сторону. На паке
это заметить трудно, на матрице ошибок — сразу.

Чего проверка НЕ показывает: как детектор ведёт себя на настоящей боковой полосе, где
текста одна подпись, а всё остальное — чертёж. Повёрнутая текстовая полоса и боковая
иллюстрация — разные задачи, и первая заметно легче.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ocr_utils.scan_markup.orientation.analysis import PageResult, analyse
from ocr_utils.scan_markup.orientation.detectors import ROTATION_NAMES, ROTATIONS, Detector

# Уверенность, с которой полоса берётся в эталон как заведомо прямая.
REFERENCE_CONFIDENCE = 0.5


@dataclass
class Trial:
    """Один поворот одной полосы: что подали и что ответили."""

    rel_path: str
    applied_cw: int
    expected_cw: int
    answers: dict[str, tuple[int, float, bool]] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)


def pick_reference(results: Sequence[PageResult], wanted: int, arbiter: str | None = None) -> list[PageResult]:
    """Заведомо прямые полосы — материал для синтетических поворотов.

    ОТБИРАЕТ АРБИТР, а не общее согласие, и это принципиально. Эталон «полоса, про которую
    НИКТО не сказал, что она повёрнута» выбрасывает ровно те полосы, на которых детекторы
    ошибаются, — то есть завышает точность каждого из них на классе «прямо», причём тем
    сильнее, чем чаще детектор ошибается. Сравнение после такого отбора говорило бы больше
    об отборе, чем о детекторах.

    Арбитр от этого свободен: он читает сам текст, а не судит о нём по геометрии, и его
    мнение не зависит от того, что скажут остальные. Арбитра в наборе нет — приходится
    возвращаться к согласию, и отчёт об этом предупреждает.
    """
    chosen = []
    for result in results:
        if result.error:
            continue
        if arbiter is not None and arbiter in result.verdicts:
            verdict = result.verdicts[arbiter]
            if verdict.rotate_cw != 0 or verdict.confidence < REFERENCE_CONFIDENCE:
                continue
        else:
            verdicts = [v for v in result.verdicts.values() if v.confidence > 0.0]
            if not verdicts or any(verdict.rotate_cw != 0 for verdict in verdicts):
                continue
            if max(verdict.confidence for verdict in verdicts) < REFERENCE_CONFIDENCE:
                continue
        chosen.append(result)
        if len(chosen) >= wanted:
            break
    return chosen


def run_trials(
    reference: Sequence[PageResult],
    detectors: Sequence[Detector],
    workers: int,
    default_dpi: int,
    gpu_side: int,
    gpu_batch: int,
    progress: bool = True,
    allowed: tuple[int, ...] = ROTATIONS,
) -> list[Trial]:
    tasks = [(result.path, result.rel_path) for result in reference]
    trials: list[Trial] = []
    # Крутим ровно на те углы, которые разрешены прогону: проверять надо ту настройку,
    # с которой потом пойдёшь на пак, а не какую-то другую.
    for applied in allowed:
        turned = analyse(
            tasks,
            detectors,
            workers=workers,
            default_dpi=default_dpi,
            gpu_side=gpu_side,
            gpu_batch=gpu_batch,
            progress=progress,
            desc=f"поворот {applied}",
            turn_cw=applied,
            allowed=allowed,
        )
        for result in turned:
            trial = Trial(result.rel_path, applied, (-applied) % 360)
            for name, verdict in result.verdicts.items():
                trial.answers[name] = (verdict.rotate_cw, verdict.confidence, verdict.axis_only)
            trial.seconds = dict(result.seconds)
            trials.append(trial)
    return trials


def sample_paths(paths: Sequence[Path], root: Path, count: int, seed: int) -> list[tuple[Path, str]]:
    generator = random.Random(seed)
    picked = generator.sample(list(paths), min(count, len(paths)))
    return [(path, path.relative_to(root).as_posix()) for path in sorted(picked)]


def correct(answer: tuple[int, float, bool], expected: int, strict: bool = False) -> bool:
    """Верен ли ответ.

    ``strict`` — совпадение один в один. Без него детектору, поднявшему ``axis_only``,
    зачитывается совпадение по ОСИ: он не брался называть сторону, и спрашивать с него за
    неё нечестно. В отчёте показаны обе величины: их расхождение и есть мера того, сколько
    работы детектор оставляет другим.
    """
    rotate, confidence, axis_only = answer
    if confidence <= 0.0:
        return False
    if strict or not axis_only:
        return rotate == expected
    return (rotate % 180) == (expected % 180)


def accuracy(trials: Sequence[Trial], name: str, strict: bool = False) -> tuple[float, int, int]:
    """Доля верных среди высказавшихся, число высказавшихся, всего попыток."""
    spoke = right = 0
    for trial in trials:
        answer = trial.answers.get(name)
        if answer is None or answer[1] <= 0.0:
            continue
        spoke += 1
        right += correct(answer, trial.expected_cw, strict=strict)
    return (right / spoke if spoke else float("nan")), spoke, len(trials)


def confusion(trials: Sequence[Trial], name: str) -> dict[int, dict[str, int]]:
    """Матрица «ждали → ответили», плюс графа «молчит»."""
    seen = sorted({trial.expected_cw for trial in trials})
    table = {expected: {ROTATION_NAMES[r]: 0 for r in ROTATIONS} | {"молчит": 0} for expected in seen}
    for trial in trials:
        answer = trial.answers.get(name)
        row = table[trial.expected_cw]
        if answer is None or answer[1] <= 0.0:
            row["молчит"] += 1
        else:
            row[ROTATION_NAMES[answer[0]]] += 1
    return table
