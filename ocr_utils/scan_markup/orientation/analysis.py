"""Прогон детекторов по дереву полос.

СХЕМА ИСПОЛНЕНИЯ. Разжатие полосы и CPU-детекторы едут в пул процессов, GPU-детекторы
остаются в родителе и работают пачками: видеопамять одна на всех, и копия сети в каждом из
шестнадцати воркеров невозможна ни по памяти, ни по смыслу (CLAUDE.md, и та же схема в
``scan_markup/detection/run.py``). Воркер отдаёт в родителя не массив, а JPEG — см.
``image_io.read_frame``.

ДВА ПРОХОДА. Первым идут быстрые детекторы по всем полосам, вторым — арбитр по кандидатам,
которых первый проход отметил. Арбитр стоит четырёх распознаваний на полосу, и по всему паку
это полсотни минут ради нескольких сотен интересных полос.
"""

from __future__ import annotations

import logging
import ctypes
import multiprocessing
import os
import time
from collections import deque
from itertools import islice
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2

from ocr_utils.scan_markup.orientation.detectors import DETECTORS, Detector, Verdict
from ocr_utils.scan_markup.orientation.detectors.base import ROTATIONS
from ocr_utils.scan_markup.orientation.image_io import decode_gpu_jpeg, read_frame

logger = logging.getLogger(__name__)

# Воркеры уступают дорогу интерактивной работе: прогон по паку идёт полчаса, и всё это время
# машина не должна быть занята под завязку.
WORKER_NICE = 10

# Библиотеки, которые сами разойдутся по всем ядрам поверх пула, если их не остановить.
THREAD_LIMIT_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# Сколько заданий на воркера держать в работе. Двух хватает, чтобы воркер не простаивал в
# ожидании следующего, и мало, чтобы готовые результаты не копились: каждый везёт с собой
# JPEG уменьшенной копии, и без границы они съедают память (см. ``_bounded``).
IN_FLIGHT_PER_WORKER = 2


def _trim_heap() -> None:
    """Возвращает системе память, которую glibc держит во фрагментированных аренах.

    ЗАЧЕМ. GPU-этап прогоняет пачками крупные временные массивы, и куча после них остаётся
    раздутой, хотя объекты давно освобождены. Замер на пяти пачках по 32 полосы: surya
    прибавляла 220 МиБ, из которых malloc_trim возвращал 209 — то есть рост был почти целиком
    фрагментацией, а не занятой памятью. На двенадцати тысячах полос такая «фрагментация»
    складывается в десятки гигабайт и выдавливает из памяти соседние процессы.

    Не универсальное лекарство: если память действительно занята, trim не вернёт ничего
    (у docTR он возвращал 9% — там настоящая утечка, и лечится она только тем, что детектор
    не берут в длинный прогон).
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # не glibc — значит и арен таких нет


# Кто голосует за ось, а кто за сторону. Порядок значим: сторону берём у первого, кто
# высказался уверенно, а первым стоит арбитр — он единственный смотрит на сам текст.
AXIS_VOTERS = ("ink_axis", "profile", "surya_lines", "osd", "doctr", "onnx", "ocr_vote")

# Детекторы, которые судят об оси по НАЙДЕННЫМ СТРОКАМ, а не по успешности распознавания.
# Если промолчали оба, структурных свидетельств об оси нет вовсе, и остаются только догадки
# OCR — а он на такой полосе как раз и ошибается.
#
# ЗАМЕР по паку-1: из 43 находок оба промолчали на семи, и ВСЕ ТРИ ошибки прогона оказались
# среди этих семи (две полосы с бланками, которым свод назначил 180 вместо cw90, и снимок,
# которому поворот не нужен вовсе). Остальные 36 находок верны без исключения. Семь полос
# на просмотр глазами при двенадцати тысячах в паке — цена, которую стоит платить.
LINE_AXIS_VOTERS = ("ink_axis", "surya_lines")
SIGN_VOTERS = ("ocr_vote", "osd", "doctr", "onnx", "ink_axis")

# Пороги уверенности при сведении. Ось — грубая геометрия, её видно уверенно; сторона —
# тонкий признак, и требовать от неё той же уверенности значило бы всё время оставаться без
# ответа (замер: у боковых полос сторона у ink_axis выходит с уверенностью 0.11-0.38).
AXIS_MIN_CONFIDENCE = 0.25
SIGN_MIN_CONFIDENCE = 0.15

# Уверенность, с которой мнение «полоса повёрнута» отправляет её к арбитру.
CANDIDATE_CONFIDENCE = 0.25


@dataclass
class PageResult:
    """Что известно об одной полосе после всех детекторов."""

    rel_path: str
    path: Path
    width: int = 0
    height: int = 0
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    combo: Verdict | None = None
    candidate: bool = False
    seconds: dict[str, float] = field(default_factory=dict)
    sign_from: str = ""
    disputed: bool = False
    error: str = ""

    @property
    def rotated(self) -> bool:
        return self.combo is not None and self.combo.rotate_cw != 0


@dataclass(frozen=True)
class Job:
    path: Path
    rel_path: str
    detector_names: tuple[str, ...]
    default_dpi: int
    gpu_side: int
    turn_cw: int = 0  # синтетический поворот, нужен только команде validate
    allowed: tuple[int, ...] = ROTATIONS


def _init_worker() -> None:
    cv2.setNumThreads(1)
    try:
        os.nice(WORKER_NICE)
    except OSError:
        pass


def _worker(job: Job) -> tuple[PageResult, bytes | None]:
    """CPU-этап одной полосы. Не бросает: сбой одной полосы не должен ронять прогон."""
    result = PageResult(rel_path=job.rel_path, path=job.path)
    try:
        frame, payload = read_frame(
            job.path, job.rel_path, default_dpi=job.default_dpi, gpu_side=job.gpu_side, allowed=job.allowed
        )
    except Exception as error:
        result.error = f"чтение: {error}"
        return result, None
    if job.turn_cw:
        frame = _turn(frame, job.turn_cw)
        payload = _turn_payload(payload, job.turn_cw)
    result.width, result.height = frame.width, frame.height
    for name in job.detector_names:
        detector = DETECTORS[name]
        started = time.perf_counter()
        try:
            result.verdicts[name] = detector.run(frame)
        except Exception as error:
            result.error = f"{name}: {error}"
        result.seconds[name] = time.perf_counter() - started
    return result, payload


def _turn(frame, degrees: int):
    """Синтетический поворот кадра — для проверки детекторов на заведомо прямых полосах."""
    from ocr_utils.scan_markup.orientation.detectors.base import Frame, rotate_cw

    return Frame(
        frame.rel_path,
        frame.path,
        frame.width,
        frame.height,
        frame.dpi,
        rotate_cw(frame.gray150, degrees),
        rotate_cw(frame.gray300, degrees),
    )


def _turn_payload(payload: bytes | None, degrees: int) -> bytes | None:
    if payload is None:
        return None
    import io

    import numpy as np
    from PIL import Image

    from ocr_utils.scan_markup.orientation.detectors.base import rotate_cw
    from ocr_utils.scan_markup.orientation.image_io import GPU_JPEG_QUALITY

    turned = Image.fromarray(rotate_cw(np.asarray(decode_gpu_jpeg(payload)), degrees))
    buffer = io.BytesIO()
    turned.save(buffer, format="JPEG", quality=GPU_JPEG_QUALITY)
    return buffer.getvalue()


def _with_progress(iterator: Iterable, total: int, enabled: bool, desc: str) -> Iterator:
    if not enabled:
        yield from iterator
        return
    from tqdm import tqdm

    yield from tqdm(iterator, total=total, desc=desc, unit="полоса")


def _bounded(
    pool: ProcessPoolExecutor, jobs: Sequence[Job], in_flight: int
) -> Iterator[tuple[PageResult, bytes | None]]:
    """Результаты по порядку, но в работе не больше ``in_flight`` заданий.

    ЗАЧЕМ НЕ ``pool.map``. Он отправляет в пул ВСЕ задания разом, и дальше воркеры считают
    так быстро, как могут, складывая готовое в очередь. Здесь потребитель заведомо медленнее
    производителей — родитель прогоняет GPU-этап, пока двенадцать воркеров разжимают полосы, —
    и очередь растёт без всякой границы. На прогоне по паку-1 это выглядело так: родитель
    занимал 8 ГиБ на 3145 полосах из 12 135, шёл к тридцати гигабайтам и попутно выдавил из
    памяти соседние процессы.

    Ограничение числа заданий в работе связывает потребителя с производителем: пока родитель
    не разобрал результат, следующее задание в пул не уходит.
    """
    queue: deque = deque()
    remaining = iter(jobs)
    for job in islice(remaining, in_flight):
        queue.append(pool.submit(_worker, job))
    while queue:
        result = queue.popleft().result()
        next_job = next(remaining, None)
        if next_job is not None:
            queue.append(pool.submit(_worker, next_job))
        yield result


def _run_pool(
    jobs: Sequence[Job], workers: int, progress: bool, desc: str
) -> Iterator[tuple[PageResult, bytes | None]]:
    if workers <= 1 or len(jobs) <= 1:
        yield from _with_progress((_worker(job) for job in jobs), len(jobs), progress, desc)
        return
    for name in THREAD_LIMIT_VARS:
        os.environ.setdefault(name, "1")
    # forkserver, а не fork: в родителе к этому моменту может быть поднят torch, а форк
    # процесса с инициализированной CUDA — верный способ получить зависший воркер.
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
        stream = _bounded(pool, jobs, max(IN_FLIGHT_PER_WORKER * workers, 8))
        yield from _with_progress(stream, len(jobs), progress, desc)


def analyse(
    tasks: Sequence[tuple[Path, str]],
    detectors: Sequence[Detector],
    workers: int,
    default_dpi: int = 600,
    gpu_side: int = 1536,
    gpu_batch: int = 32,
    progress: bool = True,
    desc: str = "ориентация",
    turn_cw: int = 0,
    allowed: tuple[int, ...] = ROTATIONS,
) -> list[PageResult]:
    """Прогон одного набора детекторов по списку полос. Порядок результатов — как на входе."""
    cpu_names = tuple(d.name for d in detectors if d.stage == "cpu")
    gpu_detectors = [d for d in detectors if d.stage == "gpu"]
    batches = {d.name: d.make_batch() for d in gpu_detectors}

    jobs = [
        Job(path, rel, cpu_names, default_dpi, gpu_side if gpu_detectors else 0, turn_cw, allowed)
        for path, rel in tasks
    ]
    results: list[PageResult] = []
    pending: list[PageResult] = []
    images: list[object] = []

    def flush() -> None:
        if not pending:
            return
        for name, batch in batches.items():
            started = time.perf_counter()
            verdicts = batch(images)
            if len(verdicts) != len(pending):
                # zip молча обрезал бы разъехавшиеся списки, и вердикты сдвинулись бы на
                # чужие полосы — ошибка, которую в отчёте не отличить от ошибки детектора.
                raise RuntimeError(f"{name} вернул {len(verdicts)} вердиктов на {len(pending)} полос")
            spent = (time.perf_counter() - started) / max(1, len(pending))
            for result, verdict in zip(pending, verdicts):
                result.verdicts[name] = verdict
                result.seconds[name] = spent
        results.extend(pending)
        pending.clear()
        images.clear()
        _trim_heap()

    for result, payload in _run_pool(jobs, workers, progress, desc):
        if not gpu_detectors or payload is None:
            results.append(result)
            continue
        pending.append(result)
        images.append(decode_gpu_jpeg(payload))
        if len(pending) >= gpu_batch:
            flush()
    flush()
    return results


def triggering_verdict(result: PageResult) -> tuple[str, Verdict] | None:
    """Самое уверенное мнение «полоса повёрнута», если такое вообще есть.

    Именно оно делает полосу кандидатом, и именно его поворот попадает в имя симлинка
    в каталоге кандидатов — иначе у отвергнутой полосы поворот назвать было бы нечем.
    """
    speaking = [
        (name, verdict)
        for name, verdict in result.verdicts.items()
        if verdict.rotate_cw != 0 and verdict.confidence >= CANDIDATE_CONFIDENCE
    ]
    return max(speaking, key=lambda item: item[1].confidence) if speaking else None


def candidates(results: Sequence[PageResult]) -> list[PageResult]:
    """Полосы, которые хоть один быстрый детектор счёл повёрнутыми, — работа для арбитра.

    Отбор НАМЕРЕННО щедрый: достаточно одного голоса с уверенностью
    ``CANDIDATE_CONFIDENCE``, согласия не требуется. Пропущенная полоса тут не всплывёт уже
    никогда, а лишний кандидат стоит одного прогона арбитра, то есть секунд. Поэтому порог
    низкий, а решает потом арбитр — он же и отменяет находку, сказав «прямо».

    Флаг ставится на самой полосе: по нему потом раскладывается каталог ``кандидаты/``,
    где видно и принятые, и отвергнутые.
    """
    chosen = []
    for result in results:
        result.candidate = triggering_verdict(result) is not None
        if result.candidate:
            chosen.append(result)
    return chosen


# Имена арбитров. Арбитр — единственный, кто проверяет ВСЕ гипотезы напрямую: он читает
# полосу на каждом допустимом повороте. Остальные детекторы судят об ориентации косвенно —
# по форме строк, по периодичности профиля, по обученной сети, — и нужны они для того, чтобы
# решить, кого вообще стоит отдать арбитру.
#
# Поэтому если арбитр высказался, его ответ и есть ответ. Раньше он проходил через то же
# голосование за ось, что и остальные, и это давало ровно одну ошибку: на бланке, где ось
# никто из структурных детекторов не разглядел, голосование выбирало «книжную» ось и
# отбрасывало верный ответ арбитра как «не на той оси». Замер на 122 размеченных вручную
# полосах: арбитр в одиночку даёт 122 из 122.
ARBITERS = ("ocr_vote",)


def combine(verdicts: dict[str, Verdict], allowed: tuple[int, ...] = ROTATIONS) -> tuple[Verdict, str, bool]:
    """Сводный вердикт: ось — голосованием, сторона — первым уверенным из ``SIGN_VOTERS``.

    Возвращает вердикт, имя детектора, давшего сторону, и признак спорности. Спорной полоса
    объявляется, когда сторону называют двое и называют по-разному: выдать в такой ситуации
    один из ответов значило бы спрятать разногласие вместо того, чтобы показать его глазам.
    """
    for name in ARBITERS:
        verdict = verdicts.get(name)
        if verdict is not None and verdict.confidence > 0.0 and verdict.rotate_cw in allowed:
            return verdict, name, False

    weight = {"up": 0.0, "side": 0.0}
    for name in AXIS_VOTERS:
        verdict = verdicts.get(name)
        if verdict is None or verdict.confidence < AXIS_MIN_CONFIDENCE:
            continue
        weight["side" if verdict.rotate_cw in (90, 270) else "up"] += verdict.confidence
    if weight["up"] == 0.0 and weight["side"] == 0.0:
        return Verdict(0, 0.0, note="ось никто не определил"), "", False

    side = weight["side"] > weight["up"]
    candidates_ = tuple(r for r in ((90, 270) if side else (0, 180)) if r in allowed)
    if not candidates_:
        return Verdict(0, 0.0, note="ось вне набора допустимых углов"), "", False
    axis_confidence = max(weight.values()) / (weight["up"] + weight["side"])

    voted: list[tuple[str, Verdict]] = []
    for name in SIGN_VOTERS:
        verdict = verdicts.get(name)
        if verdict is None or verdict.axis_only or verdict.confidence < SIGN_MIN_CONFIDENCE:
            continue
        if verdict.rotate_cw in candidates_:
            voted.append((name, verdict))

    if not voted:
        # Ось видна, сторону назвать некому. Для книжной оси это «поворот не нужен»: полоса
        # журнала вверх ногами — редкость, а вот боковая — обычное дело, и её надо показать.
        note = "" if not side else "сторона неизвестна"
        fallback = candidates_[0] if side else 0
        return Verdict(fallback, axis_confidence, axis_only=side, note=note), "", side

    blind = all(
        (verdicts.get(name) is None or verdicts[name].confidence < AXIS_MIN_CONFIDENCE) for name in LINE_AXIS_VOTERS
    )

    chosen_name, chosen = voted[0]
    # Возражение считается возражением, только если возражающий уверен НЕ МЕНЬШЕ выбранного.
    # Иначе спорной становилась бы каждая боковая полоса: сторону там называет арбитр,
    # который читает сам текст и уверен на 0.8-0.9, а вяло не соглашается ink_axis со своими
    # 0.11-0.38 — и разногласием это назвать нельзя, это разная надёжность.
    disputed = any(
        other.rotate_cw != chosen.rotate_cw and other.confidence >= chosen.confidence for _, other in voted[1:]
    )
    if blind:
        disputed = True
        note = "строк не нашёл никто — ось назначена по распознаванию"
    else:
        note = "детекторы разошлись в стороне" if disputed else ""
    return Verdict(chosen.rotate_cw, min(axis_confidence, chosen.confidence), note=note), chosen_name, disputed


def apply_combo(results: Sequence[PageResult], allowed: tuple[int, ...] = ROTATIONS) -> None:
    for result in results:
        result.combo, result.sign_from, result.disputed = combine(result.verdicts, allowed)


def rotation_counts(results: Sequence[PageResult], name: str) -> dict[int, int]:
    """Сколько полос каждый поворот собрал у одного детектора — строка сводной таблицы."""
    counts = {rotation: 0 for rotation in ROTATIONS}
    for result in results:
        verdict = result.combo if name == "combo" else result.verdicts.get(name)
        if verdict is not None and verdict.confidence > 0.0:
            counts[verdict.rotate_cw] += 1
    return counts
