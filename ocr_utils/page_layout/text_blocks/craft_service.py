"""CRAFT для фильтра псевдоглифов как служба: сервер в главном процессе (одна модель на GPU, поток-обслуживатель под общей блокировкой GPU) и клиент в воркерах пула — запросы по очереди процессов, у каждого воркера своя очередь ответов, таймаут с откатом на правила."""

from __future__ import annotations

import itertools
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from ocr_utils.page_layout.text_blocks.pseudo_glyphs import CraftScorer, LocalCraft, set_default_scorer

log = logging.getLogger(__name__)

# Сколько ждать ответа сервера, с: GPU делят SAM 3, LaMa, surya и уточнение рваных сторон под одной блокировкой, и
# очередь до CRAFT может стоять десятки секунд. Дольше — проход фильтра идёт одними правилами (предупреждение в лог).
REPLY_TIMEOUT_S = 300.0
# После стольких таймаутов подряд клиент больше не ждёт сервер (он, видимо, не поднялся) — сразу правила.
MAX_FAILURES = 2
# Сколько запросов сервер собирает за один захват блокировки GPU и сколько ждёт следующих, с.
BATCH_MAX = 32
BATCH_WAIT_S = 0.02
# Как часто поток-обслуживатель проверяет флаг остановки, с.
POLL_S = 0.2


@dataclass(frozen=True)
class CraftRequest:
    """Запрос воркера: номер, слот (его очередь ответов), вырезки и маски кандидатов."""

    request_id: int
    slot: int
    crops: list[np.ndarray]
    masks: list[np.ndarray]


class RemoteCraft(CraftScorer):
    """Клиент CRAFT-сервера в воркере пула: запрос в общую очередь, ответ — из своей очереди слота."""

    def __init__(self, requests, replies, slot: int, timeout_s: float = REPLY_TIMEOUT_S):
        """Клиент.

        Args:
            requests: Общая очередь запросов (``multiprocessing.Queue`` контекста пула).
            replies: Очередь ответов этого слота.
            slot: Номер слота (куда сервер шлёт ответ).
            timeout_s: Сколько ждать ответа.
        """
        self.requests = requests
        self.replies = replies
        self.slot = slot
        self.timeout_s = timeout_s
        self._ids = itertools.count(1)
        self._warned = False
        self._failures = 0

    @property
    def available(self) -> bool:
        """Канал до сервера есть."""
        return self.requests is not None and self.replies is not None

    def score_crops(self, crops: list[np.ndarray], masks: list[np.ndarray]) -> list[float] | None:
        """Оценки по вырезкам от сервера; ``None`` — сервер не ответил за ``timeout_s`` (дальше — одни правила).

        Args:
            crops: Вырезки uint8.
            masks: Маски кандидатов.

        Returns:
            Оценки по порядку или ``None``.
        """
        if not crops:
            return []
        if self._failures >= MAX_FAILURES:
            return None
        request_id = next(self._ids)
        try:
            self.requests.put(CraftRequest(request_id, self.slot, crops, masks))
            deadline = time.monotonic() + self.timeout_s
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise queue.Empty
                answer_id, scores = self.replies.get(timeout=left)
                # Ответ на прежний запрос, по которому истёк таймаут, — выбрасывается.
                if answer_id == request_id:
                    self._failures = 0
                    return scores
        except (queue.Empty, OSError, EOFError, ValueError) as error:
            self._failures += 1
            if not self._warned:
                log.warning("CRAFT-сервер не ответил (%s), воркер %d: фильтр псевдоглифов — одними правилами",
                            type(error).__name__, os.getpid())  # fmt: skip
                self._warned = True
            return None


class CraftServer:
    """Сервер CRAFT в главном процессе: поток забирает запросы воркеров пачками и оценивает их под блокировкой GPU."""

    def __init__(self, scorer: CraftScorer, requests, replies: list, gpu_lock=None):
        """Сервер (поток стартует :meth:`start`).

        Args:
            scorer: Оценщик с моделью (``LocalCraft(device="cuda")``; в тестах — заглушка).
            requests: Общая очередь запросов.
            replies: Очереди ответов по слотам.
            gpu_lock: Общая блокировка GPU прогона (``threading.RLock``) или ``None``.
        """
        self.scorer = scorer
        self.requests = requests
        self.replies = replies
        self.gpu_lock = gpu_lock if gpu_lock is not None else threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="craft-server", daemon=True)
        self.served = 0  # обработано вырезок
        self.batches = 0  # захватов блокировки GPU

    def start(self) -> "CraftServer":
        """Запустить поток-обслуживатель."""
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        """Остановить поток (запросы, пришедшие после остановки, клиенты не дождутся — таймаут и правила)."""
        self._stop.set()
        self._thread.join(timeout_s)

    def _batch(self) -> list[CraftRequest]:
        """Пачка запросов: первый — с ожиданием до ``POLL_S``, остальные — что успело прийти за ``BATCH_WAIT_S``."""
        try:
            batch = [self.requests.get(timeout=POLL_S)]
        except queue.Empty:
            return []
        deadline = time.monotonic() + BATCH_WAIT_S
        while len(batch) < BATCH_MAX:
            try:
                batch.append(self.requests.get(timeout=max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                break
        return batch

    def _serve(self) -> None:
        """Цикл обслуживания: пачка → оценка под блокировкой GPU → ответы по слотам; ошибка оценки — ответ ``None``."""
        while not self._stop.is_set():
            batch = self._batch()
            if not batch:
                continue
            with self.gpu_lock:
                self.batches += 1
                answers = []
                for request in batch:
                    try:
                        scores = self.scorer.score_crops(request.crops, request.masks)
                        self.served += len(request.crops)
                    except Exception:  # noqa: BLE001 — ошибка оценки не роняет сервер: клиент уйдёт на правила
                        log.exception("CRAFT-сервер: ошибка оценки запроса %d", request.request_id)
                        scores = None
                    answers.append((request, scores))
            for request, scores in answers:
                self.replies[request.slot].put((request.request_id, scores))


def init_client(requests, replies: list, slots, timeout_s: float = REPLY_TIMEOUT_S) -> None:
    """Инициализатор воркера: занять слот и поставить клиента сервера оценщиком CRAFT процесса.

    Args:
        requests: Общая очередь запросов.
        replies: Очереди ответов по слотам.
        slots: Очередь свободных номеров слотов (по одному на воркер).
        timeout_s: Таймаут ответа.
    """
    try:
        slot = slots.get(timeout=10.0)
    except queue.Empty:
        log.warning(
            "CRAFT-сервер: свободного слота нет (воркер %d) — фильтр псевдоглифов одними правилами", os.getpid()
        )
        set_default_scorer(None)
        return
    set_default_scorer(RemoteCraft(requests, replies[slot], slot, timeout_s))


def make_channels(ctx, workers: int):
    """Очереди канала CRAFT для пула из ``workers`` воркеров.

    Args:
        ctx: Контекст ``multiprocessing`` пула (forkserver).
        workers: Число воркеров (слотов).

    Returns:
        ``(очередь запросов, очереди ответов по слотам, очередь свободных слотов)``.
    """
    requests = ctx.Queue()
    replies = [ctx.Queue() for _ in range(workers)]
    slots = ctx.Queue()
    for slot in range(workers):
        slots.put(slot)
    return requests, replies, slots


def gpu_scorer() -> CraftScorer | None:
    """Оценщик для сервера: CRAFT на CUDA (fp16); без CUDA — ``None`` (воркеры считают CRAFT на CPU сами)."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    scorer = LocalCraft(device="cuda")
    if not scorer.available:
        return None
    scorer.load()
    return scorer


__all__ = ["CraftRequest", "CraftServer", "RemoteCraft", "gpu_scorer", "init_client", "make_channels"]
