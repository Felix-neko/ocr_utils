"""Тесты CRAFT-службы фильтра псевдоглифов на фиктивной модели (без GPU): запрос и ответ, пул воркеров с клиентом, таймаут с откатом на правила, пачки сервера."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from ocr_utils.page_layout.text_blocks.craft_service import CraftServer, RemoteCraft, init_client, make_channels
from ocr_utils.page_layout.text_blocks.pseudo_glyphs import CraftScorer, LocalCraft, craft_crop, default_scorer


class FakeScorer(CraftScorer):
    """Фиктивная модель: оценка — доля пикселей маски в вырезке; ``delay_s`` — задержка на вырезку."""

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s

    @property
    def available(self) -> bool:
        return True

    def score_crops(self, crops, masks):
        time.sleep(self.delay_s * len(crops))
        return [float(mask.mean()) for mask in masks]


def _crops(n: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``n`` вырезок 20 × 20 с масками разной доли."""
    crops = [np.full((20, 20), 255, np.uint8) for _ in range(n)]
    masks = []
    for i in range(n):
        mask = np.zeros((20, 20), bool)
        mask[: 2 * (i + 1)] = True
        masks.append(mask)
    return crops, masks


def _probe(n: int) -> list[float] | None:
    """Работа воркера пула: оценки ``n`` вырезок оценщиком процесса (клиентом сервера)."""
    crops, masks = _crops(n)
    return default_scorer().score_crops(crops, masks)


def test_request_reply_in_process() -> None:
    """Клиент получает ответы сервера по своему слоту; оценки — по порядку вырезок."""
    ctx = get_context("forkserver")
    requests, replies, _ = make_channels(ctx, 2)
    server = CraftServer(FakeScorer(), requests, replies).start()
    try:
        crops, masks = _crops(3)
        scores = RemoteCraft(requests, replies[1], 1, timeout_s=10).score_crops(crops, masks)
        assert scores == [0.1, 0.2, 0.3]
        assert server.served == 3
    finally:
        server.stop()


def test_pool_workers_use_server() -> None:
    """Воркеры пула forkserver с клиентом (инициализатор ``init_client``) получают оценки сервера главного процесса."""
    ctx = get_context("forkserver")
    requests, replies, slots = make_channels(ctx, 2)
    server = CraftServer(FakeScorer(delay_s=0.01), requests, replies).start()
    try:
        with ProcessPoolExecutor(2, mp_context=ctx, initializer=init_client,
                                 initargs=(requests, replies, slots)) as pool:  # fmt: skip
            results = list(pool.map(_probe, [1, 2, 3, 4, 2, 1]))
        assert results[0] == [0.1] and results[3] == [0.1, 0.2, 0.3, 0.4]
        assert server.served == 13
    finally:
        server.stop()


def test_timeout_falls_back_and_gives_up() -> None:
    """Без сервера клиент ждёт таймаут и отдаёт ``None``; после двух таймаутов подряд — ``None`` сразу."""
    ctx = get_context("forkserver")
    requests, replies, _ = make_channels(ctx, 1)
    client = RemoteCraft(requests, replies[0], 0, timeout_s=0.2)
    crops, masks = _crops(1)
    started = time.monotonic()
    assert client.score_crops(crops, masks) is None
    assert client.score_crops(crops, masks) is None
    assert time.monotonic() - started >= 0.4
    quick = time.monotonic()
    assert client.score_crops(crops, masks) is None
    assert time.monotonic() - quick < 0.1


def test_stale_reply_is_skipped() -> None:
    """Опоздавший ответ на прежний запрос выбрасывается, клиент ждёт свой."""
    ctx = get_context("forkserver")
    requests, replies, _ = make_channels(ctx, 1)
    replies[0].put((999, [0.5]))
    server = CraftServer(FakeScorer(), requests, replies).start()
    try:
        crops, masks = _crops(2)
        assert RemoteCraft(requests, replies[0], 0, timeout_s=10).score_crops(crops, masks) == [0.1, 0.2]
    finally:
        server.stop()


def test_server_survives_scorer_error() -> None:
    """Ошибка модели — ответ ``None`` клиенту, сервер работает дальше."""

    class Broken(FakeScorer):
        def score_crops(self, crops, masks):
            if len(crops) == 1:
                raise RuntimeError("сбой")
            return super().score_crops(crops, masks)

    ctx = get_context("forkserver")
    requests, replies, _ = make_channels(ctx, 1)
    server = CraftServer(Broken(), requests, replies).start()
    try:
        client = RemoteCraft(requests, replies[0], 0, timeout_s=10)
        one, two = _crops(1), _crops(2)
        assert client.score_crops(*one) is None
        assert client.score_crops(*two) == [0.1, 0.2]
    finally:
        server.stop()


def test_craft_crop_window_and_mask() -> None:
    """Вырезка — рамка с полем ``CRAFT_CONTEXT_MM`` (обрезанная краем), маска — только своя компонента."""
    gray = np.full((400, 400), 255, np.uint8)
    labels = np.zeros((400, 400), np.int32)
    labels[10:20, 10:15] = 3
    labels[30:35, 30:35] = 4
    crop, mask = craft_crop(gray, labels, 3, (10, 10, 15, 20))
    assert crop.shape == mask.shape and mask.sum() == 50
    # Поле 8 мм при 300 dpi ≈ 94.5 px: сверху обрезано краем (0), снизу до 20 + 94 = 114.
    assert crop.shape[0] == 114 and crop.shape[1] == 109


def test_local_craft_is_a_scorer() -> None:
    """Локальный оценщик — та же реализация интерфейса (модель не грузится до первой оценки)."""
    scorer = LocalCraft()
    assert isinstance(scorer, CraftScorer) and scorer._net is None


def test_filter_without_server_answer_is_rules() -> None:
    """Оценщик не ответил (``None``) — проход фильтра идёт одними правилами и не падает."""
    import cv2

    from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
    from ocr_utils.page_layout.text_blocks.page import analyse_gray
    from ocr_utils.page_layout.text_blocks.pseudo_glyphs import PseudoMode, PseudoParams, filter_pass

    class Silent(FakeScorer):
        def score_crops(self, crops, masks):
            return None

    gray = np.full((600, 900), 255, np.uint8)
    for i in range(8):
        cv2.putText(gray, "the quick brown fox jumps", (40, 80 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3)
    gray[40:500, 0:2] = 100
    gray[200:210, 120:126] = 140  # серая точка в поле строки — спорная
    analysis = analyse_gray(gray, InkEngine())
    rules_gray, rules_drops, _ = filter_pass(analysis, gray, gray, PseudoParams())
    silent_gray, silent_drops, calls = filter_pass(analysis, gray, gray, PseudoParams(craft_doubt=0.4), Silent())
    assert calls == 0 and np.array_equal(rules_gray, silent_gray)
    assert [d.box for d in rules_drops] == [d.box for d in silent_drops]
    assert analyse_gray(gray, InkEngine(), pseudo=PseudoMode.RULES).pseudo_glyphs.drops
