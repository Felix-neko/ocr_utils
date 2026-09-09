"""Ограничение числа заданий в работе: результаты по порядку и без роста очереди."""

from __future__ import annotations

from concurrent.futures import Future

from ocr_utils.scan_markup.orientation import analysis


class CountingPool:
    """Подставной пул: считает, сколько заданий висит в работе одновременно."""

    def __init__(self):
        self.in_flight = 0
        self.peak = 0
        self.order = []

    def submit(self, function, job):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.order.append(job)
        future = Future()
        future.set_result(job)
        original = future.result

        def result(*args, **kwargs):
            self.in_flight -= 1
            return original(*args, **kwargs)

        future.result = result
        return future


def test_bounded_keeps_the_queue_short_and_the_order_intact():
    """Без границы pool.map копил готовые результаты, пока родитель их разбирал, —
    на паке-1 это дало 8 ГиБ на четверти прогона."""
    pool = CountingPool()
    jobs = list(range(50))
    got = list(analysis._bounded(pool, jobs, in_flight=6))
    assert got == jobs, "порядок результатов должен совпадать с порядком заданий"
    assert pool.peak <= 6
    assert pool.order == jobs, "каждое задание отправлено ровно один раз"


def test_bounded_handles_fewer_jobs_than_the_limit():
    pool = CountingPool()
    assert list(analysis._bounded(pool, [1, 2], in_flight=10)) == [1, 2]


def test_bounded_on_an_empty_list_does_nothing():
    pool = CountingPool()
    assert list(analysis._bounded(pool, [], in_flight=4)) == []
    assert pool.order == []


def test_allowed_angles_reach_the_worker(tmp_path):
    """Набор допустимых углов обязан доехать до кадра в воркере.

    Проводка длинная — CLI → analyse → Job → read_frame → Frame, — и разрыв в любом её звене
    не заметен: детектор просто продолжает рассматривать все четыре поворота, как раньше.
    """
    import cv2

    from ocr_utils.scan_markup.orientation.detectors.base import rotate_cw
    from tests.ocr_utils.scan_markup.orientation.synthetic import text_page

    page = tmp_path / "IMG_0001_1L.png"
    cv2.imwrite(str(page), rotate_cw(text_page(), 270))
    results = analysis.analyse(
        [(page, "1967/01/IMG_0001_1L.png")],
        analysis.DETECTORS and [analysis.DETECTORS["ink_axis"]],
        workers=1,
        default_dpi=300,
        progress=False,
        allowed=(0, 90),
    )
    verdict = results[0].verdicts["ink_axis"]
    assert verdict.rotate_cw in (0, 90), "ответ вне набора допустимых углов"
