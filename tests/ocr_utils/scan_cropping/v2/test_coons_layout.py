"""Тесты холста surya по Кунсу на синтетике: тождество, перенос кривой фигуры, сгущение с двойной точкой на корешке, обратная пара, растр холста, откаты, кэш областей и готовность ``layout.json``, пунктир по длине дуги."""

import json

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2 import coons_layout
from ocr_utils.scan_cropping.v2.coons_layout import (
    SIMPLIFY_MM,
    FramesReason,
    LayoutWarp,
    densify,
    page_points,
    plan_coons,
    unwarp_coons,
    warp_coons,
    warp_polygons_coons,
)
from ocr_utils.scan_cropping.v2.page_frames import PageFrame, frames_digest, load_frames, save_frames

# Синтетика в «мм = px»: поле холста 25 px, шаг сгущения 2 px, допуск упрощения 0.1 px.
PX_PER_MM = 1.0
SHAPE = (400, 500)
SEG = (1.0, 1.0)


def line(a: tuple[float, float], b: tuple[float, float], n: int = 50) -> np.ndarray:
    """Отрезок ``a → b`` из ``n`` точек."""
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) * np.array(a) + t * np.array(b)


def arc(x0: float, x1: float, y: float, bulge: float, n: int = 200) -> np.ndarray:
    """Дуга-парабола от ``x0`` до ``x1`` на высоте ``y`` со стрелкой ``bulge`` px (вниз — положительно)."""
    xs = np.linspace(x0, x1, n)
    s = (xs - x0) / (x1 - x0)
    return np.stack([xs, y + 4 * bulge * s * (1 - s)], axis=1)


def page(x_far: float, x_spine: float, top_bulge: float, bottom_bulge: float) -> PageFrame:
    """Область страницы 200×300 (верх на y = 50, низ на y = 350) с дугами верха и низа."""
    lo, hi = min(x_far, x_spine), max(x_far, x_spine)
    top, bottom = arc(lo, hi, 50.0, top_bulge), arc(lo, hi, 350.0, bottom_bulge)
    if x_far > x_spine:
        top, bottom = top[::-1], bottom[::-1]
    far = line((x_far, top[0, 1]), (x_far, bottom[0, 1]))
    spine = np.array([top[-1], bottom[-1]])
    return PageFrame(top, bottom, far, spine, np.zeros((0, 2)))


def spread(bulge: float) -> list[PageFrame]:
    """Разворот 2×(200×300) с корешком на x = 250: верх и низ страниц — дуги со стрелкой ``bulge``."""
    return [page(50.0, 250.0, bulge, bulge), page(450.0, 250.0, bulge, bulge)]


def plan(frames, margin_mm: float = 25.0):
    """Холст по областям синтетики (должен построиться)."""
    warp, reason = plan_coons(frames, SEG, SHAPE, PX_PER_MM, margin_mm)
    assert reason is FramesReason.OK and warp is not None
    return warp


def sorted_rows(points: np.ndarray) -> np.ndarray:
    """Вершины по возрастанию (x, y) — для сравнения многоугольников без учёта начальной вершины."""
    pts = np.asarray(points, np.float64)
    return pts[np.lexsort((pts[:, 1], pts[:, 0]))]


def polyline_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Расстояния от точек до замкнутой ломаной (до ближайшего ребра)."""
    a = np.asarray(poly, np.float64)
    b = np.roll(a, -1, axis=0)
    d = b - a
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip((rel * d[None]).sum(axis=2) / np.maximum((d**2).sum(axis=1), 1e-12)[None], 0, 1)
    near = a[None] + t[..., None] * d[None]
    return np.hypot(*(points[:, None, :] - near).transpose(2, 0, 1)).min(axis=1)


def test_identity_rectangle_shifts_by_margin():
    """Прямоугольные области: прямоугольник холста переносится в тот же прямоугольник со сдвигом на поле."""
    warp = plan(spread(0.0))
    assert warp.scale == 1.0
    # Холст начинается на 25 px левее внешней стороны левой страницы и на 25 px выше верха.
    rect = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]])
    out = unwarp_coons([rect], warp, PX_PER_MM)[0]
    np.testing.assert_allclose(sorted_rows(out), sorted_rows(rect + 25.0), atol=1e-4)


def test_curved_top_follows_coons_surface():
    """Дуга верха левой страницы: углы фигуры — точки поверхности, стрелка верхнего и нижнего рёбер, допуск упрощения."""
    bulge = 20.0
    frames = [page(50.0, 250.0, bulge, 0.0), page(450.0, 250.0, 0.0, 0.0)]
    warp = plan(frames)
    rect = warp.layout.rects[0]
    u0, u1, v0, v1 = 0.2, 0.8, 0.25, 0.6
    xs = rect.x0 + np.array([u0, u1, u1, u0]) * rect.width
    ys = rect.y0 + np.array([v0, v0, v1, v1]) * rect.height
    out = unwarp_coons([np.stack([xs, ys], axis=1)], warp, PX_PER_MM)[0].astype(np.float64)
    # (а) Углы фигуры — точки поверхности Кунса левой страницы.
    corners = warp.patches[0].points(np.array([u0, u1, u1, u0]), np.array([v0, v0, v1, v1]))
    assert polyline_distance(corners, out).max() < 1e-3
    for c in corners:
        assert np.hypot(*(out - c).T).min() < 1e-3
    # (б) Стрелка рёбер: у дуги верха между долями дуги u0 и u1 — S; у ребра на высоте v — (1 − v)·S.
    top = warp.patches[0].top
    seg = top.at(np.linspace(u0, u1, 400))
    chord = seg[-1] - seg[0]
    normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
    sag_top = np.abs((seg - seg[0]) @ normal).max()
    for v in (v0, v1):
        # Вершины ребра — те, что лежат на образе этого ребра.
        exact = warp.patches[0].points(np.linspace(u0, u1, 200), np.full(200, v))
        on_edge = out[polyline_distance(out, np.vstack([exact, exact[::-1]])) < 0.5]
        a, b = exact[0], exact[-1]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        sag = np.abs((on_edge - a) @ n).max()
        assert abs(sag - (1 - v) * sag_top) < 0.5, (v, sag, (1 - v) * sag_top)
    # (в) Точная кривая верхнего ребра — не дальше допуска упрощения от упрощённой фигуры.
    exact_top = warp.patches[0].points(np.linspace(u0, u1, 500), np.full(500, v0))
    assert polyline_distance(exact_top, out).max() <= SIMPLIFY_MM * PX_PER_MM + 0.05


def test_densify_steps_and_double_spine_point():
    """Сгущение: шаг не больше заданного, углы на месте, у рёбер через корешок — по две точки на нём."""
    square = np.array([[200.0, 100.0], [300.0, 100.0], [300.0, 200.0], [200.0, 200.0]])
    dense = densify(square, 10.0, 250.0)
    steps = np.hypot(*np.diff(np.vstack([dense, dense[:1]]), axis=0).T)
    assert steps.max() <= 10.0 + 1e-9
    for corner in square:
        assert np.hypot(*(dense - corner).T).min() < 1e-12
    on_spine = dense[np.abs(dense[:, 0] - 250.0) < 1e-9]
    assert len(on_spine) == 4
    # Копии различаются стороной: одна левее корешка, другая ровно на нём.
    assert (on_spine[:, 0] < 250.0).sum() == 2 and (on_spine[:, 0] == 250.0).sum() == 2


def test_block_across_spine_uses_both_maps():
    """Блок через корешок: левые точки — по карте левой страницы, правые — по правой, излом — на линии корешка."""
    warp = plan(spread(20.0))
    s = warp.layout.spine_x
    rect = warp.layout.rects[0]
    y0, y1 = rect.y0 + 0.1 * rect.height, rect.y0 + 0.3 * rect.height
    block = np.array([[s - 60, y0], [s + 60, y0], [s + 60, y1], [s - 60, y1]])
    dense = densify(block, 2.0, s)
    left = dense[:, 0] < s
    exact = np.vstack(
        [page_points(warp, 0, dense[left, 0], dense[left, 1]), page_points(warp, 1, dense[~left, 0], dense[~left, 1])]
    )
    out = unwarp_coons([block], warp, PX_PER_MM)[0].astype(np.float64)
    # Каждая вершина итога — образ точки сгущения своей картой.
    for vtx in out:
        assert np.hypot(*(exact - vtx).T).min() < 1e-3
    # На кадре корешок — x = 250: верхнее и нижнее рёбра ломаются на нём (дуги верха страниц сходятся к углу).
    assert (np.abs(out[:, 0] - 250.0) < 1e-3).sum() >= 2


def test_inverse_pair_returns_to_canvas_rectangle():
    """``warp_polygons_coons(unwarp_coons(p))`` ложится на контур исходного прямоугольника холста."""
    warp = plan(spread(20.0))
    rect = warp.layout.rects[0]
    block = np.array(
        [
            [rect.x0 + 30, rect.y0 + 40],
            [rect.x0 + 150, rect.y0 + 40],
            [rect.x0 + 150, rect.y0 + 120],
            [rect.x0 + 30, rect.y0 + 120],
        ]
    )
    frame_poly = unwarp_coons([block], warp, PX_PER_MM)[0]
    back = warp_polygons_coons([frame_poly], warp)[0]
    assert polyline_distance(back, block).max() < 0.01


def test_raster_grid_lines_become_straight(monkeypatch):
    """Растр: линии сетки ``Patch.points`` на кадре на холсте прямые (≤ 1 px на масштабе surya); вне кадра — чёрное."""
    monkeypatch.setattr(coons_layout, "SURYA_MAX_SIDE", 300)
    frames = spread(25.0)
    warp, reason = plan_coons(frames, SEG, SHAPE, PX_PER_MM, 80.0)
    assert reason is FramesReason.OK and 0.5 < warp.scale < 1.0
    image = np.full((*SHAPE, 3), 255, np.uint8)
    vs = (0.25, 0.5, 0.75)
    us = np.linspace(-0.05, 1.05, 400)
    for v in vs:
        pts = warp.patches[0].points(us, np.full(len(us), v))
        cv2.polylines(image, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 0), 3, cv2.LINE_AA)
    canvas = warp_coons(image, warp)
    assert canvas.shape[1::-1] == warp.small_size
    gray = canvas[..., 0].astype(np.float64)
    rect = warp.layout.rects[0]
    # Столбцы малого холста внутри левой страницы (с отступом от углов).
    cols = np.arange(int((rect.x0 + 20 + 0.5) * warp.scale), int((rect.x1 - 20 + 0.5) * warp.scale))
    for v in vs:
        row = (rect.y0 + v * rect.height + 0.5) * warp.scale - 0.5
        lo, hi = int(row) - 6, int(row) + 7
        band = 255.0 - gray[lo:hi, cols]
        centre = (band * np.arange(lo, hi)[:, None]).sum(axis=0) / np.maximum(band.sum(axis=0), 1e-9)
        assert np.abs(centre - row).max() <= 1.0
    # Поле 80 px выходит за кадр слева (внешняя сторона левой страницы — на x = 50, холст левее x = 30 — за кадром):
    # там чёрное; запас 5 px — на билинейную выборку у края кадра.
    outside = int((80 - 50 - 5) * warp.scale)
    assert canvas[:, :outside].max() == 0


def test_fallback_reasons():
    """Без областей или с тремя — ``NOT_TWO_PAGES``; перекрученная область (верх ниже низа) — ``DEGENERATE``."""
    assert plan_coons(None, SEG, SHAPE, PX_PER_MM) == (None, FramesReason.NOT_TWO_PAGES)
    assert plan_coons(spread(0.0) + spread(0.0)[:1], SEG, SHAPE, PX_PER_MM) == (None, FramesReason.NOT_TWO_PAGES)
    good = spread(0.0)
    twisted = PageFrame(good[0].bottom, good[0].top, good[0].far[::-1], good[0].spine[::-1], good[0].knots)
    assert plan_coons([twisted, good[1]], SEG, SHAPE, PX_PER_MM) == (None, FramesReason.DEGENERATE)


def test_frames_cache_roundtrip_and_digest(tmp_path):
    """Области туда и обратно через npz; отпечаток устойчив к шуму < 0.05 px и меняется от сдвига."""
    frames = spread(20.0)
    path = tmp_path / "layout_frames.npz"
    save_frames(path, frames)
    loaded, reason = load_frames(path)
    assert reason == ""
    assert loaded is not None and len(loaded) == 2
    for a, b in zip(frames, loaded):
        for name in ("top", "bottom", "far", "spine", "knots"):
            np.testing.assert_allclose(getattr(a, name), getattr(b, name))
    assert frames_digest(loaded) == frames_digest(frames)
    # Шум меньше 0.05 px вокруг вершин, стоящих на сетке округления 0.1 px, отпечаток не меняет.
    fields = ("top", "bottom", "far", "spine", "knots")
    on_grid = [PageFrame(*(np.round(getattr(f, n), 1) for n in fields)) for f in frames]
    rng = np.random.default_rng(0)
    noisy = [
        PageFrame(*(getattr(f, n) + rng.uniform(-0.04, 0.04, getattr(f, n).shape) for n in fields)) for f in on_grid
    ]
    assert frames_digest(noisy) == frames_digest(on_grid)
    moved = [PageFrame(f.top + 1.0, f.bottom, f.far, f.spine, f.knots) for f in frames]
    assert frames_digest(moved) != frames_digest(frames)
    # Без областей — причина и отпечаток «none».
    save_frames(path, None, FramesReason.FIT_FAILED.value)
    assert load_frames(path) == (None, FramesReason.FIT_FAILED.value)
    assert frames_digest(None) == "none"
    assert load_frames(tmp_path / "missing.npz") == (None, FramesReason.MISSING.value)


def write_layout(path, data: dict) -> None:
    """``layout.json`` с заданным содержимым."""
    path.write_text(json.dumps(data))


def run_params(root, mode: LayoutWarp, redo: bool = False):
    """Параметры прогона с режимом выпрямления для surya (папки — заглушки)."""
    from ocr_utils.scan_cropping.v2.pipeline import RunParams

    return RunParams(root, root, root.parent, layout_warp=mode, redo=redo)


def test_layout_ready_matrix(tmp_path):
    """Готовность surya по версиям 3/4/5 × режимам × совпадению отпечатка областей."""
    from ocr_utils.scan_cropping.v2.pipeline import WorkPaths, layout_ready, layout_warp

    work = WorkPaths(tmp_path)
    frames = spread(0.0)
    save_frames(work.layout_frames, frames)
    digest = frames_digest(frames)
    block = {"blocks": [[[0, 0], [1, 0], [1, 1]]], "labels": ["Text"]}

    # Нет файла.
    assert not layout_ready(work, run_params(tmp_path, LayoutWarp.COONS))
    # Версия 3 — без типов: всегда считать заново.
    write_layout(work.layout, {"version": 3, "keystone": True, **block})
    assert layout_warp(work.layout) is LayoutWarp.KEYSTONE
    assert not any(layout_ready(work, run_params(tmp_path, m)) for m in LayoutWarp)
    # Версия 4 с keystone — годна для keystone, не для coons и none.
    write_layout(work.layout, {"version": 4, "keystone": True, **block})
    assert layout_ready(work, run_params(tmp_path, LayoutWarp.KEYSTONE))
    assert not layout_ready(work, run_params(tmp_path, LayoutWarp.COONS))
    assert not layout_ready(work, run_params(tmp_path, LayoutWarp.NONE))
    write_layout(work.layout, {"version": 4, "keystone": False, **block})
    assert layout_warp(work.layout) is LayoutWarp.NONE
    assert layout_ready(work, run_params(tmp_path, LayoutWarp.NONE))
    # Версия 5, режим coons: годна при совпавшем отпечатке, в том числе при откате на трапеции.
    v5 = {"version": 5, "mode": "coons", "warp": "coons", "reason": "", "frames": digest, **block}
    write_layout(work.layout, v5)
    assert layout_warp(work.layout) is LayoutWarp.COONS
    assert layout_ready(work, run_params(tmp_path, LayoutWarp.COONS))
    assert not layout_ready(work, run_params(tmp_path, LayoutWarp.KEYSTONE))
    write_layout(work.layout, {**v5, "frames": "другой"})
    assert not layout_ready(work, run_params(tmp_path, LayoutWarp.COONS))
    save_frames(work.layout_frames, None, FramesReason.NOT_TWO_PAGES.value)
    write_layout(work.layout, {**v5, "warp": "keystone", "reason": "not_two_pages", "frames": "none"})
    assert layout_ready(work, run_params(tmp_path, LayoutWarp.COONS))
    assert layout_warp(work.layout) is LayoutWarp.KEYSTONE
    # Пересчёт по ``redo``.
    assert not layout_ready(work, run_params(tmp_path, LayoutWarp.COONS, redo=True))
    # Версия 5, режим keystone: годна для keystone.
    write_layout(
        work.layout, {"version": 5, "mode": "keystone", "warp": "keystone", "reason": "", "frames": "", **block}
    )
    assert layout_ready(work, run_params(tmp_path, LayoutWarp.KEYSTONE))
    assert not layout_ready(work, run_params(tmp_path, LayoutWarp.COONS))


def test_dashed_poly_keeps_gaps_on_dense_polygon():
    """Пунктир по сгущённому многоугольнику (400 вершин) остаётся пунктиром: доля закрашенного ≈ DASH / (DASH + GAP)."""
    from ocr_utils.scan_cropping.v2.overlay import DASH_PX, GAP_PX, _dashed_poly

    canvas = np.zeros((600, 600, 3), np.uint8)
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.stack([300 + 250 * np.cos(t), 300 + 250 * np.sin(t)], axis=1)
    _dashed_poly(canvas, circle, (255, 255, 255), 1)
    # Доля закрашенных точек вдоль окружности.
    probe = np.stack(
        [300 + 250 * np.cos(np.linspace(0, 2 * np.pi, 5000)), 300 + 250 * np.sin(np.linspace(0, 2 * np.pi, 5000))],
        axis=1,
    )
    hit = canvas[np.round(probe[:, 1]).astype(int), np.round(probe[:, 0]).astype(int), 0] > 0
    expected = DASH_PX / (DASH_PX + GAP_PX)
    assert abs(hit.mean() - expected) < 0.12


def test_single_page_canvas():
    """Одна область: холст строится, прямоугольник переносится со сдвигом на поле, обратная пара сходится, корешка нет."""
    single = [page(50.0, 250.0, 15.0, 0.0)]
    warp = plan(single)
    assert warp.layout.single and len(warp.patches) == 1
    assert coons_layout.canvas_record(warp)["spine_x"] is None
    flat = plan([page(50.0, 250.0, 0.0, 0.0)])
    rect = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]])
    out = unwarp_coons([rect], flat, PX_PER_MM)[0]
    np.testing.assert_allclose(sorted_rows(out), sorted_rows(rect + 25.0), atol=1e-4)
    # Блок у правого края холста («корешок» одной страницы) — без двойных точек и излома; обратная пара сходится.
    r = warp.layout.rects[0]
    block = np.array([[r.x1 - 60, r.y0 + 40], [r.x1 + 10, r.y0 + 40], [r.x1 + 10, r.y0 + 120], [r.x1 - 60, r.y0 + 120]])
    back = warp_polygons_coons(unwarp_coons([block], warp, PX_PER_MM), warp)[0]
    assert polyline_distance(back, block).max() < 0.05
    canvas = warp_coons(np.full((*SHAPE, 3), 200, np.uint8), warp)
    assert canvas.shape[1::-1] == warp.small_size
