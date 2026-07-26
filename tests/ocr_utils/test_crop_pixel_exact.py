"""Тесты вырезки crop-зоны способом pixel-exact (``--crop-mode=pixel-exact``).

Главное свойство режима — содержимое crop-зоны копируется БЕЗ ресэмплинга, поэтому
проверяем побитовое совпадение с исходным кадром, а не «похожесть». Остальное —
геометрия холста и заполнение «ушей» (Вороной + размытие + выцветание).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ocr_utils.detect_and_crop import (
    CROP_FILL_REPLICATE,
    CROP_FILL_VORONOI,
    _bbox_corners,
    _clamp_to_edge,
    _rotation_matrix,
    book_mean_color,
    crop_pixel_exact,
)


def _noise_frame(h: int = 400, w: int = 600, seed: int = 0) -> np.ndarray:
    """Кадр из некоррелированного шума: любая интерполяция сразу видна как расхождение."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _box_mask(shape: "tuple[int, int]", corners: np.ndarray) -> np.ndarray:
    """Маска повёрнутого crop-bbox в координатах холста (uint8 0/255)."""
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(m, [np.round(corners).astype(np.int32)], 255)
    return m


class TestGeometry:
    """Размер холста и положение crop-зоны в нём."""

    def test_zero_angle_gives_exact_slice(self) -> None:
        """Угол 0 — «уши» вырождаются, холст равен crop-зоне, содержимое — обычный срез."""
        frame = _noise_frame()
        cx, cy = 300.0, 200.0
        ext = (-100.0, -60.0, 100.0, 60.0)

        out = crop_pixel_exact(frame, cx, cy, 0.0, ext)

        assert out.shape == (120, 200, 3)
        assert np.array_equal(out, frame[140:260, 200:400])

    def test_canvas_is_minimal_aabb_of_rotated_box(self) -> None:
        """Холст — минимальный осевой bbox, в который вписан повёрнутый crop-bbox."""
        frame = _noise_frame()
        cx, cy, angle = 300.0, 200.0, 20.0
        ext = (-100.0, -60.0, 100.0, 60.0)

        out = crop_pixel_exact(frame, cx, cy, angle, ext)

        corners = _bbox_corners(cx, cy, angle, ext)
        exp_w = int(np.ceil(corners[:, 0].max())) - int(np.floor(corners[:, 0].min()))
        exp_h = int(np.ceil(corners[:, 1].max())) - int(np.floor(corners[:, 1].min()))
        assert out.shape[:2] == (exp_h, exp_w)
        # Наклонённая зона в осевой холст «в упор» не влезает — холст обязан быть больше неё.
        assert exp_w > 200 and exp_h > 120

    @pytest.mark.parametrize("angle", [-30.0, -7.0, 3.0, 25.0])
    def test_canvas_contains_whole_rotated_box(self, angle: float) -> None:
        """При любом угле повёрнутый bbox целиком лежит внутри холста."""
        frame = _noise_frame()
        cx, cy = 300.0, 200.0
        ext = (-90.0, -50.0, 90.0, 50.0)

        out = crop_pixel_exact(frame, cx, cy, angle, ext)

        corners = _bbox_corners(cx, cy, angle, ext)
        local = corners - np.array([np.floor(corners[:, 0].min()), np.floor(corners[:, 1].min())], dtype=np.float32)
        assert local.min() >= 0
        assert local[:, 0].max() <= out.shape[1]
        assert local[:, 1].max() <= out.shape[0]


class TestPixelExactness:
    """Содержимое crop-зоны не должно меняться ни на один бит."""

    @pytest.mark.parametrize("angle", [-25.0, -11.0, 6.0, 17.0])
    def test_inside_box_is_bit_exact_copy(self, angle: float) -> None:
        """Каждый пиксель внутри повёрнутого bbox равен исходному со сдвигом на угол холста."""
        frame = _noise_frame()
        cx, cy = 300.0, 200.0
        ext = (-100.0, -60.0, 100.0, 60.0)

        out = crop_pixel_exact(frame, cx, cy, angle, ext)

        corners = _bbox_corners(cx, cy, angle, ext)
        x0, y0 = int(np.floor(corners[:, 0].min())), int(np.floor(corners[:, 1].min()))
        inside = _box_mask(out.shape[:2], corners - np.array([x0, y0], dtype=np.float32)) > 0
        # Сдвинутый исходник в координатах холста
        shifted = frame[y0 : y0 + out.shape[0], x0 : x0 + out.shape[1]]
        assert shifted.shape == out.shape  # зона целиком внутри кадра
        assert np.array_equal(out[inside], shifted[inside])

    def test_rotation_is_not_applied(self) -> None:
        """Книга остаётся НАКЛОНЁННОЙ: наклонная линия в кадре остаётся наклонной в выводе."""
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        # Горизонтальная в системе crop-зоны линия (наклонена на 15° в кадре)
        angle = 15.0
        cx, cy = 300.0, 200.0
        rad = np.deg2rad(angle)
        for t in np.arange(-80, 80, 0.25):
            x = int(round(cx + t * np.cos(rad)))
            y = int(round(cy + t * np.sin(rad)))
            frame[y, x] = (255, 255, 255)

        out = crop_pixel_exact(frame, cx, cy, angle, (-100.0, -60.0, 100.0, 60.0), fade_color=None, blur_px=0.0)

        ys, xs = np.where(out[:, :, 0] > 200)
        # Наклон сохранён: линия не превратилась в горизонталь (разброс y соизмерим с наклоном)
        assert ys.max() - ys.min() > 20
        slope = np.polyfit(xs, ys, 1)[0]
        assert slope == pytest.approx(np.tan(rad), abs=0.05)


class TestEarsFill:
    """«Уши» между наклонённым bbox и осевым холстом."""

    def _setup(self, angle: float = 20.0) -> tuple:
        frame = np.full((400, 600, 3), 200, dtype=np.uint8)
        frame[:, :, 0] = 180  # слегка цветная «бумага», чтобы отличать каналы
        cx, cy = 300.0, 200.0
        ext = (-100.0, -60.0, 100.0, 60.0)
        corners = _bbox_corners(cx, cy, angle, ext)
        x0, y0 = int(np.floor(corners[:, 0].min())), int(np.floor(corners[:, 1].min()))
        return frame, cx, cy, angle, ext, corners - np.array([x0, y0], dtype=np.float32)

    @pytest.mark.parametrize("method", [CROP_FILL_REPLICATE, CROP_FILL_VORONOI])
    def test_ears_are_filled_not_black(self, method: str) -> None:
        """Углы холста заполнены заливкой, а не остаются чёрными дырами."""
        frame, cx, cy, angle, ext, local = self._setup()

        out = crop_pixel_exact(
            frame, cx, cy, angle, ext, fade_color=None, blur_px=0.0, fade_strength=0.0, fill_method=method
        )

        ears = _box_mask(out.shape[:2], local) == 0
        assert ears.any()
        assert out[ears].min() > 0
        # Без выцветания и размытия наружу тянется цвет края страницы
        assert np.allclose(out[ears], (180, 200, 200), atol=1)

    def test_fade_pulls_far_pixels_to_mean_color(self) -> None:
        """С fade_strength=1 самый дальний от crop-зоны пиксель уходит в средний цвет."""
        frame, cx, cy, angle, ext, local = self._setup()
        fade_color = np.array([10.0, 20.0, 30.0])  # заведомо непохожий на «бумагу»

        out = crop_pixel_exact(frame, cx, cy, angle, ext, fade_color=fade_color, blur_px=0.0, fade_strength=1.0)

        box = _box_mask(out.shape[:2], local)
        dist = cv2.distanceTransform((box == 0).astype(np.uint8), cv2.DIST_L2, 3)
        fy, fx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        assert out[fy, fx] == pytest.approx(fade_color, abs=3)

    def test_fade_grows_monotonically_with_distance(self) -> None:
        """Выцветание тем сильнее, чем дальше от crop-зоны."""
        frame, cx, cy, angle, ext, local = self._setup()
        fade_color = np.array([0.0, 0.0, 0.0])

        out = crop_pixel_exact(frame, cx, cy, angle, ext, fade_color=fade_color, blur_px=0.0, fade_strength=1.0)

        box = _box_mask(out.shape[:2], local)
        dist = cv2.distanceTransform((box == 0).astype(np.uint8), cv2.DIST_L2, 3)
        ears = box == 0
        d, brightness = dist[ears], out[ears].mean(axis=1)
        # Ближние к зоне пиксели ярче дальних (выцветание в чёрный растёт с расстоянием)
        near = brightness[d <= np.percentile(d, 20)].mean()
        far = brightness[d >= np.percentile(d, 80)].mean()
        assert near > far
        assert np.corrcoef(d, brightness)[0, 1] < -0.9

    def test_fade_strength_zero_keeps_edge_color(self) -> None:
        """fade_strength=0 — выцветания нет, «уши» остаются цвета края."""
        frame, cx, cy, angle, ext, local = self._setup()

        out = crop_pixel_exact(
            frame, cx, cy, angle, ext, fade_color=np.array([0.0, 0.0, 0.0]), blur_px=0.0, fade_strength=0.0
        )

        ears = _box_mask(out.shape[:2], local) == 0
        assert np.allclose(out[ears], (180, 200, 200), atol=1)

    def test_fill_does_not_touch_crop_zone(self) -> None:
        """Заливка (в т.ч. размытие и выцветание) не залезает внутрь crop-зоны."""
        frame = _noise_frame()
        cx, cy, angle = 300.0, 200.0, 20.0
        ext = (-100.0, -60.0, 100.0, 60.0)

        out = crop_pixel_exact(
            frame, cx, cy, angle, ext, fade_color=np.array([0.0, 0.0, 0.0]), blur_px=40.0, fade_strength=1.0
        )

        corners = _bbox_corners(cx, cy, angle, ext)
        x0, y0 = int(np.floor(corners[:, 0].min())), int(np.floor(corners[:, 1].min()))
        inside = _box_mask(out.shape[:2], corners - np.array([x0, y0], dtype=np.float32)) > 0
        shifted = frame[y0 : y0 + out.shape[0], x0 : x0 + out.shape[1]]
        assert np.array_equal(out[inside], shifted[inside])

    def _ear_roughness(self, out: np.ndarray, local: np.ndarray, dist_from: float = 60.0) -> float:
        """Локальный контраст (средний |лапласиан|) в дальней части «ушей»."""
        box = _box_mask(out.shape[:2], local)
        dist = cv2.distanceTransform((box == 0).astype(np.uint8), cv2.DIST_L2, 3)
        far = (box == 0) & (dist >= np.percentile(dist[box == 0], dist_from))
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.abs(cv2.Laplacian(gray, cv2.CV_32F))[far].mean())

    def test_blur_smooths_ears(self) -> None:
        """Размытие сглаживает заливку: с ним фактура в дальней части «ушей» глаже."""
        # Шумный край страницы → Вороной растащит шум полосами по «ушам»
        frame = _noise_frame()
        cx, cy, angle = 300.0, 200.0, 20.0
        ext = (-100.0, -60.0, 100.0, 60.0)
        corners = _bbox_corners(cx, cy, angle, ext)
        local = corners - np.array([np.floor(corners[:, 0].min()), np.floor(corners[:, 1].min())], dtype=np.float32)

        sharp = crop_pixel_exact(frame, cx, cy, angle, ext, fade_color=None, blur_px=0.0, fade_strength=0.0)
        blurred = crop_pixel_exact(frame, cx, cy, angle, ext, fade_color=None, blur_px=40.0, fade_strength=0.0)

        assert self._ear_roughness(blurred, local) < 0.5 * self._ear_roughness(sharp, local)

    def test_blur_grows_with_distance(self) -> None:
        """Размытие тем сильнее, чем дальше: у шва фактура резче, чем в глубине «ушей»."""
        frame = _noise_frame()
        cx, cy, angle = 300.0, 200.0, 20.0
        ext = (-100.0, -60.0, 100.0, 60.0)
        corners = _bbox_corners(cx, cy, angle, ext)
        local = corners - np.array([np.floor(corners[:, 0].min()), np.floor(corners[:, 1].min())], dtype=np.float32)

        out = crop_pixel_exact(frame, cx, cy, angle, ext, fade_color=None, blur_px=40.0, fade_strength=0.0)

        box = _box_mask(out.shape[:2], local)
        dist = cv2.distanceTransform((box == 0).astype(np.uint8), cv2.DIST_L2, 3)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        ears = box == 0
        near = ears & (dist <= np.percentile(dist[ears], 25))
        far = ears & (dist >= np.percentile(dist[ears], 75))
        assert lap[near].mean() > lap[far].mean()


class TestReplicateFill:
    """Продление краевых пикселей по нормали к сторонам crop-зоны (--crop-fill-method=replicate)."""

    def _frame_with_gutter(self, angle: float, u_off: float = 0.0) -> tuple:
        """Кадр со «светлой бумагой» и тёмной линией корешка вдоль оси v crop-зоны."""
        frame = np.full((800, 800, 3), 230, dtype=np.uint8)
        cx = cy = 400.0
        ext = (-200.0, -150.0, 200.0, 150.0)
        r = _rotation_matrix(angle)

        def to_frame(u: float, v: float) -> "tuple[int, int]":
            p = np.array([u, v]) @ r + np.array([cx, cy])
            return int(round(p[0])), int(round(p[1]))

        cv2.line(frame, to_frame(u_off, -400), to_frame(u_off, 400), (0, 0, 0), 5)
        return frame, cx, cy, ext

    def _ear_line_slope(self, out: np.ndarray, local: np.ndarray) -> "tuple[float, int]":
        """Наклон dx/dy тёмной линии в верхнем «ухе» и число её пикселей."""
        box = _box_mask(out.shape[:2], local)
        h = out.shape[0]
        ear_top = (box == 0) & (np.arange(h)[:, None] < h // 2)
        ys, xs = np.where(ear_top & (out[:, :, 0] < 128))
        if len(xs) < 20:
            return float("nan"), len(xs)
        return float(np.polyfit(ys, xs, 1)[0]), len(xs)

    @pytest.mark.parametrize("angle", [5.0, 10.0, 20.0])
    def test_gutter_continues_at_book_tilt(self, angle: float) -> None:
        """Линия корешка продолжается в «ухе» под тем же наклоном, под которым лежит книга."""
        frame, cx, cy, ext = self._frame_with_gutter(angle)
        corners = _bbox_corners(cx, cy, angle, ext)
        local = corners - np.array([np.floor(corners[:, 0].min()), np.floor(corners[:, 1].min())], dtype=np.float32)

        out = crop_pixel_exact(
            frame, cx, cy, angle, ext, fade_color=None, blur_px=0.0, fade_strength=0.0, fill_method=CROP_FILL_REPLICATE
        )

        slope, count = self._ear_line_slope(out, local)
        assert count > 50
        # Нормаль к верхней стороне наклонена ровно на угол поворота книги
        assert slope == pytest.approx(np.tan(np.deg2rad(angle)), abs=0.03)

    def test_replicate_is_perpendicular_projection(self) -> None:
        """Пиксель «уха» берётся с проекции по нормали на сторону, а не с ближайшей точки.

        На линейном градиенте проверяем аналитически: значение равно значению исходного
        кадра в точке, зажатой в границы crop-зоны в её собственных осях.
        """
        h, w, angle = 500, 700, 18.0
        cx, cy = 350.0, 250.0
        ext = (-150.0, -110.0, 150.0, 110.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        frame = np.stack([xx * 0.2 + 20, yy * 0.2 + 20, np.full_like(xx, 128)], axis=2).astype(np.uint8)

        out = crop_pixel_exact(
            frame, cx, cy, angle, ext, fade_color=None, blur_px=0.0, fade_strength=0.0, fill_method=CROP_FILL_REPLICATE
        )

        corners = _bbox_corners(cx, cy, angle, ext)
        x0, y0 = int(np.floor(corners[:, 0].min())), int(np.floor(corners[:, 1].min()))
        box = _box_mask(out.shape[:2], corners - np.array([x0, y0], dtype=np.float32))
        r = _rotation_matrix(angle)
        ys, xs = np.where(box == 0)
        rel = np.stack([xs + x0 - cx, ys + y0 - cy], axis=1)
        local = rel @ r.T
        local[:, 0] = np.clip(local[:, 0], ext[0], ext[2])
        local[:, 1] = np.clip(local[:, 1], ext[1], ext[3])
        src = local @ r + np.array([cx, cy])
        # Кадр — линейный градиент, поэтому ожидаемое значение считается аналитически
        expected_b = np.clip(src[:, 0] * 0.2 + 20, 0, 255)
        got_b = out[ys, xs, 0].astype(np.float64)
        # Края «уха» задевают границу кадра/растеризацию — сравниваем робастно
        assert np.median(np.abs(got_b - expected_b)) < 2.0
        assert np.mean(np.abs(got_b - expected_b) < 4.0) > 0.95

    def test_voronoi_fans_out_at_convex_corner_replicate_does_not(self) -> None:
        """Регрессия на исходный дефект (IMG_0004/IMG_0034): веер у выпуклого угла книги.

        Для ЧИСТОГО прямоугольника оба способа совпадают (проекция на ближайшую точку
        прямоугольника — это и есть зажим координат). Разница появляется, когда известная
        область — криволинейный силуэт книги: у выпуклого угла (верх страницы у корешка)
        Вороной тянет цвет ОДНОЙ точки на целую область и размазывает линию корешка веером,
        а clamp-to-edge продолжает её прямо.
        """
        angle, cx, cy = 12.0, 400.0, 400.0
        ext = (-200.0, -150.0, 200.0, 150.0)
        r = _rotation_matrix(angle)

        def to_frame(u: float, v: float) -> "tuple[int, int]":
            p = np.array([u, v]) @ r + np.array([cx, cy])
            return int(round(p[0])), int(round(p[1]))

        frame = np.full((800, 800, 3), 230, dtype=np.uint8)
        cv2.line(frame, to_frame(0, -120), to_frame(0, 300), (0, 0, 0), 5)  # корешок
        # Силуэт книги с выпуклой «крышей»: вершина ровно над корешком
        content = np.zeros((800, 800), dtype=np.uint8)
        roof = np.array(
            [to_frame(-200, 150), to_frame(200, 150), to_frame(200, -60), to_frame(0, -120), to_frame(-200, -60)],
            dtype=np.int32,
        )
        cv2.fillPoly(content, [roof], 255)

        corners = _bbox_corners(cx, cy, angle, ext)
        x0, y0 = int(np.floor(corners[:, 0].min())), int(np.floor(corners[:, 1].min()))
        stats = {}
        for method in (CROP_FILL_REPLICATE, CROP_FILL_VORONOI):
            out = crop_pixel_exact(
                frame,
                cx,
                cy,
                angle,
                ext,
                fade_color=None,
                blur_px=0.0,
                fade_strength=0.0,
                fill_method=method,
                content_mask=content,
            )
            box = _box_mask(out.shape[:2], corners - np.array([x0, y0], dtype=np.float32))
            cont = content[y0 : y0 + out.shape[0], x0 : x0 + out.shape[1]]
            zone = (box > 0) & (cont == 0)  # зона экстраполяции внутри crop-bbox
            ys, xs = np.where(zone & (out[:, :, 0] < 128))
            stats[method] = (len(xs), float(np.polyfit(ys, xs, 1)[0]))

        rep_count, rep_slope = stats[CROP_FILL_REPLICATE]
        vor_count, _ = stats[CROP_FILL_VORONOI]
        # replicate продолжает корешок прямо, под наклоном книги
        assert rep_slope == pytest.approx(np.tan(np.deg2rad(angle)), abs=0.05)
        # Вороной размазывает его веером — тёмных пикселей заметно больше
        assert vor_count > 1.5 * rep_count


class TestClampToEdge:
    """Ядро replicate-заливки: продление краевых пикселей по осям."""

    def test_takes_nearest_known_along_row_when_column_ends_early(self) -> None:
        """Регрессия (IMG_0042): если столбец известен только сверху — берём соседа по строке.

        Граница книги криволинейна, и у нижних строк она уходит правее края crop-зоны.
        Раньше такие пиксели всё равно продлевались вертикально и брали цвет далёкого
        верхнего пикселя — вдоль левого «уха» шла резкая светлая полоса.
        """
        h, w = 40, 20
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 1] = np.arange(h)[:, None] * 5  # значение = 5·строка: видно, откуда взяли
        known = np.zeros((h, w), dtype=np.uint8)
        known[:, 10:] = 255  # правая половина известна везде
        known[:10, 5:10] = 255  # левее — только верхние 10 строк

        out = _clamp_to_edge(img, known)

        # Строка 35: ближайший известный — сбоку (x=10), а не сверху (строка 9)
        assert out[35, 0:10, 1].tolist() == [175] * 10
        # Строка 5: слева от известного — сосед по строке
        assert out[5, 0:5, 1].tolist() == [25] * 5
        # Строка 12, x=7: до известного вверх 3 строки, вбок 3 колонки — вертикаль не хуже
        assert out[12, 7, 1] == 45

    def test_pure_rectangle_extends_along_columns(self) -> None:
        """Когда известное — сплошной прямоугольник, продление идёт строго по осям."""
        h, w = 30, 30
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = np.arange(w)[None, :]
        img[:, :, 1] = np.arange(h)[:, None]
        known = np.zeros((h, w), dtype=np.uint8)
        known[10:20, 5:25] = 255

        out = _clamp_to_edge(img, known)

        # Выше прямоугольника — цвет его верхней строки, ниже — нижней
        assert np.array_equal(out[0, 5:25, 1], np.full(20, 10))
        assert np.array_equal(out[29, 5:25, 1], np.full(20, 19))
        # Левее/правее — цвет крайней колонки, с сохранением своей строки
        assert np.array_equal(out[10:20, 0, 0], np.full(10, 5))
        assert np.array_equal(out[10:20, 29, 0], np.full(10, 24))

    def test_seam_is_continuous_when_content_cuts_crop_zone(self) -> None:
        """Шов не даёт скачка, когда E2 срезает край crop-зоны (сквозь crop_pixel_exact)."""
        angle, cx, cy = 3.0, 400.0, 400.0
        ext = (-200.0, -180.0, 200.0, 180.0)
        rng = np.random.default_rng(3)
        frame = (rng.integers(-6, 7, size=(800, 800, 3)) + 200).clip(0, 255).astype(np.uint8)
        # Силуэт книги, уходящий вправо к низу: у нижних строк он правее края crop-зоны
        content = np.zeros((800, 800), dtype=np.uint8)
        cv2.fillPoly(content, [np.array([[150, 0], [800, 0], [800, 800], [330, 800]], dtype=np.int32)], 255)

        out = crop_pixel_exact(
            frame,
            cx,
            cy,
            angle,
            ext,
            fade_color=None,
            blur_px=0.0,
            fade_strength=0.0,
            fill_method=CROP_FILL_REPLICATE,
            content_mask=content,
        )

        corners = _bbox_corners(cx, cy, angle, ext)
        x0, y0 = int(np.floor(corners[:, 0].min())), int(np.floor(corners[:, 1].min()))
        cont = content[y0 : y0 + out.shape[0], x0 : x0 + out.shape[1]]
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # На каждой строке сравниваем заливку у шва с первым известным пикселем
        jumps = []
        for y in range(30, out.shape[0] - 30, 5):
            known_x = np.where(cont[y] > 0)[0]
            if len(known_x) == 0 or known_x[0] < 6:
                continue
            s = int(known_x[0])
            jumps.append(abs(float(gray[y, s]) - float(gray[y, s - 4])))
        assert jumps
        # Разрыва нет: заливка у шва совпадает с краем контента (допуск — зерно бумаги)
        assert np.median(jumps) < 8
        assert np.percentile(jumps, 95) < 20


class TestOutOfFrame:
    """Crop-зона, вылезающая за границы кадра (положительные припуски)."""

    def test_area_outside_frame_is_filled(self) -> None:
        """Часть холста вне кадра не остаётся чёрной, а заполняется как «уши»."""
        frame = np.full((400, 600, 3), 200, dtype=np.uint8)
        cx, cy, angle = 60.0, 50.0, 10.0  # зона свисает за левый верхний угол кадра
        ext = (-120.0, -100.0, 120.0, 100.0)

        out = crop_pixel_exact(frame, cx, cy, angle, ext, fade_color=None, blur_px=0.0, fade_strength=0.0)

        assert out.min() > 0  # чёрных дыр нет
        assert np.allclose(out, 200, atol=1)

    def test_zone_fully_outside_frame_does_not_crash(self) -> None:
        """Зона целиком вне кадра: не падаем, возвращаем холст нужного размера."""
        frame = np.full((400, 600, 3), 200, dtype=np.uint8)

        out = crop_pixel_exact(frame, -500.0, -500.0, 10.0, (-50.0, -50.0, 50.0, 50.0))

        assert out.shape[0] > 0 and out.shape[1] > 0


class TestBookMeanColor:
    """Средний цвет книги — цель выцветания."""

    def test_mean_ignores_noisy_edge_thanks_to_erosion(self) -> None:
        """Эрозия отсекает кайму: среднее берётся по «чистой бумаге» внутри маски."""
        bgr = np.zeros((400, 600, 3), dtype=np.uint8)
        mask = np.zeros((400, 600), dtype=np.uint8)
        mask[50:350, 100:500] = 255
        bgr[50:350, 100:500] = (100, 150, 200)  # «бумага»
        bgr[50:70, 100:500] = (0, 0, 0)  # тёмная кайма у края маски
        bgr[330:350, 100:500] = (0, 0, 0)

        color = book_mean_color(bgr, mask, erosion_px=40, work_side=600)

        assert color == pytest.approx((100, 150, 200), abs=1)

    def test_empty_mask_returns_none(self) -> None:
        """Пустая маска — среднего цвета нет."""
        assert book_mean_color(np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100), np.uint8)) is None

    def test_erosion_eating_mask_falls_back_to_mask(self) -> None:
        """Если эрозия съела маску целиком — берём исходную маску, а не падаем."""
        bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[48:52, 40:60] = 255  # узкая полоска, эрозия на 40 px её уничтожит

        color = book_mean_color(bgr, mask, erosion_px=40, work_side=100)

        assert color == pytest.approx((128, 128, 128), abs=1)
