"""Нарезка областей строк на куски: главное — устойчивость к наклону строк."""

import math

import numpy as np
import pytest

from ocr_utils.defocus_detection.lines.regions import MAX_SLANT_LOSS, Chunk, LineRegion, line_chunks

FRAME = (1000, 1000)


def slanted_line(angle_deg: float, length: float = 600.0, height: float = 20.0, x0: float = 100.0, y0: float = 300.0):
    """Строит область строки с заданным наклоном.

    Args:
        angle_deg: Наклон в градусах.
        length: Длина строки по горизонтали.
        height: Высота строки.
        x0: Левый край.
        y0: Верх левого края.

    Returns:
        ``LineRegion``.
    """
    dy = math.tan(math.radians(angle_deg)) * length
    return LineRegion(
        polygon=np.array(
            [[x0, y0], [x0 + length, y0 + dy], [x0 + length, y0 + dy + height], [x0, y0 + height]], dtype=np.float64
        )
    )


@pytest.mark.parametrize("angle", [0.0, 1.0, 3.0, -3.0, 6.0])
def test_height_does_not_grow_with_slant(angle):
    """Высота строки — это кегль, а не вертикальный габарит наклонного полигона.

    Если бы высота мерилась по охватывающему прямоугольнику, у наклонной строки она
    выросла бы на весь снос — и фильтр «мелкий текст» выбросил бы наклонные строки,
    то есть целый край кадра при трапециевидных искажениях.
    """
    assert slanted_line(angle).height == pytest.approx(20.0, abs=0.01)


def test_corners_survive_slant_exceeding_line_height():
    """Углы канонизируются даже когда снос от наклона больше высоты строки.

    Это не экзотика, а норма: строка газетной колонки длиной 600 px при наклоне 3°
    сносится на 31 px при высоте 20 px. Наивная сортировка «сначала по Y» относит тогда
    оба левых угла к верхнему ребру.
    """
    region = slanted_line(3.0)
    tl, tr, br, bl = region.corners()
    assert tl[0] < tr[0] and bl[0] < br[0], "левые углы должны быть левее правых"
    assert tl[1] < bl[1] and tr[1] < br[1], "верхние углы должны быть выше нижних"


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [2, 3, 0, 1], [3, 2, 1, 0], [1, 2, 3, 0]])
def test_corners_do_not_depend_on_input_order(order):
    """Порядок углов от surya не документирован, поэтому он не должен ни на что влиять."""
    region = slanted_line(3.0)
    shuffled = LineRegion(polygon=region.polygon[order])
    for a, b in zip(region.corners(), shuffled.corners()):
        assert a == pytest.approx(b)


@pytest.mark.parametrize("angle", [0.0, 1.0, 3.0, -3.0, 6.0])
def test_chunks_lie_strictly_inside_polygon(angle):
    """Каждый кусок целиком внутри полигона строки — ни пикселя бумаги сверху и снизу.

    Это и есть причина резать строку на куски: у наклонной строки охватывающий
    прямоугольник прихватил бы бумагу и выносные элементы соседних строк тем сильнее,
    чем больше наклон.
    """
    region = slanted_line(angle)
    tl, tr, br, bl = region.corners()

    def edge_y(x, left, right):
        return left[1] + (right[1] - left[1]) * (x - left[0]) / (right[0] - left[0])

    chunks = line_chunks(region, FRAME)
    assert chunks, "куски должны находиться"
    for chunk in chunks:
        for x in (chunk.x1, chunk.x2):
            assert chunk.y1 >= edge_y(x, tl, tr) - 1e-6, "верх куска выше строки"
            assert chunk.y2 <= edge_y(x, bl, br) + 1e-6, "низ куска ниже строки"


@pytest.mark.parametrize("angle", [1.0, 3.0, 6.0])
def test_slant_eats_no_more_than_promised(angle):
    """Ширина куска подстраивается под наклон, чтобы потеря высоты была ограничена.

    Без этого кусок фиксированной ширины у сильно наклонной строки становился бы тем
    тоньше, чем больше наклон, — и замеров оставалось бы меньше именно там, где
    искажения сильнее. Это ровно тот пространственно неоднородный перекос, который
    зональная метрика приняла бы за расфокус.
    """
    region = slanted_line(angle)
    heights = [c.y2 - c.y1 for c in line_chunks(region, FRAME)]
    # Плюс два пикселя на округление границ куска до целых.
    assert min(heights) >= region.height * (1.0 - MAX_SLANT_LOSS) - 2


def test_crop_is_a_view_and_never_resamples():
    """Кроп куска — срез массива, а не пересчёт.

    Любая интерполяция сама размывает край, причём тем сильнее, чем дальше угол от
    кратного 90°. Наклон по кадру меняется плавно, значит и добавленное размытие
    менялось бы плавно — получился бы идеально гладкий ложный зональный сигнал.
    """
    frame = np.arange(1000 * 1000, dtype=np.uint8).reshape(1000, 1000)
    chunk = Chunk(x1=10, y1=20, x2=130, y2=40, line_height=20.0)
    crop = chunk.crop(frame)
    assert np.shares_memory(crop, frame), "кроп обязан быть представлением, а не копией"
    assert np.array_equal(crop, frame[20:40, 10:130])


def test_degenerate_lines_give_no_chunks():
    """Слишком тонкая или слишком короткая строка не даёт кусков, а не мусор."""
    thin = LineRegion(polygon=np.array([[0.0, 0.0], [600.0, 0.0], [600.0, 3.0], [0.0, 3.0]]))
    short = LineRegion(polygon=np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 20.0], [0.0, 20.0]]))
    assert line_chunks(thin, FRAME) == []
    assert line_chunks(short, FRAME) == []


def test_chunks_are_clipped_to_the_frame():
    """Строка, вылезшая за кадр, не порождает кусков с координатами вне массива."""
    region = LineRegion(polygon=np.array([[900.0, 980.0], [1400.0, 980.0], [1400.0, 1010.0], [900.0, 1010.0]]))
    for chunk in line_chunks(region, FRAME):
        assert 0 <= chunk.x1 < chunk.x2 <= FRAME[1]
        assert 0 <= chunk.y1 < chunk.y2 <= FRAME[0]
