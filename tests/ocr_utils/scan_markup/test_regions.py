"""Сборка областей: Surya предлагает, пиксели уточняют."""

import cv2
import numpy as np
import pytest

from ocr_utils.scan_markup.detection.dots import params_for_dpi, screen_regions
from ocr_utils.scan_markup.detection.regions import SOURCE_LINEART, SOURCE_SCREEN, find_raster_boxes
from tests.ocr_utils.scan_markup import synthetic

SIZE = (1800, 1200)  # полоса 300 dpi
DPI = 300


def _analyse(page: np.ndarray):
    """Пиксельный разбор полосы плюс её копия 1/4 — то, что видит find_raster_boxes."""
    params = params_for_dpi(DPI)
    regions, stats, centroids = screen_regions(page, params)
    work = cv2.resize(page, (page.shape[1] // 4, page.shape[0] // 4), interpolation=cv2.INTER_AREA)
    return regions, stats, centroids, work, params


def _find(page: np.ndarray, surya_boxes, **kwargs):
    regions, stats, centroids, work, params = _analyse(page)
    return find_raster_boxes(
        regions, stats, centroids, work, page.shape[:2], params, surya_boxes=surya_boxes, order_index=1, **kwargs
    )


def _page_with_photo(box=(200, 200, 900, 1000)):
    return synthetic.with_screen(synthetic.paper(SIZE), box, pitch=4, radius=1)


def test_region_never_shrinks_below_the_surya_block() -> None:
    """Область — ОБЪЕДИНЕНИЕ блока и растровых клеток, а не замена блока клетками.

    Соблазн заменить есть: блок в медиане в 1.69 раза крупнее картинки и хватает бумагу.
    Но по краю картинки клетка наполовину занята бумагой, точек в ней не набирается, и
    замена уводит рамку ВНУТРЬ фотографии — замер на паке-1 дал по 228..348 px срезанных
    слева и 184..300 px сверху на четырёх полосах 1966 года. Лишняя бумага не стоит ничего,
    срезанный кусок фотографии теряется навсегда.
    """
    photo = (200, 200, 900, 1000)
    page = _page_with_photo(photo)
    block = (60, 60, 1140, 1500)  # блок вдвое шире картинки

    findings = _find(page, [block])
    assert len(findings.findings) == 1
    x1, y1, x2, y2 = findings.findings[0].box
    assert x1 <= block[0] and y1 <= block[1] and x2 >= block[2] and y2 >= block[3]


def test_cells_extend_the_block_where_they_reach_further() -> None:
    """Клетки не только не сжимают блок, но и достраивают его, если картинка шире блока.

    Ровно этот случай Surya и портит чаще всего: край фотографии она срезает, а разметчику
    нужен весь кадр.
    """
    photo = (200, 200, 900, 1000)
    page = _page_with_photo(photo)
    clipped = (400, 400, 700, 700)  # блок ВНУТРИ картинки

    findings = _find(page, [clipped])
    assert len(findings.findings) == 1
    x1, y1, x2, y2 = findings.findings[0].box
    assert x1 < clipped[0] and y1 < clipped[1] and x2 > clipped[2] and y2 > clipped[3]


def test_two_blocks_over_one_area_give_one_box() -> None:
    """Две половины одной картинки под двумя блоками Surya сливаются в один прямоугольник.

    Это и есть «объединение картинок в одну большую»: связность решает карта клеток.
    """
    page = _page_with_photo((200, 200, 900, 1000))
    findings = _find(page, [(180, 180, 1000, 560), (180, 560, 1000, 1050)])
    assert len(findings.findings) == 1


def test_separate_pictures_stay_separate() -> None:
    """Две картинки, разделённые белым полем, остаются двумя областями."""
    page = synthetic.paper(SIZE)
    synthetic.with_screen(page, (150, 150, 1050, 500), pitch=4, radius=1)
    synthetic.with_screen(page, (150, 1200, 1050, 1650), pitch=4, radius=1)
    findings = _find(page, [(140, 140, 1060, 510), (140, 1190, 1060, 1660)])
    assert len(findings.findings) == 2


def test_block_over_line_art_is_marked_lineart() -> None:
    """Блок Surya над штриховым рисунком помечается штрихом, а не растром.

    Surya считает картинкой и штрих тоже (на паке-1 — 28 полос из 31), поэтому её блоки
    обязаны проходить пиксельную проверку. Решает p99 площади связного пятна краски.
    """
    page = synthetic.paper(SIZE)
    page[200:1000, 200:900] = synthetic.line_art((800, 700), step=24, thickness=4)
    findings = _find(page, [(180, 180, 920, 1020)])
    assert findings.findings
    assert all(f.source == SOURCE_LINEART for f in findings.findings)


def test_screen_block_is_marked_screen() -> None:
    """А над настоящим растром — растром."""
    findings = _find(_page_with_photo(), [(180, 180, 920, 1020)])
    assert findings.findings and findings.findings[0].source == SOURCE_SCREEN


@pytest.mark.parametrize("photo,kept", [((200, 200, 1000, 1400), True), ((200, 200, 460, 460), False)])
def test_safety_net_keeps_only_large_unconfirmed_areas(photo, kept: bool) -> None:
    """Без блока Surya область остаётся, только если она крупная.

    Страховка от пропусков модели. Мелкие неподтверждённые области отбрасываются: строка
    отточий в оглавлении даёт ту же статистику, что растровая сетка, и занимает 0.7% полосы,
    а самая мелкая настоящая неподтверждённая область — 19.8%.
    """
    findings = _find(_page_with_photo(photo), [])
    assert bool(findings.findings) is kept


def test_without_surya_all_areas_survive() -> None:
    """Прогон без GPU: страховочный порог не применяется, работают одни пиксели.

    ``surya_boxes=None`` и ``surya_boxes=[]`` — разные вещи: первое значит «модель не
    гонялась», второе «гонялась и ничего не нашла».
    """
    page = _page_with_photo((200, 200, 460, 460))
    assert _find(page, None).findings
    assert not _find(page, []).findings


def test_black_line_art_never_reaches_the_markup() -> None:
    """Чёрный штрих в разметку не попадает, цветной — попадает как color.

    Проверяется через полный ``detect_page``: правило живёт в классификации цвета, а не в
    сборке областей, и разделение ролей тут существенное.
    """
    from pathlib import Path

    from PIL import Image

    from ocr_utils.scan_markup.db.models import KIND_COLOR
    from ocr_utils.scan_markup.detection.page import PageOptions, detect_page

    class _Stub:
        """Дублёр LayoutDetector: отдаёт заданные блоки, не трогая GPU и Surya."""

        def __init__(self, boxes):
            self._boxes = boxes

        def picture_polygons(self, bgr, gray=None, filter_raster=True):
            return [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32) for x1, y1, x2, y2 in self._boxes]

    def run(tmp: Path, ink_bgr, block_in_work) -> list:
        page = np.full((*SIZE, 3), (245, 245, 245), np.uint8)
        art = synthetic.line_art((800, 700), step=24, thickness=4) < 200
        page[200:1000, 200:900][art] = ink_bgr
        Image.fromarray(page[..., ::-1]).save(tmp, dpi=(DPI, DPI))
        result = detect_page(tmp, "a.tif", 1, PageOptions(need_digest=False), _Stub([block_in_work]))
        assert result.error == "", result.error
        return result.regions

    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "a.tif"
        block = (180 // 4, 180 // 4, 920 // 4, 1020 // 4)  # блок Surya приходит в масштабе 1/4
        assert run(path, (30, 30, 30), block) == []
        colored = run(path, (200, 60, 40), block)
        assert colored and all(region.kind == KIND_COLOR for region in colored)


def test_small_coloured_line_art_becomes_a_stamp_suspect() -> None:
    """Мелкий цветной штрих — не картинка, а подозрение на библиотечную печать.

    Печать это цветной штрих по определению (фиолетовая мастика), и от цветного рисунка её
    отличает только размер: замер по паку-1 дал у печатей 2.1 и 2.2% полосы против почти
    100% у синего рисунка обложки. Без разделения по размеру каждый оттиск уезжал бы в PDF
    как иллюстрация.
    """
    import tempfile
    from pathlib import Path

    from PIL import Image

    from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_STAMP_SUSPECT
    from ocr_utils.scan_markup.detection.page import PageOptions, detect_page

    class _Stub:
        def __init__(self, boxes):
            self._boxes = boxes

        def picture_polygons(self, bgr, gray=None, filter_raster=True):
            return [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32) for x1, y1, x2, y2 in self._boxes]

    def run(tmp: Path, patch: tuple[int, int, int, int]) -> list:
        page = np.full((*SIZE, 3), (245, 245, 245), np.uint8)
        x1, y1, x2, y2 = patch
        # Штрихи толстые намеренно: признак «это штрих, а не растр» — крупные связные пятна
        # краски (p99 площади), и тонкая сетка линий под него не подошла бы.
        art = synthetic.line_art((y2 - y1, x2 - x1), step=30, thickness=12) < 200
        page[y1:y2, x1:x2][art] = (200, 60, 40)  # цветная краска
        Image.fromarray(page[..., ::-1]).save(tmp, dpi=(DPI, DPI))
        block = tuple(v // 4 for v in patch)  # блок Surya приходит в масштабе 1/4
        result = detect_page(tmp, "a.tif", 5, PageOptions(need_digest=False), _Stub([block]))
        assert result.error == "", result.error
        return result.regions

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "a.tif"
        small = run(path, (200, 200, 600, 600))  # 7.4% полосы: мельче порога иллюстрации
        assert [r.kind for r in small] == [KIND_STAMP_SUSPECT]

        big = run(path, (100, 100, 1100, 1700))  # больше половины полосы
        assert [r.kind for r in big] == [KIND_COLOR]
