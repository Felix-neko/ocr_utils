"""Сборка областей: Surya предлагает, пиксели уточняют."""

import cv2
import numpy as np
import pytest

from ocr_utils.scan_markup.detection.dots import params_for_dpi, screen_regions
from ocr_utils.scan_markup.detection.tone import tone_maps
from ocr_utils.scan_markup.detection.regions import (
    LINEART_MAX_DOT_FRAC,
    SOURCE_LINEART,
    SOURCE_SCREEN,
    find_raster_boxes,
)
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
    kwargs.setdefault("tone", tone_maps(page, params))
    return find_raster_boxes(
        regions, stats, centroids, work, page.shape[:2], params, surya_boxes=surya_boxes, order_index=1, **kwargs
    )


def _area(box) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


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


def _two_patches(page):
    """Две растровые области, разнесённые по полосе, — как верх и низ обложки.

    Вместе их охватывающий прямоугольник занимает около 79% полосы, то есть проходит порог
    «полоса занята картинкой целиком» (0.75); зазор между ними 450 px, слиянию по зазору
    (2 мм) не поддаётся.
    """
    synthetic.with_screen(page, (80, 80, 1120, 700), pitch=4, radius=1)
    synthetic.with_screen(page, (80, 1150, 1120, 1720), pitch=4, radius=1)
    return page


def test_colour_page_becomes_one_full_page_region() -> None:
    """Цветная полоса, занятая находками почти целиком, — одна область во весь кадр.

    Обложку разрывает то, что само картинкой не является: сплошная плашка заголовка
    растровых клеток не даёт, и полоса распадается на куски (1967/11 IMG_0052_2R: 4.3% и
    68.4% при зазоре 588 px). Отличает её от двух ч/б фотографий не зазор, а то, что она
    цветная ЦЕЛИКОМ — разброс хроматичности по полосе 33.7 против 4.1..5.1.
    """
    page = _two_patches(synthetic.paper(SIZE))
    regions, stats, centroids, work, params = _analyse(page)
    blocks = [(70, 70, 1130, 710), (70, 1140, 1130, 1730)]

    # Цвет полосы приходит извне: сама сборка областей ни цвета, ни бумаги не знает.
    grey = find_raster_boxes(
        regions, stats, centroids, work, page.shape[:2], params, blocks, order_index=1, page_chroma_spread=4.5
    )
    colour = find_raster_boxes(
        regions, stats, centroids, work, page.shape[:2], params, blocks, order_index=1, page_chroma_spread=30.0
    )

    assert len(grey.findings) == 2, "ч/б полоса обязана остаться с двумя областями"
    assert len(colour.findings) == 1 and colour.findings[0].full_page
    assert colour.findings[0].box == (0, 0, SIZE[1], SIZE[0])


def test_colour_page_with_one_small_area_is_not_filled() -> None:
    """Одной маленькой картинки на цветной полосе мало, чтобы залить кадр целиком.

    Без проверки площади цветная полоса с картинкой и текстом уехала бы в разметку вся.
    """
    page = synthetic.with_screen(synthetic.paper(SIZE), (150, 150, 700, 600), pitch=4, radius=1)
    regions, stats, centroids, work, params = _analyse(page)
    findings = find_raster_boxes(
        regions,
        stats,
        centroids,
        work,
        page.shape[:2],
        params,
        [(140, 140, 710, 610)],
        order_index=1,
        page_chroma_spread=30.0,
    )
    assert len(findings.findings) == 1
    assert not findings.findings[0].full_page


def test_safety_net_rejects_dot_leaders() -> None:
    """Колонка отточий из оглавления страховкой не подхватывается, а растр — подхватывается.

    Отточия дают ровно ту статистику, по которой опознаётся растровая печать: множество
    мелких круглых пятен. По размеру, плотности и разбросу площади они от фотографии не
    отличаются — это замерено и перекрывается. Отличается строение: точки стоят по базовым
    линиям текста, между линиями бумага.
    """
    leaders = synthetic.paper(SIZE)
    leaders[300:1500, 300:800] = synthetic.dot_leaders((1200, 500), line_step=28, dot_step=15, radius=2)
    screen = _page_with_photo((200, 200, 900, 1000))

    assert not _find(leaders, []).findings
    assert _find(screen, []).findings


def test_leader_features_separate_the_two() -> None:
    """Тот же случай на уровне признаков — чтобы при сдвиге фикстуры было видно, что поехало.

    Пороги: пустых строк 0.30, периодичность 0.52.
    """
    leaders = synthetic.paper(SIZE)
    leaders[300:1500, 300:800] = synthetic.dot_leaders((1200, 500), line_step=28, dot_step=15, radius=2)
    screen = _page_with_photo((200, 200, 900, 1000))

    lead = list(_analyse(leaders)[0].leader.values())
    scr = list(_analyse(screen)[0].leader.values())
    assert lead and scr
    assert lead[0][0] >= 0.30 and lead[0][1] >= 0.52, f"отточия должны срабатывать: {lead[0]}"
    assert scr[0][0] < 0.30, f"растр не должен: {scr[0]}"


def test_safety_box_grows_to_the_edge_of_the_picture() -> None:
    """Страховочная рамка растёт до края картинки и останавливается перед подписью.

    У области без блока Surya объединять не с чем, и граница берётся по растровым клеткам,
    а они по краям снимка не добирают: светлые участки клеток не дают. Замер на паке-1:
    1973/12 IMG_0271_1L (640,1408,3200,3072) -> (384,1152,3200,3072).
    """
    page = synthetic.paper(SIZE)
    photo = (200, 300, 1000, 1200)
    synthetic.with_screen(page, photo, pitch=4, radius=1)
    # Подпись под фотографией: тонкий текст на белом, средняя яркость почти бумажная.
    page[1260:1300] = synthetic.text_page((40, SIZE[1]), line_step=40, glyph_w=14, glyph_h=30, char_step=26, margin=200)

    findings = _find(page, [])
    assert len(findings.findings) == 1
    x1, y1, x2, y2 = findings.findings[0].box

    # Рост идёт шагом GROW_STEP_PX, поэтому совпадение с краем — с точностью до шага.
    from ocr_utils.scan_markup.detection.regions import GROW_STEP_PX

    assert x1 <= photo[0] + GROW_STEP_PX and y1 <= photo[1] + GROW_STEP_PX
    assert x2 >= photo[2] - GROW_STEP_PX and y2 >= photo[3] - GROW_STEP_PX
    assert y2 < 1260, "рамка не должна дорастать до подписи"

    # И рамка действительно выросла: без роста она заметно уже.
    ungrown = _find(page, [], grow_paper_margin=255)
    assert (x2 - x1) * (y2 - y1) > 1.05 * _area(ungrown.findings[0].box)


def test_shrink_pulls_the_frame_off_the_paper_but_not_into_the_block() -> None:
    """Подрезка снимает с рамки бумагу, добавленную округлением до клетки, и только её.

    Край растровой области округляется наружу до целой клетки — при 600 dpi это 128 px,
    около 5 мм, и в этот ряд попадает подпись под фотографией. Замер на паке-1:
    1974/03 IMG_0140_1L (368,1560,3200,3456) -> (368,1560,3140,3396), то есть ровно до
    низа блока Surya. Глубже блока подрезка не идёт никогда: блок — то, что поручила
    модель, и спорить с ним пикселями значит вернуть срезание светлых краёв фотографии.
    """
    from ocr_utils.scan_markup.detection.regions import shrink_to_paper

    photo = (200, 200, 900, 1000)
    page = _page_with_photo(photo)
    work = cv2.resize(page, (SIZE[1] // 4, SIZE[0] // 4), interpolation=cv2.INTER_AREA)

    loose = (photo[0] - 128, photo[1] - 128, photo[2] + 128, photo[3] + 128)
    tight = shrink_to_paper(work, loose, photo)
    assert _area(tight) < _area(loose), "рамка по бумаге должна подрезаться"
    assert tight[0] <= photo[0] and tight[1] <= photo[1]
    assert tight[2] >= photo[2] and tight[3] >= photo[3]

    # Тот же вызов с блоком шире картинки не двигает рамку внутрь блока.
    block = (100, 100, 1100, 1400)
    assert shrink_to_paper(work, block, block) == block


def test_tone_features_are_recorded_even_when_the_rule_is_silent() -> None:
    """Признаки пишутся у каждой находки: по ним пороги крутятся без чтения оригиналов."""
    findings = _find(_page_with_photo(), [(180, 180, 920, 1020)])
    finding = findings.findings[0]
    assert finding.source == SOURCE_SCREEN
    assert finding.mid_frac is not None and finding.tone_entropy is not None
    assert finding.screen_peak is not None


def test_full_of_screen_cells_is_never_line_art() -> None:
    """Рамка, заполненная растровыми клетками сплошь, штрихом не признаётся никогда.

    Замер на паке-1: три портрета 1975/01 IMG_0048_1L по тонам и по спектру от штриха
    неотличимы (средние тона 0.18..0.22 при 0.23 у штриха, энтропия 6.2..6.5 при 6.37), и
    связка ``tone`` уводила их в «подозрение на печать». Отличает их доля растровых клеток:
    у фотографий 0.98..1.00, у штриха 0.60..0.85 — фотография заполняет свой прямоугольник
    растром сплошь, а между штрихами рисунка лежит бумага.

    Фикстура здесь намеренно НЕРЕАЛИСТИЧНА: растр без размытия двухцветен, и связка ``tone``
    честно принимает его за штрих (средние тона 0.000, энтропия 0.83). Это и нужно — так
    проверяется, что ограничение по доле клеток срабатывает поверх сработавшего правила.
    """
    page = synthetic.paper(SIZE)
    box = (200, 200, 900, 1000)
    synthetic.with_screen(page, box, pitch=4, radius=1, blur=0.0)
    block = (180, 180, 920, 1020)

    loose = _find(page, [block], lineart_max_dot_frac=1.01).findings[0]
    assert loose.source == SOURCE_LINEART, "по тонам резкая решётка и должна выглядеть штрихом"
    assert loose.dot_frac is not None and loose.dot_frac > LINEART_MAX_DOT_FRAC
    assert _find(page, [block]).findings[0].source == SOURCE_SCREEN
