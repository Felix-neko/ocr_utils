"""Проверка находки: таблица это или рамка объявления, чертёж, блок-схема."""

from __future__ import annotations

from research.legacy.table_processing.detection import ruling, verify

from tests.research.legacy.table_processing.synthetic import DPI, make_block_diagram, make_frame, make_table


def _verdict(image):
    lines = ruling.find_lines(image, DPI)
    return verify.is_table(verify.features(image, lines, DPI))


def test_ruled_table_is_accepted():
    table = make_table(
        upright={
            (0, 0): "Области",
            (0, 1): "план",
            (0, 2): "факт",
            (1, 0): "Архангельская",
            (1, 1): "22030",
            (1, 2): "24853",
        }
    )
    good, reason = _verdict(table.image)
    assert good, reason


def test_advertisement_frame_is_rejected():
    good, reason = _verdict(make_frame())
    assert not good
    assert "рамка" in reason


def test_block_diagram_is_rejected():
    good, reason = _verdict(make_block_diagram())
    assert not good


def test_table_without_internal_horizontals_is_accepted():
    """Вёрстка бывает разной: внутренних горизонталей может не быть вовсе."""
    table = make_table(
        row_heights=[120, 300],
        upright={
            (0, 0): "Наименование",
            (0, 1): "план",
            (0, 2): "факт",
            (1, 0): "Прокат",
            (1, 1): "886",
            (1, 2): "450",
        },
    )
    good, reason = _verdict(table.image)
    assert good, reason


def test_features_are_finite():
    table = make_table(upright={(0, 0): "Области", (1, 1): "22030"})
    signs = verify.features(table.image, ruling.find_lines(table.image, DPI), DPI)
    row = signs.as_row()
    assert set(row) == set(verify.HEADER)
    assert all(value == value for value in row.values())


def test_new_detector_is_a_subset_of_the_old_one():
    """Вторая версия — это первая плюс фильтр, и иначе быть не может.

    На этом свойстве стоит сверка ``check-detector``: сравнение двух версий укладывается в
    таблицу два на три только потому, что одно множество вложено в другое.

    ВНИМАНИЕ: для ТРЕТЬЕЙ версии это уже неверно, и это не поломка. Она ослабила пороги
    кластеризации и находит то, чего вторая не находила, — вложение перевернулось. Про неё
    есть отдельное утверждение ниже.
    """
    from research.legacy.table_processing.detection.ruling import detect

    images = [
        make_table(upright={(0, 0): "Области", (0, 1): "план", (1, 0): "Архангельская", (1, 1): "22030"}).image,
        make_frame(),
        make_block_diagram(),
    ]
    for image in images:
        old = {table.box.as_tuple() for table in detect(image, DPI, verify_findings=False)}
        new = {table.box.as_tuple() for table in detect(image, DPI, verify_findings=True)}
        assert new <= old, f"новый детектор нашёл то, чего нет у старого: {new - old}"


def test_third_version_finds_at_least_what_the_second_found():
    """Третья версия — надмножество второй на синтетике, а не подмножество.

    Направление вложения перевернулось намеренно: третья версия ослабила два порога
    кластеризации ради бланков, которые вторая по построению не видела. Проверяем ровно то,
    что должно остаться верным — ничего из найденного второй версией не потеряно.

    Сравнение по ПЕРЕКРЫТИЮ, а не по совпадению рамок: третья версия рамку ещё и двигает
    (доращивает, обрезает бестолковые полосы, прилипает к просвету), так что рамка обязана
    отличаться, но обязана и накрывать прежнюю.
    """
    from research.legacy.table_processing.detection.ruling import detect as detect_v2
    from research.legacy.table_processing.detection.ruling_v3 import detect as detect_v3
    from research.legacy.table_processing.geometry import intersection

    image = make_table(upright={(row, column): f"я{row}{column}" for row in range(3) for column in range(3)}).image
    old = detect_v2(image, DPI)
    new = detect_v3(image, DPI)
    assert old, "синтетическая таблица обязана находиться и второй версией"
    for table in old:
        overlaps = [
            intersection(table.box, other.box) for other in new if intersection(table.box, other.box) is not None
        ]
        best = max((box.area for box in overlaps), default=0)
        assert best >= 0.7 * table.box.area, f"третья версия потеряла находку второй: {table.box.as_tuple()}"
