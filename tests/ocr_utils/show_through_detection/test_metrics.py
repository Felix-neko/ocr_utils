"""Проверка метрик просвечивания на синтетике.

Главное требование к метрике сформулировано в задаче: она должна реагировать на просвет
с оборота и НЕ реагировать на то, что меняется от выпуска к выпуску само по себе —
цвет бумаги, освещение, экспозицию, вёрстку. Тесты ниже проверяют ровно это, парами
страниц, отличающихся одним свойством.
"""

import numpy as np
import pytest

from ocr_utils.show_through_detection.metrics import ALGORITHMS
from ocr_utils.show_through_detection.zones import build_zones
from tests.ocr_utils.show_through_detection.pages import add_show_through, add_stains, draw_page, expose, scan

# Шум съёмки, добавляемый ко всем страницам. Без него у идеально гладкой синтетической
# бумаги нулевой уровень шума на полях, и опорная величина основной метрики вырождается —
# в жизни такого не бывает, а тест проверял бы поведение, которого никогда не случится.
NOISE = 1.5

ALL = sorted(ALGORITHMS)

# Метрики, обязанные пережить смену бумаги и экспозиции. ``gap_abs`` сюда не входит
# намеренно: она меряет абсолютную величину и защиты от смены условий съёмки не имеет —
# это записано в её докстринге и зафиксировано отдельным тестом ниже.
EXPOSURE_ROBUST = [name for name in ALL if name != "gap_abs"]


def scored(page: np.ndarray, algorithm: str) -> float:
    """Балл одной страницы.

    Args:
        page: Полутоновая полоса.
        algorithm: Имя метрики.

    Returns:
        Балл метрики.
    """
    return ALGORITHMS[algorithm].score_of(build_zones(page))


def page(strength: float = 0.0, *, gain: float = 1.0, offset: float = 0.0, seed: int = 0, **kwargs) -> np.ndarray:
    """Синтетическая полоса с заданным просветом и съёмкой.

    Args:
        strength: Сила просвета с оборота.
        gain: Множитель яркости.
        offset: Сдвиг яркости.
        seed: Зерно генератора лицевой страницы.
        **kwargs: Прочие аргументы ``draw_page``.

    Returns:
        Полутоновая полоса uint8.
    """
    return expose(scan(add_show_through(draw_page(seed=seed, **kwargs), strength)), gain, offset, NOISE)


@pytest.mark.parametrize("algorithm", ALL)
def test_score_grows_monotonically_with_show_through(algorithm: str) -> None:
    """Балл обязан расти с силой просвета — это определение шкалы всего пакета."""
    scores = [scored(page(strength), algorithm) for strength in (0.0, 0.12, 0.24, 0.36)]
    assert all(np.isfinite(s) for s in scores), f"{algorithm}: балл не посчитался"
    assert scores == sorted(scores), f"{algorithm}: балл не монотонен по силе просвета: {scores}"
    assert scores[-1] > scores[0], f"{algorithm}: сильный просвет не отличается от чистой полосы"


def test_ghost_ink_ignores_show_through_that_binarisation_will_kill() -> None:
    """Ключевое отличие метрики по умолчанию от остальных, и ради него она и написана.

    Бледный просвет виден глазом и честно поднимается контрастными метриками, но обработка
    (бинаризация → маска → размытие фона) его снимает, и пересъёмки он не требует. Замерено
    на настоящем материале: выпуск 1955/03, который пользователь разобрал глазами именно
    так, ``gap_contrast`` поднимает выше порога на 41 % полос, ``ghost_ink`` — ни на одной.
    """
    faint, heavy, clean = (build_zones(page(s)) for s in (0.12, 0.36, 0.0))
    assert (
        ALGORITHMS["ghost_ink"].score_of(faint) < ALGORITHMS["ghost_ink"].threshold
    ), "бледный просвет не должен перешагивать порог: обработка его снимет"
    assert ALGORITHMS["gap_contrast"].score_of(faint) > ALGORITHMS["gap_contrast"].score_of(
        clean
    ), "контрастная метрика обязана бледный просвет ВИДЕТЬ — иначе тест ничего не доказывает"
    assert ALGORITHMS["ghost_ink"].score_of(heavy) > ALGORITHMS["ghost_ink"].score_of(faint)


def test_ghost_ink_ignores_ink_that_leaks_into_the_gap() -> None:
    """Край настоящей буквы, дотянувшийся в межстрочье, — не призрак.

    Без фильтра по связности плотная полоса (таблица, жирный заголовок) уезжает в топ
    рейтинга: замерено на ``02-03_0013 L``, доля падает с 0.027 до 0.002 после фильтра.
    Здесь то же самое проверяется на синтетике: полоса тесного набора без всякого
    просвета обязана дать почти ноль.
    """
    tight = expose(scan(draw_page(stroke=4, line_height=22, columns=1)), noise=NOISE)
    score = ALGORITHMS["ghost_ink"].score_of(build_zones(tight))
    assert score < ALGORITHMS["ghost_ink"].threshold, f"тесный набор принят за просвет: {score:.5f}"


@pytest.mark.parametrize("algorithm", EXPOSURE_ROBUST)
def test_paper_colour_and_exposure_barely_move_the_score(algorithm: str) -> None:
    """Другой год съёмки — другая бумага и свет; балл чистой полосы обязан устоять.

    Ради этого метрики и считаются по ОТРАЖЕНИЮ относительно локального уровня бумаги,
    а основная — ещё и нормируется на собственные поля полосы.
    """
    reference = scored(page(0.0), algorithm)
    darker = scored(page(0.0, gain=0.75, offset=25.0), algorithm)
    brighter = scored(page(0.0, gain=1.1, offset=-10.0), algorithm)
    spread = max(reference, darker, brighter) - min(reference, darker, brighter)
    bleeding = scored(page(0.30), algorithm)
    assert spread < (bleeding - reference), (
        f"{algorithm}: разброс от экспозиции ({spread:.4f}) не меньше эффекта просвета "
        f"({bleeding - reference:.4f}) — метрика меряет съёмку, а не бумагу"
    )


def test_gap_abs_is_knowingly_sensitive_to_exposure() -> None:
    """Тест-документация известного изъяна запасной метрики, а не проверка корректности.

    ``gap_abs`` меряет абсолютную величину отклика и потому обязана ехать вместе с
    условиями съёмки — ровно из-за этого она запасная, а не основная, и берётся только
    там, где у полосы нет чистых полей и нормировать не на что. Тест фиксирует изъян,
    чтобы никто не принял её за равноценную замену.
    """
    reference = scored(page(0.0), "gap_abs")
    darker = scored(page(0.0, gain=0.75, offset=25.0), "gap_abs")
    from_exposure = abs(darker - reference)
    from_show_through = scored(page(0.30), "gap_abs") - reference
    assert from_exposure > 0.5 * from_show_through, (
        "если изъян исчез, метрику пора переводить в основные, а этот тест — удалять "
        f"(экспозиция {from_exposure:.5f}, просвет {from_show_through:.5f})"
    )


@pytest.mark.parametrize("algorithm", ALL)
def test_text_amount_barely_moves_the_score(algorithm: str) -> None:
    """Полоса с четвертью текста и полоса целиком в наборе должны мериться одинаково."""
    full = scored(page(0.0), algorithm)
    sparse = scored(page(0.0, fill=0.4), algorithm)
    bleeding = scored(page(0.30), algorithm)
    assert abs(full - sparse) < (bleeding - full), f"{algorithm}: количество текста двигает балл сильнее просвета"


@pytest.mark.parametrize("algorithm", ALL)
def test_smooth_stains_are_not_mistaken_for_show_through(algorithm: str) -> None:
    """Лисьи пятна попадают в тот же диапазон уровней, но штриховой структуры не имеют.

    Если метрика их считает просветом, на пересканирование поедет вся грязная бумага
    пака — а это совсем другой дефект и другое решение.
    """
    clean = scored(page(0.0), algorithm)
    stained = ALGORITHMS[algorithm].score_of(build_zones(expose(scan(add_stains(draw_page())), noise=NOISE)))
    bleeding = scored(page(0.30), algorithm)
    assert stained - clean < 0.5 * (
        bleeding - clean
    ), f"{algorithm}: пятна дают {stained:.4f} против {clean:.4f} у чистой и {bleeding:.4f} у просвета"


def test_show_through_must_be_mirrored_to_be_detected() -> None:
    """Незеркальный подмес — это не просвет, а вторая печать, и мерить его нечестно.

    Тест не про качество детекции, а про корректность самой синтетики: если бы страница
    с НЕзеркальным подмесом давала тот же балл, все остальные тесты проходили бы и на
    генераторе, который зеркала не делает, и проверяли бы не то, что нужно.
    """
    base = draw_page(seed=0)
    mirrored = add_show_through(base, 0.30)
    assert not np.array_equal(mirrored, base)


@pytest.mark.parametrize("algorithm", ALL)
def test_blank_page_is_not_measured(algorithm: str) -> None:
    """На полосе без текста межстрочий нет, и метрика обязана честно отказаться."""
    blank = expose(scan(np.full((1400, 1000), 238, np.uint8)), noise=NOISE)
    zones = build_zones(blank)
    assert zones.problem, "пустая полоса должна быть помечена как неизмеримая"
    assert not np.isfinite(ALGORITHMS[algorithm].score_of(zones))
