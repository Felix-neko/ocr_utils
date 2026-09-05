"""Выбор промпта Stable Diffusion по фону под зоной (``scan_cleanup.prompts``)."""

import pytest

from ocr_utils.scan_cleanup.prompts import (
    PROMPT_COLOUR,
    PROMPT_HALFTONE,
    PROMPT_OTHER_SUFFIX,
    PROMPT_PAPER,
    PromptSet,
    overlap_frac,
    prompt_chooser,
    prompt_for,
    raster_kind_at,
)
from ocr_utils.scan_cleanup.source import PageMarkup, Rect
from ocr_utils.scan_markup.db.models import (
    KIND_COLOR,
    KIND_COLOR_TEXT,
    KIND_GRAYSCALE,
    KIND_STAMP_SUSPECT,
    MASK_HANDWRITING,
    MASK_LIBRARY_STAMP,
    MASK_OTHER_REMOVAL,
)


def markup(*regions: Rect) -> PageMarkup:
    return PageMarkup("1970/01/0010.tif", 400, 600, 600, 8, regions, ())


def rect(kind: str, box=(100, 100, 300, 300)) -> Rect:
    return Rect(*box, kind, False)


ZONE = (150, 150, 200, 200)  # целиком внутри прямоугольника выше


def test_zone_inside_colour_gets_colour_prompt():
    prompt, _ = prompt_for(ZONE, markup(rect(KIND_COLOR)), MASK_LIBRARY_STAMP)
    assert prompt == PROMPT_COLOUR


def test_zone_inside_halftone_gets_halftone_prompt():
    prompt, _ = prompt_for(ZONE, markup(rect(KIND_GRAYSCALE)), MASK_LIBRARY_STAMP)
    assert prompt == PROMPT_HALFTONE


def test_zone_on_bare_paper_gets_paper_prompt():
    prompt, _ = prompt_for((10, 10, 40, 40), markup(rect(KIND_COLOR)), MASK_LIBRARY_STAMP)
    assert prompt == PROMPT_PAPER


def test_partial_overlap_below_threshold_counts_as_paper():
    """Зона на треть заехавшая на иллюстрацию — всё ещё на бумаге."""
    # Зона 80..110 по x при левом крае области 100: внутри треть её ширины.
    zone = (80, 150, 110, 180)
    assert overlap_frac(zone, rect(KIND_COLOR)) == pytest.approx(1 / 3)
    prompt, _ = prompt_for(zone, markup(rect(KIND_COLOR)), MASK_LIBRARY_STAMP)
    assert prompt == PROMPT_PAPER


def test_color_text_does_not_count_as_raster():
    """Под цветным набором та же бумага — и заполнять его надо бумажным промптом."""
    assert raster_kind_at(ZONE, (rect(KIND_COLOR_TEXT),)) is None
    prompt, _ = prompt_for(ZONE, markup(rect(KIND_COLOR_TEXT)), MASK_LIBRARY_STAMP)
    assert prompt == PROMPT_PAPER


def test_stamp_suspect_does_not_count_as_raster():
    assert raster_kind_at(ZONE, (rect(KIND_STAMP_SUSPECT),)) is None


def test_strongest_overlap_wins():
    """Печать на стыке двух областей получает описание того фона, которого больше."""
    zone = (150, 150, 250, 250)
    regions = (rect(KIND_COLOR, (0, 0, 175, 600)), rect(KIND_GRAYSCALE, (175, 0, 400, 600)))
    assert raster_kind_at(zone, regions, min_frac=0.4) == KIND_GRAYSCALE


def test_other_removal_gets_the_suffix():
    prompt, _ = prompt_for((10, 10, 40, 40), markup(), MASK_OTHER_REMOVAL)
    assert prompt == f"{PROMPT_PAPER}, {PROMPT_OTHER_SUFFIX}"
    # У остальных видов добавки нет: правило одно, отличается только хвост.
    plain, _ = prompt_for((10, 10, 40, 40), markup(), MASK_HANDWRITING)
    assert plain == PROMPT_PAPER


def test_suffix_applies_on_top_of_the_raster_prompt():
    prompt, _ = prompt_for(ZONE, markup(rect(KIND_GRAYSCALE)), MASK_OTHER_REMOVAL)
    assert prompt == f"{PROMPT_HALFTONE}, {PROMPT_OTHER_SUFFIX}"


def test_prompts_are_overridable():
    prompts = PromptSet(paper="my paper", negative="my negative")
    prompt, negative = prompt_for((10, 10, 40, 40), markup(), MASK_LIBRARY_STAMP, prompts)
    assert (prompt, negative) == ("my paper", "my negative")


def test_chooser_is_a_plain_callable_over_the_zone_box():
    choose = prompt_chooser(markup(rect(KIND_COLOR)), MASK_LIBRARY_STAMP)
    assert choose(ZONE)[0] == PROMPT_COLOUR
    assert choose((10, 10, 40, 40))[0] == PROMPT_PAPER


# ----------------------------------------------------------------------
# Полосный прямоугольник ничего не локализует
# ----------------------------------------------------------------------


def cover_markup() -> PageMarkup:
    """Обложка: цветная область во всю полосу, включая белые поля."""
    return PageMarkup("1970/02/cover.tif", 400, 600, 600, 8, (Rect(0, 0, 400, 600, KIND_COLOR, True),), ())


def test_full_page_region_does_not_decide_the_prompt():
    """Печать на поле обложки лежит на бумаге, а не на картинке.

    Прямоугольник обложки накрывает и поля тоже; поверив ему, SD дорисовывает на месте
    печати цветную иллюстрацию — это и наблюдалось на 1970/02 IMG_0053_2R.
    """
    assert raster_kind_at(ZONE, cover_markup().regions, page_area=400 * 600) is None
    prompt, _ = prompt_for(ZONE, cover_markup(), MASK_LIBRARY_STAMP)
    assert prompt == PROMPT_PAPER


def test_tight_region_still_decides():
    """Тесный прямоугольник человек обвёл вокруг конкретной иллюстрации — ему верим."""
    assert raster_kind_at(ZONE, (rect(KIND_COLOR),), page_area=400 * 600) == KIND_COLOR


def test_background_kind_says_paper_on_blank_paper():
    import numpy as np

    from ocr_utils.scan_cleanup.prompts import background_kind

    rng = np.random.default_rng(0)
    paper = np.full((400, 400, 3), 245, np.uint8)
    paper = np.clip(paper.astype(np.int16) + rng.integers(-3, 4, size=paper.shape), 0, 255).astype(np.uint8)
    assert background_kind(paper) is None


def test_tinted_cover_paper_is_still_paper():
    """Бумага обложки серовато-сиреневая и целиком лежит в «средних тонах».

    Детектор растра на ней честно отвечает «растр есть», и без порога по разбросу
    яркости печать на поле обложки получала бы промпт «полутоновая фотография» —
    это и наблюдалось на 1970/02 IMG_0053_2R (разброс p90-p10 там 32 против 160-182
    у настоящих иллюстраций).
    """
    import numpy as np

    from ocr_utils.scan_cleanup.prompts import background_kind

    rng = np.random.default_rng(5)
    tinted = np.dstack(
        [
            rng.integers(200, 216, size=(400, 400), dtype=np.uint8),
            rng.integers(196, 212, size=(400, 400), dtype=np.uint8),
            rng.integers(206, 222, size=(400, 400), dtype=np.uint8),
        ]
    )
    assert background_kind(tinted) is None


def halftone_photo(seed: int = 1) -> "np.ndarray":
    """Полутоновая фотография: широкий тональный переход плюс зерно растра.

    Разброс p90-p10 здесь около 150 — как у настоящих иллюстраций пака (160-182), а
    не как у равномерного шума в узком диапазоне.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    ramp = np.linspace(40, 235, 400, dtype=np.float32)[:, None].repeat(400, axis=1)
    grain = rng.normal(0, 6, size=ramp.shape)
    grey = np.clip(ramp + grain, 0, 255).astype(np.uint8)
    return np.dstack([grey, grey, grey])


def test_background_kind_finds_a_halftone_picture():
    from ocr_utils.scan_cleanup.prompts import background_kind

    assert background_kind(halftone_photo()) == KIND_GRAYSCALE


def test_prompt_falls_back_to_pixels_on_a_cover():
    """Без тесного прямоугольника решает окрестность зоны."""
    prompt, _ = prompt_for(ZONE, cover_markup(), MASK_LIBRARY_STAMP, roi_bgr=halftone_photo(2))
    assert prompt == PROMPT_HALFTONE
