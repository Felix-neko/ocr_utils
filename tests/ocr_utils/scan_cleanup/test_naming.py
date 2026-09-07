"""Отпечаток исходника в имени очищенной полосы."""

import pytest

from ocr_utils.scan_cleanup.naming import (
    HASH_PREFIX_LEN,
    cleaned_file_name,
    cleaned_rel_path,
    hash_suffix,
    split_cleaned_stem,
)

DIGEST = "4c10d529198ccd0dc2100f52e047699c75991166667f5ce40a8844ac8039b0f9"


def test_name_carries_the_fingerprint():
    assert cleaned_file_name("IMG_0034_1L.tif", DIGEST, ".tif") == "IMG_0034_1L.4c10d529.tif"
    assert cleaned_rel_path("1975/04/IMG_0034_1L.tif", DIGEST, ".tif") == "1975/04/IMG_0034_1L.4c10d529.tif"


def test_output_format_changes_only_the_extension():
    assert cleaned_rel_path("1975/04/IMG_0034_1L.tif", DIGEST, ".jpg") == "1975/04/IMG_0034_1L.4c10d529.jpg"


def test_without_a_fingerprint_the_name_stays_as_it_was():
    """Полоса, не прошедшая detect, отпечатка не имеет — выдумывать его нельзя.

    Ронять из-за этого прогон тоже нельзя: цена несоразмерна, а имя без отпечатка просто
    остаётся прежним и работает как раньше.
    """
    assert cleaned_rel_path("1975/04/IMG_0034_1L.tif", None, ".tif") == "1975/04/IMG_0034_1L.tif"
    assert hash_suffix(None) == ""


def test_name_is_derivable_before_the_file_exists():
    """Главное свойство: имя выводится из ИСХОДНИКА, поэтому --skip-if-exists возможен.

    Возьми мы отпечаток результата, узнать имя можно было бы только посчитав картинку —
    и проверка «уже готово?» потеряла бы смысл целиком.
    """
    first = cleaned_rel_path("1975/04/a.tif", DIGEST, ".tif")
    second = cleaned_rel_path("1975/04/a.tif", DIGEST, ".tif")
    assert first == second


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("IMG_0034_1L.4c10d529", ("IMG_0034_1L", "4c10d529")),
        ("IMG_0034_1L", ("IMG_0034_1L", None)),
        ("0010_2R.deadbeef", ("0010_2R", "deadbeef")),
        # Не отпечаток: не тот набор знаков и не та длина.
        ("IMG_0034_1L.ZZZZZZZZ", ("IMG_0034_1L.ZZZZZZZZ", None)),
        ("IMG_0034_1L.4c10d5", ("IMG_0034_1L.4c10d5", None)),
    ],
)
def test_stem_is_parsed_back(stem, expected):
    assert split_cleaned_stem(stem) == expected


def test_prefix_length_matches_the_parser():
    """Длина в регулярном выражении и в константе — одна и та же величина."""
    assert split_cleaned_stem("a." + "0" * HASH_PREFIX_LEN)[1] == "0" * HASH_PREFIX_LEN
