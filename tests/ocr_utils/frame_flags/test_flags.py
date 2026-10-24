"""Тесты схемы сайдкара: чтение, нормализация, противоречия, запись без перезаписи, удаление."""

import json

import pytest

from ocr_utils.frame_flags.flags import (
    FlagsError,
    FrameFlag,
    FrameFlags,
    Side,
    normalize_flags,
    read_flags,
    remove_flags,
    sidecar_path,
    write_flags,
)


@pytest.fixture
def frame(tmp_path):
    """Пустой «кадр» во временной папке."""
    path = tmp_path / "IMG_0005.jpg"
    path.write_bytes(b"jpeg")
    return path


def test_no_sidecar_is_plain_frame(frame):
    flags = read_flags(frame)
    assert flags.empty and flags.split and not flags.skip_frame
    assert not flags.ignored(Side.LEFT) and flags.equalize(Side.RIGHT)


def test_sidecar_name_keeps_extension(frame):
    assert sidecar_path(frame).name == "IMG_0005.jpg.flags.json"


def test_roundtrip(frame):
    flags = FrameFlags(frozenset({FrameFlag.IGNORE_RIGHT, FrameFlag.NO_ILLUMINATION_LEFT}), "карточка", "IMG_0006.jpg")
    write_flags(frame, flags)
    back = read_flags(frame)
    assert back.flags == flags.flags and back.pair == "IMG_0006.jpg" and back.reviewed
    assert back.ignored(Side.RIGHT) and not back.ignored(Side.LEFT)
    assert not back.equalize(Side.LEFT) and back.equalize(Side.RIGHT)
    assert back.label() == "ignore-right+no-illumination-left"
    # Сам кадр не тронут.
    assert frame.read_bytes() == b"jpeg"


def test_both_sides_become_ignore_both():
    assert normalize_flags(["ignore-left", "ignore-right", "no-splitting"]) == {FrameFlag.IGNORE_BOTH}


def test_unknown_flag_is_error(frame):
    sidecar_path(frame).write_text(json.dumps({"schema": 1, "flags": ["ignore-rirght"]}))
    with pytest.raises(FlagsError, match="ignore-rirght"):
        read_flags(frame)


def test_no_splitting_with_one_side_is_error():
    with pytest.raises(FlagsError):
        normalize_flags(["no-splitting", "ignore-left"])


def test_newer_schema_is_error(frame):
    sidecar_path(frame).write_text(json.dumps({"schema": 99, "flags": ["ignore-left"]}))
    with pytest.raises(FlagsError, match="schema"):
        read_flags(frame)


def test_no_overwrite_without_replace(frame):
    write_flags(frame, FrameFlags(frozenset({FrameFlag.IGNORE_LEFT})))
    with pytest.raises(FileExistsError):
        write_flags(frame, FrameFlags(frozenset({FrameFlag.IGNORE_RIGHT})))
    write_flags(frame, FrameFlags(frozenset({FrameFlag.IGNORE_RIGHT})), replace=True)
    assert read_flags(frame).ignored(Side.RIGHT)


def test_empty_flags_not_written(frame):
    with pytest.raises(FlagsError):
        write_flags(frame, FrameFlags())


def test_remove(frame):
    write_flags(frame, FrameFlags(frozenset({FrameFlag.IGNORE_BOTH})))
    assert remove_flags(frame) and not sidecar_path(frame).exists()
    assert not remove_flags(frame)


def test_cover_flags():
    assert normalize_flags(["cover-left", "cover-right"]) == {FrameFlag.COVER_BOTH}
    flags = FrameFlags(normalize_flags(["cover-right", "ignore-left"]))
    assert flags.cover(Side.RIGHT) and not flags.cover(Side.LEFT) and flags.ignored(Side.LEFT)
    both = FrameFlags(normalize_flags(["cover-both"]))
    assert both.cover(Side.LEFT) and both.cover(Side.RIGHT) and both.split


def test_illumination_and_text_rectify_are_separate():
    """Освещение и выпрямление по строкам выключаются отдельными флагами; обложка сама их не выключает."""
    front = FrameFlags(normalize_flags(["cover-right", "no-illumination-right", "no-text-rectify-right"]))
    assert not front.equalize(Side.RIGHT) and not front.text_rectify(Side.RIGHT)
    assert front.equalize(Side.LEFT) and front.text_rectify(Side.LEFT)
    back = FrameFlags(normalize_flags(["cover-left", "ignore-right"]))
    assert back.cover(Side.LEFT) and back.equalize(Side.LEFT) and back.text_rectify(Side.LEFT)
    picture = FrameFlags(normalize_flags(["no-text-rectify-left"]))
    assert picture.equalize(Side.LEFT) and not picture.text_rectify(Side.LEFT)


def test_legacy_no_color_equalize_reads_as_no_illumination():
    """Прежнее имя ``no-color-equalize-*`` читается и пишется уже как ``no-illumination-*``."""
    flags = FrameFlags(normalize_flags(["no-color-equalize-left"]))
    assert flags.flags == {FrameFlag.NO_ILLUMINATION_LEFT} and flags.label() == "no-illumination-left"
