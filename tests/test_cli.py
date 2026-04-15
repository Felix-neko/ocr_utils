"""Тесты CLI (__main__.py)."""

from __future__ import annotations

import pytest

from ocr_utils.__main__ import _parse_pages


class TestParsePages:
    """Тесты парсинга строки с номерами страниц."""

    def test_all(self) -> None:
        assert _parse_pages("all") is None
        assert _parse_pages("ALL") is None

    def test_comma_separated(self) -> None:
        assert _parse_pages("0,2,5") == [0, 2, 5]
        assert _parse_pages("0, 2, 5") == [0, 2, 5]

    def test_single_page(self) -> None:
        assert _parse_pages("3") == [3]

    def test_slice_two_parts(self) -> None:
        s = _parse_pages("1:10")
        assert isinstance(s, slice)
        assert s == slice(1, 10)

    def test_slice_three_parts(self) -> None:
        s = _parse_pages("1:10:2")
        assert isinstance(s, slice)
        assert s == slice(1, 10, 2)

    def test_slice_open_start(self) -> None:
        s = _parse_pages(":5")
        assert isinstance(s, slice)
        assert s == slice(None, 5)

    def test_slice_open_end(self) -> None:
        s = _parse_pages("3:")
        assert isinstance(s, slice)
        assert s == slice(3, None)
