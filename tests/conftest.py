"""Общие фикстуры для тестов."""

from __future__ import annotations

import tempfile
from pathlib import Path

import fitz
import pytest


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Временная директория для тестов (pytest автоматически очистит)."""
    return tmp_path


@pytest.fixture
def portrait_pdf(tmp_path: Path) -> Path:
    """Создать простой PDF с одной портретной страницей (595×842 pt, ~A4)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Рисуем простой прямоугольник, чтобы страница не была пустой
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(50, 50, 545, 792))
    shape.finish(color=(0, 0, 0))
    shape.commit()
    path = tmp_path / "portrait.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def landscape_spread_pdf(tmp_path: Path) -> Path:
    """Создать PDF с одной альбомной страницей-разворотом (1190×842 pt, aspect ≈ 1.41)."""
    doc = fitz.open()
    page = doc.new_page(width=1190, height=842)
    shape = page.new_shape()
    # Левая половина
    shape.draw_rect(fitz.Rect(50, 50, 545, 792))
    shape.finish(color=(1, 0, 0))
    # Правая половина
    shape.draw_rect(fitz.Rect(645, 50, 1140, 792))
    shape.finish(color=(0, 0, 1))
    shape.commit()
    path = tmp_path / "spread.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def mixed_pdf(tmp_path: Path) -> Path:
    """PDF с 3 страницами: портрет, разворот, портрет."""
    doc = fitz.open()
    # Страница 0: портрет
    p0 = doc.new_page(width=595, height=842)
    shape = p0.new_shape()
    shape.draw_rect(fitz.Rect(50, 50, 545, 792))
    shape.finish(color=(0, 0, 0))
    shape.commit()
    # Страница 1: разворот
    p1 = doc.new_page(width=1190, height=842)
    shape = p1.new_shape()
    shape.draw_rect(fitz.Rect(50, 50, 1140, 792))
    shape.finish(color=(0.5, 0.5, 0.5))
    shape.commit()
    # Страница 2: портрет
    p2 = doc.new_page(width=595, height=842)
    shape = p2.new_shape()
    shape.draw_rect(fitz.Rect(50, 50, 545, 792))
    shape.finish(color=(0, 0, 0))
    shape.commit()
    path = tmp_path / "mixed.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def jpeg_spread_pdf(tmp_path: Path) -> Path:
    """PDF с разворотом, содержащим JPEG-изображение."""
    from PIL import Image
    import io

    # Создаём JPEG-изображение 2400×1700 (альбомное, aspect ≈ 1.41)
    img = Image.new("RGB", (2400, 1700), color=(200, 180, 160))
    jpeg_buf = io.BytesIO()
    img.save(jpeg_buf, format="JPEG", quality=85)
    jpeg_bytes = jpeg_buf.getvalue()

    doc = fitz.open()
    # Страница с размерами, соответствующими развороту
    page = doc.new_page(width=1200, height=850)
    page.insert_image(fitz.Rect(0, 0, 1200, 850), stream=jpeg_bytes)
    path = tmp_path / "jpeg_spread.pdf"
    doc.save(str(path))
    doc.close()
    return path
