"""Источники стенда: разбор ``page_layout`` обоих вариантов страницы, кэш v16 (метрики и поле смещений B → A), имена страниц."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr_utils.geometry_regression.field import Field
from ocr_utils.page_layout.text_blocks.store import PageGeometry, load_page

# Разрешение координат: рабочая копия текстовых блоков и поля смещений v16 — оба 150 dpi от рендера 300 dpi.
WORK_DPI = 150.0


@dataclass(frozen=True)
class PageRef:
    """Страница пары PDF: имя PDF без расширения и номер страницы с единицы (как в кэше v16 и эталоне)."""

    pdf: str
    page: int

    @property
    def layout_key(self) -> str:
        """Имя файла разбора ``page_layout``: ``<pdf>_p<номер с нуля:04d>``."""
        return f"{self.pdf}_p{self.page - 1:04d}"

    @property
    def label(self) -> str:
        """Подпись для отчётов и эталона: ``full_ГГГГ_НН с.N``."""
        return f"{self.pdf} с.{self.page}"


@dataclass(frozen=True)
class PagePair:
    """Всё о странице, что нужно мерам v17.

    Attributes:
        ref: Страница.
        b: Разбор варианта без коррекции.
        a: Разбор варианта с коррекцией.
        field: Поле смещений B → A (150 dpi) или ``None``, если v16 его не построил.
        v16: Метрики v16 страницы (плоский словарь).
        v16_raw: Сырьё v16 (рамки line art, таблиц, растра на B, размеры).
        v16_culprits: Виновники метрик v16 (рамки и отрезки в B и A, 150 dpi).
    """

    ref: PageRef
    b: PageGeometry
    a: PageGeometry
    field: Field | None
    v16: dict[str, float]
    v16_raw: dict
    v16_culprits: dict


def field_from_raw(raw: dict) -> Field | None:
    """Поле смещений из сырья кэша v16.

    В кэше лежат тайлы ``(cx, cy, ux, uy, peak)``, веса IRLS и аффинная матрица; остаток тайла — его
    смещение минус предсказание аффинной части (как в ``field.estimate_field``).

    Args:
        raw: ``raw`` из JSON кэша v16.

    Returns:
        :class:`Field` или ``None``, если поля нет.
    """
    payload = raw.get("field")
    if not payload or not payload.get("tiles"):
        return None
    tiles = np.asarray(payload["tiles"], dtype=np.float64)
    affine = np.asarray(payload["affine"], dtype=np.float64)
    weight = np.asarray(payload["weight"], dtype=np.float64)
    centres = tiles[:, :2]
    predicted = centres @ affine[:, :2].T + affine[:, 2] - centres
    resid = tiles[:, 2:4] - predicted
    width, height = raw.get("size_b", [0, 0])
    return Field(int(width), int(height), WORK_DPI, tiles, affine, resid, weight, np.zeros((0, 3)))


def load_v16(run_dir: Path, ref: PageRef) -> dict | None:
    """JSON кэша v16 страницы или ``None``, если его нет."""
    path = run_dir / "cache" / ref.pdf / f"p{ref.page:03d}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_pair(ref: PageRef, layout_root: Path, v16_dir: Path) -> PagePair:
    """Собрать пару страницы.

    Args:
        ref: Страница.
        layout_root: Корень разбора v6 (``<root>/{geo,nogeo}/pages``).
        v16_dir: Каталог прогона v16 (``<dir>/cache``).

    Returns:
        :class:`PagePair`.

    Raises:
        FileNotFoundError: Нет разбора одного из вариантов или кэша v16.
    """
    b = load_page(layout_root / "nogeo" / "pages" / f"{ref.layout_key}.json")
    a = load_page(layout_root / "geo" / "pages" / f"{ref.layout_key}.json")
    cached = load_v16(v16_dir, ref)
    if cached is None:
        raise FileNotFoundError(f"нет кэша v16 для {ref.label}")
    raw = cached.get("raw", {})
    return PagePair(ref, b, a, field_from_raw(raw), dict(cached.get("metrics", {})), raw, dict(cached.get("culprits", {})))


def list_pages(layout_root: Path, pdfs: set[str] | None = None) -> list[PageRef]:
    """Страницы, у которых есть разбор обоих вариантов.

    Args:
        layout_root: Корень разбора v6.
        pdfs: Только эти PDF (имена без расширения); ``None`` — все.

    Returns:
        Страницы по порядку.
    """
    geo = {p.stem for p in (layout_root / "geo" / "pages").glob("*.json")}
    out = []
    for path in sorted((layout_root / "nogeo" / "pages").glob("*.json")):
        if path.stem not in geo:
            continue
        pdf, _, index = path.stem.rpartition("_p")
        if pdfs is not None and pdf not in pdfs:
            continue
        out.append(PageRef(pdf, int(index) + 1))
    return out


def object_boxes(geometry: PageGeometry, classes: set[str]) -> list[tuple[float, float, float, float]]:
    """Рамки объектов разбора заданных классов в пикселях рабочей копии (150 dpi).

    Args:
        geometry: Разбор страницы.
        classes: Классы объектов (``pack_analysis.final.PageClass`` строками).

    Returns:
        Рамки ``(x0, y0, x1, y1)``.
    """
    k = geometry.dpi / geometry.dpi_native
    return [tuple(float(v) * k for v in obj["box"]) for obj in geometry.objects if obj["class"] in classes]


__all__ = ["PagePair", "PageRef", "WORK_DPI", "field_from_raw", "list_pages", "load_pair", "load_v16", "object_boxes"]
