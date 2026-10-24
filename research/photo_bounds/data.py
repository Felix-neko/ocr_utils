"""Набор стенда из базы разметки пака-1 (только чтение): эталонные рамки растровых фото из CVAT, отбор без полностраничных областей, раскладка кэша на SSD."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

# Доля площади КАДРА, с которой эталонная область считается «почти на всю полосу» (обложка, вклейка) и в набор фото не
# идёт. Замер по pack1_reviewed (404 области без ``full_page``): медиана 0.23, p95 0.63, максимум 0.73; кадр пака-1 —
# полоса плюс поля, и печатная полоса занимает ~0.75 кадра. Области 0.55–0.73 (29 штук) отсмотрены на листах:
# это снимки на всю полосу с подписью или без, у которых края полосы — край фото; они остаются отдельным поясом
# ``BIG`` (см. :class:`SizeBand`), а не выбрасываются.
NEAR_FULL_FRAC = 0.80

# Виды растра базы, которые считаются фото (``db.models.RectRegion.kind``).
PHOTO_KINDS = ("grayscale", "color")


class SizeBand(StrEnum):
    """Пояс площади эталонной области (доля кадра)."""

    SMALL = "small"  # меньше 0.05 кадра: портреты, мелкие снимки в колонке
    MEDIUM = "medium"  # 0.05–0.25
    LARGE = "large"  # 0.25–0.55
    BIG = "big"  # 0.55 и больше: снимок на всю полосу

    @classmethod
    def of(cls, frac: float) -> SizeBand:
        """Пояс по доле кадра.

        Args:
            frac: Доля площади кадра, занятая областью.

        Returns:
            Пояс.
        """
        if frac < 0.05:
            return cls.SMALL
        if frac < 0.25:
            return cls.MEDIUM
        if frac < 0.55:
            return cls.LARGE
        return cls.BIG


@dataclass(frozen=True)
class RefPhoto:
    """Эталонная рамка фото из CVAT.

    Поля:
        region_id: ``rect_regions.id`` базы.
        page_key: Полоса ``{год}/{выпуск}/{имя без расширения}`` — ключ кэша стенда и кэша surya.
        box: ``(x1, y1, x2, y2)`` в пикселях оригинала (CVAT × ``divisor``).
        kind: ``grayscale`` или ``color``.
        frac: Доля площади кадра.
        divisor: Делитель превью CVAT (точность эталона ~ ``divisor`` px оригинала).
    """

    region_id: int
    page_key: str
    box: tuple[int, int, int, int]
    kind: str
    frac: float
    divisor: int

    @property
    def key(self) -> str:
        """Ключ фото: полоса и номер области."""
        return f"{self.page_key}#{self.region_id}"


@dataclass(frozen=True)
class PageRef:
    """Полоса набора.

    Поля:
        page_key: ``{год}/{выпуск}/{имя}``.
        rel_path: Путь TIFF от корня пака.
        width, height: Размер оригинала, px.
        dpi: Разрешение оригинала.
        order_index: Номер полосы в выпуске (правило обложки детектора).
        photos: Эталонные фото полосы.
    """

    page_key: str
    rel_path: str
    width: int
    height: int
    dpi: int
    order_index: int
    photos: tuple[RefPhoto, ...]


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Открывает базу разметки ТОЛЬКО на чтение (URI ``mode=ro``): база невоспроизводима, писать в неё стенду нельзя.

    Args:
        db_path: Путь к ``pack1_reviewed.sqlite``.

    Returns:
        Соединение sqlite3 без права записи.
    """
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def load_pack_root(conn: sqlite3.Connection) -> Path:
    """Корень оригиналов пака (``packs.source_pics_root``); в базе пака-1 он один.

    Args:
        conn: Соединение с базой.

    Returns:
        Путь к корню оригиналов.
    """
    (root,) = conn.execute("select source_pics_root from packs order by id limit 1").fetchone()
    return Path(root)


def load_pages(conn: sqlite3.Connection, near_full_frac: float = NEAR_FULL_FRAC) -> list[PageRef]:
    """Полосы с эталонными фото: виды ``grayscale``/``color``, без ``full_page`` и без областей почти на весь кадр.

    Args:
        conn: Соединение с базой (только чтение).
        near_full_frac: Доля площади кадра, с которой область выбрасывается как обложка или вклейка.

    Returns:
        Полосы по порядку пути, у каждой — её фото.
    """
    rows = conn.execute(
        f"""select r.id, p.source_rel_path, p.width, p.height, p.dpi, p.order_index, p.divisor,
                   r.x1, r.y1, r.x2, r.y2, r.kind
            from rect_regions r join pages p on p.id = r.page_id
            where r.kind in ({",".join("?" * len(PHOTO_KINDS))}) and r.full_page = 0
            order by p.source_rel_path, r.id""",
        PHOTO_KINDS,
    ).fetchall()
    pages: dict[str, dict] = {}
    for rid, rel, width, height, dpi, order, divisor, x1, y1, x2, y2, kind in rows:
        frac = (x2 - x1) * (y2 - y1) / float(width * height)
        if frac >= near_full_frac:
            continue
        page_key = str(Path(rel).with_suffix(""))
        entry = pages.setdefault(page_key, dict(rel=rel, width=width, height=height, dpi=dpi, order=order, photos=[]))
        entry["photos"].append(RefPhoto(rid, page_key, (x1, y1, x2, y2), kind, round(frac, 5), int(divisor or 8)))
    return [
        PageRef(k, v["rel"], v["width"], v["height"], v["dpi"], v["order"], tuple(v["photos"]))
        for k, v in sorted(pages.items())
    ]


@dataclass(frozen=True)
class CacheLayout:
    """Раскладка кэша стенда под корнем ``root`` (SSD hotstore).

    ``pages/<полоса>/``: ``gray300.png`` (серый 300 dpi), ``bgr150.jpg`` (цвет 150 dpi для оверлеев),
    ``detector.json`` (растр ``page_layout`` и блоки surya ``Picture``).
    """

    root: Path

    def page_dir(self, page_key: str) -> Path:
        """Папка полосы."""
        return self.root / "pages" / page_key

    def gray(self, page_key: str) -> Path:
        """Серый 300 dpi."""
        return self.page_dir(page_key) / "gray300.png"

    def bgr(self, page_key: str) -> Path:
        """Цвет 150 dpi."""
        return self.page_dir(page_key) / "bgr150.jpg"

    def detector(self, page_key: str) -> Path:
        """Находки детектора."""
        return self.page_dir(page_key) / "detector.json"

    def pages_json(self) -> Path:
        """Список полос набора с эталоном (снимок базы на момент ``select``)."""
        return self.root / "pages.json"


def save_pages(path: Path, pages: list[PageRef]) -> None:
    """Пишет снимок набора в JSON (дальше стенд базу не открывает).

    Args:
        path: Куда писать.
        pages: Полосы набора.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in pages], ensure_ascii=False, indent=1))


def read_pages(path: Path) -> list[PageRef]:
    """Читает снимок набора, записанный :func:`save_pages`.

    Args:
        path: JSON набора.

    Returns:
        Полосы набора.
    """
    out = []
    for p in json.loads(path.read_text()):
        photos = tuple(RefPhoto(**{**ph, "box": tuple(ph["box"])}) for ph in p["photos"])
        out.append(PageRef(**{**p, "photos": photos}))
    return out
