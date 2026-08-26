"""База: идемпотентность записи дерева и замена разметки."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import func, select

from ocr_utils.scan_markup.db.models import Issue, Page, RasterRegion, YearPackage
from ocr_utils.scan_markup.db.repo import iter_pages, replace_raster_regions, upsert_pack
from ocr_utils.scan_markup.db.session import open_db
from ocr_utils.scan_markup.scan_tree import scan_pack


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(path)


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    root = tmp_path / "пак-1"
    _make_image(root / "1974" / "01" / "a.tif")
    _make_image(root / "1974" / "01" / "b.tif")
    _make_image(root / "1974" / "02" / "c.tif")
    return root


@pytest.fixture
def session_factory(tmp_path: Path):
    return open_db(tmp_path / "markup.sqlite")


def test_upsert_is_idempotent(pack_dir: Path, session_factory) -> None:
    """Повторный прогон по тому же паку не плодит ни годов, ни выпусков, ни полос."""
    years = scan_pack(pack_dir)
    with session_factory() as session:
        for _ in range(3):
            upsert_pack(session, "пак-1", pack_dir, years)
        assert session.scalar(select(func.count()).select_from(YearPackage)) == 1
        assert session.scalar(select(func.count()).select_from(Issue)) == 2
        assert session.scalar(select(func.count()).select_from(Page)) == 3


def test_upsert_keeps_existing_markup(pack_dir: Path, session_factory) -> None:
    """Повторный обход дерева не стирает уже записанную разметку.

    Обход и детекция — разные операции: `detect --only-year` заново перечитывает дерево
    целиком, и если бы upsert сбрасывал разметку, прогон по одному году обнулял бы все
    остальные.
    """
    years = scan_pack(pack_dir)
    with session_factory() as session:
        pack = upsert_pack(session, "пак-1", pack_dir, years)
        page = next(page for _y, _i, page in iter_pages(pack))
        replace_raster_regions(session, page, [RasterRegion(x1=1, y1=2, x2=3, y2=4, kind="color")])
        session.commit()

        upsert_pack(session, "пак-1", pack_dir, years)
        assert session.scalar(select(func.count()).select_from(RasterRegion)) == 1


def test_upsert_adds_new_pages(pack_dir: Path, session_factory) -> None:
    """Досыпанный в выпуск переcкан подхватывается следующим прогоном."""
    with session_factory() as session:
        upsert_pack(session, "пак-1", pack_dir, scan_pack(pack_dir))
        _make_image(pack_dir / "1974" / "01" / "a2.tif")
        upsert_pack(session, "пак-1", pack_dir, scan_pack(pack_dir))
        assert session.scalar(select(func.count()).select_from(Page)) == 4


def test_two_packs_live_in_one_database(pack_dir: Path, tmp_path: Path, session_factory) -> None:
    """Вторая база не заводится: новый пак дописывается в существующий файл."""
    other = tmp_path / "пак-3"
    _make_image(other / "1977" / "07" / "z.tif")
    with session_factory() as session:
        upsert_pack(session, "пак-1", pack_dir, scan_pack(pack_dir))
        upsert_pack(session, "пак-3", other, scan_pack(other))
        assert session.scalar(select(func.count()).select_from(Page)) == 4


def test_replace_regions_is_replacement_not_addition(pack_dir: Path, session_factory) -> None:
    """Повторная детекция заменяет области полосы, а не добавляется к прежним."""
    with session_factory() as session:
        pack = upsert_pack(session, "пак-1", pack_dir, scan_pack(pack_dir))
        page = next(page for _y, _i, page in iter_pages(pack))
        replace_raster_regions(session, page, [RasterRegion(x1=1, y1=2, x2=3, y2=4, kind="color")])
        session.commit()
        replace_raster_regions(session, page, [RasterRegion(x1=9, y1=9, x2=9, y2=9, kind="grayscale")])
        session.commit()

        regions = session.scalars(select(RasterRegion)).all()
        assert len(regions) == 1 and regions[0].kind == "grayscale"


def test_deleting_a_pack_takes_its_pages(pack_dir: Path, session_factory) -> None:
    """Каскад работает на уровне БД, а не только в ORM: без PRAGMA foreign_keys он молчит."""
    with session_factory() as session:
        pack = upsert_pack(session, "пак-1", pack_dir, scan_pack(pack_dir))
        session.delete(pack)
        session.commit()
        assert session.scalar(select(func.count()).select_from(Page)) == 0


def test_iter_pages_filters_and_orders(pack_dir: Path, session_factory) -> None:
    """Обход идёт год -> выпуск -> номер полосы и умеет сузиться до одного выпуска."""
    with session_factory() as session:
        pack = upsert_pack(session, "пак-1", pack_dir, scan_pack(pack_dir))
        assert [page.file_name for _y, _i, page in iter_pages(pack)] == ["a.tif", "b.tif", "c.tif"]
        assert [page.file_name for _y, _i, page in iter_pages(pack, only_issue="02")] == ["c.tif"]
        assert list(iter_pages(pack, only_year="1999")) == []
