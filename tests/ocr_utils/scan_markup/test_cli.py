"""Команда detect от начала до конца: файлы на диске -> строки в базе."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image
from sqlalchemy import select

from ocr_utils.scan_markup.cli import main
from ocr_utils.scan_markup.db.models import Page, RasterRegion
from ocr_utils.scan_markup.db.session import open_db

DPI = 600
SIZE = (873, 1513)  # 1/4 от кадра 600 dpi — на этом масштабе откалиброваны пороги


def _page_image(path: Path, dpi: int | None = DPI, with_photo: bool = True) -> None:
    """Полоса: белая бумага, тёмные строки, при необходимости — полутоновое пятно.

    Кадр рисуется сразу в РАБОЧЕМ масштабе (1/4), а размер множится на 4, чтобы детекция
    получила ту же картинку, что и на настоящем скане, но тест не гонял 21 мегапиксель.
    """
    rng = np.random.default_rng(3)
    gray = np.full(SIZE, 245, np.uint8)
    for y in range(1000, 1480, 30):
        gray[y : y + 10, 100:770] = 20
    if with_photo:
        gray[100:700, 150:700] = rng.integers(120, 210, (600, 550), dtype=np.uint8)

    image = Image.fromarray(np.stack([gray] * 3, -1)).resize((SIZE[1] * 4, SIZE[0] * 4), Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    # dpi=None не даёт файла БЕЗ тега: PIL всё равно запишет туда 1 dpi — ровно тот мусор,
    # который детекция обязана отличать от настоящего разрешения.
    image.save(path, **({"dpi": (dpi, dpi)} if dpi else {}))


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    root = tmp_path / "пак-1"
    _page_image(root / "1974" / "01" / "a.tif")
    _page_image(root / "1974" / "01" / "b.tif", with_photo=False)
    return root


def _run(pack_dir: Path, db: Path, *extra: str):
    """Прогон detect без Surya: тесту не нужен ни GPU, ни веса моделей."""
    result = CliRunner().invoke(
        main, ["detect", "--pack-dir", str(pack_dir), "--db", str(db), "--no-use-surya-layout", *extra]
    )
    assert result.exit_code == 0, result.output
    return result


def test_detect_writes_pages_and_regions(pack_dir: Path, tmp_path: Path) -> None:
    """Полосы попадают в базу с размерами и DPI, фотография — в растровые области."""
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)

    with open_db(db)() as session:
        pages = session.scalars(select(Page).order_by(Page.file_name)).all()
        assert [page.file_name for page in pages] == ["a.tif", "b.tif"]
        assert pages[0].dpi == DPI
        assert (pages[0].width, pages[0].height) == (SIZE[1] * 4, SIZE[0] * 4)
        assert pages[0].detected_at is not None

        regions = session.scalars(select(RasterRegion)).all()
        assert len(regions) == 1
        assert regions[0].page_id == pages[0].id
        assert regions[0].chroma_frac is not None


def test_detect_leaves_downscale_parameters_to_publish(pack_dir: Path, tmp_path: Path) -> None:
    """Делитель выбирает to-cvat по своему --cvat-dpi; detect пишет только прочитанное."""
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)

    with open_db(db)() as session:
        page = session.scalars(select(Page)).first()
        assert (page.width, page.height, page.dpi) == (SIZE[1] * 4, SIZE[0] * 4, DPI)
        assert page.divisor is None
        assert (page.crop_width, page.cvat_width) == (None, None)


def test_text_only_page_has_no_regions(pack_dir: Path, tmp_path: Path) -> None:
    """Полоса из одного текста растровых областей не даёт: серое у букв убирает размыкание."""
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)

    with open_db(db)() as session:
        page = session.scalars(select(Page).where(Page.file_name == "b.tif")).one()
        assert page.raster_regions == []
        assert page.detected_at is not None


def test_rerun_does_not_duplicate(pack_dir: Path, tmp_path: Path) -> None:
    """Повторный прогон переписывает разметку, а не удваивает её."""
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)
    _run(pack_dir, db)

    with open_db(db)() as session:
        assert len(session.scalars(select(Page)).all()) == 2
        assert len(session.scalars(select(RasterRegion)).all()) == 1


def test_skip_detected_leaves_pages_alone(pack_dir: Path, tmp_path: Path) -> None:
    """--skip-detected не переобрабатывает уже посчитанные полосы."""
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)
    result = _run(pack_dir, db, "--skip-detected")
    assert "пропущено: 2" in result.output


def test_missing_dpi_is_reported_not_guessed(tmp_path: Path) -> None:
    """Файл с мусорным разрешением пропускается с ошибкой, а не берётся всерьёз.

    PIL и сканеры кладут в такой TIFF 1 dpi. Принять эту единицу — значит получить
    делитель 1 и залить в CVAT полноразмерные сканы, ровно то, из-за чего разметка и
    тормозила, причём молча.
    """
    root = tmp_path / "пак-1"
    _page_image(root / "1974" / "01" / "a.tif", dpi=None)
    db = tmp_path / "markup.sqlite"
    result = _run(root, db)

    assert "ОШИБКА" in result.output and "ошибок: 1" in result.output
    with open_db(db)() as session:
        assert session.scalars(select(Page)).one().detected_at is None


def test_default_dpi_rescues_untagged_files(tmp_path: Path) -> None:
    """--default-dpi позволяет обработать пак без тегов разрешения."""
    root = tmp_path / "пак-1"
    _page_image(root / "1974" / "01" / "a.tif", dpi=None)
    db = tmp_path / "markup.sqlite"
    _run(root, db, "--default-dpi", "450")

    with open_db(db)() as session:
        page = session.scalars(select(Page)).one()
        assert page.dpi == 450


def test_limit_and_debug_dir(pack_dir: Path, tmp_path: Path) -> None:
    """--limit ограничивает прогон, --debug-dir пишет оверлеи для калибровки порогов."""
    db = tmp_path / "markup.sqlite"
    debug = tmp_path / "debug"
    _run(pack_dir, db, "--limit", "1", "--debug-dir", str(debug))

    with open_db(db)() as session:
        assert len(session.scalars(select(RasterRegion)).all()) == 1
    assert list(debug.glob("*.jpg"))


def test_detect_records_file_hash(pack_dir: Path, tmp_path: Path) -> None:
    """Отпечаток файла снимается заодно с детекцией — файл всё равно читается целиком."""
    import hashlib

    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)

    with open_db(db)() as session:
        page = session.scalars(select(Page).where(Page.file_name == "a.tif")).one()
        expected = hashlib.sha256((pack_dir / "1974" / "01" / "a.tif").read_bytes()).hexdigest()
        assert page.file_hash == expected
        assert page.hash_algo == "sha256"
        assert page.file_size == (pack_dir / "1974" / "01" / "a.tif").stat().st_size
        # В CVAT ещё ничего не заливали, поэтому расхождению взяться неоткуда.
        assert page.cvat_file_hash is None


def test_skip_detected_ignores_untouched_files(pack_dir: Path, tmp_path: Path) -> None:
    """Повторный прогон не перечитывает то, что не менялось."""
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)
    result = _run(pack_dir, db, "--skip-detected")
    assert "пропущено: 2" in result.output
    assert "изменилось с прошлого прогона: 0" in result.output


def test_skip_detected_still_reprocesses_a_replaced_file(pack_dir: Path, tmp_path: Path) -> None:
    """Подменённую полосу --skip-detected обязан заметить и пересчитать.

    Иначе флаг превращался бы в ловушку: обновил сканы, прогнал detect, а в базе остались
    старые области от прежнего файла.
    """
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)

    # Была полоса без фотографии — стала с фотографией.
    _page_image(pack_dir / "1974" / "01" / "b.tif", with_photo=True)

    result = _run(pack_dir, db, "--skip-detected")
    assert "Полос обработано: 1" in result.output
    assert "изменилось с прошлого прогона: 1" in result.output

    with open_db(db)() as session:
        page = session.scalars(select(Page).where(Page.file_name == "b.tif")).one()
        assert page.raster_regions, "у подменённой полосы должна появиться найденная область"


def test_recopied_file_with_same_content_is_not_re_detected(pack_dir: Path, tmp_path: Path) -> None:
    """Файл переписали тем же содержимым: ``stat`` разошёлся, хеш — нет, детекция не нужна."""
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)

    target = pack_dir / "1974" / "01" / "a.tif"
    target.write_bytes(target.read_bytes())  # новый mtime, то же содержимое

    result = _run(pack_dir, db, "--skip-detected")
    assert "пропущено: 2" in result.output
    assert "изменилось с прошлого прогона: 0" in result.output
