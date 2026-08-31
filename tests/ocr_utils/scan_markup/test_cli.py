"""Команда detect от начала до конца: файлы на диске -> строки в базе."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image
from sqlalchemy import select

from ocr_utils.scan_markup.cli import main
from ocr_utils.scan_markup.db.models import KIND_COLOR, Page, RasterRegion
from ocr_utils.scan_markup.db.session import open_db
from tests.ocr_utils.scan_markup import synthetic

DPI = 300
SIZE = (1800, 1200)  # полоса 300 dpi: детекция идёт по ПОЛНОМУ кадру, 21 Мп тесту ни к чему
PHOTO_BOX = (150, 150, 900, 1000)


def _page_image(path: Path, dpi: int | None = DPI, with_photo: bool = True) -> None:
    """Полоса: бумага, текст и при необходимости — растровое пятно.

    Кадр рисуется в ПОЛНОМ разрешении, а не в рабочем масштабе 1/4: детектор смотрит на
    размер связных пятен краски, и уменьшенная копия, растянутая обратно, растровой сетки
    ему не даст. Разрешение взято 300 dpi — на нём пороги пересчитываются вдвое, и тест
    заодно проверяет, что пересчёт работает.
    """
    gray = synthetic.paper(SIZE)
    # Текст в нижней половине: без него полоса выглядит как обложка-плашка, и эвристика
    # обложки накроет её целиком (она для того и написана).
    gray[1050:1750] = synthetic.text_page((700, SIZE[1]), line_step=60, glyph_w=14, glyph_h=30, char_step=26, margin=60)
    if with_photo:
        synthetic.with_screen(gray, PHOTO_BOX, pitch=4, radius=1)

    image = Image.fromarray(np.stack([gray] * 3, -1))
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
    """Прогон detect без Surya и без допущения про обложку: проверяем сами пиксели."""
    result = CliRunner().invoke(
        main,
        [
            "detect",
            "--pack-dir",
            str(pack_dir),
            "--db",
            str(db),
            "--no-use-surya-layout",
            "--no-first-page-is-cover",
            *extra,
        ],
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
        assert (pages[0].width, pages[0].height) == (SIZE[1], SIZE[0])
        assert pages[0].detected_at is not None

        regions = session.scalars(select(RasterRegion)).all()
        assert len(regions) == 1
        assert regions[0].page_id == pages[0].id
        assert regions[0].chroma_frac is not None
        assert regions[0].chroma_spread is not None
        assert regions[0].dot_frac is not None


def test_detect_leaves_downscale_parameters_to_publish(pack_dir: Path, tmp_path: Path) -> None:
    """Делитель выбирает to-cvat по своему --cvat-dpi; detect пишет только прочитанное."""
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)

    with open_db(db)() as session:
        page = session.scalars(select(Page)).first()
        assert (page.width, page.height, page.dpi) == (SIZE[1], SIZE[0], DPI)
        assert page.divisor is None
        assert (page.crop_width, page.cvat_width) == (None, None)


def test_text_only_page_has_no_regions(pack_dir: Path, tmp_path: Path) -> None:
    """Полоса из одного текста растровых областей не даёт: буквы крупнее точки растра."""
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


def test_first_page_is_cover_marks_the_whole_frame(pack_dir: Path, tmp_path: Path) -> None:
    """С флагом первая полоса выпуска — одна цветная область во весь кадр, без детекции.

    Это структурное допущение, а не измерение: на паке-1 полоса 0 занята цветным
    изображением во всех 123 выпусках. Вторая полоса при этом обрабатывается как обычно.
    """
    db = tmp_path / "markup.sqlite"
    result = CliRunner().invoke(
        main, ["detect", "--pack-dir", str(pack_dir), "--db", str(db), "--no-use-surya-layout", "--first-page-is-cover"]
    )
    assert result.exit_code == 0, result.output

    with open_db(db)() as session:
        first = session.scalars(select(Page).where(Page.file_name == "a.tif")).one()
        assert len(first.raster_regions) == 1
        region = first.raster_regions[0]
        assert region.kind == KIND_COLOR
        assert region.full_page
        assert (region.x1, region.y1, region.x2, region.y2) == (0, 0, first.width, first.height)
        # Пиксели для такой полосы не смотрели, поэтому и измерений быть не должно.
        assert region.chroma_frac is None

        second = session.scalars(select(Page).where(Page.file_name == "b.tif")).one()
        assert second.raster_regions == []


def test_jobs_do_not_change_the_result(pack_dir: Path, tmp_path: Path) -> None:
    """Пул процессов обязан давать ровно то же, что один процесс."""
    single, parallel = tmp_path / "one.sqlite", tmp_path / "many.sqlite"
    _run(pack_dir, single, "--jobs", "1")
    _run(pack_dir, parallel, "--jobs", "2")

    def boxes(db: Path):
        with open_db(db)() as session:
            return sorted(
                (r.x1, r.y1, r.x2, r.y2, r.kind, r.full_page) for r in session.scalars(select(RasterRegion)).all()
            )

    assert boxes(single) == boxes(parallel)


def test_new_detector_version_forces_a_rerun(pack_dir: Path, tmp_path: Path, monkeypatch) -> None:
    """Смена версии детектора заставляет --skip-detected перечитать полосу.

    Без этого правка алгоритма не доехала бы до базы вовсе: файлы-то не менялись, и весь
    пак был бы молча пропущен вместе с новым детектором.
    """
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)
    assert "пропущено: 2" in _run(pack_dir, db, "--skip-detected").output

    monkeypatch.setattr("ocr_utils.scan_markup.detection.run.DETECTOR_VERSION", 999)
    result = _run(pack_dir, db, "--skip-detected")
    assert "Полос обработано: 2" in result.output
    assert "пропущено: 0" in result.output

    with open_db(db)() as session:
        assert {page.detector_version for page in session.scalars(select(Page)).all()} == {999}


def test_recolor_changes_kind_without_touching_boxes(pack_dir: Path, tmp_path: Path) -> None:
    """recolor пересчитывает только тип области; координаты остаются прежними."""
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)

    with open_db(db)() as session:
        before = [(r.x1, r.y1, r.x2, r.y2) for r in session.scalars(select(RasterRegion)).all()]
    assert before

    # Порог разброса ниже нуля объявляет цветным что угодно — этого и ждём от перекраски.
    result = CliRunner().invoke(
        main, ["recolor", "--db", str(db), "--pack-name", "пак-1", "--jobs", "1", "--chroma-spread-thr", "-1"]
    )
    assert result.exit_code == 0, result.output

    with open_db(db)() as session:
        regions = session.scalars(select(RasterRegion)).all()
        assert [(r.x1, r.y1, r.x2, r.y2) for r in regions] == before
        assert {r.kind for r in regions} == {KIND_COLOR}


def test_mark_covers_needs_no_pixels(pack_dir: Path, tmp_path: Path) -> None:
    """mark-covers правит готовую базу, не открывая ни одного оригинала."""
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)

    # Оригиналы убираем: команда обязана обойтись тем, что уже лежит в базе.
    for tif in pack_dir.rglob("*.tif"):
        tif.unlink()

    result = CliRunner().invoke(main, ["mark-covers", "--db", str(db), "--pack-name", "пак-1"])
    assert result.exit_code == 0, result.output

    with open_db(db)() as session:
        first = session.scalars(select(Page).where(Page.file_name == "a.tif")).one()
        assert len(first.raster_regions) == 1
        assert first.raster_regions[0].full_page
        assert first.raster_regions[0].kind == KIND_COLOR


def test_rerun_without_skip_detected_recomputes_everything(pack_dir: Path, tmp_path: Path) -> None:
    """Прогон без --skip-detected обязан пересчитать все полосы, а не узнать их по хешу.

    Короткое замыкание «хеш совпал — не декодируем» существует ради продолжения прерванного
    прогона и живёт только под флагом. Без флага прогон означает «пересчитать всё», и молча
    пропустить пак, потому что файлы не менялись, он не имеет права.
    """
    db = tmp_path / "markup.sqlite"
    _run(pack_dir, db)
    result = _run(pack_dir, db)
    assert "Полос обработано: 2" in result.output
    assert "пропущено: 0" in result.output
