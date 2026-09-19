"""Стадия анализа (источник по растру и кэшу геометрии) и сборка выпуска со сверкой."""

import json
import shutil
from pathlib import Path

import fitz

from ocr_utils.final_pdfs import VERSION
from ocr_utils.final_pdfs.analysis import AnalysisParams, analyse_chunk, chunk_jobs, load_analysis, page_json_path
from ocr_utils.final_pdfs.assemble import AssembleParams, assemble_issue
from ocr_utils.final_pdfs.plan import IssuePlan, PagePlan, PageSource
from ocr_utils.final_pdfs.sources import IssuePair
from ocr_utils.geometry_regression import VERSION as GEOMETRY_VERSION
from ocr_utils.geometry_regression.cache import cache_path as geometry_cache_path
from ocr_utils.text_layer_fix import VERSION as TEXT_LAYER_VERSION
from ocr_utils.text_layer_fix.raster import page_raster
from tests.ocr_utils.final_pdfs.conftest import INSET, MARGINS


def _fake_geometry_cache(run_dir: Path, pdf_stem: str, page: int, bad: bool) -> None:
    """JSON детектора геометрии текущей версии: порча линеек 5 мм → bad, нули → ok."""
    metrics = {"vstroke_dev_max_delta_mm": 5.0 if bad else 0.0, "seconds": 0.1}
    path = geometry_cache_path(run_dir, pdf_stem, page)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": GEOMETRY_VERSION, "metrics": metrics, "culprits": {}, "raw": {}}))


def _analysis(pack, text_layer: bool = False) -> AnalysisParams:
    _, _, plan, pair, tmp_path = pack
    run_dir = tmp_path / "geometry"
    # Страница 1 (без растра) — «bad», остальные без растра — «ok».
    _fake_geometry_cache(run_dir, pair.geo.stem, 1, True)
    for page in (2, 3, 4):
        _fake_geometry_cache(run_dir, pair.geo.stem, page, False)
    return AnalysisParams(work_dir=tmp_path / "work", geometry_run_dir=run_dir, text_layer=text_layer)


def test_analysis_decides_sources(pack) -> None:
    _, _, plan, pair, _ = pack
    params = _analysis(pack)
    rows = [row for job in chunk_jobs(plan, pair, params, chunk=3) for row in analyse_chunk(job)]
    assert [r.source for r in rows] == ["nogeo", "nogeo", "nogeo", "nogeo"]
    assert rows[0].geometry_verdict == "bad" and rows[0].reason == "коррекция испортила геометрию"
    assert rows[1].reason == "растр в базе" and rows[1].geometry_verdict == ""
    assert all(not r.error for r in rows) and all(r.geometry_cached for r in rows if r.geometry_verdict)
    payload = load_analysis(page_json_path(params, plan, 0))
    assert payload["analysis_version"] == VERSION and payload["source_pdf"] == str(pair.nogeo)
    # Повторный прогон берёт JSON из кэша.
    again = analyse_chunk(chunk_jobs(plan, pair, params)[0])
    assert all(r.cached for r in again)


def test_assemble_places_pictures_and_verifies(pack) -> None:
    _, blurred, plan, pair, tmp_path = pack
    params = _analysis(pack)
    for job in chunk_jobs(plan, pair, params):
        analyse_chunk(job)
    assemble = AssembleParams(
        analysis=params, out_dir=tmp_path / "final", pictures_dir=blurred, margins=MARGINS, preview_pages=1
    )
    result = assemble_issue(plan, pair, assemble)
    assert result.status == "ok", result.reason
    assert result.pages == 4 and result.pages_nogeo == 4 and result.pictures == 3
    assert result.verify_failures == 0 and result.figures_removed == 3  # две фигуры под врезками + образ обложки
    with fitz.open(result.out_path) as doc:
        assert len(doc) == 4
        # Врезка на прямой полосе: в центре рамки — серый JPEG (градиент), а не заливка страницы.
        scale = 72.0 / 600
        center = fitz.Point((INSET[0] + INSET[2]) / 2 + MARGINS.x_px, (INSET[1] + INSET[3]) / 2 + MARGINS.y_px) * scale
        pix = doc[1].get_pixmap(dpi=600, clip=fitz.Rect(center.x - 1, center.y - 1, center.x + 1, center.y + 1))
        r, g, b = pix.pixel(0, 0)[:3]
        assert abs(r - g) < 8 and abs(g - b) < 8  # серый
        # Обложка: только JPEG, бинарного образа нет.
        infos = doc[2].get_image_info(xrefs=True)
        assert len(infos) == 1 and infos[0]["cs-name"] == "DeviceRGB"
        # Повёрнутая полоса: врезка лежит по повёрнутой рамке.
        placed = plan.pages[3].placed_pictures()[0]
        expect = (
            fitz.Rect(
                placed.x1 + MARGINS.x_px, placed.y1 + MARGINS.y_px, placed.x2 + MARGINS.x_px, placed.y2 + MARGINS.y_px
            )
            * scale
        )
        jpeg = [i for i in doc[3].get_image_info(xrefs=True) if i["cs-name"] == "DeviceGray" and i["width"] == 100]
        assert (
            jpeg
            and abs(fitz.Rect(jpeg[0]["bbox"]).x0 - expect.x0) < 0.5
            and abs(fitz.Rect(jpeg[0]["bbox"]).y0 - expect.y0) < 0.5
        )
    assert (params.work_dir / "preview").exists()
    # Повторная сборка пропускается.
    assert assemble_issue(plan, pair, assemble).status == "skipped"


def test_assemble_refuses_shifted_pages(pack) -> None:
    _, blurred, plan, pair, tmp_path = pack
    params = _analysis(pack)
    for job in chunk_jobs(plan, pair, params):
        analyse_chunk(job)
    # Подменяем PDF без коррекции: страницы сдвинуты — размеры с полосами уже не сходятся.
    with fitz.open(str(pair.nogeo)) as doc:
        doc.move_page(0)  # первая страница уезжает в конец: на месте повёрнутой полосы прямая
        shifted = tmp_path / "shifted.pdf"
        doc.save(str(shifted))
    shutil.copy(shifted, pair.nogeo)
    result = assemble_issue(
        plan, pair, AssembleParams(analysis=params, out_dir=tmp_path / "final", pictures_dir=blurred, margins=MARGINS)
    )
    assert result.status == "error" and "порядок страниц" in result.reason
    assert not (tmp_path / "final" / "1970_01.pdf").exists()


def test_assemble_applies_text_layer_edits(layer_pdf, tmp_path) -> None:
    """Страница с потоком FineReader: слово DELETE исчезает, принятое чтение вписывается и ищется."""
    src, _ = layer_pdf
    geo_dir, nogeo_dir = tmp_path / "geo", tmp_path / "nogeo"
    geo_dir.mkdir(), nogeo_dir.mkdir()
    shutil.copy(src, geo_dir / "full_1970_01.pdf")
    shutil.copy(src, nogeo_dir / "full_1970_01.pdf")
    plan = IssuePlan(1, "1970", "01", (PagePlan(1, "1970/01/x.tif", "1970/01/x.jpg", 400, 600, 600, (), 0, None),))
    pair = IssuePair(geo_dir / "full_1970_01.pdf", nogeo_dir / "full_1970_01.pdf", 1)
    params = AnalysisParams(work_dir=tmp_path / "work", geometry_run_dir=None, text_layer=False)
    with fitz.open(str(src)) as doc:
        raster = page_raster(doc[0])
        zone_box = [round(v) for v in fitz.Rect(300, 100, 312, 300) * raster.to_px()]
    payload = {
        "version": TEXT_LAYER_VERSION,
        "analysis_version": VERSION,
        "pdf": "full_1970_01.pdf",
        "page": 0,
        "source": PageSource.GEO.value,
        "source_pdf": str(pair.geo),
        "reason": "коррекция не навредила",
        "geometry_verdict": "ok",
        "zones": [{"box": zone_box, "kind": "standalone", "rotate_cw": 90}],
        "words": [
            {"mcid": 0, "verdict": "delete", "keep_glyphs": [], "zone_index": 0},
            {"mcid": 1, "verdict": "keep", "keep_glyphs": [], "zone_index": None},
            {"mcid": 2, "verdict": "keep_rotated", "keep_glyphs": [], "zone_index": None},
        ],
        "readings": {"0": {"text": "Склад", "accepted": True, "rotate_cw": 90, "confidence": 0.9}},
        "error": "",
    }
    path = page_json_path(params, plan, 0)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, ensure_ascii=False))
    from ocr_utils.final_pdfs.sources import Margins

    result = assemble_issue(
        plan,
        pair,
        AssembleParams(analysis=params, out_dir=tmp_path / "final", pictures_dir=tmp_path, margins=Margins(0, 0)),
    )
    assert result.status == "ok", result.reason
    record = result.records[0]
    assert (record.blanked, record.inserted, record.verify_ok) == (1, 1, True), record.verify_notes
    with fitz.open(result.out_path) as doc:
        assert not doc[0].search_for("Таблица")
        assert doc[0].search_for("Склад") and doc[0].search_for("Резервы")
