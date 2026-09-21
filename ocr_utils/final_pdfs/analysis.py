"""Стадия A: анализ страниц в пуле процессов — выбор источника и разбор текстового слоя.

Задача пула — пачка страниц одного выпуска (:data:`CHUNK_PAGES`): воркер открывает оба PDF
один раз, по каждой странице решает источник (растр → без коррекции; иначе вердикт детектора
геометрии из кэша прогона или измерение на месте), разбирает текстовый слой выбранной страницы
(:func:`ocr_utils.text_layer_fix.pipeline.process_page`) и пишет JSON в
``<work_dir>/pages/<pdf>/pNNNN.json`` — тот же формат, что у стенда ``text_layer_fix``, плюс поля
решения (``source``, ``source_pdf``, ``geometry_*``, ``analysis_version``). Второе мнение surya
(стадия B) и сборка (стадия C) читают эти JSON.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ocr_utils.final_pdfs import VERSION
from ocr_utils.final_pdfs.plan import IssuePlan, PageDecision, PageSource, decide_source
from ocr_utils.final_pdfs.sources import IssuePair
from ocr_utils.geometry_regression.cache import verdict_for_page
from ocr_utils.geometry_regression.metrics import Params as GeometryParams
from ocr_utils.page_layout.surya.source import SuryaSourceConfig
from ocr_utils.geometry_regression.scoring import Thresholds
from ocr_utils.text_layer_fix import VERSION as TEXT_LAYER_VERSION
from ocr_utils.text_layer_fix.cache import cache_path, load_page, save_page
from ocr_utils.text_layer_fix.classify import Verdict

logger = logging.getLogger(__name__)

# Страниц одного выпуска на задачу пула: оба PDF открываются в воркере один раз на пачку.
CHUNK_PAGES = 12

# Словарь для страховки вердиктов (pymorphy3) грузится в воркере один раз.
_KNOWN = None


@dataclass(frozen=True)
class AnalysisParams:
    """Параметры стадии анализа (пиклуются в воркер)."""

    work_dir: Path
    geometry_run_dir: Path | None  # каталог прогона детектора геометрии с cache/; None — всегда мерить
    geometry_thr: tuple[str, ...] = ()  # переопределения порогов «имя=значение»
    geometry_hard: float = 5.0
    geometry_min_gain: float = 1.0
    geometry_ratio: float = 0.75
    text_layer: bool = True  # разбирать ли слой (False — только выбор источника)
    allowed_rotations: tuple[int, ...] = (0, 90, 180, 270)
    lang: str = "rus"
    free_text: bool = True
    skip_done: bool = True
    # Корень кэша surya page_layout: детектор геометрии (рамки по fr_nogeo) и правка слоя (по
    # выбранной странице) берут блоки из него; в воркере промах — ошибка страницы, кэш набивает
    # стадия 0 в родителе. ``None`` — разбор без surya.
    layout_cache_dir: Path | None = None

    @property
    def pages_dir(self) -> Path:
        """Корень JSON страниц."""
        return self.work_dir / "pages"

    def thresholds(self) -> Thresholds:
        return Thresholds.parse(self.geometry_thr, self.geometry_hard, self.geometry_min_gain, self.geometry_ratio)


@dataclass
class PageAnalysis:
    """Строка отчёта по странице после стадии A."""

    pdf: str
    page: int  # с нуля
    source: str = ""
    reason: str = ""
    geometry_verdict: str = ""
    geometry_score: float = 0.0
    geometry_reason: str = ""
    geometry_cached: bool = True
    pictures: int = 0
    zones: int = 0
    readings_accepted: int = 0
    words_delete: int = 0
    words_sanitize: int = 0
    seconds: float = 0.0
    cached: bool = False  # JSON взят из кэша целиком
    error: str = ""

    @classmethod
    def from_payload(cls, payload: dict, cached: bool) -> "PageAnalysis":
        """Строка отчёта из JSON страницы."""
        words = payload.get("words", [])
        readings = payload.get("readings", {})
        return cls(
            pdf=payload["pdf"],
            page=int(payload["page"]),
            source=payload.get("source", ""),
            reason=payload.get("reason", ""),
            geometry_verdict=payload.get("geometry_verdict", ""),
            geometry_score=float(payload.get("geometry_score", 0.0)),
            geometry_reason=payload.get("geometry_reason", ""),
            geometry_cached=bool(payload.get("geometry_cached", True)),
            pictures=int(payload.get("pictures", 0)),
            zones=len(payload.get("zones", [])),
            readings_accepted=sum(1 for r in readings.values() if r.get("accepted")),
            words_delete=sum(1 for w in words if w.get("verdict") == Verdict.DELETE.value),
            words_sanitize=sum(1 for w in words if w.get("verdict") == Verdict.SANITIZE.value),
            seconds=float(payload.get("seconds", 0.0)),
            cached=cached,
            error=payload.get("error", ""),
        )

    def to_row(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisJob:
    """Пачка страниц одного выпуска для воркера."""

    plan: IssuePlan
    pair: IssuePair
    indices: tuple[int, ...]  # номера страниц с нуля
    params: AnalysisParams


def load_analysis(path: Path) -> dict | None:
    """JSON страницы после стадии A, если он текущей версии сборщика и слоя.

    В отличие от :func:`ocr_utils.text_layer_fix.cache.load_page`, запись с ошибкой разбора слоя
    НЕ отбрасывается: решение об источнике в ней действительно, а слой просто не правится.

    Args:
        path: Путь JSON.

    Returns:
        Словарь или ``None``.
    """
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("analysis_version") != VERSION or payload.get("version") != TEXT_LAYER_VERSION:
        return None
    return payload


def page_json_path(params: AnalysisParams, plan: IssuePlan, index: int) -> Path:
    """Путь JSON страницы выпуска."""
    return cache_path(params.pages_dir, plan.full_pdf_name, index)


def _known_words():
    """Словарная проверка pymorphy3, один экземпляр на воркер."""
    global _KNOWN
    if _KNOWN is None:
        from ocr_utils.external_ocr_services.hyphen_join import default_morph

        _KNOWN = default_morph().known
    return _KNOWN


def decide_page(
    plan: IssuePlan, pair: IssuePair, index: int, geo_doc, nogeo_doc, params: AnalysisParams
) -> PageDecision:
    """Решение по источнику страницы: растр из базы, иначе детектор геометрии.

    Args:
        plan: План выпуска.
        pair: Пара PDF.
        index: Номер страницы с нуля.
        geo_doc: Открытый PDF с коррекцией.
        nogeo_doc: Открытый PDF без коррекции.
        params: Параметры (каталог кэша детектора, пороги).

    Returns:
        :class:`PageDecision`.
    """
    page_plan = plan.pages[index]
    if page_plan.pictures:
        source, reason = decide_source(True, None)
        return PageDecision(source, reason)
    stem = Path(pair.geo).stem
    geometry_params = GeometryParams(layout_cache_dir=params.layout_cache_dir)
    result = verdict_for_page(
        params.geometry_run_dir, stem, index + 1, geo_doc, nogeo_doc, params.thresholds(), geometry_params
    )
    source, reason = decide_source(False, result.verdict.verdict)
    return PageDecision(
        source,
        reason,
        result.verdict.verdict,
        round(float(result.verdict.score), 3),
        result.verdict.reason,
        result.cached,
    )


def analyse_chunk(job: AnalysisJob) -> list[PageAnalysis]:
    """Разобрать пачку страниц выпуска (в воркере): JSON на страницу, строки отчёта.

    Args:
        job: Пачка.

    Returns:
        По строке :class:`PageAnalysis` на страницу; ошибки — в поле ``error``, исключений наружу нет.
    """
    import fitz
    import pikepdf

    from ocr_utils.text_layer_fix.pipeline import Options, process_page

    params = job.params
    rows: list[PageAnalysis] = []
    with fitz.open(str(job.pair.geo)) as geo, fitz.open(str(job.pair.nogeo)) as nogeo:
        docs = {PageSource.GEO: geo, PageSource.NOGEO: nogeo}
        pdfs: dict[PageSource, "pikepdf.Pdf | None"] = {PageSource.GEO: None, PageSource.NOGEO: None}
        for index in job.indices:
            target = page_json_path(params, job.plan, index)
            cached = load_analysis(target) if params.skip_done else None
            if cached is not None:
                rows.append(PageAnalysis.from_payload(cached, True))
                continue
            started = time.time()
            try:
                decision = decide_page(job.plan, job.pair, index, geo, nogeo, params)
                source_doc = docs[decision.source]
                if params.text_layer:
                    if pdfs[decision.source] is None:
                        pdfs[decision.source] = pikepdf.open(
                            str(job.pair.geo if decision.source is PageSource.GEO else job.pair.nogeo)
                        )
                    options = Options(
                        allowed=tuple(params.allowed_rotations),
                        lang=params.lang,
                        free_text=params.free_text,
                        variant="fr_geo" if decision.source is PageSource.GEO else "fr_nogeo",
                        layout=(
                            SuryaSourceConfig(params.layout_cache_dir) if params.layout_cache_dir is not None else None
                        ),
                        known=_known_words(),
                    )
                    payload = process_page(source_doc, pdfs[decision.source], index, options).to_json()
                else:
                    payload = _empty_layer_payload(job.plan.full_pdf_name, index)
                payload.update(
                    {
                        "analysis_version": VERSION,
                        "source": decision.source.value,
                        "source_pdf": str(job.pair.geo if decision.source is PageSource.GEO else job.pair.nogeo),
                        "reason": decision.reason.value,
                        "geometry_verdict": decision.geometry_verdict,
                        "geometry_score": decision.geometry_score,
                        "geometry_reason": decision.geometry_reason,
                        "geometry_cached": decision.geometry_cached,
                        "pictures": len(job.plan.pages[index].pictures),
                    }
                )
                payload["seconds"] = round(time.time() - started, 2)
                save_page(target, payload)
                rows.append(PageAnalysis.from_payload(payload, False))
            except Exception as error:  # noqa: BLE001 — страница не должна валить пачку
                logger.exception("%s с.%d: анализ не удался", job.plan.full_pdf_name, index + 1)
                rows.append(
                    PageAnalysis(
                        job.plan.full_pdf_name,
                        index,
                        seconds=round(time.time() - started, 2),
                        error=f"{type(error).__name__}: {error}",
                    )
                )
        for pdf in pdfs.values():
            if pdf is not None:
                pdf.close()
    return rows


def _empty_layer_payload(pdf_name: str, index: int) -> dict:
    """JSON страницы без разбора слоя (режим «только выбор источника»)."""
    return {
        "version": TEXT_LAYER_VERSION,
        "pdf": pdf_name,
        "page": index,
        "zones": [],
        "words": [],
        "readings": {},
        "tables": [],
        "figures": [],
        "seconds": 0.0,
        "error": "",
    }


def chunk_jobs(plan: IssuePlan, pair: IssuePair, params: AnalysisParams, chunk: int = CHUNK_PAGES) -> list[AnalysisJob]:
    """Разбить выпуск на пачки страниц."""
    indices = list(range(pair.pages))
    return [AnalysisJob(plan, pair, tuple(indices[i : i + chunk]), params) for i in range(0, len(indices), chunk)]


def second_opinion_requests(
    params: AnalysisParams, plans: list[IssuePlan], pairs: dict[int, IssuePair]
) -> list[tuple[Path, Path]]:
    """Пары «PDF-источник → JSON» для стадии surya по всем страницам выпусков.

    Args:
        params: Параметры (каталог JSON).
        plans: Выпуски.
        pairs: Пары PDF по ``issue_id``.

    Returns:
        Список пар для :func:`ocr_utils.text_layer_fix.second_opinion.revise`.
    """
    requests: list[tuple[Path, Path]] = []
    for plan in plans:
        pair = pairs.get(plan.issue_id)
        if pair is None:
            continue
        for index in range(pair.pages):
            path = page_json_path(params, plan, index)
            payload = load_page(path)
            if payload is None or not payload.get("readings"):
                continue
            requests.append((Path(payload["source_pdf"]), path))
    return requests


__all__ = [
    "CHUNK_PAGES",
    "AnalysisJob",
    "AnalysisParams",
    "PageAnalysis",
    "analyse_chunk",
    "chunk_jobs",
    "decide_page",
    "load_analysis",
    "page_json_path",
    "second_opinion_requests",
]
