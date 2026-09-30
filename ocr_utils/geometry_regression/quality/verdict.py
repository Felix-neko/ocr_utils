"""Вердикт детектора порчи геометрии v18 по странице для сборки финальных PDF: из кэша прогона, при промахе — мера на месте."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import fitz

from ocr_utils.geometry_regression.metrics import Params
from ocr_utils.geometry_regression.quality.measure import load_cached, measure_cached
from ocr_utils.geometry_regression.quality.scoring import Assessment, Thresholds17
from ocr_utils.geometry_regression.quality.sources import PageRef
from ocr_utils.geometry_regression.v16 import page as v16_page


@dataclass(frozen=True)
class QualityVerdict:
    """Вердикт страницы и откуда он взялся.

    Attributes:
        assessment: Вердикт, правило, виновник, баллы групп и выигрыш (:class:`scoring.Assessment`).
        metrics: Плоский словарь метрик страницы (v16 + v17–v18).
        cached: ``True`` — метрики прочитаны из кэша прогона, ``False`` — измерены сейчас.
        seconds: Время измерения (0 при попадании в кэш).
    """

    assessment: Assessment
    metrics: dict[str, float]
    cached: bool
    seconds: float

    @property
    def reason(self) -> str:
        """Правило и виновник одной строкой (для отчёта сборки)."""
        rule = self.assessment.rule.value
        return f"{rule}: {self.assessment.culprit}" if self.assessment.culprit else rule


def ensure_v16(
    v16_dir: Path, pdf_stem: str, page: int, geo_doc: fitz.Document, nogeo_doc: fitz.Document, params: Params
) -> None:
    """Кэш движка v16 для страницы: если его нет — измерить пару страниц и записать.

    Args:
        v16_dir: Каталог прогона v16 (``cache/``).
        pdf_stem: Имя PDF без расширения.
        page: Номер страницы, с единицы.
        geo_doc: Открытый PDF с коррекцией.
        nogeo_doc: Открытый PDF без коррекции.
        params: Параметры измерения v16 (кэш surya и размеры).
    """
    path = v16_page.cache_path(v16_dir, pdf_stem, page)
    if v16_page.load_cache(path) is None:
        v16_page.save_cache(path, v16_page.measure_page(geo_doc, nogeo_doc, page, params))


def verdict_for_page(
    run_dir: Path,
    v16_dir: Path,
    layout_root: Path,
    pdf_stem: str,
    page: int,
    geo_pdf: Path,
    nogeo_pdf: Path,
    geo_doc: fitz.Document,
    nogeo_doc: fitz.Document,
    thresholds: Thresholds17 | None = None,
    params: Params | None = None,
) -> QualityVerdict:
    """Вердикт «испортил ли FineReader геометрию страницы» по мерам v18.

    Порядок: JSON страницы версии v18 в ``run_dir`` → вердикт по порогам. При промахе — кэш v16 (при его промахе
    v16 меряет пару страниц сейчас, около 6 с), разбор ``page_layout`` обоих вариантов из ``layout_root`` (его
    здесь не строят: это прогон на GPU, ``run_scripts/page_layout/run_pack1_analysis_v6_fr.sh``), меры v17–v18 и
    запись JSON, чтобы следующий запуск его нашёл.

    Args:
        run_dir: Каталог прогона v18 (``cache/``).
        v16_dir: Каталог прогона v16 (``cache/``).
        layout_root: Корень разбора ``page_layout`` обоих вариантов (``geo/pages``, ``nogeo/pages``).
        pdf_stem: Имя PDF без расширения (одинаковое у обоих вариантов).
        page: Номер страницы, с единицы.
        geo_pdf: Путь PDF с коррекцией (для рендера мер по плотному полю).
        nogeo_pdf: Путь PDF без коррекции.
        geo_doc: Открытый PDF с коррекцией (для меры v16 при промахе).
        nogeo_doc: Открытый PDF без коррекции.
        thresholds: Пороги вердикта (по умолчанию — из кода).
        params: Параметры меры v16 (по умолчанию — для пака-1).

    Returns:
        :class:`QualityVerdict`.

    Raises:
        RuntimeError: Страница не измерена — нет разбора одного из вариантов или ошибка меры (текст ошибки).
    """
    thresholds = thresholds or Thresholds17()
    ref = PageRef(pdf_stem, page)
    cached = load_cached(run_dir, ref)
    if cached is not None:
        return QualityVerdict(thresholds.assess(cached["metrics"]), cached["metrics"], True, 0.0)
    started = time.time()
    ensure_v16(v16_dir, pdf_stem, page, geo_doc, nogeo_doc, params or Params())
    payload = measure_cached(ref, layout_root, v16_dir, run_dir, pdf_dirs=(geo_pdf.parent, nogeo_pdf.parent))
    if "error" in payload:
        raise RuntimeError(f"детектор геометрии v18, {ref.label}: {payload['error']}")
    seconds = round(time.time() - started, 2)
    return QualityVerdict(thresholds.assess(payload["metrics"]), payload["metrics"], False, seconds)


__all__ = ["QualityVerdict", "ensure_v16", "verdict_for_page"]
