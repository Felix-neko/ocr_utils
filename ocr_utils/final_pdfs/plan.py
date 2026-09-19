"""План сборки: выпуски и полосы из базы (только чтение), источник страницы и решение по ней.

База читается явным ``select`` нужных колонок, а не целыми ORM-объектами: сборщику в базу
писать нельзя (``open_db(create=False)``), а модель ``Page`` может знать колонки, которых в
рабочей базе ещё нет (``force_is_not_toc``) — целый ``Page`` на такой базе не загрузится.
Классы плана — те же, что у сборщика промежуточных PDF, чтобы поворот и рамки считались одним
кодом.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sqlalchemy import select

from ocr_utils.db.models import PICTURE_KINDS, Issue, Page, RectRegion, YearPackage
from ocr_utils.db.repo import require_pack
from ocr_utils.db.session import open_db
from ocr_utils.pdf_utils.intermediate_pdfs import IssuePlan, PagePlan, PicturePlan

__all__ = ["IssuePlan", "PagePlan", "PageSource", "PageDecision", "SourceReason", "decide_source", "load_plans"]


def load_plans(
    db_path: Path, pack_name: str, *, only_year: str | None = None, only_issue: str | None = None
) -> list[IssuePlan]:
    """Выпуски пака с полосами (по ``order_index``) и рамками иллюстраций — одним чтением базы.

    Args:
        db_path: База разметки после экспорта из CVAT (``pack1_reviewed.sqlite``).
        pack_name: Имя пака (``packs.name``).
        only_year: Оставить только этот год (имя годового комплекта).
        only_issue: Оставить только этот выпуск (имя выпуска, с ведущими нулями).

    Returns:
        Планы выпусков; ``PagePlan.full_pdf_page_idx`` — позиция полосы в выпуске, то есть номер
        страницы в обеих PDF FineReader.

    Raises:
        ValueError: У полосы нет размеров кадра (не было ``detect``).
    """
    factory = open_db(db_path, create=False)
    with factory() as session:
        pack = require_pack(session, pack_name)
        issues = session.execute(
            select(YearPackage.name, Issue.id, Issue.name)
            .join(Issue, Issue.year_package_id == YearPackage.id)
            .where(YearPackage.pack_id == pack.id)
            .order_by(YearPackage.name, Issue.name)
        ).all()
        plans: list[IssuePlan] = []
        for year_name, issue_id, issue_name in issues:
            if only_year is not None and year_name != only_year:
                continue
            if only_issue is not None and issue_name != only_issue:
                continue
            page_rows = session.execute(
                select(
                    Page.id,
                    Page.source_rel_path,
                    Page.cleaned_rel_path,
                    Page.sharpened_text_pic_rel_path,
                    Page.width,
                    Page.height,
                    Page.dpi,
                    Page.rotate_cw,
                )
                .where(Page.issue_id == issue_id)
                .order_by(Page.order_index, Page.id)
            ).all()
            if not page_rows:
                continue
            region_rows = session.execute(
                select(RectRegion.page_id, RectRegion.x1, RectRegion.y1, RectRegion.x2, RectRegion.y2, RectRegion.kind)
                .join(Page, RectRegion.page_id == Page.id)
                .where(Page.issue_id == issue_id, RectRegion.kind.in_(PICTURE_KINDS))
                .order_by(RectRegion.page_id, RectRegion.id)
            ).all()
            pictures_by_page: dict[int, list[PicturePlan]] = {}
            for page_id, x1, y1, x2, y2, kind in region_rows:
                pictures_by_page.setdefault(page_id, []).append(PicturePlan(int(x1), int(y1), int(x2), int(y2), kind))
            pages: list[PagePlan] = []
            for index, (page_id, source_rel, cleaned_rel, sharpened_rel, width, height, dpi, rotate) in enumerate(
                page_rows
            ):
                if not width or not height:
                    raise ValueError(f"{source_rel}: нет размеров кадра, сначала нужен detect")
                pictures = tuple(pictures_by_page.get(page_id, ()))
                if sharpened_rel is None:
                    # Заострённая копия отличается от очищенной полосы только расширением
                    # (collect_sharpened без --db не записал путь в базу).
                    sharpened_rel = Path(cleaned_rel or source_rel).with_suffix(".jpg").as_posix()
                pages.append(
                    PagePlan(
                        page_id=int(page_id),
                        original_rel_path=cleaned_rel or source_rel,
                        sharpened_rel_path=sharpened_rel,
                        width=int(width),
                        height=int(height),
                        dpi=int(dpi or 600),
                        pictures=pictures,
                        full_pdf_page_idx=index,
                        pages_with_pics_only_pdf_page_idx=None,
                        rotate_cw=int(rotate or 0),
                    )
                )
            plans.append(
                IssuePlan(issue_id=int(issue_id), year_name=year_name, issue_name=issue_name, pages=tuple(pages))
            )
    return plans


class PageSource(str, Enum):
    """Из какого прогона FineReader берётся страница."""

    GEO = "geo"  # с коррекцией геометрии
    NOGEO = "nogeo"  # без коррекции


class SourceReason(str, Enum):
    """Почему выбран источник."""

    PICTURES = "растр в базе"  # иллюстрации кладутся только на геометрию скана
    GEOMETRY_BAD = "коррекция испортила геометрию"
    GEOMETRY_OK = "коррекция не навредила"


@dataclass(frozen=True)
class PageDecision:
    """Решение по странице: источник, причина и вердикт детектора геометрии (если спрашивали)."""

    source: PageSource
    reason: SourceReason
    geometry_verdict: str = ""  # bad | mixed | ok | "" (детектор не вызывался)
    geometry_score: float = 0.0
    geometry_reason: str = ""
    geometry_cached: bool = True


def decide_source(has_pictures: bool, geometry_verdict: str | None) -> tuple[PageSource, SourceReason]:
    """Правило выбора источника страницы.

    Args:
        has_pictures: Есть ли на полосе размеченный растр (иллюстрации, цветной текст).
        geometry_verdict: Вердикт детектора ``bad``/``mixed``/``ok``; ``None`` — не спрашивали.

    Returns:
        Источник и причина.
    """
    if has_pictures:
        return PageSource.NOGEO, SourceReason.PICTURES
    if geometry_verdict == "bad":
        return PageSource.NOGEO, SourceReason.GEOMETRY_BAD
    return PageSource.GEO, SourceReason.GEOMETRY_OK


def final_pdf_name(plan: IssuePlan) -> str:
    """Имя финального PDF выпуска: ``{год}_{выпуск}.pdf``."""
    return plan.full_pdf_name.removeprefix("full_")


def final_pdf_path(out_dir: Path, plan: IssuePlan) -> Path:
    """Путь финального PDF выпуска в выходной папке (все выпуски в одной папке)."""
    return out_dir / final_pdf_name(plan)
