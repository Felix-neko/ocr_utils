"""Идемпотентная запись дерева пака и разметки в базу.

Ключевое требование: повторный прогон ``detect`` по тому же паку не должен ни плодить
дубликаты, ни стирать уже накопленное. Поэтому пак, год, выпуск и полоса ищутся по своим
уникальным ключам и при совпадении переиспользуются, а разметка полосы заменяется целиком
и только той полосы, которую действительно пересчитали.
"""

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ocr_utils.scan_markup.db.models import Issue, MaskAnnotation, Pack, Page, RectRegion, YearPackage
from ocr_utils.scan_markup.scan_tree import ScannedYear

logger = logging.getLogger(__name__)


def get_pack(session: Session, name: str) -> Pack | None:
    """Пак по имени или ``None``."""
    return session.scalars(select(Pack).where(Pack.name == name)).one_or_none()


def require_pack(session: Session, name: str) -> Pack:
    """Пак по имени; если его нет — понятная ошибка вместо ``NoneType`` где-то ниже."""
    pack = get_pack(session, name)
    if pack is None:
        known = ", ".join(sorted(p.name for p in session.scalars(select(Pack)))) or "(база пуста)"
        raise LookupError(f"в базе нет пака {name!r}; есть: {known}")
    return pack


def upsert_pack(session: Session, name: str, root: Path, years: list[ScannedYear]) -> Pack:
    """Заводит или дополняет пак по дереву из :func:`scan_tree.scan_pack`.

    Возвращает объект пака. Существующие годы, выпуски и полосы переиспользуются по
    уникальным ключам, недостающие добавляются. Полосы, ИСЧЕЗНУВШИЕ с диска, не удаляются:
    прогон по подмножеству (``--only-year``) не должен выглядеть как удаление остального.

    Новые объекты подвешиваются через КОЛЛЕКЦИИ СВЯЗЕЙ, а не проставлением внешнего ключа.
    Разница не косметическая: при записи ``year_package_id=...`` объект попадает в базу, но
    в уже загруженной коллекции ``pack.year_packages`` его нет, и вызывающий, который сразу
    после upsert идёт по ``iter_pages(pack)``, получает пустой обход и молча обрабатывает
    ноль полос.
    """
    pack = get_pack(session, name)
    if pack is None:
        pack = Pack(name=name, source_pics_root=str(root))
        session.add(pack)
        session.flush()
    else:
        pack.source_pics_root = str(root)

    existing_years = {year.name: year for year in pack.year_packages}
    for scanned_year in years:
        year = existing_years.get(scanned_year.name)
        if year is None:
            year = YearPackage(name=scanned_year.name, year=scanned_year.year, rel_path=scanned_year.rel_path)
            pack.year_packages.append(year)
            existing_years[year.name] = year
            session.flush()

        existing_issues = {issue.name: issue for issue in year.issues}
        for scanned_issue in scanned_year.issues:
            issue = existing_issues.get(scanned_issue.name)
            if issue is None:
                issue = Issue(name=scanned_issue.name, number=scanned_issue.number, rel_path=scanned_issue.rel_path)
                year.issues.append(issue)
                existing_issues[issue.name] = issue
                session.flush()

            existing_pages = {page.source_file_name: page for page in issue.pages}
            for scanned_page in scanned_issue.pages:
                page = existing_pages.get(scanned_page.file_name)
                if page is None:
                    issue.pages.append(
                        Page(
                            source_file_name=scanned_page.file_name,
                            source_rel_path=scanned_page.rel_path,
                            order_index=scanned_page.order_index,
                        )
                    )
                else:
                    # Порядок мог поехать, если в выпуск досыпали пересканов.
                    page.order_index = scanned_page.order_index
                    page.source_rel_path = scanned_page.rel_path
            session.flush()

    session.commit()
    return pack


def replace_rect_regions(
    session: Session, page: Page, regions: list[RectRegion], kinds: "tuple[str, ...] | None" = None
) -> None:
    """Заменяет прямоугольные области полосы: все либо только заданных видов.

    Именно замена, а не дополнение: повторная детекция — это новый ответ на тот же
    вопрос, а не добавка к старому, и накапливать оба варианта означало бы отдать
    разметчику вдвое больше прямоугольников.

    ``kinds`` ограничивает замену семейством видов: растровый детектор заменяет только
    растр (``RASTER_KINDS``), детектор таблиц — только таблицы и схемы (``TABLE_KINDS``).
    Так таблицы можно дописать в базу, где растр уже уточнён человеком, не тронув его.
    Область другого вида среди ``regions`` — ошибка вызывающего, а не тихая потеря.

    Работаем через КОЛЛЕКЦИЮ связи: ``cascade="all, delete-orphan"`` удалит выпавшие из
    неё строки сам. Через ``session.delete`` + ``session.add`` с ручным ``page_id`` было бы
    хуже — коллекция ``page.rect_regions`` осталась бы со старым содержимым, и следующий
    вызов удалил бы не то, что нужно.
    """
    if kinds is None:
        page.rect_regions = regions
    else:
        foreign = [region.kind for region in regions if region.kind not in kinds]
        if foreign:
            raise ValueError(f"области видов {sorted(set(foreign))} не входят в заменяемые {kinds}")
        page.rect_regions = [region for region in page.rect_regions if region.kind not in kinds] + list(regions)
    session.flush()


def replace_masks(session: Session, page: Page, masks: list[MaskAnnotation]) -> None:
    """Заменяет маски полосы целиком; мотивировка та же, что у :func:`replace_rect_regions`."""
    page.masks = masks
    session.flush()


def iter_pages(pack: Pack, only_year: str | None = None, only_issue: str | None = None):
    """Полосы пака в порядке год -> выпуск -> номер полосы, с необязательной фильтрацией."""
    for year in pack.year_packages:
        if only_year is not None and year.name != only_year:
            continue
        for issue in year.issues:
            if only_issue is not None and issue.name != only_issue:
                continue
            for page in issue.pages:
                yield year, issue, page
