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

from ocr_utils.scan_markup.db.models import Issue, MaskAnnotation, Pack, Page, RasterRegion, YearPackage
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
        pack = Pack(name=name, root_path=str(root))
        session.add(pack)
        session.flush()
    else:
        pack.root_path = str(root)

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

            existing_pages = {page.file_name: page for page in issue.pages}
            for scanned_page in scanned_issue.pages:
                page = existing_pages.get(scanned_page.file_name)
                if page is None:
                    issue.pages.append(
                        Page(
                            file_name=scanned_page.file_name,
                            rel_path=scanned_page.rel_path,
                            order_index=scanned_page.order_index,
                        )
                    )
                else:
                    # Порядок мог поехать, если в выпуск досыпали пересканов.
                    page.order_index = scanned_page.order_index
                    page.rel_path = scanned_page.rel_path
            session.flush()

    session.commit()
    return pack


def replace_raster_regions(session: Session, page: Page, regions: list[RasterRegion]) -> None:
    """Заменяет растровые области полосы целиком.

    Именно замена, а не дополнение: повторная детекция — это новый ответ на тот же
    вопрос, а не добавка к старому, и накапливать оба варианта означало бы отдать
    разметчику вдвое больше прямоугольников.

    Работаем через КОЛЛЕКЦИЮ связи: ``cascade="all, delete-orphan"`` удалит выпавшие из
    неё строки сам. Через ``session.delete`` + ``session.add`` с ручным ``page_id`` было бы
    хуже — коллекция ``page.raster_regions`` осталась бы со старым содержимым, и следующий
    вызов удалил бы не то, что нужно.
    """
    page.raster_regions = regions
    session.flush()


def replace_masks(session: Session, page: Page, masks: list[MaskAnnotation]) -> None:
    """Заменяет маски полосы целиком; мотивировка та же, что у :func:`replace_raster_regions`."""
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
