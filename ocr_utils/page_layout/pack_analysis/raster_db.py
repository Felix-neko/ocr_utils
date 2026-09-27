"""Растр полос из базы разметки после ревью (вместо детектора растра) и сверка его с растром прошлого прогона.

Растровые области пака выверены человеком в CVAT и лежат в ``rect_regions`` базы ``DB_REVIEWED``. Когда они
уже есть, детектор растра при разборе пака не нужен: его находки хуже выверенных. Здесь — чтение этих
областей в формате записи полосы разбора (``stages._region_json``) и сравнение с растром прошлого прогона:
по нему решается, на каких полосах пересчитывать line art (растр — его запретная зона).
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from ocr_utils.db.models import PICTURE_KINDS
from ocr_utils.scan_markup.rotation import rotate_box


# Метка источника в ``info`` области, взятой из базы.
DB_SOURCE = "db"


def load_raster(db_path: Path, pack_name: str) -> dict[str, list[dict]]:
    """Растровые области всех полос пака из базы разметки, в кадре заострённой копии.

    Берутся только картинки (``PICTURE_KINDS``: цветной растр, серый растр, цветной текст); печати
    (``stamp_suspect``, маски ``library_stamp``) не берутся — решение пользователя для разбора v3.
    Координаты в базе — пиксели исходного TIFF ДО поворота; заострённые копии лежат уже повёрнутыми
    (``pages.rotate_cw``), поэтому рамка поворачивается тем же углом.

    Args:
        db_path: База разметки (открывается только на чтение).
        pack_name: Имя пака в таблице ``packs`` (``пак-1``).

    Returns:
        ``{«год/выпуск/полоса»: [{"kind", "box", "confidence", "info"}]}`` — как ``raster`` в записи полосы
        стадии кандидатов; полосы без растра в словарь не попадают.
    """
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    marks = ",".join("?" for _ in PICTURE_KINDS)
    rows = connection.execute(
        f"""
        select p.source_rel_path, p.width, p.height, coalesce(p.rotate_cw, 0), r.kind, r.x1, r.y1, r.x2, r.y2
        from rect_regions r
        join pages p on p.id = r.page_id
        join issues i on i.id = p.issue_id
        join year_packages y on y.id = i.year_package_id
        join packs k on k.id = y.pack_id
        where k.name = ? and r.kind in ({marks})
        order by p.source_rel_path, r.id
        """,
        (pack_name, *PICTURE_KINDS),
    ).fetchall()
    connection.close()
    raster: dict[str, list[dict]] = defaultdict(list)
    for rel_path, width, height, rotate, kind, x1, y1, x2, y2 in rows:
        box = (int(x1), int(y1), int(x2), int(y2))
        if rotate:
            # Рамка из кадра оригинала — в кадр повёрнутой заострённой копии.
            box = rotate_box(box, int(width), int(height), int(rotate))
        name = str(Path(rel_path).with_suffix(""))
        raster[name].append({"kind": kind, "box": list(box), "confidence": None, "info": {"source": DB_SOURCE}})
    return dict(raster)


def same_raster(old: list[dict], new: list[dict]) -> bool:
    """Совпадает ли растр полосы в двух разборах: те же пары «вид, рамка» столько же раз, без допусков.

    Порядок областей не важен. Любое отличие — другое число рамок, смена вида (цветной ↔ серый), сдвиг
    рамки хоть на пиксель, лишняя печать в старом разборе — значит «растр изменился» (решение
    пользователя: пересчитывать line art при любом отличии).

    Args:
        old: Области ``raster`` из записи полосы прошлого прогона.
        new: Области той же полосы из базы (:func:`load_raster`).

    Returns:
        ``True``, если мультимножества пар совпадают.
    """
    return _pairs(old) == _pairs(new)


def _pairs(regions: list[dict]) -> Counter:
    """Мультимножество пар (вид, рамка) — для сравнения без учёта порядка."""
    return Counter((region["kind"], tuple(int(v) for v in region["box"])) for region in regions)


__all__ = ["DB_SOURCE", "load_raster", "same_raster"]
