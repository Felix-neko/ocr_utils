"""Выпрямители таблицы: реестр и порядок отката.

ПОРЯДОК ОТКАТА подчинён вёрстке, а она разная: у таблицы может не быть внешней рамки, может
не быть части внутренних линеек, а может не быть линеек вовсе. Поэтому берётся первый
способ, которому хватило опор:

    полная решётка → тонкая пластина по пересечениям;
    есть линейки хотя бы одной оси → раздельные поля смещений;
    линеек нет, но есть текст → готовый движок по строкам;
    ничего нет → поворот, а если и угол не измерить, оставить как было.
"""

from __future__ import annotations

from research.legacy.table_processing.warping.engines.base import Warped, Warper  # noqa: F401

REGISTRY_ORDER = ("deskew", "perspective", "rules_separable", "rules_tps", "textline")

# Набор по умолчанию для сравнения. Победитель ставится способом по умолчанию по замеру,
# а не по вкусу — см. README пакета.
DEFAULT_SET = ("deskew", "perspective", "rules_separable", "rules_tps", "textline")


def load(name: str) -> Warper:
    if name == "deskew":
        from research.legacy.table_processing.warping.engines.deskew import ALGORITHM
    elif name == "perspective":
        from research.legacy.table_processing.warping.engines.perspective import ALGORITHM
    elif name == "rules_separable":
        from research.legacy.table_processing.warping.engines.rules_separable import ALGORITHM
    elif name == "rules_tps":
        from research.legacy.table_processing.warping.engines.rules_tps import ALGORITHM
    elif name == "textline":
        from research.legacy.table_processing.warping.engines.textline_engine import ALGORITHM
    else:
        raise KeyError(f"нет выпрямителя {name!r}; есть {', '.join(REGISTRY_ORDER)}")
    return ALGORITHM


def resolve(names: "tuple[str, ...] | None") -> list[Warper]:
    chosen = names or DEFAULT_SET
    return [warper for warper in (load(name) for name in chosen) if warper.available()]
