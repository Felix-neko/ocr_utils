"""Источники стенда: кандидаты line art разбора пака (итог, запись стадии кандидатов, ответы DeepSeek обоих проходов, вырезки)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.line_art.deepseek.rules import Crop
from ocr_utils.page_layout.pack_analysis.stages import read_jsonl_map


class PdfVariant(str, Enum):
    """Вариант разбора: страницы PDF FineReader без коррекции геометрии или с ней (подпапки разбора v6)."""

    NOGEO = "nogeo"
    GEO = "geo"


@dataclass(frozen=True)
class Candidate:
    """Кандидат line art с итогом нынешнего решения.

    Attributes:
        variant: Вариант разбора.
        id: Номер кандидата (``<полоса>_<номер>``).
        page_key: Имя файла полосы (``<pdf>_pNNNN``).
        page: Имя полосы в разборе (``<pdf>/pNNNN``).
        crop: Где вырезка на полосе.
        info: Сведения классического детектора (``kind``, ``sources``, ``area_px``).
        outcome: Итог нынешнего решения («объекты», «надпись», «неясно»).
        objects: Объекты итога ``{"class", "box", "box_source"}``.
        pass2_used: Объекты дал второй проход.
    """

    variant: PdfVariant
    id: str
    page_key: str
    page: str
    crop: Crop
    info: dict
    outcome: str
    objects: tuple[dict, ...]
    pass2_used: bool

    @property
    def classes(self) -> set[str]:
        """Классы объектов итога."""
        return {obj["class"] for obj in self.objects}


@dataclass
class DeepSeekOutput:
    """Ответы DeepSeek по всем кандидатам варианта: блоки ``markdown`` и слова ``ocr`` первого прохода, ``markdown`` второго."""

    pass1_markdown: dict[str, list]
    pass1_words: dict[str, list]
    pass2_markdown: dict[str, list]


def load_deepseek(root: Path) -> DeepSeekOutput:
    """Ответы DeepSeek варианта из ``work/deepseek/pass{1,2}/*.jsonl``."""
    work = root / "work" / "deepseek"
    return DeepSeekOutput(
        read_jsonl_map(work / "pass1" / "markdown.jsonl"),
        read_jsonl_map(work / "pass1" / "ocr.jsonl"),
        read_jsonl_map(work / "pass2" / "markdown.jsonl"),
    )


def load_candidates(root: Path, variant: PdfVariant) -> list[Candidate]:
    """Все кандидаты line art варианта: итог — из ``pages/*.json``, вырезка и сведения детектора — из ``work/pages``.

    Args:
        root: Корень варианта разбора (``.../pack1_page_analysis_v6_fr/<вариант>``).
        variant: Вариант.

    Returns:
        Кандидаты по порядку полос.
    """
    out = []
    for path in sorted((root / "pages").glob("*.json")):
        final = json.loads(path.read_text(encoding="utf-8"))
        if not final.get("candidates"):
            continue
        record = json.loads((root / "work" / "pages" / path.name).read_text(encoding="utf-8"))
        by_id = {c["id"]: c for c in record["candidates"]}
        for item in final["candidates"]:
            source = by_id.get(item["id"])
            if source is None:
                continue
            out.append(
                Candidate(
                    variant=variant,
                    id=item["id"],
                    page_key=path.stem,
                    page=final["page"],
                    crop=Crop.from_json(source["crop"]),
                    info=source.get("info") or {},
                    outcome=item.get("outcome", ""),
                    objects=tuple(item.get("objects", [])),
                    pass2_used=bool(item.get("pass2_used", False)),
                )
            )
    return out


def read_gray(root: Path, candidate_id: str, stage: str = "pass1") -> np.ndarray | None:
    """Вырезка кандидата из ``work/crops/<stage>`` (серая у первого прохода, залитая бинарная у второго) или ``None``."""
    path = root / "work" / "crops" / stage / f"{candidate_id}.png"
    if not path.is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


__all__ = ["Candidate", "DeepSeekOutput", "PdfVariant", "load_candidates", "load_deepseek", "read_gray"]
