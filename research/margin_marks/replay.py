"""Пересчёт решения line art по готовому выводу DeepSeek (без GPU): нынешний код против итога разбора и нынешний код с правилом «пометка».

Для каждой полосы с кандидатами: ``stages.decide_candidates`` по записи ``work/pages``, вырезкам и ответам DeepSeek
обоих проходов (страница рендерится из PDF только ради маски краски при достройке рамок); сверка исхода и классов с
итоговым JSON; затем признаки кандидата (``features.features_of``) и правило (``rule.MarkRule``): кандидат, которого
правило признало пометкой, получает исход «пометка» и теряет объекты.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocr_utils.page_layout.image import Variant
from ocr_utils.page_layout.pack_analysis.stages import PageTask, decide_candidates, load_image
from research.margin_marks.features import as_row, features_of
from research.margin_marks.rule import MarkRule
from research.margin_marks.sources import Candidate, PdfVariant, load_deepseek, read_gray

# Исход кандидата, признанного пометкой (в боевом коде — будущий ``Outcome.MARK``).
MARK = "пометка"
# Вариант разбора → вариант картинки ``page_layout``.
IMAGE_VARIANT = {PdfVariant.NOGEO: Variant.FR_NOGEO, PdfVariant.GEO: Variant.FR_GEO}

_DEEPSEEK: dict = {}


def _payload(deepseek, candidate_ids: list[str]) -> dict:
    """Ответы DeepSeek по кандидатам полосы в форме, которую ждёт ``decide_candidates``."""
    return {
        "pass1_markdown": {i: deepseek.pass1_markdown.get(i, []) for i in candidate_ids},
        "pass2_markdown": {i: deepseek.pass2_markdown[i] for i in candidate_ids if i in deepseek.pass2_markdown},
        # Слова первого прохода: боевое решение (с 2026-09-30) само проверяет «пометку» и по ним заливает вырезку заново.
        "pass1_ocr": {i: deepseek.pass1_words.get(i, []) for i in candidate_ids},
    }


def replay_page(root: Path, variant: PdfVariant, page_key: str, pdf_dir: Path, rule: MarkRule) -> list[dict]:
    """Все кандидаты полосы: итог разбора, пересчёт нынешним кодом, исход с правилом.

    Args:
        root: Корень варианта разбора.
        variant: Вариант.
        page_key: Имя файла полосы (``<pdf>_pNNNN``).
        pdf_dir: Каталог PDF варианта.
        rule: Правило «пометка».

    Returns:
        По кандидату словарь: ``variant, id, page, final_outcome, final_classes, replay_outcome, replay_classes,
        reproduced, new_outcome, new_classes, changed``.
    """
    if variant not in _DEEPSEEK:
        _DEEPSEEK[variant] = load_deepseek(root)
    deepseek = _DEEPSEEK[variant]
    final = json.loads((root / "pages" / f"{page_key}.json").read_text(encoding="utf-8"))
    record = json.loads((root / "work" / "pages" / f"{page_key}.json").read_text(encoding="utf-8"))
    pdf, _, index = final["page"].rpartition("/p")
    task = PageTask(final["page"], pdf_dir / f"{pdf}.pdf", int(index), IMAGE_VARIANT[variant])
    image = load_image(task, record.get("rotate_cw", 0))
    ids = [c["id"] for c in record["candidates"]]
    decisions = decide_candidates(record, image, _payload(deepseek, ids), root / "work")
    finals = {c["id"]: c for c in final.get("candidates", [])}
    out = []
    for candidate, decision in decisions:
        replay_classes = "|".join(sorted({o["class"] for o in decision.objects}))
        old = finals.get(candidate["id"], {})
        old_classes = "|".join(sorted({o["class"] for o in old.get("objects", [])}))
        row = {
            "variant": variant.value,
            "id": candidate["id"],
            "page": final["page"],
            "final_outcome": old.get("outcome", ""),
            "final_classes": old_classes,
            "replay_outcome": decision.outcome.value,
            "replay_classes": replay_classes,
            "reproduced": old.get("outcome", "") == decision.outcome.value and old_classes == replay_classes,
            "new_outcome": decision.outcome.value,
            "new_classes": replay_classes,
        }
        # Правило смотрит только рисунки и «неясно» (остальное оно не трогает по условию исхода).
        if replay_classes == "рисунок" or decision.outcome.value == "неясно":
            gray = read_gray(root, candidate["id"], "pass1")
            if gray is not None:
                from ocr_utils.page_layout.line_art.deepseek.rules import Crop

                probe = Candidate(variant, candidate["id"], page_key, final["page"], Crop.from_json(candidate["crop"]),
                                  candidate.get("info") or {}, decision.outcome.value, tuple(decision.objects),
                                  decision.pass2_used)  # fmt: skip
                features = as_row(features_of(probe, gray, read_gray(root, candidate["id"], "pass2"), deepseek))
                if rule.is_mark(features):
                    row["new_outcome"], row["new_classes"] = MARK, ""
        row["changed"] = (row["new_outcome"], row["new_classes"]) != (row["replay_outcome"], row["replay_classes"])
        out.append(row)
    return out


__all__ = ["IMAGE_VARIANT", "MARK", "replay_page"]
