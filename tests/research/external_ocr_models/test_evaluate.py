"""Метрики: нормализация, CER, структура, зацикливание, поиск фраз, оценка выходов."""

import json
from pathlib import Path

from research.external_ocr_models import evaluate, report


def test_normalize_strips_markup_and_joins_hyphens():
    md = '---\npage_number: "1"\n---\n# Заголовок\n\n**Автор**\n\nснабже-\nния текст ­мягкий\n\n| а | б |\n|---|---|\n| 1 | 2 |\n\n> [картинка: портрет]\n<table><tr><td>x</td></tr></table>'
    assert evaluate.normalize(md) == "заголовок автор снабжения текст мягкий а б 1 2 x"


def test_cer():
    assert evaluate.cer("абв", "абв") == 0.0
    assert evaluate.cer("абг", "абв") == 1 / 3
    assert evaluate.cer("x", "") is None


def test_structure_counts():
    md = "### Рубрика\n# Заголовок\n## Под\n**И. Иванов,**\n*начальник*\n- пункт\n\n| а |\n|---|\n\n<table></table>\n> [блок-схема]\n> [картинка: х]\n[^1]: сноска\n[неразборчиво]"
    counts = evaluate.structure(md)
    assert (counts.h1, counts.h2, counts.h3, counts.authors, counts.positions) == (1, 1, 1, 1, 1)
    assert (
        counts.tables,
        counts.html_tables,
        counts.schemas,
        counts.pictures,
        counts.footnotes,
        counts.unreadable,
        counts.lists,
    ) == (2, 1, 1, 1, 1, 1, 1)


def test_repetition_and_phrases():
    assert evaluate.repetition_score(". " * 2000) > 0.9
    assert evaluate.repetition_score("разные слова " * 3) == 0.0
    found, total = evaluate.phrase_recall(
        "Шапка: Железнодорожный тариф, руб. и прочее", ["Железнодорожный тариф, руб.", "нет такого"]
    )
    assert (found, total) == (1, 2)


def _write_output(root: Path, model: str, page: str, body: str | None, **meta):
    base = root / model / "1966/03" / page
    base.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "page": f"1966/03/{page}.jpg",
        "model": model,
        "cost_usd": 0.001,
        "latency_s": 2.0,
        "error": None,
        "parse_error": None,
    }
    data.update(meta)
    base.with_suffix(".meta.json").write_text(json.dumps(data), encoding="utf-8")
    if body is not None:
        base.with_suffix(".json").write_text(
            json.dumps({"content_markdown": body, "page_number": meta.get("page_number"), "is_toc": False}),
            encoding="utf-8",
        )


def test_score_outputs_and_report(tmp_path):
    _write_output(tmp_path, "a", "IMG_0105_2R", "# Заголовок\n\nтекст один", page_number="1")
    _write_output(tmp_path, "b", "IMG_0105_2R", "# Заголовок\n\nтекст одна", page_number="2")
    _write_output(tmp_path, "b", "IMG_0106_1L", None, error="HTTP 500")
    _write_output(tmp_path, "c", "IMG_0105_2R", "")
    by_model = {name: evaluate.load_outputs(tmp_path / name) for name in ("a", "b", "c")}
    scores = evaluate.score_outputs(by_model, {"IMG_0105_2R": "заголовок текст один"}, {}, {"IMG_0105_2R": "1"})
    by_key = {(score.model, Path(score.page).stem): score for score in scores}
    assert by_key[("a", "IMG_0105_2R")].cer_finereader == 0.0 and by_key[("a", "IMG_0105_2R")].page_number_ok is True
    assert by_key[("b", "IMG_0105_2R")].cer_finereader > 0 and by_key[("b", "IMG_0105_2R")].page_number_ok is False
    assert by_key[("a", "IMG_0105_2R")].agreement == by_key[("b", "IMG_0105_2R")].agreement > 0
    assert not by_key[("b", "IMG_0106_1L")].ok and "500" in by_key[("b", "IMG_0106_1L")].error
    assert not by_key[("c", "IMG_0105_2R")].ok and by_key[("c", "IMG_0105_2R")].error == "пустой ответ"
    text = report.build_report(scores, "тест")
    assert "| a | 1 | 0 |" in text and "| b | 2 | 1 |" in text and "**2** (ожид. 1)" in text
    report.write_scores_csv(scores, tmp_path / "scores.csv")
    assert (tmp_path / "scores.csv").read_text(encoding="utf-8").count("\n") == 5
