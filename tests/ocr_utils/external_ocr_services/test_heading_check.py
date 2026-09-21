"""Вспомогательный текстовый запрос по статьям без заголовка: строки-кандидаты, разбор ответа, применение на сборке."""

from __future__ import annotations

import json
from pathlib import Path

from ocr_utils.external_ocr_services.assemble import assemble_issue, issue_pages
from ocr_utils.external_ocr_services.client import ChatResponse
from ocr_utils.external_ocr_services.heading_check import (
    CandidateLine,
    HeadingChecker,
    HeadingQuery,
    HintStatus,
    parse_verdicts,
)
from ocr_utils.external_ocr_services.models import resolve
from ocr_utils.external_ocr_services.reconcile import candidate_lines, PageRefs, skip_reason
from ocr_utils.external_ocr_services.schema import TocArticle
from tests.ocr_utils.external_ocr_services.test_reconcile import ISSUE, _page, _toc


class FakeClient:
    """Отвечает заданным текстом; помнит запросы."""

    def __init__(self, answer: str):
        self.answer = answer
        self.payloads: list[dict] = []

    def chat(self, payload):
        self.payloads.append(payload)
        return ChatResponse(self.answer, "stop", "Fake", "fake/model", "gen-1", 800, 40, 0, 0.0002, 1.0, 1, {})


def _query(*lines: tuple[int, str, bool]) -> HeadingQuery:
    return HeadingQuery(
        "A1",
        "На главном направлении",
        ["И. Иванов"],
        "3",
        [CandidateLine(n, 10 + n, "1974/05/IMG_0003", 3, text, in_window) for n, text, in_window in lines],
    )


def test_payload_is_text_only_and_verdicts_are_filtered(tmp_path):
    query = _query((1, "НА ОДНОМ ИЗ ГЛАВНЫХ НАПРАВЛЕНИЙ", True), (2, "АСУ МТС", True), (3, "Чужая полоса", False))
    toc = [{"id": "A1", "title": "На главном направлении", "authors": ["И. Иванов"], "page": "3"}]
    checker = HeadingChecker(FakeClient(""), resolve("deepseek-v41-flash"), tmp_path / "cache")
    payload = checker.payload(toc, [query])
    assert isinstance(payload["messages"][1]["content"], str), "картинок нет — текст одной строкой"
    assert "1. [p. 3] НА ОДНОМ ИЗ ГЛАВНЫХ НАПРАВЛЕНИЙ" in payload["messages"][1]["content"]
    assert payload["response_format"] == {"type": "json_object"} and payload["max_tokens"] == 200
    accepted = checker._finish(
        [query], '{"articles": [{"article_id": "A1", "line": 1, "confidence": "high"}]}', 0.0002, "e", False
    )
    assert accepted["A1"].status is HintStatus.ACCEPTED and accepted["A1"].block == 11
    low = checker._finish(
        [query], '{"articles": [{"article_id": "A1", "line": 1, "confidence": "low"}]}', 0.0002, "e", False
    )
    assert low["A1"].status is HintStatus.REJECTED and low["A1"].block is None
    outside = checker._finish(
        [query], '{"articles": [{"article_id": "A1", "line": 3, "confidence": "high"}]}', 0.0, "e", True
    )
    assert outside["A1"].status is HintStatus.REJECTED, "строка вне окна статьи"
    none = checker._finish(
        [query], '{"articles": [{"article_id": "A1", "line": null, "confidence": "low"}]}', 0.0, "e", True
    )
    assert none["A1"].status is HintStatus.NONE
    broken = checker._finish([query], "не JSON", 0.0, "e", False)
    assert broken["A1"].status is HintStatus.ERROR and checker.stats.errors == 1
    assert parse_verdicts('```json\n{"articles": [{"article_id": "A2", "line": 2}]}\n```') == {
        "A2": {"article_id": "A2", "line": 2}
    }


def test_check_issue_uses_cache_and_batches(tmp_path):
    query = _query((1, "НА ОДНОМ ИЗ ГЛАВНЫХ НАПРАВЛЕНИЙ", True))
    toc = [{"id": "A1", "title": "На главном направлении", "authors": [], "page": "3"}]
    client = FakeClient('{"articles": [{"article_id": "A1", "line": 1, "confidence": "high"}]}')
    checker = HeadingChecker(client, resolve("deepseek-v41-flash"), tmp_path / "cache")
    first = checker.check_issue("1974/05", toc, [query, HeadingQuery("A2", "Без строк", [], None, [])])
    assert set(first) == {"A1"} and first["A1"].status is HintStatus.ACCEPTED and not first["A1"].cache_hit
    entries = list((tmp_path / "cache" / "1974" / "05" / "_headings").iterdir())
    assert len(entries) == 1 and entries[0].name.startswith("heading.json_object.")
    again = checker.check_issue("1974/05", toc, [query])
    assert again["A1"].cache_hit and len(client.payloads) == 1 and checker.stats.cache_hits == 1


def test_candidate_lines_and_skip_reason():
    blocks = [
        "# Готовый",
        "## Подзаголовок",
        "**Жирный**",
        "Первый абзац.",
        "Второй абзац.",
        "Третий абзац.",
        "<marker>*РАЗДЕЛ*</marker>",
        "## Далеко",
    ]
    plain = [False, False, False, True, True, True, False, False]
    floating = [False, False, False, False, False, False, True, False]
    page_of = [0, 0, 0, 0, 0, 0, 0, 1]
    pages = [PageRefs("p3", number=3), PageRefs("p9", number=9)]
    lines = candidate_lines(blocks, floating, plain, page_of, pages, 3, 20)
    assert [(b, t) for b, _, t in lines] == [
        (1, "Подзаголовок"),
        (2, "Жирный"),
        (3, "Первый абзац."),
        (4, "Второй абзац."),
        (6, "РАЗДЕЛ"),
    ], "готовый `#`, третий абзац и полоса вне окна не в счёт"
    assert all(in_window for _, in_window, _ in lines)
    no_window = candidate_lines(blocks, floating, plain, page_of, pages, None, 20)
    assert [b for b, _, _ in no_window] == [1, 2, 6, 7], "без окна — только заголовочные строки со всех полос"
    articles = [
        TocArticle("О разработке систе", [], "5", None, "A1"),
        TocArticle("мы обеспечения", [], "5", None, "A2"),
    ]
    assert skip_reason(articles[1], articles) == "хвост названия предыдущей записи (A1)"
    assert (
        skip_reason(TocArticle("А00240. Подписано к печати 5/VII 1966 г.", [], "4", None, "A3"), articles) is not None
    )
    assert skip_reason(articles[0], articles) is None


def test_assembly_restores_heading_from_model_hint(tmp_path):
    """Статья без `#` и без похожей строки: модель указывает строку — она становится `#` с source model."""
    _toc(tmp_path, [(None, [("Первая", "1"), ("Совсем иначе названная статья", "3")])])
    _page(tmp_path, "IMG_0001", "# Первая\n\nТекст.", "1")
    _page(tmp_path, "IMG_0003", "## Об одном подходе к делу\n\nТекст второй статьи.", "3")
    client = FakeClient(
        '{"articles": [{"article_id": "A2", "line": 1, "confidence": "high", "notes": "перефразировано"}]}'
    )
    checker = HeadingChecker(client, resolve("deepseek-v41-flash"), tmp_path / "cache")
    assembly = assemble_issue(tmp_path, ISSUE, issue_pages(tmp_path, ISSUE), heading_checker=checker)
    assert "<!-- article A2 -->\n\n# Об одном подходе к делу\n\nТекст второй статьи." in assembly.text
    entry = next(a for a in assembly.reconcile.articles if a["id"] == "A2")
    assert (entry["status"], entry["source"], entry["model"]["status"]) == ("restored", "model", "accepted")
    assert assembly.heading_checked.requests == 1 and assembly.as_dict()["heading_checked"]["requests"] == 1
    user = client.payloads[0]["messages"][1]["content"]
    assert "Article A2: «Совсем иначе названная статья»" in user and "1. [p. 3] Об одном подходе к делу" in user
    assert "Article A1" not in user, "найденные статьи в запрос не идут"
    # Отклонённый ответ: статья остаётся missing, вердикт в sidecar.
    client2 = FakeClient('{"articles": [{"article_id": "A2", "line": 1, "confidence": "low"}]}')
    checker2 = HeadingChecker(client2, resolve("deepseek-v41-flash"), None)
    assembly2 = assemble_issue(tmp_path, ISSUE, issue_pages(tmp_path, ISSUE), heading_checker=checker2)
    entry2 = next(a for a in assembly2.reconcile.articles if a["id"] == "A2")
    assert entry2["status"] == "missing" and entry2["model"]["status"] == "rejected"
    assert "\n# Об одном подходе" not in assembly2.text


def test_marker_of_a_known_rubric_is_not_accepted_as_title(tmp_path):
    """Маркер «ОФИЦИАЛЬНЫЙ ОТДЕЛ» (рубрика оглавления) за название статьи раздела не принимается."""
    _toc(tmp_path, [("Официальный отдел", [("Типовые положения об управлениях", "5")])])
    _page(tmp_path, "IMG_0005", "<marker>*ОФИЦИАЛЬНЫЙ ОТДЕЛ*</marker>\n\n## Общие положения\n\nТекст.", "5")
    client = FakeClient('{"articles": [{"article_id": "A1", "line": 1, "confidence": "high"}]}')
    checker = HeadingChecker(client, resolve("deepseek-v41-flash"), None)
    assembly = assemble_issue(tmp_path, ISSUE, issue_pages(tmp_path, ISSUE), heading_checker=checker)
    entry = assembly.reconcile.articles[0]
    assert entry["status"] == "missing" and entry["model"]["status"] == "rejected"
    assert "маркер рубрики" in entry["model"]["notes"] and "# ОФИЦИАЛЬНЫЙ" not in assembly.text
