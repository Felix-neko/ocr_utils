"""Тесты стенда article_problems без сети: нарезка выпуска, разбор ответа, поиск цитат, ключи кэша, сводка."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocr_utils.external_ocr_services.cache import CacheVerdict, RequestCache
from ocr_utils.external_ocr_services.client import ChatResponse
from research.article_problems.articles import parse_article_key, split_issue
from research.article_problems.batch import _spread_cost, chunk_articles, custom_id_of
from research.article_problems.payload import (
    ReasoningLevel,
    Transport,
    Variant,
    cache_entry,
    cache_payload,
    request_body,
    request_info,
    sync_payload,
)
from research.article_problems.quotes import find_quote
from research.article_problems.report import load_results, write_report
from research.article_problems.schema import CulpritKind, ImprovementKind, ParseVerdict, ProblemClass, parse_answer

# Выпуск в формате сборки внешнего OCR: обложка, рубрика, две статьи, во второй — вложенное письмо «##».
ISSUE_TEXT = """---
year: "1968"
issue: "09"
---
## Материально-техническое снабжение

<!-- rubric R1 -->

<rubric>*СНАБЖЕНИЕ ПРЕДПРИЯТИЙ*</rubric>

<!-- article A1 -->

# ПРИЧИНА НЕРИТМИЧНОСТИ

Из заявленных 43 т швеллеров завод получил от Калининской конторы только 7 т.

<author>**М. ЯКОБИ,**</author>

## СТРОГО ПЛАНИРОВАТЬ ПРОИЗВОДСТВО ЗАПАСНЫХ ЧАСТЕЙ

Тяжелое положение сложилось у нас на Тбилисском метрополитене с запасными частями к вагонам.
Зачастую заводы отказываются поставлять мелкие детали, предлагая взамен целые узлы.

<!-- article A2 -->

# Итоги соревнования

Коллектив базы выполнил план на 107%.
"""


@pytest.fixture()
def issue_file(tmp_path: Path) -> Path:
    """Файл выпуска ``1968/1968_09.md`` во временной папке пака."""
    path = tmp_path / "pack" / "1968" / "1968_09.md"
    path.parent.mkdir(parents=True)
    path.write_text(ISSUE_TEXT, encoding="utf-8")
    return path


def test_split_issue_rubric_title_authors_lines(issue_file: Path) -> None:
    """Статьи режутся по маркерам, рубрика наследуется, строка начала тела — точная."""
    articles = split_issue(issue_file)
    assert [article.article_id for article in articles] == ["A1", "A2"]
    first = articles[0]
    assert first.rubric == "СНАБЖЕНИЕ ПРЕДПРИЯТИЙ"
    assert first.title == "ПРИЧИНА НЕРИТМИЧНОСТИ"
    assert first.authors == ("М. ЯКОБИ",)
    # Вложенное письмо под «##» остаётся внутри статьи A1.
    assert "Тбилисском метрополитене" in first.text
    lines = ISSUE_TEXT.splitlines()
    assert lines[first.line_start - 1] == "# ПРИЧИНА НЕРИТМИЧНОСТИ"
    assert first.key == "1968_09:A1" and first.rel == Path("1968/09/A1")


def test_parse_article_key() -> None:
    """Ключ статьи ``ГГГГ_ВВ:An`` разбирается, неверный — ошибка."""
    assert parse_article_key("1968_09:A15") == ("1968_09", "A15")
    with pytest.raises(ValueError):
        parse_article_key("1968-09:15")


def test_parse_answer_normalizes_classes_and_adds_7() -> None:
    """Классы нормализуются, неизвестные откладываются, класс 7 добавляется, резюме — только по классам случаев."""
    text = json.dumps(
        {
            "cases": [
                {"classes": ["1", "класс 2.1", "13"], "culprit_kind": "manufacturer", "quote": "q"},
                {"classes": [4], "culprit_kind": "что-то своё"},
            ],
            "class_summaries": [{"class": "1", "summary": "есть"}, {"class": "6", "summary": "выдумка"}],
        },
        ensure_ascii=False,
    )
    answer = parse_answer(f"```json\n{text}\n```", "stop")
    assert answer.verdict == ParseVerdict.OK
    first, second = answer.cases
    assert first.classes == [ProblemClass("1"), ProblemClass("2.1"), ProblemClass("7")]
    assert first.unknown_classes == ["13"]
    assert second.classes == [ProblemClass("4"), ProblemClass("7")]
    assert second.culprit_kind == CulpritKind.UNKNOWN
    assert set(answer.class_summaries) == {ProblemClass("1")}


def test_parse_answer_gossnab_culprit_adds_8() -> None:
    """Виновник-подразделение Госснаба даёт класс 8, даже если модель его не поставила."""
    answer = parse_answer('{"cases": [{"classes": ["4"], "culprit_kind": "gossnab_body"}]}', "stop")
    assert answer.cases[0].classes == [ProblemClass("4"), ProblemClass("7"), ProblemClass("8")]


def test_parse_answer_repairs_summary_without_key() -> None:
    """Резюме вида ``"class": "7": "текст"`` чинится: ключ ``summary`` вставляется."""
    text = '{"cases": [{"classes": ["8"]}], "class_summaries": [{"class": "8": "контора сорвала поставки"}]}'
    answer = parse_answer(text, "stop")
    assert answer.verdict == ParseVerdict.OK
    assert answer.class_summaries[ProblemClass("8")] == "контора сорвала поставки"


def test_parse_answer_errors() -> None:
    """Обрезанный и не-JSON ответы получают свой вердикт, пустой список случаев — это OK."""
    assert parse_answer("{}", "length").verdict == ParseVerdict.TRUNCATED
    assert parse_answer("нет JSON", "stop").verdict == ParseVerdict.BAD_JSON
    empty = parse_answer('{"cases": [], "class_summaries": []}', "stop")
    assert empty.verdict == ParseVerdict.OK and not empty.cases


def test_find_quote_exact_fuzzy_and_missing(issue_file: Path) -> None:
    """Цитата ищется после нормализации (ё, кавычки, пробелы), с опечаткой — нечётко, пересказ — не находится."""
    article = split_issue(issue_file)[0]
    exact = find_quote(article, "Зачастую  заводы отказываются поставлять мелкие детали")
    assert exact.found and exact.score == 100.0
    assert ISSUE_TEXT.splitlines()[exact.line - 1].startswith("Зачастую")
    fuzzy = find_quote(article, "Тяжелое положение сложилось у нас на Тбилиском метрополитене")
    assert fuzzy.found and fuzzy.score < 100.0
    missing = find_quote(article, "Метрополитен в Тбилиси никак не может получить запчасти от заводов")
    assert not missing.found
    ellipsis = find_quote(article, "Тяжелое положение … заводы отказываются поставлять мелкие детали")
    assert ellipsis.found


def test_payload_keys_differ_by_variant(issue_file: Path, tmp_path: Path) -> None:
    """Sync и batch дают разные ключи и тела; batch-тело без провайдера, reasoning off → effort none."""
    article = split_issue(issue_file)[0]
    sync = Variant()
    batch = Variant(transport=Transport.BATCH)
    low = Variant(reasoning=ReasoningLevel.LOW)
    assert sync_payload(article, sync)["provider"]["order"] == ["deepseek"]
    assert cache_payload(article, batch)["model"].endswith(":batch")
    assert "provider" not in cache_payload(article, batch)
    assert request_body(article, sync)["reasoning"] == {"effort": "none"}
    assert request_body(article, low)["reasoning"] == {"effort": "low"}
    entries = {cache_entry(tmp_path, article, variant) for variant in (sync, batch, low)}
    assert len(entries) == 3
    assert custom_id_of(article) == "1968_09_A1"
    assert chunk_articles([article, article], batch) == [[article, article]]


def test_spread_cost_by_tokens() -> None:
    """Цена пачки раскладывается по ответам без своей цены пропорционально токенам."""

    def response(prompt: int, completion: int, cost: float | None) -> ChatResponse:
        return ChatResponse("", "stop", None, None, None, prompt, completion, 0, cost, 0.0, 1)

    spread = _spread_cost([response(1000, 0, None), response(3000, 0, None)], 0.4, "deepseek/deepseek-v4.1-flash:batch")
    assert [round(item.cost_usd, 6) for item in spread] == [0.1, 0.3]
    kept = _spread_cost([response(1000, 0, 0.5)], 9.0, "deepseek/deepseek-v4.1-flash:batch")
    assert kept[0].cost_usd == 0.5


def test_report_from_cached_answer(issue_file: Path, tmp_path: Path) -> None:
    """Ответ из кэша разбирается, цитаты проверяются, выходы прогона пишутся."""
    articles = split_issue(issue_file)
    variant = Variant()
    cache_root = tmp_path / "cache"
    cache = RequestCache(cache_root)
    answer = {
        "cases": [
            {
                "classes": ["1", "2.1"],
                "victim": "Тбилисский метрополитен",
                "culprit": "заводы",
                "culprit_kind": "manufacturer",
                "product": "запчасти",
                "quote": "Зачастую заводы отказываются поставлять мелкие детали",
                "explanation": "…",
                "concrete": True,
            }
        ],
        "class_summaries": [{"class": "1", "summary": "метро без запчастей"}],
    }
    response = ChatResponse(
        json.dumps(answer, ensure_ascii=False), "stop", "DeepSeek", "m", "id", 100, 50, 0, 0.001, 1.0, 1
    )
    article = articles[0]
    cache.store(
        cache_entry(cache_root, article, variant),
        request_info(article, variant),
        sync_payload(article, variant),
        response,
        CacheVerdict.OK,
    )
    results = load_results(articles, variant, cache, cache_root)
    assert results[1].answer is None
    case = results[0].answer.cases[0]
    assert case.quote_found and case.quote_line is not None
    numbers = write_report(results, variant, tmp_path / "run")
    assert numbers == {"articles": 2, "csv_rows": 3}
    sheet = (tmp_path / "run" / "by_class" / "class_2.1.md").read_text(encoding="utf-8")
    assert "Тбилисский метрополитен" in sheet
    rows = [json.loads(line) for line in (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["classes"] == ["1", "2.1", "7"] and rows[1]["verdict"] == "missing"


def test_load_results_falls_back_to_sync(issue_file: Path, tmp_path: Path) -> None:
    """Обрезанный ответ пачки подменяется годным синхронным ответом той же модели."""
    article = split_issue(issue_file)[0]
    cache_root = tmp_path / "cache"
    cache = RequestCache(cache_root)
    batch, sync = Variant(transport=Transport.BATCH), Variant()

    def store(variant: Variant, text: str, finish: str) -> None:
        response = ChatResponse(text, finish, None, None, None, 1, 1, 0, 0.0, 0.0, 1)
        cache.store(
            cache_entry(cache_root, article, variant),
            request_info(article, variant),
            cache_payload(article, variant),
            response,
            CacheVerdict.OK,
        )

    store(batch, "{", "length")
    assert load_results([article], batch, cache, cache_root)[0].answer.verdict == ParseVerdict.TRUNCATED
    store(sync, '{"cases": [{"classes": ["4"]}]}', "stop")
    result = load_results([article], batch, cache, cache_root, (sync,))[0]
    assert result.answer.verdict == ParseVerdict.OK and result.answered_by == sync.label


def test_compare_overlaps_and_sheets(tmp_path: Path) -> None:
    """Совпадение по классам считается по общим статьям, лист «только у одного» перечисляет свои статьи."""
    from research.article_problems.compare import write_comparison

    def write(path: Path, rows: list[dict]) -> Path:
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return path

    base = {"title": "т", "rubric": None, "source": "f:1", "class_summaries": {}}
    case = {
        "classes": ["1", "7"],
        "victim": "завод",
        "culprit": "",
        "explanation": "e",
        "quote": "q",
        "quote_found": True,
    }
    first = write(tmp_path / "a.jsonl", [{**base, "article": "x", "classes": ["1", "7"], "cases": [case]}])
    second = write(tmp_path / "b.jsonl", [{**base, "article": "x", "classes": ["7"], "cases": []}])
    table = {item.cls.value: item for item in write_comparison(first, second, "a", "b", tmp_path / "cmp")}
    assert (table["1"].only_first, table["1"].jaccard) == (1, 0.0)
    assert table["7"].both == 1 and table["7"].jaccard == 1.0
    assert "## x" in (tmp_path / "cmp" / "only_a" / "class_1.md").read_text(encoding="utf-8")


def test_export_by_class_union_and_header(issue_file: Path, tmp_path: Path) -> None:
    """Статья попадает в папку класса, если его дала хоть одна модель; шапка называет нашедших и причины."""
    from research.article_problems.export import CLASS_DIRS, ModelRun, export_by_class, file_name

    articles = split_issue(issue_file)
    case = {
        "classes": ["1", "7"],
        "victim": "метро",
        "culprit": "заводы",
        "product": "запчасти",
        "explanation": "нет запчастей",
        "quote": "q",
        "quote_found": True,
    }
    first = ModelRun(
        "A", {"1968_09:A1": {"classes": ["1", "7"], "class_summaries": {"1": "почему A"}, "cases": [case]}}
    )
    second = ModelRun("B", {"1968_09:A1": {"classes": ["4", "7"], "class_summaries": {}, "cases": []}})
    counts = export_by_class(articles, [first, second], tmp_path)
    assert counts[ProblemClass("1")] == 1 and counts[ProblemClass("4")] == 1 and counts[ProblemClass("6")] == 0
    text = (tmp_path / CLASS_DIRS[ProblemClass("1")] / file_name(articles[0])).read_text(encoding="utf-8")
    assert "**Нашли:** A" in text and "почему A" in text and "Тбилисском метрополитене" in text
    assert "### B" not in text
    assert file_name(articles[0]) == "1968_09_A1_ПРИЧИНА_НЕРИТМИЧНОСТИ.md"


def test_parse_answer_classes_11_12_and_barrier() -> None:
    """Классы 11 и 12 распознаются, вид случая класса 11 читается в enum."""
    text = json.dumps(
        {
            "cases": [
                {"classes": ["11"], "improvement_kind": "not_launched"},
                {"classes": ["12"], "improvement_kind": None},
            ]
        }
    )
    first, second = parse_answer(text, "stop").cases
    assert first.classes == [ProblemClass("7"), ProblemClass("11")] and first.improvement_kind == ImprovementKind(
        "not_launched"
    )
    assert second.classes == [ProblemClass("7"), ProblemClass("12")] and second.improvement_kind is None
