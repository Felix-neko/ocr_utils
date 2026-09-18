"""Пайплайн выпуска на фейковом клиенте: этапы, списки в промпте, skip-done, fallback, вето, частичный повтор."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from PIL import Image

from ocr_utils.external_ocr_services import cli, pipeline
from ocr_utils.external_ocr_services.client import ChatResponse, OpenRouterError
from ocr_utils.external_ocr_services.models import resolve
from ocr_utils.external_ocr_services.ocr import PageJob, RunOptions, build_payload, is_done, recognize_page
from ocr_utils.external_ocr_services.pages import PageFlags
from ocr_utils.external_ocr_services.pipeline import (
    PipelineParams,
    PipelineStats,
    RedoReason,
    RedoScope,
    redo_reason,
    run_pipeline,
)
from ocr_utils.external_ocr_services.schema import PageResult
from ocr_utils.external_ocr_services.tiling import prepare_tiles

ISSUE = "1966/03"
PAGES = ("IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg", "IMG_0004.jpg")


def _answer(toc_kind="none", body="# Заголовок\n\nТекст.", toc=None, title=None):
    payload = {
        "is_damaged": False,
        "damage_description": "",
        "page_number": "3",
        "rubric": None,
        "title": title,
        "title_in_list": None,
        "authors": [],
        "running_header": "МТС",
        "running_footer": None,
        "toc_kind": toc_kind,
        "content_markdown": body,
        "supplied": [],
        "unclear": [],
        "gap": 0,
        "edge_words": [],
        "notes": "",
    }
    if toc is not None:
        payload["toc"] = toc
    return json.dumps(payload, ensure_ascii=False)


def _toc_answer(titles, rubric="Опыт работы", continues=False, kind="contents"):
    toc = {
        "kind": kind,
        "continues_previous": continues,  # поле оглавления (полоса продолжает список), остаётся
        "sections": [
            {
                "rubric": rubric,
                "articles": [
                    {"title": t, "authors": [{"name": "И. Фетисов", "position": None}], "page": "5", "issue": None}
                    for t in titles
                ],
            }
        ],
    }
    return _answer(kind, "# СОДЕРЖАНИЕ\n\n- И. Фетисов. …", toc)


class FakeClient:
    """Отвечает по функции от текста системного промпта и имени полосы; помнит все запросы."""

    def __init__(self, reply):
        self.reply = reply
        self.payloads: list[dict] = []

    def chat(self, payload):
        self.payloads.append(payload)
        answer = self.reply(payload)
        if isinstance(answer, Exception):
            raise answer
        return ChatResponse(
            answer, "stop", "Fake", "fake/model", "gen-1", 1200, 300, 0, 0.001, 1.5, 1, {}, cached_tokens=900
        )

    def stage_of(self, payload) -> str:
        return "toc" if '"toc": {"kind"' in payload["messages"][0]["content"] else "page"


def _make_pages(root: Path):
    for name in PAGES:
        path = root / ISSUE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (400, 600), 230).save(path)


def _flags(toc=("IMG_0001",), veto=()):
    table = {f"{ISSUE}/{name[:-4]}": PageFlags() for name in PAGES}
    for stem in toc:
        table[f"{ISSUE}/{stem}"] = PageFlags(is_toc=True)
    for stem in veto:
        table[f"{ISSUE}/{stem}"] = PageFlags(force_is_not_toc=True)
    return table


def _params(tmp_path, **kwargs):
    return PipelineParams(
        tmp_path / "in", tmp_path / "out", RunOptions(debug_dir=tmp_path / "dbg"), _flags(), jobs=2, **kwargs
    )


def _default_reply(fake):
    def reply(payload):
        if fake.stage_of(payload) == "toc":
            return _toc_answer(["Первые шаги", "Второй шаг"])
        return _answer()

    return reply


def test_two_stages_lists_in_prompt_and_outputs(tmp_path):
    _make_pages(tmp_path / "in")
    fake = FakeClient(lambda p: None)
    fake.reply = _default_reply(fake)
    params = _params(tmp_path, on_missed_toc="skip")
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    assert (stats.issues, stats.pages, stats.requests, stats.failed, stats.toc_pages) == (1, 4, 4, 0, 1)
    assert fake.stage_of(fake.payloads[0]) == "toc", "оглавление раньше остальных"
    # Список выпуска — в тексте пользовательского сообщения (первая часть до картинок), не в системном.
    page_prompts = [p["messages"][1]["content"][0]["text"] for p in fake.payloads[1:]]
    assert all("«Первые шаги» — И. Фетисов" in s and "Rubrics: «Опыт работы»" in s for s in page_prompts)
    assert all("«Первые шаги»" not in p["messages"][0]["content"] for p in fake.payloads[1:])
    toc = json.loads((params.out_dir / ISSUE / "toc.json").read_text(encoding="utf-8"))
    assert [a["title"] for a in toc["contents"]["sections"][0]["articles"]] == ["Первые шаги", "Второй шаг"]
    assert "## Опыт работы" in (params.out_dir / ISSUE / "toc.md").read_text(encoding="utf-8")
    for name in PAGES:
        base = params.out_dir / ISSUE / name[:-4]
        assert base.with_suffix(".md").is_file() and base.with_suffix(".json").is_file()
        meta = json.loads(base.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["tiling"]["nrows"] == 1 and meta["cost_usd"] == 0.001 and meta["cached_tokens"] == 900
    meta = json.loads((params.out_dir / ISSUE / "IMG_0002.meta.json").read_text(encoding="utf-8"))
    assert meta["stage"] == "page" and meta["articles_in_prompt"] == 2 and len(meta["toc_hash"]) == 12
    # «Заголовок» не из списка [Первые шаги, Второй шаг] — понижен кодом, title_in_list пересчитан.
    assert meta["structure"]["demoted_headings"] == ["Заголовок"] and meta["title_in_list"] is False
    assert (params.out_dir / ISSUE / "IMG_0002.md").read_text(encoding="utf-8").count("\n## Заголовок") == 1
    assert (tmp_path / "dbg" / ISSUE / "IMG_0002.raw.txt").is_file()
    assert (tmp_path / "dbg" / ISSUE / "IMG_0002.tile_00.jpg").is_file()
    assert "=== user ===" in (tmp_path / "dbg" / ISSUE / "IMG_0002.prompt.txt").read_text(encoding="utf-8")
    summary = (params.out_dir / "summary.csv").read_text(encoding="utf-8")
    assert summary.count("\n") == 5 and "cached_tokens" in summary.splitlines()[0] and ",900," in summary
    assert not (params.out_dir / "missed_toc.txt").exists()

    # Повтор с --skip-done ничего не шлёт, а слитое оглавление читается из готовых .json.
    before = len(fake.payloads)
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), _params(tmp_path, skip_done=True, on_missed_toc="skip"))
    assert len(fake.payloads) == before and stats.reused == 4
    assert (params.out_dir / ISSUE / "toc.md").is_file()


def test_missed_toc_skip_writes_list_and_veto_silences(tmp_path):
    _make_pages(tmp_path / "in")
    fake = FakeClient(lambda p: None)

    def reply(payload):
        if fake.stage_of(payload) == "toc":
            return _toc_answer(["Первые шаги"])
        if "IMG_0003" in payload["messages"][1]["content"][0]["text"] or True:
            pass
        return _answer("index") if len(fake.payloads) % 2 == 0 else _answer()

    fake.reply = reply
    params = _params(tmp_path, on_missed_toc="skip")
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    assert stats.missed and not stats.redone_issues
    listed = (params.out_dir / "missed_toc.txt").read_text(encoding="utf-8")
    assert "# index" in listed and ISSUE in listed

    # То же с вето на всех обычных полосах — тревоги нет.
    fake.payloads.clear()
    params = _params(tmp_path, on_missed_toc="skip")
    params.flags = _flags(veto=("IMG_0002", "IMG_0003", "IMG_0004"))
    (params.out_dir / "missed_toc.txt").unlink()
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    assert not stats.missed and not (params.out_dir / "missed_toc.txt").exists()


@pytest.mark.parametrize("redo_scope", [RedoScope.ALL, RedoScope.STRUCTURED])
def test_missed_toc_redo_rebuilds_lists_and_rerecognizes(tmp_path, redo_scope):
    _make_pages(tmp_path / "in")
    # Полоса IMG_0002 делается выше остальных: при шаге сетки 700 px только она уходит двумя тайлами,
    # и фейк узнаёт её по фразе про тайлы в пользовательском промпте — так ответ не зависит от
    # порядка запросов в пуле потоков.
    Image.new("L", (400, 900), 230).save(tmp_path / "in" / ISSUE / "IMG_0002.jpg")
    fake = FakeClient(lambda p: None)
    marker = "2 overlapping tiles"

    def reply(payload):
        user = payload["messages"][1]["content"][0]["text"]
        if fake.stage_of(payload) == "toc":
            if marker in user:  # найденная полоса — продолжение без рубрики
                return _toc_answer(["Третий шаг"], rubric=None, continues=True)
            return _toc_answer(["Первые шаги"])
        if marker in user:  # на этапе page выглядит оглавлением
            return _answer("contents")
        return _answer()

    fake.reply = reply
    params = _params(tmp_path, on_missed_toc="redo", redo_scope=redo_scope)
    params.options.max_src_tile = 700
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    assert stats.redone_issues == [ISSUE] and stats.missed
    # 1 toc + 3 page + 1 toc (найденная) + 2 page заново = 7 запросов; первая toc-полоса переиспользована.
    # При structured обе обычные полосы тоже идут заново: у них в теле есть `# Заголовок`.
    assert stats.requests == 7 and stats.reused == 1 and stats.redo_kept == 0
    toc = json.loads((params.out_dir / ISSUE / "toc.json").read_text(encoding="utf-8"))
    titles = [a["title"] for s in toc["contents"]["sections"] for a in s["articles"]]
    assert titles == ["Первые шаги", "Третий шаг"] and toc["contents"]["continuations"] == 1
    last_prompts = [p["messages"][1]["content"][0]["text"] for p in fake.payloads[-2:]]
    assert all("«Третий шаг»" in s for s in last_prompts)
    metas = {
        name: json.loads((params.out_dir / ISSUE / f"{name[:-4]}.meta.json").read_text(encoding="utf-8"))
        for name in PAGES
    }
    assert metas["IMG_0001.jpg"]["stage"] == "toc" and metas["IMG_0003.jpg"]["stage"] == "page"
    assert len({metas[n]["toc_hash"] for n in ("IMG_0003.jpg", "IMG_0004.jpg")}) == 1
    assert not (params.out_dir / "missed_toc.txt").exists()


def test_missed_toc_redo_structured_keeps_plain_pages(tmp_path):
    """Круг повтора по structured: полоса сплошного текста берётся с диска, полоса с заголовком — заново."""
    _make_pages(tmp_path / "in")
    # Полосы различаются числом тайлов: IMG_0002 (2 тайла) — найденное оглавление, IMG_0004 (3 тайла) —
    # сплошной текст без заголовков и тегов; IMG_0003 (1 тайл) — с заголовком.
    Image.new("L", (400, 900), 230).save(tmp_path / "in" / ISSUE / "IMG_0002.jpg")
    Image.new("L", (400, 2000), 230).save(tmp_path / "in" / ISSUE / "IMG_0004.jpg")
    fake = FakeClient(lambda p: None)

    def reply(payload):
        user = payload["messages"][1]["content"][0]["text"]
        if fake.stage_of(payload) == "toc":
            if "2 overlapping tiles" in user:
                return _toc_answer(["Третий шаг"], rubric=None, continues=True)
            return _toc_answer(["Первые шаги"])
        if "2 overlapping tiles" in user:
            return _answer("contents")
        if "3 overlapping tiles" in user:
            return _answer(body="Сплошной текст полосы-продолжения.\n\nЕщё абзац.")
        return _answer()

    fake.reply = reply
    params = _params(tmp_path, on_missed_toc="redo")  # умолчание — structured
    params.options.max_src_tile = 700
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    # 1 toc + 3 page + 1 toc (найденная) + 1 page заново (IMG_0003 с `#`) = 6; с диска — toc-полоса
    # из базы и оставленная IMG_0004.
    assert stats.redone_issues == [ISSUE]
    assert (stats.requests, stats.reused, stats.redo_kept) == (6, 2, 1)
    metas = {
        name: json.loads((params.out_dir / ISSUE / f"{name[:-4]}.meta.json").read_text(encoding="utf-8"))
        for name in PAGES
    }
    # У оставленной полосы — новый toc_hash (тот же, что у перезапрошенной) и пометка redo_kept;
    # список в её промпте был старый — articles_in_prompt остался от первого круга.
    assert metas["IMG_0004.jpg"]["redo_kept"] is True and metas["IMG_0004.jpg"]["articles_in_prompt"] == 1
    assert metas["IMG_0004.jpg"]["toc_hash"] == metas["IMG_0003.jpg"]["toc_hash"]
    assert not metas["IMG_0003.jpg"].get("redo_kept") and metas["IMG_0003.jpg"]["articles_in_prompt"] == 2
    # Человек поставил тег найденной полосе в базе; повтор с --skip-done: всё готово, запросов нет.
    tagged = _params(tmp_path, skip_done=True)
    tagged.flags = _flags(toc=("IMG_0001", "IMG_0002"))
    again = run_pipeline(fake, resolve("deepseek-v41-flash"), tagged)
    assert again.requests == 0 and again.reused == 4


def _page(body="Текст.", **fields) -> PageResult:
    return PageResult(content_markdown=body, **fields)


def test_redo_reason_by_structure_signs():
    meta = {"structure": {"demoted_headings": [], "markers_from_rubrics": []}}
    # Сплошной текст без признаков — оставить.
    assert redo_reason(_page(), meta, {"3"}, False) is None
    # Понижённая полоса — всегда заново; нет результата первого круга — тоже.
    assert redo_reason(_page(), meta, set(), True) is RedoReason.DEMOTED
    assert redo_reason(None, meta, set(), False) is RedoReason.NO_FIRST_PASS
    assert redo_reason(_page(), None, set(), False) is RedoReason.NO_FIRST_PASS
    # Заголовок любого уровня, теги структуры (маркер — рубрика не из списка), поля модели.
    assert redo_reason(_page("Абзац.\n\n## Подзаголовок"), meta, set(), False) is RedoReason.HEADING
    assert redo_reason(_page("<marker>*Рынок*</marker>\n\nТекст."), meta, set(), False) is RedoReason.STRUCTURE_TAG
    assert (
        redo_reason(_page("Текст.", authors=[{"name": "И. Иванов"}]), meta, set(), False) is RedoReason.TITLE_OR_AUTHORS
    )
    assert redo_reason(_page("Текст.", rubric="Опыт"), meta, set(), False) is RedoReason.TITLE_OR_AUTHORS
    # Переписанный колонтитул — как `##`, `<rubric>`/`<marker>` или поле rubric — не признак.
    header = _page(
        "## Проблемы и суждения\n\n<marker>*ПРОБЛЕМЫ И СУЖДЕНИЯ*</marker>\n\nТекст.",
        running_header="Проблемы и суждения",
        rubric="ПРОБЛЕМЫ И СУЖДЕНИЯ",
    )
    assert redo_reason(header, meta, set(), False) is None
    header_marker = {"structure": {"markers_from_rubrics": ["Проблемы и суждения"], "markers": 1}}
    assert redo_reason(_page("Текст.", running_header="ПРОБЛЕМЫ И СУЖДЕНИЯ"), header_marker, set(), False) is None
    assert redo_reason(_page("Текст."), header_marker, set(), False) is RedoReason.STRUCTURE_EDITS
    assert (
        redo_reason(_page("<rubric>*Рынок*</rubric>\n\nТекст.", running_header="Опыт"), meta, set(), False)
        is RedoReason.STRUCTURE_TAG
    )
    # Следы пост-обработки в meta и совпадение номера страницы с началом статьи по новому оглавлению.
    edited = {"structure": {"demoted_headings": ["Старый"], "markers": 0}}
    assert redo_reason(_page(), edited, set(), False) is RedoReason.STRUCTURE_EDITS
    assert redo_reason(_page(page_number=" 3 "), meta, {"3"}, False) is RedoReason.ARTICLE_START
    assert redo_reason(_page(page_number="4"), meta, {"3"}, False) is None


def test_pipeline_stats_add_sums_and_keeps_operands():
    left = PipelineStats(requests=2, cost_usd=0.5, missed=["a"], redo_kept=1)
    right = PipelineStats(requests=3, reused=1, cost_usd=0.25, missed=["b"], demoted_toc=["d"])
    total = left + right
    assert (total.requests, total.reused, total.cost_usd, total.redo_kept) == (5, 1, 0.75, 1)
    assert total.missed == ["a", "b"] and total.demoted_toc == ["d"]
    assert left == PipelineStats(requests=2, cost_usd=0.5, missed=["a"], redo_kept=1)
    assert right.missed == ["b"]


def test_ask_without_tty_behaves_like_skip(tmp_path, monkeypatch):
    _make_pages(tmp_path / "in")
    fake = FakeClient(lambda p: None)
    fake.reply = lambda p: _toc_answer(["Т"]) if fake.stage_of(p) == "toc" else _answer("contents")
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: False)
    params = _params(tmp_path, on_missed_toc="ask")
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    assert stats.missed and not stats.redone_issues and (params.out_dir / "missed_toc.txt").is_file()


def test_request_and_parse_failures_are_not_done(tmp_path):
    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    job = PageJob(rel, "page", "none", "1966", (), (), "abc")
    spec = resolve("deepseek-v41-flash")
    out = tmp_path / "out"
    meta, result = recognize_page(
        FakeClient(lambda p: OpenRouterError("HTTP 402", 402, "no credits")),
        spec,
        tmp_path / "in" / rel,
        job,
        out,
        RunOptions(),
    )
    assert "402" in meta["error"] and result is None and not is_done(out, job)
    meta, result = recognize_page(
        FakeClient(lambda p: "это не json"), spec, tmp_path / "in" / rel, job, out, RunOptions()
    )
    assert meta["parse_error"] and (out / ISSUE / "IMG_0002.raw.txt").read_text(encoding="utf-8") == "это не json"
    assert not is_done(out, job)
    meta, result = recognize_page(FakeClient(lambda p: _answer()), spec, tmp_path / "in" / rel, job, out, RunOptions())
    assert result is not None and is_done(out, job) and not (out / ISSUE / "IMG_0002.raw.txt").exists()
    assert not is_done(out, PageJob(rel, "page", "none", "1966", (), (), "другой"))
    assert not is_done(out, PageJob(rel, "toc", "contents", "1966"))


def test_schema_rejected_falls_back_and_reasoning_removed(tmp_path):
    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    job = PageJob(rel)
    answers = [
        OpenRouterError("HTTP 400", 400, "no structured outputs"),
        OpenRouterError("HTTP 400", 400, "unknown parameter: reasoning"),
        _answer(),
    ]
    fake = FakeClient(lambda p: answers.pop(0))
    meta, result = recognize_page(
        fake, resolve("gemini-31-flash-lite"), tmp_path / "in" / rel, job, tmp_path / "out", RunOptions()
    )
    assert meta["error"] is None and meta["json_mode_used"] == "json_object" and meta["reasoning_sent"] is False
    assert fake.payloads[0]["response_format"]["type"] == "json_schema" and "reasoning" not in fake.payloads[2]


def test_payload_deepseek_tiles_and_source_year(tmp_path):
    _make_pages(tmp_path / "in")
    spec = resolve("deepseek-v41-flash")
    tiles = prepare_tiles(tmp_path / "in" / ISSUE / "IMG_0002.jpg", max_src_tile=300, max_model_tile=200)
    job = PageJob(Path(ISSUE) / "IMG_0002.jpg", "page", "none", "1966", ("Р",), ({"title": "Т", "authors": []},), "h")
    payload = build_payload(spec, tiles, RunOptions(source="журнал «МТС», {year}"), job, spec.json_mode)
    assert payload["response_format"] == {"type": "json_object"} and payload["provider"]["order"] == ["deepseek"]
    parts = payload["messages"][1]["content"]
    assert "4 overlapping tiles: 2 column(s) × 2 row(s)" in parts[0]["text"]
    assert [p["type"] for p in parts[1:]] == ["image_url"] * 4 and parts[1]["image_url"]["detail"] == "high"
    assert "(this one: журнал «МТС», 1966)" in payload["messages"][0]["content"] and "«Т»" in parts[0]["text"]


def test_cli_run_with_db_and_toc_lists(tmp_path, monkeypatch):
    _make_pages(tmp_path / "in")
    lists = tmp_path / "lists" / ISSUE
    lists.mkdir(parents=True)
    (lists / "toc_pages.txt").write_text("IMG_0001.jpg  # contents 1.00 cvat\n", encoding="utf-8")
    fake = FakeClient(lambda p: None)
    fake.reply = _default_reply(fake)
    monkeypatch.setattr(cli, "OpenRouterClient", lambda *a, **k: fake)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    result = CliRunner().invoke(
        cli.main,
        [
            "run",
            "--in-dir",
            str(tmp_path / "in"),
            "--out-dir",
            str(tmp_path / "out"),
            "--toc-lists",
            str(tmp_path / "lists"),
            "--on-missed-toc",
            "skip",
            "--jobs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Выпусков: 1, полос: 4 (оглавление/указатель по базе: 1), запросов: 4" in result.output
    assert fake.stage_of(fake.payloads[0]) == "toc"
    assert "Второй проход" not in result.output, "по умолчанию второго прохода нет — и строки о нём тоже"
    summary = (tmp_path / "out" / "summary.csv").read_text(encoding="utf-8")
    assert "is_damaged" in summary.splitlines()[0]
    with_pass = CliRunner().invoke(
        cli.main,
        ["run", "--in-dir", str(tmp_path / "in"), "--out-dir", str(tmp_path / "out2"), "--second-pass", "--jobs", "1"],
    )
    assert with_pass.exit_code == 0, with_pass.output
    assert "Второй проход: 0 полос" in with_pass.output, "флаг включает проход; на чистых полосах он не срабатывает"
    bad = CliRunner().invoke(
        cli.main, ["run", "--in-dir", str(tmp_path / "in"), "--out-dir", str(tmp_path / "out"), "--model", "нет"]
    )
    assert bad.exit_code != 0 and "неизвестная модель" in bad.output


def test_response_format_echo_is_retried_without_format(tmp_path):
    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    answers = ['{"type": "json_object"}', _answer()]
    fake = FakeClient(lambda p: answers.pop(0))
    meta, result = recognize_page(
        fake, resolve("deepseek-v41-flash"), tmp_path / "in" / rel, PageJob(rel), tmp_path / "out", RunOptions()
    )
    assert result is not None and meta["json_mode_used"] == "none" and meta["cost_usd_wasted"] == 0.001
    assert fake.payloads[0]["response_format"] == {"type": "json_object"} and "response_format" not in fake.payloads[1]
    assert meta["fallbacks"][0]["error"] == "эхо response_format"


def test_page_stage_moves_author_after_listed_heading(tmp_path):
    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    body = "<author>**С. Демидов**</author>\n\n# Первые шаги\n\nТекст."
    answer = json.loads(_answer(body=body, title="Первые шаги"))
    answer["authors"] = [
        {"name": "С. Демидов", "position": None, "article": "starts_here", "printed": "running_header"}
    ]
    job = PageJob(rel, "page", "none", "1966", (), ({"title": "Первые шаги", "authors": []},), "h")
    meta, result = recognize_page(
        FakeClient(lambda p: json.dumps(answer, ensure_ascii=False)),
        resolve("deepseek-v41-flash"),
        tmp_path / "in" / rel,
        job,
        tmp_path / "out",
        RunOptions(),
    )
    assert result.content_markdown.startswith("# Первые шаги\n\n<author>**С. Демидов**</author>")
    assert meta["structure"]["moved_authors"] == ["С. Демидов"] and result.title_in_list is True


class DemotingReply:
    """Ответы для теста понижения: 0001 — настоящее оглавление; 0002 — «не оглавление», toc пустой;
    0003 — «не оглавление», но одна статья в toc; остальное — обычные полосы."""

    def __init__(self, fake: FakeClient):
        self.fake = fake

    def __call__(self, payload):
        if self.fake.stage_of(payload) != "toc":
            return _answer(title="Обычная")
        name = self.fake.page_of(payload)
        if name == "IMG_0002":
            return _answer("none", "Обычный текст.", {"kind": "contents", "continues_previous": False, "sections": []})
        if name == "IMG_0003":
            toc = {
                "kind": "contents",
                "continues_previous": False,
                "sections": [
                    {"rubric": None, "articles": [{"title": "Третья", "authors": [], "page": "9", "issue": None}]}
                ],
            }
            return _answer("none", "Похоже на список, но не оглавление.", toc)
        return _toc_answer(["Первые шаги", "Второй шаг"])


def test_toc_page_demoted_when_model_disagrees(tmp_path):
    """Полоса из базы, которую модель не признала оглавлением, идёт ещё и этапом page; её статьи
    (если есть) всё равно попадают в оглавление; повтор с --skip-done ничего не запрашивает."""
    _make_pages(tmp_path / "in")
    fake = FakeClient(lambda p: None)
    fake.reply = DemotingReply(fake)
    # Имя полосы из payload не узнать (тайлы одинаковые), поэтому оно берётся по очереди запросов:
    # этап toc при jobs=1 идёт в порядке 0001, 0002, 0003.
    order = iter(["IMG_0001", "IMG_0002", "IMG_0003"])
    fake.page_of = lambda payload: next(order, "IMG_0009")
    flags = _flags(toc=("IMG_0001", "IMG_0002", "IMG_0003"))
    params = PipelineParams(
        tmp_path / "in", tmp_path / "out", RunOptions(second_pass=False), flags, jobs=1, on_missed_toc="skip"
    )
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    # 3 запроса этапа toc + 1 обычная полоса + 2 понижённые этапом page = 6.
    assert (stats.requests, stats.failed) == (6, 0)
    assert stats.demoted_toc == [f"{ISSUE}: IMG_0002.jpg, IMG_0003.jpg"]
    out = tmp_path / "out" / ISSUE
    toc = json.loads((out / "toc.json").read_text(encoding="utf-8"))
    titles = [a["title"] for s in toc["contents"]["sections"] for a in s["articles"]]
    assert titles == ["Первые шаги", "Второй шаг", "Третья"], "статья с понижённой полосы в оглавлении"
    for stem in ("IMG_0002", "IMG_0003"):
        meta = json.loads((out / f"{stem}.meta.json").read_text(encoding="utf-8"))
        assert meta["stage"] == "page" and meta["toc_demoted"] is True and meta["articles_in_prompt"] == 3
        assert (out / f"{stem}.toc.json").is_file()
    listed = (tmp_path / "out" / "demoted_toc.txt").read_text(encoding="utf-8")
    assert "IMG_0002.jpg" in listed and "IMG_0003.jpg" in listed
    # Список статей в промпте у понижённой полосы — тот же, что у обычной.
    page_payloads = [p for p in fake.payloads if fake.stage_of(p) == "page"]
    assert len(page_payloads) == 3 and all("«Третья»" in p["messages"][1]["content"][0]["text"] for p in page_payloads)

    # Повтор с --skip-done: всё берётся с диска, оглавление то же.
    again = FakeClient(lambda p: (_ for _ in ()).throw(AssertionError("запросов быть не должно")))
    stats2 = run_pipeline(again, resolve("deepseek-v41-flash"), replace_params(params, skip_done=True))
    assert stats2.requests == 0 and stats2.reused == 6
    assert json.loads((out / "toc.json").read_text(encoding="utf-8")) == toc


def replace_params(params: PipelineParams, **changes) -> PipelineParams:
    """Копия параметров с изменёнными полями (dataclasses.replace без импорта в тесте)."""
    from dataclasses import replace

    return replace(params, **changes)
