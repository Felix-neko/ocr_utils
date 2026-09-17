"""Пайплайн выпуска на фейковом клиенте: этапы, списки в промпте, skip-done, fallback, вето."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from ocr_utils.external_ocr_services import cli, pipeline
from ocr_utils.external_ocr_services.client import ChatResponse, OpenRouterError
from ocr_utils.external_ocr_services.models import resolve
from ocr_utils.external_ocr_services.ocr import PageJob, RunOptions, build_payload, is_done, recognise_page
from ocr_utils.external_ocr_services.pages import PageFlags
from ocr_utils.external_ocr_services.pipeline import PipelineParams, run_pipeline
from ocr_utils.external_ocr_services.tiling import prepare_tiles

ISSUE = "1966/03"
PAGES = ("IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg", "IMG_0004.jpg")


def _answer(toc_kind="none", body="# Заголовок\n\nТекст.", toc=None, title=None):
    payload = {
        "damage": "",
        "page_number": "3",
        "rubric": None,
        "title": title,
        "title_in_list": None,
        "authors": [],
        "running_header": "МТС",
        "running_footer": None,
        "toc_kind": toc_kind,
        "content_markdown": body,
        "restored": [],
        "fuzzy": [],
        "unknown": 0,
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
    page_prompts = [p["messages"][0]["content"] for p in fake.payloads[1:]]
    assert all("«Первые шаги» — И. Фетисов" in s and "Rubrics: «Опыт работы»" in s for s in page_prompts)
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


def test_missed_toc_redo_rebuilds_lists_and_rerecognises(tmp_path):
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
    params = _params(tmp_path, on_missed_toc="redo")
    params.options.max_src_tile = 700
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    assert stats.redone_issues == [ISSUE] and stats.missed
    # 1 toc + 3 page + 1 toc (найденная) + 2 page заново = 7 запросов; первая toc-полоса переиспользована.
    assert stats.requests == 7 and stats.reused == 1
    toc = json.loads((params.out_dir / ISSUE / "toc.json").read_text(encoding="utf-8"))
    titles = [a["title"] for s in toc["contents"]["sections"] for a in s["articles"]]
    assert titles == ["Первые шаги", "Третий шаг"] and toc["contents"]["continuations"] == 1
    last_prompts = [p["messages"][0]["content"] for p in fake.payloads[-2:]]
    assert all("«Третий шаг»" in s for s in last_prompts)
    metas = {
        name: json.loads((params.out_dir / ISSUE / f"{name[:-4]}.meta.json").read_text(encoding="utf-8"))
        for name in PAGES
    }
    assert metas["IMG_0001.jpg"]["stage"] == "toc" and metas["IMG_0003.jpg"]["stage"] == "page"
    assert len({metas[n]["toc_hash"] for n in ("IMG_0003.jpg", "IMG_0004.jpg")}) == 1
    assert not (params.out_dir / "missed_toc.txt").exists()


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
    meta, result = recognise_page(
        FakeClient(lambda p: OpenRouterError("HTTP 402", 402, "no credits")),
        spec,
        tmp_path / "in" / rel,
        job,
        out,
        RunOptions(),
    )
    assert "402" in meta["error"] and result is None and not is_done(out, job)
    meta, result = recognise_page(
        FakeClient(lambda p: "это не json"), spec, tmp_path / "in" / rel, job, out, RunOptions()
    )
    assert meta["parse_error"] and (out / ISSUE / "IMG_0002.raw.txt").read_text(encoding="utf-8") == "это не json"
    assert not is_done(out, job)
    meta, result = recognise_page(FakeClient(lambda p: _answer()), spec, tmp_path / "in" / rel, job, out, RunOptions())
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
    meta, result = recognise_page(
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
    assert (
        "(this one: журнал «МТС», 1966)" in payload["messages"][0]["content"]
        and "«Т»" in payload["messages"][0]["content"]
    )


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
    bad = CliRunner().invoke(
        cli.main, ["run", "--in-dir", str(tmp_path / "in"), "--out-dir", str(tmp_path / "out"), "--model", "нет"]
    )
    assert bad.exit_code != 0 and "неизвестная модель" in bad.output


def test_response_format_echo_is_retried_without_format(tmp_path):
    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    answers = ['{"type": "json_object"}', _answer()]
    fake = FakeClient(lambda p: answers.pop(0))
    meta, result = recognise_page(
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
    meta, result = recognise_page(
        FakeClient(lambda p: json.dumps(answer, ensure_ascii=False)),
        resolve("deepseek-v41-flash"),
        tmp_path / "in" / rel,
        job,
        tmp_path / "out",
        RunOptions(),
    )
    assert result.content_markdown.startswith("# Первые шаги\n\n<author>**С. Демидов**</author>")
    assert meta["structure"]["moved_authors"] == ["С. Демидов"] and result.title_in_list is True
