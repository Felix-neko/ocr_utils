"""Команда run на временной папке с фейковым клиентом: зеркальная структура, skip-done, сбои."""

import json
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from research.external_ocr_models import cli, ocr
from research.external_ocr_models.client import ChatResponse, OpenRouterError
from research.external_ocr_models.models import resolve
from research.external_ocr_models.ocr import RunOptions, build_payload, recognise_page
from research.external_ocr_models.imaging import prepare

ANSWER = '{"page_number": "3", "running_header": "МТС", "running_footer": null, "is_toc": false, "content_markdown": "# Заголовок\\n\\n**И. Фетисов**\\n\\nТекст.", "notes": ""}'


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.payloads = []

    def chat(self, payload):
        self.payloads.append(payload)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return ChatResponse(answer, "stop", "Fake", "fake/model", "gen-1", 1200, 300, 0, 0.001, 1.5, 1, {})


def _make_pages(root: Path):
    for rel in ("1966/03/IMG_0104_2R.jpg", "1966/03/IMG_0105_2R.jpg"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (400, 600), 230).save(path)


def test_payload_shape_json_schema():
    spec = resolve("gemini-31-flash-lite")
    payload = build_payload(spec, [], RunOptions(), "json_schema")
    assert payload["model"] == spec.openrouter_id
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["provider"]["require_parameters"] is True
    assert payload["reasoning"] == {"max_tokens": 1024}  # gemini: thinking не выключается, только потолок
    assert payload["messages"][0]["role"] == "system" and "content_markdown" in payload["messages"][0]["content"]


def test_payload_deepseek_json_object_and_provider_order(tmp_path):
    _make_pages(tmp_path)
    spec = resolve("deepseek-v41-flash")
    images = prepare(tmp_path / "1966/03/IMG_0104_2R.jpg", max_side=100, strips=2)
    payload = build_payload(spec, images, RunOptions(strips=2), spec.json_mode)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["provider"]["order"] == ["deepseek"] and payload["provider"]["ignore"] == ["relace"]
    parts = payload["messages"][1]["content"]
    assert parts[0]["type"] == "text" and "2 horizontal strips" in parts[0]["text"]
    assert [part["type"] for part in parts[1:]] == ["image_url", "image_url"]
    assert parts[1]["image_url"]["detail"] == "high"


def test_payload_no_reasoning_for_instruct():
    payload = build_payload(resolve("qwen3-vl-235b"), [], RunOptions(reasoning="low"), "json_schema")
    assert "reasoning" not in payload


def test_payload_reasoning_off_by_default():
    payload = build_payload(resolve("claude-haiku-45"), [], RunOptions(), "json_schema")
    assert payload["reasoning"] == {"enabled": False}
    assert "provider" not in build_payload(resolve("claude-haiku-45"), [], RunOptions(), "json_object")


def test_reasoning_rejected_is_resent_without_it(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _make_pages(in_dir)
    rel = Path("1966/03/IMG_0104_2R.jpg")
    fake = FakeClient([OpenRouterError("HTTP 400", 400, "unknown parameter: reasoning"), ANSWER])
    meta = recognise_page(fake, resolve("claude-haiku-45"), in_dir / rel, rel, out_dir, RunOptions())
    assert meta["error"] is None and meta["json_mode_used"] == "json_schema" and meta["reasoning_sent"] is False
    assert "reasoning" not in fake.payloads[1]


def test_run_mirrors_tree_and_skips_done(tmp_path, monkeypatch):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _make_pages(in_dir)
    fake = FakeClient([ANSWER, ANSWER, ANSWER])
    monkeypatch.setattr(cli, "OpenRouterClient", lambda *a, **k: fake)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["run", "--in-dir", str(in_dir), "--out-dir", str(out_dir), "--model", "gemini-31-flash-lite", "--jobs", "2"],
    )
    assert result.exit_code == 0, result.output
    for name in ("IMG_0104_2R", "IMG_0105_2R"):
        base = out_dir / "1966/03" / name
        assert base.with_suffix(".json").is_file() and base.with_suffix(".md").is_file()
        meta = json.loads(base.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["cost_usd"] == 0.001 and meta["page_number"] == "3" and meta["error"] is None
    assert (out_dir / "summary.csv").read_text(encoding="utf-8").count("\n") == 3
    assert (out_dir / "1966/03/IMG_0104_2R.md").read_text(encoding="utf-8").startswith('---\npage_number: "3"')

    # повтор с --skip-done ничего не шлёт
    result = runner.invoke(
        cli.main,
        ["run", "--in-dir", str(in_dir), "--out-dir", str(out_dir), "--model", "gemini-31-flash-lite", "--skip-done"],
    )
    assert result.exit_code == 0, result.output
    assert len(fake.payloads) == 2


def test_pages_file_and_limit(tmp_path, monkeypatch):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _make_pages(in_dir)
    pages = tmp_path / "pages.txt"
    pages.write_text("# комментарий\n1966/03/IMG_0105_2R.jpg\n", encoding="utf-8")
    fake = FakeClient([ANSWER])
    monkeypatch.setattr(cli, "OpenRouterClient", lambda *a, **k: fake)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    result = CliRunner().invoke(
        cli.main,
        [
            "run",
            "--in-dir",
            str(in_dir),
            "--out-dir",
            str(out_dir),
            "--model",
            "gemini-31-flash-lite",
            "--pages",
            str(pages),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "1966/03/IMG_0105_2R.json").is_file() and not (out_dir / "1966/03/IMG_0104_2R.json").exists()


def test_parse_failure_keeps_raw_and_is_redone(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _make_pages(in_dir)
    rel = Path("1966/03/IMG_0104_2R.jpg")
    spec = resolve("gemini-31-flash-lite")
    meta = recognise_page(FakeClient(["это не json"]), spec, in_dir / rel, rel, out_dir, RunOptions())
    assert (
        meta["parse_error"] and (out_dir / "1966/03/IMG_0104_2R.raw.txt").read_text(encoding="utf-8") == "это не json"
    )
    assert not ocr.is_done(out_dir, rel)


def test_schema_rejected_falls_back_to_json_object(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _make_pages(in_dir)
    rel = Path("1966/03/IMG_0104_2R.jpg")
    fake = FakeClient([OpenRouterError("HTTP 400", 400, "no structured outputs"), ANSWER])
    meta = recognise_page(fake, resolve("gemini-31-flash-lite"), in_dir / rel, rel, out_dir, RunOptions())
    assert meta["error"] is None and meta["json_mode_used"] == "json_object"
    assert fake.payloads[1]["response_format"] == {"type": "json_object"}
    assert ocr.is_done(out_dir, rel)


def test_request_error_recorded(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _make_pages(in_dir)
    rel = Path("1966/03/IMG_0104_2R.jpg")
    meta = recognise_page(
        FakeClient([OpenRouterError("HTTP 402", 402, "no credits")]),
        resolve("gemini-31-flash-lite"),
        in_dir / rel,
        rel,
        out_dir,
        RunOptions(),
    )
    assert "402" in meta["error"] and not ocr.is_done(out_dir, rel)


def test_damage_hints_in_payload():
    options = RunOptions(damage=True, hint="общая", hints={"1966/03/IMG_0104_2R.jpg": "особая"}, damage_side="auto")
    payload = build_payload(
        resolve("deepseek-v41-flash"), [], options, "json_object", rel=Path("1966/03/IMG_0104_2R.jpg")
    )
    system, user = payload["messages"][0]["content"], payload["messages"][1]["content"][0]["text"]
    assert "<restored>" in system and "<fuzzy>" in system and "<unknown/>" in system and '"damage"' in system
    assert user.startswith("общая особая") and "LEFT page" not in user  # суффикса _L/_R нет
    right = build_payload(resolve("deepseek-v41-flash"), [], options, "json_object", rel=Path("x/IMG_0006_R.jpg"))
    assert "RIGHT page" in right["messages"][1]["content"][0]["text"]
    plain = build_payload(resolve("deepseek-v41-flash"), [], RunOptions(), "json_object")
    assert "<restored>" not in plain["messages"][0]["content"]


def test_damage_tags_recorded_in_meta(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _make_pages(in_dir)
    rel = Path("1966/03/IMG_0104_2R.jpg")
    answer = ANSWER.replace("Текст.", "<restored>Те</restored>кст <fuzzy>и</fuzzy> <unknown/> <fuzzy>хвост")
    meta = recognise_page(
        FakeClient([answer]), resolve("gemini-31-flash-lite"), in_dir / rel, rel, out_dir, RunOptions(damage=True)
    )
    assert meta["tags"] == {"restored": 1, "fuzzy": 1, "unknown": 1} and "fuzzy" in meta["tag_warning"]
    assert "<restored>Те</restored>" in (out_dir / "1966/03/IMG_0104_2R.md").read_text(encoding="utf-8")


def test_read_hints(tmp_path):
    path = tmp_path / "hints.txt"
    path.write_text("# комментарий\nа/b.jpg\tсрезан правый край\nпусто\t\n", encoding="utf-8")
    assert cli.read_hints(path) == {"а/b.jpg": "срезан правый край"}


def test_source_in_system_prompt():
    generic = build_payload(resolve("deepseek-v41-flash"), [], RunOptions(), "json_object")["messages"][0]["content"]
    assert "journal or a newspaper" in generic and "this one:" not in generic
    specific = build_payload(resolve("deepseek-v41-flash"), [], RunOptions(source="журнал «МТС», 1966"), "json_object")
    assert "(this one: журнал «МТС», 1966)" in specific["messages"][0]["content"]
