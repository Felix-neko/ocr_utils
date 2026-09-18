"""Второй проход по повреждённой полосе: триггер, страховка выбора, блок промпта, обёртка двух проходов."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ocr_utils.external_ocr_services.client import ChatResponse, OpenRouterError
from ocr_utils.external_ocr_services.models import resolve
from ocr_utils.external_ocr_services.ocr import (
    PageJob,
    RunOptions,
    SecondPass,
    choose_final,
    recognise_with_second_pass,
    second_pass_reason,
)
from ocr_utils.external_ocr_services.prompts import DEFAULT_DAMAGE_NOTE, user_prompt
from ocr_utils.external_ocr_services.schema import is_no_damage_text, parse_json_text
from tests.ocr_utils.external_ocr_services.test_pipeline import ISSUE, FakeClient, _answer, _make_pages


def _page(damaged=False, damage="", body="Текст.", edge=()):
    payload = json.loads(_answer(body=body))
    payload.update(damaged=damaged, damage=damage, edge_words=list(edge))
    return json.dumps(payload, ensure_ascii=False)


def test_trigger_reasons_and_no_damage_text():
    assert second_pass_reason(parse_json_text(_page())) is None
    assert second_pass_reason(parse_json_text(_page(damaged=True, damage="Правый край обрезан"))) == "damaged"
    assert second_pass_reason(parse_json_text(_page(body="Те<unclear>к</unclear>ст."))) == "tags"
    assert (
        second_pass_reason(parse_json_text(_page(edge=[{"seen": "а", "full": "аб", "kind": "hidden"}]))) == "edge_words"
    )
    # Без булева поля вердикт выводится из текста: «повреждений не обнаружено» — не повреждение.
    old = json.loads(_page())
    del old["damaged"]
    old["damage"] = "Повреждений не обнаружено, все буквы читаются уверенно."
    assert parse_json_text(json.dumps(old, ensure_ascii=False)).damaged is False
    old["damage"] = "Правая кромка обрезана"
    assert parse_json_text(json.dumps(old, ensure_ascii=False)).damaged is True
    assert is_no_damage_text("Помех и обрезки текста на странице не обнаружено, буквы чёткие.")
    assert not is_no_damage_text("Небольшие потертости и нечеткость по левому краю")


def test_choose_final_safeguard():
    assert choose_final({"supplied": 5, "unclear": 0, "gap": 0}, None) == ("pass1", "второй проход сбойнул")
    assert choose_final({"supplied": 5, "unclear": 0, "gap": 0}, {"supplied": 5, "unclear": 0, "gap": 9})[0] == "pass1"
    assert choose_final({"supplied": 5, "unclear": 0, "gap": 0}, {"supplied": 12, "unclear": 0, "gap": 3})[0] == "pass2"
    assert choose_final({"supplied": 0, "unclear": 0, "gap": 0}, {"supplied": 0, "unclear": 7, "gap": 0})[0] == "pass2"


def test_second_pass_prompt_block():
    hint = SecondPass(
        "Край обрезан",
        ({"seen": "снабже", "full": "снабже<supplied>ния</supplied>", "kind": "hidden"},) * 3,
        {"supplied": 3, "unclear": 0, "gap": 0},
    )
    text = user_prompt(2, 1, 2, second_pass=hint, max_lines=2)
    assert "A first reading of this page reported damage: «Край обрезан»" in text and "3 reconstructed" in text
    assert text.count("«снабже» →") == 2 and "and 1 more" in text
    assert "HIDDEN" in text and "VISIBLE BUT UNRELIABLE" in text and DEFAULT_DAMAGE_NOTE not in text
    assert "Its transcription was" not in text
    with_text = user_prompt(1, 1, 1, second_pass=SecondPass("x", (), {}, "ПОЛНЫЙ ТЕКСТ"))
    assert "<<<\nПОЛНЫЙ ТЕКСТ\n>>>" in with_text and "do not copy" in with_text
    assert "first reading" not in user_prompt(2, 1, 2)


def _run(tmp_path, answers, **options):
    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    fake = FakeClient(lambda p: answers.pop(0))
    opts = RunOptions(debug_dir=tmp_path / "dbg", second_pass=True, **options)
    meta, result = recognise_with_second_pass(
        fake, resolve("deepseek-v41-flash"), tmp_path / "in" / rel, PageJob(rel), tmp_path / "out", opts
    )
    return fake, meta, result, tmp_path / "out" / ISSUE / "IMG_0002", tmp_path / "dbg" / ISSUE / "IMG_0002"


def test_two_passes_keep_second_and_record_both(tmp_path):
    first = _page(damaged=True, damage="Правый край обрезан", body="Текст снабже.")
    second = _page(damaged=True, damage="Правый край обрезан", body="Текст снабже<supplied>ния</supplied>.")
    fake, meta, result, out, dbg = _run(tmp_path, [first, second])
    assert len(fake.payloads) == 2 and "first reading" in fake.payloads[1]["messages"][1]["content"][0]["text"]
    assert "Its transcription was" not in fake.payloads[1]["messages"][1]["content"][0]["text"]
    assert meta["second_pass"]["chosen"] == "pass2" and meta["second_pass_reason"] == "damaged"
    assert meta["second_pass"]["pass1_tags"]["supplied"] == 0 and meta["second_pass"]["pass2_tags"]["supplied"] == 1
    assert meta["cost_usd"] == 0.002 and "<supplied>ния</supplied>" in result.content_markdown
    assert "<supplied>" in out.with_suffix(".md").read_text(encoding="utf-8")
    assert json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))["second_pass_chosen"] == "pass2"
    assert dbg.with_suffix(".pass1.json").is_file() and dbg.with_suffix(".pass2.raw.txt").is_file()
    assert dbg.with_suffix(".raw.txt").is_file() and dbg.with_suffix(".pass2.prompt.txt").is_file()


def test_second_pass_with_transcript_and_safeguard(tmp_path):
    first = _page(damaged=True, damage="Край", body="Текст снабже<supplied>ния</supplied>.")
    second = _page(damaged=True, damage="Край", body="Текст снабже<gap>▒▒▒</gap>.")
    fake, meta, result, out, dbg = _run(tmp_path, [first, second], second_pass_transcript=True)
    user = fake.payloads[1]["messages"][1]["content"][0]["text"]
    assert "<<<\nТекст снабже<supplied>ния</supplied>." in user and ">>>" in user and "do not copy" in user
    assert meta["second_pass"]["chosen"] == "pass1" and "gap" in meta["second_pass"]["why"]
    assert "<supplied>ния</supplied>" in out.with_suffix(".md").read_text(encoding="utf-8")
    assert dbg.with_suffix(".pass2.json").is_file()
    assert meta["cost_usd"] == 0.002


def test_second_pass_failure_keeps_first_and_clean_page_has_one_request(tmp_path):
    first = _page(damaged=True, damage="Край")
    fake, meta, result, out, _ = _run(tmp_path, [first, OpenRouterError("HTTP 500", 500, "упс")])
    assert len(fake.payloads) == 2 and meta["second_pass"]["chosen"] == "pass1" and meta["error"] is None
    assert json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))["second_pass"]["pass2_error"]
    fake, meta, result, out, _ = _run(tmp_path, [_page()])
    assert len(fake.payloads) == 1 and meta["second_pass_reason"] is None
    fake, meta, result, out, _ = (
        _run(tmp_path, [_page(damaged=True, damage="Край")], second_pass=False)
        if False
        else (None, None, None, None, None)
    )


def test_pipeline_option_off_means_single_request(tmp_path):
    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    answers = [_page(damaged=True, damage="Край")]
    fake = FakeClient(lambda p: answers.pop(0))
    meta, _ = recognise_with_second_pass(
        fake,
        resolve("deepseek-v41-flash"),
        tmp_path / "in" / rel,
        PageJob(rel),
        tmp_path / "out",
        RunOptions(second_pass=False),
    )
    assert len(fake.payloads) == 1 and meta["second_pass_reason"] is None
