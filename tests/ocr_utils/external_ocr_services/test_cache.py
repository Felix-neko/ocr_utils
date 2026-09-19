"""Кэш запросов: ключ по payload с хэшами тайлов, попадание без запроса, проходы по папкам, эхо и ошибки."""

from __future__ import annotations

import json
from pathlib import Path

from ocr_utils.external_ocr_services.cache import (
    ERRORS_FILE,
    REQUEST_FILE,
    RESPONSE_FILE,
    CacheVerdict,
    request_key,
    strip_images,
)
from ocr_utils.external_ocr_services.client import OpenRouterError
from ocr_utils.external_ocr_services.models import resolve
from ocr_utils.external_ocr_services.ocr import PageJob, RunOptions, build_payload, recognize_page
from ocr_utils.external_ocr_services.pipeline import PipelineParams, run_pipeline
from ocr_utils.external_ocr_services.tiling import prepare_tiles
from tests.ocr_utils.external_ocr_services.test_pipeline import (
    ISSUE,
    DemotingReply,
    FakeClient,
    _answer,
    _default_reply,
    _flags,
    _make_pages,
    _params,
)


def _payload(tmp_path: Path, name: str = "IMG_0002.jpg", **job_fields) -> dict:
    """Payload для полосы из тестового входа — как его собирает recognize_page."""
    rel = Path(ISSUE) / name
    tiles = prepare_tiles(tmp_path / "in" / rel, 4500, 2200, 85)
    spec = resolve("deepseek-v41-flash")
    return build_payload(spec, tiles, RunOptions(), PageJob(rel, **job_fields), spec.json_mode)


def test_request_key_depends_on_prompt_and_tiles_not_on_dict_order(tmp_path):
    _make_pages(tmp_path / "in")
    payload = _payload(tmp_path)
    key = request_key(payload)
    assert len(key) == 64 and key == request_key(payload), "ключ детерминирован"
    # Тот же payload с другим порядком ключей — тот же ключ.
    reordered = {name: payload[name] for name in reversed(list(payload))}
    assert request_key(reordered) == key
    # Другой список статей (другой пользовательский промпт) и другой этап — другие ключи.
    assert request_key(_payload(tmp_path, articles=({"title": "Статья", "authors": [], "rubric": None},))) != key
    assert request_key(_payload(tmp_path, stage="toc", toc_kind="contents")) != key
    # Та же картинка другого имени — тот же ключ (в ключе байты тайла, не имя файла).
    (tmp_path / "in" / ISSUE / "IMG_0002.jpg").rename(tmp_path / "in" / ISSUE / "IMG_0009.jpg")
    assert request_key(_payload(tmp_path, "IMG_0009.jpg")) == key
    # Без картинок: хэши тайлов на месте частей image_url, данных нет.
    stripped, tiles = strip_images(payload)
    parts = stripped["messages"][1]["content"]
    assert len(tiles) == 1 and tiles[0].startswith("sha256:") and parts[1]["tile"] == tiles[0]
    assert "url" not in parts[1] and payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:")


def test_second_run_with_cache_sends_nothing_and_keeps_every_pass(tmp_path):
    _make_pages(tmp_path / "in")
    fake = FakeClient(lambda p: None)
    fake.reply = _default_reply(fake)
    cache_dir = tmp_path / "cache"
    params = _params(tmp_path, on_missed_toc="skip")
    params.options.cache_dir = cache_dir
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    assert (stats.requests, stats.cache_hits) == (4, 0) and stats.cost_usd > 0
    # На каждую полосу — папка с одной записью этапа: request.json без байтов картинок, response.json с вердиктом.
    entries = sorted(p for p in (cache_dir / ISSUE).glob("*/*") if p.is_dir())
    assert [e.name.split(".")[0] for e in entries] == ["toc", "page", "page", "page"]
    request = json.loads((entries[0] / REQUEST_FILE).read_text(encoding="utf-8"))
    assert (
        request["stage"] == "toc"
        and request["toc_kind_expected"] == "contents"
        and request["tiles"][0].startswith("sha256:")
    )
    assert "data:image" not in json.dumps(request["payload"])
    response = json.loads((entries[0] / RESPONSE_FILE).read_text(encoding="utf-8"))
    assert response["verdict"] == CacheVerdict.OK and response["cost_usd"] == 0.001
    meta = json.loads((params.out_dir / ISSUE / "IMG_0002.meta.json").read_text(encoding="utf-8"))
    assert meta["cache_hit"] is False and meta["cache_entry"].startswith(f"{ISSUE}/IMG_0002/page.")

    # Второй прогон БЕЗ --skip-done: все четыре запроса находятся в кэше, сеть не нужна, стоимость 0.
    again = FakeClient(lambda p: (_ for _ in ()).throw(AssertionError("запросов быть не должно")))
    params2 = _params(tmp_path, on_missed_toc="skip")
    params2.options.cache_dir = cache_dir
    stats2 = run_pipeline(again, resolve("deepseek-v41-flash"), params2)
    assert (stats2.requests, stats2.cache_hits, stats2.failed) == (4, 4, 0) and stats2.cost_usd == 0
    meta = json.loads((params.out_dir / ISSUE / "IMG_0002.meta.json").read_text(encoding="utf-8"))
    assert meta["cache_hit"] is True and meta["cost_usd"] == 0.001, "цена в meta историческая"
    summary = (params.out_dir / "summary.csv").read_text(encoding="utf-8")
    assert "cache_hit" in summary.splitlines()[0] and ",True," in summary


def test_demoted_page_keeps_both_passes_in_cache(tmp_path):
    _make_pages(tmp_path / "in")
    fake = FakeClient(lambda p: None)
    fake.reply = DemotingReply(fake)
    order = iter(["IMG_0001", "IMG_0002", "IMG_0003"])
    fake.page_of = lambda payload: next(order, "IMG_0009")
    cache_dir = tmp_path / "cache"
    params = PipelineParams(
        tmp_path / "in",
        tmp_path / "out",
        RunOptions(cache_dir=cache_dir),
        _flags(toc=("IMG_0001", "IMG_0002", "IMG_0003")),
        jobs=1,
        on_missed_toc="skip",
    )
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    assert stats.requests == 6
    # Понижённая полоса: запись этапа toc и запись этапа page рядом, обе с ответом.
    for stem in ("IMG_0002", "IMG_0003"):
        stages = sorted(p.name.split(".")[0] for p in (cache_dir / ISSUE / stem).iterdir() if p.is_dir())
        assert stages == ["page", "toc"], stem
        assert all((p / RESPONSE_FILE).is_file() for p in (cache_dir / ISSUE / stem).iterdir() if p.is_dir())
    assert sorted(p.name.split(".")[0] for p in (cache_dir / ISSUE / "IMG_0001").iterdir()) == ["toc"]


def test_format_echo_is_cached_with_verdict_and_replayed(tmp_path):
    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    answers = ['{"type": "json_object"}', _answer()]
    fake = FakeClient(lambda p: answers.pop(0))
    options = RunOptions(cache_dir=tmp_path / "cache")
    spec = resolve("deepseek-v41-flash")
    meta, result = recognize_page(fake, spec, tmp_path / "in" / rel, PageJob(rel), tmp_path / "out", options)
    assert result is not None and meta["json_mode_used"] == "none" and len(fake.payloads) == 2
    entries = {p.name.split(".")[1]: p for p in (tmp_path / "cache" / ISSUE / "IMG_0002").iterdir()}
    assert set(entries) == {"json_object", "none"}
    echo = json.loads((entries["json_object"] / RESPONSE_FILE).read_text(encoding="utf-8"))
    assert echo["verdict"] == CacheVerdict.FORMAT_ECHO
    # Повтор: эхо из кэша остаётся эхом, цепочка идёт к следующему режиму — тоже из кэша; запросов нет.
    again = FakeClient(lambda p: (_ for _ in ()).throw(AssertionError("запросов быть не должно")))
    meta2, result2 = recognize_page(again, spec, tmp_path / "in" / rel, PageJob(rel), tmp_path / "out", options)
    assert result2 is not None and meta2["cache_hit"] is True and meta2["json_mode_used"] == "none"
    assert meta2["fallbacks"][0]["error"] == "эхо response_format" and meta2["cost_usd_wasted"] == 0.001


def test_network_error_is_not_cached_but_logged(tmp_path):
    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    fake = FakeClient(lambda p: OpenRouterError("HTTP 500", 500, "boom"))
    options = RunOptions(cache_dir=tmp_path / "cache")
    spec = resolve("deepseek-v41-flash")
    meta, result = recognize_page(fake, spec, tmp_path / "in" / rel, PageJob(rel), tmp_path / "out", options)
    assert result is None and meta["error"]
    page_dir = tmp_path / "cache" / ISSUE / "IMG_0002"
    assert not [p for p in page_dir.iterdir() if p.is_dir()], "записи с ответом нет"
    lines = (page_dir / ERRORS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["body"] == "boom"
    # После ошибки тот же запрос уходит в сеть снова.
    ok = FakeClient(lambda p: _answer())
    meta2, result2 = recognize_page(ok, spec, tmp_path / "in" / rel, PageJob(rel), tmp_path / "out", options)
    assert result2 is not None and meta2["cache_hit"] is False and len(ok.payloads) == 1
