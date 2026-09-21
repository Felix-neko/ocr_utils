"""Стык полос: строки по проекции чернил, полоски, признаки сомнения, вердикт модели, проверка с кэшем."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ocr_utils.external_ocr_services.assemble import assemble_issue, issue_pages
from ocr_utils.external_ocr_services.boundary import (
    BoundaryChecker,
    BoundaryVerdict,
    DoubtReason,
    JoinKind,
    Seam,
    VerdictStatus,
    apply_verdict,
    decide_boundary,
    doubt_reason,
    parse_verdicts,
    strip_of,
    text_line_bands,
)
from ocr_utils.external_ocr_services.cache import RESPONSE_FILE
from ocr_utils.external_ocr_services.hyphen_join import default_morph
from ocr_utils.external_ocr_services.models import resolve
from tests.ocr_utils.external_ocr_services.test_assemble import ISSUE, _page
from tests.ocr_utils.external_ocr_services.test_pipeline import FakeClient

LINE_H, GAP = 14, 22  # «строка» — чёрная полоса 14 px, шаг 36 px


def _page_image(path: Path, lines: int = 8, header: bool = True, width: int = 800, height: int = 1200) -> list[int]:
    """Страница-синтетика: колонтитул сверху, затем ``lines`` строк «текста» (тёмные бруски со словами-
    промежутками), низ пустой. Возвращает верхние края строк."""
    image = Image.new("L", (width, height), 235)
    draw = ImageDraw.Draw(image)
    tops = []
    y = 60
    if header:
        draw.rectangle((300, y, 500, y + 8), fill=40)
        y += 40
    for _ in range(lines):
        tops.append(y)
        # «слова» — бруски с пробелами, чтобы доля чернил была как у текста, а не сплошная линейка
        for x in range(80, 720, 60):
            draw.rectangle((x, y, x + 45, y + LINE_H), fill=30)
        y += LINE_H + GAP
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return tops


def test_text_line_bands_and_strips(tmp_path):
    path = tmp_path / "in" / ISSUE / "IMG_0001.jpg"
    tops = _page_image(path)
    gray = np.asarray(Image.open(path).convert("L"))
    bands = text_line_bands(gray)
    assert len(bands) == 9, bands  # колонтитул + 8 строк
    assert all(abs(top - expected) <= 3 for (top, _), expected in zip(bands[1:], tops))
    tail = strip_of(path, at_bottom=True)
    head = strip_of(path, at_bottom=False)
    assert tail.lines == 3 and head.lines == 3
    # Хвост — три последние строки с запасом в полстроки; голова — колонтитул и две первые строки.
    assert (
        tail.box[1] <= tops[-3] and tail.box[3] >= tops[-1] + LINE_H and tail.box[3] - tail.box[1] < 5 * (LINE_H + GAP)
    )
    assert head.box[1] < tops[0] and head.box[3] >= tops[1] + LINE_H
    # По ширине обрезано по чернилам (± 2 %), картинка — JPEG не шире 2200 px.
    assert 60 <= tail.box[0] <= 80 and 720 <= tail.box[2] <= 745
    assert tail.image.width <= 2200 and tail.image.data[:2] == b"\xff\xd8"
    # Пустая страница — запасной вырез в 12 % высоты.
    blank = tmp_path / "in" / ISSUE / "IMG_0002.jpg"
    Image.new("L", (800, 1200), 235).save(blank)
    empty = strip_of(blank, at_bottom=True)
    assert empty.lines == 0 and empty.box == (0, 1200 - 144, 800, 1200)


def test_doubt_reasons():
    morph = default_morph()
    cases = {
        ("были направлена", "ны на смотр."): DoubtReason.DUPLICATE,
        ("посылке заказа почтой", "тратится несколько дней."): None,  # словарное слово в голове
        # «затрачивается» словарю pymorphy3 неизвестно — дыра словаря, стык уйдёт модели (дёшево).
        ("посылке заказа почтой", "затрачивается несколько дней."): DoubtReason.FRAGMENT_HEAD,
        ("только один Дружковский", "метизный завод начал."): None,  # прописная — не фрагмент
        ("текст без точки", "ыыы дальше."): DoubtReason.FRAGMENT_HEAD,
        ("текст кончается на кргл", "дальше идёт."): DoubtReason.FRAGMENT_TAIL,
        ("кончается на кргл.", "дальше идёт."): None,  # после точки — новое предложение
        ("только «Союзлифтмаш» Минстройдор-", "маша на упаковку."): DoubtReason.COMPOUND_UNKNOWN,
        ("форму снаб-", "жения и далее."): None,  # перенос по словарю сомнений не вызывает
    }
    for (tail, head), expected in cases.items():
        decision = decide_boundary(tail, head, morph)
        assert doubt_reason(tail, head, decision, morph) is expected, (tail, head, decision)
    assert doubt_reason("были направлена", "ны на смотр.", None, None) is None, "без словаря сомнений нет"


def test_parse_and_apply_verdict():
    morph = default_morph()
    parsed = parse_verdicts(
        '```json\n{"seams": [{"seam": 1, "tail_word": "направле-", "head_word": "ны", "same_word": true, '
        '"joined": "направлены"}]}\n```'
    )
    verdict = parsed[1]
    assert verdict.same_word and verdict.joined == "направлены" and verdict.notes == ""
    # Один объект без списка (модель ответила на единственный стык) и запись без номера — тоже читаются.
    assert parse_verdicts('{"tail_word": "а", "head_word": "б", "same_word": false}')[1].head_word == "б"
    assert set(parse_verdicts('{"seams": [{"tail_word": "а"}, {"tail_word": "б"}]}')) == {1, 2}
    # Одно слово: последний токен хвоста и первый головы заменяются словом модели.
    tail, head = "плакаты были направлена", "ны на смотр."
    decision = decide_boundary(tail, head, morph)  # эвристика уже дала «направлены»
    same, status = apply_verdict(tail, head, verdict, decision)
    assert status is VerdictStatus.CONFIRMED and same is decision, "модель подтвердила эвристику"
    verdict.joined = "направлено"
    other, status = apply_verdict(tail, head, verdict, decision)
    assert status is VerdictStatus.REWRITTEN and other.kind is JoinKind.MODEL
    assert other.text == "плакаты были направлено на смотр."
    # Не одно слово: слова те же — подтверждение сшивки через пробел; слова прочитаны иначе — подстановка.
    verdict = BoundaryVerdict(tail_word="почтой", head_word="затрачивается", same_word=False)
    kept, status = apply_verdict("заказа почтой", "затрачивается несколько дней.", verdict, None)
    assert status is VerdictStatus.CONFIRMED and kept is None
    verdict = BoundaryVerdict(tail_word="почтой", head_word="затрачивается", same_word=False)
    fixed, status = apply_verdict("заказа почтой", "затрачиваетсся несколько дней.", verdict, None)
    assert status is VerdictStatus.REWRITTEN and fixed.text == "заказа почтой затрачивается несколько дней."
    assert fixed.kind is JoinKind.MODEL and fixed.text[fixed.head_start :].startswith("затрачивается")
    # Опечатка транскрипции в короткой половине («ритым» ↔ «ритным»): слова похожи, вердикт применяется.
    typo = BoundaryVerdict(tail_word="малогаба-", head_word="ритным", same_word=True, joined="малогабаритным")
    fixed, status = apply_verdict("для малогаба-", "ритым контейнеров.", typo, None)
    assert status is VerdictStatus.REWRITTEN and fixed.text == "для малогабаритным контейнеров."
    # Модель прочла чужую строку (подпись к рисунку): слова не похожи на слова стыка — отклоняется.
    stray = BoundaryVerdict(tail_word="кабеля", head_word="нием", same_word=False)
    assert apply_verdict("с соответствующим сокращением", "нием затрат.", stray, None) == (None, VerdictStatus.REJECTED)
    # Битый ответ или теги на стыке — эвристика остаётся.
    assert apply_verdict(tail, head, BoundaryVerdict(error="сбой"), decision) == (decision, VerdictStatus.ERROR)
    tagged = "плакаты были <supplied>направлена</supplied>"
    rejected = apply_verdict(tagged, head, BoundaryVerdict(same_word=True, joined="направлены"), None)
    assert rejected == (None, VerdictStatus.REJECTED)


class _Reply:
    """Ответ фейкового клиента на запрос по стыкам: отвечает заданным JSON."""

    def __init__(self, answer: str):
        self.answer = answer

    def __call__(self, payload):
        return self.answer


def _seams(*pairs: tuple[str, str, str, str, DoubtReason]) -> list[Seam]:
    return [
        Seam(i, Path(ISSUE) / f"{a}.json", Path(ISSUE) / f"{b}.json", t, h, r)
        for i, (a, b, t, h, r) in enumerate(pairs, 1)
    ]


def test_checker_sends_one_request_per_issue_and_uses_cache(tmp_path):
    in_dir = tmp_path / "in"
    for name in ("IMG_0001", "IMG_0002", "IMG_0003"):
        _page_image(in_dir / ISSUE / f"{name}.jpg", header=name != "IMG_0001")
    answer = (
        '{"seams": [{"seam": 1, "tail_word": "направле-", "head_word": "ны", "same_word": true, "joined": "направлены"}, '
        '{"seam": 2, "tail_word": "точки", "head_word": "и", "same_word": false, "joined": ""}]}'
    )
    fake = FakeClient(_Reply(answer))
    checker = BoundaryChecker(fake, resolve("deepseek-v41-flash"), in_dir, tmp_path / "cache")
    seams = _seams(
        ("IMG_0001", "IMG_0002", "плакаты были направлена", "ны на смотр.", DoubtReason.DUPLICATE),
        ("IMG_0002", "IMG_0003", "Конец без точки", "ыыы дальше.", DoubtReason.FRAGMENT_HEAD),
        ("IMG_0003", "IMG_0009", "нет файла", "ы", DoubtReason.FRAGMENT_HEAD),  # полосы нет во входе
    )
    verdicts = checker.check_issue(ISSUE, seams)
    assert set(verdicts) == {1, 2}, "стык без файла полосы пропущен"
    assert verdicts[1].same_word and verdicts[1].joined == "направлены" and verdicts[2].head_word == "и"
    assert not verdicts[1].cache_hit and verdicts[1].cost_usd == 0.0005, "цена запроса делится между стыками"
    assert len(fake.payloads) == 1, "один запрос на выпуск"
    payload = fake.payloads[0]
    parts = payload["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text"] + ["image_url"] * 4, "по две полоски на стык"
    text = parts[0]["text"]
    assert "2 seam(s)" in text and "Seam 1 (IMG_0001 → IMG_0002)" in text and "«плакаты были направлена»" in text
    assert "Seam 2" in text and "fragment head" in text and "IMG_0009" not in text
    assert payload["response_format"] == {"type": "json_object"} and payload["max_tokens"] == 800
    assert "seams" in payload["messages"][0]["content"]
    entry = tmp_path / "cache" / ISSUE / "_boundaries" / verdicts[1].cache_entry.split("/")[-1]
    assert entry.name.startswith("boundary.json_object.") and (entry / RESPONSE_FILE).is_file()
    request = json.loads((entry / "request.json").read_text(encoding="utf-8"))
    assert [s["reason"] for s in request["seams"]] == ["duplicate", "fragment_head"]
    assert request["seams"][0]["tail_strip"]["lines"] == 3 and len(request["tiles"]) == 4
    # Повтор — из кэша, без запроса; другой набор стыков — новый запрос.
    again = checker.check_issue(ISSUE, seams)
    assert again[1].cache_hit and again[1].joined == "направлены" and len(fake.payloads) == 1
    checker.check_issue(ISSUE, seams[:1])
    assert len(fake.payloads) == 2
    assert (checker.stats.requests, checker.stats.cache_hits, checker.stats.errors) == (3, 1, 0)
    # Битый ответ — вердикт с ошибкой у всех стыков пачки.
    bad = BoundaryChecker(FakeClient(_Reply("не JSON")), resolve("deepseek-v41-flash"), in_dir, None)
    broken = bad.check_issue(ISSUE, seams[:2])
    assert broken[1].error and broken[2].error and bad.stats.errors == 1


def test_assemble_with_checker_applies_verdict_and_records(tmp_path):
    in_dir = tmp_path / "in"
    for name in ("IMG_0001", "IMG_0002", "IMG_0003"):
        _page_image(in_dir / ISSUE / f"{name}.jpg")
    _page(tmp_path, "IMG_0001", "плакаты были направлена")
    _page(tmp_path, "IMG_0002", "ны на заключительный смотр. Конец без точки")
    _page(tmp_path, "IMG_0003", "продолжениее дальше.")  # опечатка транскрипции — слово вне словаря
    answer = (
        '{"seams": [{"seam": 1, "tail_word": "направле-", "head_word": "ны", "same_word": true, "joined": "направлены"}, '
        '{"seam": 2, "tail_word": "точки", "head_word": "продолжение", "same_word": false, "joined": ""}]}'
    )
    fake = FakeClient(_Reply(answer))
    checker = BoundaryChecker(fake, resolve("deepseek-v41-flash"), in_dir, tmp_path / "cache")
    assembly = assemble_issue(tmp_path, ISSUE, issue_pages(tmp_path, ISSUE), checker=checker)
    assert len(fake.payloads) == 1, "оба сомнительных стыка — одним запросом"
    assert "были направлены на заключительный смотр." in assembly.text
    assert "Конец без точки продолжение дальше." in assembly.text, "модель прочла слово без опечатки"
    first, second = assembly.joins
    assert first.kind is JoinKind.DUPLICATE and first.reason is DoubtReason.DUPLICATE
    assert first.model["status"] == "confirmed"
    assert second.kind is JoinKind.MODEL and second.reason is DoubtReason.FRAGMENT_HEAD
    assert second.model["head_word"] == "продолжение" and second.model["status"] == "rewritten"
    assert assembly.checked.requests == 1 and assembly.checked.cost_usd == 0.001
    sidecar = assembly.as_dict()
    assert sidecar["joins"][1]["reason"] == "fragment_head" and sidecar["checked"]["requests"] == 1
    # Смещение третьей полосы — начало слова модели.
    assert assembly.text[assembly.pages[2].offset :].startswith("продолжение дальше.")
    # Без checker — только эвристики: опечатка остаётся, записей о проверке нет, запросов нет.
    plain = assemble_issue(tmp_path, ISSUE, issue_pages(tmp_path, ISSUE))
    assert "продолжениее дальше." in plain.text and all(j.reason is None for j in plain.joins)
    # Выпуск без сомнительных стыков — запроса нет.
    _page(tmp_path, "IMG_0001", "плакаты были направлены")
    _page(tmp_path, "IMG_0002", "на заключительный смотр. Конец без точки")
    _page(tmp_path, "IMG_0003", "продолжение.")
    fresh = FakeClient(_Reply(answer))
    clean = assemble_issue(
        tmp_path,
        ISSUE,
        issue_pages(tmp_path, ISSUE),
        checker=BoundaryChecker(fresh, resolve("deepseek-v41-flash"), in_dir, None),
    )
    assert not fresh.payloads and clean.checked.requests == 0 and "были направлены на заключительный" in clean.text


def test_run_pipeline_checks_boundaries_once_per_issue(tmp_path):
    from ocr_utils.external_ocr_services.pipeline import PipelineParams, run_pipeline
    from ocr_utils.external_ocr_services.ocr import RunOptions
    from tests.ocr_utils.external_ocr_services.test_pipeline import _answer, _flags, _make_pages, _toc_answer

    _make_pages(tmp_path / "in")
    bodies = {"IMG_0002": "плакаты были направлена", "IMG_0003": "ны на смотр. Конец", "IMG_0004": "и всё."}
    order = iter(["IMG_0002", "IMG_0003", "IMG_0004"])
    answer = '{"seams": [{"seam": 1, "tail_word": "направле-", "head_word": "ны", "same_word": true, "joined": "направлены"}]}'

    def reply(payload):
        system = payload["messages"][0]["content"]
        if "seams" in system:
            return answer
        if fake.stage_of(payload) == "toc":
            return _toc_answer(["Первые шаги"])
        return _answer(body=bodies[next(order)])

    fake = FakeClient(reply)
    params = PipelineParams(
        tmp_path / "in",
        tmp_path / "out",
        RunOptions(cache_dir=tmp_path / "cache"),
        _flags(),
        jobs=1,
        on_missed_toc="skip",
        issues_dir=tmp_path / "issues",
    )
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    boundary_payloads = [p for p in fake.payloads if "seams" in p["messages"][0]["content"]]
    assert len(boundary_payloads) == 1 and stats.boundary_requests == 1 and stats.boundary_rewritten == 0
    assert stats.cost_usd == 0.005, "4 полосы + 1 запрос по стыкам"
    text = (tmp_path / "issues" / "1966" / "1966_03.md").read_text(encoding="utf-8")
    assert "были направлены на смотр." in text
    sidecar = json.loads((tmp_path / "out" / "1966" / "1966_03.pages.json").read_text(encoding="utf-8"))
    assert sidecar["joins"][0]["reason"] == "duplicate" and sidecar["joins"][0]["model"]["status"] == "confirmed"
    assert (tmp_path / "cache" / ISSUE / "_boundaries").is_dir()
    # Без проверки — запроса по стыкам нет, эвристика та же.
    fake.payloads.clear()
    order = iter(["IMG_0002", "IMG_0003", "IMG_0004"])
    params = PipelineParams(
        tmp_path / "in",
        tmp_path / "out2",
        RunOptions(),
        _flags(),
        jobs=1,
        on_missed_toc="skip",
        check_boundaries=False,
        issues_dir=tmp_path / "issues2",
    )
    stats = run_pipeline(fake, resolve("deepseek-v41-flash"), params)
    assert stats.boundary_requests == 0 and not [p for p in fake.payloads if "seams" in p["messages"][0]["content"]]
    assert "были направлены на смотр." in (tmp_path / "issues2" / "1966" / "1966_03.md").read_text(encoding="utf-8")
