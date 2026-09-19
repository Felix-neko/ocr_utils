"""Сборка выпуска: переносы и абзацы через границу полос, плавающие блоки, пропуски, sidecar, команда CLI."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from ocr_utils.external_ocr_services import cli
from ocr_utils.external_ocr_services.assemble import JoinKind, assemble_issue, issue_pages, list_issues, write_issue
from ocr_utils.external_ocr_services.schema import PageResult, Stage

ISSUE = "1966/03"


def _page(out_dir: Path, stem: str, body: str, page_number: str | None = None, stage: Stage = Stage.PAGE) -> Path:
    """Готовая полоса на диске: .json как пишет recognize_page и meta без ошибки."""
    base = out_dir / ISSUE / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(
        PageResult(content_markdown=body, page_number=page_number).to_json(), encoding="utf-8"
    )
    base.with_suffix(".meta.json").write_text(
        json.dumps({"page": f"{ISSUE}/{stem}.jpg", "stage": stage.value, "error": None, "parse_error": None}),
        encoding="utf-8",
    )
    return Path(ISSUE) / f"{stem}.jpg"


def _assemble(out_dir: Path, **kwargs):
    rels = issue_pages(out_dir, ISSUE)
    return assemble_issue(out_dir, ISSUE, rels, **kwargs)


def _page_text(assembly, index: int, length: int = 40) -> str:
    """Текст выпуска от начала полосы номер index."""
    offset = assembly.pages[index].offset
    return assembly.text[offset : offset + length]


def test_hyphen_across_pages_with_tags_and_compound(tmp_path):
    _page(tmp_path, "IMG_0001", "# Статья\n\nУправления будут определять форму <supplied>снаб-</supplied>", "2")
    _page(tmp_path, "IMG_0002", "жения (транзитную или складскую) и далее.\n\nСвязи торгово-", "3")
    _page(tmp_path, "IMG_0003", "<unclear>экономических</unclear> стран крепнут.", "4")
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert "форму <supplied>снаб</supplied>жения (транзитную" in text, "дефис убран, теги на месте"
    assert "торгово-<unclear>экономических</unclear> стран" in text, "составное — с дефисом, без пробела"
    assert [(j.kind, j.word) for j in assembly.joins] == [
        (JoinKind.HYPHEN, "снаб-жения"),
        (JoinKind.COMPOUND, "торгово-экономических"),
    ]
    assert [p.page_number for p in assembly.pages] == ["2", "3", "4"]
    # Смещения: полоса 2 начинается с «жения» внутри сшитого абзаца, полоса 3 — с тега.
    assert _page_text(assembly, 1).startswith("жения (транзитную")
    assert _page_text(assembly, 2).startswith("<unclear>экономических</unclear>")
    assert text.startswith('---\nyear: "1966"\nissue: "03"\npages: 3\nmissing: 0\n---\n')
    for entry in assembly.pages:
        assert text.count("\n", 0, entry.offset) + 1 == entry.line


def test_paragraph_joined_only_when_sentence_is_cut(tmp_path):
    _page(tmp_path, "IMG_0001", "В значительной мере этому")
    _page(tmp_path, "IMG_0002", "будут способствовать договоры. Конец абзаца.")
    _page(tmp_path, "IMG_0003", "Новый абзац с большой буквы.\n\nВсего 35 тыс.")
    _page(tmp_path, "IMG_0004", "автомашин в год.")
    _page(tmp_path, "IMG_0005", "# Заголовок новой статьи\n\nТекст.")
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert "этому будут способствовать" in text
    assert "Конец абзаца.\n\nНовый абзац" in text, "после точки — отдельный абзац"
    assert "35 тыс. автомашин" in text, "точка после сокращения — не конец предложения"
    assert "год.\n\n# Заголовок" in text
    assert [j.kind for j in assembly.joins] == [JoinKind.PARAGRAPH, JoinKind.PARAGRAPH]
    # Выключенная склейка абзацев: полосы остаются отдельными абзацами, переносы не трогаются.
    plain = _assemble(tmp_path, join_paragraphs_across=False)
    assert "этому\n\nбудут" in plain.text and not plain.joins


def test_floating_blocks_are_skipped_and_kept_in_place(tmp_path):
    _page(
        tmp_path,
        "IMG_0001",
        "Норма установлена для перевозок. Кроме того, грузоподъемность\n\n<footnote>[^1]: Сноска.</footnote>",
    )
    _page(
        tmp_path,
        "IMG_0002",
        "Таблица 3\n\nНазвание таблицы\n\n<table>\n<tr><td>1</td></tr>\n</table>\n\n*Примечание.* К таблице.\n\n"
        "автомобилей ниже. Далее текст.",
    )
    _page(tmp_path, "IMG_0003", "```\n[фотография]\nсклад\n```\n\nРис. 2. Склад.\n\nПоследний абзац без точки")
    _page(tmp_path, "IMG_0004", "[^2]: Голая сноска.\n\nи его продолжение.")
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert "грузоподъемность автомобилей ниже. Далее текст." in text
    assert (
        "Далее текст.\n\n<footnote>[^1]: Сноска.</footnote>\n\nТаблица 3\n\nНазвание таблицы\n\n<table>" in text
    ), "сноска за абзацем, таблица следом"
    assert "</table>\n\n*Примечание.* К таблице.\n\n```" in text
    assert "без точки и его продолжение." in text
    assert text.index("[^2]: Голая сноска.") > text.index("и его продолжение.")
    assert [j.kind for j in assembly.joins] == [JoinKind.PARAGRAPH, JoinKind.PARAGRAPH]


def test_toc_blocks_rubrics_and_lists_do_not_join(tmp_path):
    _page(
        tmp_path,
        "IMG_0001",
        "# СОДЕРЖАНИЕ\n\n<toc>\n- Автор. Статья — 5\n\n- Автор. Вторая — 7\n</toc>",
        stage=Stage.TOC,
    )
    _page(tmp_path, "IMG_0002", "- продолжение списка — 9\n\nТекст без точки")
    _page(tmp_path, "IMG_0003", "<rubric>*РУБРИКА*</rubric>\n\n# Статья\n\n<author>**И. Иванов**</author>\n\nтекст.")
    assembly = _assemble(tmp_path)
    assert not assembly.joins
    assert assembly.pages[0].stage == "toc" and "</toc>\n\n- продолжение" in assembly.text
    assert "Текст без точки\n\n<rubric>" in assembly.text


def test_duplicate_half_word_is_dropped(tmp_path):
    _page(tmp_path, "IMG_0001", "с соответствующим сокращением")
    _page(tmp_path, "IMG_0002", "нием эксплуатационных затрат.")
    assembly = _assemble(tmp_path)
    assert "сокращением эксплуатационных затрат." in assembly.text
    assert [(j.kind, j.word) for j in assembly.joins] == [(JoinKind.DUPLICATE, "сокращением+нием→сокращением")]
    assert _page_text(assembly, 1).startswith("эксплуатационных")


def test_duplicate_wrong_guess_mirror_and_identical(tmp_path):
    """Три вида дубля достроенного слова: неверная догадка, зеркальный, обе стороны одинаково."""
    _page(tmp_path, "IMG_0001", "плакаты были направлена")  # модель достроила «направле-» неверно
    _page(tmp_path, "IMG_0002", "ны на заключительный смотр. Вторая полоса кончается направле-")
    _page(tmp_path, "IMG_0003", "направлены дальше. Третья кончается словом направлены")  # голова достроена
    _page(tmp_path, "IMG_0004", "направлены в конец.")  # обе стороны достроены одинаково
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert "плакаты были направлены на заключительный смотр." in text
    assert "кончается направлены дальше." in text and "направле-" not in text
    assert "словом направлены в конец." in text and text.count("направлены") == 3
    assert [(j.kind, j.word) for j in assembly.joins] == [
        (JoinKind.DUPLICATE, "направлена+ны→направлены"),
        (JoinKind.DUPLICATE, "направле-+направлены"),
        (JoinKind.DUPLICATE, "направлены=направлены"),
    ]
    assert _page_text(assembly, 1).startswith("на заключительный") and _page_text(assembly, 2).startswith(
        "направлены дальше"
    )


def test_missing_page_leaves_comment_and_sidecar_lists_it(tmp_path):
    _page(tmp_path, "IMG_0001", "Первая полоса без точки")
    rel = _page(tmp_path, "IMG_0002", "вторая.")
    (tmp_path / rel).with_suffix(".meta.json").write_text(
        json.dumps({"stage": "page", "error": "запрос не удался", "parse_error": None}), encoding="utf-8"
    )
    _page(tmp_path, "IMG_0003", "третья без точки")
    assembly = _assemble(tmp_path)
    assert "<!-- полоса 1966/03/IMG_0002 не распознана: запрос не удался -->" in assembly.text
    assert assembly.missing == [{"file": "1966/03/IMG_0002", "reason": "запрос не удался"}]
    assert not assembly.joins, "через пропущенную полосу ничего не сшивается"
    assert "missing: 1\n" in assembly.text
    path = write_issue(tmp_path, assembly)
    assert path == tmp_path / "1966" / "03.md" and path.read_text(encoding="utf-8") == assembly.text
    sidecar = json.loads((tmp_path / "1966" / "03.pages.json").read_text(encoding="utf-8"))
    assert sidecar["counts"] == {kind.value: 0 for kind in JoinKind} and len(sidecar["pages"]) == 3
    assert sidecar["missing"] == assembly.missing and sidecar["pages"][2]["file"] == "1966/03/IMG_0003"
    # Повторная сборка воспроизводит файл байт в байт.
    assert write_issue(tmp_path, _assemble(tmp_path)).read_text(encoding="utf-8") == assembly.text


def test_assemble_command_lists_issues_and_writes_files(tmp_path):
    _page(tmp_path, "IMG_0001", "Текст первой полосы снаб-")
    _page(tmp_path, "IMG_0002", "жения продолжается.")
    (tmp_path / "1966" / "03" / "toc.json").write_text("{}", encoding="utf-8")  # служебный файл выпуска — не полоса
    assert list_issues(tmp_path) == [ISSUE] and list_issues(tmp_path, only_year="1970") == []
    assert [r.name for r in issue_pages(tmp_path, ISSUE)] == ["IMG_0001.json", "IMG_0002.json"]
    result = CliRunner().invoke(cli.main, ["assemble", "--out-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Выпусков собрано: 1; переносов через границу: 1" in result.output
    assert "снабжения продолжается." in (tmp_path / "1966" / "03.md").read_text(encoding="utf-8")
    # Без переносов: граница остаётся, абзацы не сшиваются (хвост с дефисом — не конец предложения, но
    # голова строчная, а дефис остаётся как есть).
    result = CliRunner().invoke(cli.main, ["assemble", "--out-dir", str(tmp_path), "--no-join-hyphens"])
    assert result.exit_code == 0 and "снаб- жения" in (tmp_path / "1966" / "03.md").read_text(encoding="utf-8")
