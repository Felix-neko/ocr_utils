"""Сверка ответа модели с tesseract: мера троек слов, отбор полос по meta, переиспользование fidelity.csv."""

import json
from pathlib import Path

from ocr_utils.external_ocr_services import fidelity
from ocr_utils.external_ocr_services.fidelity import (
    MIN_WORDS,
    PageFidelity,
    check_pages,
    md_body,
    measure_words,
    page_jobs,
    read_report,
    shingle_share,
    summarize,
    write_report,
)

# Связный текст полосы и его пересказ: слова те же, порядок и связки другие.
READ = " ".join(f"поставщик обязан восполнить недопоставку продукции номер{i:03d}" for i in range(12))


def test_measure_words_drops_short_words_and_markup():
    words = measure_words("<rubric>*НАШ СТОЛ СПРАВОК*</rubric> Ёлочная продукция, т. е. до их исполнения")
    assert words == ["справок", "елочная", "продукция", "исполнения"]


def test_shingle_share_reading_vs_retelling():
    assert shingle_share(READ, READ.replace("номер", "-\nномер")) == (len(measure_words(READ)), 1.0)
    retold = " ".join(reversed(READ.split()))
    words, share = shingle_share(retold, READ)
    assert words == len(measure_words(READ)) and share == 0.0
    # Мало слов — меры нет, полоса не подозрительна
    assert shingle_share("поставщик обязан восполнить", READ) == (3, None)
    assert len(measure_words(READ)) >= MIN_WORDS


def test_md_body_skips_front_matter(tmp_path):
    md = tmp_path / "a.md"
    md.write_text('---\ntitle: "Заголовок статьи"\n---\nТело полосы\n', encoding="utf-8")
    assert md_body(md) == "Тело полосы\n"
    md.write_text("Без шапки", encoding="utf-8")
    assert md_body(md) == "Без шапки"


def _page(pages_dir: Path, rel: str, image: str, text: str) -> Path:
    md = pages_dir / rel
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(f'---\ntoc_kind: "none"\n---\n{text}\n', encoding="utf-8")
    md.with_name(md.stem + ".meta.json").write_text(json.dumps({"page": image}), encoding="utf-8")
    return md


def test_page_jobs_take_image_from_meta_and_filter(tmp_path):
    _page(tmp_path, "1977/02/0430_1L.md", "1977/02/0430_1L.57bcb874.tif", READ)
    _page(tmp_path, "1977/03/0010_1L.md", "1977/03/0010_1L.aa.tif", READ)
    # Полоса со сбоем: meta есть, md нет — не сверяется
    (tmp_path / "1977/03/0020_1L.meta.json").write_text(json.dumps({"page": "1977/03/0020_1L.tif"}))
    jobs = page_jobs(tmp_path)
    assert [(md.relative_to(tmp_path).as_posix(), image) for md, image in jobs] == [
        ("1977/02/0430_1L.md", "1977/02/0430_1L.57bcb874.tif"),
        ("1977/03/0010_1L.md", "1977/03/0010_1L.aa.tif"),
    ]
    assert len(page_jobs(tmp_path, only_issue="02")) == 1


def test_check_pages_reuses_unchanged_rows_and_keeps_other_issues(tmp_path, monkeypatch):
    pages_dir, in_dir = tmp_path / "pages", tmp_path / "in"
    good = _page(pages_dir, "1977/01/0010_1L.md", "1977/01/0010_1L.x.tif", READ)
    bad = _page(pages_dir, "1977/01/0020_1L.md", "1977/01/0020_1L.x.tif", " ".join(reversed(READ.split())))
    # Старая строка другого выпуска должна пережить частичный прогон
    write_report(pages_dir / fidelity.REPORT_NAME, [PageFidelity("1977/02/0010_1L.md", "img", 80, 0.9, 1)])
    calls: list[str] = []

    # tesseract подменён: «читает» эталонный текст, вызовы считаем
    def fake_check(pages_dir_, in_dir_, md_path, image):
        calls.append(image)
        words, share = shingle_share(md_body(md_path), READ)
        return PageFidelity(md_path.relative_to(pages_dir_).as_posix(), image, words, share, md_path.stat().st_mtime_ns)

    monkeypatch.setattr(fidelity, "check_page", fake_check)
    monkeypatch.setattr(fidelity.tesseract, "tesseract_available", lambda: True)
    monkeypatch.setattr(fidelity, "ProcessPoolExecutor", _InlineExecutor)
    rows = check_pages(pages_dir, in_dir, jobs=2, only_issue="01")
    assert {row.page: row.suspect for row in rows} == {"1977/01/0010_1L.md": False, "1977/01/0020_1L.md": True}
    report = read_report(pages_dir / fidelity.REPORT_NAME)
    assert set(report) == {"1977/01/0010_1L.md", "1977/01/0020_1L.md", "1977/02/0010_1L.md"}
    assert "0020_1L.md (0.00)" in summarize(rows)

    # Повтор без изменений — tesseract не зовётся; правка md — пересчёт только этой полосы
    calls.clear()
    check_pages(pages_dir, in_dir, only_issue="01")
    assert calls == []
    bad.write_text(good.read_text(encoding="utf-8"), encoding="utf-8")
    rows = check_pages(pages_dir, in_dir, only_issue="01")
    assert calls == ["1977/01/0020_1L.x.tif"] and not any(row.suspect for row in rows)


class _InlineExecutor:
    """Подмена пула процессов: задачи выполняются сразу в этом процессе (подменённая check_page видна)."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        return _Done(fn(*args))


class _Done:
    """Готовый результат в обёртке future."""

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value
