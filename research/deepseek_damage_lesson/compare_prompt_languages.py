"""Сравнение английских и русских промптов на мини-наборе повреждённых сканов.

Каждая страница мини-набора (``external_ocr_models/damaged/нарезанное по страницам``)
распознаётся дважды тем же ходом, что в ``ocr_damaged_page``: с боевыми английскими
промптами и с их русским переводом. Подсказки про страницы — из
``run_scripts/external_ocr_models/damaged_hints.txt`` (английские) и их перевод ниже.
Выход — ``выход/сравнение/{en,ru}/<раздел>/<страница>.*`` и сводка ``выход/сравнение.md``:
пометки, записи ``edge_words``, токены, цена, сходство текстов между языками и с боевым
прогоном ``deepseek-v8``.

Запуск: ``uv run python -m research.deepseek_damage_lesson.compare_prompt_languages``.
"""

from __future__ import annotations

import difflib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from research.deepseek_damage_lesson.ocr_damaged_page import HERE, recognize

DAMAGED = HERE.parent / "external_ocr_models" / "damaged"
PAGES_DIR = DAMAGED / "нарезанное по страницам"
HINTS_EN = HERE.parent.parent / "run_scripts" / "external_ocr_models" / "damaged_hints.txt"
REFERENCE_DIR = DAMAGED / "выход" / "deepseek-v8"  # последний боевой прогон — для сверки
OUT_DIR = HERE / "выход" / "сравнение"
REPORT = HERE / "выход" / "сравнение.md"
LANGS = ("en", "ru")
JOBS = 4  # параллельных запросов: это сеть, а не CPU

# Русские подсказки — перевод строк damaged_hints.txt, по разделам мини-набора.
HINTS_RU = {
    "часть букв в словах совсем закрыта корешком/IMG_0006_L.jpg": "Это ЛЕВАЯ страница туго переплетённого тома. ПРАВЫЕ концы строк уходят в корешок: последние одна-четыре буквы многих строк полностью скрыты, их не видно вовсе.",
    "часть букв в словах совсем закрыта корешком/IMG_0006_R.jpg": "Это ПРАВАЯ страница туго переплетённого тома. ЛЕВЫЕ начала строк уходят в корешок: первые одна-четыре буквы многих строк полностью скрыты, их не видно вовсе.",
    "часть букв в словах совсем закрыта корешком/IMG_0008_L.jpg": "Это ЛЕВАЯ страница туго переплетённого тома. ПРАВЫЕ концы строк уходят в корешок: последние одна-четыре буквы многих строк полностью скрыты, их не видно вовсе.",
    "часть букв в словах совсем закрыта корешком/IMG_0008_R.jpg": "Это ПРАВАЯ страница туго переплетённого тома. ЛЕВЫЕ начала строк уходят в корешок: первые одна-четыре буквы многих строк полностью скрыты, их не видно вовсе.",
    "буквы у корешка начинают расплываться/IMG_0051_L.jpg": "Это ЛЕВАЯ страница переплетённого тома. У корешка (ПРАВЫЕ концы строк) буквы не в фокусе и начинают расплываться: они видны, но ненадёжны. Ничего не скрыто; дефис в конце строки — обычный перенос.",
    "буквы у корешка начинают расплываться/IMG_0053_L.jpg": "Это ЛЕВАЯ страница переплетённого тома. У корешка (ПРАВЫЕ концы строк) буквы не в фокусе и начинают расплываться: они видны, но ненадёжны. Ничего не скрыто; дефис в конце строки — обычный перенос.",
    "буквы у корешка и сплющены и расплываются/IMG_0067_L.jpg": "Это ЛЕВАЯ страница переплетённого тома. У корешка (ПРАВЫЕ концы строк) последние буквы сплющены изгибом страницы и размыты: видны, но ненадёжны. Ничего не скрыто; дефис в конце строки — обычный перенос.",
    "буквы у корешка и сплющены и расплываются/IMG_0068_L.jpg": "Это ЛЕВАЯ страница переплетённого тома. У корешка (ПРАВЫЕ концы строк) последние буквы сплющены изгибом страницы и размыты: видны, но ненадёжны. Ничего не скрыто; дефис в конце строки — обычный перенос.",
    "буквы у корешка сплющены/0070_L.jpg": "Это ЛЕВАЯ страница переплетённого тома. У корешка (ПРАВЫЕ концы строк) последние буквы сильно сплющены по горизонтали изгибом страницы: видны, но ненадёжны. Ничего не скрыто; дефис в конце строки — обычный перенос.",
    "пересвет/IMG_0046_L.jpg": "Это ЛЕВАЯ страница переплетённого тома. Страница освещена неровно: часть областей пересвечена и выбелена, там формы букв бледные и легко путаются.",
}

_TAGS = re.compile(r"</?(restored|fuzzy)>|<unknown\s*/>")


def read_hints_en() -> dict[str, str]:
    """Файл подсказок: относительный путь<TAB>текст; строки с # — комментарии."""
    hints = {}
    for line in HINTS_EN.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rel, _tab, text = line.partition("\t")
        hints[rel.strip()] = text.strip()
    return hints


def plain(text: str) -> str:
    """Текст без тегов повреждений — чтобы сравнивать буквы, а не разметку."""
    return _TAGS.sub("", text)


def similarity(first: str, second: str) -> float:
    """Сходство двух текстов 0..1 (difflib по символам)."""
    return difflib.SequenceMatcher(None, plain(first), plain(second), autojunk=False).ratio()


def run_one(rel: str, hint: str, lang: str) -> dict:
    scan = PAGES_DIR / rel
    out_dir = OUT_DIR / lang / Path(rel).parent
    try:
        meta = recognize(scan, hint, out_dir, lang)
    except SystemExit as error:  # сбой запроса или разбора — строка сводки с ошибкой
        meta = {"error": str(error), "tags": {}, "tags_from_edge_words": 0}
    meta["rel"], meta["lang"] = rel, lang
    return meta


def load_result(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    hints_en = read_hints_en()
    jobs = [(rel, hints_en[rel], "en") for rel in hints_en] + [(rel, HINTS_RU[rel], "ru") for rel in hints_en]
    with ThreadPoolExecutor(JOBS) as pool:
        metas = list(pool.map(lambda job: run_one(*job), jobs))
    by_key = {(meta["rel"], meta["lang"]): meta for meta in metas}

    rows = []
    totals = {lang: {"restored": 0, "fuzzy": 0, "unknown": 0, "edge": 0, "tokens_in": 0, "cost": 0.0} for lang in LANGS}
    for rel in hints_en:
        results = {lang: load_result((OUT_DIR / lang / rel).with_suffix(".json")) for lang in LANGS}
        reference = load_result((REFERENCE_DIR / rel).with_suffix(".json"))
        cells = [Path(rel).stem + " (" + Path(rel).parent.name[:22] + ")"]
        for lang in LANGS:
            meta, result = by_key[(rel, lang)], results[lang]
            if meta.get("error") or result is None:
                cells.append(f"ОШИБКА: {meta.get('error', '')[:60]}")
                continue
            tags = meta["tags"]
            totals[lang]["restored"] += tags["restored"]
            totals[lang]["fuzzy"] += tags["fuzzy"]
            totals[lang]["unknown"] += tags["unknown"]
            totals[lang]["edge"] += len(result["edge_words"])
            totals[lang]["tokens_in"] += meta["prompt_tokens"] or 0
            totals[lang]["cost"] += meta["cost_usd"] or 0.0
            cells.append(
                f"{tags['restored']}/{tags['fuzzy']}/{tags['unknown']} · edge {len(result['edge_words'])} · "
                f"вх {meta['prompt_tokens']} · {meta['cost_usd'] * 100:.2f} ¢"
            )
        en, ru = results["en"], results["ru"]
        cells.append(f"{similarity(en['content_markdown'], ru['content_markdown']):.3f}" if en and ru else "—")
        for lang in LANGS:
            got = results[lang]
            cells.append(
                f"{similarity(got['content_markdown'], reference['content_markdown']):.3f}"
                if got and reference
                else "—"
            )
        rows.append(cells)

    header = [
        "страница",
        "en: restored/fuzzy/unknown · edge_words · токены входа · цена",
        "ru: то же",
        "сходство текстов en↔ru",
        "en ↔ боевой v8",
        "ru ↔ боевой v8",
    ]
    lines = [
        "# Английские и русские промпты: DeepSeek V4.1 Flash на мини-наборе повреждённых сканов",
        "",
        f"Страниц: {len(hints_en)}, по два запроса на каждую (промпты en и ru), нарезка на 2 полосы, "
        "рассуждения выключены. Сходство — difflib по тексту без тегов.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(cells) + " |" for cells in rows),
        "",
        "## Итого",
        "",
        "| промпт | restored | fuzzy | unknown | edge_words | токены входа | цена |",
        "|---|---|---|---|---|---|---|",
    ]
    for lang in LANGS:
        total = totals[lang]
        lines.append(
            f"| {lang} | {total['restored']} | {total['fuzzy']} | {total['unknown']} | {total['edge']} | "
            f"{total['tokens_in']} | {total['cost'] * 100:.1f} ¢ |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nсводка: {REPORT}")


if __name__ == "__main__":
    main()
