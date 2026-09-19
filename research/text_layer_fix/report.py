"""Сводка прогона по CSV и JSON: таблицы для отчёта в markdown.

Ничего не считает заново: читает ``survey.csv``, ``pages.csv``, ``words.csv``, ``zones.csv``,
``fix.csv``, ``lineart_eval.json``, ``llm_compare.json`` каталога прогона и печатает
markdown-фрагменты с числами, которые переносятся в ``reports/text_layer_fix.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _table(frame: pd.DataFrame, index_name: str = "") -> str:
    """DataFrame → markdown-таблица без зависимости от tabulate."""
    columns = ([index_name] if index_name else []) + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for idx, row in frame.iterrows():
        cells = ([str(idx)] if index_name else []) + [
            ("%.3g" % v if isinstance(v, float) else str(v)) for v in row.tolist()
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def survey_section(out_dir: Path) -> str:
    """Раздел по обзору слоя всего пака."""
    path = out_dir / "survey.csv"
    if not path.is_file():
        return "_survey.csv нет_"
    df = pd.read_csv(path)
    rotated = df[df.rotated_words > 0]
    lines = [
        f"Страниц: {len(df)}, слов слоя: {int(df.words.sum())}, спанов: {int(df.spans.sum())}, "
        f"страниц без слов: {int((df.words == 0).sum())}, потоков на страницу всегда 1: {bool((df.streams == 1).all())}.",
        f"Спаны с повёрнутой матрицей FineReader: {int(df.rotated_words.sum())} слов на {len(rotated)} страницах "
        f"(из них на последней странице выпуска: {int((rotated.page >= 96).sum())}).",
        f"Страниц с дополнительными картинками FineReader (куски, вырезанные как иллюстрация): {int((df.figures > 0).sum())}, "
        f"картинок всего: {int(df.figures.sum())}.",
        f"Глифов вне страницы (текст обрезанной части образа за CropBox): {int(df.offpage.sum())} на {int((df.offpage > 0).sum())} страницах; "
        f"непривязанных глифов внутри страницы: {int(df.unmatched.sum())}.",
        f"Растяжение слов (a матрицы текста): медиана медиан {df.stretch_median.median():.3f}, "
        f"слов с растяжением вне [0.7, 1.5] на странице: медиана {df.stretch_outliers.median():.0f}, p90 {df.stretch_outliers.quantile(0.9):.0f}.",
        f"Коротких слов (1–2 знака без цифр) на странице: медиана {df.short_words.median():.0f}, p90 {df.short_words.quantile(0.9):.0f} — "
        "россыпь от повёрнутого текста по этому признаку не выделяется.",
        f"Разбор: медиана {df.seconds.median():.2f} с на страницу.",
    ]
    return "\n".join(lines)


def sample_section(out_dir: Path) -> str:
    """Раздел по выборке: зоны и вердикты по категориям."""
    pages = pd.read_csv(out_dir / "pages.csv").merge(
        pd.read_csv(out_dir / "sample.csv")[["pdf", "page", "kind"]], on=["pdf", "page"], how="left"
    )
    columns = [
        "tables",
        "zones",
        "zone_table_cell",
        "zone_mixed",
        "zone_upright_missing",
        "zone_line_art",
        "zone_standalone",
        "read_accepted",
        "read_rejected",
        "delete",
        "sanitize",
        "suspect",
        "keep_rotated",
    ]
    grouped = pages.groupby("kind")[columns].sum()
    grouped.insert(0, "страниц", pages.groupby("kind").size())
    grouped.insert(
        1, "стр. с delete", pages[pages.delete > 0].groupby("kind").size().reindex(grouped.index).fillna(0).astype(int)
    )
    text = _table(grouped, "категория")
    text += f"\n\nОшибок разбора: {int(pages.error.notna().sum())}; время на страницу: медиана {pages.seconds.median():.1f} с, p90 {pages.seconds.quantile(0.9):.1f} с."
    return text


def words_section(out_dir: Path) -> str:
    """Раздел по словам: расстояния, смешивание, россыпь."""
    words = pd.read_csv(out_dir / "words.csv")
    zones = pd.read_csv(out_dir / "zones.csv")
    lines = [f"Вердикты (кроме KEEP): {words.verdict.value_counts().to_dict()}."]
    deleted = words[words.verdict == "delete"]
    if len(deleted):
        lines.append(
            f"DELETE: {len(deleted)} слов; длина текста медиана {deleted.text.astype(str).str.len().median():.0f} знаков, "
            f"доля слов целиком внутри зоны (overlap = 1): {(deleted.overlap >= 0.999).mean():.3f}, "
            f"слов с 4+ буквами подряд: {int(deleted.text.astype(str).str.contains(r'[А-Яа-яA-Za-z]{4,}', regex=True).sum())}."
        )
    suspect = words[words.verdict == "suspect"]
    if len(suspect):
        lines.append(
            f"SUSPECT: {len(suspect)}; расстояние до ближайшей зоны, мм: медиана {suspect.dist_mm.median():.2f}, p90 {suspect.dist_mm.quantile(0.9):.2f}; "
            f"причины: {suspect.reason.value_counts().to_dict()}."
        )
    sanitize = words[words.verdict == "sanitize"]
    lines.append(
        f"SANITIZE (усечение слова): {len(sanitize)} слов — задетых зоной частично почти нет: гранулярность слоя FineReader — слово."
    )
    # Смешивание: строки дерева структуры, где есть и KEEP-слова (не в words.csv), и DELETE — считается по struct_line из zones/words.
    lines_with_delete = deleted.struct_line.dropna().nunique() if len(deleted) else 0
    lines.append(f"Строк дерева структуры FineReader с удалёнными словами: {lines_with_delete}.")
    if len(zones):
        by_kind = zones.groupby("zone_kind").agg(
            zones=("zone_index", "size"),
            accepted=("read_accepted", lambda s: int((s == True).sum())),
            no_side=("rotate_cw", lambda s: int(s.isna().sum())),
        )
        lines.append("\nЗоны по видам:\n\n" + _table(by_kind, "вид зоны"))
        reasons = (
            zones[zones.read_accepted != True]
            .read_reason.fillna("")
            .str.replace(r"\d+\.\d+", "N", regex=True)
            .value_counts()
            .head(8)
        )
        lines.append("\nПричины отказа от чтения:\n\n" + _table(reasons.to_frame("зон"), "причина"))
        if "read_rotate" in zones:
            rot = zones[zones.zone_kind.isin(["table_cell", "table_cell_mixed"])].read_rotate.value_counts().to_dict()
            lines.append(f"\nСторона чтения ячеек: {rot}.")
    return "\n".join(lines)


def _symbols_only(text: str) -> bool:
    """Токен без единой буквы и цифры («^», «=Х», «i"o=» не считается — там буквы): чистая россыпь."""
    stripped = text.strip()
    return bool(stripped) and not any(ch.isalnum() for ch in stripped)


def cache_metrics(out_dir: Path) -> str:
    """Метрики по всем словам кэша (включая KEEP): россыпь вне зон, смешанные строки, расстояния.

    Args:
        out_dir: Каталог прогона с ``cache/`` и ``sample.csv``.

    Returns:
        Markdown-абзацы с числами.
    """
    sample = pd.read_csv(out_dir / "sample.csv")
    junk_dist: list[float] = []
    junk_far = 0
    mixed_lines = 0
    lines_with_delete = 0
    pages_with_zones = 0
    delete_words_total = 0
    delete_by_kind: dict[str, int] = {}
    for row in sample.itertuples():
        path = out_dir / "cache" / row.pdf / f"p{int(row.page):04d}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("error"):
            continue
        zones = payload.get("zones", [])
        active = [
            z
            for z in zones
            if z.get("rotate_cw") != 0 and (z["kind"] != "table_cell_mixed" or z.get("rotate_cw") is not None)
        ]
        if active:
            pages_with_zones += 1
        per_line: dict[str, set[str]] = {}
        for word in payload.get("words", []):
            verdict = word["verdict"]
            line = word.get("struct_line")
            if line:
                per_line.setdefault(line, set()).add(verdict)
            if verdict == "delete":
                delete_words_total += 1
                zone = zones[word["zone_index"]] if word.get("zone_index") is not None else None
                if zone:
                    delete_by_kind[zone["kind"]] = delete_by_kind.get(zone["kind"], 0) + 1
            elif active and verdict in ("keep", "suspect") and _symbols_only(word["text"]):
                distance = float(word.get("dist_mm", 999))
                junk_dist.append(distance)
                if distance > 4.0:
                    junk_far += 1
        for verdicts in per_line.values():
            if "delete" in verdicts:
                lines_with_delete += 1
                if "keep" in verdicts:
                    mixed_lines += 1
    dist = pd.Series(junk_dist) if junk_dist else pd.Series([0.0])
    lines = [
        f"Удалённых слов: {delete_words_total}; по видам зон: {delete_by_kind}.",
        (
            f"Строк дерева структуры FineReader с удалёнными словами: {lines_with_delete}, из них с оставленными словами в той же строке: {mixed_lines} "
            f"({mixed_lines / lines_with_delete:.1%} — доля «смешанных» строк)."
            if lines_with_delete
            else ""
        ),
        f"Токены без букв и цифр ВНЕ зон на страницах с зонами: {len(junk_dist)}; расстояние до ближайшей зоны, мм: "
        f"медиана {dist.median():.1f}, p90 {dist.quantile(0.9):.1f}; ближе 4 мм — {len(junk_dist) - junk_far}, дальше — {junk_far} "
        "(тире, отточия, знаки препинания обычного текста).",
    ]
    return "\n".join(line for line in lines if line)


def fix_section(out_dir: Path) -> str:
    """Раздел по исправленным копиям и сверке."""
    path = out_dir / "fix.csv"
    if not path.is_file():
        return "_fix.csv нет_"
    df = pd.read_csv(path)
    ok = df[df.page >= 0]
    return (
        f"Страниц исправлено: {len(ok)} в {ok.pdf.nunique()} выпусках; удалено слов {int(ok.blanked.sum())}, усечено {int(ok.trimmed.sum())}, "
        f"вставок {int(ok.inserted.sum())}, пропущено дублей (FineReader уже написал повёрнутым) {int(ok.skipped_duplicates.sum())}. "
        f"Сверка: не прошли {int((ok.ok != True).sum())} страниц (KEEP не на месте: {int(ok.kept_missing.sum())}, DELETE остались: {int(ok.deleted_remaining.sum())}, "
        f"вставок не найдено: {int(ok.inserts_missing.sum())}, картинка изменилась: {int((ok.image_changed == True).sum())})."
    )


def lineart_section(out_dir: Path) -> str:
    """Раздел по точности и полноте источников line art."""
    path = out_dir / "lineart_eval.json"
    if not path.is_file():
        return "_lineart_eval.json нет_"
    summary = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(summary).T[
        ["predicted", "matched", "precision", "recall", "coverage", "f1", "false_on_pages_without_truth"]
    ]
    frame.columns = [
        "предсказаний",
        "совпало (IoU≥0.5)",
        "точность",
        "полнота",
        "накрытие ≥ 0.5",
        "F1",
        "ложных на контрольных",
    ]
    return _table(frame, "источник")


def llm_section(out_dir: Path) -> str:
    """Раздел по сравнению с языковой моделью."""
    path = out_dir / "llm_compare.json"
    if not path.is_file():
        return "_llm_compare.json нет_"
    return "```\n" + json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=1) + "\n```"


def full_report(out_dir: Path) -> str:
    """Все разделы подряд."""
    parts = [
        "## Обзор слоя (survey)",
        survey_section(out_dir),
        "## Выборка (run)",
        sample_section(out_dir),
        "## Слова",
        words_section(out_dir),
        cache_metrics(out_dir),
        "## Исправленные копии (fix)",
        fix_section(out_dir),
        "## Line art (eval-lineart)",
        lineart_section(out_dir),
        "## LLM (llm-compare)",
        llm_section(out_dir),
    ]
    return "\n\n".join(parts)
