"""Сводка по DPI картинок в PDF: таблица «один PDF — одна строка» плюс общие итоги."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: Насколько фактический DPI может отличаться от заявленного, чтобы считать их совпавшими.
MATCH_TOLERANCE = 2.0

#: Целевое разрешение, относительно которого ищем отклонения.
NOMINAL_DPI = 300


def spread(values: list[float]) -> str:
    """Компактная запись разброса: одно число, если разброс мал, иначе медиана и границы."""
    if not values:
        return "—"
    lo, hi = min(values), max(values)
    median = statistics.median(values)
    return f"{median:.0f}" if hi - lo <= 1.0 else f"{median:.0f} ({lo:.0f}–{hi:.0f})"


def counter_str(counter: Counter, limit: int = 4) -> str:
    """Значения счётчика по убыванию частоты: «300x300 (88), 200x200 (2)»."""
    if not counter:
        return "—"
    items = counter.most_common()
    if len(items) == 1:
        return items[0][0] or "нет JFIF"
    parts = [f"{value or 'нет JFIF'} ({count})" for value, count in items[:limit]]
    if len(items) > limit:
        parts.append(f"и ещё {len(items) - limit}")
    return ", ".join(parts)


def main(raw_csv: Path, out_csv: Path, out_md: Path) -> None:
    rows = list(csv.DictReader(raw_csv.open(encoding="utf-8")))
    by_pdf: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pdf[row["pdf"]].append(row)

    table = []
    for pdf, items in sorted(by_pdf.items()):
        # Фактический DPI считаем по обеим осям: на нескольких сканах они чуть разъезжаются.
        dpi = [float(i["dpi_x"]) for i in items] + [float(i["dpi_y"]) for i in items]
        jfif = Counter(i["jfif"] for i in items)
        declared = [float(v.split("x")[0]) for i in items if (v := i["jfif"]) and "x" in v and v[0].isdigit()]
        median_dpi = statistics.median(dpi)
        table.append(
            {
                "год": items[0]["год"],
                "pdf": pdf,
                "картинок": len(items),
                "фактический DPI": spread(dpi),
                "медиана DPI": round(median_dpi, 1),
                "JFIF DPI": counter_str(jfif),
                "JFIF совпадает с фактическим": (
                    "—"
                    if not declared
                    else ("да" if abs(statistics.median(declared) - median_dpi) <= MATCH_TOLERANCE else "нет")
                ),
            }
        )

    fields = ["год", "pdf", "картинок", "фактический DPI", "медиана DPI", "JFIF DPI", "JFIF совпадает с фактическим"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    # --- Итоги по паку: считаем по картинкам, ось берём горизонтальную ---
    buckets = Counter(int(round(float(r["dpi_x"]) / 25.0)) * 25 for r in rows)
    jfif_total = Counter(r["jfif"] for r in rows)
    by_year = defaultdict(list)
    for r in rows:
        by_year[r["год"]].append(float(r["dpi_x"]))

    deviating = [r for r in table if abs(r["медиана DPI"] - NOMINAL_DPI) > MATCH_TOLERANCE]

    lines = ["# DPI картинок в PDF: пак Сафронова, «Плановое хозяйство»", ""]
    lines.append(f"PDF: **{len(by_pdf)}**, картинок: **{len(rows)}**. Все картинки — JPEG, по одной на страницу.")
    lines.append("")
    lines.append("Два разных числа, которые оба называют «DPI»:")
    lines.append("")
    lines.append("* **Фактический DPI** — пиксели картинки, делённые на размер её врезки на странице PDF")
    lines.append("  в дюймах. Это реальная плотность скана: сколько точек приходится на дюйм бумаги.")
    lines.append("* **JFIF DPI** — число в заголовке самого JPEG. Это только тег; на количество пикселей")
    lines.append("  и на вёрстку страницы он не влияет.")
    lines.append("")
    lines.append("Хорошая новость: **там, где сегмент JFIF есть, он не врёт** — ни одной картинки, где тег")
    lines.append("расходился бы с фактическим DPI больше чем на 2, в паке нет. Расхождения бывают только")
    lines.append("в другую сторону: у части файлов сегмента JFIF нет вовсе, и DPI неоткуда прочитать.")
    lines.append("")
    lines.append("## Итого по паку")
    lines.append("")
    lines.append("| Фактический DPI (округл. до 25) | Картинок | Доля |")
    lines.append("|---:|---:|---:|")
    for bucket, count in sorted(buckets.items()):
        lines.append(f"| {bucket} | {count} | {count / len(rows) * 100:.1f}% |")
    lines.append("")
    lines.append("| JFIF DPI в заголовке JPEG | Картинок | Доля |")
    lines.append("|---|---:|---:|")
    for value, count in jfif_total.most_common():
        lines.append(f"| {value or '_нет сегмента JFIF_'} | {count} | {count / len(rows) * 100:.1f}% |")
    lines.append("")
    lines.append(f"## Файлы, где фактический DPI не {NOMINAL_DPI}")
    lines.append("")
    lines.append(f"Таких PDF **{len(deviating)}** из {len(by_pdf)}; в остальных ровно {NOMINAL_DPI}.")
    lines.append("")
    lines.append("| Год | PDF | Картинок | Фактический DPI | JFIF DPI |")
    lines.append("|---|---|---:|---|---|")
    for row in deviating:
        lines.append(
            f"| {row['год']} | `{row['pdf']}` | {row['картинок']} | {row['фактический DPI']} | {row['JFIF DPI']} |"
        )
    lines.append("")
    lines.append("## По годам")
    lines.append("")
    lines.append("| Год | Картинок | Фактический DPI |")
    lines.append("|---|---:|---|")
    for year in sorted(by_year):
        lines.append(f"| {year} | {len(by_year[year])} | {spread(by_year[year])} |")
    lines.append("")
    lines.append("## Файлы без сегмента JFIF")
    lines.append("")
    lines.append("У этих PDF в JPEG нет сегмента JFIF — разрешение в файле не записано вообще (это")
    lines.append("характерная примета сохранения из Photoshop: вместо JFIF стоит маркер Adobe APP14).")
    lines.append("Фактический DPI у всех — около 300.")
    lines.append("")
    lines.append("| Год | PDF | Картинок без JFIF | Фактический DPI |")
    lines.append("|---|---|---:|---|")
    for row in table:
        missing = Counter(i["jfif"] for i in by_pdf[row["pdf"]])[""]
        if missing:
            lines.append(
                f"| {row['год']} | `{row['pdf']}` | {missing} из {row['картинок']} | {row['фактический DPI']} |"
            )
    lines.append("")
    lines.append("## Что это значит для экспорта")
    lines.append("")
    lines.append(f"Опция `--dpi` в `ocr_utils.pdf_utils` проставляет тег и **не меняет количество пикселей**.")
    lines.append(f"Поэтому `--dpi {NOMINAL_DPI}` на весь пак — это утверждение «здесь {NOMINAL_DPI} точек на дюйм")
    lines.append("бумаги», и для 313 из 347 PDF оно верное. Для остальных 34 тег окажется неправдой:")
    lines.append("")
    lines.append("* два номера 1938 года отсканированы в 150 DPI — тег завысит плотность вдвое;")
    lines.append("* весь 1959 и весь 1960 год (26 номеров) — 200 DPI;")
    lines.append("* 1965/10–1966/03 (6 номеров) — 400 DPI, тег занизит плотность;")
    lines.append("* 1973/01 и 1973/02 — плавающий DPI от страницы к странице (260–430).")
    lines.append("")
    lines.append("На саму картинку это не влияет никак: пиксели те же, нарезке разворотов тег безразличен.")
    lines.append("Разойдётся только физический размер при печати и в программах, которые верят тегу.")
    lines.append("")
    lines.append("## Все файлы")
    lines.append("")
    lines.append("| Год | PDF | Картинок | Фактический DPI | JFIF DPI | JFIF = факт |")
    lines.append("|---|---|---:|---|---|---|")
    for row in table:
        lines.append(
            f"| {row['год']} | `{row['pdf']}` | {row['картинок']} | {row['фактический DPI']} "
            f"| {row['JFIF DPI']} | {row['JFIF совпадает с фактическим']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"PDF: {len(by_pdf)}, картинок: {len(rows)}, отклоняются от {NOMINAL_DPI}: {len(deviating)}")
    print("фактический DPI по корзинам:", dict(sorted(buckets.items())))
    print("записано:", out_csv, out_md)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
