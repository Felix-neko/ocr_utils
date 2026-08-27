"""Обзор DPI картинок в PDF: заявленный в JFIF и фактический по вёрстке.

Фактический DPI = пиксели картинки / размер её врезки на странице в дюймах. Врезка
берётся как page.rect: на этом паке картинка всегда занимает страницу целиком (сверено
выборкой), а честный page.get_image_rects() разбирает контент-стрим и на порядок медленнее.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz

#: Сколько PDF разбирать параллельно. Задача упирается и в счёт, и в диск.
DEFAULT_JOBS = 12


def jfif_density(data: bytes) -> tuple[int, int, int] | None:
    """Плотность из сегмента JFIF APP0: (единицы, X, Y). None, если сегмента нет."""
    if not data.startswith(b"\xff\xd8"):
        return None
    offset = 2
    while offset < len(data) - 1 and data[offset] == 0xFF:
        marker = data[offset + 1]
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if marker in (0xD9, 0xDA):
            return None
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if marker == 0xE0 and data[offset + 4 : offset + 9] == b"JFIF\x00":
            payload = data[offset + 4 : offset + 2 + length]
            return payload[7], int.from_bytes(payload[8:10], "big"), int.from_bytes(payload[10:12], "big")
        offset += 2 + length
    return None


def format_density(density: tuple[int, int, int] | None) -> str:
    """Человекочитаемая запись плотности JFIF."""
    if density is None:
        return ""
    unit, x, y = density
    if unit == 1:
        return f"{x}x{y}"
    if unit == 2:
        return f"{round(x * 2.54)}x{round(y * 2.54)} (из см/дюйм)"
    return "без единиц"


def scan_pdf(args: tuple[str, str]) -> list[list]:
    """Разобрать один PDF: по строке на каждую вложенную картинку."""
    pdf_str, root_str = args
    pdf, root = Path(pdf_str), Path(root_str)
    rel = pdf.relative_to(root)
    year = rel.parts[0] if len(rel.parts) > 1 else ""
    try:
        doc = fitz.open(str(pdf))
    except Exception as e:
        return [[str(rel), year, "", f"ОШИБКА: {e}", "", "", "", "", "", "", ""]]

    rows = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        rect = page.rect
        for xref, _smask, width, height, _bpc, _cs, _alt, _name, filt, *_ in page.get_images(full=True):
            density = None
            if "DCT" in filt:
                try:
                    density = jfif_density(doc.xref_stream_raw(xref))
                except Exception:
                    density = None
            rows.append(
                [
                    str(rel),
                    year,
                    page_idx + 1,
                    filt,
                    width,
                    height,
                    round(rect.width, 2),
                    round(rect.height, 2),
                    round(width / (rect.width / 72), 1) if rect.width else 0,
                    round(height / (rect.height / 72), 1) if rect.height else 0,
                    format_density(density),
                ]
            )
    doc.close()
    return rows


def main(root: Path, csv_path: Path, jobs: int) -> None:
    pdfs = sorted(root.rglob("*.pdf"))
    tasks = [(str(p), str(root)) for p in pdfs]
    done = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["pdf", "год", "страница", "фильтр", "px_w", "px_h", "стр_pt_w", "стр_pt_h", "dpi_x", "dpi_y", "jfif"]
        )
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(scan_pdf, task) for task in tasks]
            for future in as_completed(futures):
                writer.writerows(future.result())
                done += 1
                if done % 25 == 0:
                    print(f"...{done}/{len(pdfs)}", flush=True)
    print("готово:", csv_path, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_JOBS)
