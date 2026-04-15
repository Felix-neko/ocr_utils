"""CLI-точка входа: python -m ocr_utils или ocr-utils."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ocr_utils.pipeline import process_directory, process_single_pdf


def _parse_pages(value: str) -> list[int] | slice | None:
    """Разобрать строку с номерами страниц.

    Форматы:
        "all"         → None (все страницы)
        "0,2,5"       → [0, 2, 5]
        "1:10"        → slice(1, 10)
        "1:10:2"      → slice(1, 10, 2)
    """
    value = value.strip()
    if value.lower() == "all":
        return None
    if ":" in value:
        parts = value.split(":")
        args = [int(p) if p else None for p in parts]
        return slice(*args)
    return [int(x.strip()) for x in value.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ocr-utils", description="Обработка PDF-сканов журналов: разбивка разворотов на страницы + OCR."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- Команда: single ---
    p_single = sub.add_parser("single", help="Обработать один PDF-файл")
    p_single.add_argument("src", type=Path, help="Путь к исходному PDF")
    p_single.add_argument("dst", type=Path, help="Путь к выходному PDF")
    p_single.add_argument(
        "--pages",
        type=str,
        default="all",
        help="Какие страницы обрабатывать: 'all', '0,2,5' или '1:10' (0-based, Python slice). По умолчанию: all",
    )
    p_single.add_argument("--tmp-dir", type=Path, default=None, help="Директория для временных файлов")
    p_single.add_argument("--language", type=str, default="rus", help="Язык OCR (по умолчанию: rus)")
    p_single.add_argument("--dpi", type=int, default=900, help="DPI для оверсемплинга (по умолчанию: 900)")
    p_single.add_argument("--no-deskew", action="store_true", help="Отключить выравнивание страниц")
    p_single.add_argument("--no-clean", action="store_true", help="Отключить очистку шума")
    p_single.add_argument("--no-rotate", action="store_true", help="Отключить автоповорот страниц")

    # --- Команда: dir ---
    p_dir = sub.add_parser("dir", help="Рекурсивно обработать все PDF в директории")
    p_dir.add_argument("src", type=Path, help="Путь к исходной директории")
    p_dir.add_argument("dst", type=Path, help="Путь к выходной директории")
    p_dir.add_argument("--workers", type=int, default=None, help="Количество воркеров (по умолчанию: 3/4 ядер)")
    p_dir.add_argument("--language", type=str, default="rus", help="Язык OCR (по умолчанию: rus)")
    p_dir.add_argument("--dpi", type=int, default=900, help="DPI для оверсемплинга (по умолчанию: 900)")
    p_dir.add_argument("--no-deskew", action="store_true", help="Отключить выравнивание страниц")
    p_dir.add_argument("--no-clean", action="store_true", help="Отключить очистку шума")
    p_dir.add_argument("--no-rotate", action="store_true", help="Отключить автоповорот страниц")

    # --- Общие параметры ---
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод (DEBUG)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Минимальный вывод (только ошибки)")

    args = parser.parse_args(argv)

    # Настройка логирования
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.ERROR
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.command == "single":
        pages = _parse_pages(args.pages)
        process_single_pdf(
            src_pdf=args.src,
            dst_pdf=args.dst,
            pages=pages,
            tmp_dir=args.tmp_dir,
            language=args.language,
            oversample_dpi=args.dpi,
            deskew=not args.no_deskew,
            clean=not args.no_clean,
            rotate_pages=not args.no_rotate,
        )
        return 0

    if args.command == "dir":
        results = process_directory(
            src_dir=args.src,
            dst_dir=args.dst,
            workers=args.workers,
            language=args.language,
            oversample_dpi=args.dpi,
            deskew=not args.no_deskew,
            clean=not args.no_clean,
            rotate_pages=not args.no_rotate,
        )
        errors = {k: v for k, v in results.items() if v is not None}
        if errors:
            print(f"\nОшибки ({len(errors)}):", file=sys.stderr)
            for path, err in errors.items():
                print(f"  {path}: {err}", file=sys.stderr)
            return 1
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
