"""Marker по папке полос: layout + OCR surya → markdown с заголовками и таблицами.

Выход на полосу — ``<имя>.md`` и ``<имя>.meta.json``. Запуск:
``uv run --project research/external_ocr_models/local/marker python worker.py --in-dir … --out-dir …``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Marker по папке полос")
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    # Картинка — всегда OCR; языки surya определяет сам, но подсказка не мешает.
    config = ConfigParser({"output_format": "markdown", "force_ocr": True, "languages": "ru,en"})
    models = create_model_dict()
    converter = PdfConverter(
        config=config.generate_config_dict(),
        artifact_dict=models,
        processor_list=config.get_processors(),
        renderer=config.get_renderer(),
    )
    for path in sorted(arguments.in_dir.glob("*.jpg")):
        started = time.time()
        meta: dict = {"engine": "marker"}
        try:
            rendered = converter(str(path))
            text, _, _ = text_from_rendered(rendered)
            (arguments.out_dir / (path.stem + ".md")).write_text(text, encoding="utf-8")
            meta["vram_mb"] = int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
        except Exception as error:  # одна битая полоса не должна ронять пакет
            meta["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        meta["seconds"] = round(time.time() - started, 2)
        (arguments.out_dir / (path.stem + ".meta.json")).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{path.name}: {meta.get('error') or 'ok'}, {meta['seconds']} с", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
