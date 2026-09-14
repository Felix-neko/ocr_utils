"""PaddleOCR-VL 1.6 по папке полос: PP-DocLayout + VLM → markdown (таблицы в HTML).

Выход на полосу — ``<имя>.md`` и ``<имя>.meta.json``. Запуск:
``uv run --project research/external_ocr_models/local/paddleocr_vl python worker.py --in-dir … --out-dir …``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="PaddleOCR-VL по папке полос")
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pipeline-version", default="v1.6")
    arguments = parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(pipeline_version=arguments.pipeline_version)
    for path in sorted(arguments.in_dir.glob("*.jpg")):
        started = time.time()
        meta: dict = {"engine": f"PaddleOCR-VL {arguments.pipeline_version}"}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                results = list(pipeline.predict(str(path)))
                parts: list[str] = []
                for result in results:
                    result.save_to_markdown(save_path=tmp)
                for md_path in sorted(Path(tmp).rglob("*.md")):
                    parts.append(md_path.read_text(encoding="utf-8"))
                if not parts:
                    raise RuntimeError("пайплайн не выдал markdown")
                (arguments.out_dir / (path.stem + ".md")).write_text("\n\n".join(parts), encoding="utf-8")
                meta["blocks"] = sum(
                    len(getattr(result, "json", {}).get("res", {}).get("parsing_res_list", []) or [])
                    for result in results
                )
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
