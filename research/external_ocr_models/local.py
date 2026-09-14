"""Локальные специализированные OCR-движки: запуск воркера изолированного окружения.

Каждый движок живёт в ``local/<имя>/`` со своим ``pyproject.toml`` и ``worker.py`` — как
``paddle_env`` в research/legacy/table_processing: их стеки (transformers 4.46 против 5.x,
flash-attn, vllm) несовместимы друг с другом и с основным окружением. Основной пакет
подготавливает картинки, зовёт воркер подпроцессом на папке и читает его выходы.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from research.external_ocr_models import PROMPT_VERSION
from research.external_ocr_models.imaging import prepare
from research.external_ocr_models.models import ModelSpec, local_engine_dir
from research.external_ocr_models.ocr import RunOptions, output_paths
from research.external_ocr_models.render import to_markdown
from research.external_ocr_models.schema import PageResult, unspace_letters

logger = logging.getLogger(__name__)

LOCAL_ROOT = Path(__file__).parent / "local"


def engine_project(spec: ModelSpec) -> Path:
    project = LOCAL_ROOT / local_engine_dir(spec)
    if not (project / "worker.py").is_file():
        raise FileNotFoundError(f"нет воркера для {spec.name}: {project / 'worker.py'}")
    return project


def run_local(spec: ModelSpec, in_dir: Path, rels: list[Path], out_dir: Path, options: RunOptions) -> list[dict]:
    """Подготовить полосы во временную папку, прогнать воркер, разложить выходы как у OpenRouter.

    Воркер получает ``--in-dir`` с JPEG (имена — относительные пути с ``__`` вместо ``/``)
    и ``--out-dir``, куда кладёт на полосу ``<имя>.json`` (поля PageResult) или просто
    ``<имя>.md`` (тело), плюс ``<имя>.meta.json`` (seconds, error, что захочет).
    """
    project = engine_project(spec)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="external_ocr_") as tmp:
        tmp_in, tmp_out = Path(tmp) / "in", Path(tmp) / "out"
        tmp_in.mkdir()
        tmp_out.mkdir()
        names: dict[str, Path] = {}
        for rel in rels:
            name = rel.with_suffix("").as_posix().replace("/", "__") + ".jpg"
            names[name] = rel
            image = prepare(in_dir / rel, options.max_side, options.quality)[0]
            (tmp_in / name).write_bytes(image.data)
        command = [
            "uv",
            "run",
            "--project",
            str(project),
            "python",
            str(project / "worker.py"),
            "--in-dir",
            str(tmp_in),
            "--out-dir",
            str(tmp_out),
        ]
        logger.info("%s: %d полос, воркер %s", spec.name, len(rels), project.name)
        started = time.monotonic()
        completed = subprocess.run(command, stdout=sys.stderr, stderr=subprocess.STDOUT)
        logger.info("воркер завершился с кодом %d за %.0f с", completed.returncode, time.monotonic() - started)
        for name, rel in names.items():
            paths = output_paths(out_dir, rel)
            paths["meta"].parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "page": rel.as_posix(),
                "model": spec.name,
                "openrouter_id": spec.openrouter_id,
                "prompt_version": PROMPT_VERSION,
                "provider": "local",
                "cost_usd": 0.0,
                "max_side": options.max_side,
                "error": None,
                "parse_error": None,
            }
            worker_meta = tmp_out / (Path(name).stem + ".meta.json")
            if worker_meta.is_file():
                meta.update(json.loads(worker_meta.read_text(encoding="utf-8")))
            md_path = tmp_out / (Path(name).stem + ".md")
            json_path = tmp_out / (Path(name).stem + ".json")
            raw_path = tmp_out / (Path(name).stem + ".raw.txt")
            if completed.returncode != 0 and not (md_path.is_file() or json_path.is_file()):
                meta["error"] = meta.get("error") or f"воркер вернул код {completed.returncode}"
            if raw_path.is_file():
                paths["raw"].write_bytes(raw_path.read_bytes())
            result: PageResult | None = None
            if json_path.is_file():
                fields = json.loads(json_path.read_text(encoding="utf-8"))
                result = PageResult(
                    **{
                        key: fields.get(key)
                        for key in ("page_number", "running_header", "running_footer")
                        if key in fields
                    },
                    content_markdown=unspace_letters(fields.get("content_markdown") or ""),
                    is_toc=bool(fields.get("is_toc")),
                    notes=str(fields.get("notes") or ""),
                )
            elif md_path.is_file():
                result = PageResult(content_markdown=unspace_letters(md_path.read_text(encoding="utf-8")))
            if result is not None:
                body = result.content_markdown
                if options.write_json:
                    paths["json"].write_text(result.to_json(), encoding="utf-8")
                if options.write_md:
                    paths["md"].write_text(to_markdown(result), encoding="utf-8")
                meta["content_chars"] = len(body)
                meta["latency_s"] = meta.get("seconds")
                meta["page_number"] = result.page_number
                meta["is_toc"] = result.is_toc
            elif not meta.get("error"):
                meta["error"] = "воркер не выдал ни .json, ни .md"
            paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            rows.append(meta)
    return rows
