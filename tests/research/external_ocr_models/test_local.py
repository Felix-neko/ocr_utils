"""Адаптер локальных движков с фейковым воркером и чистые функции воркера dots.ocr."""

import importlib.util
import json
from pathlib import Path

from PIL import Image

from research.external_ocr_models import local
from research.external_ocr_models.models import resolve
from research.external_ocr_models.ocr import RunOptions

FAKE_WORKER = """
import argparse, json, sys
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument("--in-dir", type=Path); p.add_argument("--out-dir", type=Path)
a = p.parse_args()
for f in sorted(a.in_dir.glob("*.jpg")):
    (a.out_dir / (f.stem + ".json")).write_text(json.dumps({"content_markdown": "# Заголовок\\n\\nТекст " + f.stem, "page_number": "7", "running_header": "МТС", "running_footer": None, "is_toc": False}), encoding="utf-8")
    (a.out_dir / (f.stem + ".meta.json")).write_text(json.dumps({"seconds": 1.5, "engine": "fake"}), encoding="utf-8")
"""


def test_run_local_with_fake_worker(tmp_path, monkeypatch):
    project = tmp_path / "fake_engine"
    project.mkdir()
    (project / "worker.py").write_text(FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(local, "engine_project", lambda spec: project)
    # без uv: подменяем команду на прямой python
    real_run = local.subprocess.run
    monkeypatch.setattr(
        local.subprocess,
        "run",
        lambda cmd, **kw: real_run([local.sys.executable] + cmd[cmd.index("python") + 1 :], **kw),
    )

    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    (in_dir / "1966/03").mkdir(parents=True)
    Image.new("L", (300, 400), 230).save(in_dir / "1966/03/IMG_0104_2R.jpg")
    rows = local.run_local(
        resolve("local-dots-ocr"), in_dir, [Path("1966/03/IMG_0104_2R.jpg")], out_dir, RunOptions(max_side=200)
    )
    assert len(rows) == 1 and rows[0]["error"] is None and rows[0]["cost_usd"] == 0.0 and rows[0]["seconds"] == 1.5
    result = json.loads((out_dir / "1966/03/IMG_0104_2R.json").read_text(encoding="utf-8"))
    assert (
        result["page_number"] == "7"
        and result["running_header"] == "МТС"
        and "IMG_0104_2R" in result["content_markdown"]
    )
    assert (out_dir / "1966/03/IMG_0104_2R.md").read_text(encoding="utf-8").startswith('---\npage_number: "7"')


def _dots_worker():
    path = Path(__file__).parents[3] / "research/external_ocr_models/local/dots_ocr/worker.py"
    spec = importlib.util.spec_from_file_location("dots_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dots_elements_to_page():
    worker = _dots_worker()
    elements = [
        {"bbox": [0, 0, 10, 10], "category": "Page-header", "text": "МТС № 3"},
        {"bbox": [0, 0, 10, 10], "category": "Title", "text": "ЗАГОЛОВОК"},
        {"bbox": [0, 0, 10, 10], "category": "Text", "text": "Абзац."},
        {"bbox": [0, 0, 10, 10], "category": "Table", "text": "<table><tr><td>1</td></tr></table>"},
        {"bbox": [0, 0, 10, 10], "category": "Picture"},
        {"bbox": [0, 0, 10, 10], "category": "Page-footer", "text": "12"},
    ]
    page = worker.to_page(elements)
    assert page["content_markdown"] == "# ЗАГОЛОВОК\n\nАбзац.\n\n<table><tr><td>1</td></tr></table>\n\n> [картинка]"
    assert page["running_header"] == "МТС № 3" and page["running_footer"] == "12" and page["page_number"] == "12"
    assert worker.parse_elements("```json\n" + json.dumps(elements) + "\n```")[1]["category"] == "Title"
