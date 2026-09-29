"""Тесты стенда краевых пометок: отбор выбросов, рамка кандидата, перевод ответов судей в вердикты, замер ресурсов, оверлей."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.edge_marks import vram
from research.edge_marks.candidates import Candidate, outermost_ink
from research.edge_marks.cli import aligned_outliers
from research.edge_marks.interpret import from_boxes, from_score
from research.edge_marks.judges.common import Verdict, overlap_share
from research.edge_marks.judges.rule import K, line_glyphs
from research.edge_marks.overlays import JudgeFrame, page_overlay


def test_aligned_outliers_keeps_only_protrusion_on_aligned_side(tmp_path):
    """Выброс на выровненной стороне берётся; на рваной стороне и слишком далёкий — нет."""
    rows = []
    # Выровненная сторона: 10 строк «на кривой», одна торчит на 2 мм и одна на 9 мм (законная вёрстка).
    for row in range(12):
        resid = {3: -2.0, 7: -9.0}.get(row, 0.0)
        rows.append({"key": "p", "block": 0, "side": "right", "row": row, "rows": 12, "resid_mm": resid,
                     "status": "on" if resid == 0.0 else "out"})  # fmt: skip
    # Рваная сторона: половина концов вне кривой.
    for row in range(12):
        rows.append({"key": "p", "block": 1, "side": "right", "row": row, "rows": 12, "resid_mm": -2.0,
                     "status": "on" if row % 2 else "out"})  # fmt: skip
    path = tmp_path / "ends.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    found = aligned_outliers(path)
    assert list(zip(found.block, found.row)) == [(0, 3)]


def test_outermost_ink_takes_speck_not_hanging_hyphen():
    """За стороной висячий дефис и соринка за ним: кандидат — только соринка, дефис в рамку не попадает."""
    binary = np.zeros((200, 400), dtype=bool)
    binary[90:110, 200:290] = True  # строка до стороны x=290
    binary[99:102, 292:302] = True  # висячий дефис за стороной
    binary[95:101, 320:326] = True  # соринка за дефисом
    box, mask = outermost_ink(binary, end_x=326, row_y=100, sx=290, x_h=20, right=True)
    assert box == (320, 95, 326, 101)
    assert mask.all()


def test_outermost_ink_keeps_parts_stacked_over_each_other():
    """Двоеточие за левой стороной: обе точки (перекрытие по x) — в одной рамке."""
    binary = np.zeros((200, 400), dtype=bool)
    binary[90:110, 110:200] = True  # строка от стороны x=110
    binary[92:96, 100:104] = True  # верхняя точка
    binary[104:108, 100:104] = True  # нижняя точка
    box, _ = outermost_ink(binary, end_x=100, row_y=100, sx=110, x_h=20, right=False)
    assert box == (100, 92, 104, 108)


def test_line_glyphs_skips_next_column_and_keeps_touching_speck():
    """Компонент соседней колонки за кандидатом не берётся, соринка выше середины строки, накрывающая кандидата, — берётся."""
    binary = np.zeros((100, 600), dtype=bool)
    binary[45:55, 100:200] = True  # буквы строки (рабочая копия)
    binary[20:26, 230:236] = True  # соринка над серединой строки
    binary[45:55, 400:450] = True  # соседняя колонка
    box = tuple(v * K for v in (228, 18, 238, 28))
    line_box = tuple(v * K for v in (0, 0, 600, 100))
    glyphs = line_glyphs(binary, line_box, row_y=50 * K, x_h=10 * K, box=box, right=True)
    xs = sorted(int(g[0]) for g in glyphs)
    assert xs == [100, 230]


def test_overlap_and_box_verdicts():
    """Рамка слова накрывает кандидата — знак; мимо — сор."""
    assert overlap_share((0, 0, 10, 10), (5, 0, 20, 10)) == 0.5
    assert from_boxes((0, 0, 10, 10), [[-5, -5, 15, 15, "слово"]])[0] is Verdict.SIGN
    assert from_boxes((0, 0, 10, 10), [[20, 0, 30, 10]])[0] is Verdict.JUNK
    assert from_boxes((0, 0, 10, 10), [])[0] is Verdict.JUNK


def test_score_verdict_thresholds():
    """Оценка «символ»: высокая — знак, низкая — сор, между — неясно."""
    assert from_score(0.9)[0] is Verdict.SIGN
    assert from_score(0.1)[0] is Verdict.JUNK
    assert from_score(0.35)[0] is Verdict.UNSURE


def test_gpu_usage_parses_nvidia_smi(monkeypatch):
    """Разбор ``nvidia-smi --query-compute-apps`` и пустой ответ при отсутствии GPU."""
    monkeypatch.setattr(vram.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="123, 4500\n456, 12\nbad\n"))
    assert vram.gpu_usage_by_pid() == {123: 4500.0, 456: 12.0}

    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(vram.subprocess, "run", missing)
    assert vram.gpu_usage_by_pid() == {}


def test_run_measured_sums_vram_over_process_tree(monkeypatch):
    """Видеопамять берётся по дереву запущенного процесса; ОЗУ и код возврата — реальные."""
    real_run = subprocess.run
    started = {}

    def fake_smi(cmd, *args, **kwargs):
        if cmd[0] == "nvidia-smi":
            pid = started.get("pid", -1)
            return SimpleNamespace(stdout=f"{pid}, 700\n999999, 5000\n")
        return real_run(cmd, *args, **kwargs)

    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        started["pid"] = process.pid
        return process

    monkeypatch.setattr(vram.subprocess, "run", fake_smi)
    monkeypatch.setattr(vram.subprocess, "Popen", popen)
    usage = vram.run_measured([sys.executable, "-c", "import time; time.sleep(0.6); raise SystemExit(3)"])
    assert usage.returncode == 3
    assert usage.vram_mb == 700.0
    assert usage.rss_mb > 0


def test_page_overlay_frames_page():
    """Оверлей: шапка и легенда в полях, полоса целиком внутри."""
    gray = np.full((300, 200), 255, dtype=np.uint8)
    candidate = Candidate("e1", "p", "x.pdf", 1, "right", "S", "junk", (150, 100, 160, 110), (0, 80, 190, 130),
                          160.0, 105.0, 148.0, 20.0)  # fmt: skip
    frame = pd.DataFrame([{"id": "e1", "scale": "a", "verdict": "junk"}, {"id": "e1", "scale": "b", "verdict": "sign"}])
    picture = page_overlay(gray, "p — судья тест", JudgeFrame("тест", frame), [candidate], None)
    assert picture.shape[1] == 200 and picture.shape[0] > 300
