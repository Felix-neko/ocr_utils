"""Сводка судей: полнота сора, вред (знаки, признанные сором), «неясно», время на кандидата и ресурсы прогона — по судье и масштабу."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.edge_marks.candidates import load
from research.edge_marks.interpret import from_boxes, from_diff, from_score


def read_results(path: Path, candidates: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Строки результатов судьи и служебная часть (время загрузки и пр.).

    Сырые ответы (поле ``kind``: ``diff``, ``boxes``, ``score``) переводятся в вердикты :mod:`interpret`; для
    этого нужны кандидаты (сторона строки, рамка).

    Args:
        path: ``<судья>.jsonl``.
        candidates: ``id → Candidate`` или ``None`` (тогда сырые ответы не переводятся).

    Returns:
        ``(таблица, служебная часть)``.
    """
    rows, meta = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "meta" in record:
            meta.update(record["meta"])
            continue
        kind = record.get("kind")
        if kind and candidates is not None:
            candidate = candidates[record["id"]]
            if kind == "diff":
                verdict, score, detail = from_diff(record["with"], record["erased"], candidate.side)
            elif kind == "boxes":
                verdict, score, detail = from_boxes(candidate.box, record["boxes"])
            else:
                verdict, score, detail = from_score(float(record["score"]))
            record.update({"verdict": verdict.value, "score": score, "detail": detail})
        rows.append(record)
    return pd.DataFrame(rows), meta


# Допустимый вред при подборе порога оценки: доля знаков, признанных сором.
MAX_HARM = 0.05


def ranking(junk_scores: np.ndarray, sign_scores: np.ndarray) -> tuple[float, float, float]:
    """Качество судьи как ранжира, без его собственного порога.

    AUC — вероятность, что случайный сор получил оценку «сор» выше случайного знака (ничьи — пополам).
    Полнота при вреде ≤ ``MAX_HARM`` — доля сора с оценкой строго выше порога, при котором знаков выше
    порога не больше ``MAX_HARM``; порог перебирается по всем встреченным оценкам.

    Args:
        junk_scores: Оценки «сор» 0…1 у кандидатов-сора (чем выше, тем увереннее «сор»).
        sign_scores: То же у кандидатов-знаков.

    Returns:
        ``(AUC, полнота сора при вреде ≤ MAX_HARM, порог)``; пустой класс — ``nan``.
    """
    if not len(junk_scores) or not len(sign_scores):
        return float("nan"), float("nan"), float("nan")
    greater = (junk_scores[:, None] > sign_scores[None, :]).mean()
    ties = (junk_scores[:, None] == sign_scores[None, :]).mean()
    auc = float(greater + ties / 2)
    best_recall, best_threshold = 0.0, 1.0
    for threshold in np.unique(np.concatenate([junk_scores, sign_scores, [-1.0]])):
        harm = float((sign_scores > threshold).mean())
        recall = float((junk_scores > threshold).mean())
        if harm <= MAX_HARM and recall > best_recall:
            best_recall, best_threshold = recall, float(threshold)
    return auc, best_recall, best_threshold


def load_seconds(meta: dict, usage: dict) -> float:
    """Время загрузки модели: из служебной строки судьи, а если судья его не знает (DeepSeek: воркер vLLM
    отчитывается только временем генерации) — весь прогон минус время генерации по промптам.

    Args:
        meta: Служебная часть результатов (``load_seconds``, ``timing``).
        usage: Замер прогона (``seconds``).

    Returns:
        Секунды.
    """
    if meta.get("load_seconds") is not None:
        return float(meta["load_seconds"])
    timing = meta.get("timing") or {}
    return max(0.0, float(usage.get("seconds", 0.0)) - sum(timing.values()))


def summary(cand_dir: Path, results_dir: Path) -> pd.DataFrame:
    """Таблица «судья × масштаб».

    Столбцы: полнота сора (сор признан сором), вред (знак признан сором), неясно (доля), точность «сор»,
    медиана и сумма секунд на кандидата, загрузка модели, пик видеопамяти и ОЗУ (из ``<судья>_usage.json``).

    Args:
        cand_dir: Папка кандидатов.
        results_dir: Папка ``<судья>.jsonl``.

    Returns:
        Таблица.
    """
    candidates = {c.id: c for c in load(cand_dir)}
    truth = {c.id: (c.truth, c.label) for c in candidates.values()}
    junk_total = sum(1 for t, _ in truth.values() if t == "junk")
    sign_total = sum(1 for t, _ in truth.values() if t == "sign")
    out = []
    for path in sorted(results_dir.glob("*.jsonl")):
        frame, meta = read_results(path, candidates)
        if frame.empty:
            continue
        usage_path = results_dir / f"{path.stem}_usage.json"
        usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
        frame["truth"] = frame.id.map(lambda i: truth.get(i, (None, None))[0])
        frame["label"] = frame.id.map(lambda i: truth.get(i, (None, None))[1])
        for scale, part in frame.groupby("scale"):
            junk = part[part.truth == "junk"]
            sign = part[part.truth == "sign"]
            called_junk = part[part.verdict == "junk"]
            auc, recall_at, threshold = ranking(junk.score.to_numpy(float), sign.score.to_numpy(float))
            out.append(
                {
                    "судья": path.stem,
                    "масштаб": scale,
                    "кандидатов": len(part),
                    "сор найден": f"{(junk.verdict == 'junk').sum()}/{junk_total}",
                    "полнота сора": round((junk.verdict == "junk").sum() / max(1, junk_total), 3),
                    "из них пометок": f"{((junk.verdict == 'junk') & (junk.label == 'M')).sum()}/{(junk.label == 'M').sum()}",
                    "вред (знак→сор)": f"{(sign.verdict == 'junk').sum()}/{sign_total}",
                    "точность сора": round((called_junk.truth == "junk").mean(), 3) if len(called_junk) else float("nan"),
                    "неясно": round((part.verdict == "unsure").mean(), 3),
                    "AUC": round(auc, 3),
                    f"полнота при вреде ≤{MAX_HARM:.0%}": round(recall_at, 3),
                    "порог «сор» >": round(threshold, 3),
                    "с/кандидат": round(float(part.seconds.median()), 3),
                    "загрузка, с": round(load_seconds(meta, usage), 1),
                    "весь прогон, с": round(float(usage.get("seconds", 0.0)), 1),
                    "VRAM пик, МБ": round(float(usage.get("vram_mb", 0.0))),
                    "ОЗУ пик, МБ": round(float(usage.get("rss_mb", 0.0))),
                }
            )
    return pd.DataFrame(out)


__all__ = ["read_results", "summary"]
