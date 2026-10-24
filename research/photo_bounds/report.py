"""Сводка мер стенда: IoU и промахи сторон (p50/p95 модуля, доля срезанных и лишних сторон) по способам, поясам площади и классам края; распределения мер чёткости сторон."""

from __future__ import annotations

import numpy as np

from research.photo_bounds.measure import SIDES

# Сторона «срезана» или «с лишним», если промах против эталона по модулю больше этого, мм. Эталон CVAT точен до
# 8 px оригинала 600 dpi (0.34 мм) плюс небрежность разметчика (см. README: эталон бывает шире снимка на 1–2.5 мм).
SIDE_TOL_MM = 1.0


def _floats(rows: list[dict], key: str) -> np.ndarray:
    """Значения колонки как float (пустые пропускаются)."""
    return np.array([float(r[key]) for r in rows if r.get(key) not in (None, "")], np.float64)


def method_summary(rows: list[dict], method: str) -> dict:
    """Меры одного способа по строкам эталонных фото.

    Args:
        rows: Строки мер (без строк ``extra``).
        method: ``det`` или ``new``.

    Returns:
        Словарь мер: число фото, найдено (IoU > 0), IoU медиана/p10, IoU ≥ 0.9 доля, промахи сторон |мм| p50/p95,
        доля сторон срезано (< −``SIDE_TOL_MM``) и лишнее (> ``SIDE_TOL_MM``).
    """
    ious = _floats(rows, f"{method}_iou")
    deltas = np.concatenate([_floats([r for r in rows if r.get(f"{method}_iou", 0) not in ("", None)
                                      and float(r[f"{method}_iou"]) > 0], f"{method}_{s.value}_mm")
                             for s in SIDES])  # fmt: skip
    found = ious[ious > 0]
    return {
        "photos": len(rows),
        "found": int((ious > 0).sum()),
        "iou_median": round(float(np.median(found)), 4) if len(found) else None,
        "iou_p10": round(float(np.percentile(found, 10)), 4) if len(found) else None,
        "iou_ge_0_9": round(float((ious >= 0.9).mean()), 3) if len(ious) else None,
        "iou_ge_0_95": round(float((ious >= 0.95).mean()), 3) if len(ious) else None,
        "side_abs_p50_mm": round(float(np.percentile(np.abs(deltas), 50)), 2) if len(deltas) else None,
        "side_abs_p95_mm": round(float(np.percentile(np.abs(deltas), 95)), 2) if len(deltas) else None,
        "side_signed_p50_mm": round(float(np.median(deltas)), 2) if len(deltas) else None,
        "sides_cut": round(float((deltas < -SIDE_TOL_MM).mean()), 3) if len(deltas) else None,
        "sides_extra": round(float((deltas > SIDE_TOL_MM).mean()), 3) if len(deltas) else None,
    }


def tightness_summary(rows: list[dict], method: str) -> dict:
    """Теснота рамок способа без эталона: доля сторон, срезающих снимок (снаружи растр) и с лишним полем (внутри бумага).

    Args:
        rows: Строки мер.
        method: ``ref``, ``det`` или ``new``.

    Returns:
        ``{"sides": n, "cut_share": ..., "spare_share": ...}``.
    """
    from research.photo_bounds.measure import CUT_COVER, SPARE_COVER

    outs = np.concatenate([_floats(rows, f"{method}_{s.value}_out") for s in SIDES])
    ins = np.concatenate([_floats(rows, f"{method}_{s.value}_in") for s in SIDES])
    outs, ins = outs[np.isfinite(outs)], ins[np.isfinite(ins)]
    return {
        "method": method,
        "sides": len(outs),
        "cut_share": round(float((outs > CUT_COVER).mean()), 3) if len(outs) else None,
        "spare_share": round(float((ins < SPARE_COVER).mean()), 3) if len(ins) else None,
        "out_cover_p50": round(float(np.median(outs)), 3) if len(outs) else None,
        "in_cover_p50": round(float(np.median(ins)), 3) if len(ins) else None,
    }


def grouped(rows: list[dict], key: str) -> dict[str, list[dict]]:
    """Строки по значению колонки ``key`` (пустое — ``—``)."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get(key) or "—"), []).append(r)
    return out


def summary_rows(rows: list[dict], group_keys: tuple[str, ...] = ("band", "new_class", "set_class")) -> list[dict]:
    """Таблица сводки: все фото и по группам, по способам ``det`` и ``new``.

    Args:
        rows: Строки мер.
        group_keys: Колонки группировки.

    Returns:
        Строки сводки.
    """
    photos = [r for r in rows if not r.get("extra")]
    out = []
    for method in ("det", "new"):
        out.append({"group": "все", "value": "", "method": method, **method_summary(photos, method)})
        for key in group_keys:
            for value, part in sorted(grouped(photos, key).items()):
                out.append({"group": key, "value": value, "method": method, **method_summary(part, method)})
    return out


def sharpness_rows(rows: list[dict]) -> list[dict]:
    """Распределения мер чёткости сторон (покрытие, остаток, ступенька) по всем сторонам всех фото: процентили.

    Args:
        rows: Строки мер.

    Returns:
        Строки «мера, p10, p25, p50, p75, p90».
    """
    out = []
    for measure in ("cover", "resid_p90_mm", "contrast", "width_mm", "bow_mm"):
        values = np.concatenate([_floats(rows, f"{s.value}_{measure}") for s in SIDES])
        if not len(values):
            continue
        q = np.percentile(values, [10, 25, 50, 75, 90])
        out.append({"measure": measure, "n": len(values), **{f"p{p}": round(float(v), 3)
                                                              for p, v in zip((10, 25, 50, 75, 90), q)}})  # fmt: skip
    return out
