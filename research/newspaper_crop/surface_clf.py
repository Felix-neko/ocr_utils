"""Классификатор «снаружи другая поверхность»: участки у сторон листа → признаки DINOv2 и цвета → свой лист или чужое (эталон CVAT)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from research.newspaper_crop.frames import OUT_ROOT
from research.newspaper_crop.text_guard import side_index
from research.newspaper_crop.zone import clean_paper_mask

# Каталог кэша участков и признаков.
SURFACE_DIR = OUT_ROOT / "surface"
# Прогон с трапециями листа до отрезки за складкой и защиты (маска SAM2.1 и квантиль 0.995).
DETECTED_RUN = "sam2_box_nofinger_all"
# Участок: глубина × длина вдоль стороны, px превью; шаг вдоль стороны.
CROP_DEPTH, CROP_ALONG, ALONG_STEP = 32, 160, 160
# Глубины центров участков от стороны внутрь, px превью: эталон уходит от стороны до ~190 px. Участки 16×128 с шагом
# 8 px (чтобы достать бахрому и волну в первых 40 px) хуже: AP 0.57 против 0.82 — DINOv2 не хватает контекста.
DEPTHS = tuple(range(16, 257, 24))
# Глубина эталона своей бумаги у стороны: участки с центрами на этих глубинах усредняются.
REF_DEPTHS = (330, 380)
# Наименьшая доля маски листа в участке: иначе это пол или фон, не «поверхность листа» (из-за этого ленты у рваной
# кромки, где лежат бахрома и волна, почти не попадают в участки).
MIN_PAPER = 0.8
# Размер входа DINOv2 (кратно патчу 14): глубина × длина.
DINO_SIZE = (56, 224)


class Surface(Enum):
    """Метка участка: свой (верхний) лист полосы или чужое (нижние листы, соседняя полоса, крышка, бахрома, тень)."""

    OWN = "own"
    OTHER = "other"


@dataclass(frozen=True)
class SideFrame:
    """Сторона трапеции в превью: начало, единичные направления вдоль и внутрь, длина."""

    p: np.ndarray
    along: np.ndarray
    inward: np.ndarray
    length: float


def side_frames(quad: np.ndarray) -> dict[str, SideFrame]:
    """Стороны трапеции по именам с нормалью внутрь.

    Args:
        quad: Вершины (4, 2) в пикселях превью.

    Returns:
        ``{left|right|top|bottom: SideFrame}``.
    """
    centre = quad.mean(axis=0)
    out = {}
    for name in ("top", "right", "bottom", "left"):
        i = side_index(quad, name)
        p, e = quad[i], quad[(i + 1) % 4]
        d = (e - p) / np.linalg.norm(e - p)
        n = np.array([-d[1], d[0]])
        if (centre - p) @ n < 0:
            n = -n
        out[name] = SideFrame(p, d, n, float(np.linalg.norm(e - p)))
    return out


def crop_at(img: np.ndarray, frame: SideFrame, u: float, depth: float, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Участок вдоль стороны: строки — глубина (наружу → внутрь), столбцы — вдоль стороны; центр в (u, depth).

    Args:
        img: Картинка (превью RGB или маска uint8).
        frame: Сторона.
        u: Положение центра вдоль стороны, px.
        depth: Глубина центра внутрь, px.
        interp: Интерполяция ``cv2.remap``.

    Returns:
        Массив (:data:`CROP_DEPTH`, :data:`CROP_ALONG`[, C]).
    """
    dv = np.arange(CROP_DEPTH, dtype=np.float32) - CROP_DEPTH / 2 + depth
    du = np.arange(CROP_ALONG, dtype=np.float32) - CROP_ALONG / 2 + u
    mx = frame.p[0] + du[None, :] * frame.along[0] + dv[:, None] * frame.inward[0]
    my = frame.p[1] + du[None, :] * frame.along[1] + dv[:, None] * frame.inward[1]
    return cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), interp, borderMode=cv2.BORDER_CONSTANT)


def gt_depth(frame: SideFrame, x0: float, y0: float, x1: float, y1: float, u: np.ndarray) -> np.ndarray:
    """Глубина эталонной прямой от стороны в точках ``u`` вдоль неё (продолжение прямой за её концы).

    Args:
        frame: Сторона.
        x0: Начало эталона, x.
        y0: Начало эталона, y.
        x1: Конец эталона, x.
        y1: Конец эталона, y.
        u: Положения вдоль стороны.

    Returns:
        Глубины (px превью).
    """
    a, b = np.array([x0, y0]), np.array([x1, y1])
    ua, ub = (a - frame.p) @ frame.along, (b - frame.p) @ frame.along
    va, vb = (a - frame.p) @ frame.inward, (b - frame.p) @ frame.inward
    return va + (vb - va) * (u - ua) / (ub - ua + 1e-9)


def detected_quad_and_paper(key: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Трапеция листа и маска листа до отрезки за складкой, в пикселях превью.

    Args:
        key: Ключ кадра.
        shape: (H, W) превью.

    Returns:
        (вершины (4, 2), маска uint8 0/1).
    """
    base = OUT_ROOT / "runs" / DETECTED_RUN / "zones" / key
    info = json.loads(Path(str(base) + "_zone.json").read_text())
    quad = np.array(info["quad_norm"]) * [shape[1], shape[0]]
    raw = cv2.imread(str(base) + "_raw.png", cv2.IMREAD_GRAYSCALE) > 0
    paper = clean_paper_mask(raw).astype(np.uint8)
    return quad, cv2.resize(paper, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def frame_crops(rgb: np.ndarray, key: str, gt: pd.DataFrame | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Участки кадра у всех четырёх сторон с метками по эталону (если он есть) и эталонные участки своей бумаги.

    Участки, перекрывающие эталонную прямую (центр ближе половины глубины участка), не метятся (``label`` пусто);
    участки с маской листа меньше :data:`MIN_PAPER` отбрасываются.

    Args:
        rgb: Превью RGB.
        key: Ключ кадра.
        gt: Строки эталона этого кадра (``side, x0, y0, x1, y1``) или None.

    Returns:
        (таблица ``key, side, u, depth, gt_depth, label, ref`` — ref=True у эталонных участков своей бумаги; стопка
        участков (N, :data:`CROP_DEPTH`, :data:`CROP_ALONG`, 3) uint8 в том же порядке).
    """
    quad, paper = detected_quad_and_paper(key, rgb.shape[:2])
    rows, crops = [], []
    for name, frame in side_frames(quad).items():
        us = np.arange(CROP_ALONG / 2, frame.length - CROP_ALONG / 2, ALONG_STEP)
        g = None if gt is None else gt[gt.side == name]
        gd = (
            np.full(len(us), np.nan)
            if g is None or g.empty
            else gt_depth(frame, g.x0.iloc[0], g.y0.iloc[0], g.x1.iloc[0], g.y1.iloc[0], us)
        )
        for u, gdu in zip(us, gd):
            for depth in DEPTHS + REF_DEPTHS:
                if crop_at(paper, frame, u, depth, cv2.INTER_NEAREST).mean() < MIN_PAPER:
                    continue
                label = ""
                if np.isfinite(gdu) and depth not in REF_DEPTHS:
                    if depth + CROP_DEPTH / 2 <= gdu:
                        label = Surface.OTHER.value
                    elif depth - CROP_DEPTH / 2 >= gdu:
                        label = Surface.OWN.value
                rows.append(dict(key=key, side=name, u=float(u), depth=depth, gt_depth=float(gdu), label=label,
                                 ref=depth in REF_DEPTHS))  # fmt: skip
                crops.append(crop_at(rgb, frame, u, depth))
    return pd.DataFrame(rows), np.array(crops, np.uint8).reshape(-1, CROP_DEPTH, CROP_ALONG, 3)


def color_features(crops: np.ndarray) -> np.ndarray:
    """Цвет и фактура участков: среднее и СКО Lab, доля тёмного (краска, тень), средний модуль градиента.

    Args:
        crops: (N, h, w, 3) RGB uint8.

    Returns:
        (N, 9) float32.
    """
    out = []
    for c in crops:
        lab = cv2.cvtColor(c, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        dark = float((gray < np.median(gray) - 40).mean())
        out.append(np.concatenate([lab.mean(0), lab.std(0), [dark, float(np.hypot(gx, gy).mean()), gray.std()]]))
    return np.array(out, np.float32)


class DinoEmbedder:
    """DINOv2-base на GPU (fp16): эмбеддинг участка — CLS и среднее по патчам."""

    def __init__(self, device: str = "cuda") -> None:
        """Грузит модель один раз на прогон.

        Args:
            device: Устройство torch.
        """
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def embed(self, crops: np.ndarray, batch: int = 64) -> np.ndarray:
        """Эмбеддинги участков.

        Args:
            crops: (N, h, w, 3) RGB uint8.
            batch: Размер пачки.

        Returns:
            (N, 1536) float32: CLS и среднее патчей.
        """
        torch = self.torch
        out = []
        for i in range(0, len(crops), batch):
            part = np.stack(
                [cv2.resize(c, DINO_SIZE[::-1], interpolation=cv2.INTER_LINEAR) for c in crops[i : i + batch]]
            )
            x = torch.from_numpy(part).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = ((x - self.mean) / self.std).half()
            with torch.no_grad():
                h = self.model(pixel_values=x).last_hidden_state.float()
            out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1536), np.float32)


def relative_features(table: pd.DataFrame, emb: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Признаки участка вместе с разностью к эталону своей бумаги у той же стороны (участки ``ref``).

    Args:
        table: Таблица участков одного или многих кадров (``key, side, ref``).
        emb: Эмбеддинги DINOv2 (N, 1536).
        col: Цветовые признаки (N, 9).

    Returns:
        (N, 2·1536 + 2·9): [emb, emb − ref_emb, col, col − ref_col]; без эталона у стороны разности нулевые.
    """
    feats = np.zeros((len(table), 2 * emb.shape[1] + 2 * col.shape[1]), np.float32)
    for (_, _), idx in table.groupby(["key", "side"]).indices.items():
        ref = idx[table.ref.values[idx]]
        e_ref = emb[ref].mean(0) if len(ref) else emb[idx].mean(0) * 0
        c_ref = col[ref].mean(0) if len(ref) else col[idx].mean(0) * 0
        has = float(len(ref) > 0)
        feats[idx] = np.hstack([emb[idx], has * (emb[idx] - e_ref), col[idx], has * (col[idx] - c_ref)])
    return feats


def out_of_fold(table: pd.DataFrame, feats: np.ndarray, folds: int = 5, max_depth: int = 160) -> np.ndarray:
    """Вероятность «чужого» у всех участков вне выборки: логистическая регрессия на признаках DINOv2 и разности к своей
    бумаге, разбиение по кадрам (кадр, на котором предсказывают, в обучение не входит).

    Args:
        table: Таблица участков (``key, label, depth``).
        feats: Признаки (:func:`relative_features`, берутся первые 2·1536 — DINOv2 и разность).
        folds: Число частей разбиения по кадрам.
        max_depth: Обучать только на участках не глубже, px: глубже — печать полосы, к краю отношения не имеет.

    Returns:
        Вероятности (N,).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = feats[:, :3072]
    keys = table.key.values
    train = (table.label.values != "") & (table.depth.values <= max_depth)
    y = (table.label.values == Surface.OTHER.value).astype(int)
    out = np.zeros(len(table))
    uniq = np.array(sorted(set(keys)))
    for tr_k, te_k in GroupKFold(folds).split(uniq, groups=uniq):
        tr, te = train & np.isin(keys, uniq[tr_k]), np.isin(keys, uniq[te_k])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
        out[te] = clf.fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return out


def side_boundary(side_crops: pd.DataFrame, tau: float, min_share: float) -> tuple[float, float]:
    """Граница своего листа у стороны по вероятностям участков: в каждом столбце — глубина, до которой от края подряд
    идёт «чужое» (p > ``tau``); прямая Хубером по столбцам с ненулевой глубиной, если их доля не меньше ``min_share``.

    Args:
        side_crops: Участки одной стороны (``u, depth, p``), без эталонных ``ref``.
        tau: Порог вероятности «чужого».
        min_share: Наименьшая доля столбцов с «чужим» у края, чтобы сдвигать сторону.

    Returns:
        (k, c): глубина границы ``k·u + c`` (px внутрь от стороны); (0, 0) — сторона остаётся.
    """
    us = np.sort(side_crops.u.unique())
    depth = np.zeros(len(us))
    for i, u in enumerate(us):
        col = side_crops[side_crops.u == u].sort_values("depth")
        for d, p in zip(col.depth, col.p):
            if p <= tau:
                break
            depth[i] = d + CROP_DEPTH / 2
    if len(us) < 3 or (depth > 0).mean() < min_share:
        return 0.0, 0.0
    pts = np.c_[us[depth > 0], depth[depth > 0]].astype(np.float32)
    if len(pts) < 2:
        return 0.0, float(pts[0, 1])
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    k = float(vy / vx)
    return k, float(y0 - k * x0)


def evaluate_sides(table: pd.DataFrame, gt: pd.DataFrame, tau: float, min_share: float) -> pd.DataFrame:
    """Ошибка стороны против эталона до (трапеция листа) и после правки по классификатору, px превью.

    Args:
        table: Участки с вероятностями ``p``.
        gt: Эталон (``key, side, x0, y0, x1, y1``).
        tau: Порог вероятности «чужого».
        min_share: Доля столбцов с «чужим», чтобы сдвигать сторону.

    Returns:
        Таблица ``key, side, base, err, moved``: base и err — наибольшее расхождение с эталоном на концах стороны.
    """
    rows = []
    for (key, side), g in table[~table.ref].groupby(["key", "side"]):
        r = gt[(gt.key == key) & (gt.side == side)].iloc[0]
        quad, _ = detected_quad_and_paper(key, (4416, 2944))
        frame = side_frames(quad)[side]
        k, c = side_boundary(g, tau, min_share)
        ends = np.array([0.0, frame.length])
        pred = np.maximum(k * ends + c, 0.0)
        ref = gt_depth(frame, r.x0, r.y0, r.x1, r.y1, ends)
        rows.append(dict(key=key, side=side, base=float(np.abs(ref).max()), err=float(np.abs(ref - pred).max()),
                         moved=bool(pred.max() > 0)))  # fmt: skip
    return pd.DataFrame(rows)
