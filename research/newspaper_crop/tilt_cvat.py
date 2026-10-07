"""Эталон наклона сторон листа в CVAT: отбор кадров, задача с полилиниями текущих сторон, забор поправленных линий в CSV."""

from __future__ import annotations

import json
import logging
import shutil
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from research.newspaper_crop.frames import OUT_ROOT, SETS_DIR
from research.newspaper_crop.text_guard import side_index

log = logging.getLogger(__name__)

# Каталог картинок задачи внутри IMAGES_DIR CVAT (docker/.env): сервер берёт кадры только из смонтированного share.
CVAT_SHARE = Path("/home/felix/Projects/mts_markup/cvat_share")
SHARE_SUBDIR = Path("newspaper_crop") / "tilt_gt"
# Проект и задача CVAT.
PROJECT_NAME = "ЭГ: края листа (эталон наклона)"
TASK_NAME = "наклон сторон, ЭГ 1969–1970"
# Эталон после забора.
GT_CSV = SETS_DIR / "tilt_gt.csv"
# Список отобранных кадров (ключ и причина отбора).
PICK_CSV = SETS_DIR / "tilt_pick.csv"


class EdgeLabel(Enum):
    """Метки CVAT: полилиния истинного края листа у каждой стороны четырёхугольника."""

    TOP = "край_верх"
    RIGHT = "край_право"
    BOTTOM = "край_низ"
    LEFT = "край_лево"


SIDE_BY_LABEL = {EdgeLabel.TOP: "top", EdgeLabel.RIGHT: "right", EdgeLabel.BOTTOM: "bottom", EdgeLabel.LEFT: "left"}
LABEL_COLORS = {
    EdgeLabel.TOP: "#ff0000",
    EdgeLabel.RIGHT: "#00a000",
    EdgeLabel.BOTTOM: "#0050ff",
    EdgeLabel.LEFT: "#ff00ff",
}


def pick_frames(sides: pd.DataFrame, extra: list[str], seed: int = 0) -> pd.DataFrame:
    """Отбор кадров под эталон по таблице расхождений сторон (``tilt-candidates``).

    Группы:
    - ``consensus`` — обе оценки (точка схода и ближайшие признаки) одного знака, разброс точки схода < 0.2°, не меньше
      6 ближайших признаков, меньшее из расхождений > 1° — по 4 кадра на сторону (2 из пояса 1–2°, 2 круче);
    - ``border`` — сторона на рамке кадра (> 30%) и хоть одна оценка расходится > 0.5° — по 3 на сторону;
    - ``control`` — у всех сторон обе оценки < 0.5° — 8 кадров;
    - ``user`` — кадры, присланные пользователем.

    Args:
        sides: Таблица ``tilt_sides.csv``.
        extra: Ключи кадров пользователя.
        seed: Зерно случайного отбора.

    Returns:
        Таблица ``key, reason, side`` без повторов кадров.
    """
    rng = np.random.default_rng(seed)
    d = sides.copy()
    d["m"] = np.minimum(d.delta_vp.abs(), d.delta_local.abs())
    agree = (np.sign(d.delta_vp) == np.sign(d.delta_local)) & (d.sigma_vp < 0.2) & (d.count_local >= 6)
    rows, used = [dict(key=key, reason="user", side="") for key in extra], set(extra)
    for side in ("top", "right", "bottom", "left"):
        s = d[agree & (d.side == side)]
        rows += _take(s[(s.m > 1) & (s.m <= 2)], 2, "consensus_1_2", used, rng)
        rows += _take(s[s.m > 2], 2, "consensus_gt2", used, rng)
        wide = (d.delta_vp.abs() > 0.5) | (d.delta_local.abs() > 0.5)
        rows += _take(d[(d.side == side) & (d.border > 0.3) & wide], 3, "border", used, rng)
    worst = d.assign(a=d.delta_vp.abs(), b=d.delta_local.abs()).groupby("key")[["a", "b"]].max()
    calm = worst[(worst.a < 0.5) & (worst.b < 0.5)].index
    rows += _take(pd.DataFrame(dict(key=calm, side="")), 8, "control", used, rng)
    return pd.DataFrame(rows)


def _take(pool: pd.DataFrame, n: int, reason: str, used: set[str], rng: np.random.Generator) -> list[dict]:
    """До ``n`` случайных кадров пула, ещё не взятых (``used`` пополняется — общий учёт взятого у вызывающего).

    Args:
        pool: Строки сторон-кандидатов.
        n: Сколько кадров взять.
        reason: Причина отбора (пишется в строку).
        used: Уже взятые ключи.
        rng: Генератор.

    Returns:
        Строки ``key, reason, side``.
    """
    pool = pool[~pool.key.isin(used)].drop_duplicates("key")
    rows = []
    for i in rng.permutation(len(pool))[:n]:
        r = pool.iloc[i]
        used.add(r.key)
        rows.append(dict(key=r.key, reason=reason, side=r.side))
    return rows


def share_name(key: str) -> str:
    """Имя файла кадра в share CVAT: ``<группа>__<имя>.jpg``.

    Args:
        key: Ключ кадра.

    Returns:
        Имя файла.
    """
    return key.replace("/", "__") + ".jpg"


def prepare_images(keys: list[str]) -> list[str]:
    """Копирует превью без пальцев (полный размер превью 2944×4416 — координаты эталона в пикселях превью) в share.

    Args:
        keys: Ключи кадров.

    Returns:
        Пути внутри share (как их ждёт сервер CVAT), в порядке ключей.
    """
    target = CVAT_SHARE / SHARE_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    out = []
    for key in keys:
        dst = target / share_name(key)
        if not dst.exists():
            shutil.copy2(OUT_ROOT / "nofinger" / (key + ".jpg"), dst)
        out.append(str(SHARE_SUBDIR / share_name(key)))
    return out


def current_sides(key: str, run_name: str) -> dict[str, np.ndarray]:
    """Стороны четырёхугольника зоны прогона в пикселях превью.

    Args:
        key: Ключ кадра.
        run_name: Прогон с зонами.

    Returns:
        ``{имя стороны: точки (2, 2)}``.
    """
    info = json.loads((OUT_ROOT / "runs" / run_name / "zones" / (key + "_zone.json")).read_text())
    quad = np.array(info["quad_norm"]) * [2944, 4416]
    out = {}
    for name in ("top", "right", "bottom", "left"):
        i = side_index(quad, name)
        out[name] = np.array([quad[i], quad[(i + 1) % 4]])
    return out


def clip_segment(p: np.ndarray, q: np.ndarray, w: int = 2944, h: int = 4416) -> np.ndarray:
    """Обрезает отрезок рамкой кадра (CVAT не держит точки за кадром); если отрезок целиком снаружи — прижимает к рамке.

    Args:
        p: Начало.
        q: Конец.
        w: Ширина кадра.
        h: Высота кадра.

    Returns:
        Точки (2, 2).
    """
    t = np.linspace(0, 1, 2001)[:, None]
    pts = p + t * (q - p)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1)
    if inside.sum() >= 2:
        sel = pts[inside]
        return np.array([sel[0], sel[-1]])
    return np.clip(np.array([p, q]), [0, 0], [w - 1, h - 1])


def export_task(picks: pd.DataFrame, run_name: str, annotator: str = "user") -> int:
    """Заводит проект и задачу CVAT: кадры из share, предразметка — 4 полилинии текущих сторон на кадр.

    Args:
        picks: Отобранные кадры (``pick_frames``).
        run_name: Прогон с зонами (текущие стороны).
        annotator: Пользователь CVAT, на которого назначаются джобы.

    Returns:
        id задачи.
    """
    from cvat_sdk import models
    from cvat_sdk.core.proxies.tasks import ResourceType

    from ocr_utils.scan_markup.cvat.client import CvatSettings, make_cvat_client
    from ocr_utils.scan_markup.cvat.project import assign_annotator, frame_index_by_name, upload_preannotations

    keys = list(picks.key)
    resources = prepare_images(keys)
    with make_cvat_client(CvatSettings()) as client:
        project = next((p for p in client.projects.list() if p.name == PROJECT_NAME), None)
        if project is None:
            labels = [
                models.PatchedLabelRequest(name=label.value, type="polyline", color=LABEL_COLORS[label])
                for label in EdgeLabel
            ]
            project = client.projects.create(spec=models.ProjectWriteRequest(name=PROJECT_NAME, labels=labels))
        if any(t.name == TASK_NAME and t.project_id == project.id for t in client.tasks.list()):
            raise RuntimeError(f"задача {TASK_NAME!r} уже есть — разметку не затираю; заберите её tilt-import")
        task = client.tasks.create_from_data(
            spec=models.TaskWriteRequest(name=TASK_NAME, project_id=project.id),
            resource_type=ResourceType.SHARE,
            resources=resources,
            data_params={"image_quality": 95, "sorting_method": "predefined"},
        )
        label_ids = {lb.name: lb.id for lb in client.api_client.labels_api.list(project_id=project.id)[0].results}
        frames = frame_index_by_name(task)
        shapes = []
        for key, res in zip(keys, resources):
            frame = frames.get(Path(res).name, frames.get(res))
            for label in EdgeLabel:
                p, q = current_sides(key, run_name)[SIDE_BY_LABEL[label]]
                seg = clip_segment(p, q)
                shapes.append(
                    models.LabeledShapeRequest(
                        type="polyline", frame=frame, label_id=label_ids[label.value], points=seg.ravel().tolist()
                    )
                )
        upload_preannotations(task, shapes)
        assign_annotator(client, task, annotator)
        log.info("CVAT: задача %s, кадров %d, полилиний %d", task.id, len(keys), len(shapes))
        return task.id


def import_task(run_name: str) -> pd.DataFrame:
    """Забирает поправленные полилинии: по стороне — прямая по МНК через все точки полилинии (пиксели превью).

    Сторона, которую разметчик не тронул, совпадает с текущей (это тоже эталон: «сторона верна»); флаг ``changed``
    — линия сдвинута больше чем на 3 px хоть в одном конце.

    Args:
        run_name: Прогон с зонами (для флага ``changed``).

    Returns:
        Таблица ``key, side, x0, y0, x1, y1, changed`` (и пишется в :data:`GT_CSV`).
    """
    from ocr_utils.scan_markup.cvat.client import CvatSettings, make_cvat_client
    from ocr_utils.scan_markup.cvat.project import frame_index_by_name

    rows = []
    with make_cvat_client(CvatSettings()) as client:
        project = next(p for p in client.projects.list() if p.name == PROJECT_NAME)
        task = next(t for t in client.tasks.list() if t.name == TASK_NAME and t.project_id == project.id)
        names = {i: name for name, i in frame_index_by_name(task).items()}
        label_by_id = {lb.id: lb.name for lb in client.api_client.labels_api.list(project_id=project.id)[0].results}
        for shape in task.get_annotations().shapes:
            label = EdgeLabel(label_by_id[shape.label_id])
            key = Path(names[shape.frame]).stem.replace("__", "/")
            pts = np.array(shape.points, float).reshape(-1, 2)
            side = SIDE_BY_LABEL[label]
            p, q = clip_segment(*current_sides(key, run_name)[side])
            changed = min(np.abs(pts[[0, -1]] - [p, q]).max(), np.abs(pts[[0, -1]] - [q, p]).max()) > 3.0
            rows.append(dict(key=key, side=side, x0=pts[0, 0], y0=pts[0, 1], x1=pts[-1, 0], y1=pts[-1, 1],
                             points=json.dumps(pts.round(1).tolist()), changed=bool(changed)))  # fmt: skip
    df = pd.DataFrame(rows).sort_values(["key", "side"])
    df.to_csv(GT_CSV, index=False)
    return df
