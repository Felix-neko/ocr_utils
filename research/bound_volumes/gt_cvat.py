"""Эталон маски верхних страниц в CVAT: отбор кадров, задача с полигоном книги SAM2 на правку, забор полигонов в маски."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from research.bound_volumes.frames import SETS_DIR, Labeled, read_bgr
from research.bound_volumes.geometry import cache_path, load_geometry, work_image

log = logging.getLogger(__name__)

# Каталог картинок задачи внутри IMAGES_DIR CVAT (docker/.env): сервер берёт кадры только из смонтированного share.
CVAT_SHARE = Path("/home/felix/Projects/mts_markup/cvat_share")
SHARE_SUBDIR = Path("bound_volumes") / "top_pages_gt"
# Масштаб картинок задачи относительно кадра: ½ (5813×4419 → 2906×2209, ~175 dpi) — кромки и полоски торца видны.
GT_SCALE = 0.5
# Проект и задача CVAT.
PROJECT_NAME = "МТО 1990: верхние страницы подшивки (эталон)"
TASK_NAME = "верхние страницы, МТО 1990, 40 кадров"
# Отобранные кадры и эталон после забора (маски — PNG в масштабе задачи).
PICK_CSV = SETS_DIR / "gt_pick.csv"
GT_DIR = SETS_DIR / "gt_masks"
# Отбор: кадров на класс, зерно.
PER_CLASS = 10
SEED = 0
# Упрощение контура предразметки, px масштаба задачи: вершины через 4–20 px, их удобно двигать.
APPROX_EPS_PX = 4.0
# Сглаживание маски книги перед контуром (σ, px масштаба задачи): зубцы маски SAM дают сотни лишних вершин.
SMOOTH_SIGMA_PX = 4.0
# Кадры, присланные пользователем (обязательно в эталон).
USER_KEYS = ("06/IMG_1296", "02/IMG_0896")


class GtLabel(Enum):
    """Метки CVAT."""

    # Полигон верхних страниц разворота (поднятая страница — вместе с плоской).
    TOP = "верхние_страницы"


def pick_frames(labeled: list[Labeled], per_class: int = PER_CLASS, seed: int = SEED) -> pd.DataFrame:
    """По ``per_class`` кадров на класс без повторов, кадры пользователя — обязательно, выпуски вперемешку.

    Args:
        labeled: Набор с классами.
        per_class: Кадров на класс.
        seed: Зерно.

    Returns:
        Таблица ``key, category`` (класс, по которому кадр взят; у кадров пользователя — ``user``).
    """
    rng = np.random.default_rng(seed)
    used = set(USER_KEYS)
    rows = [dict(key=k, category="user") for k in USER_KEYS]
    cats = sorted({c for it in labeled for c in it.categories}, key=lambda c: c.value)
    for cat in cats:
        pool = [it.frame.key for it in labeled if cat in it.categories and it.frame.key not in used]
        # Кадр пользователя с этим классом засчитывается в квоту.
        quota = per_class - sum(1 for k in USER_KEYS for it in labeled if it.frame.key == k and cat in it.categories)
        for i in rng.permutation(len(pool))[: max(quota, 0)]:
            used.add(pool[i])
            rows.append(dict(key=pool[i], category=cat.value))
    return pd.DataFrame(rows)


def share_name(key: str) -> str:
    """Имя файла кадра в share CVAT: ``<выпуск>__<имя>.jpg``.

    Args:
        key: Ключ кадра.

    Returns:
        Имя файла.
    """
    return key.replace("/", "__") + ".jpg"


def book_polygon(frame, shape: tuple[int, int]) -> np.ndarray:
    """Предразметка: внешний контур маски книги SAM2 (``layers.book_mask``) в масштабе задачи, упрощённый.

    Args:
        frame: Кадр.
        shape: Размер картинки задачи ``(H, W)``.

    Returns:
        Вершины ``(K, 2)`` float.
    """
    from research.bound_volumes.fingers import Guard, out_paths
    from research.bound_volumes.layers import book_mask

    small = work_image(read_bgr(out_paths(Guard.GLYPH, frame)[0]))
    book = book_mask(load_geometry(cache_path("glyph", frame)), cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    big = cv2.resize(book.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    big = cv2.GaussianBlur(big, (0, 0), SMOOTH_SIGMA_PX) > 0.5
    contours, _ = cv2.findContours(big.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    return cv2.approxPolyDP(contour, APPROX_EPS_PX, True)[:, 0, :].astype(float)


def export_task(picks: pd.DataFrame, annotator: str = "user") -> int:
    """Заводит проект и задачу CVAT: кадры (исходные, в ½) из share, предразметка — полигон книги на кадр.

    Args:
        picks: Отобранные кадры (``pick_frames``).
        annotator: Пользователь CVAT, на которого назначаются джобы.

    Returns:
        id задачи.

    Raises:
        RuntimeError: Если задача уже есть (разметку не затираем).
    """
    from cvat_sdk import models
    from cvat_sdk.core.proxies.tasks import ResourceType

    from ocr_utils.scan_markup.cvat.client import CvatSettings, make_cvat_client
    from ocr_utils.scan_markup.cvat.project import assign_annotator, frame_index_by_name, upload_preannotations
    from research.bound_volumes.frames import frame_by_key

    target = CVAT_SHARE / SHARE_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    resources, polygons = [], []
    for key in picks.key:
        frame = frame_by_key(key)
        img = read_bgr(frame.work_path)
        small = cv2.resize(img, None, fx=GT_SCALE, fy=GT_SCALE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(target / share_name(key)), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        resources.append(str(SHARE_SUBDIR / share_name(key)))
        polygons.append(book_polygon(frame, small.shape[:2]))
    with make_cvat_client(CvatSettings()) as client:
        project = next((p for p in client.projects.list() if p.name == PROJECT_NAME), None)
        if project is None:
            labels = [models.PatchedLabelRequest(name=GtLabel.TOP.value, type="polygon", color="#2060ff")]
            project = client.projects.create(spec=models.ProjectWriteRequest(name=PROJECT_NAME, labels=labels))
        if any(t.name == TASK_NAME and t.project_id == project.id for t in client.tasks.list()):
            raise RuntimeError(f"задача {TASK_NAME!r} уже есть — разметку не затираю; заберите её gt-import")
        task = client.tasks.create_from_data(
            spec=models.TaskWriteRequest(name=TASK_NAME, project_id=project.id),
            resource_type=ResourceType.SHARE,
            resources=resources,
            data_params={"image_quality": 95, "sorting_method": "predefined"},
        )
        label_ids = {lb.name: lb.id for lb in client.api_client.labels_api.list(project_id=project.id)[0].results}
        frames = frame_index_by_name(task)
        shapes = [
            models.LabeledShapeRequest(
                type="polygon",
                frame={Path(k).name: v for k, v in frames.items()}[Path(res).name],
                label_id=label_ids[GtLabel.TOP.value],
                points=poly.ravel().tolist(),
            )
            for res, poly in zip(resources, polygons)
        ]
        upload_preannotations(task, shapes)
        assign_annotator(client, task, annotator)
        log.info("CVAT: задача %s, кадров %d, вершин в среднем %.0f", task.id, len(resources),
                 np.mean([len(p) for p in polygons]))  # fmt: skip
        return task.id


def import_task() -> int:
    """Забирает полигоны: по кадру маска PNG в масштабе задачи (объединение всех полигонов кадра) в :data:`GT_DIR`.

    Returns:
        Число кадров с эталоном.
    """
    from ocr_utils.scan_markup.cvat.client import CvatSettings, make_cvat_client
    from ocr_utils.scan_markup.cvat.project import frame_index_by_name

    GT_DIR.mkdir(parents=True, exist_ok=True)
    masks: dict[str, np.ndarray] = {}
    with make_cvat_client(CvatSettings()) as client:
        project = next(p for p in client.projects.list() if p.name == PROJECT_NAME)
        task = next(t for t in client.tasks.list() if t.name == TASK_NAME and t.project_id == project.id)
        names = {i: name for name, i in frame_index_by_name(task).items()}
        for shape in task.get_annotations().shapes:
            name = Path(names[shape.frame]).name
            if name not in masks:
                img = cv2.imread(str(CVAT_SHARE / SHARE_SUBDIR / name), cv2.IMREAD_GRAYSCALE)
                masks[name] = np.zeros(img.shape, np.uint8)
            pts = np.array(shape.points, float).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(masks[name], [pts], 255)
    for name, mask in masks.items():
        cv2.imwrite(str(GT_DIR / name.replace(".jpg", ".png")), mask)
    return len(masks)
