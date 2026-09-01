"""Отладочные оверлеи: уменьшенная полоса с обведёнными областями и подписями.

Без них пороги не откалибровать: число в базе не говорит, ту ли область оно описывает.

Имя файла оверлея — это КОНТРАКТ, а не деталь. Разложенные по папкам оверлеи служат
валидационной выборкой (``scan_markup.validation``), и разбор имени обратно в ``rel_path``
обязан лежать здесь же, рядом со сборкой: разъехавшись, эти две функции испортят выборку
молча.
"""

from pathlib import Path

import cv2

from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_COLOR_TEXT, KIND_GRAYSCALE, KIND_STAMP_SUSPECT

# Длинная сторона оверлея. 1000 px хватает, чтобы глазами отличить фотографию от штрихового
# рисунка и увидеть, куда легла рамка; больше — только место на диске.
OVERLAY_SIDE = 1000
OVERLAY_QUALITY = 85

# Цвет рамки по типу области, BGR. Те же цвета, что у меток проекта CVAT
# (``cvat.project.LABELS``), чтобы оверлей и разметка читались одинаково: зелёный — растр
# цветной, голубой — растр серый, оранжевый — подозрение на печать.
#
# Раньше здесь было «зелёный, если color, иначе голубой», и подозрение на печать сливалось
# с серым растром — то есть ровно то, что надо различать глазами, выглядело одинаково.
BOX_COLORS = {
    KIND_COLOR: (0, 230, 118),  # #00E676
    KIND_GRAYSCALE: (255, 176, 0),  # #00B0FF
    KIND_STAMP_SUSPECT: (0, 109, 255),  # #FF6D00
    KIND_COLOR_TEXT: (98, 17, 197),  # #C51162
}
UNKNOWN_KIND_COLOR = (255, 255, 255)


def overlay_name(rel_path: str) -> str:
    """``1969/12/IMG_0115_2R.tif`` -> ``1969__12__IMG_0115_2R.tif.jpg``."""
    return f"{rel_path.replace('/', '__')}.jpg"


def overlay_to_rel_path(name: str) -> str:
    """Обратное к :func:`overlay_name`; принимает и имя файла, и путь к нему."""
    stem = Path(name).name
    if stem.endswith(".jpg"):
        stem = stem[: -len(".jpg")]
    return stem.replace("__", "/")


def region_label(region) -> str:
    """Подпись к прямоугольнику: тип, разброс хроматичности и признак «во всю полосу»."""
    parts = [region.kind]
    if region.chroma_spread is not None:
        parts.append(f"{region.chroma_spread:.1f}")
    if region.dot_frac is not None:
        parts.append(f"d{region.dot_frac:.2f}")
    if region.full_page:
        parts.append("FULL")
    return " ".join(parts)


def write_debug_overlay(debug_dir: Path, rel_path: str, image_path: Path, regions) -> Path | None:
    """Пишет оверлей по полосе; ``None``, если исходник не прочитался."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    scale = OVERLAY_SIDE / max(bgr.shape[:2])
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for region in regions:
        x1, y1, x2, y2 = region.box
        color = BOX_COLORS.get(region.kind, UNKNOWN_KIND_COLOR)
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(small, p1, p2, color, 2)
        cv2.putText(
            small, region_label(region), (p1[0] + 4, p1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    out = debug_dir / overlay_name(rel_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, OVERLAY_QUALITY])
    return out
