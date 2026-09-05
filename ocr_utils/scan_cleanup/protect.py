"""Защитная маска из размеченных в базе растровых областей.

Отличие от ``background_smoothing.layout``: там иллюстрации ищет Surya и её блоки
приходится поправлять детектором растра (визуальный блок может срезать край
фотографии). Здесь границы уже размечены руками и выверены в CVAT — поправлять
нечего, прямоугольник берётся как есть.

ЧТО ЗАЩИЩАЕМ И ЧТО НЕТ:

* ``color`` и ``grayscale`` — настоящие иллюстрации, размытие выело бы их фактуру
  островами;
* ``color_text`` — НЕ защищаем: это буквы, набранные цветной краской, их и так
  ловит бинаризация по яркости, а сплошной прямоугольник вокруг оставил бы
  неразмытым весь фон между ними;
* ``stamp_suspect`` — по умолчанию НЕ защищаем: это прямоугольники автодетекции,
  которые разметчик как раз НЕ подтвердил как печать. Флаг оставлен, потому что
  решение спорное: пропущенная бледная печать сохранит вокруг себя «перец».

Маски под удаление (печати, надписи, прочее) не защищаются никогда: к моменту
размытия они уже закрашены, защищать там нечего, а размытие заплатки как раз и
прячет шов.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from ocr_utils.scan_cleanup.source import PageMarkup, Rect
from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_GRAYSCALE, KIND_STAMP_SUSPECT

# Виды растровых областей, защищаемых целиком: PICTURE_KINDS без color_text.
PROTECT_REGION_KINDS = (KIND_COLOR, KIND_GRAYSCALE)

# Доля площади полосы, начиная с которой область считается полосной, даже если
# флаг ``full_page`` в базе не проставлен. Порог низким быть не может: у обложек
# и вкладок картинка занимает почти весь кадр (замер по паку-1: все 166 областей
# с флагом крупнее 0.9 полосы).
FULL_PAGE_MIN_FRAC = 0.9


@dataclass
class ProtectOptions:
    """Что попадает в защитную маску размытия."""

    region_kinds: "tuple[str, ...]" = field(default=PROTECT_REGION_KINDS)
    protect_stamp_suspect: bool = False
    full_page_min_frac: float = FULL_PAGE_MIN_FRAC

    def kinds(self) -> "tuple[str, ...]":
        if self.protect_stamp_suspect:
            return tuple(self.region_kinds) + (KIND_STAMP_SUSPECT,)
        return tuple(self.region_kinds)


def rects_mask(shape: "tuple[int, int]", rects: "list[Rect] | tuple[Rect, ...]") -> np.ndarray:
    """Прямоугольники → маска uint8 0/255. Пустой список → пустая маска."""
    mask = np.zeros(shape, np.uint8)
    for r in rects:
        x1, y1 = max(0, r.x1), max(0, r.y1)
        x2, y2 = min(shape[1], r.x2), min(shape[0], r.y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def is_full_page(markup: PageMarkup, opts: "ProtectOptions | None" = None) -> "Rect | None":
    """Полосная иллюстрация (обложка, вкладка) или ``None``.

    Такую полосу размывать нельзя вовсе: размывать там нечего, весь кадр —
    содержимое. Закрас при этом никто не отменяет: по паку-1 больше половины
    масок под удаление стоит именно на обложках.

    Проверяется и флаг ``full_page`` из базы, и доля площади: флаг ставит
    детекция, а полоса могла быть обведена руками уже после.
    """
    opts = opts or ProtectOptions()
    page_area = max(1, markup.width * markup.height)
    for r in markup.regions_of(tuple(PROTECT_REGION_KINDS)):
        if r.full_page or r.area >= opts.full_page_min_frac * page_area:
            return r
    return None


def build_protect(
    shape: "tuple[int, int]", markup: PageMarkup, opts: "ProtectOptions | None" = None
) -> "tuple[np.ndarray, tuple[Rect, ...]]":
    """Маска областей, защищаемых целиком, и сами эти области (для оверлея)."""
    opts = opts or ProtectOptions()
    rects = markup.regions_of(opts.kinds())
    return rects_mask(shape, rects), rects


def analysis_roi(shape: "tuple[int, int]", markup: PageMarkup) -> "np.ndarray | None":
    """Область, по которой считать пороги: полоса минус ВСЕ растровые области.

    Минус все, а не только защищаемые: средние тона фотографии тянут порог Оцу
    вверх независимо от того, защищаем мы её потом или нет, и часть бумаги вокруг
    текста уехала бы под маску. ``None``, если растровых областей на полосе нет, —
    тогда пороги считаются по всему кадру.
    """
    if not markup.regions:
        return None
    covered = rects_mask(shape, markup.regions)
    return cv2.bitwise_not(covered)
