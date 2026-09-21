"""Вход разбора: страница как картинка с вариантом, разрешением, ленивыми копиями и именем в кэше.

ЗАЧЕМ ОДИН КЛАСС. Детекторам нужна не «картинка», а набор её представлений: полный серый
кадр (растр считает точки сетки при 600 dpi), копия 150 dpi (линейки, штрих, повёрнутый
текст), битональная копия (детектор штриха и Docstrum), цветная копия 1/4 (цвет бумаги и
областей), кадр для surya. Раньше каждый потребитель готовил их сам и по-своему — отсюда
разные результаты одного детектора на одной странице. Здесь копии считаются один раз, лениво,
по единым правилам, и кадр surya для одной страницы всегда один и тот же, кем бы её ни подали.

ВАРИАНТ КАРТИНКИ — ЧАСТЬ КЛЮЧА КЭША. Одна и та же полоса бывает сканом, заострённой копией,
размытой копией, битональной страницей FineReader с коррекцией геометрии и без. Surya на них
отвечает по-разному, и кэшировать ответ надо на каждый вариант отдельно (:class:`Variant`).

СИСТЕМА КООРДИНАТ. «Родные» пиксели — это ``dpi`` картинки: у скана — тег TIFF, у страницы
PDF — разрешение образа FineReader (обычно 600). Все результаты разбора отдаются в родных
пикселях; копии меньшего разрешения — внутреннее дело детекторов.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ocr_utils.page_layout import WORK_DPI

# Длинная сторона кадра surya, px. Surya всё равно ресайзит вход под свой процессор, а на
# 21-36-мегапиксельных кадрах одна конвертация стоит секунд. Копия 150 dpi у журнальной полосы
# короче (~1400 px), у разворота А3 с камеры — длиннее, и тогда кадр ужимается до этой стороны.
SURYA_MAX_SIDE = 2048

# Разрешение, ниже которого тег файла считается мусором (0, 1, 72 «по умолчанию» у редакторов).
MIN_PLAUSIBLE_DPI = 72


class Variant(str, Enum):
    """Какое представление страницы подано: от него зависит ответ surya и ключ кэша."""

    SCAN = "scan"  # сырой скан из «Готовое» (TIFF после ScanTailor)
    SHARPENED = "sharpened"  # заострённая копия (Capture One)
    BLURRED = "blurred"  # после закраса разметки и размытия фона (scan_cleanup)
    FR_GEO = "fr_geo"  # страница PDF FineReader с коррекцией геометрии
    FR_NOGEO = "fr_nogeo"  # страница PDF FineReader без коррекции геометрии
    CAMERA = "camera"  # кадр с камеры до кадрирования (scan_cropping)


@dataclass(frozen=True)
class SourceStat:
    """Отпечаток файла-источника: по нему кэш узнаёт, что картинка та же, не читая её."""

    path: str
    size: int
    mtime: float
    page_index: int | None = None  # страница PDF (с нуля); у картинок None

    @classmethod
    def of(cls, path: Path, page_index: int | None = None) -> "SourceStat":
        stat = Path(path).stat()
        return cls(str(path), int(stat.st_size), float(stat.st_mtime), page_index)

    def to_json(self) -> dict:
        return {"path": self.path, "size": self.size, "mtime": self.mtime, "page_index": self.page_index}

    @classmethod
    def from_json(cls, payload: dict) -> "SourceStat":
        return cls(str(payload["path"]), int(payload["size"]), float(payload["mtime"]), payload.get("page_index"))


def read_image_dpi(path: Path) -> int | None:
    """Разрешение из тега файла или ``None``, если тега нет или он неправдоподобен."""
    from PIL import Image as PILImage

    try:
        with PILImage.open(path) as image:
            dpi = image.info.get("dpi")
    except OSError:
        return None
    if not dpi:
        return None
    value = int(round(float(dpi[0])))
    return value if value >= MIN_PLAUSIBLE_DPI else None


def read_image_size(path: Path) -> tuple[int, int]:
    """``(ширина, высота)`` из заголовка файла, без декодирования пикселей."""
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        return image.size


def otsu_bitonal(gray: np.ndarray) -> np.ndarray:
    """Серый кадр → битональный в шкале детекторов штриха: краска 0, бумага 255, порог Оцу."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return np.where(ink > 0, 0, 255).astype(np.uint8)


class PageImage:
    """Страница как источник кадров разных разрешений; пиксели читаются и считаются лениво.

    Собирается фабриками :meth:`from_file`, :meth:`from_array`, :meth:`from_pdf_page`.
    Копии кэшируются внутри объекта; объект переживает pickle (в воркер уезжают только уже
    посчитанные массивы, лениво загружаемый файл в дочернем процессе перечитывается заново).
    """

    def __init__(
        self,
        *,
        variant: Variant,
        dpi: int,
        cache_name: str | None,
        width: int,
        height: int,
        source: SourceStat | None = None,
        bgr: np.ndarray | None = None,
        gray: np.ndarray | None = None,
        loader: Callable[[], np.ndarray] | None = None,
        renderer: Callable[[int], np.ndarray] | None = None,
    ) -> None:
        """Прямой конструктор — для фабрик ниже; вызывающему коду нужны они.

        Args:
            variant: Вариант картинки (см. :class:`Variant`).
            dpi: Родное разрешение — в его пикселях отдаются результаты.
            cache_name: Имя страницы в кэше surya (``1966/01/IMG_0003_2R``, ``full_1967_01/p0079``);
                ``None`` — кэш не используется.
            width: Ширина родного кадра.
            height: Высота родного кадра.
            source: Отпечаток файла-источника, если он есть.
            bgr: Уже декодированный цветной кадр (BGR, родное разрешение).
            gray: Уже декодированный серый кадр (родное разрешение).
            loader: Функция, декодирующая цветной кадр BGR по требованию (файлы).
            renderer: Функция ``dpi -> серый кадр`` — рендер страницы PDF в заданном разрешении.
        """
        self.variant = Variant(variant)
        self.dpi = int(dpi)
        self.cache_name = cache_name
        self.width = int(width)
        self.height = int(height)
        self.source = source
        self._bgr = bgr
        self._gray = gray
        self._loader = loader
        self._renderer = renderer
        self._rendered = renderer is not None  # размеры копий считаются по правилу PyMuPDF
        self._gray_at: dict[int, np.ndarray] = {}
        self._bgr_at: dict[int, np.ndarray] = {}
        self._bitonal_at: dict[int, np.ndarray] = {}
        self._surya_frame: np.ndarray | None = None

    # --- Фабрики -----------------------------------------------------------------

    @classmethod
    def from_file(
        cls, path: Path, variant: Variant = Variant.SCAN, cache_name: str | None = None, default_dpi: int | None = None
    ) -> "PageImage":
        """Картинка из файла; пиксели читаются при первом обращении, размер — из заголовка.

        Args:
            path: Файл картинки.
            variant: Вариант картинки.
            cache_name: Имя в кэше surya; ``None`` — не кэшировать.
            default_dpi: Разрешение, если тега в файле нет или он неправдоподобен.

        Returns:
            Страница с ленивой загрузкой.

        Raises:
            ValueError: Разрешение не известно ни из тега, ни из ``default_dpi``: подставить
                его наугад значило бы промахнуться в разы во всех размерах детекторов.
        """
        path = Path(path)
        dpi = read_image_dpi(path)
        if dpi is None:
            if default_dpi is None:
                raise ValueError(f"{path}: нет осмысленного тега разрешения, а default_dpi не задан")
            dpi = int(default_dpi)
        width, height = read_image_size(path)

        return cls(
            variant=variant,
            dpi=dpi,
            cache_name=cache_name,
            width=width,
            height=height,
            source=SourceStat.of(path),
            loader=lambda: _load_bgr(path),
        )

    @classmethod
    def from_array(
        cls,
        image: np.ndarray,
        dpi: int,
        variant: Variant,
        cache_name: str | None = None,
        source: SourceStat | None = None,
    ) -> "PageImage":
        """Картинка из готового массива: BGR ``(H, W, 3)`` или серого ``(H, W)``.

        Args:
            image: Кадр в родном разрешении.
            dpi: Его разрешение.
            variant: Вариант картинки.
            cache_name: Имя в кэше surya.
            source: Отпечаток файла, из которого массив получен, если известен; без него
                попадание в кэш проверяется по дайджесту кадра surya.

        Returns:
            Страница с готовым кадром.
        """
        if image.ndim == 3:
            bgr, gray = image, None
        elif image.ndim == 2:
            bgr, gray = None, image
        else:
            raise ValueError(f"ожидался кадр (H, W) или (H, W, 3), получен {image.shape}")
        height, width = image.shape[:2]
        return cls(
            variant=variant,
            dpi=dpi,
            cache_name=cache_name,
            width=width,
            height=height,
            source=source,
            bgr=bgr,
            gray=gray,
        )

    @classmethod
    def from_pdf_page(
        cls, document, index: int, variant: Variant, cache_name: str | None = None, native_dpi: int | None = None
    ) -> "PageImage":
        """Страница PDF (PyMuPDF): рендер серым в нужном разрешении по требованию.

        Args:
            document: Открытый ``fitz.Document``.
            index: Номер страницы с нуля.
            variant: Вариант (``FR_GEO`` / ``FR_NOGEO``).
            cache_name: Имя в кэше; по умолчанию ``<stem>/p<index:04d>``.
            native_dpi: Родное разрешение; по умолчанию — разрешение крупнейшего образа на
                странице (FineReader кладёт скан образом), иначе 600.

        Returns:
            Страница с ленивым рендером.
        """
        import fitz

        page = document[index]
        if native_dpi is None:
            native_dpi = main_image_dpi(page) or 600
        rect = page.rect
        width = int(round(rect.width / 72.0 * native_dpi))
        height = int(round(rect.height / 72.0 * native_dpi))
        path = Path(document.name) if document.name else None
        if cache_name is None and path is not None:
            cache_name = f"{path.stem}/p{index:04d}"

        def render(dpi: int) -> np.ndarray:
            zoom = dpi / 72.0
            pixmap = document[index].get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
            return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width).copy()

        # Открытый документ в замыкании: страница рендерится без повторного открытия файла.
        return cls(
            variant=variant,
            dpi=int(native_dpi),
            cache_name=cache_name,
            width=width,
            height=height,
            source=SourceStat.of(path, index) if path is not None and path.is_file() else None,
            renderer=render,
        )

    @classmethod
    def from_pdf_file(
        cls, path: Path, index: int, variant: Variant, cache_name: str | None = None, native_dpi: int | None = None
    ) -> "PageImage":
        """Страница PDF по пути: документ открывается на каждый рендер (объект живёт дольше документа — пул, prefill)."""
        import fitz

        path = Path(path)
        with fitz.open(str(path)) as document:
            page = document[index]
            if native_dpi is None:
                native_dpi = main_image_dpi(page) or 600
            rect = page.rect
        width = int(round(rect.width / 72.0 * native_dpi))
        height = int(round(rect.height / 72.0 * native_dpi))
        return cls(
            variant=variant,
            dpi=int(native_dpi),
            cache_name=cache_name if cache_name is not None else f"{path.stem}/p{index:04d}",
            width=width,
            height=height,
            source=SourceStat.of(path, index),
            renderer=lambda dpi: _render_pdf_gray(path, index, dpi),
        )

    # --- Кадры -------------------------------------------------------------------

    @property
    def has_color(self) -> bool:
        """Есть ли у страницы цветной кадр (у страниц PDF и серых массивов — нет)."""
        return self._bgr is not None or self._loader is not None

    @property
    def bgr(self) -> np.ndarray:
        """Цветной кадр BGR в родном разрешении.

        Raises:
            ValueError: У страницы только серый кадр.
        """
        if self._bgr is None:
            if self._loader is None:
                raise ValueError("у страницы нет цветного кадра")
            self._bgr = self._loader()
            self.height, self.width = self._bgr.shape[:2]
        return self._bgr

    @property
    def gray(self) -> np.ndarray:
        """Серый кадр в родном разрешении."""
        if self._gray is None:
            if self._renderer is not None:
                self._gray = self._renderer(self.dpi)
            else:
                self._gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return self._gray

    def size_at(self, dpi: int) -> tuple[int, int]:
        """``(ширина, высота)`` копии в разрешении ``dpi``.

        Страницы PDF рендерятся PyMuPDF, а он берёт охватывающий целый прямоугольник (округление
        вверх); копии картинок режутся ``cv2.resize`` с обычным округлением. Правило здесь то же,
        что у самого кадра, иначе размер из заголовка разошёлся бы с размером пикселей.
        """
        if dpi == self.dpi:
            return self.width, self.height
        k = dpi / self.dpi
        if self._rendered:
            return max(1, math.ceil(self.width * k - 1e-6)), max(1, math.ceil(self.height * k - 1e-6))
        return max(1, int(round(self.width * k))), max(1, int(round(self.height * k)))

    def scale_to_native(self, dpi: int) -> float:
        """Во сколько раз родной кадр крупнее копии ``dpi``."""
        return self.dpi / dpi

    def gray_at(self, dpi: int = WORK_DPI) -> np.ndarray:
        """Серая копия в разрешении ``dpi``: страницы PDF рендерятся прямо в нём, остальное уменьшается INTER_AREA."""
        if dpi == self.dpi:
            return self.gray
        if dpi not in self._gray_at:
            if self._renderer is not None and self._gray is None:
                self._gray_at[dpi] = self._renderer(dpi)
            else:
                self._gray_at[dpi] = cv2.resize(self.gray, self.size_at(dpi), interpolation=cv2.INTER_AREA)
        return self._gray_at[dpi]

    def bgr_at(self, dpi: int = WORK_DPI) -> np.ndarray:
        """Цветная копия BGR в разрешении ``dpi``; у серых страниц — серый, размноженный в три канала."""
        if not self.has_color:
            gray = self.gray_at(dpi)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if dpi == self.dpi:
            return self.bgr
        if dpi not in self._bgr_at:
            self._bgr_at[dpi] = cv2.resize(self.bgr, self.size_at(dpi), interpolation=cv2.INTER_AREA)
        return self._bgr_at[dpi]

    def bitonal_at(self, dpi: int = WORK_DPI) -> np.ndarray:
        """Битональная копия (краска 0, бумага 255) порогом Оцу по серой копии ``dpi``."""
        if dpi not in self._bitonal_at:
            self._bitonal_at[dpi] = otsu_bitonal(self.gray_at(dpi))
        return self._bitonal_at[dpi]

    @property
    def surya_dpi(self) -> int:
        """Разрешение кадра surya: ``WORK_DPI``, но так, чтобы длинная сторона не превысила ``SURYA_MAX_SIDE``."""
        width, height = self.size_at(WORK_DPI)
        longest = max(width, height)
        if longest <= SURYA_MAX_SIDE:
            return WORK_DPI
        return max(1, int(WORK_DPI * SURYA_MAX_SIDE / longest))

    @property
    def surya_frame(self) -> np.ndarray:
        """Кадр для surya: RGB uint8 в разрешении :attr:`surya_dpi`. Один и тот же для любого вызывающего."""
        if self._surya_frame is None:
            dpi = self.surya_dpi
            if self.has_color:
                self._surya_frame = cv2.cvtColor(self.bgr_at(dpi), cv2.COLOR_BGR2RGB)
            else:
                self._surya_frame = cv2.cvtColor(self.gray_at(dpi), cv2.COLOR_GRAY2RGB)
        return self._surya_frame

    def surya_frame_digest(self) -> str:
        """Дайджест кадра surya (blake2b): им кэш проверяет картинку без отпечатка файла."""
        frame = self.surya_frame
        digest = hashlib.blake2b(digest_size=16)
        digest.update(f"{frame.shape[1]}x{frame.shape[0]}".encode())
        digest.update(np.ascontiguousarray(frame).tobytes())
        return digest.hexdigest()

    def drop_full_frames(self) -> None:
        """Отпустить полный кадр, оставив копии рабочего разрешения и кадр surya (перед pickle в родителя)."""
        self._bgr = self._gray = None

    def drop_pixels(self) -> None:
        """Забыть все кадры (перед отправкой объекта из воркера в родителя, когда нужен только кадр surya)."""
        keep = self._surya_frame
        self._bgr = self._gray = None
        self._gray_at.clear()
        self._bgr_at.clear()
        self._bitonal_at.clear()
        self._surya_frame = keep

    # --- pickle: функции-замыкания не переезжают, кадры — переезжают -----------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_loader"] = None
        state["_renderer"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """После pickle источник восстанавливается по ``source``: файл перечитывается, PDF переоткрывается."""
        self.__dict__.update(state)
        if self._bgr is not None or self._gray is not None or self.source is None:
            return
        path = Path(self.source.path)
        if self.source.page_index is None:
            self._loader = lambda: _load_bgr(path)
        else:
            self._renderer = lambda dpi: _render_pdf_gray(path, self.source.page_index, dpi)


def _load_bgr(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"{path}: не читается как изображение")
    return bgr


def _render_pdf_gray(path: Path, index: int, dpi: int) -> np.ndarray:
    """Серый рендер страницы PDF в ``dpi``: единственное место, где страница превращается в пиксели."""
    import fitz

    zoom = dpi / 72.0
    with fitz.open(str(path)) as document:
        pixmap = document[index].get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width).copy()


def main_image_dpi(page) -> int | None:
    """Разрешение крупнейшего образа страницы PDF по его размеру в пикселях и на странице."""
    try:
        infos = page.get_image_info()
    except Exception:  # noqa: BLE001 — страница без образов или битая: ответ «не знаю»
        return None
    if not infos:
        return None
    main = max(infos, key=lambda info: info["width"] * info["height"])
    bbox = main.get("bbox")
    if not bbox:
        return None
    width_pt = float(bbox[2]) - float(bbox[0])
    if width_pt <= 0:
        return None
    return int(round(float(main["width"]) / width_pt * 72.0))
