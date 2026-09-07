"""Страницы PDF из JPEG без перекодирования и врезки поверх уже готовых страниц.

Зачем это отдельным модулем. Обе сборки PDF в конвейере МТС — промежуточная (из
заострённых картинок, под FineReader) и финальная (из распознанных, с возвращёнными на
место иллюстрациями) — стоят на одних и тех же трёх операциях: положить JPEG в страницу,
не тронув его байты; вырезать кусок оригинала и закодировать его отдельной картинкой;
поставить эту картинку поверх страницы в нужное место.

ПОЧЕМУ БАЙТЫ НЕ ТРОГАЮТСЯ. JPEG внутри PDF хранится ровно тем же потоком DCT, каким он
лежит в файле: ``/Filter /DCTDecode`` и есть «внутри лежит JPEG». Поэтому вставка сводится
к записи байтов в поток и заполнению словаря — ни один пиксель не декодируется. Это не
оптимизация: текст на заострённой картинке уже вытянут на пределе, и лишний круг
JPEG-кодирования съел бы ровно те бледные перемычки букв, ради которых её и делали.

ЧЕГО PDF НЕ УМЕЕТ. Прогрессивный JPEG в ``DCTDecode`` не кладётся — просмотрщик покажет
пустоту или мусор. Такой файл здесь считается ошибкой, а не поводом молча перекодировать:
перекодирование — это как раз то, чего мы избегаем, и делать его втихую нельзя.

ГЕОМЕТРИЯ. Разметка приходит в пикселях оригинала (ось Y вниз, начало в левом верхнем
углу), а PDF рисует в пунктах (ось Y вверх, начало в левом нижнем углу mediabox), да ещё
и через ``/Rotate``. Пересчёт собран в одном месте — :func:`placement_matrix`, — чтобы
потребители не разводили по своим модулям четыре варианта одной и той же матрицы.
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pikepdf
from PIL import Image
from pikepdf import Array, Dictionary, Name, Pdf, Stream

logger = logging.getLogger(__name__)

# Маркеры начала кадра (SOF). 0xC4, 0xC8 и 0xCC — это DHT, JPG и DAC: они лежат в том же
# диапазоне, но кадр не начинают, и перепутать их — значит прочитать размеры из мусора.
_SOF_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
# Прогрессивные и арифметические варианты кадра.
_PROGRESSIVE_MARKERS = frozenset({0xC2, 0xC6, 0xCA, 0xCE})
# Маркеры без поля длины: RSTn, SOI, EOI, TEM.
_STANDALONE_MARKERS = frozenset({*range(0xD0, 0xDA), 0x01})

# Допуск на расхождение соотношения сторон страницы PDF и картинки, из которой она
# сделана. Берётся не с потолка: страница 3830x5892 px при 600 dpi — это 459.6x707.04 pt,
# и округление сторон до сотых даёт относительную ошибку порядка 1e-5. Порог на два
# порядка крупнее ловит настоящую подмену страницы, не цепляясь за округление.
ASPECT_TOLERANCE = 1e-3


class JpegError(ValueError):
    """JPEG не годится для вставки в PDF (не JPEG, обрезан, прогрессивный)."""


@dataclass(frozen=True)
class JpegInfo:
    """Размеры и цветность JPEG, снятые с маркеров без декодирования пикселей."""

    width: int
    height: int
    components: int
    progressive: bool

    @property
    def color_space(self) -> Name:
        """Имя цветового пространства PDF для этого числа компонент."""
        if self.components == 1:
            return Name.DeviceGray
        if self.components == 3:
            return Name.DeviceRGB
        raise JpegError(f"неподдерживаемое число компонент JPEG: {self.components}")


def read_jpeg_info(data: bytes) -> JpegInfo:
    """Размеры и цветность JPEG по его маркерам.

    Читает только заголовки: на полосе 3830x5892 полное декодирование стоило бы десятки
    миллисекунд и сотню мегабайт, а нужны отсюда четыре числа.
    """
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise JpegError("это не JPEG: нет маркера SOI")

    offset = 2
    size = len(data)
    while offset < size:
        if data[offset] != 0xFF:
            raise JpegError(f"потерян маркер на смещении {offset}")
        # Между сегментами допустимы заполняющие 0xFF — пропускаем их все.
        while offset < size and data[offset] == 0xFF:
            offset += 1
        if offset >= size:
            break
        marker = data[offset]
        offset += 1
        if marker in _STANDALONE_MARKERS:
            continue
        if marker == 0xDA:  # начало сканированных данных — дальше кадра уже не будет
            break
        if offset + 2 > size:
            raise JpegError("JPEG обрезан: нет длины сегмента")
        length = int.from_bytes(data[offset : offset + 2], "big")
        if marker in _SOF_MARKERS:
            if offset + 8 > size:
                raise JpegError("JPEG обрезан: кадр не помещается")
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            return JpegInfo(width, height, data[offset + 7], marker in _PROGRESSIVE_MARKERS)
        offset += length
    raise JpegError("в JPEG нет маркера кадра (SOF)")


def embed_jpeg(pdf: Pdf, data: bytes, info: "JpegInfo | None" = None) -> Stream:
    """JPEG -> image-XObject с ``/DCTDecode``; байты кладутся КАК ЕСТЬ.

    Прогрессивный JPEG отвергается: ``DCTDecode`` его не декодирует, и страница вышла бы
    пустой. Лучше уронить сборку выпуска, чем отдать в FineReader сотню белых полос.
    """
    info = info or read_jpeg_info(data)
    if info.progressive:
        raise JpegError("прогрессивный JPEG в PDF не встраивается — нужен baseline")
    stream = Stream(pdf, data)
    stream.Type = Name.XObject
    stream.Subtype = Name.Image
    stream.Width = info.width
    stream.Height = info.height
    stream.ColorSpace = info.color_space
    stream.BitsPerComponent = 8
    stream.Filter = Name.DCTDecode
    return stream


def encode_jpeg(array: np.ndarray, gray: bool, quality: int) -> bytes:
    """Кусок картинки -> baseline JPEG заданного качества, при ``gray`` — одноканальный.

    ``gray`` приходит снаружи и берётся из вида области (``COLOR_PICTURE_KINDS``), а не
    угадывается по содержимому: цветной набор бывает почти серым на вид, и решать за
    разметчика по пикселям значило бы иногда молча его обесцвечивать.
    """
    if array.ndim == 3 and array.shape[2] == 3:
        image = Image.fromarray(array, "RGB")
    elif array.ndim == 2:
        image = Image.fromarray(array, "L")
    else:
        raise ValueError(f"ожидался HxW или HxWx3, получено {array.shape}")
    if gray:
        image = image.convert("L")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    # progressive=False задан явно: PDF прогрессивный JPEG не примет, и полагаться на то,
    # что умолчание Pillow не сменится, тут нельзя.
    image.save(buffer, "JPEG", quality=quality, progressive=False, optimize=False)
    return buffer.getvalue()


def load_image(path: "Path") -> np.ndarray:
    """Полоса с диска как массив ``HxWx3`` или ``HxW``.

    Читается ЦЕЛИКОМ, даже когда нужен один кусок: TIFF пака сжат LZW, и вытащить из него
    прямоугольник, не разжав всю полосу, всё равно нельзя.
    """
    with Image.open(path) as image:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        return np.asarray(image)


def crop(array: np.ndarray, rect: "tuple[int, int, int, int]") -> np.ndarray:
    """Кусок массива по прямоугольнику в пикселях, с обрезкой по краям кадра."""
    x1, y1, x2, y2 = rect
    height, width = array.shape[:2]
    return array[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]


def placement_matrix(
    rect_px: "tuple[float, float, float, float]",
    page_px: "tuple[int, int]",
    mediabox: "tuple[float, float, float, float]",
    rotation: int = 0,
) -> "tuple[float, float, float, float, float, float]":
    """Матрица ``cm`` для картинки, занимающей ``rect_px`` полосы.

    ``rect_px`` — ``(x1, y1, x2, y2)`` в пикселях ОРИГИНАЛА, ось Y вниз; ``page_px`` —
    размер оригинала в пикселях; ``mediabox`` — ``(x0, y0, x1, y1)`` целевой страницы в
    пунктах; ``rotation`` — её ``/Rotate``.

    Содержимое рисуется в НЕПОВЁРНУТОМ пространстве mediabox, а ``/Rotate`` разворачивает
    уже готовую страницу при показе. Поэтому при ненулевом повороте видимая ширина отвечает
    ВЫСОТЕ mediabox, и прямоугольник приходится разворачивать вместе с осями — иначе
    врезка уедет на соседнюю сторону листа.
    """
    rotation = rotation % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"недопустимый /Rotate: {rotation}")

    x0, y0, x1, y1 = mediabox
    box_w, box_h = x1 - x0, y1 - y0
    # Размер страницы так, как её видит читатель: при повороте на четверть стороны меняются.
    visible_w, visible_h = (box_h, box_w) if rotation in (90, 270) else (box_w, box_h)

    width_px, height_px = page_px
    scale_x, scale_y = visible_w / width_px, visible_h / height_px
    if abs(scale_x - scale_y) > ASPECT_TOLERANCE * max(scale_x, scale_y):
        raise ValueError(
            f"страница {visible_w:.2f}x{visible_h:.2f} pt не подобна кадру {width_px}x{height_px} px "
            f"(масштабы {scale_x:.5f} и {scale_y:.5f}) — геометрия страницы изменилась"
        )

    vx1, vy1, vx2, vy2 = (rect_px[0] * scale_x, rect_px[1] * scale_y, rect_px[2] * scale_x, rect_px[3] * scale_y)
    w, h = vx2 - vx1, vy2 - vy1

    if rotation == 0:
        a, b, c, d, e, f = w, 0.0, 0.0, h, vx1, visible_h - vy2
    elif rotation == 90:
        a, b, c, d, e, f = 0.0, w, -h, 0.0, vy2, vx1
    elif rotation == 180:
        a, b, c, d, e, f = -w, 0.0, 0.0, -h, box_w - vx1, vy2
    else:  # 270
        a, b, c, d, e, f = 0.0, -w, h, 0.0, box_w - vy2, box_h - vx1
    return a, b, c, d, e + x0, f + y0


def _draw(name: Name, matrix: "tuple[float, float, float, float, float, float]") -> bytes:
    """Кусок содержимого, рисующий XObject по матрице. ``q``/``Q`` обязательны."""
    a, b, c, d, e, f = matrix
    return f"q {a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.6f} {f:.6f} cm {name} Do Q\n".encode()


@dataclass(frozen=True)
class Overlay:
    """Врезка: готовые байты JPEG и место под них в пикселях оригинала."""

    jpeg: bytes
    rect_px: "tuple[int, int, int, int]"


def make_page(
    pdf: Pdf,
    base_jpeg: bytes,
    dpi: float,
    overlays: "tuple[Overlay, ...]" = (),
    page_px: "tuple[int, int] | None" = None,
) -> pikepdf.Page:
    """Новая страница из ``base_jpeg`` во весь лист и врезок поверх него.

    Размер листа берётся из пикселей и ``dpi``, а не из подразумеваемых 72 dpi: страница
    600-точечного скана должна быть размером с бумажный лист, иначе FineReader увидит
    полосу метрового формата и станет искать на ней текст соответствующего кегля.

    ``page_px`` задаётся отдельно от размеров ``base_jpeg`` только для проверки: врезки
    размечены в пикселях ОРИГИНАЛА, и если заострённая копия оказалась другого размера,
    молча растянуть её нельзя.
    """
    info = read_jpeg_info(base_jpeg)
    if page_px is not None and (info.width, info.height) != tuple(page_px):
        raise JpegError(f"картинка {info.width}x{info.height} не совпадает с полосой {page_px[0]}x{page_px[1]}")
    page_px = (info.width, info.height)

    width_pt, height_pt = info.width * 72.0 / dpi, info.height * 72.0 / dpi
    mediabox = (0.0, 0.0, width_pt, height_pt)

    resources = Dictionary()
    xobjects = Dictionary()
    content = bytearray()

    base = embed_jpeg(pdf, base_jpeg, info)
    xobjects["/Im0"] = base
    content += _draw(Name("/Im0"), placement_matrix((0, 0, info.width, info.height), page_px, mediabox))

    for index, overlay in enumerate(overlays, start=1):
        name = Name(f"/Im{index}")
        xobjects[str(name)] = embed_jpeg(pdf, overlay.jpeg)
        content += _draw(name, placement_matrix(overlay.rect_px, page_px, mediabox))

    resources["/XObject"] = xobjects
    page = pikepdf.Page(
        pdf.make_indirect(
            Dictionary(
                Type=Name.Page, MediaBox=Array([*mediabox]), Resources=resources, Contents=Stream(pdf, bytes(content))
            )
        )
    )
    return page


def overlay_on_page(
    pdf: Pdf, page: pikepdf.Page, jpeg: bytes, rect_px: "tuple[int, int, int, int]", page_px: "tuple[int, int]"
) -> None:
    """Кладёт JPEG поверх УЖЕ существующей страницы, ничего на ней не трогая.

    Текстовый слой и шрифты распознанной страницы остаются как были: врезка — это ещё один
    XObject в ресурсах и ещё один кусок содержимого В КОНЦЕ. ``contents_add`` сам следит за
    тем, чтобы дописанное не унаследовало графическое состояние от предыдущего содержимого.
    """
    mediabox = tuple(float(v) for v in page.mediabox)
    rotation = int(page.obj.get("/Rotate", 0))
    matrix = placement_matrix(rect_px, page_px, mediabox, rotation)
    name = page.add_resource(embed_jpeg(pdf, jpeg), Name.XObject)
    page.contents_add(Stream(pdf, _draw(name, matrix)), prepend=False)
