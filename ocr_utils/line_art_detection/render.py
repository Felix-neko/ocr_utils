"""Рендер страницы PDF в полутоновый массив.

ПОЧЕМУ РЕНДЕР, А НЕ ИЗВЛЕЧЕНИЕ КАРТИНОК. Обычно страница такого PDF — одна
полностраничная JBIG2, и её можно было бы достать без перекодирования (так делает
``pdf_utils.extract_images``). Но после распрямления строк FineReader кладёт на страницу
НЕСКОЛЬКО фрагментов: на стр. 80 файла ``full_1967_01_bg_off_ori_off.pdf`` их два.
Собирать их обратно вручную незачем — этим занимается сам рендерер.

ПОЧЕМУ ПОЛНЫЕ 600 dpi НЕ ДОРОГИ. Замер на ``full_1966_01.pdf``: 0.17 с на страницу при
600 dpi против 0.14 с при 300 dpi. Время уходит в декодирование JBIG2, а не в масштаб,
поэтому уменьшать разрешение ради скорости смысла нет — и мы этого не делаем.
"""

import fitz
import numpy as np

# Разрешение рендера по умолчанию: родное разрешение сканов пака-1.
DEFAULT_DPI = 600


def render_page(page: fitz.Page, dpi: int = DEFAULT_DPI) -> np.ndarray:
    """Полутоновый рендер страницы в ``dpi`` точек на дюйм.

    Args:
        page: Страница открытого документа.
        dpi: Разрешение рендера.

    Returns:
        Массив ``uint8`` формы ``(height, width)``.
    """
    zoom = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)
