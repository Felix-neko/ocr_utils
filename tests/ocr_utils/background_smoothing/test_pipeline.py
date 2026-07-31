"""Проверки сквозного прохода по файлам: что попадает на выход и в каком виде.

Главное, что здесь охраняется: кадр, на котором контент не выделяется (обложка,
растровая вкладка, чистый лист), доходит до выхода НЕТРОНУТЫМ, а не размытым
целиком — размывать такое нельзя, оно обрабатывается отдельным трактом.
"""

import cv2
import numpy as np

from ocr_utils.background_smoothing.pipeline import SmoothParams, process_frame

PAPER = 250
INK = 40


def _write(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return img


def _text_page(h: int = 400, w: int = 600) -> np.ndarray:
    img = np.full((h, w, 3), PAPER, dtype=np.uint8)
    for y in range(40, h - 40, 40):
        img[y : y + 12, 50 : w - 50] = INK
    return img


def _cover(h: int = 400, w: int = 600) -> np.ndarray:
    """Пёстрая малоконтрастная картинка без разделения на чернила и бумагу."""
    yy, xx = np.mgrid[0:h, 0:w]
    flat = np.clip(190 + 20 * np.sin(xx / 30.0) + 15 * np.cos(yy / 25.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)


def _params(tmp_path, **kw) -> SmoothParams:
    return SmoothParams(input_dir=tmp_path / "in", output_dir=tmp_path / "out", **kw)


def test_cover_is_copied_unchanged(tmp_path):
    """Кадр без выделяемого контента копируется побитово, а не размывается."""
    src = _write(tmp_path / "in" / "cover.png", _cover())
    process_frame(tmp_path / "in" / "cover.png", _params(tmp_path))
    out = cv2.imread(str(tmp_path / "out" / "cover.png"))
    assert np.array_equal(out, src)


def test_halftone_page_is_copied_unchanged(tmp_path):
    """Страница с крупной растровой зоной не сглаживается, хотя контент на ней есть.

    Зерно растра — это само изображение, и размытие вне защитной маски выело бы его
    островами (проверено на обложке IMG_0104_2R из 1966/03).
    """
    page = _text_page()
    page[100:300, 100:400] = 170
    src = _write(tmp_path / "in" / "halftone.png", page)
    process_frame(tmp_path / "in" / "halftone.png", _params(tmp_path))
    out = cv2.imread(str(tmp_path / "out" / "halftone.png"))
    assert np.array_equal(out, src)


def test_text_page_is_actually_processed(tmp_path):
    """На текстовой странице фон всё-таки меняется — проверка контраста не глушит работу."""
    src = _write(tmp_path / "in" / "page.png", _text_page())
    process_frame(tmp_path / "in" / "page.png", _params(tmp_path))
    out = cv2.imread(str(tmp_path / "out" / "page.png"))
    assert not np.array_equal(out, src)
    assert np.array_equal(out[src[:, :, 0] == INK], src[src[:, :, 0] == INK])


def test_relative_paths_are_preserved(tmp_path):
    """Структура подкаталогов зеркалится в output и в debug."""
    _write(tmp_path / "in" / "1966" / "03" / "page.png", _text_page())
    process_frame(tmp_path / "in" / "1966" / "03" / "page.png", _params(tmp_path, debug_dir=tmp_path / "dbg"))
    assert (tmp_path / "out" / "1966" / "03" / "page.png").exists()
    assert (tmp_path / "dbg" / "1966" / "03" / "page.jpg").exists()


def test_overlay_written_for_untouched_frame(tmp_path):
    """Оверлей пишется и для скопированного кадра: пустой оверлей — тоже ответ."""
    _write(tmp_path / "in" / "cover.png", _cover())
    process_frame(tmp_path / "in" / "cover.png", _params(tmp_path, debug_dir=tmp_path / "dbg"))
    assert (tmp_path / "dbg" / "cover.jpg").exists()


def test_output_format_override(tmp_path):
    """``output_format`` меняет расширение выходного файла, не трогая вход."""
    _write(tmp_path / "in" / "page.tif", _text_page())
    process_frame(tmp_path / "in" / "page.tif", _params(tmp_path, output_format="png"))
    assert (tmp_path / "out" / "page.png").exists()


def test_gray_output_is_single_channel(tmp_path):
    """``to_gray`` даёт одноканальный файл."""
    _write(tmp_path / "in" / "page.png", _text_page())
    process_frame(tmp_path / "in" / "page.png", _params(tmp_path, to_gray=True))
    out = cv2.imread(str(tmp_path / "out" / "page.png"), cv2.IMREAD_UNCHANGED)
    assert out.ndim == 2


def test_skip_if_exists(tmp_path):
    """Существующий выходной файл не перезаписывается при ``skip_if_exists``."""
    _write(tmp_path / "in" / "page.png", _text_page())
    marker = _write(tmp_path / "out" / "page.png", np.zeros((10, 10, 3), np.uint8))
    process_frame(tmp_path / "in" / "page.png", _params(tmp_path, skip_if_exists=True))
    assert np.array_equal(cv2.imread(str(tmp_path / "out" / "page.png")), marker)
