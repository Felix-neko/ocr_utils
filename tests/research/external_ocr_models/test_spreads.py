"""Разрезка синтетического разворота по наклонной щели сгиба."""

import numpy as np
from PIL import Image, ImageDraw

from research.external_ocr_models.spreads import find_fold, fold_from_points, split_folder, split_spread


def _spread(path, width=2400, height=1600, x_top=1180, x_bottom=1230):
    """Две «страницы» с полосками-строками и тёмная наклонная щель между ними."""
    image = Image.new("L", (width, height), 235)
    draw = ImageDraw.Draw(image)
    # «Слова» — короткие прямоугольники с пробелами: сплошная линия сошла бы за линейку
    # таблицы, а их детектор из маски краски вычитает.
    for y in range(140, height - 140, 26):
        for x0, x1 in ((160, x_top - 90), (x_bottom + 90, width - 160)):
            for x in range(x0, x1 - 40, 52):
                draw.rectangle((x, y, x + 36, y + 9), fill=30)
    draw.polygon([(x_top - 12, 0), (x_top + 12, 0), (x_bottom + 12, height), (x_bottom - 12, height)], fill=40)
    image.save(path, quality=95)
    return image


def test_fold_found_and_pages_split(tmp_path):
    section = tmp_path / "in" / "часть букв в словах совсем закрыта корешком"
    section.mkdir(parents=True)
    _spread(section / "IMG_0001.jpg")
    (tmp_path / "in" / "пересвет").mkdir()
    _spread(tmp_path / "in" / "пересвет" / "IMG_0002.jpg")

    fold = find_fold(section / "IMG_0001.jpg")
    assert abs(fold.x_at(0) - 1180) < 6 and abs(fold.x_at(1600) - 1230) < 6

    written = split_folder(tmp_path / "in", tmp_path / "out", ("часть букв в словах совсем закрыта корешком",))
    names = sorted(path.relative_to(tmp_path / "out").as_posix() for path in written)
    assert names == [
        "пересвет/IMG_0002_L.jpg",
        "часть букв в словах совсем закрыта корешком/IMG_0001_L.jpg",
        "часть букв в словах совсем закрыта корешком/IMG_0001_R.jpg",
    ]
    assert (tmp_path / "out" / "_линии" / "IMG_0001.jpg").is_file()
    left = np.asarray(Image.open(tmp_path / "out" / names[1]).convert("L"))
    right = np.asarray(Image.open(tmp_path / "out" / names[2]).convert("L"))
    # у левой страницы строки есть, правее линии — чернота; у правой — зеркально
    assert left.shape[1] <= 1240 and (left[:, :1000] < 100).mean() > 0.05 and (left[:, -4:] < 5).mean() > 0.9
    assert (
        right.shape[1] <= 2400 - 1180 + 8 and (right[:, -1000:] < 100).mean() > 0.05 and (right[:, :4] < 5).mean() > 0.9
    )


def test_explicit_fold_overrides_fit(tmp_path):
    image = _spread(tmp_path / "s.jpg")
    fold = fold_from_points(600, 640, image.height)
    left, right = split_spread(image, fold)
    assert left.width == 640 and right.width == image.width - 600
