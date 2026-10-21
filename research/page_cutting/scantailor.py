"""Обёртка над ``scantailor-cli`` (ScanTailor Advanced без окна): профиль параметров по умолчанию (поля, цвет, выравнивание освещения, dewarp), прогон одной страницы, разбор модели искажения из проекта."""

from __future__ import annotations

import base64
import os
import struct
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

# Время на одну страницу, с: dewarp с построением модели по строкам на полноразмерной странице — десятки секунд.
TIMEOUT_S = 600


class ColorMode(str, Enum):
    """Режим вывода ScanTailor (``colorMode`` в профиле)."""

    BW = "bw"
    COLOR_GRAY = "colorOrGray"
    MIXED = "mixed"


class Dewarp(str, Enum):
    """Режим распрямления строк ScanTailor (``dewarpingOptions mode``)."""

    OFF = "off"
    AUTO = "auto"
    MARGINAL = "marginal"


@dataclass(frozen=True)
class Margins:
    """Поля вывода, мм (``hardMargins`` профиля, единицы профиля — мм).

    Поля:
        top, bottom, left, right: Поля сверху, снизу, слева, справа.
    """

    top: float
    bottom: float
    left: float
    right: float


# Поля обычной страницы — как выставлял пользователь вручную: 15 мм сверху, 10 снизу, 5 по бокам.
TEXT_MARGINS = Margins(15.0, 10.0, 5.0, 5.0)
# Цветные обложки и форзацы — без полей.
NO_MARGINS = Margins(0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class StProfile:
    """Параметры ScanTailor для страницы.

    Поля:
        margins: Поля вывода, мм.
        color_mode: Режим вывода.
        equalize: Выравнивание освещения (для цветного/серого — ``normalizeIlluminationColor``, для ч/б — ``BW``).
        dewarp: Распрямление строк.
        dpi: Разрешение вывода (по обеим осям); None — как у входа.
        deskew_auto: Автоматический доворот (иначе — без поворота).
        content_detect: Автозахват контента (иначе — вся страница).
    """

    margins: Margins
    color_mode: ColorMode
    equalize: bool
    dewarp: Dewarp
    dpi: int | None
    deskew_auto: bool = True
    content_detect: bool = True


def text_profile(dewarp: Dewarp, dpi: int | None) -> StProfile:
    """Профиль обычной (чёрно-белой по содержанию) страницы: серый вывод с выравниванием освещения, поля 15/10/5/5 мм."""
    return StProfile(TEXT_MARGINS, ColorMode.COLOR_GRAY, True, dewarp, dpi)


def colour_profile(dpi: int | None) -> StProfile:
    """Профиль цветной обложки или форзаца: цветной вывод без выравнивания освещения, без полей, без dewarp."""
    return StProfile(NO_MARGINS, ColorMode.COLOR_GRAY, False, Dewarp.OFF, dpi)


def default_profile_xml(cli: Path, path: Path) -> None:
    """Профиль ScanTailor по умолчанию (``scantailor-cli --dump-profile``) — шаблон для ``write_profile``."""
    subprocess.run([str(cli), "--dump-profile", str(path)], check=True, capture_output=True, env=cli_env())


def write_profile(template: Path, profile: StProfile, path: Path) -> None:
    """Профиль ``.stp`` по шаблону: страница уже отрезана (``single-uncut``), страницы не выравниваются по размеру.

    Args:
        template: Профиль по умолчанию (``default_profile_xml``).
        profile: Параметры страницы.
        path: Куда писать профиль.
    """
    tree = ET.parse(template)
    params = tree.getroot().find("default-params")
    params.set("units", "mm")
    params.find("deskew-params").set("mode", "auto" if profile.deskew_auto else "manual")
    params.find("page-split-params").set("layoutType", "single-uncut")
    content = params.find("select-content-params")
    content.set("contentDetectEnabled", "1" if profile.content_detect else "0")
    content.set("pageDetectMode", "disabled")
    layout = params.find("page-layout-params")
    layout.set("autoMargins", "0")
    m = profile.margins
    layout.find("hardMargins").attrib.update(
        {"top": f"{m.top:g}", "bottom": f"{m.bottom:g}", "left": f"{m.left:g}", "right": f"{m.right:g}"}
    )
    # Без выравнивания по другим страницам: размер вывода — по своему контенту и полям.
    layout.find("alignment").set("null", "1")
    out = params.find("output-params")
    if profile.dpi:
        out.find("dpi").attrib.update({"vertical": str(profile.dpi), "horizontal": str(profile.dpi)})
    colour = out.find("colorParams")
    colour.set("colorMode", profile.color_mode.value)
    colour.find("color-or-grayscale").set("normalizeIlluminationColor", "1" if profile.equalize else "0")
    colour.find("bw").set("normalizeIlluminationBW", "1" if profile.equalize else "0")
    out.find("dewarpingOptions").set("mode", profile.dewarp.value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=False)


def cli_env() -> dict[str, str]:
    """Окружение для ``scantailor-cli``: Qt без дисплея."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


@dataclass(frozen=True)
class StResult:
    """Итог прогона страницы через ScanTailor.

    Поля:
        output: Выходной TIFF или None (сбой).
        project: Проект ``.ScanTailor`` с параметрами всех стадий.
        log: Вывод CLI (stdout + stderr).
        returncode: Код возврата.
    """

    output: Path | None
    project: Path
    log: str
    returncode: int


def run_page(cli: Path, image: Path, profile_path: Path, out_dir: Path) -> StResult:
    """Одна страница через ``scantailor-cli``.

    Args:
        cli: Путь к ``scantailor-cli``.
        image: Входная страница (TIFF с dpi в метаданных).
        profile_path: Профиль ``.stp``.
        out_dir: Папка вывода (туда же — проект ``<имя>.ScanTailor``).

    Returns:
        Итог прогона.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    project = out_dir / f"{image.stem}.ScanTailor"
    cmd = [str(cli), "--profile", str(profile_path), "--out-dir", str(out_dir), "--project", str(project), str(image)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=cli_env())
    out = out_dir / f"{image.stem}.tif"
    return StResult(out if out.exists() else None, project, proc.stdout + proc.stderr, proc.returncode)


def decode_polyline(text: str) -> np.ndarray:
    """Ломаная ``<polyline>`` проекта ScanTailor: base64 подряд идущих пар float32 (x, y), little-endian.

    Args:
        text: Содержимое элемента.

    Returns:
        Точки (N, 2).
    """
    raw = base64.b64decode(text.strip())
    n = len(raw) // 8
    if n == 0:
        return np.zeros((0, 2))
    return np.array(struct.unpack(f"<{2 * n}f", raw[: 8 * n]), np.float64).reshape(n, 2)


@dataclass(frozen=True)
class Distortion:
    """Модель искажения страницы из проекта: верхняя и нижняя кривые (px исходной картинки).

    Поля:
        top: Верхняя кривая (N, 2) или пустая.
        bottom: Нижняя кривая (M, 2) или пустая.
        mode: Режим dewarp, записанный в проекте.
    """

    top: np.ndarray
    bottom: np.ndarray
    mode: str


def read_distortion(project: Path) -> Distortion | None:
    """Модель искажения первой страницы проекта (``<output><page><params><distortion-model>``) или None."""
    root = ET.parse(project).getroot()
    page = root.find("./filters/output/page")
    if page is None:
        return None
    params = page.find("params")
    if params is None:
        return None
    opts = params.find("dewarping-options")
    mode = opts.get("mode", "") if opts is not None else ""
    model = params.find("distortion-model")

    def curve(name: str) -> np.ndarray:
        el = model.find(name) if model is not None else None
        poly = el.find("polyline") if el is not None else None
        if poly is not None and poly.text:
            return decode_polyline(poly.text)
        xs = el.find("xspline") if el is not None else None
        if xs is not None:
            return np.array([[float(p.get("x")), float(p.get("y"))] for p in xs.findall("point")])
        return np.zeros((0, 2))

    return Distortion(curve("top-curve"), curve("bottom-curve"), mode)
