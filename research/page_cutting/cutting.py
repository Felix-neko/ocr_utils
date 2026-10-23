"""Нарезка кадра целиком: рамка spline со сгибом → выпрямленная страница (TIFF с dpi) → ScanTailor без dewarp и с dewarp → судья → итог и лист отсмотра."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ocr_utils.page_layout.overlay_frame import LegendEntry, framed, header_strip
from ocr_utils.units import MM_PER_INCH
from ocr_utils.scan_cropping.v2.page_boundary import FitKind, fit_boundary
from research.page_cutting.cases import load_case
from research.page_cutting.fitting import page_name, side_by_side
from research.page_cutting.judge import Verdict, judge
from research.page_cutting.runs import RunSpec, read_image, write_image
from research.page_cutting.scantailor import (
    Dewarp,
    colour_profile,
    read_distortion,
    run_page,
    text_profile,
    write_profile,
)
from research.page_cutting.straightness import measure_straightness
from research.page_cutting.warp import plan_page_warp, warp_page

# Страница цветная (обложка, форзац, цветная вклейка), если медианная цветность центра выше, Lab: желтоватая бумага
# ~4–10, цветные обложки пака-6 — 30–70 (select, группа colour).
COLOUR_CHROMA = 20.0
# Доля размера страницы, отступаемая от краёв при мере цветности.
CHROMA_INSET = 0.2
# Высота картинки на листе отсмотра, px.
SHEET_HEIGHT = 900
# Цвет кривых модели искажения ScanTailor на листе (BGR): верх — синий, низ — оранжевый.
COLOUR_MODEL_TOP = (220, 90, 20)
COLOUR_MODEL_BOTTOM = (0, 165, 255)


@dataclass(frozen=True)
class CutRow:
    """Итог по странице.

    Поля:
        run, key, page: Страница.
        colour: Страница цветная (профиль без полей, освещения и dewarp).
        dpi: Плотность выпрямленной страницы и вывода, dpi.
        verdict: Вердикт судьи (для цветной — ``off_failed`` не бывает, ставится ``colour``).
        similarity: Сходство слов «авто» с «выкл».
        words_off, words_auto: Слов tesseract.
        conf_off, conf_auto: Средняя уверенность слов tesseract.
        score_off, score_auto: Непрямизна (прогиб p90 мм + 0.5 · разброс наклонов °).
        sag_off, sag_auto: Прогиб строк p90, мм.
        spread_off, spread_auto: Разброс наклонов, °.
        model_skew_deg: Расхождение наклонов кривых модели ScanTailor, °.
        output: Выбранный вывод (путь TIFF).
    """

    run: str
    key: str
    page: str
    colour: bool
    dpi: int
    verdict: str
    similarity: float
    words_off: int
    words_auto: int
    conf_off: float
    conf_auto: float
    score_off: float
    score_auto: float
    sag_off: float
    sag_auto: float
    spread_off: float
    spread_auto: float
    model_skew_deg: float
    output: str


def centre_chroma(bgr: np.ndarray) -> float:
    """Медианная цветность центра картинки (Lab OpenCV, расстояние (a, b) от серого)."""
    h, w = bgr.shape[:2]
    dy, dx = int(h * CHROMA_INSET), int(w * CHROMA_INSET)
    lab = cv2.cvtColor(bgr[dy : h - dy, dx : w - dx], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    return float(np.median(np.hypot(lab[:, 1] - 128, lab[:, 2] - 128)))


def write_tiff(path: Path, bgr: np.ndarray, dpi: int) -> None:
    """TIFF (LZW) с dpi в метаданных — ScanTailor берёт плотность оттуда."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path, dpi=(dpi, dpi), compression="tiff_lzw")


def profiles(template: Path, out_dir: Path, dpi: int) -> dict[str, Path]:
    """Профили ScanTailor на плотность ``dpi``: текст без dewarp, текст с dewarp, цветная страница."""
    paths = {}
    for name, prof in (
        ("off", text_profile(Dewarp.OFF, dpi)),
        ("auto", text_profile(Dewarp.AUTO, dpi)),
        ("colour", colour_profile(dpi)),
    ):
        path = out_dir / "profiles" / f"{name}_{dpi}.stp"
        if not path.exists():
            write_profile(template, prof, path)
        paths[name] = path
    return paths


def thumb(img: np.ndarray, title: list[str]) -> np.ndarray:
    """Картинка высотой ``SHEET_HEIGHT`` с шапкой."""
    scale = SHEET_HEIGHT / img.shape[0]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.vstack([header_strip(title, small.shape[1]), small])


def draw_model(img: np.ndarray, model) -> np.ndarray:
    """Кривые модели искажения ScanTailor поверх выпрямленной страницы (координаты входа ScanTailor)."""
    canvas = img.copy()
    if model is not None:
        for curve, colour in ((model.top, COLOUR_MODEL_TOP), (model.bottom, COLOUR_MODEL_BOTTOM)):
            if len(curve) >= 2:
                cv2.polylines(canvas, [np.round(curve).astype(np.int32)], False, colour, 6, cv2.LINE_AA)
    return canvas


def cut_frame(run: RunSpec, key: str, out_dir: Path, cli: Path, template: Path) -> list[CutRow]:
    """Нарезка кадра: по странице — рамка, выпрямление, ScanTailor в двух режимах, судья, лист.

    Args:
        run: Прогон v2.
        key: Ключ кадра.
        out_dir: Корень выхода (``pages/``, ``st/<режим>/``, ``sheets/``, ``profiles/``).
        cli: Путь к ``scantailor-cli``.
        template: Профиль ScanTailor по умолчанию (шаблон).

    Returns:
        Строки по страницам.
    """
    case = load_case(run, key)
    src = case.output()
    dpi = int(round(run.px_per_mm * MM_PER_INCH))
    prof = profiles(template, out_dir, dpi)
    rows = []
    for i, page in enumerate(case.pages):
        name = page_name(case, i)
        stem = f"{run.name}__{key}_{name}"
        if case.fold is not None:
            b = fit_boundary(
                page.fold_mask,
                page.fold_quad,
                page.spine_right,
                FitKind.SPLINE,
                case.seg_px_per_mm,
                case.exclude,
                page.text,
                fold_points=page.spine_edge,
            )
        else:
            b = fit_boundary(
                page.mask, page.quad, page.spine_right, FitKind.SPLINE, case.seg_px_per_mm, case.exclude, page.text
            )
        warp = plan_page_warp(b, case.seg_to_frame, case.box[:2])
        img = warp_page(src, warp)
        page_path = out_dir / "pages" / f"{stem}.tif"
        write_tiff(page_path, img, dpi)
        colour = centre_chroma(img) > COLOUR_CHROMA
        if colour:
            res = run_page(cli, page_path, prof["colour"], out_dir / "st" / "colour")
            out = res.output
            rows.append(
                CutRow(
                    run.name, key, name, True, dpi, "colour", 1.0, 0, 0, 0.0, 0.0, *([float("nan")] * 6), 0.0, str(out)
                )
            )
            if out is not None:
                sheet = side_by_side([thumb(img, ["выпрямленная страница"]), thumb(read_image(out), ["ScanTailor"])])
                write_image(
                    out_dir / "sheets" / f"{stem}.jpg",
                    np.vstack([header_strip([f"{stem} — цветная"], sheet.shape[1]), sheet]),
                )
            continue
        res_off = run_page(cli, page_path, prof["off"], out_dir / "st" / "off")
        res_auto = run_page(cli, page_path, prof["auto"], out_dir / "st" / "auto")
        if res_off.output is None:
            rows.append(
                CutRow(
                    run.name, key, name, False, dpi, "st_failed", 0.0, 0, 0, 0.0, 0.0, *([float("nan")] * 6), 0.0, ""
                )
            )
            continue
        off = read_image(res_off.output)
        auto = read_image(res_auto.output) if res_auto.output is not None else None
        ppm = dpi / MM_PER_INCH
        s_off = measure_straightness(off, ppm)
        s_auto = measure_straightness(auto, ppm) if auto is not None else None
        model = read_distortion(res_auto.project) if res_auto.project.exists() else None
        j = judge(off, auto, s_off, s_auto, model)
        chosen = res_auto.output if j.verdict is Verdict.DEWARP else res_off.output
        rows.append(
            CutRow(
                run=run.name,
                key=key,
                page=name,
                colour=False,
                dpi=dpi,
                verdict=j.verdict.value,
                similarity=j.similarity,
                words_off=j.words_off,
                words_auto=j.words_auto,
                conf_off=j.conf_off,
                conf_auto=j.conf_auto,
                score_off=round(j.score_off, 3),
                score_auto=round(j.score_auto, 3),
                sag_off=s_off.sagitta_p90_mm,
                sag_auto=s_auto.sagitta_p90_mm if s_auto else float("nan"),
                spread_off=s_off.slope_spread_deg,
                spread_auto=s_auto.slope_spread_deg if s_auto else float("nan"),
                model_skew_deg=j.model_skew_deg,
                output=str(chosen),
            )
        )
        cells = [
            thumb(draw_model(img, model), ["выпрямленная страница", "кривые модели dewarp ScanTailor"]),
            thumb(
                off,
                [
                    "ScanTailor, dewarp выкл",
                    f"прогиб {s_off.sagitta_p90_mm:.2f} мм, разброс {s_off.slope_spread_deg:.2f}°",
                ],
            ),
        ]
        if auto is not None:
            cells.append(
                thumb(
                    auto,
                    [
                        "ScanTailor, dewarp авто",
                        f"прогиб {s_auto.sagitta_p90_mm:.2f} мм, разброс {s_auto.slope_spread_deg:.2f}°",
                    ],
                )
            )
        sheet = side_by_side(cells)
        head = header_strip(
            [
                f"{stem} — вердикт: {j.verdict.value}; сходство текста {j.similarity:.3f} "
                f"(слов {j.words_off} / {j.words_auto}, уверенность {j.conf_off:.0f} / {j.conf_auto:.0f}); непрямизна {j.score_off:.2f} → {j.score_auto:.2f}; "
                f"сход кривых модели {j.model_skew_deg:.1f}°"
            ],
            sheet.shape[1],
        )
        legend = [
            LegendEntry("верхняя кривая модели dewarp", COLOUR_MODEL_TOP),
            LegendEntry("нижняя кривая модели dewarp", COLOUR_MODEL_BOTTOM),
        ]
        write_image(out_dir / "sheets" / f"{stem}.jpg", framed(np.vstack([head, sheet]), [], legend))
    return rows
