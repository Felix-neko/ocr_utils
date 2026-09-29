"""Оверлеи судей: полоса с кандидатами по вердикту и тем, что увидел судья (карта, рамки), и листы «вырезка — вердикты всех судей»."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.page_layout.overlay_frame import FONT_PATH, LegendEntry, SampleStyle, framed
from research.edge_marks.candidates import Candidate, load
from research.edge_marks.judges.common import Verdict
from research.edge_marks.score import read_results

# Цвета BGR: вердикт судьи (как «принято / отвергнуто» этапных оверлеев, неясно — оранжевый).
VERDICT_COLOUR = {Verdict.JUNK.value: (0, 0, 220), Verdict.SIGN.value: (0, 150, 0), Verdict.UNSURE.value: (0, 165, 255)}
VERDICT_WORD = {Verdict.JUNK.value: "сор", Verdict.SIGN.value: "знак", Verdict.UNSURE.value: "?"}
TRUTH_WORD = {"junk": "сор", "sign": "знак"}
# Что увидел судья: рамки слов/строк движка — «подсказка», карта движка — фиолетовая заливка.
COLOUR_BOXES = (200, 140, 60)
COLOUR_MAP = (200, 60, 200)
MAP_ALPHA = 0.45
BOXES_ALPHA = 0.8
# Рамка кандидата на полосе: толщина и поле вокруг краски.
BOX_THICKNESS, BOX_PAD = 3, 4
# Подписи у кандидатов и в листах.
CAPTION_SIZE = 20
SHEET_ROW_HEIGHT = 110
SHEET_TEXT_WIDTH = 900
SHEET_ROWS = 25


@dataclass(frozen=True)
class JudgeFrame:
    """Результаты одного судьи: ``id, scale, verdict, detail`` и сырые ответы (рамки для масштаба полосы)."""

    name: str
    frame: pd.DataFrame


def _font(size: int) -> ImageFont.FreeTypeFont:
    """Шрифт DejaVu для подписей (OpenCV кириллицу не рисует)."""
    return ImageFont.truetype(FONT_PATH, size)


def _draw_captions(canvas: np.ndarray, captions: list[tuple[int, int, list[tuple[str, tuple]]]]) -> np.ndarray:
    """Подписи из цветных кусков на белой подложке.

    Args:
        canvas: Холст BGR.
        captions: ``(x, y, [(текст, цвет BGR), …])`` — левый верхний угол подписи и её куски.

    Returns:
        Холст BGR с подписями.
    """
    picture = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(picture)
    font = _font(CAPTION_SIZE)
    for x, y, parts in captions:
        # Подложка под всю подпись, чтобы краска страницы не мешала читать.
        width = sum(draw.textlength(text, font=font) for text, _ in parts)
        draw.rectangle([x - 2, y - 2, x + width + 2, y + CAPTION_SIZE + 4], fill=(255, 255, 255))
        for text, colour in parts:
            draw.text((x, y), text, font=font, fill=(colour[2], colour[1], colour[0]))
            x += draw.textlength(text, font=font)
    return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)


def _blend_map(canvas: np.ndarray, engine_map: np.ndarray) -> None:
    """Карта движка (0…255) — фиолетовой заливкой с прозрачностью, пропорциональной значению (правит ``canvas``)."""
    weight = (engine_map.astype(np.float32) / 255.0 * MAP_ALPHA)[..., None]
    tint = np.array(COLOUR_MAP, dtype=np.float32)
    canvas[:] = (canvas.astype(np.float32) * (1 - weight) + tint * weight).astype(np.uint8)


def page_overlay(
    gray: np.ndarray, title: str, judge: JudgeFrame, items: list[Candidate], maps_dir: Path | None
) -> np.ndarray:
    """Оверлей полосы для одного судьи.

    На полосе: карта движка (если судья её сохранил), рамки слов/строк движка по масштабу полосы, кандидаты —
    рамкой цвета вердикта первого масштаба и подписью «истина | вердикты по масштабам».

    Args:
        gray: Бинаризованная полоса (рендер 300 dpi).
        title: Шапка (полоса, судья).
        judge: Результаты судьи.
        items: Кандидаты полосы.
        maps_dir: Папка карт судьи ``<судья>_maps`` или ``None``.

    Returns:
        Картинка BGR со шапкой и легендой в полях.
    """
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    key = items[0].key
    has_map = maps_dir is not None and (maps_dir / f"{key}.png").exists()
    if has_map:
        engine_map = cv2.imread(str(maps_dir / f"{key}.png"), cv2.IMREAD_GRAYSCALE)
        if engine_map.shape == gray.shape:
            _blend_map(canvas, engine_map)
    # Рамки движка по полосе (масштабы «c…» с ответом-рамками): у всех кандидатов полосы они одни и те же.
    ids = {c.id for c in items}
    rows = judge.frame[judge.frame.id.isin(ids)]
    page_boxes = rows.iloc[:0]
    if "boxes" in rows:
        page_boxes = rows[rows.scale.str.startswith("c") & rows.boxes.notna()]
    layer = canvas.copy()
    drawn_boxes = False
    if len(page_boxes):
        for box in page_boxes.iloc[0]["boxes"]:
            cv2.rectangle(layer, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), COLOUR_BOXES, 2)
        drawn_boxes = True
    # Рамки кандидатов в том же слое непрозрачно поверх рамок движка.
    cv2.addWeighted(layer, BOXES_ALPHA, canvas, 1 - BOXES_ALPHA, 0, canvas)
    captions = []
    scales = sorted(rows.scale.unique())
    for c in items:
        mine = rows[rows.id == c.id].set_index("scale")
        first = mine.loc[scales[0], "verdict"] if len(scales) and scales[0] in mine.index else Verdict.UNSURE.value
        x0, y0, x1, y1 = c.box
        cv2.rectangle(
            canvas, (x0 - BOX_PAD, y0 - BOX_PAD), (x1 + BOX_PAD, y1 + BOX_PAD), VERDICT_COLOUR[first], BOX_THICKNESS
        )
        parts = [(f"{c.id} {TRUTH_WORD[c.truth]}({c.label}) |", (20, 20, 20))]
        for scale in scales:
            if scale in mine.index:
                verdict = mine.loc[scale, "verdict"]
                parts.append((f" {scale}:{VERDICT_WORD[verdict]}", VERDICT_COLOUR[verdict]))
        # Подпись снаружи блока: у правой стороны — правее рамки, у левой — левее (ширину оцениваем грубо).
        width = sum(len(text) for text, _ in parts) * CAPTION_SIZE * 0.55
        x = x1 + 12 if c.side == "right" else x0 - 12 - int(width)
        x = int(min(max(4, x), canvas.shape[1] - width - 4))
        captions.append((x, int(y0 - CAPTION_SIZE - 6), parts))
    canvas = _draw_captions(canvas, captions)
    legend = [
        LegendEntry("кандидат: судья сказал «сор»", VERDICT_COLOUR[Verdict.JUNK.value], style=SampleStyle.LINE),
        LegendEntry("кандидат: судья сказал «знак»", VERDICT_COLOUR[Verdict.SIGN.value], style=SampleStyle.LINE),
        LegendEntry("кандидат: неясно", VERDICT_COLOUR[Verdict.UNSURE.value], style=SampleStyle.LINE),
    ]
    if drawn_boxes:
        legend.append(LegendEntry("рамки слов/строк движка по полосе", COLOUR_BOXES, alpha=BOXES_ALPHA))
    if has_map:
        legend.append(
            LegendEntry("карта движка (ярче — выше вероятность)", COLOUR_MAP, alpha=MAP_ALPHA, style=SampleStyle.BOX)
        )
    header = [
        title,
        f"рамка — вердикт масштаба {scales[0] if scales else '—'}; подпись: id, истина(класс) | масштаб:вердикт",
    ]
    return framed(canvas, header, legend)


def load_judges(cand_dir: Path, results_dir: Path) -> tuple[list[Candidate], list[JudgeFrame]]:
    """Кандидаты и результаты всех судей из ``results_dir`` (``<судья>.jsonl``)."""
    candidates = load(cand_dir)
    by_id = {c.id: c for c in candidates}
    judges = []
    for path in sorted(results_dir.glob("*.jsonl")):
        frame, _ = read_results(path, by_id)
        if not frame.empty and "verdict" in frame:
            judges.append(JudgeFrame(path.stem, frame))
    return candidates, judges


def judge_pages(
    cand_dir: Path, results_dir: Path, out_dir: Path, judge: JudgeFrame, candidates: list[Candidate]
) -> int:
    """Оверлеи всех полос для одного судьи в ``out_dir/<судья>/overlays/``; вернуть число картинок."""
    target = out_dir / judge.name / "overlays"
    target.mkdir(parents=True, exist_ok=True)
    maps_dir = results_dir / f"{judge.name}_maps"
    by_key: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_key.setdefault(c.key, []).append(c)
    for key, items in by_key.items():
        gray = cv2.imread(str(cand_dir / "pages" / f"{key}.png"), cv2.IMREAD_GRAYSCALE)
        picture = page_overlay(
            gray, f"{key} — судья {judge.name}", judge, items, maps_dir if maps_dir.exists() else None
        )
        cv2.imwrite(str(target / f"{key}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(by_key)


def _row(cand_dir: Path, c: Candidate, judges: list[JudgeFrame]) -> np.ndarray:
    """Строка листа: вырезка (a) и вердикты всех судей по масштабам цветом."""
    crop = cv2.imread(str(cand_dir / "a" / f"{c.id}.png"), cv2.IMREAD_GRAYSCALE)
    scale = SHEET_ROW_HEIGHT / crop.shape[0]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)[:, : SHEET_ROW_HEIGHT * 3]
    image = np.full((SHEET_ROW_HEIGHT, SHEET_ROW_HEIGHT * 3 + SHEET_TEXT_WIDTH, 3), 255, dtype=np.uint8)
    image[:, : crop.shape[1]] = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    captions = [
        (
            SHEET_ROW_HEIGHT * 3 + 10,
            4,
            [(f"{c.id} {c.key} {c.side}: истина {TRUTH_WORD[c.truth]} ({c.label})", (20, 20, 20))],
        )
    ]
    parts: list[tuple[str, tuple]] = []
    line = 1
    for judge in judges:
        mine = judge.frame[judge.frame.id == c.id]
        for record in mine.itertuples():
            parts.append(
                (f"{judge.name}.{record.scale}:{VERDICT_WORD[record.verdict]}  ", VERDICT_COLOUR[record.verdict])
            )
            if len(parts) == 5:
                captions.append((SHEET_ROW_HEIGHT * 3 + 10, 4 + line * (CAPTION_SIZE + 6), parts))
                parts, line = [], line + 1
    if parts:
        captions.append((SHEET_ROW_HEIGHT * 3 + 10, 4 + line * (CAPTION_SIZE + 6), parts))
    image = _draw_captions(image, captions)
    cv2.line(image, (0, SHEET_ROW_HEIGHT - 1), (image.shape[1], SHEET_ROW_HEIGHT - 1), (200, 200, 200), 1)
    return image


def _wrong_count(c: Candidate, judges: list[JudgeFrame]) -> int:
    """Сколько вердиктов судей расходится с истиной: ошибка — 2, «неясно» — 1 (только для порядка листов)."""
    total = 0
    for judge in judges:
        for verdict in judge.frame[judge.frame.id == c.id].verdict:
            if verdict == Verdict.UNSURE.value:
                total += 1
            elif verdict != c.truth:
                total += 2
    return total


def sheets(cand_dir: Path, out_dir: Path, candidates: list[Candidate], judges: list[JudgeFrame]) -> int:
    """Листы «вырезка — вердикты всех судей» в ``out_dir/sheets/``: сначала спорные (судьи расходятся или ошиблись).

    Returns:
        Число листов.
    """
    target = out_dir / "sheets"
    target.mkdir(parents=True, exist_ok=True)

    ordered = sorted(candidates, key=lambda c: (-_wrong_count(c, judges), c.id))
    count = 0
    for start in range(0, len(ordered), SHEET_ROWS):
        rows = [_row(cand_dir, c, judges) for c in ordered[start : start + SHEET_ROWS]]
        legend = [LegendEntry(f"вердикт «{VERDICT_WORD[v]}»", colour) for v, colour in VERDICT_COLOUR.items()]
        header = [f"Кандидаты {start + 1}–{start + len(rows)} из {len(ordered)}: сначала те, где судьи ошибаются чаще",
                  "судья.масштаб: a — вырезка, b — строка, c — полоса"]  # fmt: skip
        cv2.imwrite(
            str(target / f"sheet_{count + 1:02d}.jpg"),
            framed(np.vstack(rows), header, legend),
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        count += 1
    return count


__all__ = ["JudgeFrame", "judge_pages", "load_judges", "page_overlay", "sheets"]
