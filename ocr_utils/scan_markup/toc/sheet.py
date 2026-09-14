"""Контактный лист окна выпуска: по нему размечается эталон и разбираются ошибки.

Миниатюры полос окна в порядке выпуска; рамка — зелёная у «Содержания», синяя у указателя,
красная — расхождение с эталоном (если метки даны). Подписи латиницей и цифрами: штатный
шрифт PIL кириллицу не рисует, а тянуть шрифт ради подписи незачем.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from ocr_utils.scan_markup.toc import KIND_CONTENTS, KIND_INDEX
from ocr_utils.scan_markup.toc.labels import Label, lookup
from ocr_utils.scan_markup.toc.run import IssueResult

THUMB_W = 220
THUMB_H = 380
CAPTION_H = 28
COLUMNS = 9
FRAME = 4

COLOURS = {KIND_CONTENTS: (0, 160, 0), KIND_INDEX: (30, 80, 220)}
MISMATCH = (220, 0, 0)
PLAIN = (200, 200, 200)


def contact_sheet(result: IssueResult, labels: dict[str, Label] | None) -> Image.Image:
    features = sorted(result.features, key=lambda f: f.order_index)
    by_index = {d.order_index: d for d in result.decisions}
    rows = (len(features) + COLUMNS - 1) // COLUMNS
    cell_h = THUMB_H + CAPTION_H
    sheet = Image.new("RGB", (COLUMNS * THUMB_W, max(1, rows) * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for slot, feature in enumerate(features):
        decision = by_index[feature.order_index]
        label = lookup(labels, feature.rel_path) if labels else None
        x0, y0 = (slot % COLUMNS) * THUMB_W, (slot // COLUMNS) * cell_h
        if feature.thumbnail:
            thumb = Image.open(BytesIO(feature.thumbnail)).convert("RGB")
            thumb.thumbnail((THUMB_W - 2 * FRAME, THUMB_H - 2 * FRAME))
            sheet.paste(thumb, (x0 + FRAME, y0 + FRAME))
        colour = COLOURS.get(decision.kind or "", PLAIN)
        if label is not None and (label.label if label.is_toc else None) != decision.kind:
            colour = MISMATCH
        draw.rectangle((x0, y0, x0 + THUMB_W - 1, y0 + THUMB_H - 1), outline=colour, width=FRAME)
        caption = f"#{feature.order_index} -{feature.idx_from_end} {decision.kind or '-'} {decision.score:.2f}"
        if label is not None:
            caption += f" [{label.label}]"
        draw.text((x0 + FRAME, y0 + THUMB_H + 4), caption, fill=(0, 0, 0))
        draw.text((x0 + FRAME, y0 + THUMB_H + 15), feature.rel_path.rsplit("/", 1)[-1], fill=(90, 90, 90))
    return sheet


FOUND_THUMB_W = 420
FOUND_THUMB_H = 720
FOUND_COLUMNS = 6


def found_sheet(result: IssueResult, kind: str) -> Image.Image | None:
    """Контактный лист НАЙДЕННЫХ полос одного вида (только они, крупнее, чем в окне);
    ``None``, если таких полос в выпуске нет."""
    by_index = {d.order_index: d for d in result.decisions}
    found = sorted((f for f in result.features if by_index[f.order_index].kind == kind), key=lambda f: f.order_index)
    if not found:
        return None
    cell_h = FOUND_THUMB_H + CAPTION_H
    columns = min(FOUND_COLUMNS, len(found))
    rows = (len(found) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * FOUND_THUMB_W, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for slot, feature in enumerate(found):
        decision = by_index[feature.order_index]
        x0, y0 = (slot % columns) * FOUND_THUMB_W, (slot // columns) * cell_h
        if feature.thumbnail:
            thumb = Image.open(BytesIO(feature.thumbnail)).convert("RGB")
            thumb.thumbnail((FOUND_THUMB_W - 2 * FRAME, FOUND_THUMB_H - 2 * FRAME))
            sheet.paste(thumb, (x0 + FRAME, y0 + FRAME))
        draw.rectangle((x0, y0, x0 + FOUND_THUMB_W - 1, y0 + FOUND_THUMB_H - 1), outline=COLOURS[kind], width=FRAME)
        caption = f"{result.rel_dir} #{feature.order_index} (-{feature.idx_from_end}) {decision.score:.2f}"
        draw.text((x0 + FRAME, y0 + FOUND_THUMB_H + 4), caption, fill=(0, 0, 0))
        draw.text((x0 + FRAME, y0 + FOUND_THUMB_H + 15), feature.rel_path.rsplit("/", 1)[-1], fill=(90, 90, 90))
    return sheet


__all__ = ["contact_sheet", "found_sheet"]
