"""Пробник: может ли VLM (DeepSeek V4.1 Flash через OpenRouter) сама увидеть порчу геометрии.

Гипотеза пользователя: модель увидит «геометрия поплыла» по картинке «было | стало», как
человек, без наших метрик. Ограничение DeepSeek — ~1024 токена на картинку, то есть страница
ужимается до ~1300 px (1 мм ≈ 4 px): наклоны в полградуса ей не видны, грубая порча — должна.

Четыре варианта входа (``VARIANTS``), чтобы сравнить на одной выборке:

* ``overlay`` — A приведена в кадр B обратной аффинной частью поля смещений, краска B синим,
  A красным, совпадение тёмным, зелёная сетка через 10 мм. Без совмещения всё двоится:
  FineReader сдвигает и поворачивает страницу целиком.
* ``side`` — «было | стало» рядом с сеткой, та же картинка, что смотрит человек.
* ``two`` — B и A двумя картинками в одном запросе: каждую модель ужмёт отдельно, деталей вдвое больше.
* ``crop`` — пара кропов области-виновника классического детектора в полном разрешении:
  роль «второе мнение по кандидату».

Сеть, модель и учёт цены — из ``ocr_utils.external_ocr_services`` (клиент OpenRouter, реестр
моделей, ``usage.cost`` из ответа). Ответы кладутся по одному JSON на запрос; повтор с
``--skip-done`` не платит дважды.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ocr_utils.external_ocr_services.client import OpenRouterClient
from ocr_utils.external_ocr_services.models import ModelSpec
from research.geometry_regression import mm_to_px
from research.geometry_regression.overlay import pair_image
from research.geometry_regression.prompts import render
from research.geometry_regression.report import PageRow

logger = logging.getLogger(__name__)

VLM_PROMPT_VERSION = 1
VARIANTS = ("overlay", "side", "two", "crop")
DEFAULT_MODEL = "deepseek-v41-flash"
# Шаг сетки на картинках для модели и размер кропа-виновника (мм).
GRID_MM = 10.0
CROP_MIN_MM = 70.0
CROP_PAD_MM = 8.0
# Картинка к отправке: длинная сторона и качество JPEG (DeepSeek всё равно ужмёт до ~1300 px).
MAX_SIDE_PX = 2200
JPEG_QUALITY = 85
MAX_TOKENS = 400
# Пояса score классического детектора для выборки.
BELTS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf")))

INK_B = (40, 40, 230)
INK_A = (230, 40, 40)
INK_BOTH = (50, 0, 60)
GRID_GREEN = (0, 160, 0)


@dataclass(frozen=True)
class Probe:
    pdf: str
    page: int
    belt: str  # "label:bad" | "label:good" | "belt:0.5-1"
    score: float


def belt_name(score: float) -> str:
    for low, high in BELTS:
        if low <= score < high:
            return f"{low:g}-{high:g}" if high != float("inf") else f"{low:g}+"
    return "?"


def sample_pages(
    rows: list[PageRow], labels: dict, per_belt: int, seed: int = 0, only_labelled: bool = False
) -> list[Probe]:
    """Эталонные страницы плюс по ``per_belt`` случайных из каждого пояса score."""
    by_key = {r.key: r for r in rows if not r.error}
    probes = [
        Probe(pdf, page, f"label:{label}", by_key[(pdf, page)].score)
        for (pdf, page), (label, _) in sorted(labels.items())
        if (pdf, page) in by_key  # эталонные страницы вне прогона (проба на одном выпуске) пропускаются
    ]
    if only_labelled:
        return probes
    rng = random.Random(seed)
    taken = {(p.pdf, p.page) for p in probes}
    for low, high in BELTS:
        pool = [r for r in by_key.values() if low <= r.score < high and r.key not in taken]
        rng.shuffle(pool)
        for r in pool[:per_belt]:
            probes.append(Probe(r.pdf, r.page, f"belt:{belt_name(r.score)}", r.score))
    return probes


# ---------- картинки -------------------------------------------------------------------------


def _grid(image: Image.Image, step_px: int, colour) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for x in range(step_px, image.width, step_px):
        draw.line([(x, 0), (x, image.height)], fill=colour, width=1)
    for y in range(step_px, image.height, step_px):
        draw.line([(0, y), (image.width, y)], fill=colour, width=1)
    return image


def overlay_image(before: np.ndarray, after: np.ndarray, affine: np.ndarray | None, dpi: float) -> Image.Image:
    """A в кадре B (обратная аффинная часть поля), краска двумя цветами, зелёная сетка."""
    h, w = before.shape
    if affine is not None:
        aligned = cv2.warpAffine(
            after,
            np.asarray(affine, dtype=np.float64),
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderValue=255,
        )
    else:
        aligned = np.full((h, w), 255, np.uint8)
        aligned[: after.shape[0], : after.shape[1]] = after[:h, :w]
    ink_b, ink_a = before < 128, aligned < 128
    rgb = np.full((h, w, 3), 255, np.uint8)
    rgb[ink_b] = INK_B
    rgb[ink_a] = INK_A
    rgb[ink_b & ink_a] = INK_BOTH
    return _grid(Image.fromarray(rgb), mm_to_px(GRID_MM, dpi), GRID_GREEN)


def side_image(before: np.ndarray, after: np.ndarray) -> Image.Image:
    return pair_image(before, after, None, "B (raw)", "A (corrected)")


def two_images(before: np.ndarray, after: np.ndarray) -> list[Image.Image]:
    out = []
    for gray in (before, after):
        image = Image.fromarray(gray).convert("RGB")
        draw = ImageDraw.Draw(image)
        for step in range(1, 12):
            draw.line(
                [(0, image.height * step // 12), (image.width, image.height * step // 12)], fill=(255, 0, 0), width=1
            )
            draw.line(
                [(image.width * step // 12, 0), (image.width * step // 12, image.height)], fill=(255, 0, 0), width=1
            )
        out.append(image)
    return out


def crop_boxes(
    culprit: dict | None, field_raw: dict | None, size_b: tuple[int, int], size_a: tuple[int, int], dpi: float
):
    """Рамки кропа в B и A (пиксели рабочей копии): виновник с припуском, не меньше ``CROP_MIN_MM``."""
    if culprit:
        box_b, box_a = culprit["b"], culprit["a"]
    elif field_raw and field_raw.get("tiles"):
        tiles = np.array(field_raw["tiles"])
        affine = np.array(field_raw["affine"])
        pred = tiles[:, :2] @ affine[:, :2].T + affine[:, 2] - tiles[:, :2]
        resid = np.hypot(*(tiles[:, 2:4] - pred).T)
        cx, cy = tiles[int(np.argmax(resid)), :2]
        box_b = (cx, cy, cx, cy)
        ax, ay = affine @ np.array([cx, cy, 1.0])
        box_a = (ax, ay, ax, ay)
    else:
        box_b = (size_b[0] / 2, size_b[1] / 2, size_b[0] / 2, size_b[1] / 2)
        box_a = (size_a[0] / 2, size_a[1] / 2, size_a[0] / 2, size_a[1] / 2)
    half = mm_to_px(CROP_MIN_MM, dpi) / 2
    pad = mm_to_px(CROP_PAD_MM, dpi)

    def grow(box, size):
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = max(x1 - x0 + 2 * pad, 2 * half)
        h = max(y1 - y0 + 2 * pad, 2 * half)
        return (
            int(max(0, cx - w / 2)),
            int(max(0, cy - h / 2)),
            int(min(size[0], cx + w / 2)),
            int(min(size[1], cy + h / 2)),
        )

    return grow(box_b, size_b), grow(box_a, size_a)


def crop_image(gray300_b: np.ndarray, gray300_a: np.ndarray, box_b, box_a, dpi: float) -> Image.Image:
    """Кропы в полном разрешении (300 dpi) рядом, с сеткой через 10 мм."""
    k = 300.0 / dpi
    panels = []
    for gray, box in ((gray300_b, box_b), (gray300_a, box_a)):
        x0, y0, x1, y1 = (int(round(v * k)) for v in box)
        panel = Image.fromarray(gray[y0:y1, x0:x1]).convert("RGB")
        panels.append(_grid(panel, mm_to_px(GRID_MM, 300.0), (255, 0, 0)))
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 12, height), (120, 120, 120))
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width + 12, 0))
    return canvas


def variant_images(
    variant: str,
    before: np.ndarray,
    after: np.ndarray,
    gray300_b: np.ndarray,
    gray300_a: np.ndarray,
    cache: dict,
    dpi: float,
) -> list[Image.Image]:
    field_raw = (cache.get("raw") or {}).get("field")
    if variant == "overlay":
        return [overlay_image(before, after, field_raw["affine"] if field_raw else None, dpi)]
    if variant == "side":
        return [side_image(before, after)]
    if variant == "two":
        return two_images(before, after)
    if variant == "crop":
        culprit = None
        flags = cache.get("flags") or {}
        culprits = cache.get("culprits") or {}
        if flags:
            culprit = culprits.get(max(flags, key=flags.get))
        box_b, box_a = crop_boxes(
            culprit, field_raw, (before.shape[1], before.shape[0]), (after.shape[1], after.shape[0]), dpi
        )
        return [crop_image(gray300_b, gray300_a, box_b, box_a, dpi)]
    raise ValueError(f"неизвестный вариант {variant!r}")


def data_url(image: Image.Image, max_side: int = MAX_SIDE_PX, quality: int = JPEG_QUALITY) -> str:
    scale = max_side / max(image.size)
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------- запрос ---------------------------------------------------------------------------


def build_payload(spec: ModelSpec, variant: str, images: list[Image.Image], json_object: bool = True) -> dict:
    content: list[dict] = [{"type": "text", "text": render("vlm_user.md.j2", variant=variant, grid_mm=int(GRID_MM))}]
    for image in images:
        part = {"url": data_url(image)}
        if spec.image_detail:
            part["detail"] = spec.image_detail
        content.append({"type": "image_url", "image_url": part})
    payload: dict = {
        "model": spec.openrouter_id,
        "messages": [{"role": "system", "content": render("vlm_system.md.j2")}, {"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    if spec.reasoning == "off":
        payload["reasoning"] = {"enabled": False}
    provider: dict = {}
    if spec.provider_order:
        provider = {"order": list(spec.provider_order), "allow_fallbacks": True}
    if spec.provider_ignore:
        provider["ignore"] = list(spec.provider_ignore)
    if provider:
        payload["provider"] = provider
    return payload


def parse_answer(text: str) -> dict | None:
    """JSON-объект из ответа: без ```-ограждений, первый объект, обязательное поле damaged."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    start = body.find("{")
    if start < 0:
        return None
    try:
        answer, _ = json.JSONDecoder().raw_decode(body[start:])
    except ValueError:
        return None
    if not isinstance(answer, dict) or "damaged" not in answer:
        return None
    return answer


def ask(client: OpenRouterClient, spec: ModelSpec, variant: str, images: list[Image.Image]) -> tuple[dict | None, dict]:
    """Один запрос; при эхе формата или мусоре — повтор без ``response_format``."""
    meta: dict = {"model": spec.name, "prompt_version": VLM_PROMPT_VERSION, "variant": variant, "cost_usd": 0.0}
    answer = None
    for json_object in (True, False):
        started = time.time()
        response = client.chat(build_payload(spec, variant, images, json_object))
        meta["cost_usd"] += response.cost_usd or 0.0
        meta.update(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_s=round(time.time() - started, 1),
            provider=response.provider,
            finish_reason=response.finish_reason,
            raw_text=response.text,
            json_object=json_object,
        )
        answer = parse_answer(response.text)
        if answer is not None:
            break
    if answer is None:
        meta["parse_error"] = "ответ без JSON-объекта с полем damaged"
    return answer, meta


def answer_path(out_dir: Path, variant: str, pdf: str, page: int, repeat: int) -> Path:
    return out_dir / "vlm" / variant / f"{pdf}_p{page:03d}_r{repeat}.json"


def is_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return record.get("answer") is not None and record.get("meta", {}).get("prompt_version") == VLM_PROMPT_VERSION


# ---------- сводка ---------------------------------------------------------------------------


def load_answers(out_dir: Path) -> list[dict]:
    records = []
    for path in sorted((out_dir / "vlm").glob("*/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        record["variant"] = path.parent.name
        records.append(record)
    return records


def vlm_markdown(records: list[dict], labels: dict) -> str:
    """Согласие с эталоном и между повторами, доля damaged по поясам, цена — по вариантам."""
    lines = ["# Пробник VLM: видит ли DeepSeek порчу геометрии", ""]
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_variant[r["variant"]].append(r)
    lines += [
        "| вариант | запросов | без ответа | $ всего | ¢/запрос | с/запрос | согласие повторов | эталон bad: damaged | эталон good: damaged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, items in sorted(by_variant.items()):
        cost = sum(r["meta"].get("cost_usd", 0.0) for r in items)
        latency = np.mean([r["meta"].get("latency_s", 0.0) for r in items]) if items else 0.0
        missing = sum(1 for r in items if r.get("answer") is None)
        by_page: dict[tuple, list[bool]] = defaultdict(list)
        for r in items:
            if r.get("answer") is not None:
                by_page[(r["pdf"], r["page"])].append(bool(r["answer"].get("damaged")))
        agree = [len(set(v)) == 1 for v in by_page.values() if len(v) >= 2]
        agreement = f"{100.0 * np.mean(agree):.0f} % ({len(agree)} стр)" if agree else "—"

        def share(label: str) -> str:
            keys = [k for k, (lab, _) in labels.items() if lab == label and k in by_page]
            if not keys:
                return "—"
            votes = [np.mean(by_page[k]) for k in keys]
            return f"{100.0 * np.mean(votes):.0f} % ({len(keys)} стр)"

        lines.append(
            f"| {variant} | {len(items)} | {missing} | {cost:.3f} | {100.0 * cost / max(1, len(items)):.3f} | {latency:.1f} | {agreement} | {share('bad')} | {share('good')} |"
        )
    lines += [
        "",
        "## Доля damaged по поясам score классического детектора",
        "",
        "| вариант | пояс | страниц | damaged (доля голосов) |",
        "|---|---|---|---|",
    ]
    for variant, items in sorted(by_variant.items()):
        by_belt: dict[str, list[float]] = defaultdict(list)
        for r in items:
            if r.get("answer") is not None:
                by_belt[r.get("belt", "?")].append(1.0 if r["answer"].get("damaged") else 0.0)
        for belt, votes in sorted(by_belt.items()):
            lines.append(f"| {variant} | {belt} | {len(votes)} | {100.0 * np.mean(votes):.0f} % |")
    lines += [
        "",
        "## По страницам",
        "",
        "| страница | метка/пояс | score | " + " | ".join(sorted(by_variant)) + " |",
        "|---|---|---|" + "---|" * len(by_variant),
    ]
    pages: dict[tuple, dict] = {}
    for r in records:
        key = (r["pdf"], r["page"])
        entry = pages.setdefault(
            key, {"belt": r.get("belt", "?"), "score": r.get("score", 0.0), "votes": defaultdict(list), "where": {}}
        )
        if r.get("answer") is not None:
            entry["votes"][r["variant"]].append("D" if r["answer"].get("damaged") else "-")
            if r["answer"].get("damaged"):
                entry["where"][r["variant"]] = str(r["answer"].get("where", ""))[:60]
    for key, entry in sorted(pages.items(), key=lambda kv: (-kv[1]["score"], kv[0])):
        cells = []
        for variant in sorted(by_variant):
            votes = "".join(entry["votes"].get(variant, [])) or "·"
            where = entry["where"].get(variant, "")
            cells.append(f"{votes} {where}".strip())
        lines.append(f"| {key[0]} с.{key[1]} | {entry['belt']} | {entry['score']:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
