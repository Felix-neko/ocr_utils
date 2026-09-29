"""Конвейер защиты сторон по полосам: разбор → выступы выровненных сторон → (CRAFT: фильтр глифов и повторный разбор) → пометка оставшихся выступов недостоверными; JSON, оверлеи, «было | стало»."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks.cli import _page_hints, make_engine
from ocr_utils.page_layout.text_blocks.overlay import draw, legend_entries
from ocr_utils.page_layout.text_blocks.page import PageAnalysis, Variant, analyse_gray, render_page
from ocr_utils.page_layout.text_blocks.report import page_json
from ocr_utils.page_layout.text_blocks.sides_overlay import side_by_side
from ocr_utils.page_layout.text_blocks.smooth_envelope import SmoothEnvelope
from research.edge_marks.guard.anomaly import BUMP_MM, SideBump, page_bumps, row_excess
from research.edge_marks.guard.craft_filter import keep_gray, side_zones
from research.edge_marks.guard.glyph_vote import Engine, load_maps, row_geometry

# Ширина страницы на оверлее, пиксели.
OVERLAY_WIDTH = 1400
# Выброшенная фильтром CRAFT краска — рамкой цвета «подсказка, вспомогательное» (палитра навыка draw-overlay).
COLOR_DROPPED = (200, 140, 60)


class GuardPass(str, Enum):
    """Вариант второго прохода (он же папка выхода). Выступы, оставшиеся по итогу, помечаются всегда."""

    PLAIN = "plain"  # без второго прохода: выступы первого разбора помечаются недостоверными
    CRAFT = "craft"  # фильтр глифов голосованием CRAFT, повторный разбор
    CRAFT_PERO = "craft_pero"  # голосование CRAFT + pero: сор, если так говорит хотя бы один
    CRAFT_PERO_DOCTR = "craft_pero_doctr"  # голосование CRAFT + pero + docTR

    @property
    def engines(self) -> tuple[Engine, ...]:
        """Голосующие детекторы варианта (пусто — второго прохода нет)."""
        return {
            GuardPass.PLAIN: (),
            GuardPass.CRAFT: (Engine.CRAFT,),
            GuardPass.CRAFT_PERO: (Engine.CRAFT, Engine.PERO),
            GuardPass.CRAFT_PERO_DOCTR: (Engine.CRAFT, Engine.PERO, Engine.DOCTR),
        }[self]

    @property
    def title(self) -> str:
        """Подпись варианта на оверлеях."""
        return {GuardPass.PLAIN: "без второго прохода", GuardPass.CRAFT: "CRAFT", GuardPass.CRAFT_PERO: "CRAFT + pero",
                GuardPass.CRAFT_PERO_DOCTR: "CRAFT + pero + docTR"}[self]  # fmt: skip


# Папка непомеченного первого разбора — только картинка «было» для склеек, не режим.
BEFORE = "before"


@dataclass(frozen=True)
class PageJob:
    """Полоса для прогона: ключ, бинаризованный PDF (nogeo) и страница с единицы."""

    key: str
    pdf: Path
    page: int


@dataclass(frozen=True)
class GuardSettings:
    """Настройки прогона: кэш подсказок page_layout, пороги выступа и карты CRAFT, папка карт."""

    layout_cache: Path | None
    bump_mm: float = BUMP_MM
    maps_dir: Path | None = None  # корень карт детекторов: ``<maps_dir>/<детектор>/<ключ>.png``


# Источник подсказок surya — один на процесс пула (открывается в инициализаторе).
_SURYA = None


def init_worker(layout_cache: Path | None) -> None:
    """Инициализатор воркера пула: hugepage и потоки BLAS выключены, источник подсказок surya открыт один раз.

    Args:
        layout_cache: Кэш ``page_layout`` или ``None``.
    """
    global _SURYA
    import threadpoolctl

    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)
    if layout_cache is not None:
        from ocr_utils.page_layout.surya import SuryaSourceConfig

        _SURYA = SuryaSourceConfig(layout_cache).open()


def analyse(gray300: np.ndarray, job: PageJob) -> PageAnalysis:
    """Разбор полосы с боевыми настройками (движок ink, гладкие границы, подсказки page_layout, вариант nogeo).

    Args:
        gray300: Серая полоса 300 dpi (исходная или с закрашенным сором).
        job: Полоса.

    Returns:
        Разбор.
    """
    hints = _page_hints(job.pdf, job.page, gray300, 150.0, _SURYA, None, Variant.NOGEO.value, False)
    engine = make_engine("ink", {"hints": hints})
    return analyse_gray(gray300, engine, name=job.pdf.stem, page=job.page, variant=Variant.NOGEO.value, hints=hints)


def mark_unreliable(analysis: PageAnalysis, bumps: list[SideBump]) -> PageAnalysis:
    """Добавить участки выступов к недостоверным участкам сторон огибающих блоков.

    Уже найденные недостоверные участки (ступеньки ``smooth_sides``) сохраняются. У огибающей без гладких сторон
    (ступенчатый запасной ход) нет полей недостоверности — она переводится в :class:`SmoothEnvelope` с теми же
    кривыми.

    Args:
        analysis: Разбор.
        bumps: Выступы (:func:`anomaly.page_bumps`).

    Returns:
        Разбор с дополненными ``unreliable_left/right``.
    """
    if not bumps:
        return analysis
    blocks = list(analysis.blocks)
    for number in sorted({bump.block for bump in bumps}):
        envelope = blocks[number].envelope
        if not isinstance(envelope, SmoothEnvelope):
            envelope = SmoothEnvelope(**{item.name: getattr(envelope, item.name) for item in fields(envelope)})
        spans = {"left": list(envelope.unreliable_left), "right": list(envelope.unreliable_right)}
        for bump in bumps:
            if bump.block == number:
                spans[bump.side].append((bump.y0, bump.y1))
        envelope = replace(
            envelope, unreliable_left=tuple(sorted(spans["left"])), unreliable_right=tuple(sorted(spans["right"]))
        )
        blocks[number] = replace(blocks[number], envelope=envelope)
    return replace(analysis, blocks=tuple(blocks))


def overlay_picture(analysis: PageAnalysis, gray300: np.ndarray, dropped: list, title: str) -> np.ndarray:
    """Оверлей разбора на ИСХОДНОЙ полосе: оси строк, границы блоков, недостоверные участки пунктиром и рамки
    краски, выброшенной фильтром CRAFT; шапка и легенда в полях.

    Args:
        analysis: Разбор.
        gray300: Исходная полоса 300 dpi.
        dropped: Рамки выброшенных компонент (пиксели рендера).
        title: Вторая строка шапки.

    Returns:
        Картинка BGR.
    """
    scale_page = OVERLAY_WIDTH / gray300.shape[1]
    page = cv2.resize(gray300, (OVERLAY_WIDTH, int(gray300.shape[0] * scale_page)), interpolation=cv2.INTER_AREA)
    canvas = draw(analysis, page, scale=OVERLAY_WIDTH / analysis.width)
    for x0, y0, x1, y1 in dropped:
        cv2.rectangle(canvas, (int(x0 * scale_page) - 3, int(y0 * scale_page) - 3),
                      (int(x1 * scale_page) + 3, int(y1 * scale_page) + 3), COLOR_DROPPED, 2)  # fmt: skip
    # Пунктир боевой легенды назван по старой причине («вынос за колонку») — здесь он же и выступ от сора.
    legend = [
        replace(entry, text="недостоверный участок стороны (выступ от сора / вынос за колонку)")
        if entry.style is SampleStyle.DASHED and "недостовер" in entry.text
        else entry
        for entry in legend_entries()
    ]
    legend.append(LegendEntry("выброшено фильтром второго прохода", COLOR_DROPPED, style=SampleStyle.BOX))
    header = [f"{analysis.name} с.{analysis.page} [{analysis.variant}] движок {analysis.engine}", title]
    return framed(canvas, header, legend)


def excess_rows(analysis: PageAnalysis) -> list[dict]:
    """Местный выступ у каждой строки выровненных сторон (для калибровки порога).

    Returns:
        Строки ``block, side, row, y, excess_mm, kind``.
    """
    from research.edge_marks.guard.anomaly import ALIGNED_SIDES

    out = []
    for number, (block, alignment) in enumerate(zip(analysis.blocks, analysis.alignments)):
        for side in ALIGNED_SIDES.get(alignment.kind, ()):
            if not getattr(alignment, side).aligned:
                continue
            for index, value in enumerate(row_excess(block, side)):
                out.append({"block": number, "side": side, "row": index, "y": float(block.rows[index].y),
                            "excess_mm": round(float(value), 3), "kind": alignment.kind.value})  # fmt: skip
    return out


def write_page(analysis: PageAnalysis, gray: np.ndarray, summary: dict, dropped: list, title: str, out_dir: Path) -> None:
    """Выход полосы: ``pages/<ключ>.json`` (разбор + поле ``edge_guard``), ``overlays/<ключ>.jpg``,
    ``excess/<ключ>.json`` (местный выступ каждой строки выровненных сторон).

    Args:
        analysis: Разбор.
        gray: Исходная полоса 300 dpi.
        summary: Сводка защиты сторон.
        dropped: Рамки выброшенной краски (пиксели рендера).
        title: Вторая строка шапки оверлея.
        out_dir: Папка прохода.
    """
    key = summary["key"]
    for sub in ("pages", "overlays", "excess"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    (out_dir / "pages" / f"{key}.json").write_text(
        json.dumps({**page_json(analysis), "edge_guard": summary}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_dir / "excess" / f"{key}.json").write_text(json.dumps(excess_rows(analysis)))
    cv2.imwrite(str(out_dir / "overlays" / f"{key}.jpg"), overlay_picture(analysis, gray, dropped, title),
                [cv2.IMWRITE_JPEG_QUALITY, 88])  # fmt: skip


def run_page(job: PageJob, variant: GuardPass, settings: GuardSettings, base: Path) -> dict:
    """Одна полоса: разбор, выступы выровненных сторон, (с CRAFT — фильтр и повторный разбор), пометка оставшихся.

    Выступы, оставшиеся по итогу, помечаются недостоверными ВСЕГДА: без CRAFT — выступы первого разбора, с CRAFT —
    выступы разбора после фильтрации (если фильтр ничего не выбросил, повторного разбора нет и это те же выступы).
    Проход без CRAFT дополнительно пишет непомеченный первый разбор в ``<base>/before`` — картинку «было».

    Args:
        job: Полоса.
        variant: Вариант второго прохода на аномальной полосе (нужны карты его детекторов в ``settings.maps_dir``).
        settings: Настройки.
        base: Папка набора; проход пишет в ``<base>/<вариант>``.

    Returns:
        Сводка полосы: выступы первого разбора и итоговые, выброшенные компоненты, время.
    """
    started = time.monotonic()
    gray = render_page(job.pdf, job.page)
    analysis = analyse(gray, job)
    first_seconds = time.monotonic() - started
    before = page_bumps(analysis, settings.bump_mm)
    name = variant
    summary = {"key": job.key, "pass": name.value, "anomalous": bool(before),
               "bumps_before": [asdict(b) | {"kind": b.kind.value} for b in before]}  # fmt: skip
    if variant is GuardPass.PLAIN:
        # Непомеченный первый разбор — «было» для склеек.
        write_page(analysis, gray, {**summary, "bumps_after": summary["bumps_before"], "dropped": []}, [],
                   f"без пометок; выступов {len(before)}", base / BEFORE)  # fmt: skip
    after, dropped, craft_seconds = before, [], 0.0
    if variant.engines and before:
        tick = time.monotonic()
        maps = load_maps(settings.maps_dir, job.key, variant.engines)
        zone = side_zones(analysis, before, gray.shape)
        blocks = {bump.block for bump in before}
        filtered, dropped = keep_gray(gray, zone, maps, partial(row_geometry, analysis, blocks))
        if dropped:
            analysis = analyse(filtered, job)
            after = page_bumps(analysis, settings.bump_mm)
        craft_seconds = time.monotonic() - tick
    # Итоговые выступы — недостоверные участки сторон, всегда.
    analysis = mark_unreliable(analysis, after)
    summary |= {
        "bumps_after": [asdict(b) | {"kind": b.kind.value} for b in after],
        "dropped": dropped,
        "seconds_analysis": round(first_seconds, 2),
        "seconds_craft_reanalysis": round(craft_seconds, 2),
        "blocks": len(analysis.blocks),
        "lines": len(analysis.axes),
        "alignments": [a.kind.value for a in analysis.alignments],
    }
    title = (f"защита сторон, {variant.title}: выступов в первом разборе {len(before)}, "
             f"итоговых (помечены) {len(after)}, выброшено фильтром {len(dropped)}")  # fmt: skip
    write_page(analysis, gray, summary, dropped, title, base / name.value)
    return summary


def run_pass(variant: GuardPass, selected: list[PageJob], jobs: int, settings: GuardSettings, base: Path) -> list[dict]:
    """Прогнать полосы пулом процессов (разбор — CPU, видеокарта не нужна): ``<base>/<проход>/…`` и
    ``<base>/summary_<проход>.json``.

    Args:
        variant: Вариант второго прохода.
        selected: Полосы.
        jobs: Процессов пула.
        settings: Настройки.
        base: Папка набора.

    Returns:
        Сводки полос в порядке ``selected``.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    context = multiprocessing.get_context("forkserver")
    worker = partial(run_page, variant=variant, settings=settings, base=base)
    with ProcessPoolExecutor(jobs, mp_context=context, initializer=init_worker, initargs=(settings.layout_cache,)) as pool:
        out = list(pool.map(worker, selected))
    base.mkdir(parents=True, exist_ok=True)
    name = variant
    (base / f"summary_{name.value}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def compare(key: str, dirs: list[Path], titles: list[str], out: Path) -> None:
    """Склейка оверлеев одной полосы из нескольких проходов рядом («было | стало»).

    Args:
        key: Полоса.
        dirs: Папки проходов (в каждой ``overlays/<ключ>.jpg``).
        titles: Подписи над картинками.
        out: Куда писать JPEG.
    """
    images = [cv2.imread(str(d / "overlays" / f"{key}.jpg")) for d in dirs]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), side_by_side(images, titles), [cv2.IMWRITE_JPEG_QUALITY, 85])


__all__ = ["BEFORE", "GuardPass", "GuardSettings", "PageJob", "analyse", "compare", "init_worker", "mark_unreliable", "run_page", "run_pass", "write_page"]
