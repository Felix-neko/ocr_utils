"""Цепочка на одну таблицу: разбор (CPU, в пуле) и перекройка с набором (CPU, в пуле).

ДВЕ ФАЗЫ, А НЕ ОДНА, потому что между ними — surya на GPU в родительском процессе
(``second_opinion``): воркер не может позвать её сам. :func:`analyse` строит сетку,
определяет поворот ячеек и читает боковые tesseract-ом; сомнительные ячейки уезжают в
родителя на второе мнение; :func:`rewrite` подбирает кегль, считает DPI, при нужде
перекраивает таблицу и набирает текст. Полноразмерная вырезка между фазами не передаётся
(6 МБ на таблицу, 900 таблиц — 5 ГБ в родителе): вторая фаза перечитывает её с диска за
доли секунды.

ЛЕСТНИЦА РЕШЕНИЙ (по условию задачи):

1. Кегль подбирается в существующих границах ячеек. Самый мелкий из вписанных даёт
   требуемый DPI страницы (``dpi.required_dpi``). Не выше потолка — увеличиваем и набираем.
2. Иначе — поворот всей таблицы на 90° в ту или другую сторону: бывшие боковые ячейки
   становятся прямыми пикселями, бывшие прямые читаются и набираются заново. Если у
   лучшего варианта DPI укладывается в потолок — берём его.
3. Иначе — расширение колонок с боковыми ячейками ровно настолько, чтобы текст влез
   кеглем, который дорастёт до порога при потолочном DPI; затем увеличение до потолка.

ИСКЛЮЧЕНИЕ — ТАБЛИЦА, НАПЕЧАТАННАЯ БОКОМ ЦЕЛИКОМ (лежит половина ячеек и больше): для
неё поворот пробуется ПЕРВЫМ. Смоук на трёх таких таблицах пака показал, почему: набор
в прежних границах формально укладывался в потолок DPI, но подменял прочитанным текстом
десятки ячеек, включая боковые числа, которых он читать не умеет, и куски приписки,
порезанной сеткой, — тогда как поворот выпрямляет их все разом, без единого чтения.

Набор всегда идёт УЖЕ на увеличенной картинке, а укладка пересчитывается в конечном
разрешении: кегль, подобранный в исходном и умноженный на масштаб, из-за округлений мог
бы вылезти за поле на пиксель.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ocr_utils.scan_markup.rotation import rotate_cw as rotate_image
from ocr_utils.page_layout.geometry import Box, Cell, Grid
from ocr_utils.scan_markup.table_detection.ruling import mm_to_px

from ocr_utils.rotated_text.tables.dpi import MAX_DPI, MIN_FONT_EM_PX, font_cap_px, font_px_for_dpi, required_dpi
from ocr_utils.rotated_text.tables.fit import FIT_MIN_PX, MAX_LINES, Fit, fit_text, required_width
from ocr_utils.rotated_text.tables.ocr import LANGUAGES, CellText, read_cell
from ocr_utils.rotated_text.tables.orientation import (
    ALNUM,
    LETTER,
    CellOrientation,
    evidence_at,
    apply_prior,
    has_word,
    is_sideways_table,
    looks_like_text,
    orient_cell,
    sideways_share,
    upright_interior,
)
from ocr_utils.rotated_text.tables.render import Replacement, paper_of, render, upscale
from ocr_utils.rotated_text.tables.reshape import rotate_grid, widen_columns
from ocr_utils.rotated_text.tables.second_opinion import (
    RELIABLE_CONFIDENCE,
    SURYA_MIN_CONFIDENCE,
    SURYA_SHORT_CONFIDENCE,
    accept,
    cyrillic,
)
from ocr_utils.rotated_text.tables.source import TableRef, load_table_crop
from ocr_utils.rotated_text.tables.structure import (
    analyse_structure,
    cell_interior,
    ink_level,
    interior_box,
    merge_split_cells,
    native_font_px,
    scale_grid,
    work_copy,
)

logger = logging.getLogger(__name__)

ACTION_NONE = "none"  # боковых ячеек нет, таблица не трогалась
ACTION_AS_IS = "as_is"  # набрано в прежних границах без увеличения
ACTION_UPSCALE = "upscale"  # набрано в прежних границах, страницу надо увеличить
ACTION_ROTATE = "rotate"  # таблица повёрнута целиком
ACTION_WIDEN = "widen"  # колонки раздвинуты

# Число без букв подменяется только при такой уверенности tesseract: знак «∅» перед
# диаметром он читает как цифру, и «∅ 90—120» превращается в «290—120».
NUMERIC_CONFIDENCE = 0.9

# Перевёрнутая ячейка подменяется только при такой уверенности: у ложных «180»
# (перевёрнутые цифры) она 0.49–0.81, у настоящих — от 0.85 (замер по 115 ячейкам пака).
CONFIDENCE_180 = 0.8

# Любая подмена — при уверенности не ниже этой. Замер по 1383 набранным ячейкам пака:
# ниже 0.6 — 16 ячеек, почти все мусор («потрё НОСТЬ Г.)» вместо «Пятидневная потребность»
# на чертёжном курсиве бланка, «О Я Хх; 52 И РЕК м»); 0.6–0.7 — 26, половина мусор;
# от 0.8 — 1333, читаемые. Сомнительные ниже 0.7 идут на второе мнение surya.
CONFIDENCE_REPLACE = 0.6

# Россыпь: больше половины токенов — в один знак («-Я 3 У м 2 Я я у сво»). Слово в такой
# россыпи бывает («сво»), но это не текст ячейки, а чтение штрихов.
JUNK_TOKEN_SHARE = 0.5

# Ячейка не крупнее стольких компонент, чья ось по форме не видна, при повороте таблицы
# боком-большинством может остаться непрочитанной: это чёрточка или одиночный знак. Ячейка
# крупнее или с уверенно стоячей осью («6П13С» — пять компонент) при повороте обязана быть
# прочитана, иначе поворот отвергается (замер: 1967/07 IMG_0044 оставалась с «6П13С» боком).
WEAK_GLYPH_COMPONENTS = 2


@dataclass
class Options:
    work_dpi: int = 300
    allowed: tuple[int, ...] = (0, 90, 180, 270)
    lang: str = LANGUAGES
    # Меньше букв в прочитанном — подменять нечем: «№№ п/п» остаётся как есть.
    min_letters: int = 3
    deskew: bool = True
    max_dpi: int = MAX_DPI
    min_font_em_px: int = MIN_FONT_EM_PX
    # Запас при расширении колонки, чтобы округления не съели последний пиксель.
    widen_slack_mm: float = 1.0
    use_surya: bool = True


@dataclass
class CellRecord:
    """Всё, что известно про ячейку: геометрия (в рабочем dpi), поворот, текст, набор."""

    row: int
    col: int
    row_span: int
    col_span: int
    is_header: bool
    box: tuple[int, int, int, int]
    inner: "tuple[int, int, int, int] | None"
    rotate_cw: "int | None" = None
    aspect: float = 0.0
    components: int = 0
    letters: dict[int, int] = field(default_factory=dict)
    orientation_note: str = ""
    # Ось по форме букв: True — стоит, False — лежит, None — неясно (мало компонент).
    axis_upright: "bool | None" = None
    # Сторона взята от большинства таблицы, а не прочитана в ячейке.
    side_from_prior: bool = False
    # Поворот ячейки ДО поворота всей таблицы — для оверлея «было».
    rotate_before: "int | None" = None
    text_tesseract: str = ""
    conf_tesseract: float = 0.0
    text_surya: "str | None" = None
    conf_surya: "float | None" = None
    surya_note: str = ""
    text: str = ""
    engine: str = ""
    # Сколько строк прочитал движок: абзац из многих строк в транспонированную ячейку не
    # переложить без потери строк.
    ocr_lines: int = 0
    candidate: bool = False
    font_px: "int | None" = None
    lines: int = 0
    replaced: bool = False
    reason: str = ""

    @property
    def key(self) -> tuple[int, int]:
        return (self.row, self.col)

    @property
    def rotated(self) -> bool:
        return self.rotate_cw not in (None, 0)

    def letter_count(self) -> int:
        return len(LETTER.findall(self.text))


def junk_share(text: str) -> float:
    """Доля токенов, в которых меньше двух букв или цифр."""
    tokens = text.split()
    if len(tokens) < 3:
        return 0.0
    return sum(1 for token in tokens if len(ALNUM.findall(token)) < 2) / len(tokens)


def replaceable(
    record: CellRecord,
    confidence: float,
    min_letters: int,
    floor: float = CONFIDENCE_REPLACE,
    short_floor: float = NUMERIC_CONFIDENCE,
) -> tuple[bool, str]:
    """Можно ли подменить прочитанным. Два пути: есть слово от ``min_letters`` букв и букв
    достаточно — текст; иначе (число, «0», «6П13С», «шт.») — только при уверенности
    tesseract от ``NUMERIC_CONFIDENCE``: короткое он читает либо чисто, либо выдумывает."""
    if not looks_like_text(record.text):
        return False, "прочитанное не похоже на текст"
    if confidence < floor:
        return False, f"уверенность {confidence:.2f} ниже {floor}"
    if junk_share(record.text) > JUNK_TOKEN_SHARE:
        return False, "россыпь одиночных знаков"
    if record.rotate_cw == 180 and confidence < CONFIDENCE_180:
        return False, f"перевёрнутая ячейка прочитана с уверенностью {confidence:.2f}, ниже {CONFIDENCE_180}"
    letters = record.letter_count()
    if letters >= min_letters and has_word(record.text, min_letters):
        return True, ""
    if confidence >= short_floor:
        return True, ""
    return False, f"коротко ({letters} букв) и неуверенно ({confidence:.2f})"


@dataclass
class TableAnalysis:
    """Результат первой фазы. ``grid`` и ``native_font_px`` — в рабочем dpi."""

    ref: TableRef
    work_dpi: int
    source_dpi: int
    deskew_deg: float = 0.0
    crop_size: tuple[int, int] = (0, 0)
    grid: Grid = field(default_factory=lambda: Grid(xs=[], ys=[], cells=[]))
    paper: int = 255
    ink: int = 0
    native_font_px: float = 0.0
    cells: list[CellRecord] = field(default_factory=list)
    second_opinion: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    # Таблица напечатана боком целиком, и в какую сторону её повернуть по часовой.
    sideways_table: bool = False
    sideways_side: "int | None" = None
    sideways_share: float = 0.0
    merged_cells: int = 0
    error: str = ""
    seconds: float = 0.0

    @property
    def has_rotated_text(self) -> bool:
        return any(cell.candidate for cell in self.cells) or self.sideways_table

    def cell(self, key: tuple[int, int]) -> CellRecord:
        for cell in self.cells:
            if cell.key == key:
                return cell
        raise KeyError(key)


def _orient_all(work: np.ndarray, grid: Grid, options: Options) -> dict[tuple[int, int], CellOrientation]:
    return {
        cell.key: orient_cell(
            cell_interior(work, cell, options.work_dpi), options.work_dpi, options.allowed, options.lang
        )
        for cell in grid.cells
    }


def analyse(ref: TableRef, options: Options) -> TableAnalysis:
    """Первая фаза: вырезка, сетка, поворот ячеек, чтение боковых."""
    started = time.time()
    result = TableAnalysis(ref=ref, work_dpi=options.work_dpi, source_dpi=ref.dpi)
    try:
        crop = load_table_crop(ref, deskew=options.deskew)
    except Exception as error:  # noqa: BLE001 — одна битая полоса не должна валить прогон
        result.error = f"не удалось вырезать таблицу: {error}"
        return result
    result.source_dpi = crop.dpi
    result.deskew_deg = crop.deskew_deg
    result.crop_size = (crop.gray.shape[1], crop.gray.shape[0])
    work = work_copy(crop.gray, crop.dpi, options.work_dpi)
    structure = analyse_structure(work, options.work_dpi)
    grid = structure.grid
    result.paper, result.ink = structure.paper, structure.ink
    if not grid.cells:
        result.grid = grid
        result.error = "сетка ячеек не построена"
        result.seconds = time.time() - started
        return result

    # Склейка объединённых ячеек, которые сетка разрезала (текст пересекает границу без линейки).
    merged = merge_split_cells(grid, structure.lines, work, options.work_dpi)
    result.merged_cells = len(grid.cells) - len(merged.cells)
    grid = merged
    result.grid = grid
    verdicts = _orient_all(work, grid, options)
    result.sideways_side = apply_prior(verdicts)
    result.sideways_table = is_sideways_table(verdicts) and result.sideways_side is not None
    result.sideways_share = round(sideways_share(verdicts)[0], 3)

    for cell in grid.cells:
        verdict = verdicts[cell.key]
        record = CellRecord(
            row=cell.row,
            col=cell.col,
            row_span=cell.row_span,
            col_span=cell.col_span,
            is_header=cell.is_header,
            box=cell.box.as_tuple(),
            inner=cell.inner.as_tuple() if cell.inner is not None else None,
            rotate_cw=verdict.rotate_cw,
            aspect=round(verdict.aspect, 3),
            components=verdict.components,
            letters=dict(verdict.letters),
            orientation_note=verdict.note,
            axis_upright=None if verdict.axis_sideways is None else not verdict.axis_sideways,
            side_from_prior=verdict.from_prior,
        )
        if record.rotated:
            upright = upright_interior(work, cell, options.work_dpi, verdict.rotate_cw)
            text = read_cell(upright, options.lang)
            record.text_tesseract, record.conf_tesseract = text.text, round(text.confidence, 3)
            record.text, record.engine, record.ocr_lines = text.text, text.engine, len(text.lines)
            record.candidate, record.reason = replaceable(record, text.confidence, options.min_letters)
            reliable = record.candidate and text.confidence >= RELIABLE_CONFIDENCE
            if options.use_surya and not reliable:
                result.second_opinion[record.key] = upright
        result.cells.append(record)

    known = [(cell, verdicts[cell.key].rotate_cw) for cell in grid.cells if verdicts[cell.key].rotate_cw is not None]
    result.native_font_px = native_font_px(work, known, options.work_dpi)
    result.seconds = time.time() - started
    return result


def apply_second_opinion(analysis: TableAnalysis, key: tuple[int, int], surya: CellText, options: Options) -> bool:
    """Подставить ответ surya, если он прошёл фильтр. Возвращает, принят ли."""
    record = analysis.cell(key)
    record.text_surya, record.conf_surya = cyrillic(surya.text), round(surya.confidence, 3)
    ours = CellText(lines=[record.text_tesseract], confidence=record.conf_tesseract, engine="tesseract")
    reliable = record.candidate and record.conf_tesseract >= RELIABLE_CONFIDENCE
    accepted, why = accept(ours, surya, reliable, ours_plausible=record.candidate)
    record.surya_note = why
    if accepted:
        record.text, record.engine = record.text_surya, "surya"
        # Пороги уверенности у surya свои: её 0.5–0.6 на курсивном бланке — верные чтения.
        record.candidate, record.reason = replaceable(
            record, surya.confidence, options.min_letters, SURYA_MIN_CONFIDENCE, SURYA_SHORT_CONFIDENCE
        )
    return accepted


@dataclass
class TableRewrite:
    """Результат второй фазы, без картинок — они отдаются рядом."""

    table_id: str
    action: str
    steps: list[str] = field(default_factory=list)
    table_rotate_cw: int = 0
    scale: float = 1.0
    required_dpi: "int | None" = None
    final_dpi: int = 0
    widened: dict[int, int] = field(default_factory=dict)
    min_font_px: float = 0.0
    native_font_px: float = 0.0
    sideways_table: bool = False
    size_before: tuple[int, int] = (0, 0)
    size_after: tuple[int, int] = (0, 0)
    cells: list[CellRecord] = field(default_factory=list)
    grid_before: dict = field(default_factory=dict)
    grid_after: dict = field(default_factory=dict)
    note: str = ""
    seconds: float = 0.0

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["cells"] = [asdict(cell) for cell in self.cells]
        return payload


@dataclass
class RewriteImages:
    before: np.ndarray  # вырезка в исходном dpi
    grid_before: Grid  # её сетка в исходном dpi
    after: np.ndarray  # результат в конечном dpi


# Одна замена: запись ячейки, внутренность (в разрешении картинки-основы) и текст.
_Spec = tuple[CellRecord, Box, str]


def _min_font(fits: list["Fit | None"]) -> float:
    if not fits or any(fit is None for fit in fits):
        return 0.0
    return float(min(fit.font_px for fit in fits if fit is not None))


def _plan_fits(spec: list[_Spec], dpi: int, native_px: float, options: Options) -> tuple[float, "int | None"]:
    """Самый мелкий кегль плана и требуемый DPI. Пустой план — ничего набирать не надо."""
    if not spec:
        return float(options.min_font_em_px), dpi
    cap = font_cap_px(dpi, native_px, options.min_font_em_px)
    fits = [fit_text(text, inner.width, inner.height, cap) for _, inner, text in spec]
    smallest = _min_font(fits)
    return smallest, required_dpi(dpi, smallest, options.min_font_em_px)


@dataclass
class _Plan:
    """Один из вариантов перекройки: чем он кончается и что в нём набирать."""

    name: str
    image: np.ndarray
    grid: Grid
    spec: list[_Spec]
    smallest: float
    required: "int | None"
    rotate_cw: int = 0
    widened: dict[int, int] = field(default_factory=dict)
    # Ячейки, которые при повороте сочтены стоящими (их поворот не выпрямляет).
    standing: set[tuple[int, int]] = field(default_factory=set)

    @property
    def feasible(self) -> bool:
        return self.required is not None


def rewrite(analysis: TableAnalysis, options: Options) -> tuple[TableRewrite, "RewriteImages | None"]:
    """Вторая фаза: подбор кегля, DPI, лестница решений, набор."""
    started = time.time()
    ref = analysis.ref
    result = TableRewrite(
        table_id=ref.table_id,
        action=ACTION_NONE,
        cells=analysis.cells,
        final_dpi=analysis.source_dpi,
        sideways_table=analysis.sideways_table,
    )
    if analysis.error:
        result.note = analysis.error
        return result, None
    crop = load_table_crop(ref, deskew=options.deskew)
    gray = crop.gray
    height, width = gray.shape[:2]
    dpi = crop.dpi
    factor = dpi / analysis.work_dpi
    grid_src = scale_grid(analysis.grid, factor)
    native_src = analysis.native_font_px * factor
    result.native_font_px = round(native_src, 1)
    result.size_before = (width, height)
    result.grid_before = grid_src.to_json()
    cells_src = list(grid_src.cells)
    if len(cells_src) != len(analysis.cells):
        result.note = "сетка разошлась с разбором"
        return result, None

    targets = [(rec, cell) for rec, cell in zip(analysis.cells, cells_src) if rec.candidate]
    if not targets and not analysis.sideways_table:
        result.size_after = result.size_before
        result.grid_after = result.grid_before
        result.seconds = time.time() - started
        return result, RewriteImages(before=gray, grid_before=grid_src, after=gray)

    def plan_in_place() -> _Plan:
        spec: list[_Spec] = [(rec, interior_box(cell, dpi), rec.text) for rec, cell in targets]
        smallest, required = _plan_fits(spec, dpi, native_src, options)
        return _Plan("в прежних границах", gray, grid_src, spec, smallest, required)

    def plan_rotated(degrees: int) -> _Plan:
        grid_rot = rotate_grid(grid_src, width, height, degrees)
        spec, why, standing = _rotated_spec(analysis, grid_rot, degrees, dpi, options)
        if spec is None:
            return _Plan(f"поворот {degrees}: {why}", gray, grid_rot, [], 0.0, None, rotate_cw=degrees)
        smallest, required = _plan_fits(spec, dpi, native_src, options)
        return _Plan(
            f"поворот {degrees}",
            rotate_image(gray, degrees),
            grid_rot,
            spec,
            smallest,
            required,
            rotate_cw=degrees,
            standing=standing,
        )

    def plan_widened(base: _Plan) -> _Plan:
        need_font = font_px_for_dpi(dpi, options.max_dpi, options.min_font_em_px)
        slack = mm_to_px(options.widen_slack_mm, dpi)
        extra: dict[int, int] = {}
        for rec, inner, text in base.spec:
            need = required_width(text, need_font, inner.height) + slack
            if need > inner.width:
                column = rec.col + rec.col_span - 1
                extra[column] = max(extra.get(column, 0), need - inner.width)
        image, grid = widen_columns(gray, grid_src, extra)
        by_key = {cell.key: cell for cell in grid.cells}
        spec = [(rec, interior_box(by_key[rec.key], dpi), text) for rec, _, text in base.spec]
        smallest, required = _plan_fits(spec, dpi, native_src, options)
        return _Plan(
            f"расширение колонок {dict(sorted(extra.items()))}", image, grid, spec, smallest, required, widened=extra
        )

    def fits_ceiling(plan: _Plan) -> bool:
        return plan.feasible and plan.required <= options.max_dpi

    def describe(plan: _Plan) -> str:
        return f"{plan.name}: кегль {plan.smallest:.0f} px, нужно {plan.required} dpi, набирать ячеек: {len(plan.spec)}"

    chosen: "_Plan | None" = None
    # Таблица боком целиком: сперва поворот в сторону большинства.
    if analysis.sideways_table and analysis.sideways_side:
        plan = plan_rotated(analysis.sideways_side)
        result.steps.append(describe(plan))
        if fits_ceiling(plan):
            chosen = plan
    if chosen is None:
        plan = plan_in_place()
        result.steps.append(describe(plan))
        if fits_ceiling(plan):
            chosen = plan
        else:
            best: "_Plan | None" = None
            for degrees in (90, 270):
                if analysis.sideways_table and degrees == analysis.sideways_side:
                    continue  # уже пробовали
                candidate = plan_rotated(degrees)
                result.steps.append(describe(candidate))
                if candidate.feasible and (best is None or candidate.required < best.required):
                    best = candidate
            if best is not None and fits_ceiling(best):
                chosen = best
            else:
                chosen = plan_widened(plan)
                result.steps.append(describe(chosen))
                if not fits_ceiling(chosen):
                    result.note = "не влезло даже после расширения колонок"
                    chosen.required = options.max_dpi

    if chosen.rotate_cw:
        _mark_after_rotation(analysis.cells, chosen)
    result.widened = dict(sorted(chosen.widened.items()))
    result.table_rotate_cw = chosen.rotate_cw

    # Набор в конечном разрешении.
    required = chosen.required
    scale = required / dpi
    final = upscale(chosen.image, scale)
    grid_final = scale_grid(chosen.grid, scale)
    cap_final = font_cap_px(required, native_src * scale, options.min_font_em_px)
    replacements: list[Replacement] = []
    for rec, inner, text in chosen.spec:
        inner_final = inner.scaled(scale)
        # Тот же подбор, что в плане (целые слова важнее кегля), только в конечном
        # разрешении; из-за округлений кегль может недобрать до порога пиксель-другой.
        fit = fit_text(text, inner_final.width, inner_final.height, cap_final, FIT_MIN_PX)
        if fit is None:
            rec.replaced, rec.reason = False, "не влезло"
            continue
        if fit.font_px < options.min_font_em_px:
            rec.reason = f"кегль {fit.font_px} px ниже порога {options.min_font_em_px} px"
        rec.replaced, rec.font_px, rec.lines, rec.text = True, fit.font_px, len(fit.lines), text
        replacements.append(Replacement(inner_final, fit, paper_of(final, inner_final)))
    after = render(final, replacements, ink_level(final))

    result.scale = round(scale, 4)
    result.required_dpi = required
    result.final_dpi = required
    result.min_font_px = round(chosen.smallest, 1)
    result.size_after = (after.shape[1], after.shape[0])
    result.grid_after = grid_final.to_json()
    if result.table_rotate_cw:
        result.action = ACTION_ROTATE
    elif result.widened:
        result.action = ACTION_WIDEN
    elif scale > 1.0 + 1e-6:
        result.action = ACTION_UPSCALE
    else:
        result.action = ACTION_AS_IS
    result.seconds = time.time() - started
    return result, RewriteImages(before=gray, grid_before=grid_src, after=after)


def _mark_after_rotation(cells: list[CellRecord], plan: _Plan) -> None:
    """Переписать поворот каждой ячейки так, как он выглядит ПОСЛЕ поворота таблицы.

    Таблица повёрнута на ``d`` по часовой — значит, ячейке, которой требовалось ``r``,
    теперь требуется ``(r − d) mod 360``: той, что лежала в ту же сторону, — 0 (её поворот
    выпрямил), той, что стояла, — ``−d`` (таблица по часовой → ячейка «270», то есть 90
    против часовой), и дальше она обрабатывается как любая повёрнутая ячейка. Ячейка с
    неизвестным поворотом считается лежавшей, если чтение так и рассудило, иначе — стоявшей.
    """
    d = plan.rotate_cw
    for rec in cells:
        if rec.components == 0:
            continue
        rec.rotate_before = rec.rotate_cw
        was = rec.rotate_cw
        if was is None or rec.side_from_prior:
            was = 0 if rec.key in plan.standing else d
        rec.rotate_cw = (was - d) % 360
        rec.side_from_prior = False
        if rec.rotate_cw == 0:
            rec.reason = "выпрямлена поворотом таблицы"
            rec.candidate = False


def _rotated_spec(
    analysis: TableAnalysis, grid_rot: Grid, degrees: int, dpi: int, options: Options
) -> tuple["list[_Spec] | None", str, set[tuple[int, int]]]:
    """Что придётся набрать, если повернуть таблицу на ``degrees``, — или почему нельзя.

    ПРАВИЛО БЕЗ ИСКЛЮЧЕНИЙ: после поворота ни одна ячейка с краской не должна остаться
    лежащей. Ячейка либо выпрямляется самим поворотом (её текст лежал в ту же сторону),
    либо читается и набирается заново; если ни то ни другое не доказано — поворот
    невозможен, и таблица идёт на набор в прежних границах или расширение колонок.
    Первый прогон по паку этого не требовал, и в 1967/07 IMG_0044 повёрнутая таблица
    вышла с «5», «10», «13» и «6П13С» боком, а в 1967/08 IMG_0059 — с пятью столбцами
    чисел боком.

    Ячейка с известным поворотом: 0 — читается здесь и набирается; равный ``degrees`` —
    выпрямляется; иной (180, другая сторона) — набирается, если прочитана надёжно.
    Ячейка с неизвестным поворотом (мало компонент или букв нет): читается прямо и так,
    как лежит большинство; где букв и цифр больше — так она и стоит. Стоит — набирается
    (с требованием уверенности к короткому: «0», «13», «шт.»); лежит — выпрямится сама;
    не читается никак — это дефис или галочка, читать в ней нечего и FineReader-у.
    Абзац из многих строк (неразлинованное тело) в транспонированную ячейку не
    перекладывается — строки слиплись бы, — и такой поворот тоже невозможен."""
    work = None

    def work_copy_lazy() -> np.ndarray:
        nonlocal work
        if work is None:
            crop = load_table_crop(analysis.ref, deskew=options.deskew)
            work = work_copy(crop.gray, crop.dpi, analysis.work_dpi)
        return work

    spec: list[_Spec] = []
    standing_keys: set[tuple[int, int]] = set()
    for index, (rec, cell_rot) in enumerate(zip(analysis.cells, grid_rot.cells)):
        if rec.components == 0:
            continue
        unknown = rec.rotate_cw is None or rec.side_from_prior
        if rec.rotate_cw == degrees and not unknown:
            continue
        where = f"ячейка ({rec.row},{rec.col})"
        cell_work = analysis.grid.cells[index]
        if unknown:
            interior = cell_interior(work_copy_lazy(), cell_work, analysis.work_dpi)
            single = rec.components <= WEAK_GLYPH_COMPONENTS
            standing = evidence_at(interior, 0, single, options.lang)
            lying = evidence_at(interior, degrees, single, options.lang)
            rec.letters = {**rec.letters, 0: standing, degrees: lying}
            # Слова нет ни с одной стороны — это чёрточки, обрезанные буквы, штрихи формул:
            # читать в ячейке нечего ни нам, ни FineReader-у, и поворот она не блокирует.
            # Ничья при читаемом слове решается осью по форме букв.
            if lying > standing or (lying == standing and (lying == 0 or rec.axis_upright is not True)):
                rec.reason = "выпрямлена поворотом таблицы" if lying else "слова не читается, остаётся как есть"
                continue
            standing_keys.add(rec.key)
            rec.rotate_cw, rec.text, rec.candidate = None, "", False  # стоит — читаем прямо
        if rec.rotated and not rec.candidate:
            return None, f"{where} повёрнута на {rec.rotate_cw} и прочитана ненадёжно", standing_keys
        text = rec.text
        if not rec.rotated and not text:
            read = read_cell(upright_interior(work_copy_lazy(), cell_work, analysis.work_dpi, 0), options.lang)
            text = read.text
            rec.text_tesseract, rec.conf_tesseract, rec.engine = text, round(read.confidence, 3), read.engine
            rec.text, rec.ocr_lines = text, len(read.lines)
        if not rec.rotated:
            ok, why = replaceable(rec, rec.conf_tesseract, options.min_letters)
            if not ok and unknown and rec.axis_upright is False:
                # Ось по форме букв лежит, а «стоит» сказали два случайных знака, которые
                # нечем подменить: кусок боковой приписки читается прямо как «я я». Форма
                # букв здесь надёжнее чтения — ячейка лежит, поворот её выпрямит.
                standing_keys.discard(rec.key)
                rec.reason = f"чтение прямо ненадёжно ({why}), по форме букв лежит"
                continue
            weak_glyph = rec.components <= WEAK_GLYPH_COMPONENTS and rec.axis_upright is None
            if not ok and unknown and weak_glyph and analysis.sideways_table:
                # Одиночный знак, не читаемый уверенно ни прямо, ни боком, в таблице, где
                # лежит большинство: «стоит» его назвал перевес в один знак, а вертикальная
                # чёрточка читается как «1» с любой стороны. Скорее всего лежит и он.
                # Отменять ради него поворот всей таблицы (1969/08: 33 боковые ячейки против
                # одной чёрточки) — значит вернуть боковой мусор. Ячейка с уверенно стоячей
                # осью («6П13С», пять компонент) под это не подпадает никогда: она стоит.
                standing_keys.discard(rec.key)
                rec.reason = f"не прочиталась ({why}); считается лежащей"
                continue
            if not ok:
                return None, f"{where} стоит и не прочиталась: {why}", standing_keys
        if rec.ocr_lines > MAX_LINES:
            return None, f"{where} — абзац из {rec.ocr_lines} строк", standing_keys
        spec.append((rec, interior_box(cell_rot, dpi), text))
    return spec, "", standing_keys
