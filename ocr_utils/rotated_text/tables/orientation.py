"""Повёрнут ли текст в ячейке и в какую сторону.

ДВА ВОПРОСА, ДВА ОТВЕТЧИКА. ОСЬ (текст стоит или лежит) даёт форма букв: кириллическая
буква выше, чем шире, повёрнутая — наоборот. Медианное отношение ширины компоненты к
высоте у боковых ячеек 1.07–1.53, у прямых 0.58–0.94; порог 1.0 стоит в пустом промежутке
с запасом 0.13 в обе стороны (замер на 132 размеченных ячейках, 36/37 боковых и 72/74
прямых, 1 мс на ячейку). Форму букв не сбивает колонка чисел, на которой ошибалось
смыкание глифов. СТОРОНУ (90 или 270, 0 или 180) форма не различает принципиально —
её называет tesseract: читаем под каждым углом-кандидатом, побеждает угол, под которым
прочиталось больше букв (36/36 на тех же ячейках).

ВЕТО. «Букв не нашлось ни под каким углом» — это утверждение, что читать нечего, а не
воздержавшийся голос: колонки чисел и пустые ячейки форма букв иногда уверенно называет
боковыми. На паке вето сняло 393 ложные ячейки из 1113.

ПРИОР ПО ТАБЛИЦЕ. Все боковые ячейки одной таблицы набраны в одну сторону — так вёрстка
и делалась. Ячейке, где tesseract отличил 90 от 270 неуверенно, сторона достаётся от
большинства.

180°. Ось у перевёрнутого текста та же, что у прямого, поэтому вопрос «0 или 180»
решается только чтением и, в отличие от бокового, без поддержки со стороны формы букв.
Отсюда ЗАПАС: ответ «180» принимается, только если букв под ним в полтора раза больше,
чем под нулём, — иначе шумное чтение мелкой ячейки перевернуло бы прямой текст.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from ocr_utils.scan_markup.rotation import rotate_cw
from ocr_utils.page_layout.geometry import Cell, Grid
from ocr_utils.scan_markup.table_detection.verify import glyph_mask

from ocr_utils.rotated_text.tables.ocr import LANGUAGES, prepare, tesseract_tsv
from ocr_utils.rotated_text.tables.structure import MIN_GLYPH_MM2, cell_interior

# Меньше этого числа компонент — медиана считается по шуму («93» — две цифры).
MIN_COMPONENTS = 3

# Граница «выше, чем шире» / «шире, чем выше» и отступ, при котором уверенность полная.
ASPECT_THRESHOLD = 1.0
ASPECT_FULL_MARGIN = 0.13

# Слово идёт в счёт букв, если tesseract уверен в нём не меньше этого: на неверном
# повороте он выдаёт россыпь мусора, и без порога повороты почти не различаются.
MIN_WORD_CONFIDENCE = 60.0

# Меньше букв на лучшем угле — читать нечего, ячейку не трогаем.
MIN_LETTERS = 3

# Во сколько раз ответ без поддержки формы букв (180 против 0, боковой при неясной оси)
# обязан обыграть ноль.
SIDE_MARGIN = 1.5

# Ответ «180» требует букв больше обычного: перевёрнутые цифры tesseract читает как буквы
# («101» → «ТОТ», «150» → «ОСТ»). Замер по паку: у 115 ячеек, названных перевёрнутыми,
# ложные дают 3–4 «буквы», настоящие («Итого», «Прочие», «кг/т. г. п.») — от 5.
MIN_LETTERS_180 = 5

# Ниже этой уверенности в стороне (90 против 270) ячейка берёт сторону от большинства.
PRIOR_BELOW = 0.3

# Доля букв боком от букв прямо, начиная с которой прямая ячейка считается смешанной.
MIXED_SHARE = 0.3

# Обратный случай — боковая ячейка с прямыми буквами: порог выше, потому что боковой
# текст, прочитанный прямо, даёт шум в несколько «букв» (у «фактическая» — 4 при 11
# боком), а настоящая смесь — заметную долю (склеенная «стоимость»: 17 при 44 боком).
MIXED_UPRIGHT_MIN = 6
MIXED_UPRIGHT_SHARE = 0.35

# Ячейка, назвавшая сторону против большинства таблицы, отстаивает её только с таким
# числом букв: «хеоен:» (5 букв под 270 среди 28 ячеек под 90 на 1971/03) — это шум чтения
# куска приписки, а не заголовок, набранный в другую сторону.
STRONG_LETTERS = 8

LETTER = re.compile(r"[а-яёА-ЯЁa-zA-Z]")
ALNUM = re.compile(r"[а-яёА-ЯЁa-zA-Z0-9]")


@dataclass
class CellOrientation:
    """Вердикт по ячейке. ``rotate_cw`` — на сколько повернуть по часовой, чтобы текст стал
    прямым; None — текста нет или судить не по чему, ячейку не трогаем."""

    rotate_cw: "int | None"
    aspect: float = 0.0
    components: int = 0
    letters: dict[int, int] = field(default_factory=dict)
    confidence: float = 0.0
    note: str = ""
    # Ось по форме букв, если она видна уверенно: True — лежит, False — стоит, None — неясно.
    axis_sideways: "bool | None" = None
    # Сторона взята от большинства таблицы, а не прочитана в самой ячейке.
    from_prior: bool = False

    @property
    def sideways(self) -> bool:
        return self.rotate_cw in (90, 270)


def median_aspect(mask: np.ndarray, dpi: int) -> tuple[float, int]:
    """Медианное отношение ширины компоненты к высоте и сколько компонент учтено."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0.0, 0
    width = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    height = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    keep = (width * height) >= MIN_GLYPH_MM2 * (dpi / 25.4) ** 2
    if int(keep.sum()) < MIN_COMPONENTS:
        return 0.0, int(keep.sum())
    return float(np.median(width[keep] / np.maximum(height[keep], 1.0))), int(keep.sum())


def _count_at(
    interior: np.ndarray, angle: int, pattern: "re.Pattern[str]", lang: str, min_confidence: float = MIN_WORD_CONFIDENCE
) -> int:
    total = 0
    for row in tesseract_tsv(prepare(rotate_cw(interior, angle)), lang):
        columns = row.split("\t")
        if len(columns) < 12:
            continue
        try:
            confidence = float(columns[10])
        except ValueError:
            continue
        if confidence >= min_confidence:
            total += len(pattern.findall(columns[11]))
    return total


def letters_at(interior: np.ndarray, angle: int, lang: str = LANGUAGES) -> int:
    """Сколько букв прочиталось, если повернуть вырезку на ``angle`` по часовой."""
    return _count_at(interior, angle, LETTER, lang)


# Порог уверенности для сравнения «стоит или лежит» по знакам: выше обычного, потому что
# сравниваются два чтения ОДНОЙ вырезки в один-два знака, и мусор с уверенностью 60–80
# на неверном повороте («5» боком → два случайных знака) перевешивал бы верный.
ALNUM_MIN_CONFIDENCE = 80.0


def alnum_at(interior: np.ndarray, angle: int, lang: str = LANGUAGES) -> int:
    """Сколько букв и цифр прочиталось под углом ``angle`` — для ячеек, где букв может не
    быть вовсе («0», «13», «6П13С»): цифры сторону не называют, но стоит ли знак или лежит,
    по ним видно."""
    return _count_at(interior, angle, ALNUM, lang, ALNUM_MIN_CONFIDENCE)


# Свидетельство, что под углом читается ТЕКСТ, а не россыпь знаков: лучшее слово. Пороги
# подобраны на шести ячейках пака, где счёт знаков ошибался в обе стороны: «6113С»
# (5 знаков, уверенность 60) и «ШТ.» (2, 93) стоят; «32 … Ва» (по 2, 53–57), «гв» (2, 65),
# «сл» (2, 76), «О»/«р» (по 1, 61–63) — мусор чтения чёрточек и обрезанных букв.
WORD_LONG, WORD_LONG_CONFIDENCE = 3, 50.0
WORD_SHORT, WORD_SHORT_CONFIDENCE = 2, 80.0
GLYPH_CONFIDENCE = 90.0


def evidence_at(interior: np.ndarray, angle: int, single_glyph: bool, lang: str = LANGUAGES) -> int:
    """Длина лучшего слова под углом ``angle``; 0 — текста не читается. ``single_glyph`` —
    ячейка в одну-две компоненты: в ней и один знак («5», «0») с уверенностью от 90 —
    свидетельство, в ячейке крупнее одиночный знак — шум."""
    best = 0
    for row in tesseract_tsv(prepare(rotate_cw(interior, angle)), lang):
        columns = row.split("\t")
        if len(columns) < 12:
            continue
        try:
            confidence = float(columns[10])
        except ValueError:
            continue
        length = len(ALNUM.findall(columns[11]))
        if (
            (length >= WORD_LONG and confidence >= WORD_LONG_CONFIDENCE)
            or (length >= WORD_SHORT and confidence >= WORD_SHORT_CONFIDENCE)
            or (single_glyph and length >= 1 and confidence >= GLYPH_CONFIDENCE)
        ):
            best = max(best, length)
    return best


def orient_cell(interior: np.ndarray, dpi: int, allowed: tuple[int, ...], lang: str = LANGUAGES) -> CellOrientation:
    aspect, components = median_aspect(glyph_mask(interior, dpi), dpi)
    if components < MIN_COMPONENTS:
        return CellOrientation(None, aspect, components, note=f"компонент {components}, судить не по чему")

    margin = (aspect - ASPECT_THRESHOLD) / ASPECT_FULL_MARGIN
    axis_sideways = aspect > ASPECT_THRESHOLD
    axis: "bool | None" = axis_sideways if abs(margin) >= 1.0 else None
    if axis is not None:
        candidates = {90, 270} if axis_sideways else {0, 180}
    else:
        candidates = {0, 90, 180, 270}
    candidates &= set(allowed)
    if not candidates:
        candidates = {0}
    if candidates == {0}:
        return CellOrientation(
            0, aspect, components, confidence=min(1.0, abs(margin)), note="ось прямая", axis_sideways=axis
        )

    scores = {angle: letters_at(interior, angle, lang) for angle in sorted(candidates)}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0
    if best_score < MIN_LETTERS:
        return CellOrientation(
            None, aspect, components, scores, note="букв не нашлось ни под каким углом", axis_sideways=axis
        )

    confidence = (best_score - second_score) / best_score
    if best == 180 and best_score < MIN_LETTERS_180:
        return CellOrientation(
            0,
            aspect,
            components,
            scores,
            confidence,
            note=f"180 при {best_score} буквах не принято",
            axis_sideways=axis,
        )
    if best != 0 and 0 in scores and best_score < SIDE_MARGIN * max(scores[0], MIN_LETTERS - 1):
        # Ответ без поддержки формы букв обязан выигрывать у нуля с запасом.
        return CellOrientation(
            0, aspect, components, scores, confidence, note=f"{best} не обыграл 0 с запасом", axis_sideways=axis
        )
    # Смешанная ячейка: букв заметно и прямо, и боком — сетка склеила боковую шапку с
    # соседней прямой (линейка между ними не нашлась). Какая бы сторона ни победила,
    # подменять такую ячейку нельзя: закраска стёрла бы и прямой текст (1966/06 IMG_0137:
    # «стоимость» пропала вместе с боковыми графами под ней). Отмечаем и не трогаем.
    sideways = max(scores.get(90, 0), scores.get(270, 0))
    upright = scores.get(0, 0)
    if best == 0 and sideways >= max(MIN_LETTERS, MIXED_SHARE * best_score):
        note = f"смешанная ячейка: букв прямо {best_score}, боком {sideways}"
        return CellOrientation(0, aspect, components, scores, confidence, note=note, axis_sideways=axis)
    if best in (90, 270) and upright >= max(MIXED_UPRIGHT_MIN, MIXED_UPRIGHT_SHARE * best_score):
        note = f"смешанная ячейка: букв боком {best_score}, прямо {upright}; не трогаем"
        return CellOrientation(None, aspect, components, scores, confidence, note=note, axis_sideways=axis)
    return CellOrientation(best, aspect, components, scores, confidence, axis_sideways=axis)


def orient_table(
    work: np.ndarray, grid: Grid, dpi: int, allowed: tuple[int, ...], lang: str = LANGUAGES
) -> dict[tuple[int, int], CellOrientation]:
    """Вердикты по всем ячейкам с общим для таблицы приором стороны."""
    verdicts = {cell.key: orient_cell(cell_interior(work, cell, dpi), dpi, allowed, lang) for cell in grid.cells}
    apply_prior(verdicts)
    return verdicts


def apply_prior(verdicts: dict[tuple[int, int], CellOrientation]) -> "int | None":
    """Сторона большинства боковых ячеек (взвешенно по буквам) — неуверенным ячейкам и
    ячейкам, где ось лежит уверенно, а букв нет (боковые числа: их сторона та же)."""
    weight = {90: 0, 270: 0}
    for verdict in verdicts.values():
        if verdict.sideways and verdict.confidence >= PRIOR_BELOW:
            weight[verdict.rotate_cw] += max(1, verdict.letters.get(verdict.rotate_cw, 1))
    if not any(weight.values()):
        return None
    prior = max(weight, key=weight.get)
    for verdict in verdicts.values():
        if verdict.sideways and verdict.rotate_cw != prior:
            weak = verdict.confidence < PRIOR_BELOW or verdict.letters.get(verdict.rotate_cw, 0) < STRONG_LETTERS
            if weak:
                verdict.note = f"сторона по большинству таблицы вместо {verdict.rotate_cw}"
                verdict.rotate_cw = prior
        elif verdict.rotate_cw is None and verdict.axis_sideways:
            # Ненадёжно: широкие заглавные («ШТ.») тоже «лежат» по форме. Помечается, чтобы
            # перед поворотом таблицы такую ячейку перепроверили чтением.
            verdict.note = "лежит, букв нет; сторона по большинству таблицы"
            verdict.rotate_cw = prior
            verdict.from_prior = True
    return prior


# Доля лежащих ячеек, начиная с которой таблица считается напечатанной боком целиком.
SIDEWAYS_TABLE_SHARE = 0.5
SIDEWAYS_TABLE_MIN = 3


def sideways_share(verdicts: dict[tuple[int, int], CellOrientation]) -> tuple[float, int]:
    """Доля ячеек с лежащей осью среди ячеек, где ось видна, и их число."""
    lying = sum(1 for v in verdicts.values() if v.axis_sideways is True or v.sideways)
    standing = sum(1 for v in verdicts.values() if v.axis_sideways is False and not v.sideways)
    total = lying + standing
    return (lying / total if total else 0.0), lying


def is_sideways_table(verdicts: dict[tuple[int, int], CellOrientation]) -> bool:
    share, lying = sideways_share(verdicts)
    return lying >= SIDEWAYS_TABLE_MIN and share >= SIDEWAYS_TABLE_SHARE


DIGIT = re.compile(r"[0-9]")


def looks_like_text(text: str) -> bool:
    """Похоже ли прочитанное на текст, а не на шум: есть хотя бы одна буква или цифра, и
    буквы с цифрами составляют не меньше половины знаков. Длина не ограничена: в ячейке
    бывает и «0», и «13». Отсев мусора вроде «я у у Я 1 т ко Е у я» — по уверенности и
    по наличию слова, см. ``pipeline.replaceable``."""
    body = re.sub(r"\s", "", text)
    if not body:
        return False
    alnum = len(LETTER.findall(text)) + len(DIGIT.findall(text))
    return alnum >= 1 and alnum >= 0.5 * len(body)


def has_word(text: str, min_letters: int = 3) -> bool:
    """Есть ли в тексте слово хотя бы из ``min_letters`` букв подряд."""
    return any(len(LETTER.findall(token)) >= min_letters for token in text.split())


def upright_interior(work: np.ndarray, cell: Cell, dpi: int, rotate: int) -> np.ndarray:
    """Внутренность ячейки, повёрнутая так, чтобы текст читался."""
    return rotate_cw(cell_interior(work, cell, dpi), rotate)
