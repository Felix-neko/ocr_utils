"""Стык двух полос при сборке выпуска: словарные эвристики, признаки сомнения и точечная проверка моделью.

Откуда задача. Модель достраивает перенесённое слово в последней строке полосы сама: на скане
«были направле-» | «ны на …» (1976/12, 0270_1L/0270_2R, чисто, без корешка), в транскрипции —
«направлены» | «ны на …»; то же «сокращением» | «нием» в 1966/03. Это правило 5 промпта
(«обрезанное слово прочитай целиком»), распространённое на честный перенос; второй сценарий —
дефис или половина буквы под корешком тугой подшивки. На 195 границах двух выпусков таких случаев
2 (~1 %), сомнительных по словарю границ — единицы на выпуск.

Что здесь:

* :func:`decide_boundary` — решение по паре «хвост, голова» без картинки: перенос по словарю
  (``hyphen_join.join_across_boundary``), дубли достроенного слова (неверная догадка «направлена» |
  «ны», зеркальный «направле-» | «направлены», обе стороны достроены одинаково), оборванное
  предложение через пробел.
* :func:`doubt_reason` — стоит ли показать стык модели: строчный фрагмент вне словаря, несловарный
  хвост, «составное» с неизвестной дефисной формой, любой сработавший дубль (текст изменён — модель
  подтверждает).
* :func:`text_line_bands`, :func:`boundary_strips` — узкие полоски последних/первых строк по проекции
  чернил (без layout-моделей), JPEG через ``tiling.encode``.
* :class:`BoundaryChecker` — **один запрос на выпуск** к той же модели: сборка первым проходом
  собирает сомнительные стыки (:class:`Seam`), они уходят вместе (по две полоски на стык и
  хвост/голова текстом, ``prompts/boundary_*.md.j2``), вердикты ложатся вторым проходом сборки;
  кэш запросов тот же (``cache.RequestCache``, псевдополоса ``_boundaries``), :func:`apply_verdict` —
  как вердикт ложится в текст.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from ocr_utils.external_ocr_services.cache import CacheVerdict, cache_for, entry_dir, request_key
from ocr_utils.external_ocr_services.client import OpenRouterClient, OpenRouterError
from ocr_utils.external_ocr_services.hyphen_join import PAGE_HEAD, PAGE_TAIL, JoinRule, Morph, join_across_boundary
from ocr_utils.external_ocr_services.models import JsonMode, ModelSpec
from ocr_utils.external_ocr_services.ocr import provider_field, reasoning_field
from ocr_utils.external_ocr_services.pages import IMAGE_SUFFIXES
from ocr_utils.external_ocr_services.prompts import BOUNDARY_PROMPT_VERSION, boundary_prompts
from ocr_utils.external_ocr_services.schema import DamageTag
from ocr_utils.external_ocr_services.tiling import DEFAULT_QUALITY, PreparedImage, TileBox, encode

logger = logging.getLogger(__name__)

BOUNDARY_STAGE = "boundary"  # имя этапа в папке кэша: cache/{год}/{выпуск}/_boundaries/boundary.…
BOUNDARY_PAGE = "_boundaries"  # псевдополоса выпуска в кэше — запрос общий на все стыки
BOUNDARY_MAX_TOKENS = 400  # ответ на один стык — короткий JSON из двух строк и двух слов
BATCH_MAX_SEAMS = 10  # стыков в одном запросе; больше — ещё запрос (на выпуск обычно единицы)
STRIP_LINES = 3  # сколько последних/первых строк уходит в полоску
# Если между хвостом и краем полосы стоят плавающие блоки (подпись к рисунку, сноска, таблица),
# нужная строка выше — берётся больше строк, а модель предупреждается (1966/03, IMG_0121_2R: три
# последние строки — подпись к чертежу, и модель читала её вместо текста).
STRIP_LINES_EXTENDED = 12
STRIP_MAX_SIDE = 2200  # длинная сторона полоски после уменьшения — как у тайлов полос
FALLBACK_FRACTION = 0.12  # доля высоты полосы, если строки по проекции не нашлись
SNIPPET_CHARS = 200  # сколько знаков хвоста и головы показывать модели текстом

# Теги повреждений — снимаются для сравнения слов; токен со стыка, где они есть, модели не показываем.
_DAMAGE_TAGS = re.compile(rf"</?(?:{'|'.join(tag.value for tag in DamageTag)})>")
# Конец предложения: точка, вопрос, восклицание, многоточие, курсив/жирный (подпись «*И. Иванов*»).
# Кавычка, скобка, двоеточие и точка с запятой концом не считаются: со строчной головой это середина.
_SENTENCE_END = re.compile(r"[.!?…*]\s*$")
# Точка после сокращения — не конец предложения: «35 тыс.» + «автомашин».
_ABBREVIATION_END = re.compile(
    r"(?:^|[\s(«])(?:тыс|млн|млрд|руб|коп|гг?|т|кг|км|м|см|мм|стр|с|шт|экз|др|пр|напр|проц|ул|им|обл|р-н|п)\.\s*$"
)
# Голова, продолжающая оборванное предложение: строчная буква, цифра или знак продолжения.
_CONTINUATION_START = re.compile(r"^[а-яёa-z0-9—,(;]")
# Последнее слово хвоста (целое, без дефиса) и первое строчное слово головы.
_LAST_WORD = re.compile(r"(?<![\w-])([а-яёА-ЯЁ]{3,})\s*$")
_FIRST_WORD = re.compile(r"^([а-яё]{2,})(?![\w-])")
# Токены на стыке для подстановки вердикта модели: последнее «слово с необязательным дефисом»
# хвоста и первое слово головы; всё, что не буквы, — не трогаем.
_LAST_TOKEN = re.compile(r"([а-яёА-ЯЁ]+-?)\s*$")
_FIRST_TOKEN = re.compile(r"^([а-яёА-ЯЁ]+)")
_WORD_ONLY = re.compile(r"^[а-яёА-ЯЁ]+(?:-[а-яёА-ЯЁ]+)*$")


class JoinKind(StrEnum):
    """Что произошло на границе полос."""

    HYPHEN = "hyphen"  # дефис-перенос убран, слово склеено по словарю
    COMPOUND = "compound"  # дефис на границе — составное слово, оставлен, абзацы сшиты без пробела
    DUPLICATE = "duplicate"  # модель достроила перенесённое слово на одной из полос — дубль убран
    PARAGRAPH = "paragraph"  # оборванное предложение сшито через пробел
    MODEL = "model"  # стык переписан по вердикту модели (проверка полосками строк)
    NONE = "none"  # не сшивался; запись есть только у стыков, показанных модели


class DoubtReason(StrEnum):
    """Почему стык стоит показать модели."""

    FRAGMENT_HEAD = "fragment_head"  # голова начинается со строчного токена вне словаря, склейки нет
    FRAGMENT_TAIL = "fragment_tail"  # хвост кончается несловарным токеном без знака конца перед строчной головой
    COMPOUND_UNKNOWN = "compound_unknown"  # дефис на границе оставлен, но дефисной формы целиком словарь не знает
    DUPLICATE = "duplicate"  # сработала эвристика дубля — текст изменён, модель подтверждает


class VerdictStatus(StrEnum):
    """Что стало с вердиктом модели на стыке (``model.status`` в sidecar)."""

    REWRITTEN = "rewritten"  # стык переписан по вердикту
    CONFIRMED = "confirmed"  # вердикт совпал с эвристикой, текст не менялся
    REJECTED = (
        "rejected"  # вердикт непригоден: слова модели не похожи на слова стыка (читала чужую строку) или не слова
    )
    ERROR = "error"  # запрос или разбор ответа не удался


@dataclass(frozen=True)
class BoundaryDecision:
    """Решение по стыку: сшитый абзац, где в нём начинается голова, вид и слово (для sidecar)."""

    text: str
    head_start: int
    kind: JoinKind
    word: str | None = None


def _strip(text: str) -> str:
    """Текст без тегов повреждений.

    Args:
        text: Кусок абзаца.
    """
    return _DAMAGE_TAGS.sub("", text)


def _duplicate(tail: str, head: str, morph: Morph) -> BoundaryDecision | None:
    """Дубль достроенного слова на стыке — три случая, см. докстринг модуля.

    Args:
        tail: Хвост предыдущей полосы.
        head: Голова следующей.
        morph: Анализатор.

    Returns:
        Решение вида ``DUPLICATE`` или ``None``.
    """
    tail_text, head_text = _strip(tail.rstrip()), _strip(head.lstrip())
    stem = tail.rstrip() + " "
    # Зеркальный случай: хвост «направле-», голова — словарное слово «направлены», начинающееся с
    # половины: половина с дефисом убирается, остаётся слово головы.
    tail_half, head_word = PAGE_TAIL.search(tail.rstrip()), _FIRST_WORD.match(head_text)
    if tail_half and head_word:
        half, word = _strip(tail_half.group("a")), head_word.group(1)
        if word.startswith(half.lower()) and len(word) > len(half) + 1 and morph.known(word):
            prefix = tail.rstrip()[: tail_half.start("a")]
            return BoundaryDecision(prefix + head.lstrip(), len(prefix), JoinKind.DUPLICATE, f"{half}-+{word}")
    last, fragment = _LAST_WORD.search(tail_text), _FIRST_WORD.match(head_text)
    if not (last and fragment):
        return None
    word, half = last.group(1), fragment.group(1)
    if _SENTENCE_END.search(tail_text) and not _ABBREVIATION_END.search(tail_text):
        return None
    rest = head.lstrip()[fragment.end() :].lstrip()
    # Обе стороны достроены одинаково: «направлены» | «направлены …» — одно слово.
    if word.lower() == half and len(half) >= 5 and morph.known(half):
        return BoundaryDecision(stem + rest, len(stem), JoinKind.DUPLICATE, f"{word}={half}")
    if not morph.known(word) or morph.known(half):
        return None
    # Неверная догадка: хвост «направлена», голова «ны» — ищем префикс хвоста, с которым фрагмент
    # даёт словарное слово; «направлены» + «ны» (хвост кончается фрагментом) — частный случай.
    for cut in range(0, len(half) + 3):
        prefix = word[: len(word) - cut] if cut else word
        if len(prefix) < 3:
            break
        candidate = prefix + half
        if morph.known(candidate):
            tail_cut = tail.rstrip()
            base = (
                tail_cut[: _LAST_WORD.search(_strip(tail_cut)).start()] if not _DAMAGE_TAGS.search(tail_cut) else None
            )
            if base is None:  # теги внутри последнего слова — прямую подстановку не делаем
                return None
            new_stem = base + candidate + " "
            return BoundaryDecision(new_stem + rest, len(new_stem), JoinKind.DUPLICATE, f"{word}+{half}→{candidate}")
    return None


def decide_boundary(
    tail: str, head: str, morph: Morph | None, rule: JoinRule = JoinRule.E, join_paragraphs_across: bool = True
) -> BoundaryDecision | None:
    """Сшить хвост и голову, если это одно слово или одно предложение.

    Args:
        tail: Хвост предыдущей полосы (простой абзац).
        head: Голова следующей (простой абзац).
        morph: Анализатор для переносов и дублей; ``None`` — только сшивание абзацев.
        rule: Правило склейки переносов.
        join_paragraphs_across: Сшивать ли оборванные предложения через пробел.

    Returns:
        :class:`BoundaryDecision` или ``None`` — не сшивать.
    """
    # Дубли — раньше переноса: зеркальный случай «направле-» | «направлены» иначе ушёл бы в
    # «составное» (слитная форма «направленаправлены» словарю неизвестна).
    if morph is not None and join_paragraphs_across:
        duplicate = _duplicate(tail, head, morph)
        if duplicate is not None:
            return duplicate
    boundary = join_across_boundary(tail, head, morph, rule) if morph is not None else None
    if boundary is not None:
        kind = JoinKind.HYPHEN if boundary.joined else JoinKind.COMPOUND
        return BoundaryDecision(boundary.text, boundary.head_start, kind, boundary.word)
    if not join_paragraphs_across:
        return None
    tail_text, head_text = _strip(tail.rstrip()), _strip(head.lstrip())
    if not _CONTINUATION_START.match(head_text):
        return None
    if _SENTENCE_END.search(tail_text) and not _ABBREVIATION_END.search(tail_text):
        return None
    stem = tail.rstrip() + " "
    return BoundaryDecision(stem + head.lstrip(), len(stem), JoinKind.PARAGRAPH, None)


def doubt_reason(tail: str, head: str, decision: BoundaryDecision | None, morph: Morph | None) -> DoubtReason | None:
    """Стоит ли показать стык модели.

    Args:
        tail: Хвост предыдущей полосы.
        head: Голова следующей.
        decision: Что решили эвристики (``None`` — абзацы оставлены раздельно).
        morph: Анализатор; ``None`` — сомнений нет (без словаря не с чем сравнивать).

    Returns:
        Причина или ``None``.
    """
    if morph is None:
        return None
    if decision is not None and decision.kind is JoinKind.DUPLICATE:
        return DoubtReason.DUPLICATE
    tail_text, head_text = _strip(tail.rstrip()), _strip(head.lstrip())
    if decision is not None and decision.kind is JoinKind.COMPOUND and decision.word and not morph.known(decision.word):
        return DoubtReason.COMPOUND_UNKNOWN
    if decision is not None and decision.kind is JoinKind.HYPHEN:
        return None
    fragment = _FIRST_WORD.match(head_text)
    if fragment and not morph.known(fragment.group(1)):
        return DoubtReason.FRAGMENT_HEAD
    # Несловарный хвост — только строчный: с прописной это, скорее всего, имя собственное («Дружковский»).
    last = _LAST_WORD.search(tail_text)
    if (
        last
        and fragment
        and last.group(1)[0].islower()
        and not morph.known(last.group(1))
        and not (_SENTENCE_END.search(tail_text) and not _ABBREVIATION_END.search(tail_text))
    ):
        return DoubtReason.FRAGMENT_TAIL
    return None


# --- полоски строк ---------------------------------------------------------------------------------


def text_line_bands(gray: np.ndarray, work_width: int = 1500) -> list[tuple[int, int]]:
    """Строки текста по горизонтальной проекции чернил: ``[(верх, низ), …]`` в пикселях исходника.

    Полоса уменьшается до ``work_width``, бинаризуется по Оцу, по строкам считается доля чернил;
    строка текста — ряд строк с долей выше порога (четверть 95-го перцентиля, не ниже 0.4 %);
    разрывы короче трети медианной высоты сливаются (диакритика, выносные элементы), полосы ниже
    четверти медианы отбрасываются (линейки, шум).

    Args:
        gray: Серая полоса ``uint8`` (H×W).
        work_width: Ширина рабочей копии, px.

    Returns:
        Полосы сверху вниз; пустой список — чернил нет.
    """
    height, width = gray.shape[:2]
    scale = min(1.0, work_width / width)
    small = cv2.resize(
        gray, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )
    threshold, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ink = (small < threshold).mean(axis=1)
    if not ink.any():
        return []
    level = max(0.004, 0.25 * float(np.percentile(ink, 95)))
    on = np.convolve(ink, np.ones(5) / 5, mode="same") > level
    runs: list[list[int]] = []
    for row, flag in enumerate(on):
        if flag and runs and runs[-1][1] == row:
            runs[-1][1] = row + 1
        elif flag:
            runs.append([row, row + 1])
    if not runs:
        return []
    median = float(np.median([bottom - top for top, bottom in runs]))
    merged: list[list[int]] = [runs[0]]
    for top, bottom in runs[1:]:
        if top - merged[-1][1] < 0.34 * median:
            merged[-1][1] = bottom
        else:
            merged.append([top, bottom])
    median = float(np.median([bottom - top for top, bottom in merged]))
    bands = [(top, bottom) for top, bottom in merged if bottom - top >= 0.25 * median]
    return [(int(top / scale), int(bottom / scale)) for top, bottom in bands]


@dataclass(frozen=True)
class Strip:
    """Полоска строк одной полосы: картинка для запроса и откуда она вырезана."""

    image: PreparedImage
    box: tuple[int, int, int, int]  # left, top, right, bottom в пикселях исходника
    lines: int  # сколько строк по проекции попало (0 — запасной вырез по доле высоты)


def _crop_box(
    gray: np.ndarray, bands: list[tuple[int, int]], at_bottom: bool, lines: int = STRIP_LINES
) -> tuple[tuple[int, int, int, int], int]:
    """Прямоугольник полоски: последние (или первые) ``STRIP_LINES`` строк с запасом в полстроки и
    обрезкой по чернилам по ширине; без строк — доля высоты с края.

    Args:
        gray: Серая полоса.
        bands: Строки по :func:`text_line_bands`.
        at_bottom: Хвост (низ полосы) или голова (верх).
        lines: Сколько строк брать.

    Returns:
        ``((left, top, right, bottom), число строк)``.
    """
    height, width = gray.shape[:2]
    chosen = bands[-lines:] if at_bottom else bands[:lines]
    if not chosen:
        band = int(height * FALLBACK_FRACTION)
        box = (0, height - band, width, height) if at_bottom else (0, 0, width, band)
        return box, 0
    pitch = int(np.median([bottom - top for top, bottom in chosen]))
    top = max(0, chosen[0][0] - pitch // 2)
    bottom = min(height, chosen[-1][1] + pitch // 2)
    # По ширине — от первого до последнего столбца с чернилами в выбранных строках, плюс 2 % ширины.
    part = gray[top:bottom]
    threshold, _ = cv2.threshold(part, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    columns = np.where((part < threshold).mean(axis=0) > 0.002)[0]
    margin = int(width * 0.02)
    left = max(0, int(columns[0]) - margin) if columns.size else 0
    right = min(width, int(columns[-1]) + margin) if columns.size else width
    return (left, top, right, bottom), len(chosen)


def strip_of(path: Path, at_bottom: bool, quality: int = DEFAULT_QUALITY, lines: int = STRIP_LINES) -> Strip:
    """Полоска последних (``at_bottom``) или первых строк полосы.

    Args:
        path: Файл полосы.
        at_bottom: Низ (хвост) или верх (голова).
        quality: Качество JPEG.
        lines: Сколько строк брать (``STRIP_LINES_EXTENDED`` — когда у края плавающие блоки).
    """
    with Image.open(path) as image:
        gray_image = ImageOps.grayscale(image)
    gray = np.asarray(gray_image)
    (left, top, right, bottom), lines = _crop_box(gray, text_line_bands(gray), at_bottom, lines)
    crop = gray_image.crop((left, top, right, bottom))
    prepared = encode(crop, TileBox(0, 0, left, top, right, bottom), STRIP_MAX_SIDE, quality)
    return Strip(prepared, (left, top, right, bottom), lines)


def source_image(in_dir: Path, rel: Path) -> Path | None:
    """Файл полосы на входе по её относительному пути с любым суффиксом (``.json`` из out-dir тоже годится).

    Args:
        in_dir: Корень входа.
        rel: Путь полосы относительно корня.
    """
    base = in_dir / rel.with_suffix("")
    for suffix in IMAGE_SUFFIXES:
        candidate = base.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


# --- запрос к модели --------------------------------------------------------------------------------


@dataclass
class BoundaryVerdict:
    """Ответ модели по стыку и учёт запроса."""

    tail_word: str = ""
    head_word: str = ""
    same_word: bool = False
    joined: str = ""
    last_line: str = ""
    first_line: str = ""
    notes: str = ""
    cache_entry: str = ""
    cache_hit: bool = False
    cost_usd: float = 0.0
    error: str | None = None  # запрос или разбор не удался — стык остаётся по эвристике

    def as_dict(self) -> dict:
        """Для sidecar."""
        return {key: value for key, value in vars(self).items() if value not in ("", None, False, 0.0)}


@dataclass
class CheckerStats:
    """Счётчики проверки стыков за выпуск/прогон."""

    requests: int = 0  # стыков, показанных модели (с попаданиями в кэш)
    cache_hits: int = 0
    cost_usd: float = 0.0
    errors: int = 0

    def __add__(self, other: CheckerStats) -> CheckerStats:
        """Сумма счётчиков (выпуски → прогон)."""
        return CheckerStats(
            self.requests + other.requests,
            self.cache_hits + other.cache_hits,
            self.cost_usd + other.cost_usd,
            self.errors + other.errors,
        )

    def __sub__(self, other: CheckerStats) -> CheckerStats:
        """Разность: счётчики проверки с момента снимка ``other`` (один выпуск из общих счётчиков checker).

        Args:
            other: Снимок счётчиков до выпуска.
        """
        return CheckerStats(
            self.requests - other.requests,
            self.cache_hits - other.cache_hits,
            self.cost_usd - other.cost_usd,
            self.errors - other.errors,
        )


@dataclass(frozen=True)
class Seam:
    """Сомнительный стык, собранный первым проходом сборки, — для одного запроса на выпуск."""

    index: int  # порядковый номер стыка в выпуске (ключ вердикта)
    rel_a: Path  # полоса перед границей (относительно корня, суффикс любой)
    rel_b: Path  # полоса после границы
    tail: str  # хвост (абзац целиком)
    head: str  # голова
    reason: DoubtReason
    tail_floating: bool = False  # после хвоста до края полосы A есть плавающие блоки (подпись, сноска, таблица)
    head_floating: bool = False  # перед головой от края полосы B есть плавающие блоки


def parse_verdicts(text: str) -> dict[int, BoundaryVerdict]:
    """JSON ответа → вердикты по номерам стыков; терпимо к ограждению ```json и тексту вокруг объекта.

    Args:
        text: Сырой текст ответа.

    Returns:
        ``{номер стыка (с 1): вердикт}``; запись без номера получает номер по порядку.

    Raises:
        ValueError: Объекта JSON или списка ``seams`` нет.
    """
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-z]*\s*|\s*```$", "", body)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise ValueError("в ответе нет объекта JSON")
    data = json.loads(body[start : end + 1])
    items = data.get("seams") if isinstance(data, dict) else None
    if items is None and isinstance(data, dict) and "tail_word" in data:
        items = [data]  # модель ответила одним объектом на единственный стык
    if not isinstance(items, list):
        raise ValueError("в ответе нет списка seams")
    verdicts: dict[int, BoundaryVerdict] = {}
    for position, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        verdict = BoundaryVerdict()
        for key in ("tail_word", "head_word", "joined", "last_line", "first_line", "notes"):
            value = item.get(key, "")
            setattr(verdict, key, str(value).strip() if value is not None else "")
        verdict.same_word = bool(item.get("same_word", False))
        number = item.get("seam", position)
        try:
            number = int(number)
        except (TypeError, ValueError):
            number = position
        verdicts[number] = verdict
    return verdicts


class BoundaryChecker:
    """Проверка сомнительных стыков выпуска той же моделью — один запрос на выпуск (до
    ``BATCH_MAX_SEAMS`` стыков в запросе, дальше — ещё запрос): по две полоски строк на стык плюс
    хвост/голова текстом.

    Args:
        client: Клиент OpenRouter.
        spec: Модель из реестра (та же, что распознаёт полосы).
        in_dir: Корень входа — откуда резать полоски.
        cache_dir: Кэш запросов (``--cache-dir``); ``None`` — без кэша.
        quality: Качество JPEG полосок.
        max_tokens: Потолок ответа на один стык (умножается на число стыков в запросе).
    """

    def __init__(
        self,
        client: OpenRouterClient,
        spec: ModelSpec,
        in_dir: Path,
        cache_dir: Path | None,
        quality: int = DEFAULT_QUALITY,
        max_tokens: int = BOUNDARY_MAX_TOKENS,
    ):
        self.client = client
        self.spec = spec
        self.in_dir = Path(in_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.quality = quality
        self.max_tokens = max_tokens
        self.stats = CheckerStats()

    def payload(self, seams: list[Seam], strips: list[tuple[Strip, Strip]]) -> dict[str, Any]:
        """Тело запроса: системный промпт, текст со стыками и по две полоски на стык.

        Args:
            seams: Стыки запроса по порядку.
            strips: Полоски (хвост, голова) для каждого стыка в том же порядке.
        """
        described = [
            {
                "number": position,
                "page_before": seam.rel_a.with_suffix("").name,
                "page_after": seam.rel_b.with_suffix("").name,
                "tail": " ".join(_strip(seam.tail).split())[-SNIPPET_CHARS:],
                "head": " ".join(_strip(seam.head).split())[:SNIPPET_CHARS],
                "reason": seam.reason.value.replace("_", " "),
                "tail_floating": seam.tail_floating,
                "head_floating": seam.head_floating,
            }
            for position, seam in enumerate(seams, 1)
        ]
        system, user = boundary_prompts(described)
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for tail_strip, head_strip in strips:
            for strip in (tail_strip, head_strip):
                part: dict[str, Any] = {"url": strip.image.data_url()}
                if self.spec.image_detail:
                    part["detail"] = self.spec.image_detail
                content.append({"type": "image_url", "image_url": part})
        payload: dict[str, Any] = {
            "model": self.spec.openrouter_id,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": self.max_tokens * len(seams),
            "response_format": {"type": JsonMode.JSON_OBJECT.value},
        }
        # Рассуждения выключаются так же, как у полос: без этого DeepSeek думает в счёт потолка
        # ответа (замер 1976/12: 800 токенов ушло на рассуждения, JSON оборван на 276 знаках).
        reasoning = reasoning_field(self.spec, None)
        if reasoning is not None:
            payload["reasoning"] = reasoning
        provider = provider_field(self.spec)
        if provider:
            payload["provider"] = provider
        return payload

    def check_issue(self, issue_key: str, seams: list[Seam]) -> dict[int, BoundaryVerdict]:
        """Показать модели все сомнительные стыки выпуска.

        Args:
            issue_key: «год/выпуск» — папка записи в кэше (``cache/{год}/{выпуск}/_boundaries/…``).
            seams: Стыки первого прохода сборки.

        Returns:
            ``{seam.index: вердикт}``; стыки без файлов полос во входе пропущены (предупреждение в лог),
            стыки из сбойного запроса получают вердикт с ``error``.
        """
        verdicts: dict[int, BoundaryVerdict] = {}
        usable: list[tuple[Seam, tuple[Strip, Strip]]] = []
        for seam in seams:
            path_a, path_b = source_image(self.in_dir, seam.rel_a), source_image(self.in_dir, seam.rel_b)
            if path_a is None or path_b is None:
                logger.warning("стык %s → %s: нет файла полосы во входе, проверка пропущена", seam.rel_a, seam.rel_b)
                continue
            tail_lines = STRIP_LINES_EXTENDED if seam.tail_floating else STRIP_LINES
            head_lines = STRIP_LINES_EXTENDED if seam.head_floating else STRIP_LINES
            usable.append(
                (
                    seam,
                    (
                        strip_of(path_a, True, self.quality, tail_lines),
                        strip_of(path_b, False, self.quality, head_lines),
                    ),
                )
            )
        for start in range(0, len(usable), BATCH_MAX_SEAMS):
            batch = usable[start : start + BATCH_MAX_SEAMS]
            verdicts.update(self._request(issue_key, [seam for seam, _ in batch], [strips for _, strips in batch]))
        return verdicts

    def _request(
        self, issue_key: str, seams: list[Seam], strips: list[tuple[Strip, Strip]]
    ) -> dict[int, BoundaryVerdict]:
        """Один запрос (из кэша или в сеть) на пачку стыков.

        Args:
            issue_key: «год/выпуск».
            seams: Стыки пачки.
            strips: Их полоски.

        Returns:
            ``{seam.index: вердикт}`` для всех стыков пачки.
        """
        payload = self.payload(seams, strips)
        self.stats.requests += 1
        cache = cache_for(self.cache_dir) if self.cache_dir is not None else None
        entry = None
        cache_entry = ""
        key = request_key(payload)
        if cache is not None:
            entry = entry_dir(
                self.cache_dir, Path(issue_key) / BOUNDARY_PAGE, BOUNDARY_STAGE, JsonMode.JSON_OBJECT.value, key
            )
            cache_entry = entry.relative_to(self.cache_dir).as_posix()
            cached = cache.lookup(entry)
            if cached is not None:
                self.stats.cache_hits += 1
                return self._finish(seams, cached.response.text, cached.response.cost_usd, cache_entry, True)
        try:
            response = self.client.chat(payload)
        except OpenRouterError as error:
            if cache is not None and entry is not None:
                cache.record_error(entry, str(error), error.body)
            self.stats.errors += 1
            return {seam.index: BoundaryVerdict(cache_entry=cache_entry, error=str(error)) for seam in seams}
        if cache is not None and entry is not None and response.text.strip():
            info = {
                "issue": issue_key,
                "stage": BOUNDARY_STAGE,
                "seams": [
                    {
                        "number": position,
                        "page_before": seam.rel_a.with_suffix("").as_posix(),
                        "page_after": seam.rel_b.with_suffix("").as_posix(),
                        "reason": seam.reason.value,
                        "tail_strip": {"box": pair[0].box, "lines": pair[0].lines},
                        "head_strip": {"box": pair[1].box, "lines": pair[1].lines},
                    }
                    for position, (seam, pair) in enumerate(zip(seams, strips), 1)
                ],
                "boundary_prompt_version": BOUNDARY_PROMPT_VERSION,
                "model": self.spec.name,
                "openrouter_id": self.spec.openrouter_id,
                "key": key,
            }
            cache.store(entry, info, payload, response, CacheVerdict.OK)
        self.stats.cost_usd += response.cost_usd or 0.0
        return self._finish(seams, response.text, response.cost_usd, cache_entry, False)

    def _finish(
        self, seams: list[Seam], text: str, cost: float | None, cache_entry: str, cache_hit: bool
    ) -> dict[int, BoundaryVerdict]:
        """Разобрать ответ пачки; битый ответ — ``error`` у всех стыков пачки, пропущенный стык — свой ``error``.

        Args:
            seams: Стыки пачки по порядку (номера в ответе — с 1 в этом порядке).
            text: Сырой ответ.
            cost: Цена ответа (делится между стыками поровну для sidecar).
            cache_entry: Папка записи кэша.
            cache_hit: Ответ взят из кэша.
        """
        share = (cost or 0.0) / max(1, len(seams))
        try:
            parsed = parse_verdicts(text)
        except (ValueError, TypeError) as error:
            self.stats.errors += 1
            return {
                seam.index: BoundaryVerdict(
                    cache_entry=cache_entry, cache_hit=cache_hit, error=f"ответ не разобран: {error}"
                )
                for seam in seams
            }
        result: dict[int, BoundaryVerdict] = {}
        for position, seam in enumerate(seams, 1):
            verdict = parsed.get(position)
            if verdict is None:
                verdict = BoundaryVerdict(error="стыка нет в ответе")
            verdict.cache_entry, verdict.cache_hit, verdict.cost_usd = cache_entry, cache_hit, share
            result[seam.index] = verdict
        return result


def _resembles(model_word: str, seam_word: str) -> bool:
    """Похоже ли слово модели на слово транскрипции: одно — начало другого (без дефиса), или общее
    начало от четырёх букв; так отсекается вердикт по чужой строке («кабеля» против «сокращением»).

    Args:
        model_word: Слово из ответа модели (может кончаться дефисом).
        seam_word: Токен транскрипции на стыке.
    """
    a, b = model_word.rstrip("-").lower(), seam_word.rstrip("-").lower()
    if not a or not b:
        return False
    if a.startswith(b) or b.startswith(a):
        return True
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    # Короткая половина с опечаткой транскрипции («ритым» против «ритным», 1968/12) — общее начало
    # короче четырёх букв, но слова почти совпадают: считаем по сходству строк.
    return common >= 4 or SequenceMatcher(None, a, b).ratio() >= 0.8


def apply_verdict(
    tail: str, head: str, verdict: BoundaryVerdict, decision: BoundaryDecision | None
) -> tuple[BoundaryDecision | None, VerdictStatus]:
    """Положить вердикт модели на стык.

    Args:
        tail: Хвост.
        head: Голова.
        verdict: Ответ модели.
        decision: Решение эвристик (может быть ``None``).

    Returns:
        ``(решение, статус)``: новое решение вида ``MODEL`` и ``REWRITTEN``, если модель переписала
        стык; иначе решение эвристик как есть и ``CONFIRMED`` (вердикт совпал), ``REJECTED`` (вердикт
        непригоден) или ``ERROR``.
    """
    if verdict.error:
        return decision, VerdictStatus.ERROR
    tail_text, head_text = tail.rstrip(), head.lstrip()
    last, first = _LAST_TOKEN.search(tail_text), _FIRST_TOKEN.match(head_text)
    if not (last and first) or _DAMAGE_TAGS.search(last.group(0)) or _DAMAGE_TAGS.search(first.group(0)):
        return decision, VerdictStatus.REJECTED
    tail_word, head_word, joined = (
        verdict.tail_word.strip('«»" '),
        verdict.head_word.strip('«»" '),
        verdict.joined.strip('«»" '),
    )
    # Модель могла прочитать не ту строку (подпись к рисунку вместо текста над ним): её слова
    # должны быть похожи на слова транскрипции на стыке, иначе вердикт не применяется.
    if not _resembles(tail_word, last.group(1)) or not _resembles(head_word, first.group(1)):
        return decision, VerdictStatus.REJECTED
    if verdict.same_word or (tail_word.endswith("-") and joined):
        if not joined or not _WORD_ONLY.match(joined):
            return decision, VerdictStatus.REJECTED
        # Одно слово: последний токен хвоста и первый токен головы заменяются на слово модели.
        prefix = tail_text[: last.start()]
        rest = head_text[first.end() :]
        text = prefix + joined + rest
        # Смещение головы — где в слове начинается её часть, если она видна; иначе конец слова.
        inner = (
            len(joined) - len(head_word) if head_word and joined.lower().endswith(head_word.lower()) else len(joined)
        )
        current = decision.text if decision is not None else None
        if current == text:
            return decision, VerdictStatus.CONFIRMED
        word = f"{tail_word}|{head_word}→{joined}"
        return BoundaryDecision(text, len(prefix) + inner, JoinKind.MODEL, word), VerdictStatus.REWRITTEN
    # Не одно слово: модель могла прочесть слова на стыке иначе, чем транскрипция; подставляем их
    # и сшиваем через пробел, если голова строчная.
    if not (_WORD_ONLY.match(tail_word) and _WORD_ONLY.match(head_word)):
        return decision, VerdictStatus.REJECTED
    new_tail = tail_text[: last.start()] + tail_word if tail_word.lower() != last.group(1).lower() else tail_text
    new_head = head_word + head_text[first.end() :] if head_word.lower() != first.group(1).lower() else head_text
    if new_tail == tail_text and new_head == head_text:
        # Слова те же; «не одно слово» согласуется с эвристикой, если та не склеивала слово.
        agreed = decision is None or decision.kind in (JoinKind.PARAGRAPH, JoinKind.NONE)
        return decision, VerdictStatus.CONFIRMED if agreed else VerdictStatus.REJECTED
    stem = new_tail + " "
    if not _CONTINUATION_START.match(_strip(new_head)):
        return decision, VerdictStatus.REJECTED
    return (
        BoundaryDecision(stem + new_head, len(stem), JoinKind.MODEL, f"{tail_word}|{head_word}"),
        VerdictStatus.REWRITTEN,
    )
