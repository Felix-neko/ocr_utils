"""Одна полоса → один запрос к модели → PageResult, файлы выхода и .meta.json.

Здесь собирается тело запроса под особенности модели (режим JSON, рассуждения, порядок
провайдеров) и этап (``toc`` или ``page`` со списками выпуска), и здесь же запасные ходы:
если провайдер не принял строгую схему, тот же запрос уходит в режиме json_object, потом
вообще без response_format; отвергнутый параметр ``reasoning`` убирается и запрос повторяется.

Готовность полосы определяется по .meta.json и отпечатку списков (``toc_hash``): полоса,
распознанная с другим списком статей, считается устаревшей и идёт заново. Ниже этого уровня —
кэш запросов (``cache.py``, ``RunOptions.cache_dir``): перед каждым обращением к модели ищется
ответ на такой же payload, и попадание обходится без сети.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from ocr_utils.external_ocr_services import PROMPT_VERSION
from ocr_utils.external_ocr_services.cache import CacheVerdict, cache_for, entry_dir, request_key
from ocr_utils.external_ocr_services.client import ChatResponse, OpenRouterClient, OpenRouterError
from ocr_utils.external_ocr_services.models import JsonMode, ModelSpec, Reasoning
from ocr_utils.external_ocr_services.hyphen_join import default_morph, join_broken_hyphens
from ocr_utils.external_ocr_services.prompts import system_prompt, user_prompt
from ocr_utils.external_ocr_services import structure
from ocr_utils.external_ocr_services import toc as toc_module
from ocr_utils.external_ocr_services.render import to_markdown
from ocr_utils.external_ocr_services.schema import (
    BlockTag,
    DamageTag,
    FORMULA_TAG,
    PageResult,
    ParseError,
    Stage,
    TocKind,
    count_illustrations,
    drop_duplicate_supplied,
    is_gap_runaway,
    json_schema,
    parse_json_text,
    strip_hyphen_supplied,
    tag_counts,
    tags_from_edge_words,
    unbalanced_fences,
    unbalanced_tags,
)
from ocr_utils.external_ocr_services.tiling import (
    DEFAULT_MAX_MODEL_TILE,
    DEFAULT_MAX_SRC_TILE,
    DEFAULT_QUALITY,
    GridSummary,
    PreparedImage,
    describe,
    prepare_tiles,
)

logger = logging.getLogger(__name__)

# Потолок выходных токенов. Полоса — 3-6 тыс. знаков ≈ 2-4 тыс. токенов, таблица в HTML вдвое
# больше; 16k оставляет запас, но не даёт зациклившейся модели наговорить на доллар.
DEFAULT_MAX_TOKENS = 16000

# Статусы, после которых имеет смысл повторить запрос в более простом режиме JSON.
FALLBACK_STATUSES = frozenset({400, 404, 422})


def is_format_echo(text: str) -> bool:
    """Ответ — эхо ``response_format`` вместо страницы: ``{"type": "json_object"}`` и ничего больше.

    DeepSeek в режиме json_object на длинном промпте (списки статей выпуска) в трети случаев
    отвечает ровно этим объектом на 7 токенов. Лечится повтором без ``response_format``: схема
    и так описана в промпте, а разбор терпимый.

    Args:
        text: Сырой текст ответа модели.

    Returns:
        ``True`` — это эхо (короткий объект с ``"type"`` и без ``content_markdown``), запрос надо повторить.
    """
    body = text.strip()
    return len(body) < 80 and body.startswith("{") and '"type"' in body and "content_markdown" not in body


@dataclass
class RunOptions:
    """Настройки прогона, общие для всех полос: тайлы, модель, второй проход, отладочный выход."""

    # Сетка тайлов: сторона исходника делится на ceil(сторона / max_src_tile) частей с перекрытием,
    # каждый тайл ужимается до max_model_tile по большей стороне. Обычная полоса пака-1 → 2 тайла,
    # склеенный разворот → 4. Подробности и умолчания — в tiling.py.
    max_src_tile: int = DEFAULT_MAX_SRC_TILE
    max_model_tile: int = DEFAULT_MAX_MODEL_TILE
    quality: int = DEFAULT_QUALITY  # JPEG-качество тайлов в запросе
    reasoning: Reasoning | None = None  # переопределение уровня из реестра (``--reasoning``)
    max_tokens: int = DEFAULT_MAX_TOKENS  # потолок выходных токенов на запрос
    # Описание издания для промпта; «{year}» подставляется годом выпуска.
    source: str = ""
    debug_dir: Path | None = None  # сырые ответы, промпты и отправленные тайлы
    # Кэш запросов (``--cache-dir``, cache.py): каждый запрос к модели — своя папка с промптами и
    # ответом; тот же запрос (по хэшу payload с хэшами тайлов) второй раз в сеть не идёт, а берётся
    # с диска. None — без кэша.
    cache_dir: Path | None = None
    # Второй проход по полосе, которую первый проход счёл повреждённой: подсказка собирается из
    # его же ответа (описание, затронутые строки, счётчики). По умолчанию ВЫКЛЮЧЕН (``--second-pass``
    # включает): на МТС 1991/02 он то чинил первый проход, то портил — снимал теги с достроек, не
    # меняя букв, подставлял слова без тега на смазе («немецкое» вместо «акционерное») и переписывал
    # чистые слова из списка подсказок («Солкомфлот» 8/12) — арбитраж 28:13 за первый проход
    # (`reports/external_ocr_hyphenation_second_pass.md`). ``second_pass_transcript`` — слать ещё и
    # весь текст первого прохода.
    second_pass: bool = False
    second_pass_transcript: bool = False
    # Склейка разорванных переносов («кре-диты» → «кредиты») по словарю pymorphy3 после доводки
    # структуры: на полосе с обрезанным краем модель в части ответов оставляет переносы как
    # напечатано (с. 47 МТС 1991/02 — 38 из 39), промптом не лечится. Правило E: 77/79 переносов,
    # 1 ложное слияние на 234 составных (`reports/external_ocr_damaged_research.md`). ``--no-join-hyphens``.
    join_hyphens: bool = True


@dataclass(frozen=True)
class SecondPass:
    """Что первый проход сказал о повреждениях — подсказка для второго прохода той же полосы.

    Всё это подставляется в пользовательский промпт второго прохода (блок ``second_pass`` в
    ``user.md.j2``): модель знает, где искать, и восстанавливает по контексту увереннее.
    """

    damage_description: str  # описание повреждения словами (по-русски), как его дал первый проход
    edge_words: tuple[dict, ...]  # затронутые строки: {"line", "text", "kind"}; режутся до SECOND_PASS_MAX_LINES
    tags: dict  # {DamageTag.SUPPLIED: n, DamageTag.UNCLEAR: n, DamageTag.GAP: n}
    transcript: str | None = None  # полный текст первого прохода, если решено его передавать


@dataclass(frozen=True)
class PageJob:
    """Что распознать: полоса, этап и известное оглавление выпуска (для этапа ``page``).

    Неизменяемый, чтобы безопасно ходить по потокам пула; второй проход делает копию с ``second_pass``.
    """

    rel: Path  # путь полосы относительно in-dir; тот же путь — под out-dir и debug-dir
    stage: Stage = Stage.PAGE  # обычная полоса или полоса оглавления/указателя
    toc_kind: TocKind = TocKind.NONE  # для этапа TOC: что сказала база (CONTENTS | INDEX)
    year: str = ""  # год выпуска — в пользовательский промпт
    rubrics: tuple[str, ...] = ()  # рубрики «Содержания» выпуска — в системный промпт этапа page
    articles: tuple[dict, ...] = ()  # статьи «Содержания»: {"title", "authors", "rubric"}
    toc_hash: str = ""  # отпечаток списков; пишется в meta, по нему is_done узнаёт устаревшую полосу
    second_pass: SecondPass | None = None  # подсказка первого прохода; None — это первый проход
    # Полоса шла этапом TOC, но модель сочла её не оглавлением, и теперь она идёт этапом PAGE;
    # пишется в meta (``toc_demoted``), чтобы повтор с --skip-done узнал её без запроса.
    demoted_from_toc: bool = False

    def __post_init__(self) -> None:
        # Строки из старых вызовов и тестов («page», «contents») приводятся к перечислениям, чтобы
        # ниже работали сравнения через ``is``; чужое значение падает здесь, а не где-то в промпте.
        object.__setattr__(self, "stage", Stage(self.stage))
        object.__setattr__(self, "toc_kind", TocKind(self.toc_kind))


# Сколько строк ``edge_words`` первого прохода показывать второму: хватает, чтобы указать место,
# и не раздувает промпт на страницах, где модель перечислила каждую строку.
SECOND_PASS_MAX_LINES = 60


class SecondPassReason(StrEnum):
    """Почему полоса ушла во второй проход; пишется в meta (``second_pass_reason``) и summary.csv.

    ``DAMAGED`` — модель подняла булев флаг ``is_damaged``; ``TAGS`` — флага нет, но в тексте есть
    теги повреждений; ``EDGE_WORDS`` — флага и тегов нет, но список повреждённых строк не пуст.
    """

    DAMAGED = "damaged"
    TAGS = "tags"
    EDGE_WORDS = "edge_words"


class PassChoice(StrEnum):
    """Какой проход ушёл в финал (``second_pass_chosen`` в meta) и суффикс его файлов в debug-dir."""

    PASS1 = "pass1"
    PASS2 = "pass2"


def second_pass_reason(result: PageResult) -> SecondPassReason | None:
    """Почему полосе нужен второй проход: первый описал повреждение, поставил теги или назвал строки.

    Главный признак — булево ``is_damaged`` (по тексту описания срабатывало на 82 % чистых полос);
    теги и ``edge_words`` — на случай, когда модель разметила повреждение, но флаг не подняла.

    Args:
        result: Разобранный ответ первого прохода.

    Returns:
        Причина второго прохода или ``None`` — полоса чистая, второго прохода не будет.
    """
    if result.is_damaged:
        return SecondPassReason.DAMAGED
    if sum(tag_counts(result.content_markdown).values()) > 0:
        return SecondPassReason.TAGS
    if result.edge_words:
        return SecondPassReason.EDGE_WORDS
    return None


def choose_final(pass1_tags: dict[DamageTag, int], pass2_tags: dict[DamageTag, int] | None) -> tuple[PassChoice, str]:
    """Какой проход оставить: второй, кроме случаев, когда он сбойнул или ушёл в ``<unknown/>``.

    Страховка от вопроса «а не окажется ли в финале пропуск там, где первый проход восстановил»:
    если во втором ``<gap>`` больше, а ``<supplied>`` не больше, — остаётся первый.
    Возвращает выбор и причину словами (пусто, если выбран второй).

    Args:
        pass1_tags: Счётчики тегов повреждений первого прохода (``schema.tag_counts``).
        pass2_tags: То же для второго; ``None`` — второй проход сбойнул.

    Returns:
        ``(выбор, причина)``: ``PASS2`` с пустой причиной или ``PASS1`` с объяснением, почему второй отвергнут.
    """
    if pass2_tags is None:
        return PassChoice.PASS1, "второй проход сбойнул"
    more_gaps = pass2_tags.get(DamageTag.GAP, 0) > pass1_tags.get(DamageTag.GAP, 0)
    no_more_supplied = pass2_tags.get(DamageTag.SUPPLIED, 0) <= pass1_tags.get(DamageTag.SUPPLIED, 0)
    if more_gaps and no_more_supplied:
        return PassChoice.PASS1, "во втором проходе больше <gap> без прироста <supplied>"
    return PassChoice.PASS2, ""


def reasoning_field(spec: ModelSpec, override: Reasoning | None) -> dict | None:
    """Поле ``reasoning`` запроса по уровню из реестра (или ``--reasoning``); ``None`` — не слать.

    Args:
        spec: Модель из реестра — её уровень по умолчанию и знает ли она параметр вообще.
        override: Уровень из ``--reasoning``; ``None`` — взять из реестра.

    Returns:
        ``{"effort": …}`` для поля ``reasoning`` запроса (``"none"`` выключает thinking) или ``None`` —
        поле не слать (модель его не знает).
    """
    level = Reasoning(override) if override is not None else spec.reasoning
    # «none» в реестре значит «модель параметра не знает»: слать нельзя даже по просьбе из CLI.
    if level is Reasoning.NONE or spec.reasoning is Reasoning.NONE:
        return None
    if level is Reasoning.OFF:
        # ``effort: none`` выключает рассуждения целиком (доки OpenRouter); прежнее ``enabled: false``
        # модель SDK не знает и молча выбросила бы — thinking включился бы обратно.
        return {"effort": "none"}
    return {"effort": level.value}


def prompts_for(job: PageJob, tiles: list[PreparedImage], options: RunOptions) -> tuple[str, str]:
    """(системный, пользовательский) промпты полосы.

    Системный зависит только от этапа и издания — он одинаков для всех полос пака, и провайдер
    кэширует его как префикс между выпусками (поэтому ``--source`` лучше давать без года).
    Пользовательский — константная часть, затем списки выпуска, затем сетка тайлов и подсказки
    второго прохода — в порядке «от общего к частному», чтобы префикс совпадал как можно дольше.

    Args:
        job: Что распознаём: этап, вид полосы, списки выпуска, год, подсказка второго прохода.
        tiles: Подготовленные тайлы — из них берётся форма сетки для описания раскладки картинок.
        options: Настройки прогона — описание издания (``source``).

    Returns:
        ``(системный промпт, пользовательский промпт)`` — готовые тексты сообщений.
    """
    info: GridSummary = describe(tiles)
    source = options.source.replace("{year}", job.year).strip()
    # Списки выпуска — в пользовательское сообщение: системный промпт остаётся общим для всего пака
    # и кэшируется провайдером между выпусками; внутри выпуска кэшируется и список.
    system = system_prompt(job.stage, source, has_list=bool(job.articles))
    user = user_prompt(
        len(tiles),
        info.ncols,
        info.nrows,
        job.stage,
        job.toc_kind,
        second_pass=job.second_pass,
        max_lines=SECOND_PASS_MAX_LINES,
        rubrics=list(job.rubrics),
        articles=[dict(a) for a in job.articles],
    )
    return system, user


def provider_field(spec: ModelSpec) -> dict[str, Any]:
    """Маршрутизация OpenRouter для payload: предпочтительные провайдеры по порядку (с откатом на
    остальных) и чёрный список — квантованные копии модели читают хуже.

    Args:
        spec: Модель из реестра.

    Returns:
        Поле ``provider`` payload (пустой словарь — не слать); ``require_parameters`` добавляет
        вызывающий для строгой схемы.
    """
    provider: dict[str, Any] = {}
    if spec.provider_order:
        provider["order"] = list(spec.provider_order)
        provider["allow_fallbacks"] = True
    if spec.provider_ignore:
        provider["ignore"] = list(spec.provider_ignore)
    return provider


def build_payload(
    spec: ModelSpec,
    tiles: list[PreparedImage],
    options: RunOptions,
    job: PageJob,
    json_mode: JsonMode,
    skip_reasoning: bool = False,
) -> dict:
    """Тело chat/completions под модель, этап и режим JSON — «сырой» payload в форме HTTP API.

    Args:
        spec: Модель из реестра: id, ``detail`` картинок, порядок и чёрный список провайдеров.
        tiles: Тайлы полосы в порядке отправки.
        options: Настройки прогона: потолок токенов, уровень рассуждений, описание издания.
        job: Полоса, этап и списки выпуска — для промптов и схемы ответа.
        json_mode: Режим JSON этой попытки (цепочка запасных ходов перебирает их по очереди).
        skip_reasoning: Не слать поле ``reasoning`` (провайдер его отверг на прошлой попытке).

    Returns:
        Словарь в форме HTTP API OpenRouter: ``model``, ``messages`` (system + user с картинками),
        ``temperature``, ``max_tokens`` и, если заданы, ``response_format``, ``reasoning``, ``provider``.
        Клиент переводит его в аргументы SDK.
    """
    system, user = prompts_for(job, tiles, options)
    # Сообщение пользователя: сначала текст, затем тайлы в порядке сетки (столбцами) как data-URL.
    content: list[dict[str, Any]] = [{"type": "text", "text": user}]
    for tile in tiles:
        part: dict[str, Any] = {"url": tile.data_url()}
        if spec.image_detail:  # DeepSeek по «low» ужал бы тайл до 512 px — текст пропадёт
            part["detail"] = spec.image_detail
        content.append({"type": "image_url", "image_url": part})
    # temperature=0: воспроизводимость важнее «живости», это транскрипция, а не сочинение.
    payload: dict[str, Any] = {
        "model": spec.openrouter_id,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": options.max_tokens,
    }
    provider = provider_field(spec)
    if json_mode is JsonMode.JSON_SCHEMA:
        # Строгая схема; require_parameters отсекает провайдеров, которые её молча игнорируют.
        payload["response_format"] = {
            "type": JsonMode.JSON_SCHEMA.value,
            "json_schema": {"name": "page_transcription", "strict": True, "schema": json_schema(job.stage)},
        }
        provider["require_parameters"] = True
    elif json_mode is JsonMode.JSON_OBJECT:
        # «Верни валидный JSON»; сама схема описана словами в системном промпте.
        payload["response_format"] = {"type": JsonMode.JSON_OBJECT.value}
    reasoning = reasoning_field(spec, options.reasoning) if not skip_reasoning else None
    if reasoning is not None:
        payload["reasoning"] = reasoning
    if provider:
        payload["provider"] = provider
    return payload


def _fallback_chain(spec: ModelSpec) -> list[JsonMode]:
    """Режимы JSON от заявленного в реестре к более простым: json_schema → json_object → none.

    Args:
        spec: Модель из реестра — с её ``json_mode`` цепочка начинается.

    Returns:
        Режимы для перебора по порядку, первый — из реестра.
    """
    chain = list(JsonMode)  # порядок объявления: от строгого к простому
    return chain[chain.index(spec.json_mode) :]


@dataclass(frozen=True)
class OutputPaths:
    """Файлы одной полосы под out-dir.

    Args:
        json: Разобранный ответ (все поля ``PageResult``) — для сборки выпуска и повторов.
        md: Тот же результат глазами: YAML-шапка и тело.
        meta: ``.meta.json`` — этап, отпечаток списков, тайлы, токены, цена, ошибки.
        raw: Сырой текст ответа — только при сбое разбора, чтобы не терять оплаченный ответ.
        toc_json: Результат этапа toc у полосы, которую модель сочла НЕ оглавлением и которая
            ушла на этап page: её вклад в оглавление выпуска, чтобы повтор с ``--skip-done`` не
            запрашивал полосу заново.
    """

    json: Path
    md: Path
    meta: Path
    raw: Path
    toc_json: Path


@dataclass(frozen=True)
class DebugPaths:
    """Файлы одной полосы под debug-dir (``--debug-dir``); для второго прохода — с суффиксом ``.pass2``.

    Args:
        raw: Сырой ответ модели (всегда, не только при сбое).
        prompt: Системный и пользовательский промпты ровно в том виде, в каком ушли.
        tiles: Основа имени для тайлов: ``<tiles>.tile_{столбец}{строка}.jpg``.
        json: Разобранный результат прохода (``.pass1.json`` / ``.pass2.json``).
        md: То же в markdown.
    """

    raw: Path
    prompt: Path
    tiles: Path
    json: Path
    md: Path


def output_paths(out_dir: Path, rel: Path) -> OutputPaths:
    """Файлы полосы под out_dir: путь полосы без суффикса плюс ``.json`` / ``.md`` / ``.meta.json`` / ``.raw.txt``.

    Args:
        out_dir: Корень выхода.
        rel: Путь полосы относительно корня входа (та же раскладка сохраняется под out_dir).
    """
    base = out_dir / rel.with_suffix("")
    return OutputPaths(
        json=base.with_suffix(".json"),
        md=base.with_suffix(".md"),
        meta=base.with_suffix(".meta.json"),
        raw=base.with_suffix(".raw.txt"),
        toc_json=base.with_suffix(".toc.json"),
    )


def debug_paths(debug_dir: Path, rel: Path, chosen: PassChoice | None = None) -> DebugPaths:
    """Файлы полосы под debug_dir; ``chosen`` добавляет суффикс ``.pass1`` / ``.pass2``, чтобы проходы не затирали друг друга.

    Args:
        debug_dir: Корень отладочного выхода (``--debug-dir``).
        rel: Путь полосы относительно корня входа.
        chosen: Проход, чьи файлы пишутся; ``None`` — первый проход без суффикса (сырой ответ,
            промпт и тайлы) — как у полос, где второго прохода не было.
    """
    suffix = f".{chosen.value}" if chosen is not None else ""
    base = debug_dir / rel.with_suffix("")
    return DebugPaths(
        raw=base.with_suffix(f"{suffix}.raw.txt"),
        prompt=base.with_suffix(f"{suffix}.prompt.txt"),
        tiles=base,
        json=base.with_suffix(f"{suffix}.json"),
        md=base.with_suffix(f"{suffix}.md"),
    )


def read_meta(out_dir: Path, rel: Path) -> dict | None:
    """``.meta.json`` полосы; нет файла или он битый — ``None`` (полоса считается не сделанной).

    Args:
        out_dir: Корень выхода.
        rel: Путь полосы относительно корня входа.
    """
    path = output_paths(out_dir, rel).meta
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_done(out_dir: Path, job: PageJob) -> bool:
    """Сделано = .json на месте, .meta.json без ошибки, та же версия промпта, тот же этап и тот же отпечаток списков.

    Args:
        out_dir: Корень выхода.
        job: Задание на полосу: этап, вид оглавления и отпечаток списков, с которыми сравнивается meta.

    Returns:
        ``True`` — полосу можно взять с диска и не запрашивать.
    """
    meta = read_meta(out_dir, job.rel)
    # Сбойная полоса (сеть или разбор) сделанной не считается: --skip-done её догонит.
    if meta is None or meta.get("error") or meta.get("parse_error"):
        return False
    paths = output_paths(out_dir, job.rel)
    if not paths.json.is_file():
        return False
    # Полоса, распознанная промптом другой версии, устарела: разметка и теги могли измениться
    # (v18: <rubricintoc>, рубрика только напечатанная); кэш запросов её всё равно не найдёт.
    if meta.get("prompt_version") != PROMPT_VERSION:
        return False
    # Понижённая полоса: финальные файлы — от этапа page, а её ответ этапа toc лежит рядом в
    # .toc.json; для задания toc это «сделано», результат читает load_result из .toc.json.
    if job.stage is Stage.TOC and meta.get("toc_demoted") and meta.get("stage") == Stage.PAGE:
        return paths.toc_json.is_file()
    # Полоса, распознанная как обычная, а теперь помеченная оглавлением (или наоборот), идёт заново.
    if meta.get("stage") != job.stage:
        return False
    # Для оглавления важен вид («Содержание»/указатель — разные промпты), для обычной полосы —
    # что список статей в промпте был тот же: иначе структура (`#`, авторы, рубрики) устарела.
    if job.stage is Stage.TOC:
        return meta.get("toc_kind_expected") == job.toc_kind
    return meta.get("toc_hash") == job.toc_hash


def load_result(out_dir: Path, job: PageJob) -> PageResult | None:
    """Готовый результат полосы из её .json (тот же разбор, что у ответа модели); битый файл — ``None``.

    Args:
        out_dir: Корень выхода.
        job: Задание на полосу — путь и этап (у ``TOC`` разбирается объект ``toc``).
    """
    paths = output_paths(out_dir, job.rel)
    # У понижённой полосы ответ этапа toc лежит отдельно (см. is_done); её .json — уже этап page.
    meta = read_meta(out_dir, job.rel) or {}
    path = paths.toc_json if job.stage is Stage.TOC and meta.get("toc_demoted") else paths.json
    try:
        return parse_json_text(path.read_text(encoding="utf-8"), job.stage)
    except (OSError, ParseError):
        return None


def save_demoted_toc(out_dir: Path, rel: Path, result: PageResult) -> Path:
    """Сохранить ответ этапа toc понижённой полосы в ``.toc.json`` рядом с её файлами.

    Полоса дальше идёт этапом page, и её .json/.md/.meta.json перезапишутся; вклад в оглавление
    выпуска (если модель всё же извлекла статьи) остаётся здесь, и повтор с ``--skip-done`` берёт
    его без запроса.

    Args:
        out_dir: Корень выхода.
        rel: Путь полосы относительно корня входа.
        result: Разобранный ответ этапа toc.

    Returns:
        Путь записанного ``.toc.json``.
    """
    path = output_paths(out_dir, rel).toc_json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json(), encoding="utf-8")
    return path


def _write_debug(options: RunOptions, job: PageJob, tiles: list[PreparedImage], payload: dict, raw: str | None) -> None:
    """В debug-dir: промпты как ушли, отправленные тайлы и сырой ответ; без ``--debug-dir`` ничего.

    Args:
        options: Настройки прогона — откуда берётся ``debug_dir``.
        job: Полоса; по ``job.second_pass`` файлы второго прохода получают суффикс ``.pass2``.
        tiles: Отправленные тайлы — пишутся только у первого прохода (у второго они те же).
        payload: Готовое тело запроса — промпты берутся из него, а не пересобираются.
        raw: Сырой текст ответа; ``None`` — запрос не удался, ответа нет.
    """
    if options.debug_dir is None:
        return
    paths = debug_paths(options.debug_dir, job.rel, PassChoice.PASS2 if job.second_pass else None)
    paths.prompt.parent.mkdir(parents=True, exist_ok=True)
    # Промпт — из готового payload, а не пересобранный: видно ровно то, что получила модель.
    system = payload["messages"][0]["content"]
    user = payload["messages"][1]["content"][0]["text"]
    paths.prompt.write_text(f"=== system ===\n{system}\n=== user ===\n{user}", encoding="utf-8")
    if not job.second_pass:  # тайлы у обоих проходов одни и те же
        for tile in tiles:
            paths.tiles.with_name(f"{paths.tiles.name}.tile_{tile.box.col}{tile.box.row}.jpg").write_bytes(tile.data)
    if raw is not None:
        paths.raw.write_text(raw, encoding="utf-8")


def _verdict(json_mode: JsonMode, text: str) -> CacheVerdict:
    """Годен ли ответ или цепочка откатов идёт дальше: эхо ``response_format``, цикл заполнителя.

    Без режима JSON повторять уже не в чем, поэтому такие ответы считаются годными и идут в разбор.

    Args:
        json_mode: Режим JSON попытки, давшей ответ.
        text: Сырой текст ответа.
    """
    if json_mode is JsonMode.NONE:
        return CacheVerdict.OK
    if is_format_echo(text):
        return CacheVerdict.FORMAT_ECHO
    if is_gap_runaway(text):
        return CacheVerdict.GAP_RUNAWAY
    return CacheVerdict.OK


def _chat_cached(
    client: OpenRouterClient,
    spec: ModelSpec,
    options: RunOptions,
    job: PageJob,
    json_mode: JsonMode,
    payload: dict,
    tiles: list[PreparedImage],
    meta: dict,
) -> tuple[ChatResponse, CacheVerdict]:
    """Ответ на payload: из кэша, если такой запрос уже задавали, иначе от модели с записью в кэш.

    Без ``options.cache_dir`` — просто ``client.chat``. Сетевая ошибка пробрасывается наружу, как у
    клиента, но перед этим дописывается в ``errors.jsonl`` папки полосы в кэше.

    Args:
        client: Клиент OpenRouter.
        spec: Модель — имя и id в ``request.json``.
        options: Настройки прогона — откуда ``cache_dir``.
        job: Полоса и этап — для папки записи и контекста в ``request.json``.
        json_mode: Режим JSON этой попытки — часть имени папки.
        payload: Тело запроса с картинками.
        tiles: Тайлы — их раскладка пишется в ``request.json``.
        meta: Meta полосы; сюда ставятся ``cache_hit`` и ``cache_entry`` (по последнему вызову — он
            и даёт ответ, который идёт в разбор).

    Returns:
        ``(ответ, вердикт)``: вердикт — как ответ оценён (``_verdict``), у записи из кэша — тот, что
        был вынесен при записи.
    """
    if options.cache_dir is None:
        response = client.chat(payload)
        return response, _verdict(json_mode, response.text)
    cache = cache_for(options.cache_dir)
    key = request_key(payload)
    entry = entry_dir(options.cache_dir, job.rel, job.stage.value, json_mode.value, key, job.second_pass is not None)
    meta["cache_entry"] = entry.relative_to(options.cache_dir).as_posix()
    cached = cache.lookup(entry)
    if cached is not None:
        meta["cache_hit"] = True
        logger.info("%s [%s]: ответ из кэша %s (%s)", job.rel, job.stage, entry.name, cached.verdict)
        return cached.response, cached.verdict
    meta["cache_hit"] = False
    try:
        response = client.chat(payload)
    except OpenRouterError as error:
        cache.record_error(entry, str(error), error.body)
        raise
    verdict = _verdict(json_mode, response.text)
    request_info = {
        "page": job.rel.as_posix(),
        "stage": job.stage.value,
        "toc_kind_expected": job.toc_kind.value if job.stage is Stage.TOC else None,
        "toc_hash": job.toc_hash,
        "toc_demoted": job.demoted_from_toc,
        "second_pass": job.second_pass is not None,
        "articles_in_prompt": len(job.articles),
        "json_mode": json_mode.value,
        "prompt_version": PROMPT_VERSION,
        "model": spec.name,
        "openrouter_id": spec.openrouter_id,
        "tiling": describe(tiles).as_dict(),
        "key": key,
    }
    cache.store(entry, request_info, payload, response, verdict)
    return response, verdict


def recognize_page(
    client: OpenRouterClient, spec: ModelSpec, in_path: Path, job: PageJob, out_dir: Path, options: RunOptions
) -> tuple[dict, PageResult | None]:
    """Распознать одну полосу и записать выходы. Возвращает (meta, результат или None при сбое).

    Порядок: тайлы → запрос с цепочкой запасных режимов JSON → разбор → теги из ``edge_words`` →
    доводка структуры (этап ``page``) → .json/.md → meta. Сбой на любом шаге пишет meta с ``error``
    или ``parse_error`` и возвращает ``None``; исключение наружу уходит только из клиента при
    неверном ключе.

    Args:
        client: Клиент OpenRouter (общий на прогон).
        spec: Модель из реестра.
        in_path: Файл полосы на диске (``in_dir / job.rel``).
        job: Задание: путь, этап, вид оглавления, списки выпуска, подсказка второго прохода.
        out_dir: Корень выхода; файлы полосы лягут под ``out_dir / job.rel`` без суффикса.
        options: Настройки запроса: тайлы, потолок токенов, рассуждения, описание издания, debug-dir.

    Returns:
        ``(meta, result)``: meta — то, что записано в ``.meta.json`` (этап, тайлы, провайдер, токены,
        цена, ``structure``, ошибки); result — разобранный ответ или ``None`` при сбое (причина в
        ``meta["error"]`` / ``meta["parse_error"]``).
    """
    paths = output_paths(out_dir, job.rel)
    paths.meta.parent.mkdir(parents=True, exist_ok=True)
    # Meta заводится до запроса: если что-то упадёт, на диске останется запись с причиной, и
    # сводка покажет полосу как сбойную, а не как отсутствующую.
    meta: dict[str, Any] = {
        "page": job.rel.as_posix(),
        "stage": job.stage,
        "toc_kind_expected": job.toc_kind if job.stage is Stage.TOC else None,
        "toc_hash": job.toc_hash,
        "toc_demoted": job.demoted_from_toc,
        "articles_in_prompt": len(job.articles),
        "second_pass_reason": None,
        "second_pass_chosen": None,
        "model": spec.name,
        "openrouter_id": spec.openrouter_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "max_src_tile": options.max_src_tile,
        "max_model_tile": options.max_model_tile,
        "reasoning": options.reasoning or spec.reasoning,
        "error": None,
        "parse_error": None,
    }
    started = time.monotonic()
    # Тайлы режутся здесь, в потоке пула: декодирование и JPEG-сжатие — единственная CPU-работа.
    try:
        tiles = prepare_tiles(in_path, options.max_src_tile, options.max_model_tile, options.quality)
    except Exception as error:  # битый файл — не повод ронять прогон
        meta["error"] = f"картинка: {error}"
        write_meta(paths.meta, meta)
        return meta, None
    meta["tiling"] = describe(tiles).as_dict()  # сетка и размеры — чтобы сверять с тем, что видела модель
    meta["image_bytes"] = sum(len(tile.data) for tile in tiles)

    # Цепочка запасных ходов. Внешний цикл — режимы JSON от строгого к простому; внутри каждого
    # ещё две поправки: убрать отвергнутый параметр reasoning и повторить, если ответ — эхо
    # response_format. Каждый неудачный шаг остаётся в meta["fallbacks"] для разбора после прогона.
    response: ChatResponse | None = None
    skip_reasoning = False
    payload: dict = {}
    chain = _fallback_chain(spec)
    for json_mode in chain:
        payload = build_payload(spec, tiles, options, job, json_mode, skip_reasoning)
        try:
            # Из кэша, если такой запрос уже был; иначе в сеть (внутри — свои повторы по 5xx/429).
            response, verdict = _chat_cached(client, spec, options, job, json_mode, payload, tiles, meta)
        except OpenRouterError as error:
            meta.setdefault("fallbacks", []).append(
                {"json_mode": json_mode, "error": str(error), "body": error.body[:300]}
            )
            # 4xx с упоминанием reasoning в теле: провайдер параметра не знает — тот же запрос без него.
            if error.status in FALLBACK_STATUSES and "reasoning" in payload and "reasoning" in error.body.lower():
                logger.warning("%s %s: параметр reasoning отвергнут (%s), повторяю без него", spec.name, job.rel, error)
                skip_reasoning = True
                try:
                    payload = build_payload(spec, tiles, options, job, json_mode, True)
                    response, verdict = _chat_cached(client, spec, options, job, json_mode, payload, tiles, meta)
                except OpenRouterError as again:
                    meta["fallbacks"].append({"json_mode": json_mode, "error": str(again), "body": again.body[:300]})
                    error = again
                else:
                    meta["json_mode_used"] = json_mode
                    break
            # Другой 4xx из FALLBACK_STATUSES — скорее всего не принят response_format: режим проще.
            if error.status in FALLBACK_STATUSES and json_mode is not JsonMode.NONE:
                logger.warning("%s %s: режим %s отвергнут (%s), пробую проще", spec.name, job.rel, json_mode, error)
                continue
            # Всё остальное (лимиты, 5xx после повторов клиента, «none» тоже отвергнут) — сбой полосы.
            meta["error"] = f"{error} {error.body[:300]}".strip()
            break
        meta["json_mode_used"] = json_mode
        # Ответ пришёл, но это {"type": "json_object"} вместо страницы: деньги за него потрачены
        # зря (считаем отдельно), запрос уходит ещё раз в следующем, более простом режиме.
        if verdict is CacheVerdict.FORMAT_ECHO:
            meta.setdefault("fallbacks", []).append({"json_mode": json_mode, "error": "эхо response_format"})
            meta["cost_usd_wasted"] = round(float(meta.get("cost_usd_wasted") or 0.0) + (response.cost_usd or 0.0), 6)
            logger.warning("%s %s: ответ — эхо response_format, повторяю без режима %s", spec.name, job.rel, json_mode)
            continue
        # Модель зациклилась на «▒» до потолка токенов — ответ обрезан. При temperature 0 тот же
        # запрос даст тот же цикл, поэтому повтор идёт в следующем режиме JSON (другой контекст).
        if verdict is CacheVerdict.GAP_RUNAWAY:
            meta.setdefault("fallbacks", []).append({"json_mode": json_mode, "error": "цикл заполнителя <gap>"})
            meta["cost_usd_wasted"] = round(float(meta.get("cost_usd_wasted") or 0.0) + (response.cost_usd or 0.0), 6)
            logger.warning(
                "%s %s: ответ зациклился на заполнителе <gap>, повторяю без режима %s", spec.name, job.rel, json_mode
            )
            continue
        break
    meta["reasoning_sent"] = not skip_reasoning
    meta["seconds"] = round(time.monotonic() - started, 2)  # с тайлами и всеми повторами, в отличие от latency_s
    if response is None:
        meta.setdefault("error", "запрос не удался")
        _write_debug(options, job, tiles, payload, None)
        write_meta(paths.meta, meta)
        return meta, None

    # Учёт удачного ответа: кто обслужил, сколько токенов (в том числе из кэша префикса), цена по
    # usage.cost провайдера, число попыток клиента.
    meta.update(
        provider=response.provider,
        served_model=response.model,
        request_id=response.request_id,
        finish_reason=response.finish_reason,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        reasoning_tokens=response.reasoning_tokens,
        cached_tokens=response.cached_tokens,
        cost_usd=response.cost_usd,
        latency_s=round(response.latency_s, 2),
        attempts=response.attempts,
        response_chars=len(response.text),
    )
    _write_debug(options, job, tiles, payload, response.text)
    # Разбор терпимый (обрезка ```json, лишние ключи, старые поля); не вышло — сырой текст
    # сохраняется рядом с meta, чтобы не терять оплаченный ответ, полоса — сбойная.
    try:
        result = parse_json_text(response.text, job.stage)
    except ParseError as error:
        meta["parse_error"] = str(error)
        paths.raw.write_text(response.text, encoding="utf-8")
        write_meta(paths.meta, meta)
        return meta, None
    if result.edge_words:
        # Продолжение переноса, объявленное «восстановленным» («ва-» → «ва<supplied>л</supplied>ютных»),
        # — не повреждение: тег снимается, буквы остаются.
        result.content_markdown, stripped = strip_hyphen_supplied(result.content_markdown, result.edge_words)
        if stripped:
            meta["hyphen_supplied_stripped"] = stripped
        # Модель охотнее заполняет список повреждённых строк, чем ставит теги в тексте.
        result.content_markdown, edge_report = tags_from_edge_words(result.content_markdown, result.edge_words)
        meta["tags_from_edge_words"] = edge_report.inserted
        meta["edge_words_empty"] = edge_report.empty
    # «трудностей <supplied>стей</supplied>» — слово целое, хвост в теге лишний.
    result.content_markdown, dropped = drop_duplicate_supplied(result.content_markdown)
    if dropped:
        meta["duplicate_supplied_dropped"] = dropped
    if job.stage is Stage.TOC and result.toc is not None and result.toc.sections:
        # Блок <toc>…</toc> вокруг списка ставит код по границам элементов: модель его не просят
        # (тег она искажала — «< toc>», «<тoc>»); написанный ею по памяти тег нормализован разбором.
        result.content_markdown, wrapped = toc_module.ensure_toc_block(result.content_markdown)
        meta["toc_wrapped"] = wrapped
        # Сверка тела с объектом toc в обе стороны: статьи только из тела — в объект, потом блок
        # <toc> строится заново по достроенному объекту; расхождения — в лог и в messages полосы.
        result.content_markdown, result.toc, check = toc_module.reconcile_toc(result.content_markdown, result.toc)
        meta["toc_check"] = check.as_dict()
        if (message := check.message()) is not None:
            logger.warning("%s: %s", job.rel, message)
            result.messages.append(message)
    if job.stage is Stage.PAGE:
        # `#` только из оглавления, авторы при своей статье, рубрика перед `#` — доводится кодом.
        # Оглавление — источник истины: заголовки не из списка понижаются, утёкшие названия
        # с полос-продолжений снимаются, рубрики подтягиваются к `#`, чужие — в <marker>.
        titles = [article["title"] for article in job.articles if article.get("title")]
        result.content_markdown, report = structure.apply(
            result.content_markdown,
            [dict(article) for article in job.articles],
            list(job.rubrics),
            result.authors,
            result.running_header,
            result.running_footer,
        )
        meta["structure"] = report.as_dict()  # что именно доводка переставила — для проверки глазами
        # title_in_list модель проставляет сама; при непустом списке вердикт кода точнее.
        if titles and report.title_in_list is not None:
            result.title_in_list = report.title_in_list
    if options.join_hyphens:
        # Разорванные переносы («кре-диты») склеиваются по словарю; список склеек — в meta, чтобы при
        # слиянии с FineReader видеть, где вмешался словарь, а не модель.
        result.content_markdown, join_report = join_broken_hyphens(result.content_markdown, default_morph())
        meta["hyphens_joined"] = join_report.joined
    # Два представления одного результата: .json — для сборки выпуска и повторов, .md — глазами.
    paths.json.write_text(result.to_json(), encoding="utf-8")
    paths.md.write_text(to_markdown(result), encoding="utf-8")
    if paths.raw.exists():
        paths.raw.unlink()  # сырой текст от прошлого сбоя разбора больше не нужен
    # Показатели содержимого для сводки: теги повреждений, формулы, флаг повреждения, заголовок.
    meta.update(
        page_number=result.page_number,
        toc_kind=result.toc_kind,
        title=result.title,
        title_in_list=result.title_in_list,
        content_chars=len(result.content_markdown),
        has_header=result.running_header is not None,
        tags=tag_counts(result.content_markdown),
        formulas=result.content_markdown.count(f"<{FORMULA_TAG}>"),
        # Блоки: оглавление и сноски (теги) плюс иллюстрации трёх видов (fenced-блоки) — для
        # регрессии «объекты не хуже»; ключи иллюстраций — photo / schema / line-art.
        blocks={
            **{tag.value: result.content_markdown.count(f"<{tag}>") for tag in BlockTag},
            **{kind.key: count for kind, count in count_illustrations(result.content_markdown).items()},
        },
        is_damaged=result.is_damaged,
        damage_description=result.damage_description,
        messages="; ".join(result.messages),
    )
    if job.stage is Stage.TOC and result.toc is not None:
        meta["toc_articles"] = sum(len(section.articles) for section in result.toc.sections)
    # Непарный <supplied>, <unclear> или <gap> ломает разметку при нарезке; не ошибка, но в meta должно быть видно.
    broken = unbalanced_tags(result.content_markdown)
    if broken:
        meta["tag_warning"] = "непарные теги: " + ", ".join(tag.value for tag in broken)
    # Незакрытый fenced-блок иллюстрации утащит «в цитату» весь остаток полосы.
    if unbalanced_fences(result.content_markdown):
        meta["tag_warning"] = (meta.get("tag_warning") or "").rstrip("; ") + "; незакрытый fenced-блок ```"
        meta["tag_warning"] = meta["tag_warning"].lstrip("; ")
    write_meta(paths.meta, meta)
    return meta, result


def write_meta(path: Path, meta: dict) -> None:
    """``.meta.json`` целиком (перезапись); indent=1 — чтобы diff между прогонами читался построчно.

    Args:
        path: Куда писать (``OutputPaths.meta``).
        meta: Словарь meta; StrEnum внутри уходят строковыми значениями.
    """
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def _write_outputs(out_dir: Path, rel: Path, result: PageResult, meta: dict) -> None:
    """Финальные .json/.md/.meta.json полосы — тем же набором, что пишет :func:`recognize_page`.

    Args:
        out_dir: Корень выхода.
        rel: Путь полосы относительно корня входа.
        result: Результат, выбранный финалом (первый или второй проход).
        meta: Meta финала (с блоком ``second_pass``).
    """
    paths = output_paths(out_dir, rel)
    paths.json.write_text(result.to_json(), encoding="utf-8")
    paths.md.write_text(to_markdown(result), encoding="utf-8")
    write_meta(paths.meta, meta)


def _write_debug_copy(options: RunOptions, rel: Path, chosen: PassChoice, result: PageResult) -> None:
    """Разобранный результат прохода (``.pass1`` / ``.pass2``) в debug-dir — сравнить оба глазами.

    Args:
        options: Настройки прогона — откуда берётся ``debug_dir``; без него ничего не пишется.
        rel: Путь полосы относительно корня входа.
        chosen: Какой проход сохраняем — определяет суффикс файлов.
        result: Разобранный результат этого прохода.
    """
    if options.debug_dir is None:
        return
    paths = debug_paths(options.debug_dir, rel, chosen)
    paths.json.parent.mkdir(parents=True, exist_ok=True)
    paths.json.write_text(result.to_json(), encoding="utf-8")
    paths.md.write_text(to_markdown(result), encoding="utf-8")


def recognize_with_second_pass(
    client: OpenRouterClient, spec: ModelSpec, in_path: Path, job: PageJob, out_dir: Path, options: RunOptions
) -> tuple[dict, PageResult | None]:
    """Первый проход; если он счёл полосу повреждённой — второй с подсказкой из его ответа.

    Финал — второй проход, кроме страховки :func:`choose_final`. Оба прохода остаются в debug
    (``имя.pass1.*`` и ``имя.pass2.*``), в meta финала — ``second_pass`` с причиной, тегами обоих
    проходов и выбором; ``cost_usd`` — сумма обоих запросов.

    Args:
        client: Клиент OpenRouter (общий на прогон).
        spec: Модель из реестра.
        in_path: Файл полосы на диске.
        job: Задание первого прохода; второй получает его копию с ``second_pass``.
        out_dir: Корень выхода — финал перезаписывает файлы полосы целиком.
        options: Настройки прогона: ``second_pass`` (по умолчанию выключен, ``--second-pass``),
            ``second_pass_transcript`` (слать ли текст первого прохода), debug-dir.

    Returns:
        ``(meta, result)`` выбранного прохода, как у :func:`recognize_page`; при втором проходе в meta
        добавлены ``second_pass`` (подробности), ``second_pass_reason``, ``second_pass_chosen`` и
        ``cost_usd`` — сумма обоих запросов.
    """
    # Первый проход — обычный запрос; его выход уже лежит под out-dir.
    meta1, result1 = recognize_page(client, spec, in_path, job, out_dir, options)
    # Сбой, этап TOC (оглавление не восстанавливаем) или проход выключен (умолчание) — без второго.
    if result1 is None or job.stage is not Stage.PAGE or not options.second_pass:
        return meta1, result1
    reason = second_pass_reason(result1)
    if reason is None:  # чистая полоса — большинство: второй проход стоит ×2 только повреждённым
        return meta1, result1

    tags1 = tag_counts(result1.content_markdown)
    # Подсказка второму проходу — всё, что первый сказал о повреждениях. Полный текст первого
    # прохода по умолчанию не шлём: в замерах он не помог, а промпт удваивал.
    hint = SecondPass(
        damage_description=result1.damage_description,
        edge_words=tuple(result1.edge_words),
        tags=tags1,
        transcript=result1.content_markdown if options.second_pass_transcript else None,
    )
    _write_debug_copy(options, job.rel, PassChoice.PASS1, result1)
    # Та же полоса, тот же этап и списки, плюс подсказка; второй проход перезапишет .json/.md/meta.
    job2 = replace(job, second_pass=hint)
    meta2, result2 = recognize_page(client, spec, in_path, job2, out_dir, options)
    tags2 = tag_counts(result2.content_markdown) if result2 is not None else None
    chosen, why = choose_final(tags1, tags2)
    # Всё о втором проходе — в meta финала под ключом second_pass; плоские second_pass_reason /
    # second_pass_chosen — для колонок summary.csv.
    info = {
        "reason": reason,
        "pass1_tags": tags1,
        "pass2_tags": tags2,
        "pass1_damage": result1.damage_description,
        "pass2_damage": result2.damage_description if result2 is not None else None,
        "chosen": chosen,
        "why": why,
        "cost_usd_pass1": meta1.get("cost_usd"),
        "cost_usd_pass2": meta2.get("cost_usd"),
        "pass2_error": meta2.get("error") or meta2.get("parse_error"),
    }
    total_cost = float(meta1.get("cost_usd") or 0.0) + float(meta2.get("cost_usd") or 0.0)
    if chosen is PassChoice.PASS2:
        meta, result = meta2, result2
    else:
        meta, result = meta1, result1
        # Второй проход проиграл, но его результат стоит сохранить в debug: по нему видно, почему.
        if result2 is not None:
            _write_debug_copy(options, job.rel, PassChoice.PASS2, result2)
    meta = dict(meta)
    meta.update(second_pass=info, second_pass_reason=reason, second_pass_chosen=chosen, cost_usd=total_cost)
    # Второй проход мог записать сбойную meta или чужой результат — финал перезаписывается целиком.
    _write_outputs(out_dir, job.rel, result, meta)
    logger.info(
        "%s: второй проход (%s): %s → %s, оставлен %s%s",
        job.rel,
        reason,
        tags1,
        tags2,
        chosen,
        f" ({why})" if why else "",
    )
    return meta, result
