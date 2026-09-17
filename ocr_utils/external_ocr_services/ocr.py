"""Одна полоса → один запрос к модели → PageResult, файлы выхода и .meta.json.

Здесь собирается тело запроса под особенности модели (режим JSON, рассуждения, порядок
провайдеров) и этап (``toc`` или ``page`` со списками выпуска), и здесь же запасные ходы:
если провайдер не принял строгую схему, тот же запрос уходит в режиме json_object, потом
вообще без response_format; отвергнутый параметр ``reasoning`` убирается и запрос повторяется.

Готовность полосы определяется по .meta.json и отпечатку списков (``toc_hash``): полоса,
распознанная с другим списком статей, считается устаревшей и идёт заново.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ocr_utils.external_ocr_services import PROMPT_VERSION
from ocr_utils.external_ocr_services.client import ChatResponse, OpenRouterClient, OpenRouterError
from ocr_utils.external_ocr_services.models import JsonMode, ModelSpec
from ocr_utils.external_ocr_services.prompts import system_prompt, user_prompt
from ocr_utils.external_ocr_services import structure
from ocr_utils.external_ocr_services.render import to_markdown
from ocr_utils.external_ocr_services.schema import (
    PageResult,
    ParseError,
    json_schema,
    parse_json_text,
    tag_counts,
    tags_from_edge_words,
    unbalanced_tags,
)
from ocr_utils.external_ocr_services.tiling import (
    DEFAULT_MAX_MODEL_TILE,
    DEFAULT_MAX_SRC_TILE,
    DEFAULT_QUALITY,
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
    reasoning: str | None = None  # переопределение уровня из реестра
    max_tokens: int = DEFAULT_MAX_TOKENS  # потолок выходных токенов на запрос
    # Описание издания для промпта; «{year}» подставляется годом выпуска.
    source: str = ""
    debug_dir: Path | None = None  # сырые ответы, промпты и отправленные тайлы
    # Второй проход по полосе, которую первый проход счёл повреждённой: подсказка собирается из
    # его же ответа (описание, затронутые строки, счётчики). Включён всегда (опции в CLI нет;
    # False — только в тестах). ``second_pass_transcript`` — слать ещё и весь текст первого прохода.
    second_pass: bool = True
    second_pass_transcript: bool = False


@dataclass(frozen=True)
class SecondPass:
    """Что первый проход сказал о повреждениях — подсказка для второго прохода той же полосы.

    Всё это подставляется в пользовательский промпт второго прохода (блок ``second_pass`` в
    ``user.md.j2``): модель знает, где искать, и восстанавливает по контексту увереннее.
    """

    damage: str  # описание повреждения словами (по-русски), как его дал первый проход
    edge_words: tuple[dict, ...]  # затронутые строки: {"line", "text", "kind"}; режутся до SECOND_PASS_MAX_LINES
    tags: dict  # {"restored": n, "fuzzy": n, "unknown": n}
    transcript: str | None = None  # полный текст первого прохода, если решено его передавать


@dataclass(frozen=True)
class PageJob:
    """Что распознать: полоса, этап и известное оглавление выпуска (для этапа ``page``).

    Неизменяемый, чтобы безопасно ходить по потокам пула; второй проход делает копию с ``second_pass``.
    """

    rel: Path  # путь полосы относительно in-dir; тот же путь — под out-dir и debug-dir
    stage: str = "page"  # page | toc
    toc_kind: str = "none"  # для этапа toc: contents | index (что сказала база)
    year: str = ""  # год выпуска — в пользовательский промпт
    rubrics: tuple[str, ...] = ()  # рубрики «Содержания» выпуска — в системный промпт этапа page
    articles: tuple[dict, ...] = ()  # статьи «Содержания»: {"title", "authors", "rubric"}
    toc_hash: str = ""  # отпечаток списков; пишется в meta, по нему is_done узнаёт устаревшую полосу
    second_pass: SecondPass | None = None  # подсказка первого прохода; None — это первый проход


# Сколько строк ``edge_words`` первого прохода показывать второму: хватает, чтобы указать место,
# и не раздувает промпт на страницах, где модель перечислила каждую строку.
SECOND_PASS_MAX_LINES = 60


def second_pass_reason(result: PageResult) -> str | None:
    """Почему полосе нужен второй проход: первый описал повреждение, поставил теги или назвал строки.

    Главный признак — булево ``damaged`` (по тексту описания срабатывало на 82 % чистых полос);
    теги и ``edge_words`` — на случай, когда модель разметила повреждение, но флаг не подняла.
    """
    if result.damaged:
        return "damaged"
    if sum(tag_counts(result.content_markdown).values()) > 0:
        return "tags"
    if result.edge_words:
        return "edge_words"
    return None


def choose_final(pass1_tags: dict, pass2_tags: dict | None) -> tuple[str, str]:
    """Какой проход оставить: второй, кроме случаев, когда он сбойнул или ушёл в ``<unknown/>``.

    Страховка от вопроса «а не окажется ли в финале unknown там, где первый проход восстановил»:
    если во втором ``<unknown/>`` больше, а ``<restored>`` не больше, — остаётся первый.
    """
    if pass2_tags is None:
        return "pass1", "второй проход сбойнул"
    if pass2_tags.get("unknown", 0) > pass1_tags.get("unknown", 0) and pass2_tags.get("restored", 0) <= pass1_tags.get(
        "restored", 0
    ):
        return "pass1", "во втором проходе больше <unknown/> без прироста <restored>"
    return "pass2", ""


def reasoning_field(spec: ModelSpec, override: str | None) -> dict | None:
    """Поле ``reasoning`` запроса по уровню из реестра (или ``--reasoning``); ``None`` — не слать."""
    level = override or spec.reasoning
    # «none» в реестре значит «модель параметра не знает»: слать нельзя даже по просьбе из CLI.
    if level == "none" or spec.reasoning == "none":
        return None
    if level == "off":
        return {"enabled": False}
    if spec.reasoning_max_tokens:
        # У OpenRouter effort и max_tokens взаимоисключающие; потолок важнее уровня.
        return {"max_tokens": spec.reasoning_max_tokens}
    return {"effort": level}


def prompts_for(job: PageJob, tiles: list[PreparedImage], options: RunOptions) -> tuple[str, str]:
    """(системный, пользовательский) промпты полосы.

    Системный зависит от этапа, издания и списков выпуска — он одинаков для всех полос выпуска,
    и провайдер кэширует его как префикс (поэтому ``--source`` лучше давать без года).
    Пользовательский — про эту полосу: сетка тайлов, вид полосы, подсказки второго прохода.
    """
    info = describe(tiles)
    source = options.source.replace("{year}", job.year).strip()
    system = system_prompt(job.stage, source, list(job.rubrics), [dict(a) for a in job.articles])
    user = user_prompt(
        len(tiles),
        info["ncols"],
        info["nrows"],
        job.stage,
        job.toc_kind,
        second_pass=job.second_pass,
        max_lines=SECOND_PASS_MAX_LINES,
    )
    return system, user


def build_payload(
    spec: ModelSpec,
    tiles: list[PreparedImage],
    options: RunOptions,
    job: PageJob,
    json_mode: JsonMode,
    skip_reasoning: bool = False,
) -> dict:
    """Тело chat/completions под модель, этап и режим JSON; ``skip_reasoning`` — без поля reasoning."""
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
    # Маршрутизация OpenRouter: предпочтительные провайдеры по порядку (с откатом на остальных)
    # и чёрный список — квантованные копии модели читают хуже.
    provider: dict[str, Any] = {}
    if spec.provider_order:
        provider["order"] = list(spec.provider_order)
        provider["allow_fallbacks"] = True
    if spec.provider_ignore:
        provider["ignore"] = list(spec.provider_ignore)
    if json_mode == "json_schema":
        # Строгая схема; require_parameters отсекает провайдеров, которые её молча игнорируют.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "page_transcription", "strict": True, "schema": json_schema(job.stage)},
        }
        provider["require_parameters"] = True
    elif json_mode == "json_object":
        # «Верни валидный JSON»; сама схема описана словами в системном промпте.
        payload["response_format"] = {"type": "json_object"}
    reasoning = reasoning_field(spec, options.reasoning) if not skip_reasoning else None
    if reasoning is not None:
        payload["reasoning"] = reasoning
    if provider:
        payload["provider"] = provider
    return payload


def _fallback_chain(spec: ModelSpec) -> list[JsonMode]:
    """Режимы JSON от заявленного в реестре к более простым: json_schema → json_object → none."""
    chain: list[JsonMode] = ["json_schema", "json_object", "none"]
    return chain[chain.index(spec.json_mode) :]


def output_paths(out_dir: Path, rel: Path) -> dict[str, Path]:
    """Файлы полосы под out_dir: разобранный .json, .md для чтения, .meta.json, .raw.txt при сбое разбора."""
    base = out_dir / rel.with_suffix("")
    return {
        "json": base.with_suffix(".json"),
        "md": base.with_suffix(".md"),
        "meta": base.with_suffix(".meta.json"),
        "raw": base.with_suffix(".raw.txt"),
    }


def debug_paths(debug_dir: Path, rel: Path, suffix: str = "") -> dict[str, Path]:
    """``suffix`` — «.pass2» для файлов второго прохода, чтобы не затирать первый."""
    base = debug_dir / rel.with_suffix("")
    return {
        "raw": base.with_suffix(f"{suffix}.raw.txt"),
        "prompt": base.with_suffix(f"{suffix}.prompt.txt"),
        "tiles": base,
        "json": base.with_suffix(f"{suffix}.json"),
        "md": base.with_suffix(f"{suffix}.md"),
    }


def read_meta(out_dir: Path, rel: Path) -> dict | None:
    """``.meta.json`` полосы; нет файла или он битый — ``None`` (полоса считается не сделанной)."""
    path = output_paths(out_dir, rel)["meta"]
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_done(out_dir: Path, job: PageJob) -> bool:
    """Сделано = .json на месте, .meta.json без ошибки, тот же этап и тот же отпечаток списков."""
    meta = read_meta(out_dir, job.rel)
    # Сбойная полоса (сеть или разбор) сделанной не считается: --skip-done её догонит.
    if meta is None or meta.get("error") or meta.get("parse_error"):
        return False
    if not output_paths(out_dir, job.rel)["json"].is_file():
        return False
    # Полоса, распознанная как обычная, а теперь помеченная оглавлением (или наоборот), идёт заново.
    if meta.get("stage") != job.stage:
        return False
    # Для оглавления важен вид («Содержание»/указатель — разные промпты), для обычной полосы —
    # что список статей в промпте был тот же: иначе структура (`#`, авторы, рубрики) устарела.
    if job.stage == "toc":
        return meta.get("toc_kind_expected") == job.toc_kind
    return meta.get("toc_hash") == job.toc_hash


def load_result(out_dir: Path, job: PageJob) -> PageResult | None:
    """Готовый результат полосы из её .json (тот же разбор, что у ответа модели)."""
    path = output_paths(out_dir, job.rel)["json"]
    try:
        return parse_json_text(path.read_text(encoding="utf-8"), job.stage)
    except (OSError, ParseError):
        return None


def _write_debug(options: RunOptions, job: PageJob, tiles: list[PreparedImage], payload: dict, raw: str | None) -> None:
    """В debug-dir: промпты как ушли, отправленные тайлы и сырой ответ; без ``--debug-dir`` ничего."""
    if options.debug_dir is None:
        return
    paths = debug_paths(options.debug_dir, job.rel, ".pass2" if job.second_pass else "")
    paths["prompt"].parent.mkdir(parents=True, exist_ok=True)
    # Промпт — из готового payload, а не пересобранный: видно ровно то, что получила модель.
    system = payload["messages"][0]["content"]
    user = payload["messages"][1]["content"][0]["text"]
    paths["prompt"].write_text(f"=== system ===\n{system}\n=== user ===\n{user}", encoding="utf-8")
    if not job.second_pass:  # тайлы у обоих проходов одни и те же
        for tile in tiles:
            paths["tiles"].with_name(f"{paths['tiles'].name}.tile_{tile.box.col}{tile.box.row}.jpg").write_bytes(
                tile.data
            )
    if raw is not None:
        paths["raw"].write_text(raw, encoding="utf-8")


def recognise_page(
    client: OpenRouterClient, spec: ModelSpec, in_path: Path, job: PageJob, out_dir: Path, options: RunOptions
) -> tuple[dict, PageResult | None]:
    """Распознать одну полосу и записать выходы. Возвращает (meta, результат или None при сбое).

    Порядок: тайлы → запрос с цепочкой запасных режимов JSON → разбор → теги из ``edge_words`` →
    доводка структуры (этап ``page``) → .json/.md → meta. Сбой на любом шаге пишет meta с ``error``
    или ``parse_error`` и возвращает ``None``; исключение наружу уходит только из клиента при
    неверном ключе.
    """
    paths = output_paths(out_dir, job.rel)
    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    # Meta заводится до запроса: если что-то упадёт, на диске останется запись с причиной, и
    # сводка покажет полосу как сбойную, а не как отсутствующую.
    meta: dict[str, Any] = {
        "page": job.rel.as_posix(),
        "stage": job.stage,
        "toc_kind_expected": job.toc_kind if job.stage == "toc" else None,
        "toc_hash": job.toc_hash,
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
        _write_meta(paths["meta"], meta)
        return meta, None
    meta["tiling"] = describe(tiles)  # сетка, размеры и перекрытие — чтобы сверять с тем, что видела модель
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
            response = client.chat(payload)  # внутри — свои повторы по сети и 5xx/429
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
                    response = client.chat(payload)
                except OpenRouterError as again:
                    meta["fallbacks"].append({"json_mode": json_mode, "error": str(again), "body": again.body[:300]})
                    error = again
                else:
                    meta["json_mode_used"] = json_mode
                    break
            # Другой 4xx из FALLBACK_STATUSES — скорее всего не принят response_format: режим проще.
            if error.status in FALLBACK_STATUSES and json_mode != "none":
                logger.warning("%s %s: режим %s отвергнут (%s), пробую проще", spec.name, job.rel, json_mode, error)
                continue
            # Всё остальное (лимиты, 5xx после повторов клиента, «none» тоже отвергнут) — сбой полосы.
            meta["error"] = f"{error} {error.body[:300]}".strip()
            break
        meta["json_mode_used"] = json_mode
        # Ответ пришёл, но это {"type": "json_object"} вместо страницы: деньги за него потрачены
        # зря (считаем отдельно), запрос уходит ещё раз в следующем, более простом режиме.
        if json_mode != "none" and is_format_echo(response.text):
            meta.setdefault("fallbacks", []).append({"json_mode": json_mode, "error": "эхо response_format"})
            meta["cost_usd_wasted"] = round(float(meta.get("cost_usd_wasted") or 0.0) + (response.cost_usd or 0.0), 6)
            logger.warning("%s %s: ответ — эхо response_format, повторяю без режима %s", spec.name, job.rel, json_mode)
            continue
        break
    meta["reasoning_sent"] = not skip_reasoning
    meta["seconds"] = round(time.monotonic() - started, 2)  # с тайлами и всеми повторами, в отличие от latency_s
    if response is None:
        meta.setdefault("error", "запрос не удался")
        _write_debug(options, job, tiles, payload, None)
        _write_meta(paths["meta"], meta)
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
        paths["raw"].write_text(response.text, encoding="utf-8")
        _write_meta(paths["meta"], meta)
        return meta, None
    if result.edge_words:
        # Модель охотнее заполняет список повреждённых строк, чем ставит теги в тексте.
        result.content_markdown, inserted = tags_from_edge_words(result.content_markdown, result.edge_words)
        meta["tags_from_edge_words"] = inserted
    if job.stage == "page":
        # `#` только из оглавления, авторы при своей статье, рубрика перед `#` — доводится кодом.
        # Оглавление — источник истины: заголовки не из списка понижаются, утёкшие названия
        # с полос-продолжений снимаются, рубрики подтягиваются к `#`, чужие — в <marker>.
        titles = [article["title"] for article in job.articles if article.get("title")]
        result.content_markdown, report = structure.apply(
            result.content_markdown, [dict(article) for article in job.articles], list(job.rubrics), result.authors
        )
        meta["structure"] = report.as_dict()  # что именно доводка переставила — для проверки глазами
        # title_in_list модель проставляет сама; при непустом списке вердикт кода точнее.
        if titles and report.title_in_list is not None:
            result.title_in_list = report.title_in_list
    # Два представления одного результата: .json — для сборки выпуска и повторов, .md — глазами.
    paths["json"].write_text(result.to_json(), encoding="utf-8")
    paths["md"].write_text(to_markdown(result), encoding="utf-8")
    if paths["raw"].exists():
        paths["raw"].unlink()  # сырой текст от прошлого сбоя разбора больше не нужен
    # Показатели содержимого для сводки: теги повреждений, формулы, флаг повреждения, заголовок.
    meta.update(
        page_number=result.page_number,
        toc_kind=result.toc_kind,
        title=result.title,
        title_in_list=result.title_in_list,
        content_chars=len(result.content_markdown),
        has_header=result.running_header is not None,
        tags=tag_counts(result.content_markdown),
        formulas=result.content_markdown.count("<latex>"),
        damaged=result.damaged,
        damage_seen=result.damage,
    )
    if job.stage == "toc" and result.toc is not None:
        meta["toc_articles"] = sum(len(section.articles) for section in result.toc.sections)
    # Непарный <restored> или <latex> ломает разметку при нарезке; не ошибка, но в meta должно быть видно.
    broken = unbalanced_tags(result.content_markdown)
    if broken:
        meta["tag_warning"] = "непарные теги: " + ", ".join(broken)
    _write_meta(paths["meta"], meta)
    return meta, result


def _write_meta(path: Path, meta: dict) -> None:
    """``.meta.json`` целиком (перезапись); indent=1 — чтобы diff между прогонами читался построчно."""
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def _write_outputs(out_dir: Path, rel: Path, result: PageResult, meta: dict) -> None:
    """Финальные .json/.md/.meta.json полосы — тем же набором, что пишет :func:`recognise_page`."""
    paths = output_paths(out_dir, rel)
    paths["json"].write_text(result.to_json(), encoding="utf-8")
    paths["md"].write_text(to_markdown(result), encoding="utf-8")
    _write_meta(paths["meta"], meta)


def _write_debug_copy(options: RunOptions, rel: Path, suffix: str, result: PageResult) -> None:
    """Разобранный результат прохода (``.pass1`` / ``.pass2``) в debug-dir — сравнить оба глазами."""
    if options.debug_dir is None:
        return
    paths = debug_paths(options.debug_dir, rel, suffix)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(result.to_json(), encoding="utf-8")
    paths["md"].write_text(to_markdown(result), encoding="utf-8")


def recognise_with_second_pass(
    client: OpenRouterClient, spec: ModelSpec, in_path: Path, job: PageJob, out_dir: Path, options: RunOptions
) -> tuple[dict, PageResult | None]:
    """Первый проход; если он счёл полосу повреждённой — второй с подсказкой из его ответа.

    Финал — второй проход, кроме страховки :func:`choose_final`. Оба прохода остаются в debug
    (``имя.pass1.*`` и ``имя.pass2.*``), в meta финала — ``second_pass`` с причиной, тегами обоих
    проходов и выбором; ``cost_usd`` — сумма обоих запросов.
    """
    # Первый проход — обычный запрос; его выход уже лежит под out-dir.
    meta1, result1 = recognise_page(client, spec, in_path, job, out_dir, options)
    # Сбой, этап toc (оглавление не восстанавливаем) или проход выключен тестом — без второго.
    if result1 is None or job.stage != "page" or not options.second_pass:
        return meta1, result1
    reason = second_pass_reason(result1)
    if reason is None:  # чистая полоса — большинство: второй проход стоит ×2 только повреждённым
        return meta1, result1

    tags1 = tag_counts(result1.content_markdown)
    # Подсказка второму проходу — всё, что первый сказал о повреждениях. Полный текст первого
    # прохода по умолчанию не шлём: в замерах он не помог, а промпт удваивал.
    hint = SecondPass(
        damage=result1.damage,
        edge_words=tuple(result1.edge_words),
        tags=tags1,
        transcript=result1.content_markdown if options.second_pass_transcript else None,
    )
    _write_debug_copy(options, job.rel, ".pass1", result1)
    # Та же полоса, тот же этап и списки, плюс подсказка; второй проход перезапишет .json/.md/meta.
    job2 = PageJob(job.rel, job.stage, job.toc_kind, job.year, job.rubrics, job.articles, job.toc_hash, hint)
    meta2, result2 = recognise_page(client, spec, in_path, job2, out_dir, options)
    tags2 = tag_counts(result2.content_markdown) if result2 is not None else None
    chosen, why = choose_final(tags1, tags2)
    # Всё о втором проходе — в meta финала под ключом second_pass; плоские second_pass_reason /
    # second_pass_chosen — для колонок summary.csv.
    info = {
        "reason": reason,
        "pass1_tags": tags1,
        "pass2_tags": tags2,
        "pass1_damage": result1.damage,
        "pass2_damage": result2.damage if result2 is not None else None,
        "chosen": chosen,
        "why": why,
        "cost_usd_pass1": meta1.get("cost_usd"),
        "cost_usd_pass2": meta2.get("cost_usd"),
        "pass2_error": meta2.get("error") or meta2.get("parse_error"),
    }
    total_cost = float(meta1.get("cost_usd") or 0.0) + float(meta2.get("cost_usd") or 0.0)
    if chosen == "pass2":
        meta, result = meta2, result2
    else:
        meta, result = meta1, result1
        # Второй проход проиграл, но его результат стоит сохранить в debug: по нему видно, почему.
        if result2 is not None:
            _write_debug_copy(options, job.rel, ".pass2", result2)
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
