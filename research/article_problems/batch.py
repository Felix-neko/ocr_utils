"""Batch API OpenRouter: отправка статей пачками, опрос статуса, перенос готовых ответов в общий кэш запросов.

Протокол (https://openrouter.ai/docs/batch-quickstart, проверено 2026-09-29): ``POST /api/v1/batches`` с
``endpoint``, ``model`` (слаг ``:batch``) и массивом ``requests`` из ``{custom_id, body}`` — в JSON поля
верхнего уровня обязаны идти **до** ``requests``. Ответ 202 с id; ``GET /api/v1/batches/{id}`` отдаёт
статус (validating → in_progress → finalizing → completed; failed/expired/cancelled — тоже конечные), а у
завершённого — ``results`` прямо в ответе. Окно — 24 ч, результаты хранятся 30 дней. Chat/completions
со слагом ``:batch`` отвечает 404 («cannot be used with the chat/completions endpoint»).

Состояние каждой отправленной пачки — JSON в ``{out_root}/batches/{id}.json``: какие статьи в ней, куда
класть ответы; ``collect`` перечитывает его, забирает готовые пачки и кладёт ответы в кэш под тем же
ключом, под которым их ищет ``report``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

import httpx

from ocr_utils.external_ocr_services.cache import CacheVerdict, RequestCache
from ocr_utils.external_ocr_services.client import APP_REFERER, APP_TITLE, ChatResponse, OpenRouterError
from research.article_problems.articles import Article, split_issue
from research.article_problems.payload import (
    PRICE_PER_M,
    ReasoningLevel,
    Transport,
    Variant,
    cache_entry,
    cache_payload,
    request_body,
    request_info,
)

logger = logging.getLogger(__name__)

BATCHES_URL = "https://openrouter.ai/api/v1/batches"
ENDPOINT = "/v1/chat/completions"
# Пачка не больше стольких запросов и мегабайт JSON: лимитов в документации нет, берём с запасом,
# чтобы один сбой не хоронил весь пак и тело запроса не росло до сотен мегабайт.
MAX_REQUESTS_PER_BATCH = 500
MAX_BATCH_MB = 25.0
HTTP_TIMEOUT = 300.0


class BatchStatus(StrEnum):
    """Статус пачки по документации OpenRouter."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """Конечный ли статус (дальше не изменится)."""
        return self in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED)


@dataclass(frozen=True)
class CollectStats:
    """Итоги сбора пачек.

    Args:
        batches_done: Пачек, собранных в этот раз (перешли в конечный статус).
        batches_pending: Пачек, ещё не готовых.
        stored: Ответов, положенных в кэш.
        failed: Запросов, вернувшихся с ошибкой.
        cost_usd: Сумма ``usage.cost`` собранных пачек.
    """

    batches_done: int = 0
    batches_pending: int = 0
    stored: int = 0
    failed: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: CollectStats) -> CollectStats:
        """Сложить итоги поэлементно."""
        return CollectStats(
            *(getattr(self, name) + getattr(other, name) for name in self.__dataclass_fields__)  # type: ignore[arg-type]
        )


def _headers(api_key: str) -> dict[str, str]:
    """Заголовки запросов к Batch API (ключ + визитка приложения, как у синхронного клиента).

    Args:
        api_key: Ключ OpenRouter.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_REFERER,
        "X-Title": APP_TITLE,
    }


def custom_id_of(article: Article) -> str:
    """``custom_id`` запроса статьи в пачке: ``1968_09_A15`` (уникален в паке).

    Args:
        article: Статья.
    """
    return f"{article.year}_{article.issue}_{article.article_id}"


def chunk_articles(articles: list[Article], variant: Variant) -> list[list[Article]]:
    """Разбить статьи на пачки по числу запросов и объёму JSON.

    Args:
        articles: Статьи к отправке.
        variant: Вариант (объём тела зависит от промпта).

    Returns:
        Списки статей; каждый — одна пачка.
    """
    chunks: list[list[Article]] = []
    current: list[Article] = []
    size_mb = 0.0
    for article in articles:
        item_mb = len(json.dumps(request_body(article, variant), ensure_ascii=False).encode("utf-8")) / 1e6
        if current and (len(current) >= MAX_REQUESTS_PER_BATCH or size_mb + item_mb > MAX_BATCH_MB):
            chunks.append(current)
            current, size_mb = [], 0.0
        current.append(article)
        size_mb += item_mb
    if current:
        chunks.append(current)
    return chunks


def submit_chunk(articles: list[Article], variant: Variant, api_key: str, cache_root: Path, state_dir: Path) -> str:
    """Отправить одну пачку и записать её состояние.

    Args:
        articles: Статьи пачки.
        variant: Вариант (транспорт ``BATCH``).
        api_key: Ключ OpenRouter.
        cache_root: Корень кэша (пути записей идут в состояние).
        state_dir: Папка состояний пачек.

    Returns:
        Id пачки у OpenRouter.
    """
    # Порядок ключей важен: endpoint и model до requests (сервер разбирает тело потоком).
    body = {
        "endpoint": ENDPOINT,
        "model": variant.model_slug,
        "requests": [
            {"custom_id": custom_id_of(article), "body": request_body(article, variant)} for article in articles
        ],
    }
    response = httpx.post(
        BATCHES_URL,
        headers=_headers(api_key),
        content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        timeout=HTTP_TIMEOUT,
    )
    if response.status_code not in (200, 202):
        raise OpenRouterError(
            f"пачка не принята: HTTP {response.status_code}", response.status_code, response.text[:2000]
        )
    batch = response.json()
    state = {
        "id": batch["id"],
        "variant": {
            "model": variant.model,
            "transport": variant.transport.value,
            "reasoning": variant.reasoning.value,
            "prompt_version": variant.prompt_version,
        },
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": batch.get("status"),
        "collected": False,
        # По custom_id — где статья лежит (чтобы перечитать её при сборе) и куда класть ответ.
        "requests": {
            custom_id_of(article): {
                "article": article.key,
                "source": str(article.source_path),
                "article_id": article.article_id,
                "entry": str(cache_entry(cache_root, article, variant)),
            }
            for article in articles
        },
    }
    _write_state(state_dir / f"{batch['id']}.json", state)
    logger.info("пачка %s: %d статей, статус %s", batch["id"], len(articles), batch.get("status"))
    return batch["id"]


def submit(
    articles: list[Article], variant: Variant, api_key: str, cache: RequestCache, cache_root: Path, state_dir: Path
) -> list[str]:
    """Отправить статьи, которых ещё нет в кэше и нет в неcобранных пачках, пачками.

    Args:
        articles: Статьи прогона.
        variant: Вариант (транспорт ``BATCH``).
        api_key: Ключ OpenRouter.
        cache: Кэш запросов.
        cache_root: Его корень.
        state_dir: Папка состояний пачек.

    Returns:
        Id отправленных пачек.
    """
    if variant.transport != Transport.BATCH:
        raise ValueError("submit — только для варианта с транспортом batch")
    pending = pending_entries(state_dir)
    todo = []
    for article in articles:
        entry = cache_entry(cache_root, article, variant)
        if cache.lookup(entry) is None and str(entry) not in pending:
            todo.append(article)
    logger.info("к отправке %d из %d статей (остальные в кэше или в пачках)", len(todo), len(articles))
    return [submit_chunk(chunk, variant, api_key, cache_root, state_dir) for chunk in chunk_articles(todo, variant)]


def pending_entries(state_dir: Path) -> set[str]:
    """Записи кэша, ответы на которые ждутся в ещё не собранных пачках.

    Args:
        state_dir: Папка состояний пачек.
    """
    entries: set[str] = set()
    for path in sorted(state_dir.glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if not state.get("collected"):
            entries.update(item["entry"] for item in state["requests"].values())
    return entries


def _write_state(path: Path, state: dict) -> None:
    """Записать состояние пачки атомарно (временный файл + replace).

    Args:
        path: Файл состояния.
        state: Содержимое.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _response_from(body: dict, model_slug: str) -> ChatResponse:
    """Тело ответа одного запроса пачки → ``ChatResponse`` (как у синхронного клиента).

    Args:
        body: ``results[i].response.body`` — ответ chat/completions.
        model_slug: Слаг модели пачки (на случай, если в теле его нет).
    """
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    cost = usage.get("cost")
    return ChatResponse(
        text=message.get("content") or "",
        finish_reason=choice.get("finish_reason"),
        provider=body.get("provider"),
        model=body.get("model") or model_slug,
        request_id=body.get("id"),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        latency_s=0.0,
        attempts=1,
        raw_usage=usage,
        cached_tokens=int(prompt_details.get("cached_tokens") or 0),
    )


def _spread_cost(responses: list[ChatResponse], total_cost: float | None, model_slug: str) -> list[ChatResponse]:
    """Если у ответов нет своей цены, разложить цену пачки пропорционально токенам по справочному прайсу.

    Args:
        responses: Ответы пачки.
        total_cost: ``usage.cost`` всей пачки; ``None`` — не сообщили, раскладывать нечего.
        model_slug: Слаг модели (для весов входа и выхода).

    Returns:
        Те же ответы; у кого цены не было — с долей цены пачки.
    """
    if total_cost is None or all(response.cost_usd is not None for response in responses):
        return responses
    price_in, price_out = PRICE_PER_M.get(model_slug, (1.0, 1.0))
    weights = [response.prompt_tokens * price_in + response.completion_tokens * price_out for response in responses]
    total_weight = sum(weights) or 1.0
    result = []
    for response, weight in zip(responses, weights):
        if response.cost_usd is None:
            response = ChatResponse(**{**response.__dict__, "cost_usd": total_cost * weight / total_weight})
        result.append(response)
    return result


def collect_one(state_path: Path, api_key: str, cache: RequestCache) -> CollectStats:
    """Проверить одну пачку и, если она завершена, разложить ответы в кэш.

    Args:
        state_path: Файл состояния пачки.
        api_key: Ключ OpenRouter.
        cache: Кэш запросов.

    Returns:
        Итоги по пачке (готова / ждёт, сколько ответов сохранено).
    """
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("collected"):
        return CollectStats()
    response = httpx.get(f"{BATCHES_URL}/{state['id']}", headers=_headers(api_key), timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    batch = response.json()
    status = BatchStatus(batch["status"])
    state["status"] = status.value
    state["request_counts"] = batch.get("request_counts")
    if not status.terminal:
        _write_state(state_path, state)
        return CollectStats(batches_pending=1)
    variant = Variant(
        model=state["variant"]["model"],
        transport=Transport(state["variant"]["transport"]),
        reasoning=ReasoningLevel(state["variant"]["reasoning"]),
        prompt_version=state["variant"]["prompt_version"],
    )
    # Статьи перечитываются из выпусков: в состоянии хранятся только адреса, не тексты.
    issues: dict[str, dict[str, Article]] = {}
    articles_by_id: dict[str, Article] = {}
    for custom_id, item in state["requests"].items():
        if item["source"] not in issues:
            issues[item["source"]] = {article.article_id: article for article in split_issue(Path(item["source"]))}
        articles_by_id[custom_id] = issues[item["source"]][item["article_id"]]
    usage = batch.get("usage") or {}
    results = batch.get("results") or []
    parsed: list[tuple[str, ChatResponse]] = []
    failed = 0
    for result in results:
        custom_id = result.get("custom_id")
        item = state["requests"].get(custom_id)
        if item is None:
            continue
        reply = result.get("response") or {}
        if result.get("error") or reply.get("status_code") not in (200, None):
            failed += 1
            cache.record_error(
                Path(item["entry"]), json.dumps(result.get("error"), ensure_ascii=False), str(reply)[:300]
            )
            continue
        parsed.append((custom_id, _response_from(reply.get("body") or {}, variant.model_slug)))
    responses = _spread_cost([response for _, response in parsed], usage.get("cost"), variant.model_slug)
    for (custom_id, _), chat in zip(parsed, responses):
        article = articles_by_id[custom_id]
        entry = Path(state["requests"][custom_id]["entry"])
        verdict = CacheVerdict.TRUNCATED if chat.finish_reason == "length" else CacheVerdict.OK
        cache.store(
            entry,
            {**request_info(article, variant), "batch_id": state["id"]},
            cache_payload(article, variant),
            chat,
            verdict,
        )
    state["collected"] = True
    state["usage"] = usage
    state["collected_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if status != BatchStatus.COMPLETED:
        state["error"] = batch.get("error")
        logger.warning("пачка %s завершилась со статусом %s: %s", state["id"], status.value, batch.get("error"))
    _write_state(state_path, state)
    logger.info(
        "пачка %s: %s, ответов %d, ошибок %d, $%s", state["id"], status.value, len(parsed), failed, usage.get("cost")
    )
    return CollectStats(batches_done=1, stored=len(parsed), failed=failed, cost_usd=float(usage.get("cost") or 0.0))


def collect(state_dir: Path, api_key: str, cache: RequestCache) -> CollectStats:
    """Пройти все несобранные пачки: готовые — в кэш, остальные — обновить статус.

    Args:
        state_dir: Папка состояний пачек.
        api_key: Ключ OpenRouter.
        cache: Кэш запросов.

    Returns:
        Сумма итогов по пачкам.
    """
    stats = CollectStats()
    for path in sorted(state_dir.glob("*.json")):
        stats = stats + collect_one(path, api_key, cache)
    return stats
