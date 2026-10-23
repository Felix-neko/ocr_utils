"""Синхронный прогон статей через chat/completions с кэшем и бюджетом; чтение готовых ответов из кэша."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ocr_utils.external_ocr_services.cache import CachedResponse, CacheVerdict, RequestCache
from ocr_utils.external_ocr_services.client import OpenRouterClient, OpenRouterError
from research.article_problems.articles import Article
from research.article_problems.payload import Transport, Variant, cache_entry, cache_payload, request_info, sync_payload

logger = logging.getLogger(__name__)


class Outcome(StrEnum):
    """Чем кончилась обработка статьи в прогоне."""

    CACHED = "cached"  # ответ уже был в кэше
    DONE = "done"  # запрос ушёл, ответ сохранён
    FAILED = "failed"  # запрос не удался (сеть, 4xx/5xx после ретраев)
    OVER_BUDGET = "over_budget"  # не отправлен: бюджет прогона исчерпан


@dataclass(frozen=True)
class RunStats:
    """Итоги прогона: сколько статей чем кончились, токены и деньги новых запросов.

    Args:
        cached: Статей, взятых из кэша.
        done: Статей, по которым пришёл новый ответ.
        failed: Статей с ошибкой запроса.
        over_budget: Статей, не отправленных из-за бюджета.
        cost_usd: Сумма ``usage.cost`` новых ответов.
        prompt_tokens: Токенов входа новых ответов.
        completion_tokens: Токенов выхода новых ответов (с рассуждениями).
    """

    cached: int = 0
    done: int = 0
    failed: int = 0
    over_budget: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: RunStats) -> RunStats:
        """Сложить итоги поэлементно (счётчики складываются у вызывающего)."""
        return RunStats(
            *(getattr(self, name) + getattr(other, name) for name in self.__dataclass_fields__)  # type: ignore[arg-type]
        )


class Budget:
    """Потокобезопасный счётчик потраченного: новые запросы не уходят, когда сумма дошла до потолка.

    Args:
        limit_usd: Потолок трат прогона, $.
    """

    def __init__(self, limit_usd: float):
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()

    def exhausted(self) -> bool:
        """Дошли ли траты до потолка."""
        with self._lock:
            return self.spent_usd >= self.limit_usd

    def add(self, cost_usd: float | None) -> None:
        """Учесть цену ответа (``None`` — провайдер не сообщил, не учитывается).

        Args:
            cost_usd: Цена ответа, $.
        """
        with self._lock:
            self.spent_usd += cost_usd or 0.0


def lookup(cache: RequestCache, cache_root: Path, article: Article, variant: Variant) -> CachedResponse | None:
    """Ответ по статье и варианту из кэша.

    Args:
        cache: Кэш запросов.
        cache_root: Его корень.
        article: Статья.
        variant: Вариант прогона.

    Returns:
        Запись кэша или ``None``, если статью этим вариантом ещё не спрашивали.
    """
    return cache.lookup(cache_entry(cache_root, article, variant))


def process_article(
    article: Article, variant: Variant, client: OpenRouterClient, cache: RequestCache, cache_root: Path, budget: Budget
) -> RunStats:
    """Одна статья синхронно: из кэша, иначе запрос и запись ответа в кэш.

    Args:
        article: Статья.
        variant: Вариант прогона (транспорт ``SYNC``).
        client: Клиент OpenRouter.
        cache: Кэш запросов.
        cache_root: Корень кэша.
        budget: Общий на прогон счётчик трат.

    Returns:
        Итоги по одной статье (одно из полей-счётчиков = 1, плюс токены и цена нового ответа).
    """
    entry = cache_entry(cache_root, article, variant)
    if cache.lookup(entry) is not None:
        return RunStats(cached=1)
    if budget.exhausted():
        return RunStats(over_budget=1)
    payload = sync_payload(article, variant)
    try:
        response = client.chat(payload)
    except OpenRouterError as error:
        logger.error("%s: запрос не удался: %s", article.key, error)
        cache.record_error(entry, str(error), error.body)
        return RunStats(failed=1)
    budget.add(response.cost_usd)
    # Обрезанный ответ сохраняется с пометкой: повтор с тем же payload дал бы то же самое.
    verdict = CacheVerdict.TRUNCATED if response.finish_reason == "length" else CacheVerdict.OK
    cache.store(entry, request_info(article, variant), payload, response, verdict)
    logger.info(
        "%s: %d+%d ток., $%.5f, %.0f с, %s",
        article.key,
        response.prompt_tokens,
        response.completion_tokens,
        response.cost_usd or 0.0,
        response.latency_s,
        response.provider,
    )
    return RunStats(
        done=1,
        cost_usd=response.cost_usd or 0.0,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )


def run_sync(
    articles: list[Article],
    variant: Variant,
    client: OpenRouterClient,
    cache: RequestCache,
    cache_root: Path,
    jobs: int,
    budget_usd: float,
) -> RunStats:
    """Прогнать статьи синхронно пулом потоков (задача упирается в сеть, а не в CPU).

    Args:
        articles: Статьи прогона.
        variant: Вариант (транспорт ``SYNC``).
        client: Клиент OpenRouter (потокобезопасен: httpx внутри SDK).
        cache: Кэш запросов.
        cache_root: Корень кэша.
        jobs: Одновременных запросов.
        budget_usd: Потолок трат прогона, $.

    Returns:
        Сумма итогов по статьям.
    """
    if variant.transport != Transport.SYNC:
        raise ValueError("run_sync — только для синхронного варианта; batch идёт через batch.submit")
    budget = Budget(budget_usd)
    stats = RunStats()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(process_article, article, variant, client, cache, cache_root, budget) for article in articles
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            stats = stats + future.result()
            if index % 50 == 0:
                logger.info("%d/%d статей, потрачено $%.4f", index, len(articles), stats.cost_usd)
    return stats
