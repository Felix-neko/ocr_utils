#!/usr/bin/env bash
#
# Отбор статей пака-1 о проблемах предприятий: весь пак через Batch API OpenRouter, затем сводка.
#
# Что делает: отправляет 2880 статей (по статье на запрос, промпт v3, классы 1–10) в
#   deepseek/deepseek-v4.1-flash:batch с рассуждениями low, ждёт пачки, складывает ответы в кэш и
#   собирает сводку: results.jsonl, cases.csv, by_class/class_*.md, summary.md.
# Читает: $PACK_MD_DIR (Я.Диск, только чтение).
# Пишет: $WORK_DIR/{cache,batches,runs,logs} на SSD.
# Сколько идёт: пачки Batch API — от десятков минут до 24 ч (окно OpenRouter); тестовая пачка из
#   одного запроса 2026-09-29 шла дольше 35 мин.
# Сколько стоит: ≈ $7 (пилот v2/v3 low на 55 статьях: 0,183–0,207 $ по sync-цене, пересчёт на пак по
#   объёму текста и batch-прайсу 0,112/0,336 $ за млн токенов). Бюджет — BUDGET_USD: отправка
#   отменяется, если прикидка выше.
# Качество (пилот, эталон 55 статей, классы кроме 7): точность ≈ 83 %, полнота ≈ 86 %;
#   см. reports/article_problems_pack1.md.
# Повторный запуск идемпотентен: статьи, уже лежащие в кэше или в ждущих пачках, не отправляются.

set -euo pipefail
source "$(dirname "$0")/common.sh"

REASONING="low"      # off дешевле вдвое, но на пилоте полнота 80 % против 86 %
BUDGET_USD=10
POLL_SECONDS=600

cd "$(dirname "$0")/../.."
mkdir -p "$WORK_DIR/logs"
uv run python -m research.article_problems batch-submit --all --reasoning "$REASONING" \
    --work-dir "$WORK_DIR" --pack-dir "$PACK_MD_DIR" --budget-usd "$BUDGET_USD"
uv run python -m research.article_problems batch-collect --wait --poll-seconds "$POLL_SECONDS" --work-dir "$WORK_DIR"
uv run python -m research.article_problems report --all --transport batch --reasoning "$REASONING" \
    --work-dir "$WORK_DIR" --pack-dir "$PACK_MD_DIR" --tag pack1
