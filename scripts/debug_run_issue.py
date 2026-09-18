"""Отладочный запуск одного выпуска через ``run_issue`` без CLI — для пошагового разбора OCR в PyCharm.

Все параметры прогона собираются здесь, в коде, в том же виде, в каком их собирает
``python -m ocr_utils.external_ocr_services run``: флаги оглавления — из базы разметки (только
чтение), настройки запроса — ``RunOptions``, параметры прогона — ``PipelineParams``. Ставьте точку
останова в ``run_issue`` / ``recognise_page`` / ``structure.apply`` и запускайте файл из IDE.

Ключ OpenRouter — из ``$OPENROUTER_API_KEY``. Выход пишется в отдельную папку ``out_debug``,
чтобы не трогать боевой ``out``. Набор полос: все полосы оглавления/указателя выпуска (иначе
этап ``page`` пойдёт без списка статей) плюс первые ``REGULAR_PAGES`` обычных полос.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ocr_utils.external_ocr_services import models as registry
from ocr_utils.external_ocr_services.client import OpenRouterClient, api_key_from
from ocr_utils.external_ocr_services.ocr import RunOptions
from ocr_utils.external_ocr_services.pages import flags_for, flags_from_db, list_pages
from ocr_utils.external_ocr_services.pipeline import PipelineParams, run_issue

# --- Что распознаём (пути и числа те же, что в run_scripts/external_ocr_services/common.sh) ---
YEAR, ISSUE = "1976", "12"
IN_DIR = Path("/mnt/SYSTEM/raw/mts/pack1_background_blurred_v2/sharpened")
DB_PATH = Path("/home/felix/Projects/mts_markup/pack1_reviewed.sqlite")
PACK_NAME = "пак-1"
ROOT = Path("/mnt/SYSTEM/raw/mts/pack1_external_ocr_services")
OUT_DIR = ROOT / "out_debug"
DEBUG_DIR = ROOT / "debug_debug"  # промпты, тайлы и сырые ответы по полосе
SOURCE = "журнал «Материально-техническое снабжение», Москва, 1966–1976"
MODEL = registry.DEFAULT_MODEL
REGULAR_PAGES: int | None = 2  # обычных полос помимо оглавления; None — весь выпуск (99 полос, ~$0.10)
JOBS = 1  # один поток: точки останова срабатывают по очереди, лог читается линейно


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    spec = registry.resolve(MODEL)
    # Флаги оглавления/указателя и вето из базы разметки; база открывается только на чтение.
    flags = flags_from_db(DB_PATH, PACK_NAME)
    options = RunOptions(source=SOURCE, debug_dir=DEBUG_DIR)  # тайлы, потолки и второй проход — по умолчанию
    params = PipelineParams(
        in_dir=IN_DIR,
        out_dir=OUT_DIR,
        options=options,
        flags=flags,
        jobs=JOBS,
        skip_done=False,  # True — не запрашивать полосы, у которых выход уже есть
        on_missed_toc="skip",  # без повтора выпуска: для отладки один круг понятнее
        only_year=YEAR,
        only_issue=ISSUE,
    )
    # То, что run_pipeline делает перед циклом по выпускам: отбор полос выпуска. В 1976/12 это
    # 2 полосы «Содержания» и 6 полос годового указателя; их берём все, обычных — первые REGULAR_PAGES.
    all_pages = list_pages(IN_DIR, None, YEAR, ISSUE)
    toc_pages = [rel for rel in all_pages if flags_for(rel, flags).toc_kind is not None]
    regular = [rel for rel in all_pages if rel not in toc_pages][:REGULAR_PAGES]
    pages = sorted(toc_pages + regular)  # run_issue ждёт порядок по имени: по нему сливается оглавление
    issue_key = f"{YEAR}/{ISSUE}"
    client = OpenRouterClient(api_key_from(None))

    stats = run_issue(client, spec, params, issue_key, pages)  # <- точка останова здесь

    print(
        f"полос {len(pages)}, запросов {stats.requests}, сбоев {stats.failed}, "
        f"второй проход {stats.second_passes}, стоимость ${stats.cost_usd:.4f}; выход {OUT_DIR / issue_key}"
    )
    if stats.missed:
        print("оглавления вне базы:", "; ".join(stats.missed))


if __name__ == "__main__":
    main()
