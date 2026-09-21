"""Отладочный прогон «плохого выпуска» через ``run_issue`` без CLI — понижение, fallback и частичный повтор глазами.

Второй сценарий рядом с ``scripts/debug_run_issue.py``: один выпуск, но нарочно плохой —
31 полоса МТС 1991/02 из фото-разворотов (``scripts/cut_damaged_issue.py``), с. 92 с размазанной
печатью (второй проход по умолчанию выключен с v16; ``SECOND_PASS`` ниже включает) и **намеренно неверные теги оглавления**, как будто в CVAT тег
поставлен на соседней полосе:

* ``IMG_0624_R`` — первая полоса «Содержания», помечена верно;
* ``IMG_0625_L`` — вторая полоса «Содержания» (продолжение + редколлегия), **не помечена** — её
  должен найти fallback на этапе page (``toc_kind=contents`` → круг повтора);
* ``IMG_0625_R`` — с. 3, начало статьи, помечена оглавлением **ошибочно** — модель должна её
  понизить (``demoted_toc.txt``, ``.toc.json``), и она пойдёт этапом page.

На круге повтора по ``redo_scope=STRUCTURED`` заново должны уйти только понижённая полоса и полосы
с заголовками/рубриками/авторами (с. 3, 82, 84, 86, 89, 91, 93, 94, реклама), а сплошной текст
(с. 4–9, 85, 87–88, 90, 95–100) — остаться с пометкой ``redo_kept`` в meta.

Без аргументов командной строки: всё задано ниже, чтобы запускать из PyCharm в режиме Debug.
Ключ OpenRouter — из ``$OPENROUTER_API_KEY``. Выход — в ``scripts/debug_bad_issue_out/``
(вне git): ``out/`` — .json/.md/.meta.json по полосам, ``toc.json``/``toc.md``, ``demoted_toc.txt``;
``debug/`` — промпты, тайлы и сырые ответы. Стоимость круга: ~31 + ~12 полос, около $0.05–0.10.

``VARIANT`` выбирает вход: ``None`` — полосы, нарезанные по сгибу из разворотов (``1991/02``, с чёрными
полями и куском соседней страницы); ``"cut"`` — те же полосы, откадрированные руками по краю
бумаги (``1991_cut/02``). У варианта свои папки ``out_<вариант>`` / ``debug_<вариант>``, чтобы
сравнивать прогоны, ничего не затирая. Года в промпте нет, поэтому «1991_cut» в ключе выпуска безвреден.

``NEW_PAGES`` — режим «только новые полосы»: в выпуск добавили кадры, а гонять всё заново незачем.
Оглавление берётся из старого прогона: файлы полос оглавления (и понижённой полосы) копируются из
``out_<вариант>`` в ``out_new_<вариант>``, найденная fallback'ом полоса помечается оглавлением уже
«в базе» (как после правки тегов в CVAT), и ``run_issue`` со ``skip_done`` берёт их с диска, а в
сеть отправляет только новые полосы — с теми же списками рубрик и статей в промпте.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

# Пакет не установлен в окружение: при запуске файлом (терминал, PyCharm без content root) корень репо — руками.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_utils.external_ocr_services import models as registry
from ocr_utils.external_ocr_services.client import OpenRouterClient, api_key_from
from ocr_utils.external_ocr_services.ocr import RunOptions
from ocr_utils.external_ocr_services.pages import PageFlags, list_pages, page_key
from ocr_utils.external_ocr_services.pipeline import OnMissedToc, PipelineParams, RedoScope, run_issue

REPO = Path(__file__).resolve().parents[1]
VARIANT: str | None = "cut"  # None — нарезка по сгибу (1991/02); "cut" — ручное кадрирование (1991_cut/02)
YEAR, ISSUE = ("1991" if VARIANT is None else f"1991_{VARIANT}"), "02"
IN_DIR = REPO / "research" / "external_ocr_models" / "damaged_issue"  # {год}/{выпуск}/IMG_xxxx_{L|R}.jpg
ROOT = Path(__file__).resolve().with_name("debug_bad_issue_out")
# Метка прогона: с ней папки выхода получают суффикс ``_<метка>`` (``out_v15_cut``), чтобы сравнивать
# версии промпта, ничего не затирая; None — без метки.
RUN_TAG: str | None = "v15"
SUFFIX = ("" if VARIANT is None else f"_{VARIANT}") + ("" if RUN_TAG is None else f"_{RUN_TAG}")
OUT_DIR = ROOT / f"out{SUFFIX}"
DEBUG_DIR = ROOT / f"debug{SUFFIX}"
SOURCE = "журнал «Материально-техническое снабжение», Москва, 1991"
MODEL = registry.DEFAULT_MODEL
JOBS = 1  # один поток: точки останова срабатывают по очереди, лог читается линейно
SKIP_DONE = False  # True — не запрашивать полосы, у которых выход уже есть (догнать сбойные)
SECOND_PASS = False  # True — второй проход по полосам с is_damaged (в пакете выключен с v16, см. README)

# Теги «из базы» руками: верный тег на первой полосе оглавления, пропуск на второй, ложный на третьей.
TOC_TAGGED = ("IMG_0624_R", "IMG_0625_R")
TOC_MISSED = ("IMG_0625_L",)  # для пояснений в логе; в флагах эта полоса обычная

# Только новые полосы (см. докстринг модуля); None — обычный прогон всего выпуска.
NEW_PAGES: tuple[str, ...] | None = (
    "IMG_0622_R",  # обложка
    "IMG_0625_R",  # эмблема-треугольник
    "IMG_0648_L",  # логотип «Маркетинг»
    "IMG_0668_L",  # пиктограмма-конверт
    "IMG_0673_R",  # пиктограмма-конверт
    "IMG_0674_L",  # щит
    "IMG_0677_R",  # реклама с логотипом и кривыми
)
# Откуда брать готовое оглавление в режиме NEW_PAGES: выход полного прогона того же варианта (без метки).
PREVIOUS_OUT_DIR = ROOT / ("out" if VARIANT is None else f"out_{VARIANT}")
# Файлы полосы, которые нужны, чтобы is_done счёл её готовой (у понижённой — ещё и .toc.json).
PAGE_SUFFIXES = (".json", ".md", ".meta.json", ".toc.json")


def build_flags(pages: list[Path], tagged: tuple[str, ...]) -> dict[str, PageFlags]:
    """Флаги полос выпуска как их отдала бы база: все известны, оглавление — только ``tagged``.

    Args:
        pages: Относительные пути всех полос выпуска.
        tagged: Имена полос (без суффикса), помеченных оглавлением.

    Returns:
        ``{page_key: PageFlags}`` — тот же вид, что у ``flags_from_db``.
    """
    flags = {page_key(rel): PageFlags() for rel in pages}
    for stem in tagged:
        flags[page_key(Path(YEAR) / ISSUE / stem)] = PageFlags(is_toc=True)
    return flags


def copy_toc_pages(previous: Path, target: Path, stems: tuple[str, ...]) -> list[Path]:
    """Перенести готовые файлы полос оглавления из старого прогона в новый out-dir.

    Копируются только файлы полос ``stems`` (.json/.md/.meta.json/.toc.json, какие есть) плюс
    ``toc.json``/``toc.md`` выпуска; остальные полосы старого прогона не трогаются.

    Args:
        previous: Корень выхода старого прогона.
        target: Корень выхода нового прогона (создаётся).
        stems: Имена полос без суффикса.

    Returns:
        Скопированные файлы; пусто — старого прогона нет (тогда оглавление придётся распознать).
    """
    src, dst = previous / YEAR / ISSUE, target / YEAR / ISSUE
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in [f"{stem}{suffix}" for stem in stems for suffix in PAGE_SUFFIXES] + ["toc.json", "toc.md"]:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
            copied.append(dst / name)
    return copied


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    spec = registry.resolve(MODEL)
    log = logging.getLogger(__name__)
    pages = list_pages(IN_DIR, None, YEAR, ISSUE)
    if not pages:
        raise SystemExit(f"нет полос в {IN_DIR / YEAR / ISSUE}: сначала scripts/cut_damaged_issue.py")
    out_dir, debug_dir, tagged, skip_done, on_missed = OUT_DIR, DEBUG_DIR, TOC_TAGGED, SKIP_DONE, OnMissedToc.REDO
    if NEW_PAGES is not None:
        # Только новые полосы: свои папки, оглавление — из старого прогона, найденная полоса теперь «в базе»,
        # повтора выпуска не нужно. skip_done берёт скопированные полосы с диска, новые идут в сеть.
        out_dir, debug_dir = ROOT / f"out_new{SUFFIX}", ROOT / f"debug_new{SUFFIX}"
        tagged = tuple(sorted(TOC_TAGGED + TOC_MISSED))
        copied = copy_toc_pages(PREVIOUS_OUT_DIR, out_dir, tagged)
        if not copied:
            raise SystemExit(f"нет старого прогона в {PREVIOUS_OUT_DIR}: сначала полный прогон (NEW_PAGES = None)")
        log.info("оглавление из %s: скопировано файлов %d", PREVIOUS_OUT_DIR, len(copied))
        wanted = set(tagged) | set(NEW_PAGES)
        pages = [rel for rel in pages if rel.stem in wanted]
        skip_done, on_missed = True, OnMissedToc.SKIP
    flags = build_flags(pages, tagged)
    # Тайлы и потолки — по умолчанию; второй проход — по константе (с v16 в пакете выключен).
    options = RunOptions(source=SOURCE, debug_dir=debug_dir, second_pass=SECOND_PASS)
    params = PipelineParams(
        in_dir=IN_DIR,
        pages_dir=out_dir,
        issues_dir=out_dir.with_name(out_dir.name + "_issues"),  # md выпусков — отдельно от полос
        options=options,
        flags=flags,
        jobs=JOBS,
        skip_done=skip_done,
        on_missed_toc=on_missed,  # найденное оглавление → круг повтора: это и проверяем
        redo_scope=RedoScope.STRUCTURED,  # на повторе заново — только чувствительные к спискам полосы
        only_year=YEAR,
        only_issue=ISSUE,
    )
    issue_key = f"{YEAR}/{ISSUE}"
    client = OpenRouterClient(api_key_from(None))
    log.info("полос %d; в «базе» оглавление: %s; пропущено нарочно: %s", len(pages), tagged, TOC_MISSED)

    stats = run_issue(client, spec, params, issue_key, pages)  # <- точка останова здесь

    print(
        f"полос {len(pages)}, запросов {stats.requests}, с диска {stats.reused}, сбоев {stats.failed}, "
        f"второй проход {stats.second_passes}, оставлено без повтора {stats.redo_kept}, "
        f"стоимость ${stats.cost_usd:.4f}; выход {out_dir / issue_key}"
    )
    print("понижены (в «базе» оглавление, модель — нет):", "; ".join(stats.demoted_toc) or "—")
    print("оглавления вне «базы»:", "; ".join(stats.missed) or "—")
    print("круг повтора:", ", ".join(stats.redone_issues) or "не было")
    toc_md = out_dir / issue_key / "toc.md"
    if toc_md.is_file():
        print(f"\n--- {toc_md} ---\n{toc_md.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
