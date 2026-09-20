"""Повторы распознавания полос боевым пакетом external_ocr_services под A/B промптов: N прогонов, сводка по повторам.

Зачем: разметка повреждений у DeepSeek V4.1 Flash плавает между одинаковыми запросами (IMG_0008_R:
16 против 64 ``<supplied>`` при том же промпте), поэтому сравнивать версии промпта по одному прогону
нельзя. Раньше такие повторы собирались в scratchpad-скриптах и терялись вместе с сессией (а один
раз ушли впустую $0.30, потому что руками собранный payload не выключал ``reasoning``). Здесь
запрос идёт через ``ocr.recognize_page`` — тот же код, что в прогоне: тайлы, цепочка режимов
JSON, ``reasoning.effort none``, разбор, теги из ``edge_words``, доводка структуры — только без
второго прохода; оглавление выпуска (список статей и рубрик с id) уходит в промпт, если задан
``--toc-root`` — корень выхода боевого прогона с ``{год}/{выпуск}/toc.json`` (v20: по нему в
сводке видно, сколько `#` получили id от модели, а сколько код присвоил по названию).

Команды::

    # 2 повтора мини-набора повреждённых полос текущими промптами (PROMPT_VERSION пакета)
    uv run python scripts/replay_page.py run \\
        --in-dir "research/external_ocr_models/damaged/нарезанное по страницам" \\
        --out-dir /mnt/system/raw/mts/replay/damaged_v16 --repeats 2

    # то же промптами другой версии: папка с system.md.j2, user.md.j2 и, если нужно, damage_note.txt
    uv run python scripts/replay_page.py run ... --out-dir .../damaged_v15 --prompts-dir /tmp/prompts_v15

    # полосы пака со списком статей выпуска в промпте (toc.json из выхода боевого прогона)
    uv run python scripts/replay_page.py run --in-dir .../sharpened --pages полосы.txt \
        --toc-root /mnt/system/raw/mts/pack1_external_ocr_services/out --out-dir .../headings_v20 --repeats 2

    # сводка по повторам (и сходство с другим прогоном тех же полос)
    uv run python scripts/replay_page.py summarize --out-dir .../damaged_v16 [--baseline-dir .../damaged_v15]

Выход ``run``: ``<out-dir>/run1/``, ``run2/`` … с той же раскладкой, что у прогона
(``.json``/``.md``/``.meta.json`` по полосе); сырые ответы, промпты и тайлы первого повтора — в
``<out-dir>/debug/``. Сводка ``summarize`` печатается в терминал и пишется в ``<out-dir>/summary.md``.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import click

# Пакет не установлен в окружение: при запуске файлом корень репо — руками.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_utils.experimental import damage_hints, strips  # noqa: E402
from ocr_utils.external_ocr_services import PROMPT_VERSION, prompts  # noqa: E402
from ocr_utils.external_ocr_services import models as registry  # noqa: E402
from ocr_utils.external_ocr_services import ocr as ocr_module  # noqa: E402
from ocr_utils.external_ocr_services.client import OpenRouterClient, api_key_from  # noqa: E402
from ocr_utils.external_ocr_services.ocr import PageJob, RunOptions, recognize_page  # noqa: E402
from ocr_utils.external_ocr_services.pages import list_pages  # noqa: E402
from ocr_utils.external_ocr_services.schema import DamageTag, TocKind  # noqa: E402
from ocr_utils.external_ocr_services.toc import from_dict as toc_from_dict, prompt_lists, toc_hash  # noqa: E402
from ocr_utils.external_ocr_services.tiling import DEFAULT_MAX_MODEL_TILE, DEFAULT_MAX_SRC_TILE  # noqa: E402

logger = logging.getLogger("replay_page")

# Теги повреждений и структуры вместе с содержимым <gap> — снимаются перед сравнением текстов.
_TAGS = re.compile(r"<gap>.*?</gap>|</?[a-z-]+>", re.S)
# Дефис между двумя кириллическими буквами: настоящие составные слова + разорванные переносы.
_INNER_HYPHEN = re.compile(r"[а-яё]-[а-яё]", re.IGNORECASE)
_UNREADABLE = "[неразборчиво]"


def use_prompts_dir(prompts_dir: Path) -> None:
    """Подменить шаблоны промптов пакета папкой ``prompts_dir`` (A/B версий без правки репо).

    В папке ожидаются ``system.md.j2`` и ``user.md.j2``; если рядом лежит ``damage_note.txt``,
    его текст заменяет ``DEFAULT_DAMAGE_NOTE`` (фраза о повреждениях живёт не в шаблоне, а в
    ``prompts/__init__.py``). Старую версию проще всего достать из git:
    ``git show <коммит>:ocr_utils/external_ocr_services/prompts/system.md.j2 > /tmp/p/system.md.j2``.

    Args:
        prompts_dir: Папка с шаблонами.
    """
    for name in ("system.md.j2", "user.md.j2"):
        if not (prompts_dir / name).is_file():
            raise click.ClickException(f"в {prompts_dir} нет {name}")
    prompts.PROMPTS_DIR = prompts_dir
    prompts._environment.cache_clear()  # окружение Jinja кэшировано с прежним loader'ом
    note = prompts_dir / "damage_note.txt"
    if note.is_file():
        prompts.DEFAULT_DAMAGE_NOTE = note.read_text(encoding="utf-8").strip()
    logger.info("промпты из %s%s", prompts_dir, " (+ damage_note.txt)" if note.is_file() else "")


def issue_job_lists(toc_root: Path | None, rel: Path, cache: dict[str, tuple]) -> tuple:
    """Рубрики, статьи и отпечаток списков выпуска полосы из ``toc.json`` боевого прогона.

    Args:
        toc_root: Корень выхода боевого прогона (``{год}/{выпуск}/toc.json``); ``None`` — списков нет.
        rel: Полоса относительно входа: ``{год}/{выпуск}/имя``.
        cache: ``{«год/выпуск»: (рубрики, статьи, отпечаток)}`` — toc.json читается один раз на выпуск.

    Returns:
        ``(рубрики, статьи, отпечаток)`` как для ``PageJob``; пустые, если toc.json нет.
    """
    key = rel.parent.as_posix()
    if toc_root is None or len(rel.parts) < 3:
        return (), (), ""
    if key not in cache:
        path = toc_root / key / "toc.json"
        if path.is_file():
            tocs = toc_from_dict(json.loads(path.read_text(encoding="utf-8")))
            rubrics, articles = prompt_lists(tocs.get(TocKind.CONTENTS))
            cache[key] = (tuple(rubrics), tuple(articles), toc_hash(rubrics, articles))
            logger.info("%s: в промпт уходят %d статей и %d рубрик из %s", key, len(articles), len(rubrics), path)
        else:
            logger.warning("%s: нет %s — полосы выпуска идут без списка", key, path)
            cache[key] = ((), (), "")
    return cache[key]


def _recognize_one(
    client: OpenRouterClient,
    spec,
    in_dir: Path,
    out_dir: Path,
    options: RunOptions,
    joiner,
    toc_root: Path | None,
    toc_cache: dict[str, tuple],
    rel: Path,
) -> tuple[Path, dict]:
    """Одна полоса одним проходом — для пула потоков.

    Args:
        client: Клиент OpenRouter, общий на прогон.
        spec: Модель из реестра.
        in_dir: Корень входа.
        out_dir: Корень выхода этого повтора.
        options: Настройки запроса.
        joiner: ``(Morph, JoinRule)`` для склейки переносов после разбора или ``None``.
        toc_root: Корень с ``toc.json`` выпусков (списки в промпт) или ``None``.
        toc_cache: Кэш списков по выпускам (общий на прогон).
        rel: Полоса относительно ``in_dir``.

    Returns:
        ``(rel, meta)`` — meta как записана в ``.meta.json`` (при склейке — плюс ``hyphens_joined``).
    """
    rubrics, articles, digest = issue_job_lists(toc_root, rel, toc_cache)
    job = PageJob(rel, rubrics=rubrics, articles=articles, toc_hash=digest)
    meta, result = recognize_page(client, spec, in_dir / rel, job, out_dir, options)
    if joiner is not None and result is not None:
        # Склейка — поверх готового выхода: .json/.md перезаписываются, число склеек — в meta.
        from ocr_utils.external_ocr_services.hyphen_join import join_broken_hyphens  # noqa: PLC0415

        result.content_markdown, report = join_broken_hyphens(result.content_markdown, *joiner)
        meta = dict(meta, hyphens_joined=len(report.joined), hyphens_kept=len(report.kept))
        ocr_module._write_outputs(out_dir, rel, result, meta)
    return rel, meta


def use_strips(rows: int) -> None:
    """Подменить нарезку полосы в боевом коде на горизонтальные полосы 1×``rows`` (``experimental.strips``)."""
    ocr_module.prepare_tiles = partial(_strips_like_tiles, rows)
    logger.info("тайлы: %d горизонтальных полос вместо сетки", rows)


def _strips_like_tiles(rows: int, path: Path, max_src_tile: int, max_model_tile: int, quality: int):
    """Сигнатура ``tiling.prepare_tiles`` → ``strips.prepare_strips`` (шаг сетки игнорируется)."""
    return strips.prepare_strips(path, rows, max_model_tile, quality)


def use_hints(csv_path: Path) -> damage_hints.HintedPrompts:
    """Подменить сборку промптов: фраза о повреждениях — из CSV детектора корешка по имени кадра."""
    hinted = damage_hints.HintedPrompts(damage_hints.hints_from_gutter_csv(csv_path))
    ocr_module.prompts_for = hinted
    logger.info("подсказки о повреждениях: %d страниц из %s", len(hinted.hints), csv_path)
    return hinted


@click.group()
def main() -> None:
    """Повторы полос под A/B промптов и сводка по ним."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command("run")
@click.option("--in-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--pages", "pages_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--only-year", default=None)
@click.option("--only-issue", default=None)
@click.option("--limit", type=int, default=None)
@click.option("--repeats", type=int, default=2, show_default=True, help="Сколько раз распознать каждую полосу.")
@click.option(
    "--prompts-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Папка с system.md.j2, user.md.j2 [, damage_note.txt] вместо шаблонов пакета.",
)
@click.option("--model", "model_name", default=registry.DEFAULT_MODEL, show_default=True)
@click.option("--source", default="", help="Описание издания в системный промпт.")
@click.option("--max-src-tile-size", type=int, default=DEFAULT_MAX_SRC_TILE, show_default=True)
@click.option("--max-model-tile-size", type=int, default=DEFAULT_MAX_MODEL_TILE, show_default=True)
@click.option("--jobs", type=int, default=4, show_default=True, help="Параллельных запросов (сеть, не CPU).")
@click.option("--api-key", default=None, help="Ключ OpenRouter; по умолчанию $OPENROUTER_API_KEY.")
@click.option("--rows", type=int, default=None, help="Горизонтальные полосы 1×N вместо сетки (experimental.strips).")
@click.option(
    "--hints-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="CSV детектора корешка: подсказка о повреждении на страницу вместо нейтральной фразы.",
)
@click.option(
    "--join-hyphens",
    default=None,
    help="Дополнительная склейка переносов другим бэкендом/правилом: БЭКЕНД:ПРАВИЛО (боевая pymorphy3:E идёт всегда, --no-join нет).",
)
@click.option(
    "--toc-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Корень выхода боевого прогона с {год}/{выпуск}/toc.json: список статей и рубрик выпуска с id уходит в промпт.",
)
def run_command(
    in_dir: Path,
    out_dir: Path,
    pages_file: Path | None,
    only_year: str | None,
    only_issue: str | None,
    limit: int | None,
    repeats: int,
    prompts_dir: Path | None,
    model_name: str,
    source: str,
    max_src_tile_size: int,
    max_model_tile_size: int,
    jobs: int,
    api_key: str | None,
    rows: int | None,
    hints_csv: Path | None,
    join_hyphens: str | None,
    toc_root: Path | None,
) -> None:
    """Распознать полосы ``--repeats`` раз каждую; повтор i пишется в ``<out-dir>/run<i>/``."""
    if prompts_dir is not None:
        use_prompts_dir(prompts_dir)
    if rows is not None:
        use_strips(rows)
    hinted = use_hints(hints_csv) if hints_csv is not None else None
    joiner = None
    if join_hyphens is not None:
        from ocr_utils.external_ocr_services.hyphen_join import JoinRule, Morph  # noqa: PLC0415

        backend, _, rule = join_hyphens.partition(":")
        joiner = (Morph(backend), JoinRule(rule or "E"))
    spec = registry.resolve(model_name)
    rels = list_pages(in_dir, pages_file, only_year, only_issue, limit)
    if not rels:
        raise click.ClickException(f"в {in_dir} нет полос")
    client = OpenRouterClient(api_key_from(api_key))
    logger.info("полос %d × повторов %d, модель %s, промпт v%d", len(rels), repeats, spec.name, PROMPT_VERSION)
    total_cost = 0.0
    toc_cache: dict[str, tuple] = {}  # списки выпусков — один раз на прогон
    for index in range(1, repeats + 1):
        run_dir = out_dir / f"run{index}"
        # Сырые ответы, промпты и тайлы — только у первого повтора: тайлы весят как вход.
        options = RunOptions(
            max_src_tile=max_src_tile_size,
            max_model_tile=max_model_tile_size,
            source=source,
            debug_dir=out_dir / "debug" if index == 1 else None,
            second_pass=False,
        )
        work = partial(_recognize_one, client, spec, in_dir, run_dir, options, joiner, toc_root, toc_cache)
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for rel, meta in pool.map(work, rels):
                total_cost += float(meta.get("cost_usd") or 0.0)
                status = meta.get("error") or meta.get("parse_error") or "ok"
                logger.info(
                    "run%d %s: %s, теги %s, is_damaged=%s", index, rel, status, meta.get("tags"), meta.get("is_damaged")
                )
    (out_dir / "prompt_version.txt").write_text(
        f"{PROMPT_VERSION}\n{prompts_dir or 'шаблоны пакета'}\nrows={rows or 'сетка'} hints={hints_csv or '-'} "
        f"join={join_hyphens or '-'} toc_root={toc_root or '-'}\n",
        encoding="utf-8",
    )
    if hinted is not None:
        logger.info("подсказка нашлась для %d запросов", hinted.used)
    click.echo(f"Готово: {len(rels)} полос × {repeats}, ${total_cost:.4f}; сводка — summarize --out-dir {out_dir}")


@dataclass(frozen=True)
class PageMetrics:
    """Показатели одного ответа по полосе — то, по чему сравниваются версии промпта."""

    is_damaged: bool | None
    supplied: int
    unclear: int
    gap: int
    edge_words: int
    edge_words_empty: int
    from_edge_words: int  # тегов дослал код из edge_words
    hyphen_stripped: int  # снято <supplied> с продолжений переносов
    duplicates_dropped: int
    unreadable: int  # «[неразборчиво]» в теле
    inner_hyphens: int  # дефисов между буквами (составные слова + разорванные переносы)
    chars: int
    cost_usd: float
    error: str | None
    text: str  # тело без тегов — для сходства между повторами
    headings: int = 0  # `#` на полосе после доводки
    ids_model: int = 0  # из них с id статьи от модели (v20, согласован с текстом)
    ids_title: int = 0  # с id по названию (модель id не дала или дала чужой)
    ids_wrong: int = 0  # id модели противоречил тексту
    rubric_tags: int = 0  # `<rubric>` на полосе
    rubric_ids_model: int = 0  # из них с id от модели
    header_ids: int = 0  # сколько id (статьи/рубрики) модель дала колонтитулам (0–4)

    def cell(self) -> str:
        """Ячейка таблицы: «s/u/g · ew(пусто) · нераз · дефисы · знаков · # ids · rub · hdr» или причина сбоя."""
        if self.error:
            return f"сбой: {self.error[:40]}"
        damaged = "D" if self.is_damaged else "-"
        return (
            f"{damaged} {self.supplied}/{self.unclear}/{self.gap} · ew {self.edge_words}({self.edge_words_empty})"
            f"+{self.from_edge_words} · нр {self.unreadable} · деф {self.inner_hyphens} · {self.chars}"
            f" · # {self.headings} id {self.ids_model}/{self.ids_title}/{self.ids_wrong}"
            f" · rub {self.rubric_tags}/{self.rubric_ids_model} · hdr {self.header_ids}"
        )


def read_page(base: Path) -> PageMetrics:
    """Показатели полосы по её ``.json`` и ``.meta.json``.

    Args:
        base: Путь полосы без суффикса (``…/IMG_0006_L``).

    Returns:
        Заполненный ``PageMetrics``; при сбое — с ``error`` и нулями.
    """
    meta = json.loads(base.with_suffix(".meta.json").read_text(encoding="utf-8"))
    error = meta.get("error") or meta.get("parse_error")
    json_path = base.with_suffix(".json")
    if error or not json_path.is_file():
        return PageMetrics(
            None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(meta.get("cost_usd") or 0), error or "нет .json", ""
        )
    result = json.loads(json_path.read_text(encoding="utf-8"))
    body = result.get("content_markdown") or ""
    tags = meta.get("tags") or {}
    return PageMetrics(
        is_damaged=result.get("is_damaged", result.get("damaged")),
        supplied=int(tags.get(DamageTag.SUPPLIED.value, 0)),
        unclear=int(tags.get(DamageTag.UNCLEAR.value, 0)),
        gap=int(tags.get(DamageTag.GAP.value, 0)),
        edge_words=len(result.get("edge_words") or []),
        edge_words_empty=int(meta.get("edge_words_empty") or 0),
        from_edge_words=int(meta.get("tags_from_edge_words") or 0),
        hyphen_stripped=int(meta.get("hyphen_supplied_stripped") or 0),
        duplicates_dropped=int(meta.get("duplicate_supplied_dropped") or 0),
        unreadable=body.count(_UNREADABLE),
        inner_hyphens=len(_INNER_HYPHEN.findall(body)),
        chars=len(body),
        cost_usd=float(meta.get("cost_usd") or 0.0),
        error=None,
        text=_TAGS.sub("", body),
        headings=len(result.get("headings") or []),
        ids_model=int(((meta.get("structure") or {}).get("heading_ids") or {}).get("model") or 0),
        ids_title=int(((meta.get("structure") or {}).get("heading_ids") or {}).get("title") or 0),
        ids_wrong=int(((meta.get("structure") or {}).get("heading_ids") or {}).get("wrong") or 0),
        rubric_tags=len(result.get("rubrics") or []),
        rubric_ids_model=int(((meta.get("structure") or {}).get("rubric_ids") or {}).get("model") or 0),
        header_ids=sum(
            1
            for key in (
                "running_header_article_id",
                "running_header_rubric_id",
                "running_footer_article_id",
                "running_footer_rubric_id",
            )
            if result.get(key)
        ),
    )


def reference_texts(reference_pdf: Path, issue_dir: Path | None) -> dict[str, str]:
    """Текстовый слой PDF FineReader по полосам выпуска: страница i PDF ↔ i-я полоса папки по имени.

    Args:
        reference_pdf: PDF FineReader выпуска (``full_1966_03.pdf``).
        issue_dir: Папка полос выпуска — порядок страниц берётся из сортировки имён файлов.

    Returns:
        ``{stem полосы: нормализованный текст страницы}``; при расхождении числа страниц —
        предупреждение в лог, лишнее отбрасывается.
    """
    from research.external_ocr_models import evaluate  # noqa: PLC0415 — стенд: pymupdf, rapidfuzz

    names = list_pages(issue_dir) if issue_dir is not None else []
    texts = evaluate.finereader_pages(reference_pdf)
    if len(names) != len(texts):
        logger.warning("в PDF %d страниц, полос в %s — %d; сопоставляю по порядку", len(texts), issue_dir, len(names))
    return {name.stem: evaluate.normalize(text) for name, text in zip(names, texts)}


def page_cer(text: str, reference: str) -> float | None:
    """CER текста без тегов к нормализованному эталону (``evaluate.cer``); ``None`` при пустом эталоне."""
    from research.external_ocr_models import evaluate  # noqa: PLC0415

    return evaluate.cer(evaluate.normalize(text), reference)


def similarity(a: str, b: str) -> float:
    """Сходство двух текстов без тегов (``difflib``, 0–1); пусто с обеих сторон — 1."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def collect(out_dir: Path) -> dict[str, list[PageMetrics | None]]:
    """``{полоса: [показатели по повторам]}`` из ``<out-dir>/run*/``; в прогоне без повторов — из самой папки.

    Args:
        out_dir: Корень прогона ``run`` этого скрипта или выход боевого прогона (тогда один «повтор»).

    Returns:
        Полосы в отсортированном порядке; ``None`` там, где у повтора нет meta.
    """
    runs = sorted((p for p in out_dir.glob("run*") if p.is_dir()), key=lambda p: int(p.name[3:]))
    if not runs:
        runs = [out_dir]
    pages: dict[str, list[PageMetrics | None]] = {}
    for run_index, run_dir in enumerate(runs):
        for meta_path in sorted(run_dir.rglob("*.meta.json")):
            if meta_path.name.endswith((".pass1.meta.json", ".pass2.meta.json")):
                continue
            base = meta_path.with_name(meta_path.name[: -len(".meta.json")])
            key = base.relative_to(run_dir).as_posix()
            pages.setdefault(key, [None] * len(runs))[run_index] = read_page(base)
    return pages


def _totals(pages: dict[str, list[PageMetrics | None]], run_index: int) -> dict[str, float]:
    """Суммы показателей одного повтора по всем полосам (сбойные полосы — только в счётчик сбоев)."""
    items = [runs[run_index] for runs in pages.values() if run_index < len(runs) and runs[run_index] is not None]
    good = [m for m in items if not m.error]
    return {
        "полос": len(items),
        "сбоев": len(items) - len(good),
        "is_damaged": sum(1 for m in good if m.is_damaged),
        "supplied": sum(m.supplied for m in good),
        "unclear": sum(m.unclear for m in good),
        "gap": sum(m.gap for m in good),
        "edge_words": sum(m.edge_words for m in good),
        "пустых ew": sum(m.edge_words_empty for m in good),
        "из ew": sum(m.from_edge_words for m in good),
        "снято с переносов": sum(m.hyphen_stripped for m in good),
        "дублей": sum(m.duplicates_dropped for m in good),
        "[неразборчиво]": sum(m.unreadable for m in good),
        "дефисов": sum(m.inner_hyphens for m in good),
        "знаков": sum(m.chars for m in good),
        "#": sum(m.headings for m in good),
        "id от модели": sum(m.ids_model for m in good),
        "id по названию": sum(m.ids_title for m in good),
        "id чужой": sum(m.ids_wrong for m in good),
        "<rubric>": sum(m.rubric_tags for m in good),
        "rubric id от модели": sum(m.rubric_ids_model for m in good),
        "id колонтитулов": sum(m.header_ids for m in good),
        "$": round(sum(m.cost_usd for m in items), 4),
    }


@main.command("summarize")
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--baseline-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Другой прогон тех же полос (этого скрипта или боевой): сходство текстов с его первым повтором.",
)
@click.option(
    "--reference-pdf",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="PDF FineReader выпуска: CER текста без тегов к его текстовому слою (страница i ↔ i-я полоса по имени).",
)
@click.option(
    "--issue-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Папка полос выпуска (для порядка страниц PDF), например SHARPENED_DIR/1966/03.",
)
def summarize_command(
    out_dir: Path, baseline_dir: Path | None, reference_pdf: Path | None, issue_dir: Path | None
) -> None:
    """Таблица по полосам и повторам: флаг, теги, edge_words, «[неразборчиво]», дефисы, длина, сходство, CER."""
    pages = collect(out_dir)
    if not pages:
        raise click.ClickException(f"в {out_dir} нет результатов")
    baseline = collect(baseline_dir) if baseline_dir is not None else {}
    reference = reference_texts(reference_pdf, issue_dir) if reference_pdf is not None else {}
    if reference_pdf is not None and issue_dir is None:
        raise click.ClickException("--reference-pdf требует --issue-dir")
    n_runs = max(len(runs) for runs in pages.values())
    lines = [
        f"# Повторы: {out_dir}",
        "",
        f"Промпт: {(out_dir / 'prompt_version.txt').read_text().strip() if (out_dir / 'prompt_version.txt').is_file() else '?'}",
        "",
    ]
    lines.append(
        "Ячейка: D — is_damaged; supplied/unclear/gap; ew — записей edge_words (из них пустых) + дослано кодом; нр — «[неразборчиво]»; деф — дефисов между буквами; знаков в теле."
    )
    lines.append("")
    header = "| полоса | " + " | ".join(f"run{i + 1}" for i in range(n_runs)) + " | сходство повторов |"
    if baseline:
        header += " сходство с базой |"
    if reference:
        header += " CER run1 |" + (" CER базы |" if baseline else "")
    extra = (1 if baseline else 0) + ((2 if baseline else 1) if reference else 0)
    lines += [header, "|---|" + "---|" * (n_runs + 1 + extra)]
    cers: dict[str, list[float]] = {"run1": [], "база": []}
    for key, runs in pages.items():
        cells = [m.cell() if m is not None else "—" for m in runs] + ["—"] * (n_runs - len(runs))
        texts = [m.text for m in runs if m is not None and not m.error]
        sim = f"{similarity(texts[0], texts[1]):.3f}" if len(texts) >= 2 else "—"
        row = f"| {key} | " + " | ".join(cells) + f" | {sim} |"
        if baseline:
            base_runs = baseline.get(key) or []
            base_text = next((m.text for m in base_runs if m is not None and not m.error), None)
            row += f" {similarity(texts[0], base_text):.3f} |" if texts and base_text is not None else " — |"
        if reference:
            ref = reference.get(Path(key).stem)
            own = page_cer(texts[0], ref) if texts and ref else None
            row += f" {own:.4f} |" if own is not None else " — |"
            if own is not None:
                cers["run1"].append(own)
            if baseline:
                base_runs = baseline.get(key) or []
                base_text = next((m.text for m in base_runs if m is not None and not m.error), None)
                base_cer = page_cer(base_text, ref) if base_text is not None and ref else None
                row += f" {base_cer:.4f} |" if base_cer is not None else " — |"
                if base_cer is not None:
                    cers["база"].append(base_cer)
        lines.append(row)
    lines += ["", "## Итого по повторам", ""]
    keys = list(_totals(pages, 0))
    lines.append("| | " + " | ".join(keys) + " |")
    lines.append("|---|" + "---|" * len(keys))
    for run_index in range(n_runs):
        totals = _totals(pages, run_index)
        lines.append(f"| run{run_index + 1} | " + " | ".join(str(totals[k]) for k in keys) + " |")
    if baseline:
        totals = _totals(baseline, 0)
        lines.append(f"| база run1 | " + " | ".join(str(totals[k]) for k in keys) + " |")
    if reference:
        lines += ["", "## CER к FineReader (текст без тегов)", ""]
        for name, values in cers.items():
            if values:
                values = sorted(values)
                lines.append(
                    f"- {name}: полос {len(values)}, медиана {values[len(values) // 2]:.4f}, "
                    f"среднее {sum(values) / len(values):.4f}, максимум {values[-1]:.4f}"
                )
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    click.echo(text)


if __name__ == "__main__":
    main()
