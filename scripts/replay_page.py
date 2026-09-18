"""Повторы распознавания полос боевым пакетом external_ocr_services под A/B промптов: N прогонов, сводка по повторам.

Зачем: разметка повреждений у DeepSeek V4.1 Flash плавает между одинаковыми запросами (IMG_0008_R:
16 против 64 ``<supplied>`` при том же промпте), поэтому сравнивать версии промпта по одному прогону
нельзя. Раньше такие повторы собирались в scratchpad-скриптах и терялись вместе с сессией (а один
раз ушли впустую $0.30, потому что руками собранный payload не выключал ``reasoning``). Здесь
запрос идёт через ``ocr.recognise_page`` — тот же код, что в прогоне: тайлы, цепочка режимов
JSON, ``reasoning.effort none``, разбор, теги из ``edge_words``, доводка структуры — только без
второго прохода и без оглавления выпуска (список статей в промпт не уходит).

Команды::

    # 2 повтора мини-набора повреждённых полос текущими промптами (PROMPT_VERSION пакета)
    uv run python scripts/replay_page.py run \\
        --in-dir "research/external_ocr_models/damaged/нарезанное по страницам" \\
        --out-dir /mnt/SYSTEM/raw/mts/replay/damaged_v16 --repeats 2

    # то же промптами другой версии: папка с system.md.j2, user.md.j2 и, если нужно, damage_note.txt
    uv run python scripts/replay_page.py run ... --out-dir .../damaged_v15 --prompts-dir /tmp/prompts_v15

    # сводка по повторам (и сходство с другим прогоном тех же полос)
    uv run python scripts/replay_page.py summarise --out-dir .../damaged_v16 [--baseline-dir .../damaged_v15]

Выход ``run``: ``<out-dir>/run1/``, ``run2/`` … с той же раскладкой, что у прогона
(``.json``/``.md``/``.meta.json`` по полосе); сырые ответы, промпты и тайлы первого повтора — в
``<out-dir>/debug/``. Сводка ``summarise`` печатается в терминал и пишется в ``<out-dir>/summary.md``.
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

from ocr_utils.external_ocr_services import PROMPT_VERSION, prompts  # noqa: E402
from ocr_utils.external_ocr_services import models as registry  # noqa: E402
from ocr_utils.external_ocr_services.client import OpenRouterClient, api_key_from  # noqa: E402
from ocr_utils.external_ocr_services.ocr import PageJob, RunOptions, recognise_page  # noqa: E402
from ocr_utils.external_ocr_services.pages import list_pages  # noqa: E402
from ocr_utils.external_ocr_services.schema import DamageTag  # noqa: E402
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


def _recognise_one(
    client: OpenRouterClient, spec, in_dir: Path, out_dir: Path, options: RunOptions, rel: Path
) -> tuple[Path, dict]:
    """Одна полоса одним проходом — для пула потоков.

    Args:
        client: Клиент OpenRouter, общий на прогон.
        spec: Модель из реестра.
        in_dir: Корень входа.
        out_dir: Корень выхода этого повтора.
        options: Настройки запроса.
        rel: Полоса относительно ``in_dir``.

    Returns:
        ``(rel, meta)`` — meta как записана в ``.meta.json``.
    """
    meta, _ = recognise_page(client, spec, in_dir / rel, PageJob(rel), out_dir, options)
    return rel, meta


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
) -> None:
    """Распознать полосы ``--repeats`` раз каждую; повтор i пишется в ``<out-dir>/run<i>/``."""
    if prompts_dir is not None:
        use_prompts_dir(prompts_dir)
    spec = registry.resolve(model_name)
    rels = list_pages(in_dir, pages_file, only_year, only_issue, limit)
    if not rels:
        raise click.ClickException(f"в {in_dir} нет полос")
    client = OpenRouterClient(api_key_from(api_key))
    logger.info("полос %d × повторов %d, модель %s, промпт v%d", len(rels), repeats, spec.name, PROMPT_VERSION)
    total_cost = 0.0
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
        work = partial(_recognise_one, client, spec, in_dir, run_dir, options)
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for rel, meta in pool.map(work, rels):
                total_cost += float(meta.get("cost_usd") or 0.0)
                status = meta.get("error") or meta.get("parse_error") or "ok"
                logger.info(
                    "run%d %s: %s, теги %s, is_damaged=%s", index, rel, status, meta.get("tags"), meta.get("is_damaged")
                )
    (out_dir / "prompt_version.txt").write_text(
        f"{PROMPT_VERSION}\n{prompts_dir or 'шаблоны пакета'}\n", encoding="utf-8"
    )
    click.echo(f"Готово: {len(rels)} полос × {repeats}, ${total_cost:.4f}; сводка — summarise --out-dir {out_dir}")


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

    def cell(self) -> str:
        """Ячейка таблицы: «s/u/g · ew(пусто) · нераз · дефисы · знаков» или причина сбоя."""
        if self.error:
            return f"сбой: {self.error[:40]}"
        damaged = "D" if self.is_damaged else "-"
        return (
            f"{damaged} {self.supplied}/{self.unclear}/{self.gap} · ew {self.edge_words}({self.edge_words_empty})"
            f"+{self.from_edge_words} · нр {self.unreadable} · деф {self.inner_hyphens} · {self.chars}"
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
    )


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
        "$": round(sum(m.cost_usd for m in items), 4),
    }


@main.command("summarise")
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--baseline-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Другой прогон тех же полос (этого скрипта или боевой): сходство текстов с его первым повтором.",
)
def summarise_command(out_dir: Path, baseline_dir: Path | None) -> None:
    """Таблица по полосам и повторам: флаг, теги, edge_words, «[неразборчиво]», дефисы, длина, сходство."""
    pages = collect(out_dir)
    if not pages:
        raise click.ClickException(f"в {out_dir} нет результатов")
    baseline = collect(baseline_dir) if baseline_dir is not None else {}
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
    lines += [header, "|---|" + "---|" * (n_runs + 1) + ("---|" if baseline else "")]
    for key, runs in pages.items():
        cells = [m.cell() if m is not None else "—" for m in runs] + ["—"] * (n_runs - len(runs))
        texts = [m.text for m in runs if m is not None and not m.error]
        sim = f"{similarity(texts[0], texts[1]):.3f}" if len(texts) >= 2 else "—"
        row = f"| {key} | " + " | ".join(cells) + f" | {sim} |"
        if baseline:
            base_runs = baseline.get(key) or []
            base_text = next((m.text for m in base_runs if m is not None and not m.error), None)
            row += f" {similarity(texts[0], base_text):.3f} |" if texts and base_text is not None else " — |"
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
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    click.echo(text)


if __name__ == "__main__":
    main()
