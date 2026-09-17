"""Промпты — Jinja-шаблоны рядом с этим файлом; ``PROMPT_VERSION`` пакета поднимать при любой правке."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).parent

# Нейтральная фраза про повреждения по умолчанию: правило про теги действует всегда, но без
# подсказки модель повреждений не ищет, а с завышенной — выдумывает (стенд, разделы 8 и 10).
# Первая редакция («помечай только то, что видишь») делала модель осторожной: на скрытых
# корешком буквах она ставила <unknown/> вместо достроенных <restored> (55 против 47 на
# IMG_0006_L). Восстановление по контексту нужно обязательно, поэтому фраза требует его явно.
DEFAULT_DAMAGE_NOTE = (
    "Scans of bound volumes may have letters cut off, squashed or blurred near the binding gutter, and overexposed "
    "spots anywhere on the page. No specific damage has been reported for this page, so look for it yourself: decide "
    "from the image which edge, if any, runs into the gutter (it can be the left or the right one) and where the print "
    "is unreliable. "
    "Wherever letters are hidden, cut off or unreadable, DO reconstruct them from the context and the visible remains "
    "of the word and mark the reconstructed letters with <restored>; mark visible but unreliable letters with <fuzzy>; "
    "use <unknown/> only where no confident reconstruction is possible. Do not tag clean print or ordinary "
    "end-of-line hyphenation."
)


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def render(template_name: str, **variables: object) -> str:
    return _environment().get_template(template_name).render(**variables).strip() + "\n"


def system_prompt(
    stage: str, source: str = "", rubrics: list[str] | None = None, articles: list[dict] | None = None
) -> str:
    """Системный промпт этапа.

    ``stage`` — ``toc`` (полоса оглавления или указателя: извлечь структуру) или ``page`` (обычная
    полоса). ``rubrics`` и ``articles`` (``[{"title", "authors": [str]}]``) — известное оглавление
    выпуска для этапа ``page``; пустые списки — общие правила без него. ``source`` — описание
    издания; пусто — советская и постсоветская экономическая пресса вообще.
    """
    return render(
        "system.md.j2", stage=stage, source=source.strip(), rubrics=list(rubrics or []), articles=list(articles or [])
    )


def user_prompt(
    ntiles: int,
    ncols: int,
    nrows: int,
    stage: str = "page",
    toc_kind: str = "contents",
    second_pass: object | None = None,
    max_lines: int = 60,
) -> str:
    """Текст рядом с картинками: раскладка тайлов, фраза про повреждения, задача этапа.

    ``second_pass`` — сводка первого прохода (``ocr.SecondPass``: ``damage``, ``edge_words``,
    ``tags``, ``transcript``); с ней вместо нейтральной фразы идёт блок «первое чтение нашло…».
    """
    return render(
        "user.md.j2",
        ntiles=ntiles,
        ncols=ncols,
        nrows=nrows,
        damage_note=DEFAULT_DAMAGE_NOTE,
        stage=stage,
        toc_kind=toc_kind,
        second_pass=second_pass,
        max_lines=max_lines,
    )
