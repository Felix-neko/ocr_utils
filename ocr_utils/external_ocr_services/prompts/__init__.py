"""Промпты — Jinja-шаблоны рядом с этим файлом; ``PROMPT_VERSION`` пакета поднимать при любой правке."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ocr_utils.external_ocr_services.schema import Stage, TocKind

# Шаблоны лежат рядом с этим файлом: system.md.j2, user.md.j2 (+ переводы *.ru.md для чтения).
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
    """Окружение Jinja для шаблонов промптов; одно на процесс.

    ``StrictUndefined`` — опечатка в имени переменной шаблона падает сразу, а не уходит в промпт
    пустым местом. Пробелы и переводы строк шаблонов не трогаются: промпт должен читаться как написан.
    """
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def render(template_name: str, **variables: object) -> str:
    """Отрисовать шаблон из ``PROMPTS_DIR``; результат без краевых пробелов и с одним переводом строки в конце.

    Args:
        template_name: Имя файла шаблона, например ``system.md.j2``.
        **variables: Переменные шаблона (все обязательны — ``StrictUndefined``).
    """
    return _environment().get_template(template_name).render(**variables).strip() + "\n"


def system_prompt(
    stage: Stage, source: str = "", rubrics: list[str] | None = None, articles: list[dict] | None = None
) -> str:
    """Системный промпт этапа — одинаковый для всех полос выпуска, поэтому кэшируется провайдером как префикс.

    Args:
        stage: ``TOC`` — полоса оглавления или указателя (извлечь структуру); ``PAGE`` — обычная полоса.
        source: Описание издания для первого абзаца; пусто — советская и постсоветская
            экономическая пресса вообще. Без года — иначе префикс не совпадает между выпусками.
        rubrics: Рубрики «Содержания» выпуска для этапа ``PAGE``; пустой список — общие правила без них.
        articles: Статьи «Содержания» (``[{"title", "authors": [str], "rubric"}]``) для этапа ``PAGE``.
    """
    return render(
        "system.md.j2",
        stage=Stage(stage),
        source=source.strip(),
        rubrics=list(rubrics or []),
        articles=list(articles or []),
    )


def user_prompt(
    ntiles: int,
    ncols: int,
    nrows: int,
    stage: Stage = Stage.PAGE,
    toc_kind: TocKind = TocKind.CONTENTS,
    second_pass: object | None = None,
    max_lines: int = 60,
) -> str:
    """Текст рядом с картинками: раскладка тайлов, фраза про повреждения, задача этапа.

    Args:
        ntiles: Сколько картинок приложено к сообщению.
        ncols: Столбцов в сетке тайлов (порядок картинок — по столбцам, сверху вниз).
        nrows: Строк в сетке тайлов.
        stage: Этап: у ``TOC`` — задача извлечь оглавление, у ``PAGE`` — прочитать полосу.
        toc_kind: Для этапа ``TOC`` — что это: «Содержание» или годовой указатель (разные подсказки).
        second_pass: Сводка первого прохода (``ocr.SecondPass``: ``damage``, ``edge_words``,
            ``tags``, ``transcript``); с ней вместо нейтральной фразы о повреждениях идёт блок
            «первое чтение нашло…». ``None`` — первый проход.
        max_lines: Сколько строк ``edge_words`` первого прохода показывать второму.
    """
    return render(
        "user.md.j2",
        ntiles=ntiles,
        ncols=ncols,
        nrows=nrows,
        damage_note=DEFAULT_DAMAGE_NOTE,
        stage=Stage(stage),
        toc_kind=TocKind(toc_kind),
        second_pass=second_pass,
        max_lines=max_lines,
    )
