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
# корешком буквах она ставила пропуск вместо достроенных букв (55 против 47 на
# IMG_0006_L). Восстановление по контексту нужно обязательно, поэтому фраза требует его явно.
DEFAULT_DAMAGE_NOTE = (
    "Scans of bound volumes may have letters cut off, squashed, shadowed or blurred near the binding gutter and "
    "smeared or washed-out spots anywhere on the page. Nothing specific is known about this page: decide from the "
    "image which edge, if any, runs into the gutter (it can be the left or the right one) and where the print is "
    "unreliable. Hidden or cut-off letters → restore the whole word and mark the restored letters with <supplied>; "
    "visible but unreliable letters → <unclear>; unrecoverable → <gap>N</gap> (N — the approximate number of missing "
    "characters, only the number). Clean print and ordinary end-of-line hyphenation are not damage."
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


# Промпт проверки стыка полос (boundary.py) версионируется отдельно: он не входит в запросы полос,
# и его правка не касается их кэша. Поднимать при любой правке boundary_*.md.j2.
BOUNDARY_PROMPT_VERSION = 1


def boundary_prompts(seams: list[dict]) -> tuple[str, str]:
    """Системный и пользовательский промпты проверки стыков выпуска (по две полоски строк на стык, один запрос).

    Args:
        seams: Стыки по порядку: ``{"number", "page_before", "page_after", "tail", "head", "reason"}`` —
            номер с 1, имена полос, конец транскрипции полосы N и начало N+1 без переводов строк,
            причина сомнения фразой.

    Returns:
        ``(system, user)``; картинки (по две на стык, в том же порядке) к user добавляет вызывающий.
    """
    return render("boundary_system.md.j2"), render("boundary_user.md.j2", seams=seams)


def system_prompt(stage: Stage, source: str = "", has_list: bool = False) -> str:
    """Системный промпт этапа — один на весь пак: без списка статей выпуска, поэтому кэшируется провайдером
    как префикс между выпусками и годами; общие правила идут первыми, чтобы toc и page делили префикс.

    Args:
        stage: ``TOC`` — полоса оглавления или указателя (извлечь структуру); ``PAGE`` — обычная полоса.
        source: Описание издания для первого абзаца; пусто — советская и постсоветская
            экономическая пресса вообще. Без года — иначе префикс не совпадает между выпусками.
        has_list: Есть ли в пользовательском сообщении список статей выпуска (этап ``PAGE``):
            с ним действуют правила «`#` только из списка», без него — общие правила.

    Returns:
        Текст системного сообщения.
    """
    return render("system.md.j2", stage=Stage(stage), source=source.strip(), has_list=bool(has_list))


def user_prompt(
    ntiles: int,
    ncols: int,
    nrows: int,
    stage: Stage = Stage.PAGE,
    toc_kind: TocKind = TocKind.CONTENTS,
    second_pass: object | None = None,
    max_lines: int = 60,
    rubrics: list[str] | None = None,
    articles: list[dict] | None = None,
) -> str:
    """Текст рядом с картинками: фраза про повреждения, задача этапа, список выпуска, раскладка тайлов.

    Порядок блоков — под кэш префикса: константы первыми, список выпуска (меняется от выпуска к
    выпуску) после них, раскладка тайлов и второй проход (меняются от полосы к полосе) — в конце.

    Args:
        ntiles: Сколько картинок приложено к сообщению.
        ncols: Столбцов в сетке тайлов (порядок картинок — по столбцам, сверху вниз).
        nrows: Строк в сетке тайлов.
        stage: Этап: у ``TOC`` — задача проверить и извлечь оглавление, у ``PAGE`` — прочитать полосу.
        toc_kind: Для этапа ``TOC`` — что предположил детектор: «Содержание» или годовой указатель.
        second_pass: Сводка первого прохода (``ocr.SecondPass``: ``damage``, ``edge_words``,
            ``tags``, ``transcript``); с ней вместо нейтральной фразы о повреждениях идёт блок
            «первое чтение нашло…». ``None`` — первый проход.
        max_lines: Сколько строк ``edge_words`` первого прохода показывать второму.
        rubrics: Рубрики «Содержания» выпуска для этапа ``PAGE``; пусто — списка нет.
        articles: Статьи «Содержания» (``[{"title", "authors": [str], "rubric"}]``) для этапа ``PAGE``.

    Returns:
        Текст пользовательского сообщения (без картинок).
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
        rubrics=list(rubrics or []),
        articles=list(articles or []),
    )
