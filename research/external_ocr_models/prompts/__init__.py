"""Промпты — Jinja-шаблоны рядом с этим файлом, а не строки в коде.

Формулировки правятся чаще кода, и в отдельных файлах их удобно сравнивать между
прогонами. ``PROMPT_VERSION`` пакета поднимать при любой правке шаблонов.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).parent


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


def system_prompt(output_mode: str, damage: bool = False, source: str = "") -> str:
    """Системный промпт под режим ответа: ``json`` или ``markdown`` с YAML-шапкой.

    ``damage`` — правило про повреждённые буквы: достраивать и помечать ``<restored>``,
    сомнительные — ``<fuzzy>``, нечитаемые — ``<unknown/>``. ``source`` — описание издания,
    если известно; по умолчанию промпт говорит про советскую и постсоветскую экономическую
    прессу вообще, журналы и газеты.
    """
    return render("system.md.j2", output_mode=output_mode, damage=damage, source=source.strip())


def user_prompt(strips: int, hint: str = "", damage: bool = False) -> str:
    return render("user.md.j2", strips=strips, hint=hint.strip(), damage=damage)
