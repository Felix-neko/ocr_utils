"""Промпты пробника VLM — Jinja-шаблоны рядом; ``VLM_PROMPT_VERSION`` в ``vlm.py`` поднимать при правке."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)), undefined=StrictUndefined, keep_trailing_newline=False
    )


def render(template_name: str, **variables: object) -> str:
    return _environment().get_template(template_name).render(**variables).strip()
