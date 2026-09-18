"""Учебный скрипт: одна повреждённая страница → DeepSeek V4.1 Flash → JSON и Markdown с пометками.

Это развёрнутая, построчно откомментированная копия того, что делает
``research.external_ocr_models run --model deepseek-v41-flash --strips 2 --damage --hints …``
для одной страницы: те же промпты (``system_prompt.md``, ``user_prompt.md`` — отрендеренные
Jinja-шаблоны из ``external_ocr_models/prompts``), та же нарезка картинки на две
перекрывающиеся полосы, то же тело запроса к OpenRouter, тот же разбор ответа и та же
структура выхода (``.json``, ``.md``, ``.meta.json``). CLI нет: пути и подсказка — константы
в ``ocr_damaged_page.py``. Запуск: ``uv run python -m research.deepseek_damage_lesson.ocr_damaged_page``.
"""
