"""Стенд исследования текстового слоя FineReader: обзор по паку, выборки, оверлеи, оценка источников.

Ядро (разбор слоя, зоны, чтение, вердикты, правка, кэш) переехало в ``ocr_utils.text_layer_fix``
и вызывается сборщиком финальных PDF. Здесь остались команды ``survey``/``run``/``fix``/
``overlay``/``second-opinion``/``eval-lineart``/``llm-compare``/``reclassify``/``report``
(``cli.py``), выборка страниц по базам (``pages.py``, ``db_models.py``), точность источников
line art (``lineart_eval.py``), сравнение с LLM (``llm_sanitize.py``), оверлеи и сводка
(``overlay.py``, ``report.py``). Отчёт — ``reports/text_layer_fix.md``.
"""
