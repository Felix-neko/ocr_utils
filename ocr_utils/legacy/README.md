# `ocr_utils.legacy` — помойка

Неподдерживаемый код: старые эксперименты и части пайплайна, которыми больше не
пользуются. Импортироваться должно, работать — не обязано. Тестами не покрыто.

## Что тут лежит

| Путь | Что это |
|---|---|
| `dewarp/` | Выпрямление страниц: обвязка над DewarpNet / DocScanner / UVDoc / docTR / page-dewarp. Заменено внешним ScanTailor. |
| `defocus_detection/` | Детекция расфокуса сканов по метрике муара (NEAREST−AREA), FFT-HF и лапласиану + CLI `detect_defocus`. |
| `stamp_and_writing_detection/` | Поиск печатей и рукописного текста на сканах. |
| `page_layout_processing/` | Пробные прогоны Surya layout и OpenCV-сегментации разворота. |
| `finger_removal/` | Части пакета удаления пальцев, не участвующие в `scan_cropping`: самостоятельные CLI, инпейнтинг через Stable Diffusion и «продолжение кромки», батчевая детекция. |
| `ocr.py`, `pipeline.py`, `config.py`, `cli.py`, `run_ocr_batch.py` | Старый пайплайн «PDF → OCR-слой» на ocrmypdf. `cli.py` — бывшая точка входа `ocr-utils` (entry-point из `pyproject.toml` снят). |
| `pdf_utils_flat.py` | Мёртвый плоский дубль пакета `ocr_utils.pdf_utils`: лежал рядом с одноимённым каталогом и был им затенён, то есть не импортировался в принципе. |

## Как запустить что-то отсюда

Модульные пути изменились на `ocr_utils.legacy.*`:

```bash
uv run python -m ocr_utils.legacy.defocus_detection --help
uv run python -m ocr_utils.legacy.dewarp --help
```
