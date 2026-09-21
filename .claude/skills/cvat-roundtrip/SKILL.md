---
name: cvat-roundtrip
description: Обмен разметкой с локальным CVAT (cvat_mts) — поднять инстанс, залить или дозалить находки детектора, забрать ручную правку в базу, проверить здоровье. Использовать при словах CVAT, to-cvat, from-cvat, теги, задачи разметки.
---

# CVAT: туда и обратно

Подробности путей — `docs/data_layout.md` (раздел CVAT), шаги 2–4 конвейера —
`docs/pack1_pipeline.md`. Команды — `ocr_utils/scan_markup/README.md`.

## Инстанс

- Этот проект: compose-проект **`cvat_mts`**, `docker/up.sh` / `docker/down.sh`, порт **8081**
  (`docker/.env`). Не путать с dev-CVAT пользователя (`~/Projects/cvat`, проект `cvat`): оба
  хотят 8080, тома разные.
- Проверка: `docker ps --filter label=com.docker.compose.project=cvat_mts` и
  `curl -s -H 'Accept: application/json' 'http://localhost:8081/api/server/health/?format=json'`.
  «Services are not healthy» в UI при живых контейнерах — заполнен `/` выше
  `CVAT_HEALTH_DISK_USAGE_MAX` (95): `df -h /`, `docker builder prune -af`, `docker image prune -f`.
  `image prune -a` и `volume prune` — никогда.
- Картинки только из `IMAGES_DIR` (`docker/.env`): `--share-root` обязан лежать внутри него.
  `IMAGES_DIR` при живых контейнерах не переименовывать; после смены — `down.sh` + `up.sh`.

## Залить (to-cvat)

- Первый раз по паку: `run_2_to_cvat.sh` — с `--annotator user`, иначе разметчик видит пустой список.
- Дозалить находки нового детектора в уже размеченные задачи: `to-cvat --append-kinds table,line_art_schema,stroke_table,stroke_drawing`
  или теги `--append-tags` — это PATCH, ручная правка цела.
- **`--force-annotations` затирает ручную разметку задачи целиком**; `--recreate-stale`
  пересоздаёт задачу-года — перед ними прогнать `from-cvat` как снимок. Оба — только по явному
  решению пользователя.
- Точечное обновление изменившихся полос — `run_4_update_pages.sh` (сам бэкапит sqlite и pg_dump).

## Забрать (from-cvat)

- `run_3_from_cvat.sh` → `DB_REVIEWED`; автодетекция в `DB` не трогается. Можно гонять как
  снимок в любой момент.
- Координаты приходят в масштабе уменьшенных копий: умножать на `divisor` из базы (у полосы свой).
- После забора — проверить число шейпов/тегов против ожидания (`count-shapes`), базы бэкапятся
  `common.sh`.

## Метки

Растр = rectangle (`color`/`grayscale`), печать и рукописная надпись = mask, таблицы и схемы =
`table`/`line_art_schema`, теги полосы «Оглавление» / «Годовой указатель». Новый вид метки —
сначала в `scan_markup/cvat`, потом в проект CVAT, потом в базу.
