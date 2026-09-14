# Внешние OCR-модели для полос «Материально-технического снабжения»

Задача: получить вторую, структурную версию каждой полосы (заголовки, авторы с
должностями, таблицы с боковыми шапками, схемы, номер страницы и колонтитулы), чтобы
потом слить её с буквами FineReader. Пакет — `research/external_ocr_models`
(README там же), прогоны — `run_scripts/external_ocr_models/`, выходы —
`/mnt/SYSTEM/raw/mts/pack1_external_ocr_probe/` (пробник, 13 полос × все модели) и
`/mnt/SYSTEM/raw/mts/pack1_external_ocr/` (весь выпуск 1966/03, 4 модели).

## Коротко

* **Что брать.** Для структуры полосы (заголовки, авторы с должностями, таблицы с боковыми
  шапками, номер страницы) достаточно дешёвых моделей: **Gemini 3.1 Flash Lite** (0,22 ¢ за
  полосу, 21 ¢ за выпуск, ≈ $27 за весь пак из 12 135 полос) — лучшая по таблицам и самая
  ровная; **Qwen3.8 Flash** (0,09 ¢, ≈ $10 за пак) и **DeepSeek V4.1 Flash двумя кусками**
  (0,10 ¢, ≈ $12 за пак) — почти не хуже и втрое дешевле. Gemini 3.8 Flash (0,52 ¢, ≈ $63 за
  пак) на этих полосах не даёт ничего сверх Lite. Три модели на один пак — около $50, и в
  трёх независимых версиях есть из чего выбирать при слиянии.
* **Буквы.** На чистом тексте лучшие VLM расходятся с текстовым слоем FineReader на 0,1 %
  знаков — то есть буквы они читают не хуже FineReader, а таблицы и оглавление — лучше
  (там FineReader сам даёт мусор). Претензия к FineReader остаётся только по кривым
  строкам и пятнистому фону, которых в 1966/03 после очистки почти нет.
* **Не брать.** Claude Haiku 4.5 (в 4-9 раз дороже, теряет шапки таблиц, дописывает рубрику
  из примера в промпте), Mistral Small 4 (выдумывает шапки таблиц), Qwen3-VL-235B (уходит в
  цикл по отточиям, 3 ¢ и 200 с на такой полосе), GPT-5.6 Luna (прилично, но слабее на
  большой таблице и не лучше дешёвых).
* **DeepSeek V4.1 Flash** целой полосой теряет текст на плотных страницах: потолок 1024
  токена на картинку в самой модели, параметром не поднимается; два куска в одном
  запросе снимают проблему (ответ один, сшивать не нужно).
* **Локально** на RTX 5060 Ti завелись dots.ocr (таблицы с боковыми шапками читает
  идеально, буквы хуже облачных, 27 с/полоса, 6,7 ГБ), DeepSeek-OCR-2 (20 с, 7 ГБ; теряет тело
  таблиц с боковыми шапками, выдумывает их, нестабильна от прогона к прогону, не
  инструктивна) и Marker 1.x (29 с; таблицы рассыпаются). Ни один не даёт полей «автор /
  должность / номер страницы» — для них нужен второй проход языковой моделью, и тогда
  дешевле сразу отдать полосу Gemini Lite. PaddleOCR-VL: завёлся на GPU после ручной загрузки колеса paddlepaddle-gpu cu130 с источника Baidu (CDN
  из индекса не отвечает): 19 с/полоса, 1,3 ГБ VRAM, буквы на уровне облачных (CER 0,002 на
  чистом тексте), таблицы с боковыми шапками структурно верны, но в самих шапках ошибки
  («фи-знческого», «ИСПОЛЬЗО-ванне»), широкие заголовки читает без пробелов
  («ВНОВЫХУСЛОВИЯХ»), таблицы отдаёт с инлайн-стилями HTML.
* **Специализированные OCR-API** (Mistral OCR 4.1 — $4 за 1000 стр., Yandex Vision — $1.08,
  Google/Azure — $1.50) не дешевле VLM через OpenRouter и не дают семантики статьи;
  их плюс — координаты слов и отсутствие галлюцинаций.
* **Потрачено** на всё исследование (пробник на 11 прогонах, повторы, выпуск целиком на четырёх моделях) — **$1.65** из $30.

## 1. Как устроены современные VLM-OCR и что им подавать

**Что это.** Все модели ниже — не «OCR-движки» в старом смысле, а мультимодальные языковые
модели: картинка кодируется в токены (у Gemini — плитки 768×768 по 258 токенов, у
DeepSeek — не больше 1024 токенов на картинку, у Qwen — ~1 токен на 28×28 px), и дальше
модель *пишет* текст страницы так же, как писала бы ответ в чате. Отсюда их сильные и
слабые стороны: они понимают структуру (что заголовок, что автор, что боковая шапка
таблицы) и склеивают переносы сами, но могут исправить, дописать или зациклиться.

**Картинка.** 600-dpi сканы по 8-22 МБ слать нельзя и незачем — модели всё равно ужимают
до своей сетки. Берём серую копию с длинной стороной 2200 px (≈216 dpi: строчные буквы
основного текста 9-10 pt ≈ 20 px, петит ≈ 13 px), JPEG q85 — 0,5-0,7 МБ, base64 в
`image_url`. Резать полосу на куски не нужно **кроме DeepSeek** (см. ниже).

**Промпт.** Системный промпт (`research/external_ocr_models/prompts/system.md.j2`) описывает
документ (советский журнал 1966-76, русский, орфография 1960-х), требует дословную
транскрипцию без исправлений, задаёт разметку (`#` статья, `##` подзаголовок, `###` рубрика,
`**автор**`, `*должность*`, таблицы GFM или `<table>` с rowspan/colspan, повёрнутый текст —
как обычный, `> [блок-схема]`, `> [картинка: …]`, разрядка → курсив без пробелов) и просит
колонтитулы и номер страницы отдельными полями, а не в теле. Инструкции на английском:
мелкие модели держат их надёжнее. Два урока пробника:
* *Отточия*: без явного запрета «. . . . .» в оглавлении и таблицах любая модель может
  уйти в цикл до `max_tokens` (16 тыс. токенов, 2,5-3 ¢ и 200 с на полосу). Запрет в
  промпте снял это у всех, кроме Qwen3-VL-235B.
* *Примеры в промпте протекают*: Claude Haiku дописала рубрику «Опыт работы
  территориальных управлений» (пример из промпта) на страницу, где её нет.

**Схема ответа.** Просим `response_format: json_schema` (strict) с полями
`page_number, running_header, running_footer, is_toc, content_markdown, notes` и
`provider.require_parameters` — чтобы запрос не ушёл к провайдеру без поддержки схемы. Если
провайдер отверг (404/400) — тот же запрос в режиме `json_object`, потом без
`response_format`; что использовано, записано в `.meta.json`. У DeepSeek родной эндпоинт
схем не умеет — сразу `json_object`. Разбор терпимый: снимаем ```-ограждения, берём первый
JSON-объект, что бы ни шло следом (модели дописывают эхо `{"type":"json_object"}`).

**Рассуждения.** Выключены (`reasoning: {enabled: false}`): на DeepSeek V4.1 Flash с
`effort: low` полоса стоила втрое дороже (4238 токенов рассуждений) при побайтно том же
тексте. У Gemini 3.x thinking не выключается — только `low`. GLM отвергает параметр
`reasoning` (400) — клиент повторяет запрос без него.

**DeepSeek V4.1 Flash и потолок 1024 токенов.** Зрение у него нативное (DeepSeek-ViT в
претрейне; Vision-Exp снят), но любая картинка ужимается до ~1300×1300 px = 1024 токенов;
параметром это не поднимается (`detail` только `low`/`high`/`original`). Целая полоса
6096 px превращается в ≈128 dpi, и на плотных полосах модель теряет текст: IMG_0114_1L
(таблица + текст) — CER 0,28 против 0,07 у остальных. Лекарство — `--strips 2`: полоса режется
на нашей стороне на 2 куска с перекрытием 8 % высоты, оба уходят **одним** запросом, ответ
один, сшивать нечего; CER падает до 0,07, цена растёт с 0,075 до 0,094 ¢. 3 куска ничего не
добавляют.

**Стоимость.** OpenRouter кладёт цену каждого запроса в ответ (`usage.cost`, $), отчёт
считает по ней, а не по прайсу — у одной модели десяток провайдеров с разными ценами.
Настройка приватности аккаунта «провайдеры, обучающиеся на запросах» отсекала родной
эндпоинт DeepSeek (самый дешёвый, bf16); после её снятия он в приоритете
(`provider.order=["deepseek"]`), fp4-копия Relace исключена.

## 2. Кандидаты на OpenRouter

В каталоге 445 моделей, с картинками на входе — около 200. Специализированных OCR-моделей
(Mistral OCR, PaddleOCR-VL, DeepSeek-OCR, dots.ocr) на OpenRouter **нет** — только общие VLM.
Отобраны по цене и репутации в документных задачах (цены — $ за M токенов вход/выход):

| имя в реестре | модель | $/M in | $/M out | зачем |
|---|---|---|---|---|
| deepseek-v41-flash | deepseek/deepseek-v4.1-flash | 0.15 | 0.60 | просьба заказчика; нативное зрение |
| gemini-31-flash-lite | google/gemini-3.1-flash-lite | 0.25 | 1.50 | дешёвый Gemini, плитки 768 px |
| qwen38-flash | qwen/qwen3.8-flash | 0.15 | 0.47 | дешёвый Qwen с картинками |
| gpt-56-luna | openai/gpt-5.6-luna | 0.20 | 1.20 | самый дешёвый OpenAI с картинками |
| mistral-small-4 | mistralai/mistral-small-2603 | 0.15 | 0.60 | дешёвый европейский |
| glm-53-flash | z-ai/glm-5.3-flash | 0.15 | 0.50 | Z.ai, нативная мультимодальность |
| gemini-38-flash | google/gemini-3.8-flash | 0.75 | 3.75 | лидер OCR Arena среди flash |
| qwen3-vl-235b | qwen/qwen3-vl-235b-a22b-instruct | 0.21 | 1.90 | Qwen «для document parsing» |
| claude-haiku-45 | anthropic/claude-haiku-4.5 | 1.00 | 5.00 | самый дешёвый Claude |

В реестре, но не в пробнике: gemini-31-pro и claude-sonnet-5 (эталоны по $2/M), gpt-54-mini,
qwen37-flash ($0.03/M — самая дешёвая с картинками), gemma-4-31b, seed-20-mini.

## 3. Пробник: 13 полос × 11 прогонов

Полосы выпуска 1966/03 подобраны по базе разметки и текстовому слою FineReader
(`run_scripts/external_ocr_models/probe_pages_1966_03.txt`): обложка, шапка журнала с началом
первой статьи, чистая текстовая полоса, три начала статей с авторами и должностями (одна с
рубрикой, одна с двумя авторами), четыре таблицы (три с боковыми шапками, одна на 33
ячейки), чертёж с таблицей, схема, оглавление. Прогон: `run_probe_1966_03.sh`, промпт v2,
первая попытка (сбои первой попытки ниже, в таблице — состояние после повтора сбойных
полос); локальные движки — `run_local_probe_1966_03.sh`. Итого 143 запроса, **$0.59**.

Сбои первой попытки: Gemini 3.1 Lite и Qwen3-VL-235B по 2 полосы, Qwen3.8 Flash — 1 (все —
цикл по отточиям в оглавлении и таблице до промпта v2 с запретом отточий; после него у
Gemini и Qwen3.8 сбоев нет, Qwen3-VL-235B зациклился снова); DeepSeek — 2 полосы ошибок
разбора (валидный JSON + мусор следом; разбор сделан терпимее). В таблице сбоями считаются и
пустые ответы (GPT-5.6 Luna и Mistral отдали пустое тело на обложке, dots.ocr — рамки без
текста на IMG_0123_2R).

| модель | полос | сбоев | обрезано | CER к FR (медиана) | CER к FR (текст) | согласие | h1/h2/h3 | авторов | должн. | таблиц | повёрн. | схем | карт. | зацикл. | № стр. верно | с/полоса | ¢/полоса | ¢/выпуск | $/пак |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-45 | 13 | 0 | 0 | 0.071 | 0.126 | 0.196 | 9/11/2 | 6 | 6 | 6 | 5/13 | 0 | 3 | 0 | 11/12 | 16 | 0.847 | 82.1 | 102.76 |
| deepseek-v41-flash | 13 | 0 | 0 | 0.087 | 0.111 | 0.143 | 7/1/9 | 6 | 6 | 6 | 11/13 | 1 | 2 | 0 | 12/12 | 6 | 0.074 | 7.2 | 8.96 |
| deepseek-v41-flash-s2 | 13 | 0 | 0 | 0.072 | 0.092 | 0.129 | 7/9/1 | 6 | 6 | 5 | 13/13 | 2 | 1 | 0 | 12/12 | 7 | 0.092 | 8.9 | 11.18 |
| deepseek-v41-flash-s3 | 13 | 0 | 0 | 0.083 | 0.096 | 0.157 | 9/8/1 | 6 | 7 | 5 | 13/13 | 1 | 1 | 0 | 12/12 | 6 | 0.102 | 9.9 | 12.43 |
| gemini-31-flash-lite | 13 | 0 | 0 | 0.031 | 0.049 | 0.144 | 9/0/10 | 6 | 6 | 5 | 13/13 | 0 | 2 | 0 | 11/12 | 5 | 0.192 | 18.6 | 23.32 |
| gemini-38-flash | 13 | 0 | 0 | 0.030 | 0.049 | 0.144 | 7/3/9 | 8 | 7 | 7 | 13/13 | 0 | 2 | 0 | 11/12 | 5 | 0.477 | 46.3 | 57.89 |
| glm-53-flash | 13 | 0 | 0 | 0.036 | 0.052 | 0.142 | 7/10/1 | 8 | 7 | 5 | 13/13 | 0 | 3 | 0 | 11/12 | 43 | 0.255 | 24.7 | 30.94 |
| gpt-56-luna | 13 | 1 | 0 | 0.070 | 0.064 | 0.068 | 7/0/10 | 6 | 6 | 6 | 9/13 | 0 | 2 | 0 | 11/12 | 9 | 0.191 | 18.5 | 23.13 |
| local-deepseek-ocr2 | 13 | 0 | 0 | 0.026 | 0.100 | 0.208 | 4/15/0 | 0 | 0 | 5 | 4/13 | 0 | 0 | 0 | 0/12 | 22 | 0.000 | 0.0 | 0.00 |
| local-dots-ocr | 13 | 1 | 0 | 0.059 | 0.060 | 0.158 | 1/14/0 | 0 | 6 | 9 | 13/13 | 0 | 2 | 0 | 10/11 | 30 | 0.000 | 0.0 | 0.00 |
| local-marker | 13 | 0 | 0 | 0.036 | 0.193 | 0.355 | 0/14/0 | 0 | 0 | 6 | 11/13 | 0 | 0 | 1 | 0/12 | 29 | 0.000 | 0.0 | 0.00 |
| local-paddleocr-vl | 13 | 0 | 0 | 0.037 | 0.089 | 0.177 | 5/4/0 | 0 | 0 | 5 | 11/13 | 0 | 0 | 0 | 0/12 | 19 | 0.000 | 0.0 | 0.00 |
| mistral-small-4 | 13 | 1 | 0 | 0.087 | 0.141 | 0.121 | 6/2/12 | 6 | 6 | 5 | 11/13 | 2 | 2 | 0 | 11/12 | 8 | 0.094 | 9.2 | 11.46 |
| qwen3-vl-235b | 13 | 2 | 2 | 0.069 | 0.090 | 0.162 | 8/10/1 | 6 | 6 | 3 | 10/13 | 0 | 1 | 0 | 9/10 | 28 | 0.645 | 62.6 | 78.28 |
| qwen38-flash | 13 | 0 | 0 | 0.073 | 0.052 | 0.148 | 8/1/9 | 6 | 7 | 7 | 13/13 | 0 | 2 | 0 | 11/12 | 11 | 0.077 | 7.5 | 9.40 |

Как читать: *CER к FR* — расстояние Левенштейна до текстового слоя FineReader, делённое на
его длину (оба текста нормализованы: без разметки, переносов, регистра); это не истина —
на таблицах и оглавлении FineReader сам ошибается, и там CER 0,08-0,15 у всех, — поэтому
рядом *согласие*: средний CER до остальных моделей на той же полосе. На чистом тексте
(IMG_0107_2R) лучшие модели расходятся с FineReader на 0,1 % знаков — то есть буквы они
читают не хуже. *повёрн.* — сколько из 13 уверенно прочитанных tesseract'ом боковых шапок
(`pack1_rotated_tables/info`) нашлось в выходе модели.

### Что видно глазами

*Таблица с боковыми шапками (IMG_0114_2R, «Таблица 3»).* Gemini 3.1 Lite, Gemini 3.8, GLM-5.3
Flash и dots.ocr отдали безупречный `<table>` с `rowspan`/`colspan` и всеми четырьмя
повёрнутыми шапками («остатки, тыс. т физического веса», «использование складских
емкостей, %»), все 48 чисел верны. DeepSeek V4.1 и Qwen3.8 Flash — то же в GFM (DeepSeek
сократил шапку до «использование емкостей»). GPT-5.6 Luna схлопнул двухуровневую шапку в
одну строку — допустимо. Claude Haiku потеряла шапку целиком (первая строка данных стала
заголовком) и выронила «в размере 1958» из примечания. Mistral Small 4 **выдумала** шапку
«остаток на 1-е число, тыс. т». DeepSeek-OCR-2 (локально) выбросил все 12 строк данных и
придумал шапки («Средняя стоимость, руб.», в другом прогоне — «всего»), год прочёл как
«1994». Marker прочёл повёрнутые шапки (surya умеет), но строки таблицы перепутал
(«Ha I thermand», сдвиги значений).

*Чертёж с таблицей (IMG_0121_2R).* Все облачные модели вытащили номер страницы «33» и
колонтитул «3 Материально-техническое снабжение, № 3» в поля (Gemini Lite и Haiku положили
колонтитул в `running_header`, хотя он внизу). В таблице «Состав складов» Haiku и Mistral
сдвинули строки (у «Резервного парка» стоят значения строки выше), остальные верны.
Чертёж все, кроме DeepSeek целой полосой и Mistral, описали как `[картинка: чертёж
секции стеллажного хранения …]`, Gemini 3.8 — дважды.

*Рубрика + заголовок + два автора (IMG_0123_2R).* Все облачные модели дали ровно ожидаемое:
`### ОПЫТ РАБОТЫ ТЕРРИТОРИАЛЬНЫХ УПРАВЛЕНИЙ`, `# РАЗВИВАТЬ РАЦИОНАЛЬНЫЕ ХОЗЯЙСТВЕННЫЕ
СВЯЗИ`, `**И. КОМАРОВСКИЙ,**` / `*начальник Управления …*`, `**М. КРУГМАН,**` /
`*начальник отдела …*`, номер «37». Haiku: «КРУГЛАН». dots.ocr на этой полосе вернул 11
рамок без текста (пустой выход — единственный его сбой), DeepSeek-OCR-2 и Marker дали
текст без разметки автора и должности.

*Шапка журнала + первая статья (IMG_0105_2R).* Разброс только в том, куда положить шапку
(«МАТЕРИАЛЬНО-ТЕХНИЧЕСКОЕ СНАБЖЕНИЕ № 3 СЕНТЯБРЬ Год издания 1-й»): DeepSeek, GLM, Qwen,
GPT, Mistral — в `running_header`, Gemini — в тело. Haiku дописала рубрику из примера в
промпте. Mistral записала номер выпуска «3» как номер страницы.

*Оглавление (IMG_0151_2R).* С правилом про отточия все, кроме Qwen3-VL-235B, отдали список
«Автор. Название — страница» и `is_toc: true`; DeepSeek V4.1 в `notes` честно перечислил
сомнительные фамилии («В. Капамкаров», «А. Пропыгин»).

*Чистый текст (IMG_0107_2R).* DeepSeek, Gemini, GLM, Qwen — CER 0,000-0,001 к FineReader;
GPT 0,002; Haiku 0,014; dots.ocr 0,014; Mistral 0,022 (в ней же одно «[неразборчиво]»).

## 4. Весь выпуск 1966/03: 97 полос × 4 модели

По итогам пробника на полный выпуск взяты три дешёвые модели и Gemini 3.8 Flash как
эталон. Прогон `run_issue_1966_03.sh`, промпт v3 (добавлено правило про разрядку).

| модель | полос | сбоев | обрезано | CER к FR (медиана) | CER к FR (текст) | согласие | h1/h2/h3 | авторов | должн. | таблиц | повёрн. | схем | карт. | зацикл. | № стр. верно | с/полоса | ¢/полоса | ¢/выпуск | $/пак |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v41-flash-s2 | 97 | 0 | 0 | 0.003 | 0.031 | 0.022 | 37/16/19 | 38 | 20 | 12 | 16/17 | 3 | 3 | 0 | 83/96 | 7 | 0.097 | 9.4 | 11.77 |
| gemini-31-flash-lite | 97 | 0 | 0 | 0.004 | 0.023 | 0.029 | 42/9/14 | 30 | 22 | 11 | 17/17 | 3 | 5 | 0 | 82/96 | 6 | 0.225 | 21.9 | 27.34 |
| gemini-38-flash | 97 | 0 | 0 | 0.003 | 0.022 | 0.030 | 33/25/21 | 38 | 20 | 15 | 17/17 | 0 | 7 | 0 | 82/96 | 10 | 0.516 | 50.0 | 62.58 |
| qwen38-flash | 97 | 0 | 0 | 0.004 | 0.023 | 0.028 | 45/5/21 | 27 | 27 | 13 | 17/17 | 0 | 5 | 0 | 82/96 | 13 | 0.086 | 8.4 | 10.45 |

Сбои первой попытки: Gemini Lite — 1 полоса (thinking-спираль на 15 тыс. токенов при
`effort: low`; после этого для Gemini поставлен потолок `reasoning.max_tokens=1024`, повтор
прошёл), Qwen3.8 Flash — 2 полосы (модель вернула `[]` вместо JSON; повтор прошёл),
DeepSeek и Gemini 3.8 — без сбоев. Повторный запуск того же скрипта с `--skip-done`
добирает только сбойные полосы.

Номер страницы: у 14 полос выпуска номер не напечатан (концы статей, реклама) — там все
модели верно отдали `null`; на остальных 82 все четыре модели совпали с ожидаемым
(порядковый номер полосы в выпуске), кроме двух галлюцинаций на табличных полосах (Gemini
Lite «45» на IMG_0140_2R; DeepSeek «11» и Gemini Lite «51» на IMG_0145_1L). Оглавление
(`is_toc`) нашли все четыре. Из 17 уверенно прочитанных tesseract'ом боковых шапок таблиц
выпуска все четыре модели нашли 17 (DeepSeek — 16).

Согласие между четырьмя моделями на тексте — 0,02-0,03 (два-три знака на сто), с
FineReader — 0,02-0,03 на текстовых полосах; разница между моделями — в счёте
подзаголовков (`##`): Gemini 3.8 и DeepSeek выделяют внутренние подзаголовки статей
(25 и 16), Qwen почти нет (5) — он чаще делает их `#`. Это вопрос вкуса промпта, не
качества чтения.

Полные выходы: `/mnt/SYSTEM/raw/mts/pack1_external_ocr/<модель>/1966/03/*.{json,md,meta.json}`,
построчная сводка — `reports/external_ocr_models_issue_1966_03.md`.

## 5. Локальные специализированные OCR-модели

Машина: RTX 5060 Ti 16 ГБ (Blackwell, sm_120), torch 2.12 + CUDA 13. Каждый движок — в своём
окружении (`research/external_ocr_models/local/<движок>/pyproject.toml`, `uv sync --project …`),
основной пакет зовёт `worker.py` подпроцессом; запуск — `run --model local-<движок>`.

| движок | размер | завёлся | с/полоса | VRAM | что умеет / где ломается |
|---|---|---|---|---|---|
| **dots.ocr** (rednote-hilab, 3B, MIT) | 6 ГБ весов | да: transformers 4.53 (4.54+ ломает процессор — `video_processor`), веса в папку без точки в имени, башне зрения принудительно `sdpa` (иначе eager и OOM) | 27 | 6,7 ГБ | layout + текст одной моделью, таблицы с боковыми шапками — идеальный HTML, колонтитулы отдельными элементами (из них берём номер страницы); буквы хуже облачных (CER 0,014 на чистом тексте), одна полоса из 13 — рамки без текста |
| **DeepSeek-OCR-2** (3B, Apache) | 6 ГБ | да: transformers 4.46.3 (пин модели) на torch 2.12 без flash-attn, `attn=eager`, веса грузить сразу в bf16 (иначе OOM на загрузке) | 20 | 7,2 ГБ | markdown с заголовками; таблицы с боковыми шапками теряет (выбрасывает тело, выдумывает шапки), нестабилен между прогонами, разрядку пишет буквально, промпт дописывать нельзя — не инструктивен |
| **Marker 1.10** (datalab, surya 0.17) | ~1 ГБ моделей | да; Marker 2.x / Surya 2 требуют vllm в docker с nvidia-runtime или бинарник llama-server — не ставились | 29 | 5,5 ГБ | markdown с уровнями заголовков, повёрнутые шапки читает, но строки таблиц перепутывает и отточия превращает в «20-20-20»; авторов/должностей/номеров нет |
| **PaddleOCR-VL 1.6** (0.9B, Apache) | 2 ГБ рантайм + модели | да, на GPU: колесо `paddlepaddle-gpu` cu130 качается только с источника `paddle-whl.bj.bcebos.com` (CDN из индекса молчит), без torch в окружении (конфликт по cuDNN 9.13 / 9.20) | 19 | 1,3 ГБ | полный пайплайн PP-DocLayout + VLM; буквы на уровне облачных (CER 0,002 на чистом тексте), таблицы структурно верны (rowspan/colspan, все числа), но боковые шапки с ошибками («фи-знческого», латинское «r.» вместо «г.»), широкий заголовок без пробелов («ВНОВЫХУСЛОВИЯХ»), переносы внутри ячеек не склеены, HTML с инлайн-стилями; авторов/должностей/номеров страниц нет |
| GLM-OCR (0.9B, MIT) | — | не пробовался | — | — | как и PaddleOCR-VL, поэлементная модель: нужен PP-DocLayoutV3 из paddleocr; смысл появляется только если paddle-стек заведётся |
| olmOCR-2, Chandra OCR 2 | 7-9B | не пробовались | — | — | в 16 ГБ только в 4 битах, ориентированы на английский |

Вывод по локальным: они годятся как бесплатный третий голос по таблицам (dots.ocr) и как
резерв, если облако недоступно, но полей «автор / должность / рубрика / номер страницы» не
дают, а буквы читают хуже Gemini/Qwen/DeepSeek. Второй проход дешёвой текстовой моделью
по их markdown стоил бы столько же, сколько сразу отдать картинку Gemini Lite.

## 6. Специализированные OCR-API у других провайдеров

Их нет на OpenRouter, нужен отдельный ключ. Все они дают текст + layout (заголовки по
размеру шрифта, таблицы, картинки), но **не** семантику «автор / должность / рубрика» — её
всё равно пришлось бы доставать языковой моделью вторым проходом. Цены за 1000 страниц:

| сервис | цена | языки | что даёт | замечание |
|---|---|---|---|---|
| Mistral OCR 4.1 (`mistral-ocr-latest`) | $4 (батч $2); с аннотациями $5 | 170, русский есть | markdown с заголовками, таблицы, картинки, bbox | самый близкий по духу к нашей задаче; за выпуск 39 ¢ |
| Yandex Vision OCR | $1.08 текст; $10 таблицы; $12.5 рукопись | русский родной | строки, блоки, таблицы (JSON) | структуры статьи нет; дёшево для «букв» |
| Google Document AI, Enterprise OCR | $1.50 (свыше 5 млн стр. — $0.60) | русский есть | текст, блоки, параграфы; Layout Parser — отдельно $10+ | нужен GCP-проект |
| Azure Document Intelligence | Read $1.50; Layout $10 | русский есть | Read — текст; Layout — таблицы, заголовки (markdown) | нужна подписка Azure |
| Datalab (Marker/Chandra API) | ~$3-6 (по тарифу) | 90+ | markdown/JSON с layout | облачная версия локального Marker |

Для сравнения: Gemini 3.1 Flash Lite через OpenRouter — $1.9-2.5 за 1000 полос, Qwen3.8
Flash — $0.85, DeepSeek V4.1 Flash (2 куска) — $0.9, причём со структурой статьи. То есть
специализированные API не дешевле и не структурнее; их преимущество — координаты слов и
предсказуемость (нет галлюцинаций и циклов), что важно для слияния с FineReader.

## 7. Рекомендации

1. **Основной вариант — Gemini 3.1 Flash Lite** по всему паку: ≈ $27 за 12 135 полос,
   ~6 с на полосу при 4 потоках (≈ 5 часов на пак; можно 8 потоков). Промпт v3, JSON-схема,
   `reasoning.max_tokens=1024`, картинка 2200 px целиком.
2. **Второй и третий голос — Qwen3.8 Flash и DeepSeek V4.1 Flash (2 куска)**: ещё ≈ $22 на
   пак. Три независимые версии структуры дают голосование при слиянии: заголовок, автор,
   должность и номер страницы берутся по большинству; таблицы — из Gemini (HTML), при
   расхождении чисел — сверка с FineReader.
3. **Не тратить** на Gemini 3.8 Flash / Pro, Claude, Qwen3-VL-235B: качество не выше, цена
   в 3-10 раз больше, у Qwen-235B — циклы.
4. **Слияние с FineReader** — следующая задача: по чистому тексту версии совпадают на
   99,9 %, так что выравнивание построчно тривиально; спорные места — таблицы (брать
   VLM), кривые строки и пятнистый фон (брать FineReader, он видел распрямлённую
   бинаризованную полосу), оглавление (VLM).
5. **Что ещё проверить перед паком**: год 1966 — самый чистый; на 1974-76 (пятнистый
   фон, петит) стоит прогнать тот же пробник (`run_probe_1966_03.sh` с другим списком
   полос) и посмотреть, не растёт ли расхождение с FineReader. И включить в пак-скрипт
   `--skip-done` + повтор: сбои первой попытки — 1-2 %, повтор их закрывает.

## 8. Достраивание повреждённых букв (`--restore`)

Вопрос: может ли модель догадываться по контексту, что было в буквах, срезанных корешком
или расплывшихся, и помечать достроенное. Опыт на чистой полосе IMG_0107_2R (у неё есть
точный текст): слева срезано 1-3 знака каждой строки (как под тугой подшивкой), полоса из
пяти строк в середине размыта. Три прогона DeepSeek V4.1 Flash (2 куска) и один Gemini
3.1 Flash Lite:

| вариант | тегов `<restored>` | расхождений с чистой полосой (из 3557 знаков) |
|---|---|---|
| правило про повреждения подпунктом в промпте, без подсказки | 0 | 4 |
| правило отдельным пунктом + поле `restored` в JSON, без подсказки | 0 | 3 |
| то же + подсказка про полосу («левый край срезан корешком, в середине размытая полоса») | **52** | 3 |
| Gemini 3.1 Flash Lite, то же с подсказкой | 1 | 3 |

Выводы:
* **Достраивают все и всегда, молча.** Это языковые модели: срезанное «ногоассортиментным»
  они читают как «многоассортиментным», не замечая, что буквы не видели. Без явной
  подсказки DeepSeek считает это чтением и не помечает ничего.
* **С подсказкой про конкретную полосу DeepSeek помечает честно**: 52 фрагмента,
  `<restored>ра</restored>йона`, `может <restored>бы</restored>ть`, размытые строки
  обёрнуты целиком, слова продублированы в поле `restored`. Точность самого достраивания —
  3 ошибки на 3557 знаков, все в размытой полосе. Gemini Lite подсказку почти игнорирует
  (1 тег из ~50).
* Отсюда схема: детектор повреждений на нашей стороне (`ocr_utils/gutter_loss_detection`
  уже умеет находить полосы с уходом под корешок и сторону) → для таких полос запуск с
  `--restore --hint "…"`, где подсказка говорит, какой край и что именно повреждено →
  в выходе теги `<restored>` и список слов; при слиянии с FineReader такие места
  проверяются по его версии (он видел ту же бинаризованную полосу) или по словарю
  выпуска (`gutter_loss_restoration/lexicon.py`).
* Что реализовано: `run --restore [--hint "…"]` — правило в промпте (Jinja-блок в
  `prompts/system.md.j2`), поле `restored` в схеме и в `.json`, подсказка в
  пользовательском сообщении. Подсказка пока одна на прогон; следующий шаг — брать её
  по полосе из отчёта детектора корешка.

## 9. Промпты и настройки запросов — как есть

Всё ниже — актуальное состояние `research/external_ocr_models` (промпт v12; блоки
`--damage` — в 9.7); история версий — в 9.6.

### 9.1. Системный промпт (`prompts/system.md.j2`, режим `json`, v12 — без указания издания)

```text
You are a meticulous OCR and document-structure transcriber. You receive a scan of ONE page of Soviet or post-Soviet economic press — a journal or a newspaper printed between the 1920s and the 1990s. The text is Russian in the orthography of its time, with occasional Latin abbreviations, brand names and formulas. Newspaper pages are usually set in several narrow columns with small type; journal pages in one or two columns.

Transcribe the page exactly and mark up its structure. Rules:

1. Verbatim text. Keep the printed spelling, punctuation, numbers and units. Do not correct, modernise, translate, summarise or reorder. Never invent text that is not on the page; write [неразборчиво] for an unreadable fragment.
2. Lines and paragraphs. Join words hyphenated across a line break («снабже-» + «ния» → «снабжения»), but keep real hyphens in compound words («материально-техническое»). Merge the lines of a paragraph into one line; separate paragraphs with a blank line. On multi-column pages read the columns in order, left to right, each column fully before the next; an article that continues in the next column or under a heading spanning several columns is one text — keep its paragraphs together.
   Dot leaders — rows of dots or dashes that fill the space before a number (in tables of contents, in table rows) — are NOT text: never reproduce them; write the entry, then « — », then the number («В. Тычинин. Первые шаги работы по-новому — 1»).
3. Structure (Markdown):
   - Article title → `# Заголовок статьи` (case as printed).
   - Headings inside an article → `## Подзаголовок`.
   - Rubric printed above the title («Опыт работы территориальных управлений», «Письма читателей», «Консультация», «Информация») → `### Рубрика`, placed before the title. A page may carry several independent articles or news items (typical for newspapers): give each its own `#` title; a subtitle or lead paragraph set in larger or bold type right under the title → `## Подзаголовок`.
   - Author name → its own paragraph in bold: `**И. Фетисов**` — wherever it is printed (under the title, at the end of the article, or as a signature like «Наш корр.»).
   - Author's position and regalia → its own paragraph in italics right after the name: `*начальник УМТС Московского городского района, член коллегии Госснаба СССР*`.
   - Tables → ALWAYS an HTML `<table>` (never a Markdown pipe table): one `<tr>` per printed line of the table — a sub-item printed on its own line («в том числе хлопка») is its own row, never several lines stacked in one cell with `<br>`; `<th>` for header cells, `rowspan`/`colspan` for merged cells and multi-level headers, one `<td>` per cell even if it is empty; a section heading inside the table («А. Ресурсы») → one row with a cell spanning all columns. Several row labels joined by a brace «}» to one shared value → keep each label in its own row and give the shared value cells `rowspan` over those rows. A cell never contains `<br>`. In the HEADER, a column title printed on several lines is ONE `<th>` with the lines joined by a space and a hyphenated word joined WITHOUT the hyphen, as in rule 2 («Тип дви-» / «гателя» → `<th>Тип двигателя</th>`, «зарпла-» / «та» → «зарплата»). In the BODY this does not apply: every printed line stays its own row — a label line without numbers («Остаток на начало периода:») is its own row with empty value cells, and the sub-items under it are the following rows. Text printed vertically (rotated 90°) inside cells must be read and written as normal horizontal text. Keep the caption («Таблица 3») and the table title as paragraphs before the table, not inside it.
   - Flowcharts and block diagrams → a block quote starting with `> [блок-схема]`, then the text of every block in reading order, one block per line, with `→` between connected blocks.
   - Photographs, drawings, decorative graphics → `> [картинка: краткое описание]`, e.g. `> [картинка: портрет мужчины в костюме]`.
   - Footnotes → `[^1]` in the text and `[^1]: текст сноски` at the end. Lists → Markdown lists. Text printed in bold → **bold**. Letter-spaced text (разрядка: «П р и м е ч а н и е») → write the word normally, without spaces between letters, in italics: *Примечание*.
4. Running header and footer (publication name, issue number, date, page number printed in the top or bottom margin; on a newspaper front page — the masthead with the name, date, price and founder) are NOT part of the body: put them into the dedicated fields and leave them out of content_markdown. `page_number` is only the page number; an issue number («№ 3») or a year is not a page number.
5. If the page is the issue's table of contents («СОДЕРЖАНИЕ»), set is_toc to true and transcribe it as a Markdown list of «Автор. Название — страница».
Return ONLY a JSON object with exactly these keys and nothing else:
{"page_number": string or null (page number as printed, e.g. "12"),
 "running_header": string or null,
 "running_footer": string or null,
 "is_toc": boolean,
 "content_markdown": string (the body of the page in Markdown per the rules; use \n for line breaks),
"notes": string (uncertainties such as unreadable areas; empty string if none)}
```

В режиме `--output-mode markdown` хвост после правила 5 заменяется на:

```text
Return the page as Markdown preceded by a YAML front matter block, exactly in this shape:
---
page_number: "12" or null
running_header: "text" or null
running_footer: "text" or null
is_toc: false
notes: ""
---
<the body of the page in Markdown per the rules>
```

С `--damage` после правила 5 добавляется правило 6, а в описание JSON — поля повреждений
(подробно — 9.7; здесь показана первая редакция, `--restore`, из раздела 8):

```text
6. Damaged text. Some letters may be physically missing or unreadable: cut off by the binding gutter at the page edge, smeared, faded, torn. Reconstruct such letters from context — this is expected and desirable — but you MUST mark every reconstructed run of characters inline with `<restored>…</restored>`, wrapping only the characters you did not actually see (e.g. `<restored>м</restored>ногоассортиментным`, `снаб<restored>же</restored>ния`). If a fragment cannot be reconstructed with confidence, write `[неразборчиво]`. Never wrap clearly legible text. Also list every reconstructed word in the `restored` field.
"restored": array of strings (every word containing reconstructed characters, as written in content_markdown; empty array if none),
```

### 9.2. Пользовательское сообщение (`prompts/user.md.j2`)

Идёт первой текстовой частью, за ним — картинка (или N кусков) как `image_url`:

```text
Transcribe this page.
```

При `--strips N` (N ≥ 2) перед этим:

```text
The page is given as 3 horizontal strips in order from top to bottom; neighbouring strips overlap slightly. Treat them as ONE page: transcribe it once, in reading order, without repeating the overlapping lines.
Transcribe this page.
```

При `--hint "…"` текст подсказки ставится перед «Transcribe this page.» — в опыте с
повреждённой полосой он был такой: «The left edge of this page is cut off by the binding
gutter: the first one to three letters of many lines are missing. One band of lines in the
middle of the page is smeared.»

### 9.3. Тело запроса к OpenRouter (`ocr.py::build_payload`)

```json
{
  "model": "<id из реестра>",
  "messages": [
    {"role": "system", "content": "<системный промпт>"},
    {"role": "user", "content": [
      {"type": "text", "text": "<пользовательское сообщение>"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,…", "detail": "high"}}
    ]}
  ],
  "temperature": 0,
  "max_tokens": 16000,
  "response_format": {"type": "json_schema", "json_schema": {"name": "page_transcription", "strict": true, "schema": {…6 полей, additionalProperties=false…}}},
  "reasoning": {"enabled": false},
  "provider": {"require_parameters": true}
}
```

* `temperature: 0` везде — нам нужна транскрипция, не вариативность.
* `max_tokens: 16000` — полоса это 2-4 тыс. токенов, таблица в HTML вдвое больше;
  потолок отсекает зацикливание (цикл по отточиям стоил 2,5-3 ¢ — ровно 16k токенов).
* `response_format`: сначала `json_schema` со строгой схемой (`page_number`, `running_header`,
  `running_footer`: string|null; `is_toc`: boolean; `content_markdown`, `notes`: string) и
  `provider.require_parameters=true`, чтобы маршрутизатор не отдал запрос провайдеру без
  structured outputs. Если ответ 400/404/422 — тот же запрос с `{"type": "json_object"}`, затем
  без `response_format`; использованный режим пишется в `.meta.json` (`json_mode_used`).
  У DeepSeek стартовый режим — `json_object` (родной эндпоинт схем не умеет), GPT-5.6 Luna
  по факту всегда падал в `json_object` (404 на `json_schema`).
* `reasoning`: `{"enabled": false}` — если провайдер отвечает 400 с упоминанием
  `reasoning`, запрос повторяется без поля (так ведёт себя GLM-5.3 Flash). Для Gemini
  thinking не выключается — там `{"max_tokens": 1024}` (после thinking-спирали на 15 тыс.
  токенов). Для Qwen3-VL-235B-Instruct поле не шлётся (модель без thinking).
* `provider`: `order=["deepseek"]` + `allow_fallbacks=true` и `ignore=["relace"]` (fp4)
  только у DeepSeek; остальные — на усмотрение маршрутизатора (самый дешёвый живой).
* Заголовки: `Authorization: Bearer $OPENROUTER_API_KEY`, `HTTP-Referer`, `X-Title`; ключ
  в логи и meta не пишется.

Настройки по моделям пробника:

| имя | id | режим JSON | reasoning | провайдеры | detail |
|---|---|---|---|---|---|
| deepseek-v41-flash | `deepseek/deepseek-v4.1-flash` | json_object | `{'enabled': False}` | order=deepseek, ignore=relace | high |
| gemini-31-flash-lite | `google/gemini-3.1-flash-lite` | json_schema | `{'max_tokens': 1024}` | — | high |
| qwen38-flash | `qwen/qwen3.8-flash` | json_schema | `{'enabled': False}` | — | high |
| gpt-56-luna | `openai/gpt-5.6-luna` | json_schema | `{'enabled': False}` | — | high |
| mistral-small-4 | `mistralai/mistral-small-2603` | json_schema | `{'enabled': False}` | — | high |
| glm-53-flash | `z-ai/glm-5.3-flash` | json_schema | `{'enabled': False}` | — | high |
| gemini-38-flash | `google/gemini-3.8-flash` | json_schema | `{'max_tokens': 1024}` | — | high |
| qwen3-vl-235b | `qwen/qwen3-vl-235b-a22b-instruct` | json_schema | `None` | — | high |
| claude-haiku-45 | `anthropic/claude-haiku-4.5` | json_schema | `{'enabled': False}` | — | high |

Оговорка по хронологии: пробник и первый проход полного выпуска шли для Gemini с
`{'effort': 'low'}`; потолок `max_tokens=1024` появился после thinking-спирали на полосе
IMG_0116_1L и использовался при повторе сбойных полос. Локальные движки промптов не
получают вовсе (у dots.ocr и DeepSeek-OCR-2 — их штатные промпты, см. `local/*/worker.py`).

### 9.4. Картинка (`imaging.py`)

* Исходник 600 dpi RGB → серый (`ImageOps.grayscale`) → длинная сторона
  `--max-side 2200` px (LANCZOS; ≈216 dpi для полосы 10″) → JPEG `--quality 85`,
  `optimize=True` → 0,5-0,7 МБ → base64 data-URL. Меньше исходника — не увеличиваем.
* `--strips N`: полоса режется в исходном разрешении на N горизонтальных кусков с
  перекрытием 8% высоты (≈5-6 строк), каждый кусок уменьшается до `--max-side`
  отдельно, все N идут одним запросом. Использовано только для DeepSeek (N=2, N=3).
* `image_url.detail = "high"` (для DeepSeek — оригинал до потолка 1024 токенов; остальные
  провайдеры поле игнорируют или трактуют как у OpenAI).

### 9.5. Сеть и прогон (`client.py`, `cli.py`)

* Таймаут 300 с, 5 попыток с паузой `min(60, 2·2^k)` + джиттер на 408/409/425/429/5xx и
  сетевых ошибках; 400/401/402/403 не повторяются. Ошибка провайдера внутри 200-го ответа
  обрабатывается так же.
* `--jobs 4` потоков на модель (это сеть; на 8 потоках Fireworks ловил 429). `--skip-done`
  пропускает полосы с `.meta.json` без `error`/`parse_error`, так что повтор скрипта
  добирает только сбойные.
* Стоимость — `usage.cost` из ответа (OpenRouter кладёт его всегда), токены —
  `usage.prompt_tokens`/`completion_tokens`/`completion_tokens_details.reasoning_tokens`.
* Разбор ответа (`schema.py`): снять ```-ограждения → `json.JSONDecoder.raw_decode` первого
  объекта (хвост игнорируется) → поля; пустые строки → `null`; сбой разбора → `.raw.txt` и
  `parse_error` в meta. После разбора к телу применяется `unspace_letters()`:
  «П р и м е ч а н и е» → «*Примечание*».

### 9.6. История версий промпта (`PROMPT_VERSION`)

* **v1** — исходная формулировка (правила 1-5, разметка, поля).
* **v2** — добавлен запрет отточий в правило 2 («Dot leaders … are NOT text: never reproduce
  them; write the entry, then « — », then the number»): в пробнике v1 три полосы (оглавление
  и таблица с отточиями) ушли в цикл до `max_tokens`.
* **v3** — правило про разрядку в пункте 3 («Letter-spaced text (разрядка: «П р и м е ч а н и е»)
  → write the word normally … in italics») по замечанию заказчика; полный выпуск гнался на v3.
* блок `--restore` (не меняет версию: включается флагом) — правило 6 и поле `restored`;
  первая редакция была подпунктом правила 3 и без подсказки не работала.
* **v4-v7** — режим `--damage` вместо `--restore`: три тега, поля `damage`/`restored`/`fuzzy`/
  `unknown`, подсказки по страницам (v4); `damage` первым полем, требование тегов у
  картинки (v5); `edge_words` (v6); правило про перенос (v7). Формат — 9.7, замеры — раздел 10.
* **v8** — промпт обобщён с одного журнала на советскую и постсоветскую экономическую
  прессу 1920-х — 1990-х, журналы и газеты: описание источника стало общим (конкретное
  издание — опцией `--source "журнал «…», Москва, 1966"`, попадает в первую фразу), добавлены
  газетные особенности (несколько узких колонок мелким шрифтом, несколько независимых
  статей на полосе — каждой свой `#`, лид под заголовком → `##`, подпись автора в конце
  статьи или «Наш корр.», шапка первой полосы газеты с датой и ценой — в колонтитул) и
  уточнение «номер выпуска и год — не номер страницы».
  Проверка на DeepSeek V4.1 Flash (2 куска): 13 полос пробника — 9 побайтно те же, что с v2,
  3 — в пределах 0,4 % знаков, на IMG_0134_1L модель отдала GFM-таблицу со схлопнутой
  шапкой вместо HTML (содержание то же), на обложке не описала картинку; «№ 3» из шапки
  попал в `page_number` — исправлено уточнением, повтор дал `null`. Мини-набор повреждённых
  страниц: пометки на тех же страницах, что с v7; один прогон дал 46 `<unknown/>` на
  сплющенной IMG_0068_L, два повтора той же страницы — 1/0/0: это стохастика провайдера
  при `temperature 0`, а не промпт.
* **v9-v10** — таблицы всегда в HTML `<table>` (GFM запрещён): по полному выпуску Gemini Lite
  и так отдавал 11 из 11 таблиц в HTML, а DeepSeek — 3 из 11, остальные GFM со схлопнутой
  многоуровневой шапкой. v9 («always HTML, one `<tr>` per printed row») — DeepSeek на всех
  9 табличных полосах выпуска отдал HTML, текст вне таблиц не изменился (CER к прежнему
  выходу 0,000-0,02), на 5 полосах из 9 структура совпала с Gemini до ячейки (строки,
  ячейки, объединения); но подпункты нумерованных строк («в том числе хлопка») он складывал
  в одну ячейку через `<br>` (IMG_0144_1L: 10 строк вместо 23). v10 («one `<tr>` per printed
  line, sub-item on its own line — its own row, never `<br>`-stacking; section heading inside
  the table — a row spanning all columns») — IMG_0144_1L: 22 строки, `<br>` нет, столбец с
  номерами пунктов сохранён (Gemini его выбросил); на IMG_0134_1L (шапка в три уровня)
  DeepSeek теперь **лучше Gemini**: rowspan/colspan по всем трём уровням и по строке на
  двигатель, тогда как Gemini сложил все четыре двигателя в одну строку через `<br>`.
  Осталось: скобка «}», объединяющая две строки в одно значение, разбита на две строки
  (значения в первой) вместо rowspan.
* **v11-v12** — доводка таблиц по замечаниям: «}» между строками → `rowspan` на общих
  значениях (сработало: IMG_0134_1L — `rowspan="2"` у ДК-259/ДК-207, на IMG_0145_1L
  объединений стало 6 вместо 1); `<br>` в ячейках запрещён — многострочная шапка
  склеивается пробелом (сработало: «Всего годовая экономия», «коллектор на миканите»);
  перенос внутри шапки склеивать без дефиса — **не сработало** даже с явным
  отрицательным примером: «Тип дви-гателя», «пласт-массе» модель пишет с дефисом в
  каждом из четырёх прогонов. Побочный эффект первой редакции v12 («строки ячейки
  склеивать пробелом») — заголовочная строка тела («Остаток на начало периода:»)
  сливалась с первым подпунктом (18 строк вместо 22); исправлено оговоркой «только в
  шапке, в теле каждая печатная строка — своя». Итог по 9 табличным полосам: `<br>` нет
  нигде, IMG_0144_1L — 22 строки (Gemini 23), IMG_0134_1L — 8 строк с 16 объединениями
  (Gemini 5 строк, все двигатели в одной ячейке через `<br>`). Повторы одной полосы
  показывают стохастику: один прогон v11 отдал IMG_0144_1L с 10 строками и 28 `<br>`,
  три повтора — 22 строки без `<br>`; на IMG_0122_2R «в том числе вычислительный центр
  25/30/30» внутри ячеек модель то пишет в тех же ячейках, то отдельными строками.

### 9.7. Режим повреждённых сканов (`--damage`): что уходит в запрос и что приходит в ответ

Включается флагом `run --damage`; без него ни правила, ни полей ниже в запросе нет.
Промпт v8.

**Теги в теле (`content_markdown`)** — размечают только повреждённые места, остальной текст
остаётся обычным markdown по правилам 1-5:

| тег | что значит | пример |
|---|---|---|
| `<restored>…</restored>` | буквы, которых на скане физически нет (ушли под корешок, оторваны, закрыты), достроены по контексту и по видимым остаткам слова; оборачиваются только невидимые символы | `снабже<restored>ния</restored>`, `<restored>м</restored>ногоассортиментным` |
| `<fuzzy>…</fuzzy>` | буквы видны, но ненадёжны (размыты или сплющены у сгиба, выцвели в пересвете — «е/с», «н/и», «п/л»); прочитаны по форме и контексту | `пр<fuzzy>е</fuzzy>дприятий` |
| `<unknown/>` | фрагмент, который не удалось ни прочитать, ни достроить; ставится на месте пропавших букв, видимая часть слова остаётся как есть | `предпри<unknown/>`, `<unknown/>ности` |

Правила для модели: чётко читаемое не помечать, теги не вкладывать, текст вне
повреждённых мест не «улучшать», дефис в конце строки с продолжением на следующей —
обычный перенос, а не повреждение (склеить молча).

**Правило 6 системного промпта** (добавляется к правилам 1-5 из 9.1, дословно):

```text
6. Damaged text. This scan is damaged (see the note in the user message): some letters are physically missing or unreliable. Handle them like this and mark EVERY such place inline:
   - Letters that are NOT visible at all (hidden in the binding gutter, torn off, covered) but can be reconstructed with confidence from context and the visible remains of the word → write them wrapped in `<restored>…</restored>`, wrapping ONLY the invisible characters: `<restored>м</restored>ногоассортиментным`, `снабже<restored>ния</restored>`.
   - Letters that are visible but unreliable (smeared or squashed near the gutter, washed out by overexposure, so that «е»/«с», «н»/«и», «п»/«л» could be confused) → read them by shape and context and wrap the doubtful characters in `<fuzzy>…</fuzzy>`: `пр<fuzzy>е</fuzzy>дприятий`.
   - A fragment that can be neither read nor reconstructed with confidence → put `<unknown/>` exactly where the missing letters are and keep the visible part of the word as it is: `предпри<unknown/>`, `<unknown/>ности`.
   Never wrap clearly legible text; never nest tags; do not «improve» text outside the damaged places. A hyphen at the end of a line with the rest of the word on the next line is ordinary hyphenation, not damage: join the word silently and do not list or tag it.
   Besides the inline tags, fill `edge_words`: one entry for EVERY line that touches the damaged edge or area, with the affected word exactly as you see it (`seen`), the word as you write it in the text (`full`, with the same tags) and `kind`: "hidden" (letters not visible, reconstructed), "fuzzy" (visible but unreliable) or "unknown" (could not reconstruct). Go through the lines top to bottom; a damaged line with no entry is an error.
```

**Описание JSON** в системном промпте в этом режиме (поле `damage` намеренно первое —
модель описывает повреждение до того, как начнёт писать текст, и потом помечает
последовательнее):

```text
Return ONLY a JSON object with exactly these keys and nothing else:
{"damage": string (FIRST: what damage you actually see on this page — which edge or area, of what kind, roughly how many lines are affected; empty string if none),
 "page_number": string or null (page number as printed, e.g. "12"),
 "running_header": string or null,
 "running_footer": string or null,
 "is_toc": boolean,
 "content_markdown": string (the body of the page in Markdown per the rules; use \n for line breaks),
"restored": array of strings (every word containing <restored> characters, as written in content_markdown; empty array if none),
 "fuzzy": array of strings (every word containing <fuzzy> characters, as written; empty array if none),
 "unknown": integer (how many <unknown/> markers are in content_markdown),
 "edge_words": array of objects {"seen": string, "full": string, "kind": "hidden" | "fuzzy" | "unknown"}, one per damaged line in reading order,
"notes": string (uncertainties such as unreadable areas; empty string if none)}
```

**Пользовательское сообщение** (перед картинками): общая подсказка `--hint`, строка для этой
страницы из `--hints` (файл «путь<TAB>текст», для мини-набора — `run_scripts/external_ocr_models/damaged_hints.txt`),
автоподсказка по стороне при `--damage-side auto` (по суффиксу `_L`/`_R`: правый или левый
край строк у корешка), затем требование про теги — оно повторено здесь, рядом с картинкой,
потому что на длинной полосе правило из системного промпта модель теряет. Пример для
`IMG_0008_L` с двумя кусками:

```text
The page is given as 2 horizontal strips in order from top to bottom; neighbouring strips overlap slightly. Treat them as ONE page: transcribe it once, in reading order, without repeating the overlapping lines.
This is the LEFT page of a tightly bound volume. The RIGHT ends of the lines disappear into the binding gutter: the last one to four letters of many lines are completely hidden, not visible at all.
Transcribe this page.
 Apply rule 6 strictly: in the damaged zone every letter you did not actually see goes inside <restored>…</restored>, every letter you read by shape or context rather than by clear print goes inside <fuzzy>…</fuzzy>, and a fragment you cannot read or reconstruct becomes <unknown/>. Writing a reconstructed letter without a tag is an error.
```

Подсказка должна описывать ровно то, что есть на странице (какой край; буквы скрыты или
только искажены): без подсказки модель достраивает молча, с завышенной — записывает
переносы строк в «скрытые» буквы.

**Схема ответа** (`response_format: json_schema`, strict; для DeepSeek — `json_object` с тем же
описанием в промпте). Порядок ключей: `damage, page_number, running_header, running_footer, is_toc, content_markdown, notes, restored, fuzzy, unknown, edge_words`. Новые поля:

* `damage` — что модель видит: край/область, характер, сколько строк задето (заполняет
  точно на всех проверенных страницах — годится как самопроверка подсказки);
* `restored`, `fuzzy` — слова с соответствующими тегами, как они записаны в тексте;
* `unknown` — число маркеров `<unknown/>`;
* `edge_words` — по одной записи на каждую повреждённую строку: `seen` (слово как
  видно), `full` (как записано в тексте, с тегами), `kind` (`hidden` | `fuzzy` | `unknown`).
  Списки модель заполняет надёжнее, чем ставит инлайн-теги, поэтому после разбора
  `tags_from_edge_words()` доставляет теги в текст там, где их нет: невидимая часть — это
  `full` минус `seen` с начала или конца слова; записи с дефисом в `seen` (переносы)
  пропускаются; уже помеченные слова не трогаются.

**Пример ответа** (DeepSeek V4.1 Flash, IMG_0008_L, v7; текст сокращён):

```json
{
 "damage": "Правая страница (правая колонка) обрезана у правого края: последние 1–4 буквы многих строк не видны (уходят в корешок). Повреждены примерно 30 строк правой колонки; левая колонка читается полностью.",
 "page_number": null,
 "content_markdown": "…ает Россия. Но для того, что<restored>бы</restored> добывать нефть в необходимом коли<restored>честве</restored>, нужно нефтепромысловое обору<restored>дование</restored>. Значительная его ча<restored>сть</restored> изготавливается в Азербайдж<restored>ане</restored>, которому нет нужды наращив<restored>ать</restored> его производство. Чтобы не п<restored>окупать</restored> неф…",
 "restored": [
  "оборудование",
  "часть",
  "Азербайджане",
  "наращивать",
  "…"
 ],
 "fuzzy": [],
 "unknown": 0,
 "edge_words": [
  {
   "seen": "обору",
   "full": "обору<restored>дование</restored>",
   "kind": "hidden"
  },
  {
   "seen": "ча",
   "full": "ча<restored>сть</restored>",
   "kind": "hidden"
  },
  {
   "seen": "Азербайдж",
   "full": "Азербайдж<restored>ане</restored>",
   "kind": "hidden"
  },
  "…"
 ]
}
```

**Что пишется на диск.** `.json` — все поля выше (после доставки тегов из `edge_words`),
`.md` — тело с тегами и YAML-шапкой, `.meta.json` — дополнительно `tags`
(`{restored, fuzzy, unknown}` — счётчики по тексту), `tags_from_edge_words` (сколько
пометок вставлено из списка), `damage_seen` (поле `damage`), `tag_warning` при непарных
тегах. `report` добавляет таблицу «Повреждения: пометки модели» (полоса, счётчики, из
списка, непарные, что видит модель).

**Остальные настройки** те же, что в 9.3-9.5: `temperature 0`, `max_tokens 16000`,
`reasoning` выключен, DeepSeek — 2 куска (`--strips 2`), 2200 px, `detail: high`.

## 10. Повреждённые сканы: `<restored>`, `<fuzzy>`, `<unknown/>` (режим `--damage`)

Задача: буквы, совсем ушедшие под корешок тугой подшивки, достраивать по контексту и
помечать `<restored>`; невосстановимые — `<unknown/>`; буквы видимые, но ненадёжные
(расплывшиеся у сгиба, сплющенные, в пересвете) — читать по форме и контексту и помечать
`<fuzzy>`. Материал — 8 фото-разворотов пяти типов повреждений
(`research/external_ocr_models/damaged/сырые сканы/`), разрезанных на 10 страниц
(`run_split_damaged.sh`; линии сгиба заданы руками — автоподгонка `fit_fold` на
двухколонных страницах в 3 случаях из 8 попадала в межколонный промежуток). Модель —
DeepSeek V4.1 Flash, 2 куска, подсказка по странице (`damaged_hints.txt`), четыре
итерации промпта (v4-v7), каждая — 10 страниц, ≈3 ¢.

### Что менялось и что это дало

| итерация | что добавлено | помечено на 4 «срезанных» страницах | fuzzy на 6 «размытых/сплющенных/пересвет» |
|---|---|---|---|
| v4: правило 6 с тремя тегами, поля `restored/fuzzy/unknown/damage` в JSON, подсказка по странице | — | 2 из 4 (41 и 58 тегов), на двух — 0 при том, что поле `damage` перечисляет срезанные слова | 0 |
| v5: поле `damage` первым в JSON; требование тегов повторено в пользовательском сообщении рядом с картинкой | | 3 из 4 (46, 28, 20), IMG_0008_L — 0 | 1 |
| v6: поле `edge_words` — список «как видно / как записано / вид» по каждой повреждённой строке; из него теги доставляются в текст кодом (`tags_from_edge_words`) | | 3 из 4 (48, 30, 62); IMG_0008_L — 0; на сплющенной 0070_L список из 35 «скрытых» — все ложные (переносы строк) | 9 (0070_L, из списка) |
| v7: правило «дефис в конце строки — перенос, не повреждение»; подсказки без «часть букв может быть скрыта» там, где ничего не скрыто | | **4 из 4** (48, 30, 49, 67) | 14 (IMG_0051_L, из списка), 0070_L — 1 |

### Что видно глазами (кропы краёв в полном разрешении)

* **Достраивает правильно.** На IMG_0008_L все 14 проверенных срезанных слов правой
  колонки восстановлены верно (поддер→поддержания, ур→уровне, сконцен→сконцентрировать,
  веде→ведение, това→товарообращения, реали→реализации…); на IMG_0051_L (размытие) 19
  проверенных концов строк прочитаны верно и правильно склеены через перенос; на 0070_L
  (сплющено) — все проверенные концы верны. Ошибок чтения в повреждённых зонах в
  выборке не нашёл.
* **Помечает неохотно и приблизительно.** Достроенные буквы модель по умолчанию считает
  прочитанными: в v4-v6 одна и та же страница (IMG_0008_L) три раза подряд шла без единого
  тега, хотя в поле `damage` модель сама писала «последние 1-4 буквы скрыты» и в `notes` —
  «восстановлены по смыслу». Помогло сочетание: описание повреждения *первым* полем,
  повтор требования у картинки и структурный список `edge_words` (списки модель заполняет
  надёжнее, чем ставит инлайн-теги). Границы тегов приблизительные: на левом крае
  IMG_0008_R модель ставит `<restored>` ровно на одну первую букву строки («<restored>ч</restored>иновник»),
  даже если видна и она; на IMG_0008_L «обору<restored>дование</restored>» при видимом только «о».
  Слова верные, граница «видел/не видел» ±1-4 буквы.
* **`fuzzy` почти не ставит сама.** Размытые и сплющенные буквы она читает уверенно (и
  верно), поэтому «сомнительными» их не считает; пометки `fuzzy` появляются только через
  список `edge_words` (v7: 14 слов на IMG_0051_L — концы строк у сгиба, все прочитаны верно).
  Пересвет (IMG_0046) модель не считает повреждением вовсе: 0 пометок, текст верный.
* **`unknown` не понадобился ни разу**: на этих страницах всё восстановимо по контексту.
* **Подсказка — обоюдоострая.** Без неё пометок нет (раздел 8); с завышенной («часть букв
  может быть скрыта») модель выдумывает скрытые буквы там, где обычные переносы
  («адми-» → «административные» как hidden). Подсказка должна описывать ровно то, что
  есть: какой край, скрыты буквы или только искажены.
* Поле `damage` модель заполняет точно на всех 10 страницах (край, характер, число строк) —
  его можно использовать как самопроверку подсказки.

### Заключение

DeepSeek V4.1 Flash с повреждёнными сканами справляется **хорошо по сути и удовлетворительно
по форме**: текст у корешка, в размытии и в пересвете читает и достраивает без ошибок в
проверенной выборке; пометки после v7 стоят на всех страницах с реально срезанными буквами,
но их границы приблизительны, `fuzzy` без списка `edge_words` не появляется, а поведение
между прогонами одной страницы плавает (одна и та же страница в трёх версиях промпта не
помечалась, в четвёртой — 49 тегов). Для слияния с FineReader этого хватает как сигнала
«в этих словах есть достроенные буквы»; для точной разметки «какая именно буква не видна»
нужен второй источник — наш детектор корешка (где физически кончается бумага) или
`gutter_loss_restoration` со словарём выпуска.

Что реализовано: `run --damage --hints файл [--damage-side auto]`, промпт v7
(`prompts/system.md.j2`, блок `damage`; `prompts/user.md.j2`), поля `damage`, `restored`,
`fuzzy`, `unknown`, `edge_words` в схеме и `.json`, теги в `.md`, счётчики в `.meta.json` и в
сводке (`report` → таблица «Повреждения»), скрипты `run_split_damaged.sh`,
`run_damaged_deepseek.sh`, подсказки `damaged_hints.txt`. Выходы итераций —
`research/external_ocr_models/damaged/выход/deepseek-v{4..7}/` (вне git).
