# Состояние направлений

Обновлять при завершении направления.

## Сводная таблица

| Направление | Пакет | Статус | Лучший результат / выбранный метод | Отчёт |
|---|---|---|---|---|
| scan_cropping | `ocr_utils/scan_cropping` | рабочее | YOLO-World + SAM → силуэт разворота, `pixel-exact`, закрас пальца LaMa, заливка фона `nearest` | `ocr_utils/scan_cropping` (README корня) |
| scan_cropping: многопроцессность | — | заморожено | план: 1 GPU-процесс + 3–4 CPU-воркера, потолок ускорения 3.7×; модулей в коде нет | `reports/scan_cropping_multiprocess_report.md` |
| background_fill (экстраполяция бумаги в кроп) | `ocr_utils/scan_cropping/background_fill.py` | исследование | рекомендован B2 «приграничная полоса + диффузия наружу»; в бою пока средний цвет | `reports/background_fill_extrapolation_report.md` |
| shadow_removal (тень от пальца) | — | исследование | PoC: локализация зоны у пальца + коррекция с сохранением текста; в пайплайн не встроено | `reports/shadow_removal_report.md` |
| finger_removal: касание рамки | `ocr_utils/scan_cropping/finger_removal` | исследование | проверять касание рамки по СЫРОЙ маске до дилатации | `reports/finger_border_check_predilate_report.md` |
| defocus_detection | `ocr_utils/defocus_detection` | рабочее | `dom` (Kumar–Chen–Doermann) с агрегацией `best`, пресет `dom-si` | `reports/defocus_detection_validation_report.md`, `reports/defocus_validation_si_report.md`, `reports/defocus_detection_state_of_the_art.md` |
| defocus: зональный | `ocr_utils/defocus_detection/zonal.py` | рабочее | перепад ширины края по тайлам; заметная доля ложных, решает человек | `reports/defocus_detection_validation_report.md` (разд. 9), `reports/defocus_moire_improvement_plan.md` |
| выбор лучшего дубля серии | `reports/focus_series_report.md` / скрипты корня | исследование | регистрация SIFT+RANSAC, спектральная метрика без ресемплинга | `reports/focus_series_report.md` |
| select_best_raws | `select_best_raws.py` | рабочее | AKAZE (`local`) для газет, MobileNet (`cnn`) для фотосессий; резкость — lapvar | `reports/select_best_raws_report.md` |
| scan_markup (растр, печати) | `ocr_utils/scan_markup` | рабочее | автодетект → CVAT → SQLite; автодетектора печатей нет, только ручная разметка | `reports/scan_markup_pack1_annotation_report.md` |
| table_detection | `ocr_utils/scan_markup/table_detection` | рабочее | детектор v4.2: линейки → ядра → проверка «а таблица ли это» | `reports/table_processing_report.md` |
| таблицы: выпрямление и боковой текст | `research/legacy/table_processing` | заморожено | `rules_separable` (раздельное поле смещений); живой код переехал в `scan_markup` | `reports/table_processing_report.md`, `reports/table_geometry.md` |
| toc (оглавления) | `ocr_utils/scan_markup/toc` | рабочее | surya `TableOfContents` + позиционное окно + tesseract-признаки; 43/43 на эталоне | `reports/toc_detection.md` |
| orientation | `ocr_utils/scan_markup/orientation` | рабочее | 4 быстрых детектора → арбитр; 122/122 на кандидатах, `--angles 0,90` | `reports/orientation_method.md`, `orientation_pack1.md` |
| curved_lines | `ocr_utils/scan_markup/curved_lines` | рабочее | `skew_map` + `line_fit` (+`surya_lines` вторым голосом) + `end_curl` | `reports/curved_lines_detection_report.md`, `curved_lines_pack1.md` |
| dewarp | `ocr_utils/dewarp` | исследование | годится только `textline`; нейросетевые движки на сканах вредны | `reports/dewarp_report.md` |
| line_art | `ocr_utils/line_art_detection` | рабочее | связное пятно + скопление линеек, порог покрытия 5 %; формулы — только surya | `reports/line_art_detection_report.md`, `reports/line_art_detection_state_of_the_art.md` |
| show_through | `ocr_utils/show_through_detection` | рабочее | доля призрака, пережившего предварительную бинаризацию, в долях уровня бумаги | `reports/show_through_detection_report.md` |
| gutter_loss detection | `ocr_utils/gutter_loss_detection` | рабочее | «осталась ли бумага между знаком и сгибом»; отдельно помечаются таблицы | `reports/gutter_loss_report.md` |
| gutter_loss restoration | `ocr_utils/gutter_loss_restoration` | исследование | surya + достройка по переносу/словарю + набор литерами той же гарнитуры | `reports/gutter_loss_report.md` (разд. 4) |
| background_smoothing | `ocr_utils/background_smoothing` | рабочее | маска контента + нормированная свёртка (masked blur); под маской побитово исходник | `ocr_utils/background_smoothing/README.md` |
| scan_cleanup (inpaint) | `ocr_utils/scan_cleanup` | рабочее | закрас печатей/надписей LaMa по разметке из CVAT, затем размытие фона | докстринг `ocr_utils/scan_cleanup/__init__.py` |
| faint ink (пересвет, бледные перемычки) | `find_faint_ink.py` | исследование | признаки `ink_depth`, `clip_frac`, `bridge_loss`, `break_ratio` | `faint_ink_pack1.md` |
| faint zone restore | `restore_faint_zone.py` | исследование | поле «во сколько придавлена краска» + усиление + Ричардсон–Люси; разовый прогон | `reports/faint_zone_restore_IMG_0056.md` |
| rotated_text | `ocr_utils/rotated_text/tables` | рабочее | tesseract читает боковые ячейки, текст набирается прямо, DPI страницы поднимается | `reports/rotated_text_tables.md`, `reports/table_processing_rotation.md`, `reports/table_processing_ocr.md` |
| external OCR (OpenRouter) | `research/external_ocr_models` | исследование | Gemini 3.1 Flash Lite основной; Qwen3.8 Flash и DeepSeek V4.1 Flash (2 куска) вторым/третьим голосом; промпт v13 | `reports/external_ocr_models.md`, `reports/external_ocr_models_issue_1966_03.md`, `reports/external_ocr_models_probe_1966_03.md` |
| external OCR: боевой прогон | `ocr_utils/external_ocr_services` | рабочее | два этапа на выпуск: полосы оглавления (теги CVAT + вето `force_is_not_toc`) → рубрики и статьи в промпт остальных полос; `#` строго из оглавления, авторы при своей статье (`structure.py` доводит кодом); тайлы по сетке 4500 px, теги повреждений всегда, второй проход по полосам с `damaged=true` (мини-набор: 111/75 → 216/234 `<restored>`, на чистых 0 из 40 срабатываний); fallback «оглавление вне базы» | `ocr_utils/external_ocr_services/README.md` |
| external OCR: повреждённые буквы | `research/external_ocr_models` (`--damage`) | исследование | теги `<restored>/<fuzzy>/<unknown/>` + список `edge_words`, промпт v7 | `reports/external_ocr_models.md` (разд. 8, 10) |
| geometry_regression (FineReader ухудшил геометрию) | `research/geometry_regression` | исследование | попарные метрики «с коррекцией − без»: поле смещений (тайлы без пары в line art), LSD-штрихи (мм отклонения конца, параллельность), строки (наклон, волна), кромки; 7 из 8 эталонных при 0 ложных на 8 | `research/geometry_regression/README.md` |
| pdf_utils / FineReader | `ocr_utils/pdf_utils` | рабочее | `intermediate_pdfs` (две промежуточные), `final_pdfs`, `collect_sharpened` | `finereader_compare_pack1.md` |
| MRC-остатки FineReader | — | исследование | буквы, провалившиеся в фоновый слой MRC; поиск по расхождению маски и фона | `reports/mrc_leftovers_report.md` |
| docx_md | `ocr_utils/docx_md` | рабочее | `docx_to_md` + `split_md_by_articles` (нарезка по статьям под Long Context) | `ocr_utils/docx_md/README.md` |
| zonal_deblur | `ocr_utils/zonal_deblur` | рабочее | относительная PSF по спектру эталонных окон + Винер по ячейкам | `ocr_utils/zonal_deblur/README.md` |
| scantailor-эксперименты | — | исследование | разбор чужого кода: page_split и equalize illumination (color) | `reports/scantailor_page_split_report.md`, `reports/scantailor_equalize_illumination_color.md` |
| gigapixel | `analyze_gigapixel.py` | заморожено | внешний Topaz, суффикс `-gigapixel-text-shapes-4x`; отчёт — инвентаризация 526 папок | `reports/gigapixel.md` |
| инвентаризация паков | `run_scripts/pdf_utils`, `run_scripts/file_names` | рабочее | DPI картинок в PDF, альбомные страницы, рискованные имена, даты RAF | `reports/planhoz_pack_1_dpi_report.md`, `reports/planhoz_pack_1_landscape_report.md`, `mts_risky_names.md`, `reports/raf_dates_report.md` |
| списки на перескан/доскан | — | рабочее | машинные списки по прогонам `defocus_detection` + просмотр глазами | `reports/rescan_si_89_01_03.md`, `reports/rescan_si_89_04_06.md`, `reports/rescan_si_89_10_12.md`, `reports/eg_1982_04_dockan.md`, `reports/eg_1984_01-03_dockan.md` |
| cbr_library | `run_scripts/cbr_library` | рабочее | скачивание и переименование выпусков | — |
| legacy (dewarp, печати, page_layout, ocrmypdf) | `ocr_utils/legacy` | legacy | не поддерживается, тестами не покрыто, импортируется | `ocr_utils/legacy/README.md` |

## Что пробовали и отвергли

**defocus_detection**
- `edge_width` как рабочая метрика — AP 0.330 против 0.732 у `dom`, уровень плывёт между сессиями на 12.9 % (`reports/defocus_validation_si_report.md`).
- DeQA-Doc (7B, GPU) — AUC 0.68–0.77 против 1.00 у `dom` на тех же пикселях (`reports/defocus_validation_si_report.md`).
- `pyiqa`, DeFusionNET/BTBNet/MSDU-Net, SMBlurDetect, коммерческий куллинг — обучены на натурных сценах и глубине резкости, нет понятия «читается ли петит» (`reports/defocus_detection_state_of_the_art.md`).
- Выравнивание уровней `dom_eq` — нулевой эффект: `dom` есть отношение сумм, линейное растяжение сокращается (`reports/defocus_validation_si_report.md`).
- Первая версия `edge_dir` (поворот кадра на 4 угла) — сам поворот размывает, резкий кадр получал анизотропию 1.40 (`reports/defocus_validation_si_report.md`).
- Детектор смаза на реальных данных — на синтетике безупречен, на съёмке не подтвердился (`reports/defocus_validation_si_report.md`).
- Счёт балла только по тайлам с ритмом строк — средняя точность падает, верное срабатывание теряется (`reports/defocus_validation_si_report.md`).
- Нормировка порога на шаг строк — не понадобилась: уровень балла перенёсся на другую газету и гарнитуру (`reports/defocus_validation_si_report.md`).
- Сырой `moire` = std(NEAREST−AREA) — на 50 % объясняется количеством краски, а не фокусом (`reports/defocus_moire_improvement_plan.md`).

**dewarp**
- DocScanner, UVDoc, DocTr, DocTr-Plus, DewarpNet — обучены на Doc3D (фото мятых листов), на флэтбед-скане сдвигают, режут и «курсивят» страницу (`reports/dewarp_report.md`).
- `page-dewarp` (cubic sheet) — модель камеры и цельного листа, поворачивает всю полосу (`reports/dewarp_report.md`).
- ScanTailor заменил собой весь старый подпакет dewarp (`ocr_utils/legacy/README.md`).

**curved_lines**
- Край колонки (Leptonica edge curvature, ScanTailor) — абзацные отступы и висячие строки шумят сильнее прогиба (`reports/curved_lines_detection_report.md`).
- `strip_shift` — на ровных полосах гребёнка цепляется за соседнюю строку, лишён права одиночного голоса (`reports/curved_lines_detection_report.md`).
- Базовая линия по нижней огибающей — AUC 0.50 (`reports/curved_lines_detection_report.md`).
- Ridge-центр-линии (Bukhari) — хребет прыгает между строками, AUC 0.56 (`reports/curved_lines_detection_report.md`).
- Базовые точки слов (Ulges) — слишком мало точек на строку (`reports/curved_lines_detection_report.md`).
- Kraken blla — векторизация упрощает линию до двух точек, формы кончика нет; 10–25 с на полосу (`reports/curved_lines_detection_report.md`).

**geometry_regression**
- Профильный deskew (сдвиг столбцов, `warpAffine` + профиль) на БИНАРНОМ рендере — все углы ровно 0.00°: при нулевом угле нет интерполяции и округления сдвигов, профиль всегда резче; годятся только центр-линии строк, `fitLine`/LSD и блочное сопоставление (`research/geometry_regression/README.md`).
- Линейки морфологическим открытием длинным ядром — наклонённая на 6° дробная черта (1967/01 с.85) рвётся в пыль; заменено LSD на 300 dpi со склейкой коллинеарных кусков.
- Наклон линейки в градусах как флаг — хорошая с.2 1967/02 давала 2° на 9-мм вертикалях (0.2 мм, глазу незаметно); мерится отклонение конца в мм.
- Остаток поля смещений внутри line art как флаг — велик и внутри таблиц на хороших страницах (там FineReader правит законно); решает доля тайлов со слабым пиком корреляции (61 % на погнутой блок-схеме с.80 против ≤ 6 % везде).
- Клин высоты строки (верх/низ букв по третям) для наклонённых заголовков — 0.056 на 1967/03 с.36 при шуме 0.05–0.25 на обычных строках.
- Сопоставление строк по перекрытию от короткой — строке, распавшейся в A на куски, доставался короткий кусок с шумным наклоном (1967/01 с.4); перекрытие ≥ 0.75 от длинной.
- Наклон строки и кромки в градусах как флаг — короткие куски строк дают 1.5° шума (1973/01 с.43); мерится уход конца в мм (длина × sin).
- DeepSeek V4.1 Flash как детектор порчи геометрии (`vlm-probe`, 4 варианта картинки, 2 редакции промпта, 1104 запроса за $0.26) — 12 % на эталонных плохих, 8–13 % в поясе score ≥ 2, 0 ложных: наклоны в 0.5–1.5° ниже её разрешения (`research/geometry_regression/README.md`).
- Кромки по номеру колонки — сегментация колонок в A и B расходится, кромке всей колонки доставалась кромка нижней трети (1967/04 с.9); сопоставление по стороне и перекрытию по вертикали, кромки считаются по блокам строк.

**таблицы**
- Тонкая пластина по пересечениям линеек (`rules_tps`) — сработала на 2 из 15: у большинства таблиц пересечений меньше шести (`reports/table_processing_report.md`).
- Deskew и гомография — беда не в перекосе и не в трапеции: разброс углов 0.19°, сужение меньше полупроцента (`reports/table_processing_report.md`).
- Движок по строкам текста на таблицах — правит строки, а не линейки, сагитту не меняет (`reports/table_processing_report.md`).
- Гипотеза «FineReader пропускает таблицы из-за кривой геометрии» — из 99 пропущенных кривых 15, «коррекция сделала хуже» — ноль случаев (`reports/table_processing_report.md`).
- Доля букв в краске как единственный признак таблицы — выбрасывала разрежённые бланки (`reports/table_processing_report.md`).
- Заполненность ячеек и медианная длина линейки как признаки — не разделяют схему и график (`reports/table_processing_report.md`).
- surya как распознаватель боковых ячеек — CER 2.359 против 0.034 у tesseract, выдумывает зацикленный текст (`reports/table_processing_ocr.md`, `reports/rotated_text_tables.md`).
- paddle там же — CER 0.074 и 2.15 с на ячейку (`reports/table_processing_ocr.md`).
- Детектор поворота `profile` на ячейках — 0/28 боковых (`reports/table_processing_rotation.md`).

**toc**
- Геометрия отточий — таблицы норм отгрузки давали бы ложные (`reports/toc_detection.md`).
- DeepSeek «да/нет» по полосам или контактным листам — сеть и недетерминизм, на листе из 18 миниатюр шрифт нечитаем (`reports/toc_detection.md`).
- `is_toc` из полного прогона DeepSeek — узнаём слишком поздно, список нужен ДО прогона (`reports/toc_detection.md`).
- «СОДЕРЖАНИЕ» без проверки регистра и длины строки — ложное оглавление на каждой третьей полосе окна (`reports/toc_detection.md`).
- «ПЕРЕЧЕН» с допуском в одну букву — совпадал с «переменного» в таблице (`reports/toc_detection.md`).
- Пара «полосы 3 и 4» жёстко по номерам — ломается на 1970/10 (`reports/toc_detection.md`).

**orientation**
- Детектор `profile` — 79 из 80 ложных кандидатов и ни одной уникальной находки, из набора по умолчанию убран (`reports/orientation_method.md`).

**line_art**
- Отсечение по размеру дыр и доле длинных прогонов (отделение полутоновых фото) — на этом материале не разделяет и стоило эталонной стр. 80 (`reports/line_art_detection_report.md`).
- Отсев декоративных рамок рубрик — без потери настоящих врезок в рамке не вышло (`reports/line_art_detection_report.md`).
- Поиск формул пиксельным порогом — формула занимает 0.4–1.3 % полосы, достаётся только через surya `Equation` (`reports/line_art_detection_report.md`).

**show_through**
- Детекция просвета «как такового» — выпуск 1955/03 просвечивает сплошь, но бинаризацию призрак не переживает; метрика переписана под критерий «переживёт ли обработку» (`reports/show_through_detection_report.md`).
- Абсолютный уровень серого как признак — пак охватывает 40 лет, бумага и освещение разные (`ocr_utils/show_through_detection/README.md`).
- Отказ от нормировки на уровень бумаги полосы — AUC падает с 0.98 до 0.76 (`reports/show_through_detection_report.md`).

**external OCR**
- Claude Haiku 4.5 — в 4–9 раз дороже, теряет шапки таблиц, дописывает рубрику из примера промпта (`reports/external_ocr_models.md`).
- Mistral Small 4 — выдумывает шапки таблиц (`reports/external_ocr_models.md`).
- Qwen3-VL-235B — уходит в цикл по отточиям, 3 ¢ и 200 с на полосу (`reports/external_ocr_models.md`).
- GPT-5.6 Luna и Gemini 3.8 Flash — не лучше дешёвых при цене в 3–6 раз выше (`reports/external_ocr_models.md`).
- DeepSeek V4.1 Flash целой полосой — потолок 1024 токена на картинку, теряет текст; лечится подачей двумя кусками (`reports/external_ocr_models.md`).
- DeepSeek-OCR-2 локально — теряет тело таблиц с боковыми шапками, нестабилен, не инструктивен (`reports/external_ocr_models.md`).
- Marker 1.10 локально — строки таблиц перепутывает, отточия превращает в «20-20-20» (`reports/external_ocr_models.md`).
- Специализированные OCR-API (Mistral OCR, Yandex Vision, Google, Azure) — не дешевле VLM и не дают семантики статьи (`reports/external_ocr_models.md`).
- Формулы без правила в промпте — три разных нотации на шести полосах 1976/04 (HTML `<sub>`, линейная запись, `$$`), линейная дробь без скобок двусмысленна; введено правило «LaTeX в долларах внутри `<latex>`», после него 52 тега на шести полосах, все парные, старых нотаций нет.
- Поле полосы `continues_previous` — модель на странице-продолжении (МТС 1991/02, с. 95) дописывала `## <название статьи из списка>`, которого на странице нет: 5 из 5 прогонов. Удаление поля само по себе не помогло (4 из 5), помог прямой запрет в промпте «страница-продолжение не получает название из списка»: 0 из 5 и 0 из 20 на 1966/03. Поле убрано (потребителей не было), запрет в промпте v7, страховка `structure.drop_leaked_titles` осталась.
- Рубрика по месту печати — в МТС 1991/02 маркер рубрики стоит в нижнем углу страницы и в колонтитуле, тег `<rubric>` там разрезал бы статью при нарезке по рубрикам; теперь рубрика статьи берётся из оглавления, код подтягивает тег к `#`, прочие надписи — `<marker>` (`structure.place_rubrics`).
- Оговорка «`#` и для отдельной заметки с автором» в промпте — модель ставила `#` подзаголовку вверху страницы-продолжения (1966/03 `IMG_0106_1L`); теперь `#` строго из списка, остальное понижает код (`structure.py`).
- Поле `damage` без ограничения длины — DeepSeek ушёл в цикл, перечисляя номера строк (16 000 токенов на странице 1946 «Планового хозяйства»); ограничено одним предложением.
- Нейтральная фраза «помечай только то, что видишь» без подсказки о повреждении — модель перестаёт достраивать скрытые буквы: на IMG_0006_L 55 `<unknown/>` вместо 47 `<restored>`; заменена на «ищи повреждения сам, скрытые буквы достраивай по контексту» — 52 `<restored>`, 0 `<unknown/>` в двух прогонах. Без подсказки часть повреждённых страниц модель всё равно не видит (IMG_0006_R: 0 против 26 с подсказкой), подсказки по полосам `--hints` остаются полезными.
- Триггер второго прохода по тексту `damage` — с фразой «ищи повреждения сам» модель пишет «повреждений не обнаружено» на 33 из 40 чистых полос (82 % ложных срабатываний); заменён булевым `damaged` в схеме — 0 из 40.
- Полный текст первого прохода во втором запросе (`--second-pass-transcript`) — 213/243 `<restored>` против 216/234 без текста, выигрыша нет, оставлен опцией.
- Опция стороны корешка (`--damage-side`, в т. ч. по суффиксу `_L`/`_R` имени файла) — убрана: суффикс бывает ложным, а сторона плавает от части к части выпуска; сторону модель определяет сама по картинке.
- DeepSeek `json_object` с длинным промптом (список статей выпуска) — в 20–40 % ответов эхо `{"type": "json_object"}` вместо страницы; лечится повтором без `response_format` (`ocr_utils/external_ocr_services/ocr.py`, `is_format_echo`).
- Промпт: правило «склеивать перенос внутри шапки без дефиса» — не сработало даже с явным отрицательным примером (`reports/external_ocr_models.md`).
- Завышенная подсказка «часть букв может быть скрыта» — модель выдумывает скрытые буквы на обычных переносах (`reports/external_ocr_models.md`).
- Штатный SDK `openrouter` «как есть» — его модель запроса не знает `reasoning.enabled`/`max_tokens` (молча выбрасывает: thinking у DeepSeek включился бы обратно), в ответе нет `provider`, ретраи только на 5xx; поэтому клиент переводит `enabled:false` → `effort:none`, просит `openrouter_metadata` и держит свой цикл ретраев (`ocr_utils/external_ocr_services/client.py`).
- Пропуск как повтор символа `<gap>▒▒▒</gap>` — DeepSeek при `temperature 0` зациклился на «▒» до потолка 16k токенов (повреждённая полоса IMG_0008_L, 15 тыс. символов); потолок «не больше шести» в промпте помогал, но правильное решение — число: модель пишет `<gap>4</gap>`, заполнители расставляет разбор (`schema.expand_gaps`), длинные пропуски — `[N symbols]`; ряд от 20 «▒» в сыром ответе по-прежнему считается обрезанным и уходит на повтор.

**scan_cropping**
- «По экземпляру `GpuModels` на воркер» — один экземпляр занимает 6.2 ГБ VRAM, в 16 ГБ влезает два (`reports/scan_cropping_multiprocess_report.md`).
- Наивный RPC через `mp.Queue` — 368 МБ на кадр через границу, потолок падает с 3.7× до 2.7× (`reports/scan_cropping_multiprocess_report.md`).
- Один средний цвет бумаги для заливки за краем страницы — в среднее попадают чернила, свет по краям разный (`reports/background_fill_extrapolation_report.md`).
- Наивный глобальный детектор тени — путает пальцевую тень со складкой у корешка и виньетированием (`reports/shadow_removal_report.md`).
- Подбор дилатации маски пальца против склейки с ложной детекцией — арифметически не лечится (`reports/finger_border_check_predilate_report.md`).

**background_smoothing**
- `blur(I_source)` без маски — в окно попадают чернила соседних букв, фон темнеет до 168–190 (`ocr_utils/background_smoothing/README.md`).

**прочее**
- Наивные lapvar/Tenengrad по кропу для выбора дубля серии — растут к мелким кадрам, побеждает самый мелкий (`reports/focus_series_report.md`).
- Восстановление текста под корешком вместо пересъёмки — придумывает пиксели, для архива хуже настоящих (`reports/gutter_loss_report.md`).

## Действующие пороги и параметры

| Параметр | Значение | Где задано | Откуда взято |
|---|---|---|---|
| `dom` теги тяжести | heavy 2.92 / medium 2.96 / light 3.00 / ultralight 3.04 | `ocr_utils/defocus_detection/thresholds.py` (пресет `dom-si`) | `reports/defocus_validation_si_report.md` |
| `--worst-percent` / `--zonal-percent` | 5–15 (в прогонах 10) | `run_scripts/defocus_detection/*` | `ocr_utils/defocus_detection/README.md` |
| show_through: порог отбора | 0.010 (операционный, привязан к `--threshold-bias`) | `run_scripts/show_through_detection/*` | `reports/show_through_detection_report.md` |
| gutter_loss: порог | 0.35 (калиброван по одной папке; для пака возможно 0.8–0.9) | `run_scripts/gutter_loss_detection/*` | `reports/gutter_loss_report.md` |
| line_art: рабочее покрытие | ≥ 5 % полосы (610 страниц из 8656) | `run_scripts/line_art_detection/*` | `reports/line_art_detection_report.md` |
| line_art: формулы | `--min-coverage 0.003 --source surya:Equation` | `run_scripts/line_art_detection/*` | `reports/line_art_detection_report.md` |
| line_art: `--min-age-minutes` | 10 | `ocr_utils/line_art_detection` | `reports/line_art_detection_report.md` |
| curved_lines: сводный флаг | голосов ≥ 2 ИЛИ max score ≥ 2.5; щедрый режим — пороги −40 % | `run_scripts/scan_markup/pack1/run_curved_lines.sh` | `curved_lines_pack1.md`, `reports/curved_lines_detection_report.md` |
| curved_lines: `skew_map` | spread 0.9° / max_dev 1.2° / resid 0.36° | там же | `curved_lines_pack1.md` |
| curved_lines: `line_fit` | sagitta_rel_p90 0.11 / max3 0.17 / slope_spread 0.54° / resid 0.18° | там же | `curved_lines_pack1.md` |
| curved_lines: `surya_lines` | sagitta_rel_p90 0.09 / max3 0.13 / slope_spread 0.54° / resid 0.21° | там же | `curved_lines_pack1.md` |
| curved_lines: `end_curl` | end_slope 0.8° / edge_angle_diff 0.8° / tensor_edge_diff 1.0° | там же | `reports/curved_lines_detection_report.md` |
| geometry_regression: пороги (калибровка по паку-1, p98–p99) | vstroke 0.7 мм, hstroke 0.8 мм, parallel_spread 1.5°, field_lineart_weak_frac 0.3, line_dev 0.9 мм, line_wobble 0.1, edge_dev 2.4 мм + line_stretch_mm 0.2 мм → 6 % страниц (без stretch) | `research/geometry_regression/scoring.py` | `research/geometry_regression/README.md` |
| orientation: допустимые углы | `--angles 0,90` | `run_scripts/scan_markup/pack1/run_orientation.sh` | `reports/orientation_method.md` |
| background_smoothing: `--threshold-bias` | 0.5 (в прогонах до 0.7) | `ocr_utils/background_smoothing`, `run_scripts/background_smoothing/*` | `ocr_utils/background_smoothing/README.md` |
| background_smoothing: `--blur-mult` / `--blur-mode` | 4.0 / `masked` | там же | `ocr_utils/background_smoothing/README.md` |
| rotated_text: подмена ячейки | уверенность ≥ 0.6 (массово ≥ 0.8), слово от 3 букв; «180» — от 5 букв и 0.8 | `ocr_utils/rotated_text/tables` | `reports/rotated_text_tables.md` |
| rotated_text: предельный DPI страницы | 1350 | `ocr_utils/rotated_text/tables` | `задачи на перспективу/rotated_text_fragments.txt` |
| table_detection: константы детектора | FRAGMENT_MM 3, CHAIN_GAP_MM 8, MAX_ISOLATION 0.55, MAX_INK_SHARE 0.25, GROW_CAP_MM 40 и др. | `ocr_utils/scan_markup/table_detection` | `ocr_utils/scan_markup/table_detection/README.md` |
| external OCR: модель и настройки | Gemini 3.1 Flash Lite, JSON-схема, `reasoning.max_tokens=1024`, картинка 2200 px | `research/external_ocr_models`, `run_scripts/external_ocr_models/*` | `reports/external_ocr_models.md` |
| external OCR: версия промпта | `PROMPT_VERSION = 13` | `research/external_ocr_models/__init__.py` | `reports/external_ocr_models.md` (разд. 9.6) |
| external_ocr_services: промпт | `PROMPT_VERSION = 11` (теги TEI `<supplied>/<unclear>/<gap>N</gap>` — число утраченных букв, заполнитель ставит разбор, `<toc>` + `<rubric_in_toc>`, «предварительно оглавление» с понижением, `<schema>/<photo>/<line_art>/<footnote>`, формулы — LaTeX в `<latex>$…$</latex>`); регрессия v9 → v10 на 33 полосах: оглавления те же, CER ≤ 0.015, сбоев 0; v2 → v6 на 1966/03: CER к FineReader 0.036 → 0.028 (среднее), эхо формата 9 → 0 | `ocr_utils/external_ocr_services/__init__.py` | `ocr_utils/external_ocr_services/README.md` |
| external_ocr_services: тайлы | `--max-src-tile-size 4500` (обычная полоса 1×2, разворот 2×2), `--max-model-tile-size 2200`, перекрытие 8 % | `ocr_utils/external_ocr_services/tiling.py`, `run_scripts/external_ocr_services/*` | замер размеров по базе пака-1 (2998–4499 × 4961–6692, разворот 6444×5336) |
| scan_cropping: пальцы и кроп | `--finger-dilate-px` 60–120, `--max-asymmetric-dilation-ratio` 1.6–2.0, `--extra-erosion-px` 80–110, `--layout-pad-px` 12–18, `--bg-fill-blur-px=16` | `run_scripts/scan_cropping/*` | `reports/finger_border_check_predilate_report.md` |
| select_best_raws | `--min-match-ratio 0.2`, `--n-search 5`, `--max-scale-change 1.15` | `run_scripts/select_best_raws/*` | `reports/select_best_raws_report.md` |
| параллелизм по умолчанию | `--jobs 16` (16 физических ядер), на медленном NTFS меньше | `run_scripts/*`, `CLAUDE.md` | `CLAUDE.md` |

## Открытые задачи

- После очистки пака-1 — отдельный фильтр усиления пересвеченных бледных перемычек букв (решение
  пользователя от 2026-09-05). Поэтому размытие фона в `scan_cleanup` сильнее прежнего, а
  защитная маска щедрее (Sauvola, припуск 25 px вместо 15): при выборе параметров не оптимизировать
  чистоту фона в ущерб бледным штрихам.
- Боевой внешний OCR (`ocr_utils/external_ocr_services`): прогнать пак целиком (`run_pack_sharpened.sh`), разобрать `missed_toc.txt` через теги CVAT, проверить долю ложных `<fuzzy>` от нейтральной фразы про повреждения на чистых полосах.
- Перекрёстная проверка декабрьского годового указателя по спискам всех выпусков года (`reports/toc_detection.md`).
- Слияние версии FineReader и версии VLM в один markdown: голосование по заголовкам/авторам, таблицы из VLM, кривые строки из FineReader (`reports/external_ocr_models.md`).
- Прогнать пробник внешних OCR на 1974–76 (пятнистый фон, петит) и проверить рост расхождения с FineReader (`reports/external_ocr_models.md`).
- Собрать второй комплект промежуточных PDF с исправленными таблицами и проверить его на FineReader через Hot Folder (`reports/table_processing_report.md`, `reports/rotated_text_tables.md`).
- Разрежённые бланки — главная причина пропуска таблиц FineReader: дорисовка линеек, дробление, невидимый якорь в пустых ячейках (`reports/table_processing_report.md`).
- Таблицы, напечатанные боком целиком (30 находок) — отдельная задача (`reports/table_processing_report.md`).
- Смешанные ячейки (298): делить ячейку по смене оси текста; включить в прогон второе мнение surya (`reports/rotated_text_tables.md`).
- Вписывание увеличенной таблицы в страницу по кэшу surya layout (`reports/rotated_text_tables.md`).
- Блок-схемы, прочий повёрнутый текст, стадия `detect` с видом `SIMPLE_ROTATED_TEXT` и его импорт-экспорт в CVAT; разделение выходов на `intermediate_pdfs_as_is` и `..._for_detection` (`задачи на перспективу/rotated_text_fragments.txt`).
- Улучшить поиск line art / блок-схем на стадии `detect` с использованием surya layout (`задачи на перспективу/rotated_text_fragments.txt`).
- Разметить 30–50 полос из пояса score 1.0–1.6 и уточнить пороги кривых строк; отсечь оглавления признаком отточий (`reports/curved_lines_detection_report.md`).
- Прикрутить флаг кривых строк к пайплайну: колонка в базе или список страниц для FineReader (`reports/curved_lines_detection_report.md`).
- Горизонтальная поправка `textline_h` по краям колонок; проверка dewarp на распознавании, а не на картинках (`reports/dewarp_report.md`).
- Сверка с оборотом вторым этапом детектора просвета; нижняя граница на площадь межстрочий против фотопортретов (`reports/show_through_detection_report.md`).
- Перепроверить порог gutter_loss на других годах; табличный признак не видит таблицы без линеек (`reports/gutter_loss_report.md`).
- Разметить случайную выборку из 12 013 отсеянных полос, чтобы узнать настоящую полноту детектора ориентации (`reports/orientation_method.md`).
- Смаз оставлен открытым вопросом до целенаправленной разметки; `pyiqa` замерить и закрыть вопрос числом (`reports/defocus_validation_si_report.md`, `reports/defocus_detection_state_of_the_art.md`).
- Реализовать многопроцессность `scan_cropping`, начиная с сужения интерфейса `GpuModels` (368 → 87 МБ на кадр) (`reports/scan_cropping_multiprocess_report.md`).
- Внедрить заливку фона вариантом B2 и провалидировать (`reports/background_fill_extrapolation_report.md`).
- Встроить детекцию и коррекцию теневой зоны у пальца в пайплайн (`reports/shadow_removal_report.md`).
- Проверять касание рамки по сырой маске пальца до дилатации (`reports/finger_border_check_predilate_report.md`).
- Файлы для ИИ-агентов в проекте (этот справочник — часть задачи) (`задачи на перспективу/облегчение разбора проекта для ИИ.md`).
