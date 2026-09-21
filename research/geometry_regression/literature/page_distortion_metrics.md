<!-- Обзор: метрики дисторсии страницы и парного сравнения версий (2026-09-22, агент-обзор). Источник: фоновый агент WebSearch/WebFetch; ссылки — в тексте. Сводка и решения — reports/geometry_regression_v15.md -->

## Обзор: численная оценка геометрической дисторсии страницы и сравнение пары «до/после» без эталона

Главный вывод сразу: **готового инструмента, который берёт две версии одной страницы и отвечает «вторая геометрически хуже», в литературе нет.** Все парные метрики (LD, AD, Li-D, AAD, DD, энергия TPS) измеряют *величину* нежёсткой разницы между картинками, но **не знак** — выпрямление волнистой строки и погнутие прямой линейки дают одинаковый остаток. Все безэталонные метрики (DW, кривизна, скос, ортогональность штрихов) измеряют одну версию. Рабочая схема — комбинация: (а) разложение поля смещений A↔B по элементам на аффинную часть (поворот, сдвиг/шир, масштаб) + остаток, (б) безэталонная прямизна каждого элемента в обеих версиях, знак — по разности (б), «что именно сделал FineReader» — по (а). Это по сути то, что уже делает v14 проекта; ниже — что из литературы и библиотек может усилить именно эти два звена.

---

### 1. Безэталонные (no-reference) метрики и оценка скоса/кривизны

**1.1. DW — dewarping evaluation measure (Stamatopoulos et al., IET IP 2012)**
- Суть: по каждой строке j берутся ключевые точки на искажённой и на выпрямленной версии (сопоставление point-to-point, в оригинале — SIFT-ключи в началах слов), через них проводится кубический полином p_j(x); S_j = Σ∫|p_j(x)|dx (отклонение от горизонтальной прямой) до, R_j — после; D_j = 1 − R_j/S_j при R_j < S_j, иначе 0; DW = средняя по N строкам ×100 %; есть взвешенный вариант wDW с весами S_j/ΣS (формулы 23–25 в [Dasgupta et al. 2020](https://arxiv.org/pdf/2003.06872)).
- Параметры: степень полинома (3), выбор ключевых точек, число «критических» строк (у авторов 6 на страницу).
- Реализация: публичного кода нет (внутренний Matlab); формула — 20 строк numpy поверх уже имеющихся baseline surya.
- К нашей паре: **плюс** — это единственная в литературе метрика, которая по построению сравнивает *относительное* улучшение строки «до/после» и обрезается в 0, когда стало хуже; **минус** — только строки текста; в оригинале «эталон» — горизонталь, у нас FineReader ещё и глобально довернул страницу, поэтому отклонение надо мерить не от горизонтали, а от собственной наилучшей прямой строки (иначе доворот запишется в улучшение); отдельно порог «стало хуже» надо калибровать (D=0 «слишком строго» на волнистых строках, где FineReader ухудшил на 0,1 мм).

**1.2. Кривизна строки / сагитта baseline**
- Суть: точки baseline (surya, tesseract, низы CC) → устойчивая подгонка прямой/параболы; сагитта = max отклонения от хорды; кривизна ≈ 8·sagitta/L²; результат в мм при известном dpi.
- Параметры: степень (2 или 3), минимальная длина строки, робастность (Huber / RANSAC), допуск на выбросы (дескендеры).
- Реализация: numpy.polyfit, `skimage.measure.ransac`, sklearn HuberRegressor.
- К нашей паре: **плюс** — уже есть в проекте (v14), знак разности A−B отвечает на вопрос; ничего нового в литературе сверх этого не нашлось.

**1.3. Безэталонный оценщик качества дьюворпа — Dasgupta, Das, Nasipuri 2020 ([arXiv 2003.06872](https://arxiv.org/abs/2003.06872))**
- Суть: страница режется на блоки сетки; в блоке i считаются нормаль N_i, «главный поток» G_i (направление строк) и «побочный» H_i (направление штрихов). Пять метрик: μ1 ортогональность проекции ‖N·P‖, μ2 параллельность строк ‖G_i − Ḡ‖, μ3 геодезичность ‖ΔG·P‖², **μ4 ортогональность штрихов направлению строк ‖G_iᵀ·H_i‖**, μ5 относительная высота букв |1 − ‖N‖²|. Итог μ̄ = ⅕(μ1 + (1−μ2) + μ3 + μ4 + (1−μ5)); решения: ≥0,95 — хорошо, 0,90–0,95 — править строки, <0,90 — повторить (табл. 2 статьи, коррелирует с точностью Tesseract на CBDAR 2007).
- Реализация: кода нет (Python 3.7 + NumPy/SciPy/skimage у авторов, не выложен).
- К нашей паре: **плюс** — μ4 это ровно «блок стал параллелограммом»: угол между доминирующим направлением вертикальных штрихов и направлением baseline в блоке; считается без всякого сопоставления версий — гистограмма ориентаций градиента (Sobel) или осей CC (`cv2.fitEllipse`/моменты) в маске блока, сравнить A и B. Для гарнитур с наклоном (курсив) нужна калибровка «нормального» угла по паку. **Минус** — на блоках из 2–3 строк ориентационная статистика шумная; в целом μ1/μ3/μ5 завязаны на их собственную модель гомографии и нам не нужны.

**1.4. Оценка скоса (skew)**
- Проекционный профиль (Postl 1986; ocropy/ocrd_cis: 32 углов в ±2°, максимум дисперсии сумм по строкам; Bloomberg/Leptonica `pixFindSkew` — то же с coarse-to-fine и оценкой уверенности). Параметры: диапазон, шаг, downsampling. Реализация: [ocrd_cis deskew.py](https://github.com/cisocrgroup/ocrd_cis/blob/master/ocrd_cis/ocropy/deskew.py) (MIT, последний push 2024-08) — 30 строк, тянуть OCR-D-стек ради них не стоит. **Плюс** — работает на отдельном блоке/заголовке; **минус** — при 1–2 строках профиль плоский, точность ~0,1°.
- Hough по строкам/CC: pip `deskew` (sbrunner, MIT, push 2026-09, 526★; `skimage.transform.hough_line` на границах, угол ±45°, есть `determine_skew_debug_images`). Плюс: простой, но только целая картинка; на блоке — самим `hough_line_peaks` по центроидам CC.
- Fourier: pip `jdeskew` (Pham Quy Luan, ICIP 2022, MIT, push 2026-08, 169★; адаптивная радиальная проекция спектра, `get_angle(image)`), точен на страницах, на мелких кропах — не проверено, требует достаточно строк.
- Ближайший сосед (docstrum, O'Gorman 1993): углы к k ближайшим CC → пик гистограммы; работает на заголовке из 5 букв. Реализация — своя (scipy cKDTree, 40 строк).
- К нашей паре: скос глобально известен; полезно только как **локальный** угол элемента (заголовок, кромка) — и там проще линейная подгонка по baseline/кромке с RANSAC, чем любая из библиотек.

**1.5. «Warping degree» через глобальную модель листа: `page-dewarp`** ([lmmx/page-dewarp](https://github.com/lmmx/page-dewarp), MIT, push 2026-09-21, 245★, Python ≥3.10, NumPy/SciPy/SymPy/OpenCV/msgspec, опционально JAX/CUDA; renovated Zucker 2016)
- Суть: детектирует «спаны» (строки как цепочки CC), подгоняет кубическую модель листа (параметры α, β кривизны + поворот + сдвиг + фокус); величина α, β — степень изгиба.
- К нашей паре: **минус** — модель одного гладкого листа, наши дефекты FineReader локальные, по-элементные; интерфейс CLI-first, параметры модели наружу не документированы. Отвергнуть как метрику.

**1.6. Что есть в дьюворп-репозиториях по оценке**
- OCR-D `ocrd_anybaseocr` (Apache-2.0, push 2025-05): dewarp = pix2pixHD GAN, deskew — правило-based; **никакой оценки качества**; [issue #103 «dewarp: quality?»](https://github.com/OCR-D/ocrd_anybaseocr/issues/103) открыт.
- DocTr / DocGeoNet / DocScanner / DewarpNet / UVDoc: оценка MS-SSIM + LD только через **MATLAB-код бенчмарка DocUNet** (`ssim_ld_eval.m`, SIFT-flow mex Ce Liu); UVDoc `docUnet_eval.py` требует `matlab.engine`. Python-реализаций LD/AD в них нет. Лицензии DocTr/DocScanner/DocGeoNet — нестандартные (NOASSERTION, только research), DewarpNet и UVDoc — MIT.
- `docdewarp` как пакета нет; есть pip `docuwarp` — инференс UVDoc, без метрик.

---

### 2. Метрики сравнения двух изображений как дисторсии

**2.1. LD — Local Distortion (Ma et al. DocUNet 2018)**
- Суть: плотный SIFT-flow эталон→результат, LD = среднее L2 смещений. Параметры: SIFT-flow (размер ячейки dense-SIFT, α/d/γ регуляризации BP, пирамида), в бенчмарке картинки масштабируют к 598 000 px.
- Реализация: оригинал MATLAB/C++; Python-порты: [hmorimitsu/sift-flow-gpu](https://github.com/hmorimitsu/sift-flow-gpu) (MIT, 2020 — **только дескрипторы**, без BP-оптимизации потока), [chienerh/SIFT-Flow](https://github.com/chienerh/SIFT-Flow), [caomw/sift-flow](https://github.com/caomw/sift-flow) (обёртка C++).
- К нашей паре: **минус** — доминируется глобальным сдвигом/масштабом (что и породило AD), SIFT-flow медленный (BP), на бинарных текстах дескрипторы вырождены. Заменяется DIS-flow (2.5).

**2.2. AD — Aligned Distortion (Jiang et al., CVPR 2022, [arXiv 2203.16850](https://arxiv.org/abs/2203.16850))**
- Суть: тот же поток, но из него вычитается оптимальная глобальная аффинная/подобия (сдвиг + масштаб) часть, а остаток усредняется с весом |∇I| эталона (ошибки на пустой бумаге не считаются). Код метрики в [DocGeoNet](https://github.com/fh2019ustc/DocGeoNet) — MATLAB.
- **Li-D** (DocScanner, IJCV 2025): std Δx по столбцам и std Δy по строкам поля потока — мера «строки/столбцы перестали быть прямыми». **AAD** (Wang et al., AAAI 2026, [arXiv 2507.15000](https://arxiv.org/html/2507.15000)): то же, но с Sobel-весами построчно/постолбцово; репозиторий [chaoyunwang/AADD](https://github.com/chaoyunwang/AADD) — Apache-2.0, **на момент проверки только README, кода нет**. **DD** (Zhang et al., PRL 2025): поток берётся из DocAligner вместо SIFT-flow; [DocAligner](https://github.com/ZZZHANG-jx/DocAligner) — torch 1.11 + CuPy-корреляционный слой, только CUDA, лицензия не указана, обучен на «фото→скан».
- К нашей паре: идея AD (снять аффинную часть, взвесить краской) — правильная и её стоит взять; сами реализации — нет. **Ключевой минус всех** — без знака.

**2.3. Аффинный/прокрустов остаток по сопоставленным точкам**
- Суть: точки-соответствия A↔B (центроиды CC через cKDTree после глобальной привязки; концы baseline surya; ORB/AKAZE) → `cv2.estimateAffine2D` (RANSAC, маска инлаеров) или `skimage.transform.estimate_transform('affine')`; `AffineTransform.rotation / .shear / .scale` дают **поворот, шир, масштаб** элемента напрямую; RMS остатка после аффина — «локальный варп». `scipy.spatial.procrustes` — только подобие, возвращает disparity.
- Параметры: порог RANSAC (1–2 px при 300 dpi), минимум точек (≥20 для устойчивого shear), модель (euclidean/similarity/affine).
- К нашей паре: **самый прямой ответ** на «наклонил заголовок» (δθ) и «блок стал параллелограммом» (γ). При 300 dpi поворот 0,1° на заголовке 100 мм = 2 px — по сотням CC различимо. **Минус** — надо аккуратно сопоставлять CC (слипания/разрывы при разной бинаризации двух прогонов; фильтровать по площади и взаимной ближайшести).

**2.4. Энергия изгиба TPS как мера нежёсткости**
- Суть: TPS f: B→A по соответствиям; энергия изгиба E = tr(WᵀKW) (аффинная часть бесплатна, поэтому E — чисто неаффинный компонент).
- Реализация: `skimage.transform.ThinPlateSplineTransform` (0.22+; W и K не публичны — лежат в `_spline_mappings`, K восстанавливается по `src`), `scipy.interpolate.RBFInterpolator(kernel='thin_plate_spline', smoothing=λ)` (коэффициенты в `_coeffs`), pip `thin-plate-spline` 1.2.2 (numpy/scipy, параметр alpha регуляризации), [cheind/py-thin-plate-spline](https://github.com/cheind/py-thin-plate-spline) (MIT, 2018, ноутбук).
- К нашей паре: **плюс** — одно число «сколько локального варпа», не зависит от поворота/шира; **минус** — нет знака, чувствительна к выбросам сопоставления, масштаб зависит от числа/плотности точек — нормировать нужно самим. Второстепенная метрика.

**2.5. Плотное поле смещений (оптический поток) и его разложение**
- Суть: после грубой привязки (`cv2.phaseCorrelate` или аффин по ORB) считаем плотный поток B→A; затем по маске элемента: (а) МНК-аффин на поле → δθ, γ, масштабы; (б) остаток; (в) локальные деформации: F = I + ∇u (np.gradient), `scipy.linalg.polar(F)` → R (угол) и U (растяжения; сдвиг = внедиагональ), либо инфинитезимально ω = ½(∂v/∂x − ∂u/∂y), γ = ∂u/∂y + ∂v/∂x, ε_xx, ε_yy — то же, что карты деформаций в DIC.
- Реализация: `cv2.DISOpticalFlow_create(PRESET_MEDIUM)` (параметры `finest_scale`, `patch_size` 8, `patch_stride`, `variational_refinement_iterations`, `use_spatial_propagation`), `cv2.calcOpticalFlowFarneback`; нейросетевые RAFT/GMFlow/… через pip `ptlflow` ([hmorimitsu/ptlflow](https://github.com/hmorimitsu/ptlflow), Apache-2.0, push 2026-07, Python 3.10–3.12, torch 2.3–2.6, GPU).
- К нашей паре: **плюс** — DIS быстрый (десятки мс на страницу), ничего сопоставлять руками не нужно, даёт и аффин элемента, и локальные варпы, и Li-D-подобные профили; **минус** — на бинарных картинках проблема апертуры и «залипание» на пустой бумаге: подавать размытые (σ≈1–2 px) или distance-transform версии, взвешивать по краске (как AD), учитывать, что при разной бинаризации двух прогонов толщина штрихов различается (даёт ложный «масштаб»). RAFT/GMFlow — избыточно: обучены на натуральных сценах, GPU одна на всех.

**2.6. Библиотеки DIC (digital image correlation)**
- [muDIC](https://github.com/PolymerGuy/muDIC) — MIT, B-spline-КЭ сетка, деформационный градиент и деформации; **последний push 2022-02**, PyPI 0.2.1. [py2DIC](https://github.com/Geod-Geom/py2DIC) — template matching, деформации Грина–Лагранжа, GUI, активен (push 2026-05), но лицензия **только некоммерческая**. [pydic](https://gitlab.com/damien.andre/pydic) — GPLv3, OpenCV-template matching.
- К нашей паре: суть DIC = сетка окон + `cv2.matchTemplate` с субпиксельной интерполяцией → разреженное поле → сглаживание → градиенты. Это 50–80 строк на OpenCV; библиотеки не дают ничего сверх п. 2.5, зато тянут GUI/лицензии. Отвергнуть как зависимости, взять подход (окно 48–64 px, шаг 16, поиск ±8 px) как альтернативу DIS там, где DIS плывёт.

---

### 3. Структурный уровень

- **Tesseract**: baseline строки — квадратичный сплайн (`FitBaselineSplines` в textord/baselinedetect), наружу через `ResultIterator.Baseline(RIL_WORD)` только отрезки по словам — кривизну строки можно собрать из углов пословных отрезков, но surya-baseline у нас уже есть; `--psm 0` OSD даёт только ориентацию кратно 90°; tabfind — колонки. Нового не даёт.
- **Shafait–Breuel page frame** (IJDAR 2008): геометрическое сопоставление рамки по «свойству выравнивания текста»; из него практично только сам принцип — **кромка выключенного блока**: левый (правый) край каждой строки → RANSAC-прямая → угол кромки и RMS «дрожания»; разность A−B по углу = «перекосил кромку». Библиотек нет, своя реализация.
- **Линейки таблиц / рамки / кромки фото**: `cv2.createLineSegmentDetector` (LSD, снова доступен с OpenCV 4.5.1), `cv2.ximgproc.createFastLineDetector`, `EdgeDrawing` (ximgproc), `skimage.transform.probabilistic_hough_line`; для одной линейки — скелет длинного горизонтального прогона → прямая → сагитта и угол (это v14). Сдвиг граф таблицы: положения вертикальных линеек из проекции по x после аффинной привязки A↔B — разность позиций в мм.
- **Поле ориентаций** (`skimage.feature.structure_tensor` + собственные векторы по окнам, либо docstrum): даёт одновременно направление строк и направление штрихов на блок — это дешёвая реализация μ4 из п. 1.3 без сопоставления версий.
- Kim & Koo 2015, Kil et al. ICDAR 2017 ([taeho-kil/Document-Image-Dewarping](https://github.com/taeho-kil/Document-Image-Dewarping), 2019, без лицензии, Windows-бинарь): их cost = прямизна строк + прямизна/ортогональность отрезков — та же идея, кода как библиотеки нет.

---

### 4. Сводка библиотек

| Пакет | Что умеет | Лицензия / живость | Встраивать? |
|---|---|---|---|
| OpenCV `DISOpticalFlow`, `estimateAffine2D`, `phaseCorrelate`, LSD | плотный поток, аффин с RANSAC, привязка, отрезки | есть | **да**, всё уже в проекте |
| skimage `AffineTransform` (.rotation/.shear/.scale), `ThinPlateSplineTransform`, `ransac`, `structure_tensor`, `hough_line` | разложение аффина, TPS, робастная подгонка, ориентации | есть | **да** |
| scipy `linalg.polar`, `spatial.procrustes`, `interpolate.RBFInterpolator('thin_plate_spline')`, `cKDTree` | полярное разложение, прокруст, TPS, сопоставление | есть | **да** |
| `deskew` (sbrunner) | глобальный скос по Hough | MIT, push 2026-09 | не нужен (на блоках — сами) |
| `jdeskew` | глобальный скос по Фурье | MIT, push 2026-08 | не нужен, разве что для сверки |
| `page-dewarp` (lmmx) | кубическая модель листа | MIT, push 2026-09 | нет — не та модель дисторсии |
| `ptlflow` | RAFT/GMFlow и др. | Apache-2.0, push 2026-07, GPU | резерв, если DIS не хватит |
| `thin-plate-spline` (pip) | TPS с регуляризацией | 1.2.2, лицензия не указана | не нужен, есть skimage/scipy |
| muDIC / py2DIC / pydic | поля деформаций | MIT-2022 мёртв / некоммерч. / GPL | нет |
| ocrd_anybaseocr, ocrd_cis | dewarp GAN, deskew проекцией | Apache/MIT, 2025/2024 | нет |
| DocTr / DocGeoNet / DocScanner / DewarpNet / UVDoc / docuwarp | дьюворп-сети; оценка — MATLAB | research-лицензии / MIT | нет |
| AADD, DocAligner | AAD / DD метрики | код не выложен / CUDA+CuPy, без лицензии | нет |

---

### Кандидаты на реальный тест (на эталоне 70 и паке)

1. **Аффин по элементу из сопоставленных CC + разложение (поворот δθ, шир γ, масштаб, RMS остатка)** — п. 2.3. Прямой измеритель «наклонил заголовок» и «параллелограмм». ~6–10 ч (сопоставление CC через cKDTree после глобальной привязки, фильтрация слипаний, пороги в мм/градусах).
2. **Ортогональность штрихов к строке (μ4 Dasgupta) через structure tensor по блоку** — п. 1.3/3. Без сопоставления версий, независимая проверка шира; калибровка нормального угла по гарнитуре. ~4–6 ч.
3. **DIS-поток на размытых бинарях после привязки → аффин по маске элемента + карты ω/γ/ε + AD-подобный остаток с весом по краске** — п. 2.5. Один прогон даёт всё для всех типов элементов, включая фото и таблицы. ~8–12 ч, из них половина — подбор параметров DIS и проверка на разной толщине штрихов.
4. **DW Стаматопулоса по строкам surya с вычетом собственной прямой строки** — п. 1.1. Знаковая, дешёвая, baseline уже есть. ~3–5 ч; главное — порог «стало хуже» вместо жёсткого 0.
5. **Энергия изгиба TPS по тем же соответствиям** — п. 2.4, как вторичный скаляр «локальный варп» для ранжирования mixed. ~2–3 ч поверх кандидата 1.

### Отвергнуть сразу

- **LD/AD/Li-D через SIFT-flow** — единственная рабочая реализация в MATLAB (mex), Python-порт только дескрипторов; идею взять, реализацию — нет.
- **AADD, DD/DocAligner** — код не выложен / CUDA+CuPy без лицензии, домен «фото→скан».
- **DIC-библиотеки** — мёртвые или с неподходящей лицензией; подход воспроизводится на OpenCV.
- **Нейросетевые дьюворперы и их бенчмарки** — дают варп, а не метрику, требуют GT, обучены на фотографиях.
- **`page-dewarp`, ocrd_anybaseocr, ocrd_cis** — глобальная модель листа / GAN / проекционный deskew: не по-элементно, лишние зависимости.
- **RAFT/GMFlow (ptlflow)** — избыточно для субпиксельных смещений бинарного текста, занимает общую GPU; оставить в резерве.
- **Глобальные deskew-пакеты** (`deskew`, `jdeskew`) — верны, но меряют не то: нужен угол элемента, а не страницы.

Sources: [Stamatopoulos, IET IP](https://digital-library.theiet.org/doi/10.1049/iet-ipr.2011.0208) · [Dasgupta et al. 2020, arXiv 2003.06872](https://arxiv.org/abs/2003.06872) · [Jiang et al. CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.html) · [Axis-Aligned Document Dewarping](https://arxiv.org/html/2507.15000) · [AADD repo](https://github.com/chaoyunwang/AADD) · [Enhancing document dewarping evaluation (DD), PRL 2025](https://www.sciencedirect.com/science/article/abs/pii/S0167865525001801) · [DocAligner](https://github.com/ZZZHANG-jx/DocAligner) · [DocScanner](https://github.com/fh2019ustc/DocScanner) · [DocGeoNet](https://github.com/fh2019ustc/DocGeoNet) · [DocTr](https://github.com/fh2019ustc/DocTr) · [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) · [UVDoc](https://github.com/tanguymagne/UVDoc) · [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) · [sift-flow-gpu](https://github.com/hmorimitsu/sift-flow-gpu) · [ptlflow](https://github.com/hmorimitsu/ptlflow) · [OpenCV DIS](https://docs.opencv.org/3.4/da/d06/classcv_1_1optflow_1_1DISOpticalFlow.html) · [skimage TPS](https://scikit-image.org/docs/stable/auto_examples/transform/plot_tps_deformation.html) · [skimage _thin_plate_splines.py](https://github.com/scikit-image/scikit-image/blob/v0.25.0/skimage/transform/_thin_plate_splines.py) · [thin-plate-spline на PyPI](https://pypi.org/project/thin-plate-spline/) · [py-thin-plate-spline](https://github.com/cheind/py-thin-plate-spline) · [muDIC](https://github.com/PolymerGuy/muDIC) · [py2DIC](https://github.com/Geod-Geom/py2DIC) · [pydic](https://gitlab.com/damien.andre/pydic) · [обзор skew estimation](https://www.tandfonline.com/doi/full/10.1080/08839514.2011.607009) · [jdeskew](https://github.com/phamquiluan/jdeskew) · [deskew](https://github.com/sbrunner/deskew/blob/master/README.md) · [ocrd_cis deskew](https://github.com/cisocrgroup/ocrd_cis/blob/master/ocrd_cis/ocropy/deskew.py) · [ocrd_anybaseocr](https://github.com/OCR-D/ocrd_anybaseocr) · [page-dewarp](https://github.com/lmmx/page-dewarp) · [Tesseract baselinedetect](https://github.com/tesseract-ocr/tesseract/blob/main/src/textord/baselinedetect.h) · [Tesseract ICDAR 2007](https://tesseract-ocr.github.io/docs/tesseracticdar2007.pdf) · [Shafait–Breuel page frame](https://link.springer.com/article/10.1007/s10032-008-0071-7) · [Kil et al. 2017](https://github.com/taeho-kil/Document-Image-Dewarping) · [polar decomposition](https://www.sciencedirect.com/topics/engineering/polar-decomposition)
