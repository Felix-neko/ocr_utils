<!-- Обзор: нежёсткость деформации line art, дробные черты, ступеньки и дуги заголовков (2026-09-22, агент-обзор WebSearch/WebFetch). Сводка и решения — reports/geometry_regression_v15.md, раздел v16 -->

Обзор готов (по WebSearch/WebFetch; код репо не смотрел, кроме того, что видно из истории коммитов v15 — якобиан поля и кромки уже есть).

## 1. Line art: локальная нежёсткость поля смещений

**1a. Полярное разложение локального якобиана (DIC-подход, «strain»).** Суть: в каждом окне поля F = I + ∇u (2×2), F = R·U; R — локальный поворот, U — симметричное растяжение с собственными λ₁ ≥ λ₂. Неконформность = (λ₁−λ₂)/(λ₁+λ₂) (это и есть |μ| Бельтрами в первом порядке), сдвиг осей = угол между образами ортов F·e_x и F·e_y минус 90° — ровно «угол между осями графика изменился на 1–2°». Неоднородность R по площади рисунка = «ребро повёрнуто на 3°, а соседний текст нет». Параметры: окно оценки (3×3 тайла поля или локальный МНК-аффин по узлам, как в py2DIC/pydic «strain radius»), порог: 1° сдвига ≈ 0.017 rad, 2 % разности λ. Реализация: numpy на 2×2 в закрытом виде или `scipy.linalg.polar`; py2DIC/pydic считают Green–Lagrange из поля, но тянуть их не нужно. Применимость: прямая, поле уже есть; в v15 якобиан для перекоса блоков уже считается — это расширение той же ветки. **4–6 ч.**

**1b. ASAP/ARAP-остаток на тайл.** По углам/узлам тайла подобрать similarity (`skimage.transform.estimate_transform('similarity')`) и rigid (`scipy.linalg.orthogonal_procrustes`), остаток — «энергия» Igarashi/Sorkine. Разница «affine-остаток − similarity-остаток» даёт ту же неконформность, что 1a, а «поле − affine» — нелинейность (кривизна). Это дубль 1a в другой упаковке; полезно только как контроль. **2–3 ч**, но брать одно из двух.

**1c. Прямизна и углы по самим линиям (Kil 2017, UVDoc LineAcc).** Для длинной LSD/трассированной линии в A пронести её образ через поле в B (или трассировать в B напрямую): стрелка прогиба (max отклонение от хорды) в мм и в толщинах штриха; для пар пересекающихся линий — Δ угла A→B ≥ 1°; для пучков «перспективных» рёбер — согласованность поворота. Kil et al. кодируют ровно эти два свойства (straightness + alignment) в кост, UVDoc меряет прямизну образов прямых, но обоим нужен GT-варп — нам нет, у нас есть A как эталон. В v15 изгиб и параллельность уже есть; добавить Δ угла на пересечениях и «прямизну образа». **3–5 ч.**

**Отвергнуть сразу:** TPS bending energy (глобальный интеграл, не локализует, на редких узлах шумит); Beltrami/квазиконформный анализ как отдельный метод (в первом порядке = 1a); MIPS/symmetric Dirichlet (энергии для оптимизации сеток, для детекции эквивалентны 1a); LD/AD/AAD из дьюворпинга (нужны SIFT-flow и GT, AAD меряет только выровненность по осям, наши оси в A не обязаны быть горизонтальны); «strain alignment» из медицинской регистрации — 3D-специфика.

## 2. Дробные черты

**2a. Компонентный детектор + контекст «краска над и под».** Как в Tesseract EquationDetect и патентах по MER: горизонтальная компонента с шириной/высотой ≥ 4–5 и толщиной 1–3 px; дробная черта — есть краска и над, и под чертой в пределах её x-проекции (перекрытие ≥ 50 % длины, зазор ≤ 1–1.5 высоты строки); тире/минус — краска слева-справа на общей базовой линии, над/под пусто; подчёркивание — краска только над. Twaakyondo–Okamoto и Infty делают то же через структурный анализ, ничего сложнее не нужно. **1.5–2 ч.**

**2b. Наклон черты.** Не LSD (штрих 1–2 px даёт две кромки и режется), а по компоненте: центроид краски в каждом столбце → Тейл–Сен (как у кромок в v15) или `cv2.fitLine`/`regionprops.orientation`. Точность ~ atan(1 px / длина): черта 5–8 мм при 300–600 dpi даёт 0.3–0.5°, значит 1–4° ловятся. Сравнивать угол той же черты A↔B (матчинг по полю), порог |Δθ| ≥ 1° при длине ≥ 4–5 мм, и требовать, чтобы длинные линейки таблиц рядом остались ровными (иначе это общий поворот). **2–3 ч.**

**Отвергнуть:** полное распознавание формул (InftyReader, pix2tex, Tesseract equation mode — тяжело и без выигрыша для угла); Хаф по черте (та же точность, хуже локализация).

## 3. Заголовки: ступенька, дуга, растяжение

**3a. Базовая линия по словам с остатками.** Компоненты строки → слова по зазорам; для каждого слова низ букв без нижних выносных (мода нижних точек, отбор ±0.3 x-height); базовая линия строки = Тейл–Сен по всем; per-word residual. Ступенька = скачок медиан residual между соседними словами ≥ 0.6–0.7 мм (для 1 мм); дуга = стрелка квадратичной/кубической аппроксимации против прямой (Stamatopoulos DW тоже строит cubic по строке, но с ручными точками и SIFT — идею берём, методику нет). Сравнивать с той же строкой в A, чтобы не ловить нарочитую вёрстку. surya-полигоны для 1 мм грубы; считать по своим компонентам. **4–6 ч.**

**3b. Вертикальный масштаб по словам.** Заголовки чаще капителью — высота всех компонент одна, удобно: по словам мода высоты компонент, коэффициент вариации вдоль строки в B против A; порог — отношение высот соседних слов отклоняется от 1 на ≥ 5 %. Альтернатива без сегментации: компонента U_yy из 1a внутри рамки строки и её разброс вдоль x — та же величина из поля, можно взять как перекрёстную проверку. **2–4 ч.**

**Отвергнуть:** DW Stamatopoulos и UVDoc line-straightness как есть (ручные точки/GT-варп), нейросетевые baseline-детекторы (dhSegment, Kraken — оверкилл и точность хуже 1 мм), верхняя линия/x-height через OCR-движок.

## Итог

| Пункт | Взять | Часы |
|---|---|---|
| 1 | 1a полярное разложение + 1c Δ угла на пересечениях/прямизна образа | 7–11 |
| 2 | 2a компонент + контекст, 2b Тейл–Сен по столбцам, Δθ A↔B | 4–5 |
| 3 | 3a per-word residual базовой + 3b вариация высоты по словам | 6–10 |

Sources: [Kil 2017 README](https://github.com/taeho-kil/Document-Image-Dewarping/blob/master/README.md), [Kil et al. ICDAR 2017](https://www.semanticscholar.org/paper/Robust-Document-Image-Dewarping-Method-Using-and-Kil-Seo/0051ee7ac39222ba7c5202957c1210821902f4ce), [UVDoc](https://arxiv.org/html/2302.02887v2), [Axis-Aligned Document Dewarping (AAD)](https://arxiv.org/html/2507.15000), [Grid Regularization](https://www.semanticscholar.org/paper/Revisiting-Document-Image-Dewarping-by-Grid-Jiang-Long/3ae9ceb45f24f6f863559f1aae384e619a9bfe87), [DocUNet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ma_DocUNet_Document_Image_CVPR_2018_paper.pdf), [Stamatopoulos DW](https://digital-library.theiet.org/doi/10.1049/iet-ipr.2011.0208), [Multistage dewarping quality estimator](https://arxiv.org/pdf/2003.06872), [ARAP Sorkine–Alexa](https://igl.ethz.ch/projects/ARAP/arap_web.pdf), [ARAP Igarashi](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/rigid.pdf), [Local/Global parameterization](https://www.eecs.harvard.edu/~sjg/papers/arap.pdf), [Beltrami coefficient](https://www.researchgate.net/figure/llustration-of-how-the-Beltrami-coefficient-m-measures-the-distortion-of-a_fig1_221363877), [Strain alignment](https://link.springer.com/article/10.1007/s11548-026-03729-6), [py2DIC](https://github.com/Geod-Geom/py2DIC), [skimage TPS](https://scikit-image.org/docs/stable/auto_examples/transform/plot_tps_deformation.html), [Tesseract EquationDetect](https://github.com/tesseract-ocr/tesseract/blob/main/src/ccmain/equationdetect.h), [MER patent: fraction vs minus](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10346681), [Infty OCR](https://link.springer.com/chapter/10.1007/978-3-540-27817-7_97), [Twaakyondo–Okamoto / OCR of printed math](https://link.springer.com/chapter/10.1007/978-1-84628-726-8_11), [Baseline detection](https://www.researchgate.net/publication/258651794_Novel_Approach_for_Baseline_Detection_and_Text_Line_Segmentation).
