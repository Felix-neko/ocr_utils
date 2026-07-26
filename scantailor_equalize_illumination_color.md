# Реализация опции «Equalize Illumination (Color)» в ScanTailor Advanced

Отчёт по анализу исходного кода в `/home/felix/Projects/scantailor-advanced`.

> Краткая суть: для цветных (и серых) изображений эта опция оценивает фон страницы
> в виде гладкой **полиномиальной поверхности**, а затем «приподнимает» каждый пиксель
> относительно этого фона, выравнивая неравномерную засветку (тени от корешка, виньетирование).
> Для цветных изображений яркостная коррекция вычисляется по серой версии, а потом
> переносится на цветные каналы через YUV-подобное преобразование.

---

## 1. Где это находится (файлы)

### Слой UI / параметров

| Файл | Роль |
|------|------|
| `src/core/filters/output/OptionsWidget.ui` (строка 260) | Чекбокс с подписью **`Equalize illumination (Color)`** (`equalizeIlluminationColorCB`). Парный B&W-вариант на стр. 250. |
| `src/app/DefaultParamsDialog.ui` (строка 1994) | Тот же чекбокс в диалоге параметров по умолчанию. |
| `src/core/filters/output/OptionsWidget.cpp` | Логика чекбокса: слоты `equalizeIlluminationColorToggled()` (стр. 254) и взаимосвязь с B&W-вариантом `equalizeIlluminationToggled()` (стр. 235); инициализация состояния (стр. 626–630). |
| `src/core/filters/output/ColorCommonOptions.{h,cpp}` | Хранение флага `m_normalizeIllumination`, сериализация в XML как атрибут **`normalizeIlluminationColor`**. |
| `src/core/filters/output/RenderParams.{h,cpp}` | Преобразует флаги опций в битовую маску. Бит `NORMALIZE_ILLUMINATION_COLOR` (RenderParams.cpp, стр. 61–63). |

### Слой обработки изображения (собственно алгоритм)

| Файл | Роль |
|------|------|
| `src/core/filters/output/OutputGenerator.cpp` | Главный конвейер вывода. Ключевые функции: `transformToWorkingCs()` (стр. 2583), `normalizeIlluminationGray()` (стр. 1893), struct `RaiseAboveBackground` (стр. 366), вызовы для цветного пути (стр. 1419, 1556–1576, 1801). |
| `src/core/EstimateBackground.{h,cpp}` | `estimateBackground()` — оценка фона полиномиальной поверхностью + морфологическая предобработка и построение маски (EstimateBackground.cpp, стр. 141–289). |
| `src/imageproc/PolynomialSurface.{h,cpp}` | Аппроксимация поверхности фона полиномом методом наименьших квадратов. |
| `src/imageproc/AdjustBrightness.{h,cpp}` | `adjustBrightnessGrayscale()` — перенос яркостной коррекции с серого изображения на цветное (стр. 74). |
| `src/imageproc/WienerFilter.{h,cpp}` | `wienerColorFilterInPlace()` — сопутствующее подавление шума (выполняется после нормализации). |

---

## 2. Как связаны флаги (B&W vs Color)

Есть два независимых, но связанных флага:

- `BlackWhiteOptions::normalizeIllumination()` → бит **`NORMALIZE_ILLUMINATION`** → чекбокс *Equalize illumination (B&W)*.
- `ColorCommonOptions::normalizeIllumination()` → бит **`NORMALIZE_ILLUMINATION_COLOR`** → чекбокс *Equalize illumination (Color)*.

Логика связи (`OptionsWidget.cpp`):

- В режиме **MIXED** «цветная» нормализация доступна **только если включена** B&W-нормализация
  (`equalizeIlluminationColorCB->setEnabled(checked)`, стр. 246; при выключении B&W сбрасывается и Color, стр. 240–245).
- В режиме чистого **COLOR_GRAYSCALE** «цветной» чекбокс доступен всегда (стр. 630).
- Видимость: B&W-чекбокс скрыт в чисто цветном режиме, Color-чекбокс скрыт в чисто B&W (стр. 627–629).

В `RenderParams.cpp` бит `NORMALIZE_ILLUMINATION` для цветного пути выставляется тоже от
`colorCommonOptions.normalizeIllumination()` (стр. 46–48 в ветке «не бинаризация»), а
`NORMALIZE_ILLUMINATION_COLOR` — безусловно от того же флага (стр. 61–63). То есть в выводе
`normalizeIllumination()` управляет тем, выполнять ли нормализацию вообще, а
`normalizeIlluminationColor()` — переносить ли результат на **цветные** каналы (а не только на серый прокси).

---

## 3. Алгоритм по шагам

Точка входа для цветного изображения — `transformToWorkingCs(bool normalize)` (OutputGenerator.cpp, стр. 2583):

```cpp
QImage OutputGenerator::Processor::transformToWorkingCs(bool normalize) const {
  QImage dst;
  if (normalize) {
    // 1. Считаем серую нормализованную версию.
    dst = normalizeIlluminationGray(m_inputGrayImage, m_preCropAreaInOriginalCs,
                                    m_xform.transform(), m_workingBoundingRect);
    if (m_colorOriginal) {
      // 2. Берём исходное ЦВЕТНОЕ изображение и переносим на него яркость серого.
      QImage colorImg = transform(m_inputOrigImage, ...);
      adjustBrightnessGrayscale(colorImg, dst);  // <-- ключевой шаг для Color
      dst = colorImg;
    }
  } else {
    ... // просто геометрическое преобразование без нормализации
  }
  return dst;
}
```

Аналогичная логика в пути с распрямлением (dewarping), OutputGenerator.cpp стр. 1556–1576:
сначала строится серый нормализованный `normalizedGray`, затем
`adjustBrightnessGrayscale(normalizedOriginal /*цветное*/, normalizedGray)`.

### Шаг A. Серая нормализация засветки — `normalizeIlluminationGray()` (стр. 1893)

```cpp
GrayImage toBeNormalized = transformToGray(input, xform, targetRect, assumeWeakNearest());
// Оценка фона полиномиальной поверхностью:
const PolynomialSurface bgPs = estimateBackground(toBeNormalized, considerationArea, ...);
GrayImage bgImg(bgPs.render(toBeNormalized.size()));   // фон, отрендеренный в полный размер
// «Приподнимаем» изображение над фоном:
grayRasterOp<RaiseAboveBackground>(bgImg, toBeNormalized);
return bgImg;   // bgImg теперь содержит нормализованный результат
```

### Шаг B. Оценка фона — `estimateBackground()` (EstimateBackground.cpp, стр. 141)

1. **Даунскейл** входа до размера, вписанного в **300×300** px (`reducedSize.scale(300, 300, KeepAspectRatio)`),
   через `scaleToGray`. Все дальнейшие вычисления — на этой уменьшенной картинке (для скорости),
   а поверхность потом рендерится в полный размер.

2. **Морфологическая предобработка** `morphologicalPreprocessingInPlace()` (стр. 65). Выбирается один из двух методов:
   - **Метод 1**: `seedFillGrayInPlace` от рамочного изображения (`createFramedImage`) с **CONN8**,
     затем `openGray` ядром **1×20** — убирает остатки букв. Хорош, когда тёмная область в середине,
     касается вертикальных краёв.
   - **Метод 2**: `seedFillTopBottomInPlace()` — заливка только по вертикали (тени от корешка/переплёта).
   - **Выбор метода**: берётся разность двух методов, аппроксимируется полиномом степени **(3, 3)**.
     Считается остаток между разностью и её аппроксимацией. Если пикселей со значением **> 10**
     меньше **1 %** от площади — это «тень», берётся метод 1; иначе «картинка» — метод 2
     (порог `sum < 0.01 * width * height`, стр. 128).

3. **Построение маски «что считать фоном»** (`BinaryImage mask`, изначально весь BLACK = учитывать):
   - Учитывается заданная область рассмотрения `areaToConsider` (`PolygonRasterizer::fillExcept`),
     остальное помечается «не учитывать».
   - **Горизонтальное сглаживание**: каждый столбец аппроксимируется `PolynomialLine` степени **2**;
     если пиксель стал значительно светлее (`*pBg + 30 < line[y]`, порог **30**) — он маскируется
     (вероятно текст/тёмный объект, а не фон).
   - **Вертикальное сглаживание**: каждая строка аппроксимируется `PolynomialLine` степени **4**;
     тот же порог **30**.
   - **Эрозия** маски ядром **3×3** (`erodeBrick`).
   - **Чистка строк/столбцов**: если в горизонтальной линии «учитываемых» (чёрных) пикселей
     меньше **width/4**, строка целиком обнуляется; аналогично для вертикали при < **height/4**.

4. **Финальная аппроксимация фона**: `PolynomialSurface(8, 5, background, mask)` —
   полином степени **8 по горизонтали** и **5 по вертикали**, по неотмаскированным пикселям.

### Шаг C. Полиномиальная поверхность — `PolynomialSurface` (PolynomialSurface.cpp)

- Метод наименьших квадратов через **нормальные уравнения**: `AᵀA · x = Aᵀb`. Матрицы `AtA`, `Atb`
  накапливаются инкрементально, без явного построения `A` (стр. 47–58).
- Координаты пикселей нормируются в диапазон **[0, 1]** (`calcScale`), значения яркости масштабируются
  коэффициентом **1/255** (`dataScale = 1.0 / 255.0`).
- Число коэффициентов = `(horDegree+1) * (vertDegree+1)`. При нехватке точек данных степени
  понижаются (`maybeReduceDegrees`).
- Перед решением — `fixSquareMatrixRankDeficiency(AtA)` для устойчивости (борьба с вырожденностью).
- Решение СЛАУ — `DynamicMatrixCalc<double>::solve()`; при исключении коэффициенты остаются нулевыми.
- `render()` восстанавливает значение поверхности в каждой точке (с округлением `+0.5/255`,
  итог зажимается в [0, 255]).

### Шаг D. «Подъём над фоном» — `RaiseAboveBackground` (OutputGenerator.cpp, стр. 366)

Это и есть сама нормализация яркости (деление на фон), per-pixel:

```cpp
static uint8_t transform(uint8_t src /*orig*/, uint8_t dst /*background, dst >= src*/) {
  if (dst - src < 1) return 0xff;                       // фон == объект → чисто белый
  return (orig * 255 + background / 2) / background;    // нормировка к диапазону фона
}
```

Смысл: `out = orig / background * 255` (с округлением `+ background/2`). Там, где исходный пиксель
равен фону — получаем 255 (белый); тёмные детали (текст) сохраняют контраст относительно
выровненного до белого фона.

### Шаг E. Перенос яркости на цвет — `adjustBrightnessGrayscale()` (AdjustBrightness.cpp, стр. 74)

Именно этот шаг отличает Color-вариант от B&W: рассчитанная серая «новая яркость» (`brightness`)
накладывается на исходные RGB-пиксели, сохраняя цветность (хрому).

```cpp
void adjustBrightnessGrayscale(QImage& rgb, const QImage& brightness) {
  adjustBrightness(rgb, brightness, 11.0/32.0, 5.0/32.0);   // wr=0.34375, wb=0.15625
}
```

Внутри `adjustBrightness(rgbImage, brightness, wr, wb)` (стр. 10) для каждого пикселя:

- веса каналов: `wr`, `wb` заданы, `wg = 1 - wr - wb` (для grayscale-варианта `wg = 16/32 = 0.5`);
- вычисляется яркость `Y = wr·R + wg·G + wb·B` и «цветоразностные» компоненты
  `U = (B − Y)/(1−wb)`, `V = (R − Y)/(1−wr)` (YUV-подобное разложение);
- **новая яркость** `new_Y` берётся из серого нормализованного изображения (`brLine[x]`);
- обратное восстановление RGB при сохранённых U, V:
  `new_R = new_Y + V·(1−wr)`, `new_B = new_Y + U·(1−wb)`, `new_G = (new_Y − new_R·wr − new_B·wb)/wg`;
- результат зажимается в [0, 255], альфа-канал сохраняется (`RGB &= 0xFF000000`).

> Замечание: есть и второй пресет `adjustBrightnessYUV()` с коэффициентами **0.299 / 0.114**
> (стандартные BT.601), но для выравнивания засветки используется именно `adjustBrightnessGrayscale`
> с весами **11/32 и 5/32**.

### Шаг F. Сопутствующий Wiener-фильтр (не часть нормализации, но в том же конвейере)

Сразу после `transformToWorkingCs(...)` к результату применяется
`wienerColorFilterInPlace(maybeNormalized, QSize(winSize, winSize), wienerCoef)`
(OutputGenerator.cpp, стр. 1289 и 1581). Параметры — из `ColorCommonOptions`:
`wienerWindowSize` (по умолчанию **5**, минимум 3) и `wienerCoef` (по умолчанию **0.0** = выключен,
диапазон [0, 1]). Фильтр работает по серой версии (`noise_sigma = 255 · coef`) и масштабирует
цветные каналы (`colscale`, `coldelta`) — WienerFilter.cpp, стр. 116. При `coef == 0` ничего не делает.

---

## 4. Сводка констант и параметров

| Константа / параметр | Значение | Где | Назначение |
|----------------------|----------|-----|------------|
| Размер даунскейла для оценки фона | вписать в **300×300** px | EstimateBackground.cpp:146 | скорость аппроксимации |
| Степень полинома итогового фона | **(8, 5)** (гор., верт.) | EstimateBackground.cpp:288 | гладкая поверхность фона |
| Степень полинома для выбора метода | **(3, 3)** | EstimateBackground.cpp:102 | оценка «тень vs картинка» |
| Степень `PolynomialLine` (столбцы / строки) | **2 / 4** | EstimateBackground.cpp:189,210 | маскирование светлых пикселей |
| Порог «стал светлее» | **+30** | EstimateBackground.cpp:196,215 | отбрасывание текста из фона |
| Ядро `openGray` (метод 1) | **1×20** | EstimateBackground.cpp:80 | удаление остатков букв |
| Эрозия маски | **3×3** | EstimateBackground.cpp:230 | сужение учитываемой области |
| Порог чистки строк/столбцов | **width/4**, **height/4** | EstimateBackground.cpp:257,275 | удаление почти пустых линий |
| Порог «картинка vs тень» | пикселей >10 должно быть ≥ **1 %** | EstimateBackground.cpp:128 | выбор метода предобработки |
| Подъём над фоном (полностью белый) | `dst − src < 1 → 0xFF` | OutputGenerator.cpp:370 | защита от деления на ~0 |
| Формула нормализации | `(orig·255 + bg/2)/bg` | OutputGenerator.cpp:375 | деление на фон с округлением |
| Веса яркости (Color) | `wr=11/32`, `wb=5/32`, `wg=0.5` | AdjustBrightness.cpp:75 | перенос яркости на RGB |
| Веса яркости (YUV-вариант) | `wr=0.299`, `wb=0.114` | AdjustBrightness.cpp:71 | альтернативный пресет (не используется здесь) |
| Масштаб данных в полиноме | **1/255** | PolynomialSurface.cpp:213,291 | нормировка яркости |
| Координаты в полиноме | диапазон **[0, 1]** | PolynomialSurface.cpp | нормировка координат |
| Wiener: размер окна | **5** (мин. 3) | ColorCommonOptions.cpp:15,31 | сглаживание (опц.) |
| Wiener: коэффициент | **0.0** (диап. [0, 1]) | ColorCommonOptions.cpp:14,27 | сила фильтра (0 = выкл.) |
| `noise_sigma` Wiener | **255 · coef** | WienerFilter.cpp:134 | связь coef и дисперсии шума |

---

## 5. Итоговый поток данных (Color)

```
исходное цветное изображение (m_inputOrigImage)
        │
        ├─► серый прокси (m_inputGrayImage)
        │        │
        │        ▼
        │   estimateBackground():
        │     downscale 300×300 → морфол. предобработка (метод 1/2)
        │     → маска фона (полин. линии, эрозия, чистка)
        │     → PolynomialSurface(8,5)  → render() = фон
        │        │
        │        ▼
        │   RaiseAboveBackground: out = orig/фон·255  →  серая «новая яркость»
        │        │
        ▼        ▼
   adjustBrightnessGrayscale(цвет, новая_яркость)   [wr=11/32, wb=5/32]
        │   (Y заменяется на новую яркость, U/V — хрома — сохраняются)
        ▼
   wienerColorFilterInPlace (опционально, coef>0)
        ▼
   нормализованное цветное изображение → дальнейший вывод
```

Ключевой момент: **геометрия/величина** выравнивания засветки полностью определяется серым
прокси-изображением (полиномиальная модель фона), а Color-вариант лишь корректно переносит
эту яркостную правку на цветные каналы, не разрушая цветовой тон.

---

## 6. Зависимости от библиотек

Важный вывод: **вся математика и обработка изображений реализованы внутри самого ScanTailor**
(собственные статические библиотеки `imageproc`, `math`, `foundation`). Внешних
численных/CV-библиотек (OpenCV, Eigen, LAPACK/BLAS, GSL) алгоритм **не использует**.

### 6.1. Внешние библиотеки

| Библиотека | Версия | Что используется в этом алгоритме | Где видно |
|------------|--------|-----------------------------------|-----------|
| **Qt** (Qt6, либо Qt5 ≥ как fallback) | модули `Core`, `Gui`, `Xml` | `QImage` (контейнер изображения, доступ к битам/каналам RGB), `QSize`, `QRect`, `QTransform`, `QPolygonF` (геометрия областей и преобразований), `QDomElement`/`QDomDocument` (сериализация опций в XML, атрибут `normalizeIlluminationColor`), `QDebug`, `qBound` | `#include <QImage>` в AdjustBrightness.cpp; `<QSize>`/`<QtGlobal>` в WienerFilter.cpp; `<QDomDocument>` в ColorCommonOptions.cpp; CMake `foundation → Qt_Core/Qt_Xml/Qt_Gui` |
| **Boost** (≥ 1.60, только header-only части) | `boost::lambda` (`bind`, `lambda`, `control_structures`) и `boost::scoped_array` | Лямбда-функторы для попиксельных операций `rasterOpGeneric` при выборе метода предобработки фона; `scoped_array` — в решателе СЛАУ | `boost/lambda/*` в EstimateBackground.cpp:21–23; `boost/scoped_array.hpp` в `math/LinearSolver.h` |
| **libjpeg / zlib / libpng / libtiff** | — | Только ввод-вывод файлов; в самом алгоритме выравнивания засветки **не участвуют** (линкуются к `core` для чтения/записи страниц) | CMake `find_package(JPEG/ZLIB/PNG/TIFF REQUIRED)`, линкуются в `core` как PRIVATE |

> Стандарт языка — **C++17** (`set(CMAKE_CXX_STANDARD 17)`). Из стандартной библиотеки
> используются `<cmath>`, `<algorithm>`, `<cstdint>`, `<stdexcept>`, `<vector>`, `<cassert>`.

### 6.2. Внутренние (собственные) библиотеки проекта

Это не внешние зависимости, а модули самого ScanTailor — здесь сосредоточена вся «начинка»:

| Внутр. библиотека (CMake target) | Используемые компоненты | Назначение в алгоритме |
|----------------------------------|--------------------------|------------------------|
| **`imageproc`** (`src/imageproc`, статическая) | `GrayImage`, `BinaryImage`, `PolynomialSurface`, `PolynomialLine`, `Morphology` (`openGray`, `erodeBrick`, `createFramedImage`), `SeedFill` (`seedFillGrayInPlace`), `Scale` (`scaleToGray`), `Transform`/`transformToGray`, `GrayRasterOp`/`RasterOpGeneric`, `IntegralImage` (для Wiener), `WienerFilter`, `AdjustBrightness`, `Grayscale`/`GrayscaleHistogram`, `PolygonRasterizer`, `BitOps` (`countNonZeroBits`), `AlignedArray`, `Connectivity` (CONN8) | Все операции над изображениями: оценка фона, полином, морфология, нормализация, перенос яркости на цвет, Wiener |
| **`math`** (`src/math`, статическая) | `PolynomialSurface` решает `AᵀAx=Aᵀb` через `MatrixCalc`/`DynamicMatrixCalc` → `LinearSolver`; `MatT`/`VecT`/`MatMNT`/`VecNT` | Линейная алгебра: метод наименьших квадратов. Решатель — **собственная реализация LU-разложения с частичным выбором ведущего элемента** (`LinearSolver.h`), а не Eigen/LAPACK |
| **`foundation`** (`src/foundation`, статическая) | базовые утилиты, `NonCopyable`, обёртки над Qt | Низкоуровневая инфраструктура |
| **`core`** (`src/core`) | `OutputGenerator`, `EstimateBackground`, `RenderParams`, `BackgroundColorCalculator`, `ImageTransformation`, `TaskStatus`, `DebugImages` | Оркестрация конвейера вывода и сама опция |

Граф линковки (по CMake): `core → imageproc → (math → foundation → Qt)`; плюс `core`
приватно линкует `TIFF/PNG/ZLIB/JPEG` только для I/O.

### 6.3. Итог

- Для **портирования/повторения** алгоритма достаточно эквивалента `QImage` (буфер пикселей)
  и решателя плотной СЛАУ малого размера — например, в Python это `numpy`
  (`numpy.linalg.lstsq`/`solve`) или `scipy`, плюс `Pillow`/`OpenCV` как контейнер изображения.
- В оригинале же **нет ни OpenCV, ни Eigen, ни LAPACK** — линейная алгебра и вся обработка
  изображений написаны вручную внутри ScanTailor; внешние зависимости сводятся к **Qt** и
  **header-only Boost** (+ форматные библиотеки только для чтения/записи файлов).

---

## 7. Обёртка через pybind11 vs порт на Python

> **Контекст использования:** инструмент внутренний, поэтому лицензия **GPLv3 приемлема** —
> и это снимает главный нетехнический блокер обёртки. Вывод: для внутреннего использования
> **заворачивание существующего C++-кода в pybind11 предпочтительнее порта.**

### 7.1. Почему обёртка реальна и несложна

Алгоритм «разрезается» по очень удобным швам — переписывать `OutputGenerator` не нужно:

- **`estimateBackground()` — свободная функция** (`src/core/EstimateBackground.cpp`), а не метод
  тяжёлого `OutputGenerator::Processor`. Ей нужны только `GrayImage`, `QPolygonF` (можно пустой),
  `TaskStatus` и `DebugImages*` (можно `nullptr`).
- **`TaskStatus` — чистый абстрактный интерфейс из 3 методов** (`src/foundation/TaskStatus.h`).
  No-op заглушка занимает 5 строк:
  ```cpp
  struct NoopStatus : TaskStatus {
    void cancel() override {}
    bool isCancelled() const override { return false; }
    void throwIfCancelled() const override {}
  };
  ```
- **`GrayImage` строится из `QImage`** (`GrayImage(const QImage&)`), а `QImage` умеет оборачивать
  чужой буфер (`QImage(uchar*, w, h, bytesPerLine, Format_Grayscale8)`). То есть numpy ↔ изображение —
  пара строк через buffer protocol pybind11.
- Вся «начинка» опции — это `estimateBackground → render → RaiseAboveBackground →
  adjustBrightnessGrayscale`, ~15 строк glue, которые переписываются в обёртке.
- **GUI-event-loop (QGuiApplication) не нужен:** в задействованной цепочке нет `QPainter`-рендеринга
  (`PolygonRasterizer` использует только `QPainterPath`-геометрию со своим растеризатором), нет
  `Q_OBJECT`/moc в `imageproc`/`math`/`foundation`. Всё работает headless.

### 7.2. Трудности обёртки (по убыванию)

| # | Трудность | Серьёзность | Комментарий |
|---|-----------|-------------|-------------|
| 1 | Лицензия **GPLv3** | 🟢 снято (инструмент внутренний) | Линковка с кодом ScanTailor делает обёртку производным произведением → GPLv3. Для внутреннего использования допустимо. |
| 2 | Зависимость от **Qt Core+Gui** | 🟡 средняя | `QImage` живёт в QtGui (но без QApplication). Нужны dev-пакеты Qt при сборке и `libQt6Core`+`libQt6Gui` в окружении. Не нужны Widgets/OpenGL/событийный цикл. |
| 3 | Вырезание **standalone-сборки** | 🟡 механическая | Монолитный CMake собирает приложение целиком. Нужен маленький отдельный `CMakeLists`, собирающий 3 статические либы (`foundation`, `math`, `imageproc`) + один `EstimateBackground.cpp` + обёртку как pybind-модуль. `EstimateBackground.cpp` формально в `core`, но зависит только от `imageproc` + пары хедеров — его проще компилировать напрямую, чем линковать весь `core`. |
| 4 | **Marshalling + glue** | 🟢 лёгкая | numpy↔QImage и ~15 строк логики. Полдня. |

### 7.3. Минимальный план (прототип за 1–2 дня)

1. `CMakeLists.txt`: `add_subdirectory` на `foundation`, `math`, `imageproc` из дерева ScanTailor
   (они уже STATIC и самодостаточны) + `pybind11_add_module`.
2. Один `wrapper.cpp`: `#include` нужных хедеров, `NoopStatus`, функция
   `equalize_illumination_color(numpy rgb) -> numpy rgb`, повторяющая ~15 строк из
   `normalizeIlluminationGray` / `transformToWorkingCs` / `adjustBrightnessGrayscale`.
3. Прямо включить в таргет `src/core/EstimateBackground.cpp`.

### 7.4. Сравнение подходов

| Критерий | Обёртка pybind11 | Порт на Python |
|----------|------------------|----------------|
| Объём кода | ~1 `CMakeLists` + ~1 `wrapper.cpp` | переписать полином/МНК, морфологию, seed-fill, пороги |
| Риск расхождения результата | нет (тот же LU-решатель, те же пороги/морфология) | есть — нужно валидировать совпадение пиксель-в-пиксель |
| Зависимости рантайма | Qt Core+Gui (тяжело) | numpy + Pillow/OpenCV (легко) |
| Лицензия | GPLv3 (для внутреннего — ок) | свободная |
| Скорость до результата | быстрее | медленнее |

**Рекомендация для данного (внутреннего) случая:** заворачивать существующий код в **pybind11**.
Меньше работы, нет риска численных расхождений; единственная цена — присутствие Qt Core+Gui
в окружении сборки/исполнения.
