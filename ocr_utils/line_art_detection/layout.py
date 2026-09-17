"""Предложения разметки страницы от Surya: таблицы без линеек и многострочные формулы.

ЗАЧЕМ, ЕСЛИ ЕСТЬ ПИКСЕЛЬНЫЙ ДЕТЕКТОР. Пиксели видят штрих — то, что даёт длинное связное
пятно или скопление линеек. Двух вещей из заказанного они не видят по построению:

* ТАБЛИЦА БЕЗ ЛИНЕЕК — это просто колонки текста, никаких длинных штрихов в ней нет;
* МНОГОСТРОЧНАЯ ФОРМУЛА — набор из тех же глифов, что и текст, только расставленных иначе.

Оба случая — вопрос разметки, а не пикселей, и на них Surya отвечает своими классами.
В ``surya/layout/label.py`` есть ровно нужные: ``Table``, ``Equation`` (``<equation-block>``),
``Figure`` (рисунки, схемы, графики) и ``Form``.

РЕШАЕТ ПО-ПРЕЖНЕМУ НЕ МОДЕЛЬ. Блок Surya здесь — только ПРЕДЛОЖЕНИЕ; годится оно или нет,
проверяют те же пиксельные правила, что и для связных пятен. Так сделано не из
осторожности, а по замеру: у Newspaper Navigator (Faster R-CNN, дообучен на 3559 газетных
полосах — ближайший к нашему домен) AP по классу *Illustration* равен 30.9% при mAP 63.4%,
а в самом этом проекте замерено, что Surya считает штриховой рисунок фотографией
(``Picture``) на 28 полосах из 31 (``scan_markup.detection.regions``). Поэтому ``Picture``
идёт не в находки, а в ИСКЛЮЧЕНИЯ — наравне с прямоугольниками из базы разметки.

ЧТО ЭТО ДАЁТ НА САМОМ ДЕЛЕ, ЗАМЕРЕНО. Прогон по эталонному выпуску (97 полос) нашёл
10 блоков ``Table``, 7 ``Equation`` и 1 ``Figure``. Почти все они лежат под находками,
которые пиксели видят и сами; ЧИСТАЯ прибавка — четыре страницы с многострочными
формулами (23, 25, 85, 87). И вот что про них важно знать: формула занимает
0.4-1.3% полосы, то есть порогом покрытия, настроенным на рисунки (5%), её не достать
никогда. Формулы выбираются не порогом, а источником: ``export --source surya:Equation``.

ЦЕНА И КЭШ. Около 0.7 с на страницу на GPU, то есть порядка двух часов на весь пак против
двух с половиной минут у пиксельного прохода. Поэтому, во-первых, флаг выключен по
умолчанию, а во-вторых, результат кладётся в кэш на диск: пороги калибруются итеративно,
и повторный прогон не должен трогать GPU вовсе (тот же довод и то же устройство, что у
``defocus_detection.lines.detect.DetectCache``).

GPU В ПУЛ НЕ ЗАВОРАЧИВАЕТСЯ (CLAUDE.md): видеопамять одна на всех. Surya крутится
последовательно в родительском процессе, а в воркеры уезжают только четвёрки чисел.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.background_smoothing.layout import LAYOUT_WORK_SIDE

logger = logging.getLogger(__name__)

# Классы, которые считаем предложениями находки.
FINDING_LABELS = ("Figure", "Table", "Equation", "Form")
# Классы, которые считаем полутоновой печатью и вычитаем из находок.
EXCLUDE_LABELS = ("Picture",)

CACHE_VERSION = 1


class LayoutProposals:
    """Surya LayoutPredictor: блоки страницы вместе с их классами.

    Своя обёртка, а не ``background_smoothing.layout.LayoutDetector``, по одной причине:
    тот отдаёт полигоны БЕЗ меток и только класса ``Picture``, а здесь метка и есть суть —
    по ней блок попадает либо в находки, либо в исключения. Загрузка модели устроена так
    же (ленивая, ровно один ``LayoutPredictor``, без тяжёлого набора из ``gpu_models``).
    """

    def __init__(self, cache_dir: "Path | None" = None) -> None:
        self._predictor = None
        self._cache_dir = cache_dir
        self._cache: "dict[str, dict]" = {}

    def _load(self):
        """Ленивая загрузка предиктора — импорт surya не быстрый, а он может не понадобиться."""
        if self._predictor is None:
            from surya.foundation import FoundationPredictor
            from surya.layout import LayoutPredictor
            from surya.settings import settings

            logger.info("Загружаю Surya layout (--use-surya-layout)")
            predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
            predictor.disable_tqdm = True  # иначе на каждой странице рвётся полоса прогресса
            self._predictor = predictor
        return self._predictor

    def _cache_path(self, pdf_path: Path) -> "Path | None":
        return None if self._cache_dir is None else Path(self._cache_dir) / f"{pdf_path.stem}.json"

    def _cached(self, pdf_path: Path) -> dict:
        """Кэш одного PDF; читается с диска один раз за прогон."""
        key = str(pdf_path)
        if key not in self._cache:
            path = self._cache_path(pdf_path)
            data = {}
            if path and path.exists():
                try:
                    stored = json.loads(path.read_text(encoding="utf-8"))
                    if stored.get("version") == CACHE_VERSION and stored.get("size") == pdf_path.stat().st_size:
                        data = stored.get("pages", {})
                except (OSError, ValueError):
                    logger.warning("Кэш разметки %s не прочитался, считаю заново", path)
            self._cache[key] = data
        return self._cache[key]

    def flush(self, pdf_path: Path) -> None:
        """Сохранить кэш одного PDF на диск."""
        path = self._cache_path(pdf_path)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_VERSION, "size": pdf_path.stat().st_size, "pages": self._cached(pdf_path)}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def boxes(self, pdf_path: Path, page_no: int, gray: np.ndarray) -> list[tuple[tuple[int, int, int, int], str]]:
        """Блоки страницы как ``(прямоугольник, метка)`` в координатах поданного кадра.

        Кадр ужимается до ``LAYOUT_WORK_SIDE`` по длинной стороне (модель всё равно
        ресайзит вход под свой процессор, а на 28-мегапиксельном рендере одна только
        конвертация стоит секунд), координаты возвращаются пересчитанными обратно.
        """
        cache = self._cached(pdf_path)
        key = str(page_no)
        if key in cache:
            return [((b[1], b[2], b[3], b[4]), b[0]) for b in cache[key]]

        from PIL import Image as PILImage

        height, width = gray.shape[:2]
        scale = min(1.0, LAYOUT_WORK_SIDE / max(height, width))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
        rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)

        found = []
        for block in self._load()([PILImage.fromarray(rgb)])[0].bboxes:
            points = np.asarray(block.polygon, dtype=np.float32).reshape(-1, 2) / scale
            box = (
                int(np.floor(points[:, 0].min())),
                int(np.floor(points[:, 1].min())),
                int(np.ceil(points[:, 0].max())),
                int(np.ceil(points[:, 1].max())),
            )
            found.append((box, block.label))

        cache[key] = [[label, *box] for box, label in found]
        return found


def split(blocks) -> tuple[list, list]:
    """Блоки Surya -> ``(предложения находок, исключения)``; прочие классы отбрасываются."""
    findings = [(box, label) for box, label in blocks if label in FINDING_LABELS]
    excludes = [box for box, label in blocks if label in EXCLUDE_LABELS]
    return findings, excludes
