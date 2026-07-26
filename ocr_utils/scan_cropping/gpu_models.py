"""Единая точка входа во все нейросети пайплайна: класс ``GpuModels``.

Раньше модели грузились лениво из пяти разных мест, у каждого был свой
модульный кэш (``_MODEL_CACHE`` в ``page_detection`` и ``masking``, ``_CACHE`` в
``finger_inpaint``, ``_LAYOUT_PREDICTOR`` в ``text_protection``, ``_NET_CACHE`` в
``finger_shadow``), и сквозь весь пайплайн передавалась строка ``device``. Из-за
этого нельзя было ни узнать, что загружено, ни выгрузить это, ни понять по коду,
где именно тратится VRAM.

Теперь всё живёт здесь. Объект создаётся ОДИН раз (в ``scan_cropping.cli``),
в конструкторе грузятся все нужные модели, и дальше по пайплайну передаётся
он сам, а не ``device``. Методы принимают и возвращают обычные numpy-массивы:
никакой тензор наружу не выходит и внутрь не заходит.

Что грузится всегда: YOLO-World для страниц, YOLO-World для рук, SAM, LaMa.
Что по флагам: Surya layout (``with_layout``), DocShadow (``shadow_variant``) —
и то и другое нужно лишь при отдельных опциях CLI и стоит дорого.

Вся эвристика (пороги площадей, отбор боксов, подавление вложенных) остаётся
СНАРУЖИ, в ``page_detection`` и ``finger_removal.masking``: класс отвечает только
за прогон сетей, чтобы правила отбора можно было читать и править, не думая
про GPU.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# Папка для весов нейромоделей — в корне проекта. Единственное определение на
# весь пакет (раньше их было два: parents[1] в page_detection, parents[2] в
# finger_removal, и при переезде файлов их приходилось править по отдельности).
MODELS_DIR = Path(__file__).resolve().parents[2] / "finger_models"

DEFAULT_YOLO_WORLD = "yolov8x-worldv2.pt"
DEFAULT_SAM = "sam_b.pt"

# Классы open-vocabulary детектора, описывающие страницу/разворот книги.
PAGE_CLASSES = ["page", "book page", "open book", "sheet of paper", "paper", "document"]

# Классы фона/подложки — конкурируют с PAGE_CLASSES за боксы, чтобы боксы,
# распознанные как ткань/подложка, не попадали в маску страницы (см.
# ``page_detection.detect_page_mask``). CLIP путает светлую однотонную бумагу
# (форзац без текста) с тканью по текстуре волокна, независимо от того, что
# написано в промпте про цвет/яркость — поэтому «тёмное/светлое» разделяем не
# промптом, а напрямую по пикселям (см. FABRIC_MAX_MEAN_BRIGHTNESS там же).
FABRIC_CLASSES = ["fabric", "cloth", "fabric backdrop", "tablecloth"]

# Классы open-vocabulary детектора для руки/пальца (см. finger_removal.masking).
HAND_CLASSES = ["hand", "finger", "thumb", "fingertip", "human hand", "fingernail", "nail"]

# Веса LaMa: тот же torchscript-вес, что использует simple-lama-inpainting.
LAMA_WEIGHTS = MODELS_DIR / "big-lama.pt"
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"

# Веса DocShadow по вариантам (кладутся руками в finger_models/docshadow/).
DOCSHADOW_DIR = MODELS_DIR / "docshadow"
DOCSHADOW_WEIGHTS = {"sd7k": "SD7K.pth", "kligler": "Kligler.pth", "jung": "Jung.pth"}

# Сторона уменьшенной копии кадра для Surya layout.
LAYOUT_WORK_SIDE = 2048


def resolve_model_path(name: str) -> str:
    """Путь к весам в ``finger_models/`` (ultralytics докачает ассет по имени сам)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODELS_DIR / name)


def _ensure_lama_weights() -> None:
    """Качает big-lama.pt в ``finger_models/``, если его ещё нет."""
    if LAMA_WEIGHTS.exists():
        return
    import urllib.request

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Качаю веса LaMa: %s", LAMA_URL)
    urllib.request.urlretrieve(LAMA_URL, str(LAMA_WEIGHTS))


def _empty_detections() -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Пустой результат детектора: боксы (0, 4), уверенности (0,), классы (0,)."""
    return (np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int))


class GpuModels:
    """Все нейромодели пайплайна вырезки сканов и операции над ними.

    Создаётся один раз на процесс; конструктор грузит модели, дальше объект
    передаётся по пайплайну вместо строки ``device``. Можно использовать как
    контекстный менеджер — на выходе позовётся :meth:`close`.

    Аргументы:
        device: ``"cuda"`` / ``"cpu"``; ``None`` — cuda, если доступна.
        with_layout: грузить Surya LayoutPredictor (нужен только при
            ``--protect-text-layout``; это отдельная foundation-модель, грузится
            долго и занимает заметную часть VRAM).
        shadow_variant: вариант весов DocShadow (``sd7k`` / ``kligler`` / ``jung``)
            либо ``None`` — не грузить (нужен только при ``--shadow-method=docshadow-*``).
        yolo_weights, sam_weights: имена файлов весов в ``finger_models/``.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        with_layout: bool = False,
        shadow_variant: Optional[str] = None,
        yolo_weights: str = DEFAULT_YOLO_WORLD,
        sam_weights: str = DEFAULT_SAM,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Загружаю модели на %s: YOLO-World×2, SAM, LaMa%s%s",
            self._device,
            ", Surya layout" if with_layout else "",
            f", DocShadow ({shadow_variant})" if shadow_variant else "",
        )

        from ultralytics import SAM, YOLOWorld

        # ДВА инстанса одних и тех же весов YOLO-World с разными наборами классов.
        # ``set_classes`` кодирует список классов текстовым энкодером CLIP и
        # запоминает эмбеддинги в модели, поэтому один инстанс потребовал бы
        # переключения классов перед каждым из двух прогонов НА КАЖДОМ кадре.
        # Веса детектора невелики (~150 МБ), лишняя копия дешевле переключения.
        self._yolo_page = YOLOWorld(resolve_model_path(yolo_weights))
        self._yolo_page.set_classes(PAGE_CLASSES + FABRIC_CLASSES)
        self._yolo_hand = YOLOWorld(resolve_model_path(yolo_weights))
        self._yolo_hand.set_classes(HAND_CLASSES)

        # SAM общий: страницам и пальцам нужен один и тот же сегментатор по боксам.
        self._sam = SAM(resolve_model_path(sam_weights))

        _ensure_lama_weights()
        self._lama = torch.jit.load(str(LAMA_WEIGHTS), map_location=self._device)
        self._lama.eval().to(self._device)

        self._layout = self._load_layout() if with_layout else None
        self._shadow_variant = shadow_variant
        self._docshadow = self._load_docshadow(shadow_variant) if shadow_variant else None

    # --------------------------------------------------------
    # Свойства и жизненный цикл
    # --------------------------------------------------------

    @property
    def device(self) -> str:
        """Устройство, на котором реально живут модели (``"cuda"`` / ``"cpu"``)."""
        return self._device

    def close(self) -> None:
        """Отпускает модели и чистит кэш аллокатора CUDA."""
        self._yolo_page = self._yolo_hand = self._sam = None
        self._lama = self._layout = self._docshadow = None
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    def __enter__(self) -> "GpuModels":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --------------------------------------------------------
    # Детекция: YOLO-World
    # --------------------------------------------------------

    def detect_page_boxes(self, bgr: np.ndarray, conf: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Боксы-кандидаты области страницы/разворота (YOLO-World, page-классы).

        Аргументы:
            bgr: кадр BGR uint8 (H, W, 3) — обычно копия, уменьшенная до ``WORK_SIDE``
                (уменьшает вызывающий; метод размеров не меняет).
            conf: порог уверенности детектора.

        Возвращает ``(boxes, confs, is_fabric)``:
            boxes — float32 (N, 4) xyxy в координатах ПОДАННОГО кадра;
            confs — float32 (N,) уверенности;
            is_fabric — bool (N,), True для боксов с классом из ``FABRIC_CLASSES``
                (ткань/подложка), т.е. кандидатов на отбрасывание.
        Если детекций нет — массивы нулевой длины, не ``None``.

        Никакой фильтрации здесь нет: пороги площади, ярусы уверенности для
        near-full-frame боксов и подавление вложенных живут в ``page_detection``.
        """
        det = self._yolo_page.predict(bgr, conf=conf, device=self._device, verbose=False)
        boxes, confs, cls = self._unpack_detections(det)
        return boxes, confs, cls >= len(PAGE_CLASSES)

    def detect_hand_boxes(self, bgr: np.ndarray, conf: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Боксы-кандидаты руки/пальца (YOLO-World, ``HAND_CLASSES``).

        Аргументы:
            bgr: кадр BGR uint8 (H, W, 3) — здесь полного разрешения.
            conf: порог уверенности детектора.

        Возвращает ``(boxes, confs, cls)``:
            boxes — float32 (N, 4) xyxy; confs — float32 (N,);
            cls — int (N,), индекс класса в ``HAND_CLASSES``. Индекс нужен
            ``_select_finger_boxes``, чтобы отличать «части» (fingertip/fingernail/nail)
            от руки целиком. Пусто → массивы нулевой длины.

        Отбраковка по площади бокса, вложенности и площади масок — снаружи,
        в ``finger_removal.masking``.
        """
        det = self._yolo_hand.predict(bgr, conf=conf, device=self._device, verbose=False)
        return self._unpack_detections(det)

    @staticmethod
    def _unpack_detections(det) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Результат ultralytics → (boxes, confs, cls) в numpy; пусто → массивы длины 0."""
        if not det or det[0].boxes is None or len(det[0].boxes) == 0:
            return _empty_detections()
        b = det[0].boxes
        return (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))

    # --------------------------------------------------------
    # Сегментация: SAM
    # --------------------------------------------------------

    def segment_boxes(self, bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Силуэты объектов внутри боксов (SAM по bbox-подсказкам).

        Аргументы:
            bgr: кадр BGR uint8 (H, W, 3);
            boxes: (N, 4) xyxy в координатах ЭТОГО кадра.

        Возвращает bool-массив (N, H, W) — по одной бинарной маске на бокс, уже
        приведённой к размеру кадра (SAM отдаёт маски в своём внутреннем
        разрешении, ресайз ``INTER_NEAREST`` делается здесь). При пустом ``boxes``
        или пустом ответе SAM — массив формы (0, H, W).

        Отбор масок по площади (у страниц и пальцев пороги разные) и объединение
        их в одну маску — задача вызывающего.
        """
        h, w = bgr.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, h, w), dtype=bool)

        seg = self._sam.predict(bgr, bboxes=boxes, device=self._device, verbose=False)
        if not seg or seg[0].masks is None:
            return np.zeros((0, h, w), dtype=bool)

        out = []
        for m in seg[0].masks.data.cpu().numpy():
            m_bin = m > 0.5
            if m_bin.shape != (h, w):
                m_bin = cv2.resize(m_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            out.append(m_bin)
        return np.asarray(out, dtype=bool) if out else np.zeros((0, h, w), dtype=bool)

    # --------------------------------------------------------
    # Закраска: LaMa
    # --------------------------------------------------------

    def inpaint(
        self, rgb: np.ndarray, mask: np.ndarray, padding: int = 64, feather: int = 9, roi_scale: float = 1.5
    ) -> np.ndarray:
        """Закрашивает область под маской (LaMa) — «зарисовать пальцы».

        Аргументы:
            rgb: исходная картинка RGB uint8 (H, W, 3);
            mask: что закрасить, uint8 0/255 (H, W); может быть многокомпонентной
                (например, два пальца с разных краёв кадра);
            padding: контекстное поле вокруг компоненты маски, пикс.;
            feather: ширина растушёвки шва при вклеивании, пикс.;
            roi_scale: во сколько раз растянуть ROI от центра после padding.

        Возвращает закрашенную картинку RGB uint8 (H, W, 3) того же размера.
        Пиксели вне маски и вне полосы растушёвки совпадают с исходными.
        Пустая маска → входной массив возвращается как есть, сеть не запускается.

        Почему покомпонентно и по ROI: палец входит с края книги, где в кадре
        доминирует ЧЁРНЫЙ фон. По всему снимку 5696×4272 LaMa «затягивает» дыру
        этим доминирующим чёрным. В тесном ROI сеть видит локальный контекст —
        кромку переплёта, поле страницы — и достраивает именно его.
        """
        from ocr_utils.scan_cropping.finger_removal.inpaint_roi import blend_roi, mask_components, roi_bounds

        comps = mask_components(mask)
        if not comps:
            return rgb

        result = rgb.copy()
        for comp in comps:
            bounds = roi_bounds(comp, padding, roi_scale, rgb.shape[:2])
            if bounds is None:
                continue
            x1, y1, x2, y2 = bounds
            roi = result[y1:y2, x1:x2]
            mroi = comp[y1:y2, x1:x2]
            result[y1:y2, x1:x2] = blend_roi(roi, self._lama_fill_roi(roi, mroi), mroi, feather)
        return result

    def _lama_fill_roi(self, roi: np.ndarray, mroi: np.ndarray) -> np.ndarray:
        """Прогон LaMa по одному ROI; возвращает заполненный ROI (RGB uint8, тот же размер)."""
        img = (roi.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW
        msk = ((mroi > 0).astype(np.float32))[None, ...]  # 1HW

        img_p, (oh, ow) = _pad_to_modulo(img)
        msk_p, _ = _pad_to_modulo(msk)

        it = torch.from_numpy(img_p).unsqueeze(0).to(self._device)
        mt = torch.from_numpy(msk_p).unsqueeze(0).to(self._device)
        mt = (mt > 0).float()

        with torch.inference_mode():
            out = self._lama(it, mt)
        res = out[0].permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(res * 255.0, 0, 255).astype(np.uint8)[:oh, :ow]

    # --------------------------------------------------------
    # Разметка страницы: Surya layout
    # --------------------------------------------------------

    def _load_layout(self):
        """Загрузка Surya LayoutPredictor."""
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.settings import settings

        predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
        predictor.disable_tqdm = True
        return predictor

    def layout_blocks(self, rgb: np.ndarray) -> list:
        """Блоки разметки страницы (текст, заголовки, картинки, таблицы...).

        Аргументы:
            rgb: кадр RGB uint8 (H, W, 3) полного разрешения.

        Возвращает список блоков Surya в координатах ПОДАННОГО кадра; у каждого
        есть ``polygon`` (4 точки), ``label`` и ``confidence``. Отсев мусорных
        блоков здесь не делается — он в ``finger_removal.text_protection``.

        Кадр подаётся как есть; уменьшение до ``LAYOUT_WORK_SIDE`` и обратный
        пересчёт координат — задача вызывающего: фильтровать блоки удобнее в том
        же разрешении, в котором их нашла сеть.
        """
        if self._layout is None:
            raise RuntimeError(
                "GpuModels создан без Surya layout: пересоздайте объект с with_layout=True "
                "(в CLI это флаг --protect-text-layout)"
            )
        from PIL import Image as PILImage

        return self._layout([PILImage.fromarray(rgb)])[0].bboxes

    # --------------------------------------------------------
    # Коррекция тени: DocShadow
    # --------------------------------------------------------

    def _load_docshadow(self, variant: str):
        """Загрузка DocShadow с весами варианта (sd7k / kligler / jung)."""
        from ocr_utils.scan_cropping.finger_removal.docshadow_net import Model

        if variant not in DOCSHADOW_WEIGHTS:
            raise ValueError(f"неизвестный вариант DocShadow: {variant}")
        model = Model().to(self._device).eval()
        ckpt = torch.load(DOCSHADOW_DIR / DOCSHADOW_WEIGHTS[variant], map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {(k[7:] if k.startswith("module") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        return model

    def remove_shadow(self, bgr: np.ndarray, max_side: int = 2048) -> np.ndarray:
        """Выравнивает освещённость документа нейросетью DocShadow.

        Аргументы:
            bgr: кадр BGR uint8 (H, W, 3);
            max_side: длинная сторона копии, на которой считается инференс.

        Возвращает BGR uint8 (H, W, 3) того же размера. Сеть частотно-осведомлённая
        (детали восстанавливаются пирамидой), поэтому считать её на уменьшенной
        копии и растягивать результат допустимо; стороны копии приводятся к
        кратности 4 — требование пирамиды.

        Вариант весов фиксируется при создании объекта (``shadow_variant``).
        Классические методы (``classic``/``retinex``) GPU не требуют и живут в
        ``finger_removal.finger_shadow``.
        """
        if self._docshadow is None:
            raise RuntimeError(
                "GpuModels создан без DocShadow: пересоздайте объект с shadow_variant="
                "'sd7k'|'kligler'|'jung' (в CLI это --shadow-method=docshadow-*)"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = min(1.0, max_side / max(h, w))
        small = cv2.resize(rgb, (int(w * s) // 4 * 4, int(h * s) // 4 * 4), interpolation=cv2.INTER_AREA)
        t = torch.tensor(small, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self._device) / 255.0
        with torch.no_grad():
            out = self._docshadow(t).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        out = cv2.resize((out * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _pad_to_modulo(arr: np.ndarray, mod: int = 8) -> "tuple[np.ndarray, tuple[int, int]]":
    """Симметрично дополняет CHW-массив до кратности ``mod``; возвращает (arr, (h, w))."""
    _, h, w = arr.shape
    ph = (mod - h % mod) % mod
    pw = (mod - w % mod) % mod
    padded = np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="symmetric")
    return padded, (h, w)
