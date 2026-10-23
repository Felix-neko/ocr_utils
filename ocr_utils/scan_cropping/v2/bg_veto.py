"""Вето фона: пиксель фона ``flood`` отдаётся книге, где SAM 3 видит страницу, а SAM 2 по точкам коврика — не фон.

Порог и водораздел (``background.bg_flood``) относят к фону всё тёмное и бесцветное, что связано с ковриком. Рукописный
библиотечный номер, тёмный логотип или линейка, касающиеся края листа, уводят фон на страницу, и книга теряет
поле до первого сильного перепада (hp91 IMG_1091: надпись «30 84» над заголовком, РИСК 1991/03 IMG_0347: плашка
«INFORM MATERIAL»). SAM 3 эту бумагу видит, SAM 2 по точкам на коврике фоном её не считает. По отдельности модели для
вето не годятся: сырая страница SAM 3 выходит каймой на коврик, а SAM 2 считает книгой тёмную крышку переплёта —
поэтому фон отменяется только там, где согласны обе.

Стенд — ``research/bound_volumes`` (способы фона) и ``runs/bg_bays`` (2026-10-21): по кэшу ``pack6_v6`` и hp91 фон
заходил на страницу SAM 3 в 35 кадрах из 2565, бумагу терял в 9; с вето на 274 кадрах (35 + три проверочные папки)
бумага добавилась в 24 кадрах (+6.4 тыс. мм², только краска на бумаге и поле до края листа), убрано 11 мм², число
страниц не изменилось.
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.background import book_seeds, clean_background, mat_seeds
from ocr_utils.scan_cropping.v2.book import book_close_open, content_hulls
from ocr_utils.scan_cropping.v2.ink_fill import fill_bold_ink

# Модель SAM 2.1 (transformers, кэш Hugging Face).
SAM2_MODEL = "facebook/sam2.1-hiera-large"
# Длинная сторона входа SAM 2, px.
SAM2_SIDE = 1024
# Подсказки: положительные точки на уверенном коврике у рамки, отрицательные — на своей печати; зерно выбора точек.
SAM2_MAT_POINTS = 12
SAM2_PRINT_POINTS = 6
SAM2_SEED = 0
# Штраф кандидата за накрытую печать при выборе маски из трёх: доля коврика минус SAM2_PRINT_PENALTY × доля печати.
SAM2_PRINT_PENALTY = 5.0
# Версия кэша ``sam2_bg.npz``.
SAM2_CACHE_VERSION = 1


class Sam2Background:
    """SAM 2.1 по точкам коврика: маска фона кадра сегментации (модель грузится один раз на прогон)."""

    def __init__(self, device: str = "cuda") -> None:
        """Грузит модель и процессор.

        Args:
            device: Устройство torch.
        """
        from transformers import Sam2Model, Sam2Processor

        self.device = device
        self.model = Sam2Model.from_pretrained(SAM2_MODEL).to(device).eval()
        self.processor = Sam2Processor.from_pretrained(SAM2_MODEL)

    def masks(self, rgb: np.ndarray, points: list[list[float]], labels: list[int]) -> np.ndarray:
        """Три маски-кандидата SAM 2 по точкам.

        Args:
            rgb: Картинка RGB входа сети.
            points: Точки ``[[x, y], ...]`` в px картинки.
            labels: Метки точек: 1 — коврик, 0 — печать.

        Returns:
            Маски ``(3, H, W)`` bool размера ``rgb``.
        """
        import torch
        from PIL import Image

        inputs = self.processor(
            images=Image.fromarray(rgb), input_points=[[points]], input_labels=[[labels]], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy()
        return masks.astype(bool)

    def masks_scored(
        self, rgb: np.ndarray, points: list[list[float]], labels: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Три маски-кандидата SAM 2 по точкам и их оценки IoU (для уточнения края, ``edge_refine``).

        Args:
            rgb: Картинка RGB входа сети.
            points: Точки ``[[x, y], ...]`` в px картинки.
            labels: Метки точек: 1 — объект, 0 — не объект.

        Returns:
            ``(маски (3, H, W) bool, оценки (3,))``.
        """
        import torch
        from PIL import Image

        inputs = self.processor(
            images=Image.fromarray(rgb), input_points=[[points]], input_labels=[[labels]], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy()
        return masks.astype(bool), out.iou_scores[0, 0].float().cpu().numpy()

    def __call__(self, bgr: np.ndarray, content: np.ndarray) -> np.ndarray:
        """Фон кадра сегментации по SAM 2.

        Точки берутся из тех же затравок, что у ``bg_flood``: коврик — тёмные бесцветные пиксели полосы у рамки
        (``mat_seeds``), печать — своя печать (``book_seeds``). Из трёх кандидатов — тот, что лучше всех накрывает
        коврик и меньше всех печать; маска поднимается до масштаба сегментации и чистится как у ``bg_flood``.

        Args:
            bgr: Кадр BGR масштаба сегментации.
            content: Своя печать (bool) того же размера.

        Returns:
            Маска фона (bool) размера ``content``. Нет затравок коврика — пустая маска (вето ничего не отменяет).
        """
        h, w = content.shape
        scale = SAM2_SIDE / max(h, w)
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        size = (small.shape[1], small.shape[0])
        # Затравки — на масштабе сегментации, затем ближайшим соседом на вход сети.
        mat = cv2.resize(mat_seeds(bgr).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        books = cv2.resize(book_seeds(content).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        if not mat.any():
            return np.zeros_like(content)
        # Случайные точки из затравок с постоянным зерном: прогон воспроизводим.
        rng = np.random.default_rng(SAM2_SEED)
        pos = np.argwhere(mat)
        neg = np.argwhere(books)
        pos = pos[rng.choice(len(pos), min(SAM2_MAT_POINTS, len(pos)), replace=False)]
        neg = neg[rng.choice(len(neg), min(SAM2_PRINT_POINTS, len(neg)), replace=False)] if len(neg) else neg
        points = [[float(x), float(y)] for y, x in pos] + [[float(x), float(y)] for y, x in neg]
        labels = [1] * len(pos) + [0] * len(neg)
        masks = self.masks(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), points, labels)
        # Выбор кандидата: доля накрытого коврика минус штраф за накрытую печать.
        cover = [
            (m & mat).sum() / max(mat.sum(), 1) - SAM2_PRINT_PENALTY * (m & books).sum() / max(books.sum(), 1)
            for m in masks
        ]
        best = masks[int(np.argmax(cover))]
        bg = cv2.resize(best.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR) > 0
        return clean_background(bg)


def veto_background(bg: np.ndarray, sam3_page: np.ndarray, sam2_bg: np.ndarray) -> np.ndarray:
    """Фон без пикселей, где SAM 3 видит страницу, а SAM 2 — не фон.

    Фон только уменьшается: вето не добавляет ни одного пикселя.

    Args:
        bg: Фон ``bg_flood`` (bool).
        sam3_page: Сырая маска страниц SAM 3, первый проход (``sam3.npz: raw``, bool).
        sam2_bg: Фон SAM 2 (:class:`Sam2Background`, bool).

    Returns:
        Маска фона (bool).
    """
    return bg & ~(sam3_page & ~sam2_bg)


def veto_book(img: np.ndarray, book: dict, sam3_page: np.ndarray, sam2_bg: np.ndarray) -> dict:
    """Маски шага «фон и книга», пересобранные по фону-вето: книга, оболочки печати и крупная краска — как в v2.

    Второй проход SAM 3 (по кадру без краски) не пересчитывается: он идёт до вето и остаётся из кэша; пересчитывается
    только маска закрашенной краски — по ней принимаются куски добавки второго прохода (``ink_fill.touching_ink``).

    Args:
        img: Кадр BGR масштаба сегментации.
        book: Маски шага 2 (``content``, ``bg``, ``book``, ``hulls`` и, при втором проходе SAM 3, ``ink``).
        sam3_page: Сырая маска страниц SAM 3 (bool).
        sam2_bg: Фон SAM 2 (bool).

    Returns:
        Словарь масок с теми же ключами, что ``book``.
    """
    content = book["content"]
    bg = veto_background(book["bg"], sam3_page, sam2_bg)
    # Как в ``pipeline.cpu_book``: оболочки для книги — внутри «не фона», для дальнейших шагов — внутри книги.
    new_book = book_close_open(bg, content_hulls(content, ~bg))
    hulls = content_hulls(content, new_book)
    out = dict(content=content, bg=bg, book=new_book, hulls=hulls)
    if "ink" in book:
        _, out["ink"] = fill_bold_ink(img, new_book, bg, hulls)
    return out
