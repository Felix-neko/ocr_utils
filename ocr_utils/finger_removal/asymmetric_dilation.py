"""Асимметричная (разная по X и Y) дилатация зон пальца.

ЗАЧЕМ ЭТО НУЖНО
---------------
Палец, придерживающий страницу, отбрасывает тень, и эта тень выходит за силуэт
пальца НЕ равномерно во все стороны, а преимущественно ВДОЛЬ той стороны книги,
к которой палец прилегает:

  - палец слева/справа (прижимает боковой край) — тень тянется вверх-вниз,
    т.е. по Y; значит расширять маску надо сильнее по вертикали;
  - палец сверху/снизу — тень тянется влево-вправо, т.е. по X;
  - палец в углу — тень размазана по обеим осям примерно поровну.

Равномерная (круглая) дилатация, чтобы захватить тень, вынуждена раздуваться во
все стороны одинаково — и залезает вглубь страницы на текст. Асимметричная
дилатация даёт тот же охват тени, но «экономнее»: растёт там, где тень реально
есть, и почти не растёт внутрь страницы.

КАК СЧИТАЮТСЯ КОЭФФИЦИЕНТЫ
--------------------------
Для каждой зоны пальца (связной компоненты маски) берём её bbox и считаем
расстояния до четырёх сторон кадра. Затем нормируем их к полукадру, чтобы
величины были безразмерными и сравнимыми между собой:

    d_lr = min(до левого края, до правого края) / (W / 2)   # «насколько далеко от боковых сторон»
    d_tb = min(до верхнего края, до нижнего края) / (H / 2)  # «насколько далеко от верх/низ сторон»

Обе величины лежат в [0, 1]: 0 — зона вплотную прилегает к стороне, 1 — она в
середине кадра по этой оси.

Дальше — ключевой момент. «Боковой» палец узнаётся не по тому, что d_lr мало
само по себе, а по тому, что d_lr МАЛО ОТНОСИТЕЛЬНО d_tb. Поэтому веса берём
перекрёстно (в числителе — расстояние по ДРУГОЙ оси):

    w_lr = (d_tb + eps) / (d_lr + d_tb + 2*eps)   # мера «это боковой (левый/правый) палец»
    w_tb = (d_lr + eps) / (d_lr + d_tb + 2*eps)   # мера «это верхний/нижний палец»

По построению w_lr + w_tb = 1. Проверим предельные случаи:

  - палец у левого края, по вертикали в середине: d_lr≈0, d_tb≈0.5
    → w_lr≈1, w_tb≈0;
  - палец у верхнего края: d_lr≈0.5, d_tb≈0
    → w_lr≈0, w_tb≈1;
  - палец в углу: d_lr≈0, d_tb≈0
    → eps спасает от деления 0/0 и даёт ровно w_lr = w_tb = 0.5.

Наконец, коэффициенты дилатации (боковому пальцу растим Y, верхнему/нижнему — X):

    x_ratio = 1 + MAX_ASYMMETRIC_DILATION_RATIO * w_tb
    y_ratio = 1 + MAX_ASYMMETRIC_DILATION_RATIO * w_lr

что даёт ровно требуемое поведение:

  - боковой палец:      x_ratio → 1,                 y_ratio → 1 + MAX;
  - верхний/нижний:     x_ratio → 1 + MAX,           y_ratio → 1;
  - угловой:            x_ratio → 1 + MAX/2,         y_ratio → 1 + MAX/2.

``eps`` (``corner_softness``) задаёт, насколько «широким» получается угол: чем он
больше, тем раньше зона начинает считаться угловой и тем плавнее переход между
режимами. При eps=0 переход был бы разрывным вблизи угла (0/0).
"""

from typing import Optional

import cv2
import numpy as np

# Максимальная ДОБАВКА к коэффициенту дилатации по «выгодной» оси.
# Итоговый коэффициент по этой оси = 1 + MAX_ASYMMETRIC_DILATION_RATIO.
MAX_ASYMMETRIC_DILATION_RATIO = 2.0

# Глобальный выключатель асимметрии. Если False — дилатация круговая (как раньше),
# т.е. x_ratio = y_ratio = 1 для всех зон.
ASYMMETRIC_DILATION_ENABLED = True

# Смягчение угла: насколько плавно зона переходит в режим «угловая» (см. docstring).
DEFAULT_CORNER_SOFTNESS = 0.05


class FingerZoneDilation:
    """Считает меру близости зоны пальца к сторонам кадра и коэффициенты дилатации.

    Экземпляр привязан к размеру кадра (``image_shape``), поэтому его удобно
    создать один раз на кадр и переиспользовать для всех зон пальца.

    Пример::

        helper = FingerZoneDilation(mask.shape)
        x_ratio, y_ratio = helper.ratios((x_min, y_min, x_max, y_max))
        kx, ky = helper.kernel_radii(bbox, base_dilate_px=60)
    """

    def __init__(
        self,
        image_shape,
        max_ratio: float = MAX_ASYMMETRIC_DILATION_RATIO,
        enabled: bool = ASYMMETRIC_DILATION_ENABLED,
        corner_softness: float = DEFAULT_CORNER_SOFTNESS,
    ) -> None:
        self.height, self.width = int(image_shape[0]), int(image_shape[1])
        self.max_ratio = float(max_ratio)
        self.enabled = bool(enabled)
        self.corner_softness = float(corner_softness)

    # ------------------------------------------------------------------
    # Близость к сторонам
    # ------------------------------------------------------------------

    def side_distances_px(self, bbox) -> "dict[str, float]":
        """Расстояния (в пикселях) от bbox зоны до каждой стороны кадра.

        ``bbox`` — (x_min, y_min, x_max, y_max). Значения не могут быть < 0.
        """
        x_min, y_min, x_max, y_max = (float(v) for v in bbox)
        return {
            "left": max(0.0, x_min),
            "right": max(0.0, self.width - x_max),
            "top": max(0.0, y_min),
            "bottom": max(0.0, self.height - y_max),
        }

    def nearest_side(self, bbox) -> str:
        """К какой стороне кадра зона ближе всего: 'left' / 'right' / 'top' / 'bottom'."""
        d = self.side_distances_px(bbox)
        return min(d, key=d.__getitem__)

    def normalized_axis_distances(self, bbox) -> "tuple[float, float]":
        """Нормированные к полукадру расстояния (d_lr, d_tb), каждое в [0, 1].

        d_lr — до ближайшей БОКОВОЙ стороны (левой/правой), нормировано на W/2;
        d_tb — до ближайшей ГОРИЗОНТАЛЬНОЙ стороны (верх/низ), нормировано на H/2.
        """
        d = self.side_distances_px(bbox)
        half_w = max(1.0, self.width / 2.0)
        half_h = max(1.0, self.height / 2.0)
        d_lr = min(d["left"], d["right"]) / half_w
        d_tb = min(d["top"], d["bottom"]) / half_h
        return float(np.clip(d_lr, 0.0, 1.0)), float(np.clip(d_tb, 0.0, 1.0))

    def side_weights(self, bbox) -> "tuple[float, float]":
        """Веса (w_lr, w_tb), в сумме 1: насколько зона «боковая» и насколько «верх/низ».

        Веса перекрёстные: близость к боковой стороне определяется тем, что
        расстояние по X мало ОТНОСИТЕЛЬНО расстояния по Y (см. docstring модуля).
        """
        d_lr, d_tb = self.normalized_axis_distances(bbox)
        eps = self.corner_softness
        denom = d_lr + d_tb + 2.0 * eps
        w_lr = (d_tb + eps) / denom
        w_tb = (d_lr + eps) / denom
        return float(w_lr), float(w_tb)

    # ------------------------------------------------------------------
    # Коэффициенты и ядро дилатации
    # ------------------------------------------------------------------

    def ratios(self, bbox) -> "tuple[float, float]":
        """Коэффициенты (x_ratio, y_ratio), на которые домножается ``finger_dilate_px``.

        Если асимметрия выключена — обе единицы (круговая дилатация, прежнее поведение).
        """
        if not self.enabled:
            return 1.0, 1.0
        w_lr, w_tb = self.side_weights(bbox)
        x_ratio = 1.0 + self.max_ratio * w_tb  # верхний/нижний палец растим по X
        y_ratio = 1.0 + self.max_ratio * w_lr  # боковой палец растим по Y
        return x_ratio, y_ratio

    def kernel_radii(self, bbox, base_dilate_px: int) -> "tuple[int, int]":
        """Радиусы эллиптического ядра дилатации (kx, ky) в пикселях, не меньше 1."""
        x_ratio, y_ratio = self.ratios(bbox)
        kx = max(1, int(round(base_dilate_px * x_ratio)))
        ky = max(1, int(round(base_dilate_px * y_ratio)))
        return kx, ky


def dilate_finger_zones(
    mask: np.ndarray,
    dilate_px: int,
    enabled: bool = ASYMMETRIC_DILATION_ENABLED,
    max_ratio: float = MAX_ASYMMETRIC_DILATION_RATIO,
    corner_softness: float = DEFAULT_CORNER_SOFTNESS,
) -> np.ndarray:
    """Дилатирует КАЖДУЮ зону пальца (связную компоненту) со своими коэффициентами.

    Компоненты обрабатываются раздельно, потому что пальцы на одном кадре бывают
    с разных сторон (слева и справа), и каждому нужен свой перекос дилатации.
    Результат — объединение раздутых компонент.
    """
    if dilate_px <= 0 or int(np.count_nonzero(mask)) == 0:
        return mask
    helper = FingerZoneDilation(mask.shape, max_ratio=max_ratio, enabled=enabled, corner_softness=corner_softness)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        kx, ky = helper.kernel_radii((x, y, x + w, y + h), dilate_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kx + 1, 2 * ky + 1))
        comp = (labels == i).astype(np.uint8) * 255
        out = cv2.bitwise_or(out, cv2.dilate(comp, kernel, iterations=1))
    return out
