"""Реестр алгоритмов оценки резкости.

``COMBO_NAME`` — не самостоятельная метрика, а сведение нескольких: балл файла считается
как средний ранг по алгоритмам из ``COMBO_MEMBERS``. Ранги, а не сами значения, потому что
шкалы у метрик несопоставимы (ширина края в пикселях, доля энергии, [0,1]-мера), а вот
порядок файлов сравним всегда.

По умолчанию используется НЕ combo, а одиночный ``DEFAULT_ALGORITHM``: на размеченной
папке 1979 года консенсус двух метрик оказался слабее лучшей из них поодиночке
(AUC 0.858 против 0.871). Combo оставлен как более осторожный режим для незнакомого
материала, где непонятно, какой метрике верить.
"""

from ocr_utils.defocus_detection.metrics import edge_width, hf_mid, laplacian, moire, reblur
from ocr_utils.defocus_detection.metrics.base import Algorithm

ALGORITHMS: dict[str, Algorithm] = {
    a.name: a for a in (edge_width.ALGORITHM, reblur.ALGORITHM, hf_mid.ALGORITHM, moire.ALGORITHM, laplacian.ALGORITHM)
}

COMBO_NAME = "combo"
# В консенсус входят только метрики, не зависящие ни от количества текста, ни от кегля,
# ни от экспозиции. hf_mid и laplacian исключены намеренно: оба принимают крупный кегль
# за расфокус (см. tests/.../test_known_metrics_are_fooled_by_font_size), moire — потому
# что на страницах без типографского растра он меряет неизвестно что.
COMBO_MEMBERS = ("edge_width", "reblur")

DEFAULT_ALGORITHM = "edge_width"

CHOICES = (*ALGORITHMS.keys(), COMBO_NAME)


def resolve(name: str) -> tuple[str, ...]:
    """Разворачивает имя алгоритма в список метрик, которые надо посчитать.

    Args:
        name: Имя из ``CHOICES``.

    Returns:
        Кортеж имён отдельных алгоритмов.
    """
    return COMBO_MEMBERS if name == COMBO_NAME else (name,)
