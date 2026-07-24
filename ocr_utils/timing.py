"""Утилита для замеров времени ключевых операций пайплайна и вывода их в лог.

Общий модуль, чтобы и ``detect_and_crop``, и подпакеты ``finger_removal``
писали тайминги единообразно. Замер идёт по ``timeit.default_timer`` —
высокоточному монотонному таймеру. Сообщения печатаются на уровне ``INFO``,
поэтому по умолчанию (уровень логирования ``WARNING``) молчат; чтобы увидеть
тайминги, нужно поднять уровень до ``INFO`` (в ``detect_and_crop`` — флагом
``--log-level=INFO``).

Вложенность показывается не отступом, а полным путём операции: активные
``log_timing`` образуют стек, и при выходе печатается цепочка от самой внешней
операции к текущей. Пример (вложенность видна из самих строк)::

    ... INFO      22 мс: remove_fingers -> build_finger_mask -> neural_hand_mask -> yolo_predict (IMG_0084.tif)
    ... INFO     746 мс: remove_fingers -> build_finger_mask -> neural_hand_mask (IMG_0084.tif)
    ... INFO    1174 мс: remove_fingers -> build_finger_mask (IMG_0084.tif)
    ... INFO     915 мс: page_mask (IMG_0084.tif)
"""

import logging
import timeit

logger = logging.getLogger(__name__)

# Стек активных (вошедших, но не вышедших) таймеров — для построения пути
# вложенности. Общий на процесс: пайплайн однопоточный, и все модули используют
# один и тот же класс log_timing из этого модуля, поэтому вложенность считается
# сквозной (напр. таймер в masking виден вложенным в таймер из detect_and_crop).
_STACK: "list[str]" = []


class log_timing:
    """Контекстный менеджер: меряет длительность блока и пишет её в лог (INFO).

    ``label`` — имя операции (без отступов; вложенность выводится автоматически по
    стеку активных таймеров). ``name`` — имя обрабатываемого файла (необязательно).
    ``log`` — логгер, в который писать (по умолчанию — общий логгер этого модуля);
    все логгеры пакета всё равно всплывают к корневому обработчику, так что уровень,
    выставленный в ``main``, действует и здесь.

    Строка лога: ``<внешняя операция> -> ... -> <текущая операция> -> N мс (файл)``.
    """

    def __init__(self, label: str, name: str = "", log: "logging.Logger | None" = None):
        self.label = label
        self.name = name
        self.log = log or logger

    def __enter__(self) -> "log_timing":
        _STACK.append(self.label)
        self._t0 = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        dt_ms = (timeit.default_timer() - self._t0) * 1000.0
        path = " -> ".join(_STACK)
        _STACK.pop()
        suffix = f" ({self.name})" if self.name else ""
        # Время печатаем ПЕРВЫМ (и колонкой фиксированной ширины), чтобы его было
        # легко выхватывать глазами; следом — цепочка вложенности операции.
        self.log.info("%7.0f мс: %s%s", dt_ms, path, suffix)
        return False
