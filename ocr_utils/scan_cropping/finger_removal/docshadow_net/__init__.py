"""Вендоренная (MIT, © 2023 Nick Chen) архитектура DocShadow-SD7K (FSENet) для
удаления теней с документов. Источник: https://github.com/CXH-Research/DocShadow-SD7K

Используется как один из вариантов коррекции теневой зоны у пальца (см.
``ocr_utils.scan_cropping.finger_removal.finger_shadow``). Веса (SD7K/Kligler/Jung) лежат в
``finger_models/docshadow/``.
"""

from .model import Model

__all__ = ["Model"]
