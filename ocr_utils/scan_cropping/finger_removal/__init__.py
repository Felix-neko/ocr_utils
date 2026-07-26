"""Удаление придерживающего страницу пальца: детекция маски → закраска.

Верхний уровень — ``removal.remove_fingers``; он собирает вместе построение маски
(``masking``), асимметричную дилатацию зоны под тень (``asymmetric_dilation``),
защиту контента от закраски по блокам Surya layout (``text_protection``),
геометрию ROI под инпейнтер (``inpaint_roi``) и коррекцию теневой зоны
(``finger_shadow``). Сами сети — в ``scan_cropping.gpu_models.GpuModels``.
"""
