"""Показать рисунок примера: окно matplotlib, а если его нет — системный просмотрщик.

Общее для примеров подпакетов (`tables/example.py`, `text_blocks/example.py`): у uv-сборки Python
бывает несовместимый с matplotlib Tk («Failed to import tkagg backend… uv python upgrade
--reinstall»), Qt в окружении может не быть, и backend молча откатывается на Agg — окно не
открывается, а пример делает вид, что показал. Поэтому окно сначала проверяется, а при его
отсутствии рисунок сохраняется и открывается тем, что есть в системе.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def is_interactive_backend() -> bool:
    """Умеет ли выбранный backend matplotlib открывать окно.

    Returns:
        ``False`` для Agg и прочих файловых backend'ов (pdf, svg…), ``True`` для TkAgg, QtAgg и т. п.
    """
    import matplotlib
    from matplotlib import backends

    return matplotlib.get_backend().lower() not in {
        name.lower() for name in backends.backend_registry.list_builtin(backends.BackendFilter.NON_INTERACTIVE)
    }


def open_in_viewer(path: Path) -> None:
    """Открыть картинку системным просмотрщиком (``xdg-open``), не дожидаясь его закрытия.

    Args:
        path: Сохранённый PNG.
    """
    viewer = shutil.which("xdg-open")
    if viewer is None:
        print("окно matplotlib недоступно и xdg-open не найден — откройте файл сами")
        return
    # Отдельная сессия и заглушенный вывод: просмотрщик живёт своей жизнью и не пишет в консоль примера.
    subprocess.Popen([viewer, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


__all__ = ["is_interactive_backend", "open_in_viewer"]
