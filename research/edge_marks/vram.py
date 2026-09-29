"""Замер ресурсов судьи: запуск в подпроцессе с опросом видеопамяти (``nvidia-smi``, по дереву процессов) и ОЗУ раз в ``PERIOD_S`` секунд."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

import psutil

# Период опроса, с.
PERIOD_S = 0.2


@dataclass(frozen=True)
class Usage:
    """Пики ресурсов процесса и его потомков.

    Attributes:
        seconds: Время работы, с.
        vram_mb: Пик видеопамяти, МБ (сумма по процессам дерева).
        rss_mb: Пик резидентной памяти, МБ (сумма по дереву).
        returncode: Код возврата.
    """

    seconds: float
    vram_mb: float
    rss_mb: float
    returncode: int


def gpu_usage_by_pid() -> dict[int, float]:
    """Видеопамять процессов по ``nvidia-smi --query-compute-apps``, МБ; пусто — нет GPU или ошибка."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    usage: dict[int, float] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            try:
                usage[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
    return usage


def tree_pids(root: int) -> set[int]:
    """PID процесса и всех его потомков (живых на момент вызова)."""
    try:
        process = psutil.Process(root)
        return {root, *(child.pid for child in process.children(recursive=True))}
    except psutil.NoSuchProcess:
        return set()


def _poll(root: int, peaks: dict, stop: threading.Event) -> None:
    """Опрос пиков видеопамяти и ОЗУ дерева процессов ``root`` до ``stop`` (пишет в ``peaks``)."""
    while not stop.is_set():
        pids = tree_pids(root)
        gpu = gpu_usage_by_pid()
        peaks["vram"] = max(peaks["vram"], sum(gpu.get(pid, 0.0) for pid in pids))
        rss = 0.0
        for pid in pids:
            try:
                rss += psutil.Process(pid).memory_info().rss / 2**20
            except psutil.NoSuchProcess:
                continue
        peaks["rss"] = max(peaks["rss"], rss)
        stop.wait(PERIOD_S)


def run_measured(cmd: list[str], env: dict | None = None, cwd: str | None = None, log=None) -> Usage:
    """Запустить команду и замерить пики видеопамяти и ОЗУ её дерева процессов.

    Args:
        cmd: Команда.
        env: Окружение (``None`` — текущее).
        cwd: Рабочая папка.
        log: Открытый файл для stdout и stderr (``None`` — в никуда).

    Returns:
        :class:`Usage`.
    """
    started = time.monotonic()
    process = subprocess.Popen(
        cmd, env=env, cwd=cwd, stdout=log or subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True
    )
    peaks = {"vram": 0.0, "rss": 0.0}
    stop = threading.Event()
    watcher = threading.Thread(target=_poll, args=(process.pid, peaks, stop), daemon=True)
    watcher.start()
    returncode = process.wait()
    stop.set()
    watcher.join()
    return Usage(time.monotonic() - started, peaks["vram"], peaks["rss"], returncode)


__all__ = ["Usage", "gpu_usage_by_pid", "run_measured", "tree_pids"]
