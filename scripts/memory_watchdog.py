"""Сторож памяти для тяжёлого запуска: лог процессов сессии по секундам и убийство всей сессии при превышении потолка.

Зачем. ``systemd-run --user -p MemoryMax=…`` на этой машине не ограничивает память (гибридная
схема cgroup: контроллер памяти v1 не отдан пользовательскому systemd — проверено: 1.5 ГБ при
потолке 1 ГБ выделились). Swap 2 ГБ, earlyoom/systemd-oomd выключены, поэтому процесс, съевший
всю память, не убивается, а вешает рабочий стол (2026-09-27: vLLM + компиляция ядер на 32
потока, зависание GUI на 8 минут до перезагрузки). Сторож следит за сессией запуска (``setsid``)
и убивает её целиком раньше, чем кончится память системы.

    setsid bash job.sh > job.log 2>&1 < /dev/null & JOB=$!
    python scripts/memory_watchdog.py --sid "$JOB" --limit-gb 64 --log watchdog.log &
    while kill -0 "$JOB" 2>/dev/null; do sleep 5; done
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Размер страницы памяти — RSS в /proc/<pid>/statm записан в страницах.
PAGE = os.sysconf("SC_PAGE_SIZE")
GIB = 1024**3


@dataclass(frozen=True)
class Proc:
    """Процесс сессии: pid, RSS в байтах, имя и начало командной строки."""

    pid: int
    rss: int
    comm: str
    cmdline: str


def session_processes(sid: int) -> list[Proc]:
    """Все живые процессы с идентификатором сессии ``sid``.

    Args:
        sid: Идентификатор сессии (PID процесса, запущенного через ``setsid``).

    Returns:
        Процессы сессии с RSS; исчезнувшие во время обхода пропускаются.
    """
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            # Имя процесса в скобках может содержать пробелы — поля после него режем по «)».
            fields = stat[stat.rindex(")") + 2 :].split()
            session = int(fields[3])  # state ppid pgrp session …
            if session != sid:
                continue
            rss = int((entry / "statm").read_text().split()[1]) * PAGE
            comm = stat[stat.index("(") + 1 : stat.rindex(")")]
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            continue
        found.append(Proc(int(entry.name), rss, comm, cmdline[:120]))
    return found


def mem_available() -> int:
    """Доступная память системы (``MemAvailable``), байт."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def kill_session(processes: list[Proc]) -> None:
    """Послать SIGKILL всем процессам сессии (по группам и поштучно — на случай своих групп у детей)."""
    for proc in processes:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Сторож памяти для сессии тяжёлого запуска")
    parser.add_argument("--sid", type=int, required=True, help="Сессия запуска (PID после setsid)")
    parser.add_argument("--limit-gb", type=float, default=64.0, help="Потолок суммы RSS сессии, ГБ")
    parser.add_argument(
        "--min-available-gb", type=float, default=16.0, help="Убить, если у системы свободно меньше, ГБ"
    )
    parser.add_argument("--period-s", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=12, help="Сколько самых тяжёлых процессов писать в лог")
    parser.add_argument("--log", type=Path, required=True)
    arguments = parser.parse_args()

    peak = 0
    with arguments.log.open("a", encoding="utf-8") as log:
        log.write(
            f"# сторож: сессия {arguments.sid}, потолок {arguments.limit_gb} ГБ, "
            f"минимум свободной {arguments.min_available_gb} ГБ\n"
        )
        while True:
            processes = session_processes(arguments.sid)
            if not processes:
                log.write(f"{time.strftime('%H:%M:%S')} сессия завершилась; пик RSS {peak / GIB:.1f} ГБ\n")
                return 0
            total = sum(p.rss for p in processes)
            available = mem_available()
            peak = max(peak, total)
            stamp = time.strftime("%H:%M:%S")
            log.write(
                f"{stamp} процессов {len(processes)}, RSS сессии {total / GIB:.1f} ГБ, "
                f"свободно в системе {available / GIB:.1f} ГБ\n"
            )
            for proc in sorted(processes, key=lambda p: p.rss, reverse=True)[: arguments.top]:
                log.write(f"    {proc.pid:>8} {proc.rss / GIB:6.2f} ГБ  {proc.comm:<16} {proc.cmdline}\n")
            reason = None
            if total > arguments.limit_gb * GIB:
                reason = f"RSS сессии {total / GIB:.1f} ГБ > потолка {arguments.limit_gb} ГБ"
            elif available < arguments.min_available_gb * GIB:
                reason = f"свободно {available / GIB:.1f} ГБ < {arguments.min_available_gb} ГБ"
            if reason:
                log.write(f"{stamp} УБИВАЮ СЕССИЮ: {reason}\n")
                log.flush()
                kill_session(processes)
                return 1
            log.flush()
            time.sleep(arguments.period_s)


if __name__ == "__main__":
    sys.exit(main())
