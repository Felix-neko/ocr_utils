"""Запуск батч-воркера DeepSeek-OCR-2 (vLLM, своё окружение) подпроцессом под сторожем памяти и ожидание конца."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Окружение и воркер vLLM (рядом с логикой DeepSeek).
DEEPSEEK_DIR = Path(__file__).resolve().parent / "vllm_env"
WATCHDOG = Path(__file__).resolve().parents[4] / "scripts" / "memory_watchdog.py"
# Сторож: потолок памяти сессии vLLM и минимум свободной памяти системы, ГБ.
WATCHDOG_LIMIT_GB = 90
WATCHDOG_MIN_AVAILABLE_GB = 12


def run_worker(jobs_file: Path, out_dir: Path, prompts: str, extra_args: tuple[str, ...] = ()) -> None:
    """Воркер vLLM подпроцессом (своя сессия) под сторожем памяти; ждёт окончания.

    Тяжёлый запуск — только под сторожем: ``systemd-run --user MemoryMax`` на этой машине память не
    ограничивает, а сборка ядер FlashInfer без ``MAX_JOBS`` съедала всю память (2026-09-27).

    Args:
        jobs_file: JSONL заданий ``{"id", "crop" — путь к PNG}``; пустой файл — ничего не делать.
        out_dir: Куда воркер дописывает ``<промпт>.jsonl``; лог и лог сторожа — рядом с ней.
        prompts: Промпты через запятую (ключи ``parse.PROMPTS``).
        extra_args: Дополнительные аргументы воркера (``--max-new-tokens`` и т. п.).

    Raises:
        RuntimeError: Воркер завершился с ненулевым кодом.
    """
    if not jobs_file.read_text().strip():
        return
    log = out_dir.parent / f"{out_dir.name}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "uv", "run", "--project", str(DEEPSEEK_DIR), "python", str(DEEPSEEK_DIR / "worker.py"),
        "--jobs", str(jobs_file), "--out-dir", str(out_dir), "--prompts", prompts, *extra_args,
    ]  # fmt: skip
    with log.open("a") as handle:
        worker = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
        )
    watchdog = subprocess.Popen(
        [
            sys.executable, str(WATCHDOG), "--sid", str(worker.pid), "--limit-gb", str(WATCHDOG_LIMIT_GB),
            "--min-available-gb", str(WATCHDOG_MIN_AVAILABLE_GB), "--log", str(out_dir.parent / f"{out_dir.name}_watchdog.log"),
        ],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )  # fmt: skip
    code = worker.wait()
    watchdog.wait()
    if code != 0:
        raise RuntimeError(f"воркер DeepSeek завершился с кодом {code}, лог {log}")
    logger.info("DeepSeek %s: готово (%s)", out_dir.name, prompts)


__all__ = ["DEEPSEEK_DIR", "WATCHDOG", "run_worker"]
