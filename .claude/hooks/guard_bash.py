#!/usr/bin/env python3
"""PreToolUse-хук на Bash: ловит команды, которые в этом проекте уже стоили данных или часов.

Читает JSON вызова из stdin (``tool_input.command``), отвечает одним из трёх способов:

* exit 2 и текст в stderr — команда блокируется, текст видит агент;
* JSON ``permissionDecision: ask`` в stdout — команда идёт на подтверждение пользователю;
* exit 0 без вывода — пропустить.

Только stdlib: хук запускается перед каждой командой, стартовать uv ради него дорого.
Проверить руками::

    echo '{"tool_input":{"command":"pgrep -f exiftool"}}' | python3 .claude/hooks/guard_bash.py
"""

from __future__ import annotations

import json
import re
import shlex
import sys

YANDEX_ROOT = "/mnt/dump3/yandex_disk_linux_baby_zergling"
DUMP3 = "/mnt/dump3"
MARKUP_ROOT = "/home/felix/Projects/mts_markup"

# Команды, у которых последний позиционный аргумент — место записи.
COPY_LIKE = {"cp", "mv", "rsync", "scp", "install"}
# Команды, у которых любой путь-аргумент — место записи или удаления.
WRITE_ANY = {"mkdir", "touch", "rm", "rmdir", "truncate", "tee", "dd", "ln", "chmod", "chown", "unzip", "tar"}
# Имена опций, значение которых — каталог/файл вывода.
OUT_OPTION = re.compile(r"^--?(out|output|dest|destination|target|debug|save|report|share)[\w-]*$")


def deny(reason: str) -> None:
    """Заблокировать команду: агент получает объяснение и должен переделать."""
    print(reason, file=sys.stderr)
    sys.exit(2)


def ask(reason: str) -> None:
    """Отдать решение пользователю, показав причину."""
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": reason,
                }
            },
            ensure_ascii=False,
        )
    )
    sys.exit(0)


# Тело heredoc — от конца строки с маркером до строки-терминатора; сама строка с маркером
# (в ней могут быть перенаправления вроде `<<EOF > файл`) остаётся.
HEREDOC = re.compile(r"(<<-?\s*['\"]?(\w+)['\"]?[^\n]*)\n.*?^\2\s*$", re.S | re.M)


def strip_text_payloads(command: str) -> str:
    """Убрать из команды текст, который не является аргументами: тела heredoc и сообщения коммитов.

    Иначе упоминание /mnt/system или pgrep -f в записываемом документе или в сообщении
    коммита блокирует саму команду, хотя на диски она не пишет.
    """
    command = HEREDOC.sub(r"\1", command)
    command = re.sub(r"(git\s+commit\b[^\n]*?)\s-m\s+(\"(?:[^\"\\]|\\.)*\"|'[^']*')", r"\1 -m MSG", command)
    return command


def split_commands(command: str) -> list[list[str]]:
    """Разбить строку шелла на простые команды по ; && || | и переводам строк."""
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        tokens = command.split()
    commands: list[list[str]] = [[]]
    for tok in tokens:
        if tok in {";", "&&", "||", "|", "&"} or tok.endswith((";", "&&", "||")):
            if tok not in {";", "&&", "||", "|", "&"}:
                commands[-1].append(tok.rstrip(";&|"))
            commands.append([])
        else:
            commands[-1].append(tok)
    return [c for c in commands if c]


def check_pgrep(command: str, simple: list[list[str]]) -> None:
    """pgrep/pkill -f без разрыва шаблона матчат сам шелл: цикл висит вечно, pkill убивает вызов."""
    for argv in simple:
        # Пропускаем env-присваивания и sudo/time перед командой.
        while argv and ("=" in argv[0] or argv[0] in {"sudo", "time", "nice", "ionice", "env"}):
            argv = argv[1:]
        if not argv or argv[0] not in {"pgrep", "pkill"}:
            continue
        uses_f = any(a.startswith("-") and not a.startswith("--") and "f" in a for a in argv[1:]) or "--full" in argv
        if not uses_f:
            continue
        patterns = [a for a in argv[1:] if not a.startswith("-")]
        if any("[" not in p for p in patterns):
            deny(
                f"{argv[0]} -f «{' '.join(patterns)}» совпадёт с командной строкой этого же шелла: "
                "pgrep будет вечно отвечать «жив», pkill убьёт текущий вызов (код 144). "
                "Ждать и убивать — по сохранённому PID (`kill -0 $PID`, `kill -- -$PGID`); "
                "если по имени никак — разорвать шаблон классом символов: pgrep -f '[e]xiftool'. См. CLAUDE.md."
            )
    if re.search(r"\bwhile\s+pgrep\b", command):
        deny('`while pgrep ...` — та же ловушка самосовпадения. Ждать по PID: `while kill -0 "$PID"`.')


def write_targets(simple: list[list[str]], command: str) -> list[str]:
    """Пути, в которые команда пишет или которые удаляет (эвристика по типичным утилитам)."""
    targets: list[str] = []
    for argv in simple:
        while argv and ("=" in argv[0] or argv[0] in {"sudo", "time", "nice", "ionice", "env"}):
            argv = argv[1:]
        if not argv:
            continue
        name = argv[0].rsplit("/", 1)[-1]
        positional = [a for a in argv[1:] if not a.startswith("-")]
        if name in COPY_LIKE and positional:
            targets.append(positional[-1])
        elif name in WRITE_ANY:
            targets.extend(positional)
        # Опции вида --output-dir /path и --output=/path у любой команды.
        for i, a in enumerate(argv):
            if "=" in a and a.startswith("-"):
                opt, _, val = a.partition("=")
                if OUT_OPTION.match(opt):
                    targets.append(val)
            elif OUT_OPTION.match(a) and i + 1 < len(argv):
                targets.append(argv[i + 1])
    # Перенаправления вывода: > path, >> path, 2> path.
    targets += re.findall(r"(?:^|[\s\d])>>?\s*[\"']?(/\S+)", command)
    return targets


def check_disks(command: str, simple: list[list[str]]) -> None:
    """Диски с ловушками: корень Яндекс.Диска, NTFS-3G, регистр /mnt/SYSTEM, база разметки."""
    bad_case = [m for m in re.findall(r"/mnt/[sS][yY][sS][tT][eE][mM]\b", command) if m != "/mnt/SYSTEM"]
    if bad_case:
        deny(
            f"Путь «{bad_case[0]}» — не тот том: регистр в /mnt/SYSTEM значим, строчный /mnt/system это пустой "
            "каталог на системном диске, вывод тихо уедет туда. Писать /mnt/SYSTEM. См. docs/data_layout.md."
        )

    targets = write_targets(simple, command)
    for t in targets:
        if t.startswith(YANDEX_ROOT):
            deny(
                f"Запись в «{t}»: это корень живой синхронизации Яндекс.Диска, демон переименует новый файл поверх "
                "исходника скана (уже терялись оригиналы). Результаты писать на /mnt/SYSTEM или в ~/Projects/mts_markup, "
                "внутрь переносит пользователь руками при остановленном демоне. См. docs/data_layout.md."
            )
    for t in targets:
        if t.startswith(DUMP3):
            ask(
                f"Запись в «{t}» на /mnt/dump3 — это медленный NTFS-3G на шпинделе, выход прогона обычно кладут на "
                "/mnt/SYSTEM. Подтвердите, если запись туда действительно нужна."
            )

    for argv in simple:
        if argv and argv[0].rsplit("/", 1)[-1] == "rm":
            for a in argv[1:]:
                if a.endswith(".sqlite") or a.startswith(MARKUP_ROOT) or a.startswith("~/Projects/mts_markup"):
                    deny(
                        f"rm «{a}»: базы и файлы разметки пака невоспроизводимы (ручная разметка неделями). "
                        "Удалять их может только пользователь."
                    )

    if re.search(r"\bfind\s+/\s", command + " ") or re.search(r"\b(find|grep\s+-r\w*|rg)\s+.*\s/\s*$", command):
        deny(
            "Рекурсивный обход от корня зацепит /mnt/dump3 (11 ТБ NTFS-3G): seek-шторм, запись пайплайна падает в 8 раз. "
            "Искать только внутри каталога проекта."
        )
    if re.search(r"\b(find|du|tree)\s+[^;&|]*" + re.escape(DUMP3), command):
        ask("Рекурсивный обход /mnt/dump3 (медленный NTFS-3G на шпинделе) замедлит идущие прогоны. Подтвердите.")


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    command = (payload.get("tool_input") or {}).get("command") or ""
    if not command:
        return
    command = strip_text_payloads(command)
    simple = split_commands(command)
    check_pgrep(command, simple)
    check_disks(command, simple)


if __name__ == "__main__":
    main()
