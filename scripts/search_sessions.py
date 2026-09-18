"""Поиск по переписке прошлых сессий Claude Code: список сессий, поиск по регулярке, выгрузка диалога."""

from __future__ import annotations

import enum
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import click

# Куда Claude Code складывает JSONL-транскрипты: ~/.claude/projects/<слаг каталога проекта>/<id сессии>.jsonl.
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
# Кэш разобранных сессий: JSON на сессию с только текстовыми репликами, без выводов инструментов.
CACHE_ROOT = Path.home() / ".cache" / "claude_recall"
# Служебные вставки, которые в реплики пользователя подмешивает сам Claude Code; в поиске они только мешают.
SERVICE_TAG_RE = re.compile(
    r"<(system-reminder|local-command-caveat|local-command-stdout|command-name|command-message|command-args)>"
    r".*?</\1>",
    re.S,
)


class Role(enum.Enum):
    """Кто говорит: пользователь, ассистент, сводка контекста после сжатия или уведомление фоновой задачи."""

    USER = "U"
    ASSISTANT = "A"
    SUMMARY = "S"
    NOTIFICATION = "N"


@dataclass(frozen=True)
class Message:
    """Одна текстовая реплика диалога: время (ISO), роль и очищенный текст."""

    timestamp: str
    role: str
    text: str


@dataclass(frozen=True)
class Session:
    """Разобранная сессия: идентификатор, название от Claude, границы по времени и текстовые реплики."""

    session_id: str
    title: str
    started: str
    finished: str
    messages: tuple[Message, ...]


def project_slug(project_dir: Path) -> str:
    """Слаг каталога проекта так, как его строит Claude Code: все не-буквенно-цифровые символы → «-».

    Args:
        project_dir: абсолютный путь к каталогу проекта (обычно текущий cwd).

    Returns:
        Имя подкаталога в `~/.claude/projects/`, например `-home-felix-Projects-ocr-utils`.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(project_dir.resolve()))


def _content_text(content: str | list) -> str:
    """Собрать текст из поля `message.content`: строка как есть, список — только блоки `text`.

    Args:
        content: строка либо список блоков (`text`, `tool_use`, `tool_result`, `thinking`, `image`).

    Returns:
        Склеенный текст блоков `text` без служебных вставок; пустая строка, если текста нет.
    """
    if isinstance(content, str):
        parts = [content]
    else:
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    text = "\n".join(part for part in parts if part)
    return SERVICE_TAG_RE.sub("", text).strip()


def parse_session(path: Path) -> Session:
    """Разобрать один JSONL-транскрипт в `Session`: оставить только реплики с текстом.

    Пропускаются служебные записи (`attachment`, `mode`, …), результаты инструментов, размышления модели и
    реплики с флагом `isMeta` (их вставляет сам Claude Code). Сводки после сжатия контекста
    (`isCompactSummary`) сохраняются с ролью `S`, уведомления фоновых задач (`<task-notification>`, приходят
    от имени пользователя) — с ролью `N`. Название сессии — последнее `ai-title` в файле.

    Args:
        path: путь к файлу `<id сессии>.jsonl`.

    Returns:
        `Session` с репликами в порядке файла; при отсутствии времени в записях границы — пустые строки.
    """
    title = ""
    messages: list[Message] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            # Дешёвый отсев до разбора JSON: нужны только три типа записей.
            if '"type":"user"' not in line and '"type":"assistant"' not in line and '"type":"ai-title"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = record.get("type")
            if kind == "ai-title":
                title = record.get("aiTitle") or title
                continue
            if kind not in ("user", "assistant") or record.get("isMeta"):
                continue
            text = _content_text(record.get("message", {}).get("content", ""))
            if not text:
                continue
            if record.get("isCompactSummary"):
                role = Role.SUMMARY
            elif kind == "assistant":
                role = Role.ASSISTANT
            elif text.startswith("<task-notification>"):
                role = Role.NOTIFICATION
            else:
                role = Role.USER
            messages.append(Message(timestamp=record.get("timestamp", ""), role=role.value, text=text))
    stamps = [m.timestamp for m in messages if m.timestamp]
    return Session(
        session_id=path.stem,
        title=title,
        started=min(stamps) if stamps else "",
        finished=max(stamps) if stamps else "",
        messages=tuple(messages),
    )


def load_session(path: Path, cache_dir: Path) -> Session:
    """Взять сессию из кэша, если файл не менялся, иначе разобрать заново и положить в кэш.

    Args:
        path: путь к JSONL-транскрипту.
        cache_dir: каталог кэша для этого проекта; создаётся при необходимости.

    Returns:
        `Session` — из кэша или свежеразобранная.
    """
    stat = path.stat()
    stamp = f"{int(stat.st_mtime)}-{stat.st_size}"
    cache_file = cache_dir / f"{path.stem}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("stamp") == stamp:
                return Session(
                    **{**cached["session"], "messages": tuple(Message(**m) for m in cached["session"]["messages"])}
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    session = parse_session(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"stamp": stamp, "session": asdict(session)}, ensure_ascii=False), encoding="utf-8"
    )
    return session


def load_sessions(project_dir: Path, since: str | None, session_prefix: str | None) -> list[Session]:
    """Загрузить все сессии проекта, отфильтровать по дате и префиксу идентификатора, отсортировать по началу.

    Args:
        project_dir: каталог проекта, по которому вычисляется слаг в `~/.claude/projects/`.
        since: нижняя граница даты `YYYY-MM-DD` по последней реплике сессии; `None` — без фильтра.
        session_prefix: оставить только сессии, чей идентификатор начинается с этой строки; `None` — все.

    Returns:
        Список сессий с хотя бы одной репликой, от старых к новым.
    """
    slug = project_slug(project_dir)
    transcripts_dir = CLAUDE_PROJECTS_DIR / slug
    if not transcripts_dir.is_dir():
        raise click.ClickException(f"нет каталога транскриптов {transcripts_dir}")
    cache_dir = CACHE_ROOT / slug
    sessions = []
    for path in transcripts_dir.glob("*.jsonl"):
        if session_prefix and not path.stem.startswith(session_prefix):
            continue
        session = load_session(path, cache_dir)
        if not session.messages:
            continue
        if since and session.finished[:10] < since:
            continue
        sessions.append(session)
    return sorted(sessions, key=lambda s: s.started)


def _fmt_time(iso: str) -> str:
    """ISO-время из транскрипта → `ГГГГ-ММ-ДД ЧЧ:ММ` в локальном поясе; пустая строка остаётся пустой."""
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso[:16]


def _clip(text: str, max_chars: int, pattern: re.Pattern | None) -> str:
    """Обрезать реплику до `max_chars`, при наличии `pattern` — окном вокруг первого совпадения.

    Args:
        text: полный текст реплики.
        max_chars: предел длины; 0 — не обрезать.
        pattern: регулярка поиска, чтобы совпадение гарантированно попало в окно; `None` — резать с начала.

    Returns:
        Текст, возможно с `…` по краям обрезки.
    """
    if not max_chars or len(text) <= max_chars:
        return text
    start = 0
    if pattern is not None:
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - max_chars // 3)
    end = min(len(text), start + max_chars)
    clipped = text[start:end]
    return ("…" if start else "") + clipped + ("…" if end < len(text) else "")


def _print_session_header(session: Session) -> None:
    """Напечатать заголовок сессии: время начала, короткий id, название."""
    click.echo(click.style(f"=== {_fmt_time(session.started)} | {session.session_id[:8]} | {session.title}", bold=True))


def _print_message(message: Message, index: int, max_chars: int, pattern: re.Pattern | None) -> None:
    """Напечатать одну реплику с номером, ролью и временем; текст обрезается окном вокруг совпадения."""
    click.echo(click.style(f"[{index} {message.role} {_fmt_time(message.timestamp)}]", fg="cyan"))
    click.echo(_clip(message.text, max_chars, pattern))
    click.echo()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    show_default=True,
    help="Каталог проекта; по нему находится папка транскриптов в ~/.claude/projects/.",
)
@click.pass_context
def cli(ctx: click.Context, project_dir: Path) -> None:
    """Поиск по переписке прошлых сессий Claude Code этого проекта."""
    ctx.obj = project_dir


@cli.command()
@click.option("--since", help="Только сессии, закончившиеся не раньше даты YYYY-MM-DD.")
@click.pass_obj
def titles(project_dir: Path, since: str | None) -> None:
    """Список сессий: начало, конец, id, число реплик, название."""
    for session in load_sessions(project_dir, since, None):
        click.echo(
            f"{_fmt_time(session.started)}  {_fmt_time(session.finished)}  {session.session_id[:8]}  "
            f"{len(session.messages):4d}  {session.title}"
        )


@cli.command()
@click.argument("pattern")
@click.option("--since", help="Только сессии, закончившиеся не раньше даты YYYY-MM-DD.")
@click.option("--session", "session_prefix", help="Только сессия с этим префиксом id.")
@click.option(
    "--role",
    type=click.Choice([r.value for r in Role]),
    multiple=True,
    help="Искать только в репликах этих ролей (U — пользователь, A — ассистент, S — сводка, N — уведомление задачи); можно несколько.",
)
@click.option(
    "--context", "-C", default=1, show_default=True, help="Сколько соседних реплик показывать вокруг совпадения."
)
@click.option(
    "--max-chars", default=1500, show_default=True, help="Обрезать каждую реплику до стольких символов; 0 — не резать."
)
@click.option("--limit", default=50, show_default=True, help="Не больше стольких совпадений всего.")
@click.pass_obj
def search(
    project_dir: Path,
    pattern: str,
    since: str | None,
    session_prefix: str | None,
    role: tuple[str, ...],
    context: int,
    max_chars: int,
    limit: int,
) -> None:
    """Найти реплики по регулярке (без учёта регистра) и показать их с соседними репликами."""
    regex = re.compile(pattern, re.I)
    roles = set(role) if role else {r.value for r in Role}
    shown = 0
    for session in load_sessions(project_dir, since, session_prefix):
        hits = [i for i, m in enumerate(session.messages) if m.role in roles and regex.search(m.text)]
        if not hits:
            continue
        _print_session_header(session)
        # Соседние окна сливаем, чтобы одна реплика не печаталась дважды.
        printed: set[int] = set()
        for hit in hits:
            if shown >= limit:
                break
            shown += 1
            for i in range(max(0, hit - context), min(len(session.messages), hit + context + 1)):
                if i in printed:
                    continue
                printed.add(i)
                _print_message(session.messages[i], i, max_chars, regex if i == hit else None)
        if shown >= limit:
            click.echo(click.style(f"… достигнут --limit {limit}", fg="yellow"))
            break
    click.echo(f"совпадений показано: {shown}", err=True)


@cli.command()
@click.argument("session_prefix")
@click.option("--start", default=0, show_default=True, help="Номер первой реплики.")
@click.option(
    "--end", default=None, type=int, help="Номер реплики, до которой (не включая) печатать; по умолчанию до конца."
)
@click.option("--max-chars", default=0, show_default=True, help="Обрезать каждую реплику; 0 — печатать целиком.")
@click.pass_obj
def dump(project_dir: Path, session_prefix: str, start: int, end: int | None, max_chars: int) -> None:
    """Выгрузить диалог одной сессии (по префиксу id) в текстовом виде, с номерами реплик."""
    sessions = load_sessions(project_dir, None, session_prefix)
    if len(sessions) != 1:
        raise click.ClickException(f"по префиксу {session_prefix!r} найдено сессий: {len(sessions)}, нужна ровно одна")
    session = sessions[0]
    _print_session_header(session)
    for i, message in enumerate(session.messages[start:end], start=start):
        _print_message(message, i, max_chars, None)


if __name__ == "__main__":
    sys.exit(cli())
