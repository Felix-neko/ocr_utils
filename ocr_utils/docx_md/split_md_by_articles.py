#!/usr/bin/env python3
"""Нарезка MD-файла (OCR журнала/книги) на куски по статьям для подачи в LLM.

Задача: подавать LLM не весь документ целиком (это размывает внимание модели по
стратегии Long Context), а куски по одной / несколько статей. Жёсткое условие —
**одна статья не должна разрываться между кусками**. Дополнительно: кусок не крупнее
половины исходного объёма и (желательно) не совсем мелкий.

Как определяются границы статей
-------------------------------
В экспортах ``docx_to_md.py`` начало статьи/рубрики надёжно помечено «сильным»
заголовком Markdown:

* статьи — жирный ``######`` (h6), обычно с курсивным байлайном автора над ним;
* рубрики — жирный ``###`` (h3) («КРИТИКА И БИБЛИОГРАФИЯ» и т. п.);
* обложки — ``#`` / ``##``.

Заголовки ``####`` / ``#####`` и одиночные ``#`` внутри тела — это OCR-мусор
(одиночная ``*``, пунктуация, ``/ V 520``), их границей считать нельзя. Поэтому у
настоящего заголовка-границы после снятия разметки должно остаться достаточно букв.

Результат
---------
Рядом с исходником создаётся папка ``<имя>.chunks/`` с файлами ``chunk_01.md`` … и
``manifest.json`` (что в каком куске, размеры, диапазоны строк).

--------------------------------------------------------------------------------
КАК ЭТО РАБОТАЕТ (обзор пайплайна ``split_file``)
--------------------------------------------------------------------------------
Работаем на уровне СТРОК исходного .md (нарезка идёт по границам строк, содержимое
дословно не меняется — это гарантирует, что сумма кусков == исходник).

  1. ``parse_units``      — делит документ на АТОМАРНЫЕ юниты (неделимые единицы:
                            статья / рубрика / обложка). Граница юнита = «сильный»
                            заголовок; если над ним есть курсивный байлайн автора,
                            юнит начинается с байлайна (автор не отрывается от статьи).
                            Всё до первого заголовка — юнит «фронт-материя» (обложка+TOC).
  2. ``merge_tiny_units`` — вливает слишком короткие юниты в соседа: рубрику-заголовок
                            — ВПЕРЁД к её статье, хвостовые огрызки (задняя обложка) —
                            НАЗАД. Так исчезают «мусорные» микро-куски.
  3. ``group_units``      — собирает юниты в куски: либо по одному (``--per-article``),
                            либо жадной упаковкой подряд, пока не превышен ``--max-chars``.
                            Юнит НИКОГДА не режется → статья не рвётся.
  4. ``write_chunks``     — пишет ``chunk_NN.md`` (каждый с шапкой-оглавлением) и
                            ``manifest.json``.

Ключевые гарантии (проверяемые инварианты):
  * покрытие строк сплошное, без дыр и нахлёстов → сумма символов кусков == исходник;
  * каждый «сильный» заголовок (=статья) целиком внутри ровно одного куска;
  * ни один кусок не крупнее 50% файла (иначе предупреждение в stderr).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Заголовки этих уровней считаем кандидатами в границы статей/рубрик/обложек.
# 4 и 5 исключены НАМЕРЕННО: в OCR это почти всегда мусорные псевдозаголовки внутри
# тела статьи (обрывок строки, ошибочно распознанный как заголовок).
BOUNDARY_LEVELS = {1, 2, 3, 6}

# Минимум букв в очищенном тексте заголовка, чтобы считать его настоящим (а не
# мусором вида ``*``, ``/ V 520``, ``.  . *``).
MIN_HEADING_LETTERS = 8

# Дефолтные бюджеты (в символах).
# 150k ≈ 25% типичного номера журнала → в кусок влезает 2–4 статьи, комфортно для LLM.
DEFAULT_MAX_CHARS = 150_000
# Юнит короче этого сливается с соседом (обложка, TOC, одинокий рубрика-заголовок).
DEFAULT_MIN_CHARS = 4_000

# Заголовок Markdown: 1..6 знаков ``#``, пробел, текст.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# «Буква» в смысле Unicode: не-словосимвол (\W), цифра или подчёркивание исключены.
# Считаем и кириллицу, и латиницу — на случай смешанного OCR.
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass
class Unit:
    """Атомарная (неделимая) единица документа — статья, рубрика или обложка.

    Хранит свой диапазон строк ``[start_line, end_line)`` и сами строки (``lines``),
    чтобы запись кусков была дословной. ``title``/``byline`` — только для метаданных
    (шапки кусков и manifest), на нарезку они не влияют.
    """

    start_line: int
    end_line: int  # не включительно (полуинтервал)
    title: str
    byline: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        # +1 на каждый перевод строки — компенсируем ``\n``, «съеденный» при split("\n").
        # Благодаря этому сумма chars по всем юнитам точно равна размеру исходного файла.
        return sum(len(l) + 1 for l in self.lines)

    @property
    def label(self) -> str:
        """Человекочитаемая подпись юнита («автор — заголовок») для шапок/логов."""

        if self.byline and self.title:
            return f"{self.byline} — {self.title}"
        return self.title or self.byline or "[без заголовка]"


def _clean_heading_text(raw: str) -> str:
    """Убирает markdown-разметку (``*``, ``_``, `` ` ``) из текста заголовка/байлайна."""

    return re.sub(r"[*_`]", "", raw).strip()


def _heading_level(line: str) -> int | None:
    """Уровень «сильного» заголовка-границы или ``None``.

    Отсекаем: колонтитулы-комментарии (``<!-- … -->``), заголовки неподходящих уровней
    (4/5), и мусорные заголовки, где после снятия разметки слишком мало букв. Именно
    последний фильтр спасает от ложных границ на OCR-артефактах обложки.
    """

    s = line.strip()
    if s.startswith("<!--"):
        return None
    m = _HEADING_RE.match(s)
    if not m:
        return None
    level = len(m.group(1))
    if level not in BOUNDARY_LEVELS:
        return None
    text = _clean_heading_text(m.group(2))
    if len(_LETTER_RE.findall(text)) < MIN_HEADING_LETTERS:
        return None
    return level


def _is_byline(line: str) -> bool:
    """Похоже ли на курсивный байлайн автора (``*И. Фамилия*``)?

    Признаки: строка целиком в курсиве/жирном (``*…*`` / ``_…_``), без тире ``—``
    (тире встречается в оглавлении «Автор — Название», байлайн же — только имя), и это
    именно короткая строка-имя (2..60 букв, ≤ 80 символов), а не абзац текста.
    """

    s = line.strip()
    if s.startswith("<!--") or "—" in s:
        return False
    if not re.match(r"^[*_].+[*_]$", s):
        return False
    inner = _clean_heading_text(s)
    letters = len(_LETTER_RE.findall(inner))
    return 2 <= letters <= 60 and len(inner) <= 80


def _prev_nonempty(lines: list[str], idx: int) -> tuple[int, str]:
    """Индекс и текст ближайшей непустой строки ВЫШЕ ``idx`` (или ``(-1, "")``)."""

    j = idx - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    return (j, lines[j]) if j >= 0 else (-1, "")


def parse_units(lines: list[str]) -> list[Unit]:
    """Разбивает документ на атомарные юниты по сильным заголовкам."""

    # 1) Находим стартовые строки юнитов = строки с сильным заголовком. Если прямо над
    #    заголовком стоит байлайн автора — сдвигаем старт на строку байлайна, чтобы
    #    автор попал в ТУ ЖЕ статью, а не в конец предыдущей.
    starts: list[int] = []
    for i, line in enumerate(lines):
        if _heading_level(line) is None:
            continue
        prev_idx, prev_line = _prev_nonempty(lines, i)
        start = prev_idx if (prev_idx >= 0 and _is_byline(prev_line)) else i
        # Защита от «наезда»: если из-за байлайна старт заехал бы в предыдущий юнит
        # (или совпал с ним), берём саму строку заголовка.
        if starts and start <= starts[-1]:
            start = i
        starts.append(start)

    # 2) Фронт-материя: всё от начала файла до первого заголовка — отдельный юнит 0
    #    (обложка + оглавление). Если первый заголовок и так на строке 0 — юнита 0 нет.
    boundaries = starts[:]
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries

    # 3) Границы → полуинтервалы [start, next_start) → объекты Unit.
    units: list[Unit] = []
    for k, start in enumerate(boundaries):
        end = boundaries[k + 1] if k + 1 < len(boundaries) else len(lines)
        block = lines[start:end]
        title, byline = _unit_title_byline(block)
        units.append(Unit(start_line=start, end_line=end, title=title, byline=byline, lines=block))
    return units


def _unit_title_byline(block: list[str]) -> tuple[str, str]:
    """Достаёт заголовок и (если есть) байлайн автора из строк юнита — для метаданных."""

    title = ""
    byline = ""
    for idx, line in enumerate(block):
        if _heading_level(line) is not None:
            m = _HEADING_RE.match(line.strip())
            title = _clean_heading_text(m.group(2)) if m else ""
            prev_idx, prev_line = _prev_nonempty(block, idx)
            if prev_idx >= 0 and _is_byline(prev_line):
                byline = _clean_heading_text(prev_line)
            break
    if not title:
        title = "[фронт-материя / обложка]"
    return title, byline


def merge_tiny_units(units: list[Unit], min_chars: int) -> list[Unit]:
    """Вливает слишком короткие юниты в соседний (рубрику — вперёд, хвост — назад).

    Зачем: рубрика — это отдельный заголовок ``###``, за которым сразу идёт статья
    ``######``. Как самостоятельный юнит рубрика — это несколько строк заголовка, т. е.
    «мусорный» микро-кусок. Сливаем её ВПЕРЁД, в следующую статью (рубрика остаётся при
    своей статье, как в оглавлении). Прочие короткие огрызки в конце (задняя обложка) —
    НАЗАД, в предыдущий юнит.

    Реализация — повторяем один проход, пока есть изменения: каждое слияние меняет
    список, поэтому проще перезапускать цикл, чем аккуратно двигать индексы.
    """

    if not units:
        return units

    changed = True
    while changed and len(units) > 1:
        changed = False
        for i, u in enumerate(units):
            if u.chars >= min_chars:
                continue
            if i + 1 < len(units):
                # Есть следующий — сливаем вперёд: строки текущего (рубрика) + следующий.
                # Метаданные (title/byline) берём у следующего — это «настоящая» статья.
                nxt = units[i + 1]
                merged = Unit(
                    start_line=u.start_line,
                    end_line=nxt.end_line,
                    title=nxt.title,
                    byline=nxt.byline,
                    lines=u.lines + nxt.lines,
                )
                units = units[:i] + [merged] + units[i + 2 :]
            else:
                # Это последний юнит и он мал — сливаем назад, в предыдущий.
                prev = units[i - 1]
                merged = Unit(
                    start_line=prev.start_line,
                    end_line=u.end_line,
                    title=prev.title,
                    byline=prev.byline,
                    lines=prev.lines + u.lines,
                )
                units = units[: i - 1] + [merged]
            changed = True
            break  # список изменился — начинаем проход заново
    return units


def group_units(units: list[Unit], max_chars: int, per_article: bool) -> list[list[Unit]]:
    """Группирует юниты в куски: по одному, либо жадной упаковкой под бюджет.

    Инвариант обоих режимов: юнит целиком в одном куске (статья не разрывается).
    """

    if per_article:
        return [[u] for u in units]

    # Жадная упаковка: добавляем юниты в текущий кусок, пока помещаются в бюджет.
    chunks: list[list[Unit]] = []
    current: list[Unit] = []
    current_chars = 0
    for u in units:
        # Один юнит сам крупнее бюджета — не режем (нельзя рвать статью): закрываем
        # текущий кусок и кладём этот юнит отдельным куском целиком, с предупреждением.
        if u.chars > max_chars:
            if current:
                chunks.append(current)
                current, current_chars = [], 0
            chunks.append([u])
            print(
                f"ВНИМАНИЕ: юнит «{u.label[:60]}» ({u.chars} симв) больше бюджета "
                f"{max_chars}; кладу отдельным куском целиком.",
                file=sys.stderr,
            )
            continue
        # Не влезает в текущий кусок — начинаем новый.
        if current and current_chars + u.chars > max_chars:
            chunks.append(current)
            current, current_chars = [], 0
        current.append(u)
        current_chars += u.chars
    if current:
        chunks.append(current)
    return chunks


def _chunk_header(idx: int, total: int, chunk: list[Unit]) -> str:
    """Служебная шапка-оглавление куска (HTML-комментарий в начале ``chunk_NN.md``).

    Даёт LLM (и человеку) быстрый обзор: какой это кусок из скольких, диапазон строк
    исходника и список статей внутри с их размерами.
    """

    start = chunk[0].start_line
    end = chunk[-1].end_line
    chars = sum(u.chars for u in chunk)
    head = [f"<!-- CHUNK {idx}/{total} | строки {start}–{end} | {chars} симв -->"]
    head.append("<!-- Статьи в этом куске:")
    for u in chunk:
        head.append(f"     - {u.label} ({u.chars} симв)")
    head.append("-->")
    return "\n".join(head)


def write_chunks(
    src: Path, lines: list[str], chunks: list[list[Unit]], out_dir: Path, mode: str, max_chars: int
) -> dict:
    """Пишет ``chunk_NN.md`` (+ ``manifest.json``) и возвращает данные manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(chunks)
    total_chars = sum(len(l) + 1 for l in lines)
    # Ширина номера в имени файла: минимум 2 знака (chunk_01), больше — если кусков >99.
    width = max(2, len(str(total)))

    manifest_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        fname = f"chunk_{i:0{width}d}.md"
        # Тело куска — дословные строки всех его юнитов подряд.
        body = "\n".join(u_line for u in chunk for u_line in u.lines)
        header = _chunk_header(i, total, chunk)
        (out_dir / fname).write_text(header + "\n\n" + body.rstrip("\n") + "\n", encoding="utf-8")

        chunk_chars = sum(u.chars for u in chunk)
        manifest_chunks.append(
            {
                "id": i,
                "file": fname,
                "start_line": chunk[0].start_line,
                "end_line": chunk[-1].end_line,
                "chars": chunk_chars,
                "pct": round(100 * chunk_chars / total_chars, 1),
                "units": [{"title": u.title, "byline": u.byline, "chars": u.chars} for u in chunk],
            }
        )

    manifest = {
        "source": str(src),
        "total_chars": total_chars,
        "mode": mode,
        "max_chars": max_chars,
        "n_chunks": total,
        "chunks": manifest_chunks,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def split_file(src: Path, out_dir: Path | None, max_chars: int, min_chars: int, per_article: bool) -> dict:
    """Полный цикл нарезки одного файла (parse → merge → group → write)."""

    lines = src.read_text(encoding="utf-8").split("\n")
    units = parse_units(lines)
    units = merge_tiny_units(units, min_chars)
    chunks = group_units(units, max_chars, per_article)

    # По умолчанию — папка ``<имя>.chunks/`` рядом с исходником.
    if out_dir is None:
        out_dir = src.with_suffix("").with_name(src.stem + ".chunks")
    mode = "per-article" if per_article else f"pack<= {max_chars}"
    manifest = write_chunks(src, lines, chunks, out_dir, mode, max_chars)

    # Инвариант «≤ половины файла»: если нарушен — не падаем, но громко предупреждаем.
    biggest = max((c["chars"] for c in manifest["chunks"]), default=0)
    if biggest > 0.5 * manifest["total_chars"]:
        print(
            f"ВНИМАНИЕ: самый большой кусок — {biggest} симв "
            f"({100 * biggest / manifest['total_chars']:.0f}% файла), больше половины.",
            file=sys.stderr,
        )
    # Компактный отчёт в stdout: файл, размер, доля, число юнитов и первые слова названий.
    print(f"{src}  ->  {out_dir}/  ({manifest['n_chunks']} кусков)")
    for c in manifest["chunks"]:
        titles = ", ".join((u["byline"] or u["title"] or "?").split(" ")[0].rstrip(",.") for u in c["units"])
        print(f"  {c['file']}: {c['chars']:>7} симв ({c['pct']:>4}%)  [{len(c['units'])}] {titles}")
    return manifest


def _iter_md_files(paths: list[Path], recursive: bool) -> list[Path]:
    """Разворачивает пути (файлы/каталоги) в плоский список .md-файлов."""

    result: list[Path] = []
    for p in paths:
        if p.is_dir():
            pattern = "**/*.md" if recursive else "*.md"
            result.extend(sorted(q for q in p.glob(pattern)))
        elif p.suffix.lower() == ".md":
            result.append(p)
        else:
            print(f"Пропускаю (не .md): {p}", file=sys.stderr)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Нарезка MD на куски по статьям для LLM.")
    parser.add_argument("paths", nargs="+", type=Path, help="MD-файлы или каталоги с ними")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Бюджет символов на кусок при упаковке (по умолчанию {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Порог слияния крошечных юнитов (по умолчанию {DEFAULT_MIN_CHARS})",
    )
    parser.add_argument("--per-article", action="store_true", help="Один юнит (статья/рубрика) = один кусок")
    parser.add_argument(
        "--out", type=Path, default=None, help="Каталог для кусков (по умолчанию <имя>.chunks/ рядом с файлом)"
    )
    parser.add_argument("--recursive", action="store_true", help="Искать .md в подкаталогах")
    args = parser.parse_args(argv)

    files = _iter_md_files(args.paths, args.recursive)
    if not files:
        print("Не найдено ни одного .md", file=sys.stderr)
        return 1

    exit_code = 0
    for md in files:
        # При нескольких файлах общий --out игнорируем, чтобы куски разных файлов не
        # перезатирали друг друга (каждый идёт в свою <имя>.chunks/).
        out_dir = args.out if (args.out and len(files) == 1) else None
        try:
            split_file(md, out_dir, args.max_chars, args.min_chars, args.per_article)
        except Exception as exc:  # noqa: BLE001 — продолжаем с остальными файлами
            print(f"ОШИБКА при обработке {md}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
