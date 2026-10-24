"""Схема сайдкара ``<кадр>.flags.json``: флаги кадра, чтение с проверкой, запись без перезаписи и удаление.

Сайдкар лежит рядом с фото и называется полным именем кадра плюс ``.flags.json``
(``IMG_0005.jpg.flags.json``), поэтому ``.jpg`` и ``.tif`` с одной основой не путаются, а
``collect_images`` его не подхватывает. У обычного кадра сайдкара нет. Пример::

    {
      "schema": 1,
      "flags": ["ignore-right"],
      "note": "библиотечная карточка закрывает правую страницу; правая плоско — IMG_0006.jpg",
      "pair": "IMG_0006.jpg",
      "reviewed": "2026-10-23"
    }

Схема расширяется добавлением значения в ``FrameFlag``. Неизвестный флаг при чтении — ошибка,
а не молчание: опечатка в сайдкаре не должна тихо пропасть.
"""

import datetime as dt
import json
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path

# Версия схемы, которую понимает этот код. Сайдкар с бо́льшей версией — ошибка чтения.
SCHEMA_VERSION = 1

# Суффикс сайдкара, дописываемый к полному имени кадра.
SIDECAR_SUFFIX = ".flags.json"


class FrameFlag(StrEnum):
    """Флаг кадра; значение — строка в сайдкаре."""

    IGNORE_LEFT = "ignore-left"  # левая страница кадра не идёт в выход страниц
    IGNORE_RIGHT = "ignore-right"  # правая страница кадра не идёт в выход страниц
    IGNORE_BOTH = "ignore-both"  # кадр не обрабатывается вовсе (обложка подшивки, форзац)
    NO_SPLITTING = "no-splitting"  # разворот не резать на страницы (сквозная таблица, схема)
    NO_ILLUMINATION_LEFT = "no-illumination-left"  # не выравнивать освещение левой страницы
    NO_ILLUMINATION_RIGHT = "no-illumination-right"  # не выравнивать освещение правой страницы
    NO_TEXT_RECTIFY_LEFT = "no-text-rectify-left"  # левую страницу не выпрямлять по строкам (второе преобразование)
    NO_TEXT_RECTIFY_RIGHT = "no-text-rectify-right"  # правую страницу не выпрямлять по строкам
    COVER_LEFT = "cover-left"  # левая страница — обложка своего номера (зона контента — вся страница, без обрезки)
    COVER_RIGHT = "cover-right"  # правая страница — обложка своего номера
    COVER_BOTH = "cover-both"  # обе страницы — обложка (разворот обложки)
    SINGLE_PAGE = "single-page"  # в кадре одна страница: не резать, писать одной страницей ``_P`` (перекрывает автомат)
    SPREAD = "spread"  # кадр — разворот: автомат «одна страница» на нём выключен


class Side(Enum):
    """Сторона разворота."""

    LEFT = "left"
    RIGHT = "right"


class FlagsError(ValueError):
    """Сайдкар не проходит проверку схемы (неизвестный флаг, противоречие, версия)."""


# Прежние имена флагов → нынешние: сайдкар со старым именем читается, при записи — новое имя.
LEGACY_FLAGS = {
    "no-color-equalize-left": FrameFlag.NO_ILLUMINATION_LEFT,
    "no-color-equalize-right": FrameFlag.NO_ILLUMINATION_RIGHT,
}

# Флаги «игнорировать страницу», «не выравнивать освещение», «не выпрямлять по строкам» по сторонам.
_IGNORE_OF = {Side.LEFT: FrameFlag.IGNORE_LEFT, Side.RIGHT: FrameFlag.IGNORE_RIGHT}
_NO_EQUALIZE_OF = {Side.LEFT: FrameFlag.NO_ILLUMINATION_LEFT, Side.RIGHT: FrameFlag.NO_ILLUMINATION_RIGHT}
_NO_TEXT_RECTIFY_OF = {Side.LEFT: FrameFlag.NO_TEXT_RECTIFY_LEFT, Side.RIGHT: FrameFlag.NO_TEXT_RECTIFY_RIGHT}
_COVER_OF = {Side.LEFT: FrameFlag.COVER_LEFT, Side.RIGHT: FrameFlag.COVER_RIGHT}
# Флаги, относящиеся к одной стороне разворота или к его разрезу: у кадра ``single-page`` сторон нет (освещение,
# выпрямление по строкам и обложка одной страницы — ``no-illumination-*``, ``no-text-rectify-*`` любой стороны и
# ``cover-both``).
SIDED_FLAGS = frozenset(
    {FrameFlag.IGNORE_LEFT, FrameFlag.IGNORE_RIGHT, FrameFlag.NO_SPLITTING, FrameFlag.COVER_LEFT, FrameFlag.COVER_RIGHT}
)


@dataclass(frozen=True)
class FrameFlags:
    """Содержимое сайдкара кадра: флаги и пояснения к ним; пустой набор — обычный кадр."""

    flags: frozenset[FrameFlag] = field(default_factory=frozenset)
    note: str = ""  # почему так помечено (для человека)
    pair: str = ""  # имя парного кадра в той же папке (вторая съёмка разворота), если есть
    reviewed: str = ""  # дата отсмотра, ISO

    @property
    def empty(self) -> bool:
        """True, если флагов нет (кадр обычный)."""
        return not self.flags

    @property
    def skip_frame(self) -> bool:
        """True, если кадр не обрабатывается вовсе (``ignore-both``)."""
        return FrameFlag.IGNORE_BOTH in self.flags

    @property
    def split(self) -> bool:
        """True, если разворот можно резать на страницы (нет ``no-splitting``)."""
        return FrameFlag.NO_SPLITTING not in self.flags

    def ignored(self, side: Side) -> bool:
        """Игнорируется ли страница стороны ``side``.

        Args:
            side: Сторона разворота.

        Returns:
            True при ``ignore-both`` или флаге игнорирования этой стороны.
        """
        return self.skip_frame or _IGNORE_OF[side] in self.flags

    def equalize(self, side: Side) -> bool:
        """Разрешено ли выравнивать освещение страницы стороны ``side``.

        Args:
            side: Сторона разворота.

        Returns:
            False, если стоит ``no-illumination-<сторона>``.
        """
        return _NO_EQUALIZE_OF[side] not in self.flags

    def text_rectify(self, side: Side) -> bool:
        """Разрешено ли выпрямлять страницу стороны ``side`` по строкам (поле второго преобразования).

        Args:
            side: Сторона разворота.

        Returns:
            False, если стоит ``no-text-rectify-<сторона>`` (передняя обложка, страница-картинка); игнорируемую
            страницу вызывающий проверяет отдельно (``ignored``).
        """
        return _NO_TEXT_RECTIFY_OF[side] not in self.flags

    def cover(self, side: Side) -> bool:
        """Обложка ли страница стороны ``side`` (``cover-both`` или флаг этой стороны).

        Args:
            side: Сторона разворота.

        Returns:
            True для обложки: зона контента равна рамке страницы (обрезки краёв нет). Освещение и выпрямление по
            строкам флаг обложки не выключает — для этого ``no-illumination-*`` и ``no-text-rectify-*``.
        """
        return FrameFlag.COVER_BOTH in self.flags or _COVER_OF[side] in self.flags

    @property
    def single_page(self) -> bool:
        """True, если в кадре одна страница по флагу (``single-page``)."""
        return FrameFlag.SINGLE_PAGE in self.flags

    @property
    def spread(self) -> bool:
        """True, если кадр — разворот по флагу (``spread``): автомат «одна страница» не применяется."""
        return FrameFlag.SPREAD in self.flags

    def label(self) -> str:
        """Флаги одной строкой через ``+`` в порядке enum (для сводок и логов); пусто — ``""``."""
        return "+".join(f.value for f in FrameFlag if f in self.flags)


def sidecar_path(image_path: Path) -> Path:
    """Путь сайдкара кадра: полное имя кадра плюс ``.flags.json``.

    Args:
        image_path: Путь к файлу кадра.

    Returns:
        Путь сайдкара рядом с кадром.
    """
    return image_path.with_name(image_path.name + SIDECAR_SUFFIX)


def normalize_flags(raw: list[str] | tuple[str, ...] | frozenset[FrameFlag]) -> frozenset[FrameFlag]:
    """Строки флагов → проверенный набор ``FrameFlag``.

    ``ignore-left`` вместе с ``ignore-right`` сводятся к ``ignore-both``; при ``ignore-both``
    остальные флаги теряют смысл и отбрасываются. Прежние имена (``LEGACY_FLAGS``) переводятся в нынешние.

    Args:
        raw: Флаги строками (как в сайдкаре или CLI) или уже enum.

    Returns:
        Нормализованный набор флагов.

    Raises:
        FlagsError: неизвестный флаг или ``no-splitting`` вместе с игнорированием одной стороны.
    """
    known = {f.value for f in FrameFlag}
    unknown = sorted(str(r) for r in raw if str(r) not in known and str(r) not in LEGACY_FLAGS)
    if unknown:
        raise FlagsError(f"неизвестные флаги {unknown}; допустимые: {sorted(known)}")
    # Прежние имена (``no-color-equalize-*``) переводятся в нынешние.
    flags = {LEGACY_FLAGS.get(str(r)) or FrameFlag(str(r)) for r in raw}
    # Обе стороны — обложки: одним флагом.
    if {FrameFlag.COVER_LEFT, FrameFlag.COVER_RIGHT} <= flags:
        flags = (flags - {FrameFlag.COVER_LEFT, FrameFlag.COVER_RIGHT}) | {FrameFlag.COVER_BOTH}
    if FrameFlag.COVER_BOTH in flags:
        flags -= {FrameFlag.COVER_LEFT, FrameFlag.COVER_RIGHT}
    # Обе стороны игнорируются — это то же, что весь кадр.
    if {FrameFlag.IGNORE_LEFT, FrameFlag.IGNORE_RIGHT} <= flags:
        flags = {FrameFlag.IGNORE_BOTH}
    if FrameFlag.IGNORE_BOTH in flags:
        return frozenset({FrameFlag.IGNORE_BOTH})
    # «Не резать» и «выбросить одну страницу» друг другу противоречат: неразрезанный разворот
    # пишется одним файлом, выкинуть из него половину нельзя.
    if FrameFlag.NO_SPLITTING in flags and flags & {FrameFlag.IGNORE_LEFT, FrameFlag.IGNORE_RIGHT}:
        raise FlagsError("no-splitting нельзя сочетать с ignore-left/ignore-right")
    # У одной страницы нет сторон: флаги одной стороны и «не резать разворот» к ней не относятся.
    if FrameFlag.SINGLE_PAGE in flags and flags & SIDED_FLAGS:
        raise FlagsError(f"single-page нельзя сочетать с {sorted(f.value for f in flags & SIDED_FLAGS)}")
    if {FrameFlag.SINGLE_PAGE, FrameFlag.SPREAD} <= flags:
        raise FlagsError("single-page и spread противоречат друг другу")
    return frozenset(flags)


def parse_flags(data: dict) -> FrameFlags:
    """Словарь из JSON сайдкара → ``FrameFlags`` с проверкой схемы.

    Args:
        data: Разобранный JSON сайдкара.

    Returns:
        Флаги кадра.

    Raises:
        FlagsError: нет или не та версия схемы, ``flags`` не список, неизвестный флаг, противоречие.
    """
    version = data.get("schema")
    if not isinstance(version, int) or version < 1:
        raise FlagsError(f"нет поля schema или оно не целое ≥ 1: {version!r}")
    if version > SCHEMA_VERSION:
        raise FlagsError(f"schema {version} новее поддерживаемой {SCHEMA_VERSION}: обновите ocr_utils")
    raw = data.get("flags")
    if not isinstance(raw, list) or not raw:
        raise FlagsError("поле flags должно быть непустым списком строк")
    return FrameFlags(
        flags=normalize_flags(raw),
        note=str(data.get("note", "")),
        pair=str(data.get("pair", "")),
        reviewed=str(data.get("reviewed", "")),
    )


def read_flags(image_path: Path) -> FrameFlags:
    """Флаги кадра из его сайдкара; нет сайдкара — пустой набор.

    Args:
        image_path: Путь к файлу кадра.

    Returns:
        Флаги кадра.

    Raises:
        FlagsError: сайдкар есть, но не проходит проверку (с именем файла в тексте).
    """
    path = sidecar_path(image_path)
    if not path.is_file():
        return FrameFlags()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise FlagsError("сайдкар должен быть JSON-объектом")
        return parse_flags(data)
    except (FlagsError, json.JSONDecodeError) as exc:
        raise FlagsError(f"{path}: {exc}") from exc


def to_json(flags: FrameFlags) -> str:
    """``FrameFlags`` → текст сайдкара (флаги в порядке enum, отступ 2, UTF-8 без экранирования).

    Args:
        flags: Флаги кадра.

    Returns:
        Текст JSON с переводом строки в конце.
    """
    data = {
        "schema": SCHEMA_VERSION,
        "flags": [f.value for f in FrameFlag if f in flags.flags],
        "note": flags.note,
        "pair": flags.pair,
        "reviewed": flags.reviewed or dt.date.today().isoformat(),
    }
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_flags(image_path: Path, flags: FrameFlags, replace: bool = False) -> Path:
    """Пишет сайдкар кадра. Файл кадра не трогается; пишется только ``*.flags.json``.

    Args:
        image_path: Путь к существующему файлу кадра.
        flags: Флаги (непустые).
        replace: Перезаписать существующий сайдкар; без него существующий — ошибка.

    Returns:
        Путь записанного сайдкара.

    Raises:
        FileNotFoundError: нет файла кадра.
        FileExistsError: сайдкар уже есть, а ``replace`` не задан.
        FlagsError: пустые флаги.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"нет кадра {image_path}")
    if flags.empty:
        raise FlagsError("пустой набор флагов: обычному кадру сайдкар не нужен")
    path = sidecar_path(image_path)
    # Режим "x" — создать только новый файл; "w" — только при явной перезаписи.
    mode = "w" if replace else "x"
    try:
        with path.open(mode, encoding="utf-8") as fh:
            fh.write(to_json(normalized(flags)))
    except FileExistsError:
        raise FileExistsError(f"сайдкар уже есть: {path} (перезапись — replace)") from None
    return path


def normalized(flags: FrameFlags) -> FrameFlags:
    """Копия ``flags`` с нормализованным набором (см. ``normalize_flags``).

    Args:
        flags: Флаги кадра.

    Returns:
        Те же флаги после сведения ``ignore-left + ignore-right`` и проверки противоречий.
    """
    return FrameFlags(normalize_flags(flags.flags), flags.note, flags.pair, flags.reviewed)


def remove_flags(image_path: Path) -> bool:
    """Удаляет сайдкар кадра, если он есть.

    Args:
        image_path: Путь к файлу кадра.

    Returns:
        True, если сайдкар был и удалён.
    """
    path = sidecar_path(image_path)
    if not path.is_file():
        return False
    path.unlink()
    return True
