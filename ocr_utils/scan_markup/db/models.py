"""Схема SQLite: пак -> год -> выпуск -> полоса -> (растровые области | печати).

Одна и та же схема обслуживает обе базы — предварительную (результат ``detect``) и
уточнённую (результат ``from-cvat``). Различает их только колонка ``source`` у разметки,
поэтому любой потребитель ниже по конвейеру читает их одним и тем же кодом.

Все координаты разметки — в пикселях ОРИГИНАЛЬНОГО файла, а не уменьшенной копии, которую
видел разметчик в CVAT. Пересчёт делается один раз на импорте (см. ``scan_markup.geometry``),
чтобы каждый потребитель не таскал за собой коэффициент и не ошибался в нём.

Миграций нет: схема заводится ``Base.metadata.create_all``. При изменении моделей базу
пересоздают прогоном ``detect`` — она целиком выводится из файлов на диске.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Значения колонок ``kind`` у растровой области.
KIND_COLOR = "color"
KIND_GRAYSCALE = "grayscale"
RASTER_KINDS = (KIND_COLOR, KIND_GRAYSCALE)

# Значения колонки ``kind`` у маски. Пока заводится только печать, но метка «Рукописная
# надпись» в CVAT уже есть (docker/bootstrap.py), и когда её начнут размечать, хватит
# нового значения вместо новой таблицы с той же структурой.
MASK_LIBRARY_STAMP = "library_stamp"

# Значения колонок ``source``: чем поставлена разметка.
SOURCE_AUTO = "auto"
SOURCE_CVAT = "cvat"


def _utcnow() -> datetime:
    """Текущее время в UTC. Отдельной функцией — чтобы подменяться в тестах."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Pack(Base):
    """Пак сканов — папка вида ``.../Готовое/пак-1``, в CVAT ей отвечает проект."""

    __tablename__ = "packs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    root_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    cvat_project_id: Mapped[int | None] = mapped_column(Integer, default=None)

    year_packages: Mapped[list["YearPackage"]] = relationship(
        back_populates="pack", cascade="all, delete-orphan", order_by="YearPackage.name"
    )


class YearPackage(Base):
    """Годовой комплект — подпапка пака вида ``1974``, в CVAT ей отвечает задача."""

    __tablename__ = "year_packages"
    __table_args__ = (UniqueConstraint("pack_id", "name", name="uq_year_in_pack"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pack_id: Mapped[int] = mapped_column(ForeignKey("packs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    year: Mapped[int | None] = mapped_column(Integer, default=None)
    rel_path: Mapped[str] = mapped_column(Text)
    cvat_task_id: Mapped[int | None] = mapped_column(Integer, default=None)

    pack: Mapped[Pack] = relationship(back_populates="year_packages")
    issues: Mapped[list["Issue"]] = relationship(
        back_populates="year_package", cascade="all, delete-orphan", order_by="Issue.name"
    )


class Issue(Base):
    """Выпуск — подпапка года вида ``05``, в CVAT ему отвечает джоб.

    Имя хранится как есть, а не нормализованным числом: в паке-1 встречаются пересканы
    ``05 (2)``, ``06 (2)``, ``10 (2)``, и по одному лишь номеру они схлопнулись бы с
    основным выпуском.
    """

    __tablename__ = "issues"
    __table_args__ = (UniqueConstraint("year_package_id", "name", name="uq_issue_in_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    year_package_id: Mapped[int] = mapped_column(ForeignKey("year_packages.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    number: Mapped[int | None] = mapped_column(Integer, default=None)
    rel_path: Mapped[str] = mapped_column(Text)
    cvat_job_id: Mapped[int | None] = mapped_column(Integer, default=None)

    year_package: Mapped[YearPackage] = relationship(back_populates="issues")
    pages: Mapped[list["Page"]] = relationship(
        back_populates="issue", cascade="all, delete-orphan", order_by="Page.order_index"
    )


class Page(Base):
    """Полоса выпуска — конкретный файл скана вместе с параметрами его уменьшения.

    Про параметры уменьшения. ``divisor`` = ``round(dpi / CVAT_DPI)``: 600 dpi -> 8,
    450 dpi -> 6. Перед уменьшением кадр обрезается справа и снизу до размера, кратного
    ``divisor`` (``crop_width`` x ``crop_height``), и только потом делится — тогда масштаб
    ровно 1:divisor и обратный пересчёт разметки точен на всей ширине кадра. Без обрезки
    3492 -> 436 дало бы 8.0092, и промах рос бы тем сильнее, чем правее объект.

    Все три величины хранятся, а не выводятся из ``dpi`` на лету: ``dpi`` может быть не
    записан в теге файла и подставлен из ``--default-dpi``, а импорт разметки обязан
    пересчитывать координаты ровно тем же коэффициентом, каким они были получены.

    Про отпечаток файла. ``file_hash`` — sha256 содержимого (см. ``scan_markup.hashing``),
    ``file_size`` и ``file_mtime`` — дешёвый признак, позволяющий не перечитывать
    неизменившийся файл. ``cvat_file_hash`` — тот же хеш, но снятый В МОМЕНТ ЗАЛИВКИ полосы
    в CVAT. Две колонки, а не одна, именно потому, что вопрос стоит не «менялся ли файл
    когда-нибудь», а «показывает ли CVAT сейчас то, что лежит на диске»: расхождение этих
    двух значений и есть список полос, чья разметка больше не относится к делу.
    """

    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("issue_id", "file_name", name="uq_page_in_issue"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    issue_id: Mapped[int] = mapped_column(ForeignKey("issues.id", ondelete="CASCADE"), index=True)
    file_name: Mapped[str] = mapped_column(String(255))
    rel_path: Mapped[str] = mapped_column(Text)
    order_index: Mapped[int] = mapped_column(Integer)

    width: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[int | None] = mapped_column(Integer, default=None)
    dpi: Mapped[int | None] = mapped_column(Integer, default=None)

    divisor: Mapped[int | None] = mapped_column(Integer, default=None)
    crop_width: Mapped[int | None] = mapped_column(Integer, default=None)
    crop_height: Mapped[int | None] = mapped_column(Integer, default=None)

    file_size: Mapped[int | None] = mapped_column(Integer, default=None)
    file_mtime: Mapped[float | None] = mapped_column(Float, default=None)
    file_hash: Mapped[str | None] = mapped_column(String(64), index=True, default=None)
    hash_algo: Mapped[str | None] = mapped_column(String(16), default=None)
    cvat_file_hash: Mapped[str | None] = mapped_column(String(64), default=None)

    cvat_rel_path: Mapped[str | None] = mapped_column(Text, default=None)
    cvat_width: Mapped[int | None] = mapped_column(Integer, default=None)
    cvat_height: Mapped[int | None] = mapped_column(Integer, default=None)
    cvat_frame: Mapped[int | None] = mapped_column(Integer, default=None)

    detected_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    issue: Mapped[Issue] = relationship(back_populates="pages")
    raster_regions: Mapped[list["RasterRegion"]] = relationship(back_populates="page", cascade="all, delete-orphan")
    masks: Mapped[list["MaskAnnotation"]] = relationship(back_populates="page", cascade="all, delete-orphan")


class RasterRegion(Base):
    """Прямоугольник растрового изображения на полосе, координаты — ОРИГИНАЛА.

    ``chroma_frac`` пишется всегда, даже когда ``kind`` уже проставлен: это доля
    хроматичных пикселей, по которой ``kind`` и получен. Без неё перекалибровать порог
    color/grayscale по 12 тысячам полос значило бы второй проход по полутерабайту
    оригиналов с медленного диска.
    """

    __tablename__ = "raster_regions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), index=True)

    x1: Mapped[int] = mapped_column(Integer)
    y1: Mapped[int] = mapped_column(Integer)
    x2: Mapped[int] = mapped_column(Integer)
    y2: Mapped[int] = mapped_column(Integer)

    kind: Mapped[str] = mapped_column(String(16))
    full_page: Mapped[bool] = mapped_column(Boolean, default=False)
    chroma_frac: Mapped[float | None] = mapped_column(Float, default=None)

    source: Mapped[str] = mapped_column(String(16), default=SOURCE_AUTO)
    cvat_shape_id: Mapped[int | None] = mapped_column(Integer, default=None)

    page: Mapped[Page] = relationship(back_populates="raster_regions")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class MaskAnnotation(Base):
    """Битовая маска на полосе (пока — только библиотечная печать).

    Хранится ровно в том виде, в каком её отдаёт CVAT: серия длин пробегов (RLE) плюс
    охватывающий прямоугольник. Отличие одно — всё пересчитано в разрешение ОРИГИНАЛА,
    поэтому читается без коэффициентов::

        from cvat_sdk.masks import decode_mask
        points = [*map(int, row.rle.split(",")), row.left, row.top,
                  row.left + row.width - 1, row.top + row.height - 1]
        mask = decode_mask(points, image_width=page.width, image_height=page.height)

    ``source_divisor`` фиксирует, что маска рисовалась на копии 1/divisor, то есть реальная
    точность её краёв — divisor пикселей оригинала (6-8). Потребителю под LaMa это важно:
    маску всё равно надо дилатировать, и знать зернистость полезно.
    """

    __tablename__ = "mask_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), index=True)

    kind: Mapped[str] = mapped_column(String(32), default=MASK_LIBRARY_STAMP)

    left: Mapped[int] = mapped_column(Integer)
    top: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    rle: Mapped[str] = mapped_column(Text)

    source_divisor: Mapped[int | None] = mapped_column(Integer, default=None)
    source: Mapped[str] = mapped_column(String(16), default=SOURCE_CVAT)
    cvat_shape_id: Mapped[int | None] = mapped_column(Integer, default=None)

    page: Mapped[Page] = relationship(back_populates="masks")
