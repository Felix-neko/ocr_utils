"""Схема SQLite: пак -> год -> выпуск -> полоса -> (прямоугольные области | маски | точки).

Одна и та же схема обслуживает обе базы — предварительную (результат ``detect``) и
уточнённую (результат ``from-cvat``). Различает их только колонка ``source`` у разметки,
поэтому любой потребитель ниже по конвейеру читает их одним и тем же кодом.

Все координаты разметки — в пикселях ОРИГИНАЛЬНОГО файла, а не уменьшенной копии, которую
видел разметчик в CVAT. Пересчёт делается один раз на импорте (см. ``scan_markup.geometry``),
чтобы каждый потребитель не таскал за собой коэффициент и не ошибался в нём.

Про изменение схемы. Полноценных миграций нет, но и пересоздавать базу нельзя: в ней уже
лежит ручная разметка из CVAT, которой на диске больше нигде нет. Поэтому новые колонки
объявляются здесь ДОПУСКАЮЩИМИ NULL, а дописывает их в существующие таблицы
``db.session.add_missing_columns`` при каждом открытии базы. Колонка NOT NULL и любое
переименование — это уже настоящая миграция с заполнением, автоматически они не проедут.

ПЕРЕИМЕНОВАНИЯ делает отдельный скрипт ``db.migrate``, запускаемый руками, а не открытие
базы. Разница принципиальная: дописать пустую колонку безопасно при любом исходе, а
переименование либо прошло, либо нет, и делать его украдкой посреди чужого прогона нельзя.
``db.session.open_db`` старую схему только РАСПОЗНАЁТ и отказывается открывать, называя
команду миграции.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Значения колонок ``kind`` у растровой области.
KIND_COLOR = "color"
KIND_GRAYSCALE = "grayscale"

# Не иллюстрация, а ПОДОЗРЕНИЕ НА БИБЛИОТЕЧНУЮ ПЕЧАТЬ: цветной штриховой рисунок, слишком
# мелкий, чтобы быть картинкой. Печать — это цветной штрих по определению (фиолетовая
# мастика), и от цветного рисунка её отличает только размер: замер по паку-1 дал у печатей
# 2.1 и 2.2% полосы против почти 100% у синего рисунка обложки.
#
# Отдельный ``kind``, а не маска: у нас есть только прямоугольник, а метка «Библиотечная
# печать» в CVAT имеет тип mask, и прямоугольник под неё не положить. Отдельный ``kind``
# заодно не даёт вырезать печать в PDF как картинку — потребители фильтруют по color и
# grayscale. Печати всё равно предстоит закрашивать через LaMa, и знать заранее, где они
# стоят, полезно.
KIND_STAMP_SUSPECT = "stamp_suspect"

# Область, набранная ЦВЕТНОЙ КРАСКОЙ: заголовок, поздравление, список — не растр и не
# рисунок, а обычный типографский набор синим или красным (пак-1: 1976/12 0020_1L —
# новогоднее обращение целиком синим, 0500_2R — анонс издательства).
#
# Бинаризовать такое нельзя: цвет пропадёт. Пока эти области просто сохраняются картинкой,
# как растр, — отсюда участие в ``PICTURE_KINDS``. Отдельный ``kind``, а не ``color``,
# потому что это РЕШЕНИЕ, а не наблюдение: когда-нибудь цветной текст захочется вынести в
# отдельный слой PDF, и тогда его надо будет отличить от настоящих фотографий.
#
# Автодетектора у него нет: ставится только руками, меткой «Цветной текст или штрих».
KIND_COLOR_TEXT = "color_text"

# Виды, которые ставит РАСТРОВЫЙ детектор (``detection``) и которые означают «здесь не
# набор, а картинка или оттиск»: только их он и заменяет при пересчёте полосы.
RASTER_KINDS = (KIND_COLOR, KIND_GRAYSCALE, KIND_STAMP_SUSPECT, KIND_COLOR_TEXT)

# Таблица с линейками и блок-схема / штриховой рисунок (line art). Ставит детектор таблиц
# ``table_detection`` (тот же этап ``detect``, но своя версия и свои колонки у полосы —
# см. ``Page.table_detector_version``). В PDF картинкой не вырезаются: это набор, его
# распознают, а рамка нужна, чтобы обойтись с таблицей и схемой иначе, чем со сплошным
# текстом (не распрямлять строки, не расширять графы схемы). Оба вида — прямоугольники в
# ``rect_regions`` рядом с растром, потому что структура у них та же, а в CVAT это те же
# rectangle-метки, которые разметчик уточняет тем же инструментом.
#
# «Схема» и «рисунок» детектора (блок-схема с текстом в коробках против сетки графика или
# чертежа) в базе — один вид: лечат их одинаково, а тонкий вид лежит в ``detector_info``.
KIND_TABLE = "table"
KIND_LINE_ART_SCHEMA = "line_art_schema"

TABLE_KINDS = (KIND_TABLE, KIND_LINE_ART_SCHEMA)

# Все виды прямоугольников, которые вообще бывают в ``rect_regions``.
RECT_KINDS = RASTER_KINDS + TABLE_KINDS

# Типы, которые действительно означают ИЛЛЮСТРАЦИЮ: их вырезают из оригинала и вклеивают
# в PDF. ``KIND_STAMP_SUSPECT`` сюда не входит намеренно.
PICTURE_KINDS = (KIND_COLOR, KIND_GRAYSCALE, KIND_COLOR_TEXT)

# Какие из них сохраняются В ЦВЕТЕ. Список нужен именно списком: потребитель, который
# написал бы ``color если kind == KIND_COLOR иначе серый``, молча обесцветил бы цветной
# текст — ровно то, ради чего его и выделяли.
COLOR_PICTURE_KINDS = (KIND_COLOR, KIND_COLOR_TEXT)

# Значения колонки ``kind`` у маски. Всё это — объекты ПОД УДАЛЕНИЕ: разметчик обводит их
# кистью, а закрашивает потом LaMa. Разные виды разнесены значениями одной колонки, а не
# отдельными таблицами: структура у них одна и та же (RLE плюс охватывающий прямоугольник),
# и добавление вида не стоит ни новой таблицы, ни миграции.
MASK_LIBRARY_STAMP = "library_stamp"
MASK_HANDWRITING = "handwriting"
MASK_OTHER_REMOVAL = "other_removal"

MASK_KINDS = (MASK_LIBRARY_STAMP, MASK_HANDWRITING, MASK_OTHER_REMOVAL)

# Значения колонки ``kind`` у точки. Точка — это не объект, а МЕСТО: куда вставить экслибрис.
POINT_EXLIBRIS = "exlibris"

# Значения колонок ``source``: чем поставлена разметка.
SOURCE_AUTO = "auto"
SOURCE_CVAT = "cvat"


def _utcnow() -> datetime:
    """Текущее время в UTC. Отдельной функцией — чтобы подменяться в тестах."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Pack(Base):
    """Пак сканов — папка вида ``.../Готовое/пак-1``, в CVAT ей отвечает проект.

    Корней у пака несколько, потому что один и тот же комплект полос живёт на диске в
    нескольких видах, и путь к каждому нужен разным шагам конвейера::

        source_pics_root    нарезанные сканы до разметки (бывший root_path)
        cleaned_pics_root   после scan_cleanup: печати закрашены, фон размыт
        sharpened_text_pics_root   после Capture One: усилены детали текста
        full_intermediate_pdf_root                  промежуточные PDF, все полосы
        pages_with_pics_only_intermediate_pdf_root  промежуточные PDF, только с растром
        final_pdfs_root     собранные PDF с текстовым слоем

    Обязателен только первый: остальные появляются по мере прохождения конвейера, и до
    своего шага честно пусты. Хранятся они здесь, а не в аргументах запуска, ровно затем,
    зачем в базе лежит всё остальное: следующий шаг должен уметь узнать, откуда брать
    вход, не полагаясь на память запускающего.
    """

    __tablename__ = "packs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    source_pics_root: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    cvat_project_id: Mapped[int | None] = mapped_column(Integer, default=None)

    cleaned_pics_root: Mapped[str | None] = mapped_column(Text, default=None)
    sharpened_text_pics_root: Mapped[str | None] = mapped_column(Text, default=None)
    full_intermediate_pdf_root: Mapped[str | None] = mapped_column(Text, default=None)
    pages_with_pics_only_intermediate_pdf_root: Mapped[str | None] = mapped_column(Text, default=None)
    final_pdfs_root: Mapped[str | None] = mapped_column(Text, default=None)

    # Какие повороты вообще рассматривать на этом паке — углы по часовой через запятую,
    # например «0,90,270». NULL значит «умолчание» (``rotation.DEFAULT_ALLOWED``).
    #
    # Набор задаётся паком, а не прогоном, ровно затем же, зачем здесь лежат корни: шаг
    # конвейера должен уметь узнать условия, не полагаясь на память запускающего. Сужение
    # набора — самый дешёвый способ поднять точность: ответ, которого в наборе нет, не может
    # быть дан в принципе, а арбитру достаётся меньше прогонов распознавания.
    allowed_rotations: Mapped[str | None] = mapped_column(String(32), default=None)

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

    # Имена собранных по выпуску PDF — без директории: где лежит каждый вид, знает пак
    # (см. его корни). Пусто, пока соответствующий шаг не отработал.
    #
    # ``pages_with_pics_only_intermediate_pdf_name`` остаётся пустым НАВСЕГДА у выпуска,
    # в котором не размечено ни одной картинки: такой PDF ему просто не из чего собрать.
    # Отличить «ещё не собирали» от «нечего собирать» по одной этой колонке нельзя —
    # для этого есть ``full_intermediate_pdf_name``, который заполняется всегда.
    full_intermediate_pdf_name: Mapped[str | None] = mapped_column(String(255), default=None)
    pages_with_pics_only_intermediate_pdf_name: Mapped[str | None] = mapped_column(String(255), default=None)
    final_pdf_name: Mapped[str | None] = mapped_column(String(255), default=None)

    # Переопределение пакового набора допустимых поворотов для этого выпуска. NULL —
    # обычный случай: берётся паковый. Нужно на выпуск, где вёрстка отличается от остального
    # пака (скажем, номер целиком набран альбомными таблицами).
    allowed_rotations: Mapped[str | None] = mapped_column(String(32), default=None)

    # Поля, добавленные полосам при сборке ПОЛНОЙ промежуточной PDF, в миллиметрах.
    # Они нужны затем, что распрямление строк в FineReader увеличивает кадр и обрезает
    # всё, что вылезло за MediaBox; поле даёт ему куда расти. Пишутся по факту, уже с
    # округлением до размера MCU, — а не то, что просили ключами командной строки.
    #
    # Хранятся не как одно число, а по всем четырём сторонам: поля заведомо разные по
    # горизонтали и вертикали (вылет вверх у FineReader почти постоянный, вправо — зависит
    # от перекоса полосы), и однажды может понадобиться и несимметричное поле.
    #
    # ЭТО ПРО ПОЛНУЮ PDF. В PAGES_WITH_PICS_ONLY полей нет: её распознают без распрямления,
    # геометрия там должна остаться ровно такой, в какой размечались иллюстрации.
    #
    # None означает «выпуск этим прогоном не собирался» — в том числе пропущен как уже
    # готовый. Ноль означает «собирался без полей».
    full_intermediate_pdf_margin_left_mm: Mapped[float | None] = mapped_column(Float, default=None)
    full_intermediate_pdf_margin_right_mm: Mapped[float | None] = mapped_column(Float, default=None)
    full_intermediate_pdf_margin_top_mm: Mapped[float | None] = mapped_column(Float, default=None)
    full_intermediate_pdf_margin_bottom_mm: Mapped[float | None] = mapped_column(Float, default=None)

    year_package: Mapped[YearPackage] = relationship(back_populates="issues")
    pages: Mapped[list["Page"]] = relationship(
        back_populates="issue", cascade="all, delete-orphan", order_by="Page.order_index"
    )


class Page(Base):
    """Полоса выпуска — конкретный файл скана вместе с параметрами его уменьшения.

    Про параметры уменьшения. Их заполняет шаг ``to-cvat`` (а ``detect`` пишет только то,
    что прочитал из файла: ``width``, ``height``, ``dpi``) — тогда разрешение разметки можно
    сменить опцией ``--cvat-dpi``, не перечитывая пак заново.
    ``divisor`` = ``round(dpi / cvat_dpi)``: 600 dpi -> 8,
    450 dpi -> 6. Перед уменьшением кадр обрезается справа и снизу до размера, кратного
    ``divisor`` (``crop_width`` x ``crop_height``), и только потом делится — тогда масштаб
    ровно 1:divisor и обратный пересчёт разметки точен на всей ширине кадра. Без обрезки
    3492 -> 436 дало бы 8.0092, и промах рос бы тем сильнее, чем правее объект.

    Все три величины хранятся, а не выводятся из ``dpi`` на лету: ``dpi`` может быть не
    записан в теге файла и подставлен из ``--default-dpi``, а импорт разметки обязан
    пересчитывать координаты ровно тем же коэффициентом, каким они были получены. По той же
    причине полосе, уже залитой в CVAT, ``to-cvat`` делитель не меняет, каким бы ни был
    ``--cvat-dpi``: её разметка нарисована в прежнем масштабе.

    Про отпечаток файла. ``file_hash`` — sha256 содержимого (см. ``scan_markup.hashing``),
    ``file_size`` и ``file_mtime`` — дешёвый признак, позволяющий не перечитывать
    неизменившийся файл. ``cvat_file_hash`` — тот же хеш, но снятый В МОМЕНТ ЗАЛИВКИ полосы
    в CVAT. Две колонки, а не одна, именно потому, что вопрос стоит не «менялся ли файл
    когда-нибудь», а «показывает ли CVAT сейчас то, что лежит на диске»: расхождение этих
    двух значений и есть список полос, чья разметка больше не относится к делу.
    """

    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("issue_id", "source_file_name", name="uq_page_in_issue"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    issue_id: Mapped[int] = mapped_column(ForeignKey("issues.id", ondelete="CASCADE"), index=True)
    source_file_name: Mapped[str] = mapped_column(String(255))
    source_rel_path: Mapped[str] = mapped_column(Text)
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

    # Очищенная полоса (после ``scan_cleanup``: печати закрашены, фон размыт). Имя
    # отличается от исходного отпечатком — ``IMG_0034_1L.a1b2c3d4.tif``, см.
    # ``scan_cleanup.naming``, — и хранится, а не выводится: расширение зависит от
    # ``--output-format`` прогона, и вывести его задним числом уже неоткуда.
    cleaned_file_name: Mapped[str | None] = mapped_column(String(255), default=None)
    cleaned_rel_path: Mapped[str | None] = mapped_column(Text, default=None)
    # Сохранена ли очищенная полоса одним серым каналом. Цвет остаётся только там, где он
    # размечен (``COLOR_PICTURE_KINDS``), и потребителю полезно знать это заранее, не
    # открывая файл.
    cleaned_grayscale: Mapped[bool | None] = mapped_column(Boolean, default=None)

    # Заострённая копия полосы (Capture One усилил детали текста). Имя отличается от
    # ``source_file_name`` расширением — .jpg против .tif, — поэтому хранится, а не
    # выводится подстановкой суффикса: следующий пак может выгружаться иначе.
    sharpened_text_pic_file_name: Mapped[str | None] = mapped_column(String(255), default=None)
    sharpened_text_pic_rel_path: Mapped[str | None] = mapped_column(Text, default=None)

    # Номера страниц полосы в промежуточных PDF, с нуля. Нужны на обратном ходе: после
    # FineReader из распознанных PDF собирается финальная, и единственный способ понять,
    # какая её страница отвечает какой полосе, — это записанный при сборке номер.
    #
    # ``pages_with_pics_only_pdf_page_idx`` пуст у полосы, которой в PDF типа
    # PAGES_WITH_PICS_ONLY нет, то есть у полосы без размеченных картинок.
    full_pdf_page_idx: Mapped[int | None] = mapped_column(Integer, default=None)
    pages_with_pics_only_pdf_page_idx: Mapped[int | None] = mapped_column(Integer, default=None)

    # --- Ориентация полосы ---------------------------------------------------
    # На сколько повернуть полосу ПО ЧАСОВОЙ, чтобы стало прямо: 0, 90, 180 или 270.
    #
    # NULL и 0 — РАЗНОЕ. NULL значит «не считали», 0 — «считали, поворот не нужен». Слить их
    # в одно нельзя: тогда непосчитанная полоса выдала бы себя за проверенную, и прогон с
    # ``--skip-detected`` больше никогда бы к ней не вернулся.
    #
    # Речь о СОДЕРЖИМОМ, а не о кадре: полоса как страница ориентирована правильно, боком
    # напечатана сама иллюстрация — генплан, оргсхема, широкая таблица. Пропорции кадра тут
    # не говорят ничего, обе такие полосы книжные.
    rotate_cw: Mapped[int | None] = mapped_column(Integer, default=None)
    orientation_confidence: Mapped[float | None] = mapped_column(Float, default=None)
    # Версия набора детекторов ориентации (``orientation.ORIENTATION_VERSION``). Отдельная
    # от ``detector_version`` намеренно: слив их в одну, правка порога ориентации заставила
    # бы перечитать весь пак вместе со всей растровой детекцией — полтерабайта и часы.
    orientation_version: Mapped[int | None] = mapped_column(Integer, default=None)
    orientation_detected_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    # ``auto`` — посчитано детектором, ``cvat`` — поставлено или снято человеком в CVAT.
    # По этому полю видно, чему верить при расхождении.
    orientation_source: Mapped[str | None] = mapped_column(String(16), default=None)

    # Угол, с которым полоса РЕАЛЬНО записана очисткой. Отличается от ``rotate_cw``, пока
    # очистка не прогонялась после смены решения. Нужен потребителям ниже по конвейеру:
    # ``width``/``height`` описывают ОРИГИНАЛ, а файл на диске может быть повёрнут.
    cleaned_rotate_cw: Mapped[int | None] = mapped_column(Integer, default=None)

    detected_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    # Версия алгоритма детекции (``detection.DETECTOR_VERSION``), которой получена разметка
    # этой полосы. Без неё ``--skip-detected`` пропустил бы весь пак после правки детектора:
    # отпечаток файла отвечает на вопрос «менялся ли файл», а не «менялся ли алгоритм».
    detector_version: Mapped[int | None] = mapped_column(Integer, default=None)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    # --- Таблицы и блок-схемы -------------------------------------------------
    # Своя версия и своё время, как у ориентации, и по той же причине: детектор таблиц
    # правится отдельно от растрового, и его правка не должна заставлять перечитывать пак
    # ради растра — и наоборот. NULL значит «таблицы на этой полосе не искали».
    table_detector_version: Mapped[int | None] = mapped_column(Integer, default=None)
    tables_detected_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    # --- Оглавление -----------------------------------------------------------
    # Два признака, а не один «вид»: в CVAT им отвечают два тега («Оглавление» и «Годовой
    # указатель»), и разметчик ставит их независимо. ``is_toc`` — полоса «Содержания»
    # выпуска, ``is_year_index`` — полоса указателя статей за год (декабрьские номера).
    # NULL — не искали (различает ``toc_version``), False — искали и не нашли.
    # ``toc_score`` — сила решения (1.0 — сильный признак, меньше — по строкам с номерами
    # страниц), ``toc_source`` — ``auto`` или ``cvat``: ручное решение прогон не затирает.
    # Версия и время — свои, по той же причине, что у ориентации и таблиц.
    is_toc: Mapped[bool | None] = mapped_column(Boolean, default=None)
    is_year_index: Mapped[bool | None] = mapped_column(Boolean, default=None)
    toc_score: Mapped[float | None] = mapped_column(Float, default=None)
    toc_source: Mapped[str | None] = mapped_column(String(16), default=None)
    toc_version: Mapped[int | None] = mapped_column(Integer, default=None)
    toc_detected_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    # ВЕТО ЧЕЛОВЕКА: полоса точно не оглавление и не указатель, что бы ни сказали детектор или
    # внешняя модель. Внешний OCR (``ocr_utils.external_ocr_services``) просит модель проверять
    # каждую полосу «не оглавление ли это», и на таблицах-перечнях она поднимает ложную тревогу;
    # этот признак её гасит. Отдельная колонка, а не ``is_toc = False``: False там означает лишь
    # «тега нет», и от него ложная тревога модели ничем не отличается. В CVAT — тег «Не оглавление»,
    # ставится только руками; NULL — не ставился.
    force_is_not_toc: Mapped[bool | None] = mapped_column(Boolean, default=None)

    issue: Mapped[Issue] = relationship(back_populates="pages")
    rect_regions: Mapped[list["RectRegion"]] = relationship(back_populates="page", cascade="all, delete-orphan")
    masks: Mapped[list["MaskAnnotation"]] = relationship(back_populates="page", cascade="all, delete-orphan")
    points: Mapped[list["PointAnnotation"]] = relationship(back_populates="page", cascade="all, delete-orphan")


class RectRegion(Base):
    """Прямоугольная область на полосе, координаты — ОРИГИНАЛА.

    Вид области — колонка ``kind``: растр (``RASTER_KINDS``: цветная или серая картинка,
    подозрение на печать, цветной текст) либо таблица и блок-схема (``TABLE_KINDS``). Одна
    таблица на все виды, потому что структура одна — четыре координаты, источник, шейп
    CVAT, — а разные семейства различаются только тем, кто их ставит и как ими пользуются
    ниже по конвейеру. Раньше таблица звалась ``raster_regions``; переименована, когда в
    ней появились таблицы, — ``db.migrate`` переименовывает её в старых базах.

    Измерения растра пишутся всегда, даже когда ``kind`` уже проставлен: перекалибровать порог по
    12 тысячам полос иначе значило бы второй проход по полутерабайту оригиналов с
    медленного диска. Их три, и они не заменяют друг друга:

    * ``chroma_spread`` — разброс хроматичности области, ``hypot(std(a), std(b))`` в Lab
      после снятия налёта бумаги. ПО НЕЙ И ПРИНИМАЕТСЯ решение color/grayscale;
    * ``chroma_self_frac`` — доля пикселей, чей оттенок отличается от собственной медианы
      области. Подстраховка для маленькой СПЛОШНОЙ цветной плашки: у неё разброс мал;
    * ``chroma_frac`` — доля хроматичных пикселей в абсолютном выражении. Это метрика
      ПРОШЛОГО алгоритма, по ней размечены первые 777 областей. Считается и пишется дальше
      только затем, чтобы старое решение можно было сравнить с новым, не перечитывая пак.

    ``dot_frac`` — доля «точечных» связных компонент краски внутри области (см.
    ``detection.dots``).

    ``mid_frac``, ``tone_entropy``, ``screen_peak`` — признаки «растр или штрих» из
    ``detection.tone``: доля средних тонов, энтропия гистограммы и выступ пика растровой
    сетки в спектре. Решение принимает их КОНЪЮНКЦИЯ, по отдельности ни один не разделяет.
    Пишутся всегда, даже когда правило не сработало: по ним пороги детектора
    перекалибровываются без чтения оригиналов.

    ``ink_contrast`` — «бумага минус краска» внутри области. В решении растр/штрих не
    участвует: по нему бледный оттиск библиотечной печати отличается от чёрной виньетки.

    У таблиц и схем растровые измерения пусты, зато заполнен ``detector_info`` — JSON с
    тем, что знает о находке детектор таблиц: тонкий вид (``схема``/``рисунок``), балл,
    наклон линеек, признаки решётки и проверки. Тоже «измерения пишутся всегда»: по ним
    пороги детектора перекалибровываются без чтения оригиналов.
    """

    __tablename__ = "rect_regions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), index=True)

    x1: Mapped[int] = mapped_column(Integer)
    y1: Mapped[int] = mapped_column(Integer)
    x2: Mapped[int] = mapped_column(Integer)
    y2: Mapped[int] = mapped_column(Integer)

    kind: Mapped[str] = mapped_column(String(16))
    full_page: Mapped[bool] = mapped_column(Boolean, default=False)
    chroma_frac: Mapped[float | None] = mapped_column(Float, default=None)
    chroma_spread: Mapped[float | None] = mapped_column(Float, default=None)
    chroma_self_frac: Mapped[float | None] = mapped_column(Float, default=None)
    dot_frac: Mapped[float | None] = mapped_column(Float, default=None)
    mid_frac: Mapped[float | None] = mapped_column(Float, default=None)
    tone_entropy: Mapped[float | None] = mapped_column(Float, default=None)
    screen_peak: Mapped[float | None] = mapped_column(Float, default=None)
    ink_contrast: Mapped[float | None] = mapped_column(Float, default=None)

    # JSON детектора таблиц; у растровых областей пуст.
    detector_info: Mapped[str | None] = mapped_column(Text, default=None)

    source: Mapped[str] = mapped_column(String(16), default=SOURCE_AUTO)
    cvat_shape_id: Mapped[int | None] = mapped_column(Integer, default=None)

    page: Mapped[Page] = relationship(back_populates="rect_regions")

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


class PointAnnotation(Base):
    """Точка на полосе, координаты — ОРИГИНАЛА. Пока единственный вид — место экслибриса.

    Отдельная таблица, а не вырожденный прямоугольник в ``rect_regions`` и не маска из
    одного пикселя: точка отвечает на другой вопрос. Прямоугольник и маска говорят «вот
    объект, вот его границы», а точка — «вот МЕСТО, куда положить свой знак». Границ у неё
    нет вовсе, и хранить их нулями значило бы врать потребителю.

    ``source_divisor`` — как и у маски, зернистость: точка ставилась на копии 1/divisor,
    поэтому её положение известно с точностью до divisor пикселей оригинала (6-8).
    """

    __tablename__ = "point_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), index=True)

    kind: Mapped[str] = mapped_column(String(32), default=POINT_EXLIBRIS)

    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)

    source_divisor: Mapped[int | None] = mapped_column(Integer, default=None)
    source: Mapped[str] = mapped_column(String(16), default=SOURCE_CVAT)
    cvat_shape_id: Mapped[int | None] = mapped_column(Integer, default=None)

    page: Mapped[Page] = relationship(back_populates="points")
