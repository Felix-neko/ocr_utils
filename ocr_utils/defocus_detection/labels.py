"""Разбор ручной разметки брака: суффиксы в именах файлов и подпапки.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. Разметку читают три разных потребителя — валидационный свип,
сравнение алгоритмов и калибровка порогов, — и все три обязаны понимать её одинаково.
Правила же неочевидные и добывались разбором конкретной выборки, а не придумывались:
см. «две ловушки» ниже.

КАК ЧЕЛОВЕК РАЗМЕЧАЕТ. Плохой кадр получает суффикс в имени (``_defocus_light``,
``.blurry``) либо переезжает в подпапку ``defocus_for_debugging``. Суффиксы копились
годами и не унифицированы: есть опечатка ``_utralight``, есть два порядка слов
(``_zonal_defocus`` и ``_defocus_zonal``), а в части подшивок степень тяжести не
указана вовсе (``.blurry``).

ДВЕ ЛОВУШКИ, ИЗ-ЗА КОТОРЫХ НАИВНЫЙ ОБХОД ПАПКИ ДАЁТ НЕВЕРНЫЕ ЦИФРЫ.

1. **Пересъёмка.** Файл в ``defocus_for_debugging`` и файл того же номера на верхнем
   уровне — это РАЗНЫЕ кадры: плохой оригинал и снятая заново замена (в выборке СИ они
   расходятся на месяц по EXIF и на тысячи кадров по счётчику затвора). Отсюда следует
   и то, что обходить надо рекурсивно (иначе положительных не окажется вовсе), и то,
   что верхний близнец — не просто «неразмеченный», а ПРОВЕРЕННО хороший: его сняли
   именно потому, что первый забраковали.

2. **Дубль от синхронизации.** Один и тот же снимок может лежать дважды — с меткой и
   без, — если синхронизация облака не довела переименование до конца. Без склейки такой
   кадр попадает одновременно в положительные и в отрицательные. Склеиваем по паре
   «счётчик затвора + время съёмки», метка побеждает её отсутствие. Молча такое не
   чиним: ``find_duplicates`` возвращает найденное, чтобы вызывающий сообщил о проблеме
   и её починили на диске.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

# Подпапка, куда складывают забракованные кадры.
DEBUG_SUBDIR = "defocus_for_debugging"

# Суффиксы разметки, ОБЯЗАТЕЛЬНО от длинного к короткому: "_defocus_light" содержит
# "_defocus" как префикс, и наивная проверка отнесла бы лёгкий расфокус к тяжёлым.
# Порядок в этом кортеже — часть контракта, а не оформление.
LABEL_SUFFIXES = (
    "_defocus_light_motion_blur",
    "_defocus_motion_blur",
    "_zonal_defocus_light",
    "_defocus_ultralight",
    "_defocus_utralight",  # опечатка в исходной разметке, встречается один раз
    "_zonal_defocus",
    "_defocus_zonal",
    "_defocus_light",
    "_cool_focus",
    "_defocus",
    ".blurry",
    "_хп",
)

# Степени тяжести. "unknown" — метка есть, но степень человеком не указана: такие кадры
# годятся для полноты (recall) и AUC, но НЕ для калибровки границ между уровнями.
HEAVY, MEDIUM, LIGHT, UNKNOWN = "heavy", "medium", "light", "unknown"
SEVERITY_ORDER = (HEAVY, MEDIUM, LIGHT, UNKNOWN)

_SEVERITY = {
    "_defocus": HEAVY,
    "_defocus_motion_blur": HEAVY,
    "_zonal_defocus": HEAVY,
    "_defocus_zonal": HEAVY,
    "_defocus_light": MEDIUM,
    "_defocus_light_motion_blur": MEDIUM,
    "_zonal_defocus_light": MEDIUM,
    "_defocus_ultralight": LIGHT,
    "_defocus_utralight": LIGHT,
    ".blurry": UNKNOWN,
}

# Метки, которые НЕ означают брак фокуса.
#   _хп        — часть текста ушла под скотч; к резкости отношения не имеет, кадр
#                выбрасывается совсем: он не положительный и не отрицательный.
#   _cool_focus — наоборот, эталон удачного кадра: проверенно хороший.
IGNORED_SUFFIX = "_хп"
GOOD_SUFFIX = "_cool_focus"

# Шаблон имени кадра в выборке СИ: NNNN_N. Проверяется не ради красоты — если имя не
# разобралось, значит суффикс отрезан неверно, и связать пересъёмку с оригиналом
# по номеру уже нельзя.
FRAME_STEM = re.compile(r"^\d{4}_\d$")


@dataclass(frozen=True)
class Label:
    """Что человек сказал про кадр.

    Attributes:
        suffix: Найденный суффикс ("" — метки нет).
        severity: Степень тяжести из ``SEVERITY_ORDER`` либо None, если кадр не помечен
            как брак.
        motion_blur: Человек указал именно смаз, а не промах фокуса.
        zonal: Человек указал, что мягкая только часть кадра.
        verified_good: Кадр ПРОВЕРЕННО хороший (эталон ``_cool_focus`` либо пересъёмка
            взамен забракованного). Отличается от «просто неразмеченного»: на таком
            кадре ложное срабатывание считается по-настоящему.
        ignored: Кадр вообще не участвует в оценке (брак не по фокусу).
    """

    suffix: str = ""
    severity: str | None = None
    motion_blur: bool = False
    zonal: bool = False
    verified_good: bool = False
    ignored: bool = False

    @property
    def is_bad(self) -> bool:
        """True, если кадр помечен человеком как брак фокуса."""
        return self.severity is not None


@dataclass
class Sample:
    """Кадр выборки вместе с разметкой и связями с другими кадрами.

    Attributes:
        path: Путь к файлу.
        batch: Подшивка (папка), к которой кадр относится; у кадров из
            ``defocus_for_debugging`` — папка-родитель, а не подпапка.
        stem: Номер кадра без суффикса разметки.
        label: Разбор метки.
        reshoot_of: Путь к забракованному оригиналу, если этот кадр — пересъёмка.
        replaced_by: Путь к пересъёмке, если этот кадр забракован и переснят.
    """

    path: Path
    batch: str
    stem: str
    label: Label
    reshoot_of: Path | None = None
    replaced_by: Path | None = None
    extra: dict = field(default_factory=dict)


def split_suffix(name: str) -> tuple[str, str]:
    """Отделяет суффикс разметки от номера кадра.

    Args:
        name: Имя файла с расширением (``0080_2_defocus_light.RAF``).

    Returns:
        Пара (номер кадра, суффикс); суффикс пустой, если метки нет.
    """
    stem = Path(name).stem
    for suffix in LABEL_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], suffix
    return stem, ""


def parse_label(name: str) -> Label:
    """Разбирает метку по имени файла.

    Args:
        name: Имя файла с расширением.

    Returns:
        Разбор метки.
    """
    _, suffix = split_suffix(name)
    if not suffix:
        return Label()
    if suffix == IGNORED_SUFFIX:
        return Label(suffix=suffix, ignored=True)
    if suffix == GOOD_SUFFIX:
        return Label(suffix=suffix, verified_good=True)
    return Label(
        suffix=suffix, severity=_SEVERITY[suffix], motion_blur="motion_blur" in suffix, zonal="zonal" in suffix
    )


def collect(root: Path, suffixes: set[str] | None = None) -> list[Sample]:
    """Собирает выборку из дерева папок, связывая пересъёмки с оригиналами.

    Обход всегда рекурсивный: забракованные кадры лежат в подпапке, и без рекурсии
    подшивка молча остаётся без единого положительного примера.

    Args:
        root: Корень дерева с подшивками.
        suffixes: Расширения файлов в нижнем регистре; None — ``{".raf"}``.

    Returns:
        Список кадров выборки, отсортированный по пути.
    """
    suffixes = suffixes or {".raf"}
    samples: list[Sample] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        stem, _ = split_suffix(path.name)
        # Кадр из defocus_for_debugging относится к подшивке-родителю, а не к подпапке:
        # иначе брак уедет в отдельную «подшивку» и не сойдётся со своей пересъёмкой.
        folder = path.parent.parent if path.parent.name == DEBUG_SUBDIR else path.parent
        samples.append(Sample(path=path, batch=str(folder.relative_to(root)), stem=stem, label=parse_label(path.name)))
    _link_reshoots(samples)
    return samples


def _link_reshoots(samples: list[Sample]) -> None:
    """Связывает забракованные оригиналы с их пересъёмками, правя список на месте.

    Пара опознаётся так: в одной подшивке два кадра с одним номером, один лежит в
    ``defocus_for_debugging`` и помечен браком, другой — на верхнем уровне и не помечен.
    Второй получает статус проверенно хорошего.

    Args:
        samples: Список кадров; изменяется на месте.
    """
    by_key: dict[tuple[str, str], list[Sample]] = {}
    for sample in samples:
        by_key.setdefault((sample.batch, sample.stem), []).append(sample)

    for group in by_key.values():
        if len(group) != 2:
            continue
        in_debug = [s for s in group if s.path.parent.name == DEBUG_SUBDIR]
        on_top = [s for s in group if s.path.parent.name != DEBUG_SUBDIR]
        if len(in_debug) != 1 or len(on_top) != 1:
            continue
        bad, good = in_debug[0], on_top[0]
        if not bad.label.is_bad or good.label.is_bad:
            continue
        bad.replaced_by = good.path
        good.reshoot_of = bad.path
        good.label = Label(suffix=good.label.suffix, verified_good=True)


def find_duplicates(samples: list[Sample], fingerprint: dict[Path, tuple]) -> list[list[Sample]]:
    """Находит группы кадров, которые на самом деле являются одним снимком.

    Отпечаток задаётся снаружи (обычно «модель камеры + счётчик затвора + время съёмки»
    из EXIF), потому что читать EXIF — не дело этого модуля.

    Args:
        samples: Выборка.
        fingerprint: Отображение «путь -> отпечаток кадра»; кадры без отпечатка
            в поиске не участвуют.

    Returns:
        Список групп по два и более кадра с одинаковым отпечатком.
    """
    groups: dict[tuple, list[Sample]] = {}
    for sample in samples:
        key = fingerprint.get(sample.path)
        if key is None:
            continue
        groups.setdefault(key, []).append(sample)
    return [group for group in groups.values() if len(group) > 1]
