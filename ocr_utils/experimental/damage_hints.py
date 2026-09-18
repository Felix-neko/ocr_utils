"""Подсказка о повреждении на страницу из CSV детектора корешка вместо нейтральной фразы «ищи повреждения сам».

Откуда идея: в стенде (`reports/external_ocr_models.md`, разд. 8) ручная подсказка «правый край
срезан корешком» давала 52 честных тега ``<supplied>`` против 0 без неё — модель достраивает
буквы всегда, но помечает достроенное только когда ей сказано, где искать. Подсказки на
полосу были в боевом пакете (``--hints``) и убраны в 259de7e в пользу второго прохода; второй
проход выключен с v16, и источником подсказки мог бы стать наш детектор
``ocr_utils.gutter_loss_detection`` — он смотрит на РАЗВОРОТ и говорит, у какой полосы поле
съедено и насколько (в шагах строк). Здесь его CSV переводится в подсказку на страницу,
а страница узнаётся по имени кадра: ``IMG_0006`` → ``IMG_0006_L`` и ``IMG_0006_R``
(нарезка мини-набора и 1991/02) или ``IMG_0006_1L`` / ``IMG_0006_2R`` (пак-1).

Колонки CSV детектора (``gutter_loss_detection.report``): ``файл``, ``балл`` (0–1), ``вердикт``
(``текст`` / ``таблица`` / ``ок`` / …), ``полосы`` (``L``, ``R``, ``LR``), ``поле_L``, ``поле_R``
(внутреннее поле в шагах строк), ``замечание``.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ocr_utils.external_ocr_services.ocr import PageJob, RunOptions, prompts_for
from ocr_utils.external_ocr_services.prompts import DEFAULT_DAMAGE_NOTE, render
from ocr_utils.external_ocr_services.tiling import PreparedImage, describe


class PageSide(StrEnum):
    """Какая это страница разворота: левая (у корешка её ПРАВЫЙ край) или правая (ЛЕВЫЙ край)."""

    LEFT = "L"
    RIGHT = "R"


# Имя страницы: кадр + суффикс стороны — «IMG_0006_L», «IMG_0006_1L», «0070_R».
_PAGE_NAME = re.compile(r"^(?P<frame>.+?)_(?:\d)?(?P<side>[LR])$")


@dataclass(frozen=True)
class GutterVerdict:
    """Что детектор сказал о развороте: балл, вердикт и поля обеих полос в шагах строк."""

    score: float
    verdict: str
    field_left: float | None  # внутреннее поле левой полосы, шагов строк; None — не измерено
    field_right: float | None


# Ниже этого поля (в шагах строк) полоса считается съеденной; норма у детектора 0.60 на полосу, на
# фото-сканах мини-набора у съеденных и у просто прижатых к сгибу полос поле 0.03–0.05, у пересвета 0.30.
HIDDEN_FIELD_MAX = 0.2


def _float_or_none(value: str) -> float | None:
    """Число из ячейки CSV; пусто или «nan» → ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def read_gutter_csv(csv_path: Path) -> dict[str, GutterVerdict]:
    """CSV детектора → ``{имя кадра без расширения: вердикт}``.

    Args:
        csv_path: ``--csv`` детектора ``gutter_loss_detection``.

    Returns:
        Кадры с измеренным баллом; кадры без балла (одиночные страницы, сбои) пропускаются.
    """
    verdicts: dict[str, GutterVerdict] = {}
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            score = _float_or_none(row.get("балл", ""))
            if score is None:
                continue
            frame = Path(row["файл"]).stem
            verdicts[frame] = GutterVerdict(
                score=score,
                verdict=row.get("вердикт", ""),
                field_left=_float_or_none(row.get("поле_L", "")),
                field_right=_float_or_none(row.get("поле_R", "")),
            )
    return verdicts


def hint_for_page(side: PageSide, verdict: GutterVerdict, threshold: float = 0.35) -> str:
    """Текст подсказки для страницы по вердикту детектора о её развороте (по-английски, как промпт).

    Args:
        side: Левая или правая страница разворота.
        verdict: Что детектор сказал о развороте.
        threshold: Порог балла детектора (его умолчание 0.35).

    Returns:
        Фраза вместо ``DEFAULT_DAMAGE_NOTE``: где корешок у этой страницы, скрыты ли буквы
        (поле у корешка меньше ``HIDDEN_FIELD_MAX`` шагов строк при балле выше порога) или всё
        видно; в обоих случаях — что делать с тегами.
    """
    gutter_edge = "RIGHT" if side is PageSide.LEFT else "LEFT"
    field = verdict.field_left if side is PageSide.LEFT else verdict.field_right
    page_word = "LEFT" if side is PageSide.LEFT else "RIGHT"
    hidden = verdict.score >= threshold and field is not None and field <= HIDDEN_FIELD_MAX
    if hidden:
        return (
            f"This is the {page_word} page of a tightly bound volume. The {gutter_edge} ends of the lines run into the "
            f"binding gutter (inner margin ≈ {field:.1f} line heights): the last one to four letters of many lines "
            "near that edge are hidden or cut off. Restore every such word as the whole word from its visible remains "
            "and the context and mark the restored letters with <supplied>; visible but unreliable letters → <unclear>; "
            "<gap>N</gap> only where no confident reconstruction is possible. Clean print and ordinary end-of-line "
            "hyphenation are not damage."
        )
    return (
        f"This is the {page_word} page of a bound volume; its {gutter_edge} edge is the binding gutter, but the text "
        "there is fully visible: nothing is hidden. Letters near that edge may be squashed or blurred — read them by "
        "shape and context and mark doubtful ones with <unclear>. Do not use <supplied> or <gap> on this page unless "
        "letters are really missing; clean print and ordinary end-of-line hyphenation are not damage."
    )


def hints_from_gutter_csv(csv_path: Path, threshold: float = 0.35) -> dict[str, str]:
    """``{имя страницы: подсказка}`` для обеих страниц каждого измеренного разворота.

    Args:
        csv_path: CSV детектора.
        threshold: Порог балла.

    Returns:
        Ключи — ``<кадр>_L`` и ``<кадр>_R`` (без расширения); страницы пака-1 с суффиксами
        ``_1L``/``_2R`` находит :func:`hint_for_rel` по регулярке имени.
    """
    hints: dict[str, str] = {}
    for frame, verdict in read_gutter_csv(csv_path).items():
        for side in PageSide:
            hints[f"{frame}_{side.value}"] = hint_for_page(side, verdict, threshold)
    return hints


def hint_for_rel(hints: dict[str, str], rel: Path) -> str | None:
    """Подсказка для полосы по её пути: имя файла → кадр и сторона → ключ ``<кадр>_<сторона>``.

    Args:
        hints: Результат :func:`hints_from_gutter_csv`.
        rel: Путь полосы относительно входа (``1991_cut/02/IMG_0648_L.jpg``).

    Returns:
        Подсказка или ``None``, если кадр детектором не измерен или имя не разобрать.
    """
    match = _PAGE_NAME.match(rel.stem)
    if match is None:
        return None
    return hints.get(f"{match.group('frame')}_{match.group('side')}")


class HintedPrompts:
    """Замена ``ocr.prompts_for``: та же сборка промптов, но фраза о повреждениях — из подсказок по полосе.

    Args:
        hints: ``{имя страницы: подсказка}``; полосы без подсказки получают ``DEFAULT_DAMAGE_NOTE``.
    """

    def __init__(self, hints: dict[str, str]):
        self.hints = hints
        self.used = 0  # сколько раз подсказка нашлась — для лога стенда

    def __call__(self, job: PageJob, tiles: list[PreparedImage], options: RunOptions) -> tuple[str, str]:
        """(системный, пользовательский) промпты — как ``prompts_for``, с подсказкой вместо нейтральной фразы.

        Args:
            job: Полоса и её этап (списки выпуска и второй проход — как в боевом коде).
            tiles: Тайлы полосы — из них раскладка для промпта.
            options: Настройки прогона (``source``).

        Returns:
            Пара текстов сообщений.
        """
        hint = hint_for_rel(self.hints, job.rel)
        if hint is None or job.second_pass is not None:
            return prompts_for(job, tiles, options)
        self.used += 1
        system, _ = prompts_for(job, tiles, options)
        info = describe(tiles)
        user = render(
            "user.md.j2",
            ntiles=len(tiles),
            ncols=info.ncols,
            nrows=info.nrows,
            damage_note=hint,
            stage=job.stage,
            toc_kind=job.toc_kind,
            second_pass=None,
            max_lines=60,
            rubrics=list(job.rubrics),
            articles=[dict(a) for a in job.articles],
        )
        assert DEFAULT_DAMAGE_NOTE not in user
        return system, user
