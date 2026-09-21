"""Сверка заголовков и рубрик собранного выпуска с оглавлением: на статью — один `#`, на рубрику — один `<rubric>`.

Модель на полосе-продолжении подставляет название статьи из списка в промпте (по рубрике в
колонтитуле, по самому колонтитулу или просто из списка), и в целиковом тексте выпуска у статьи
оказывается два-три `#` — при нарезке по статьям такой фантом режет её пополам. По одной полосе
фантом не отличить от настоящего заголовка (все они «из списка»), поэтому сверка идёт по выпуску:
для каждой статьи «Содержания» (``toc.json``) собираются все `#` с её id, настоящим считается тот,
что стоит на странице с номером из оглавления (напечатанным или выведенным по соседям —
``numbering``), остальные — фантомы: переписанный колонтитул или заголовок сверху полосы-продолжения
удаляется, напечатанный текст (лозунг под фото) понижается до `##`. Статья без `#` вовсе, но с
названием в колонтитуле по мнению модели (v20: ``running_header_article_id``) — заголовок
восстанавливается из колонтитула. Рубрики — так же: `<rubric>` перед `#` первой статьи раздела
настоящий, остальные с тем же id — колонтитул (удалить) или напечатанный маркер (`<marker>`);
рубрика, ушедшая в колонтитул целиком (шапка раздела в «Плановом хозяйстве» стоит там, где на
следующих полосах колонтитул), восстанавливается перед `#` первой статьи раздела. Перед настоящим
`#` и `<rubric>` ставится HTML-комментарий с id — по нему нарезка находит статью в md.

Модуль не знает о блоках сборки: на входе тексты абзацев выпуска, принадлежность полосам и данные
полос, на выходе — правки по индексам (:class:`BlockEdits`), которые применяет ``assemble``.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from ocr_utils.external_ocr_services.schema import Stage, StructureTag, TocArticle
from ocr_utils.external_ocr_services.structure import _H1, _MARKER_TEXT, _RUBRIC, header_key
from ocr_utils.external_ocr_services.toc import IssueToc, best_title_match, normalize_title, title_matches

# Заголовок списка на полосе оглавления/указателя — единственный законный `#` там.
_TOC_HEADING = re.compile(r"содержание|оглавление|указатель|перечень", re.IGNORECASE)
_COMMENT = re.compile(r"^<!--")
# Рубрика и маркер перед `#` — не «содержательные» блоки: `#` за ними всё ещё стоит сверху полосы.
_BEFORE_HEADING = re.compile(rf"^<(?:{StructureTag.RUBRIC}|{StructureTag.MARKER})>|^<!--")


def article_comment(article_id: str) -> str:
    """HTML-комментарий перед настоящим `#` статьи: ``<!-- article A3 -->``."""
    return f"<!-- article {article_id} -->"


def rubric_comment(rubric_id: str) -> str:
    """HTML-комментарий перед настоящим `<rubric>`: ``<!-- rubric R2 -->``."""
    return f"<!-- rubric {rubric_id} -->"


@dataclass
class PageRefs:
    """Что сверка знает о полосе: id в заголовках и колонтитулах по ответу модели, номер, этап.

    Args:
        file: Полоса (путь без суффикса) — для отчёта.
        stage: Этап, ответ которого вошёл в текст (``page`` / ``toc``).
        number: Номер страницы (напечатанный или выведенный, ``numbering``); ``None`` — неизвестен.
        headings: ``[{"text", "article_id"}]`` полосы (``PageResult.headings``).
        rubrics: ``[{"text", "rubric_id"}]`` полосы (``PageResult.rubrics``).
        running_header: Колонтитул сверху по ответу модели.
        running_footer: Колонтитул снизу.
        header_article_id: Id статьи, названной в верхнем колонтитуле (``running_header_article_id``).
        header_rubric_id: Id рубрики в верхнем колонтитуле.
        footer_article_id: То же для нижнего.
        footer_rubric_id: То же для нижнего.
        toc_merged: Блок ``<toc>`` полосы слит с предыдущей (полоса-продолжение оглавления).
        toc_kind: Вид полосы по ответу модели (``contents`` / ``index`` / ``none``) — по нему две
            подряд полосы оглавления одного вида считаются одним списком.
    """

    file: str
    stage: str = Stage.PAGE.value
    number: int | None = None
    headings: list[dict] = field(default_factory=list)
    rubrics: list[dict] = field(default_factory=list)
    running_header: str | None = None
    running_footer: str | None = None
    header_article_id: str | None = None
    header_rubric_id: str | None = None
    footer_article_id: str | None = None
    footer_rubric_id: str | None = None
    toc_merged: bool = False
    toc_kind: str = "none"

    @property
    def header_keys(self) -> set[str]:
        """Ключи сравнения текста с колонтитулами полосы (без цифр, регистра и пунктуации)."""
        return {header_key(text) for text in (self.running_header, self.running_footer) if text}

    def header_text(self, title: str, article_id: str | None = None, rubric_id: str | None = None) -> str | None:
        """Текст колонтитула, где модель нашла эту статью или рубрику, без номера страницы; ``None`` — нет.

        Id мало: модель ставит id статьи и колонтитулу с одной лишь рубрикой (единственная статья
        раздела — «Экономическое образование кадров» 1974/02, 1974/05), и по нему заголовок встал бы
        посреди статьи с текстом рубрики. Поэтому текст колонтитула ещё должен быть похож на само
        название (``title_matches``: равенство, вхождение, похожесть ≥ 0.75).

        Args:
            title: Название статьи или рубрики из оглавления, на которое должен быть похож колонтитул.
            article_id: Искомая статья.
            rubric_id: Искомая рубрика.
        """
        for text, a_id, r_id in (
            (self.running_header, self.header_article_id, self.header_rubric_id),
            (self.running_footer, self.footer_article_id, self.footer_rubric_id),
        ):
            if text and ((article_id and a_id == article_id) or (rubric_id and r_id == rubric_id)):
                cleaned = " ".join(re.sub(r"\d+", " ", text).split())
                if cleaned and title_matches(cleaned, [title]):
                    return cleaned
        return None


@dataclass
class BlockEdits:
    """Правки блоков выпуска по индексам исходного списка — применяет ``assemble``.

    Args:
        delete: Индексы блоков, которые убираются.
        replace: Новый текст блока по индексу (понижение `#` → `##`, `<rubric>` → `<marker>`).
        insert_before: Блоки, вставляемые перед блоком с индексом: ``[(текст, метка | None)]``; метка
            (``("article", "A3")`` / ``("rubric", "R2")``) — чтобы после правок найти вставленный блок.
        anchors: Метки настоящих заголовков и рубрик, оставшихся на месте: ``[(метка, индекс блока)]``
            (у рубрики, чей раздел идёт в выпуске дважды, — две записи).
    """

    delete: set[int] = field(default_factory=set)
    replace: dict[int, str] = field(default_factory=dict)
    insert_before: dict[int, list[tuple[str, tuple[str, str] | None]]] = field(default_factory=dict)
    anchors: list[tuple[tuple[str, str], int]] = field(default_factory=list)

    def insert(self, index: int, text: str, label: tuple[str, str] | None = None, first: bool = False) -> None:
        """Добавить вставку перед блоком.

        Args:
            index: Перед каким блоком.
            text: Текст нового блока.
            label: Метка вставленного блока или ``None``.
            first: Поставить раньше уже запланированных вставок перед тем же блоком (рубрика — перед заголовком).
        """
        items = self.insert_before.setdefault(index, [])
        items.insert(0, (text, label)) if first else items.append((text, label))


@dataclass
class ReconcileReport:
    """Итог сверки для sidecar выпуска.

    Args:
        articles: По статье оглавления: ``{"id", "title", "toc_page", "status", "page", "page_number", "text"}``;
            ``status``: ``found`` — `#` на месте, ``restored`` — вставлен из колонтитула, ``missing`` — нет.
        rubrics: То же по рубрикам.
        phantom_headings: Лишние `#`: ``{"article_id", "page", "text", "action": deleted | demoted}``.
        phantom_rubrics: Лишние `<rubric>`: ``{"rubric_id", "page", "text", "action": deleted | marker}``.
        toc_page_headings: `#` на полосах оглавления, которые не заголовок списка: ``{"page", "text", "action"}``.
    """

    articles: list[dict] = field(default_factory=list)
    rubrics: list[dict] = field(default_factory=list)
    phantom_headings: list[dict] = field(default_factory=list)
    phantom_rubrics: list[dict] = field(default_factory=list)
    toc_page_headings: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Поля для sidecar как есть."""
        return {
            "articles": self.articles,
            "rubrics": self.rubrics,
            "phantom_headings": self.phantom_headings,
            "phantom_rubrics": self.phantom_rubrics,
            "toc_page_headings": self.toc_page_headings,
        }


@dataclass(frozen=True)
class HeadingHint:
    """Подсказка вспомогательного запроса по статье без заголовка: блок с названием (или ``None``) и вердикт для sidecar."""

    block: int | None
    verdict: dict


# Запись оглавления, которую искать в тексте бессмысленно: выходные данные, принятые за статью.
_IMPRINT = re.compile(r"подписано к печати|сдано в набор|формат \d+\s*[×x]\s*\d+|тираж \d|заказ \d", re.IGNORECASE)


def skip_reason(article: TocArticle, articles: list[TocArticle]) -> str | None:
    """Почему статью без заголовка не стоит искать дальше: мусор оглавления.

    Args:
        article: Статья.
        articles: Все статьи оглавления по порядку (для проверки «хвост предыдущей записи»).

    Returns:
        Причина или ``None``, если запись похожа на настоящую статью.
    """
    title = article.title.strip()
    if _IMPRINT.search(title):
        return "выходные данные приняты за статью"
    # Хвост названия предыдущей статьи, разрезанного переносом на две записи: со строчной буквы, без авторов.
    index = next((i for i, item in enumerate(articles) if item.id == article.id), None)
    if index and title and title[0].islower() and not article.authors:
        return f"хвост названия предыдущей записи ({articles[index - 1].id})"
    return None


def candidate_lines(
    blocks: list[str],
    floating: list[bool],
    plain: list[bool],
    page_of: list[int],
    pages: list[PageRefs],
    toc_page: int | None,
    limit: int,
) -> list[tuple[int, bool, str]]:
    """Строки-кандидаты для вспомогательного запроса по статье без заголовка.

    В окне «страница статьи ±1» — заголовки `##`, жирные/курсивные абзацы, маркеры и первые два абзаца
    каждой полосы; без окна (номер не выведен) — только заголовочные строки со всех полос. Готовые
    `#` не в счёт — они принадлежат другим статьям.

    Args:
        blocks: Тексты блоков.
        floating: Плавающие ли блоки.
        plain: Простой ли текст.
        page_of: Полоса каждого блока.
        pages: Данные полос.
        toc_page: Страница статьи по оглавлению; ``None`` — окна нет.
        limit: Потолок строк.

    Returns:
        ``[(индекс блока, в окне ли, текст без разметки)]`` по порядку выпуска.
    """
    found: list[tuple[int, bool, str]] = []
    seen_plain: dict[int, int] = {}  # полоса → сколько простых абзацев уже взято
    for index, text in enumerate(blocks):
        page = pages[page_of[index]]
        in_window = toc_page is not None and page.number is not None and abs(page.number - toc_page) <= 1
        if toc_page is not None and not in_window:
            continue
        if _H1.match(text) or (floating[index] and not _MARKER_TEXT.match(text)):
            continue
        marker = _MARKER_TEXT.match(text)
        heading_like = marker is not None or bool(re.match(r"^(?:#{2,6}\s|\*\*|\*|_|<u>)", text.strip()))
        if not heading_like:
            if toc_page is None or not plain[index] or seen_plain.get(page_of[index], 0) >= 2:
                continue
            seen_plain[page_of[index]] = seen_plain.get(page_of[index], 0) + 1
        cleaned = marker.group(1) if marker is not None else strip_title_markup(text)
        found.append((index, in_window or toc_page is None, cleaned[:LINE_MAX_CHARS]))
        if len(found) >= limit:
            break
    return found


def _restore_from_hint(
    blocks: list[str], page_of: list[int], pages: list[PageRefs], article_id: str, block: int, edits: BlockEdits
) -> tuple[int, str, int, str] | None:
    """Абзац, указанный вспомогательным запросом, становится `#` статьи.

    Args:
        blocks: Тексты блоков.
        page_of: Полоса каждого блока.
        pages: Данные полос.
        article_id: Статья.
        block: Индекс блока по подсказке.
        edits: Правки — сюда пишется замена и якорь.

    Returns:
        ``(индекс полосы, текст, индекс блока, "model")`` или ``None``, если блок уже тронут или не годится.
    """
    if block >= len(blocks) or block in edits.delete or block in edits.replace or _H1.match(blocks[block]):
        return None
    marker = _MARKER_TEXT.match(blocks[block])
    text = marker.group(1) if marker is not None else strip_title_markup(blocks[block])
    edits.replace[block] = f"# {text}"
    edits.anchors.append((("article", article_id), block))
    return page_of[block], text, block, "model"


@dataclass
class _Candidate:
    """`#` или `<rubric>` в тексте выпуска, отнесённый к статье или рубрике."""

    block: int  # индекс блока
    page: int  # индекс полосы
    text: str  # текст заголовка или рубрики


def _ref_id(text: str, refs: list[dict], id_key: str, titles: list[str], ids: list[str | None]) -> str | None:
    """Id статьи или рубрики для текста из тела: по записи полосы с тем же текстом (если её id
    согласован с названием в оглавлении), иначе по названию.

    Args:
        text: Текст `#` или `<rubric>`.
        refs: ``headings`` / ``rubrics`` полосы.
        id_key: ``article_id`` / ``rubric_id``.
        titles: Названия из оглавления по порядку.
        ids: Их id в том же порядке.
    """
    key = normalize_title(text)
    title_of = {ref_id: title for ref_id, title in zip(ids, titles) if ref_id}
    for ref in refs:
        ref_id = ref.get(id_key)
        if (
            ref_id in title_of
            and normalize_title(ref.get("text") or "") == key
            and title_matches(text, [title_of[ref_id]])
        ):
            return ref_id
    index = best_title_match(text, titles)
    return ids[index] if index is not None else None


def _first_plain_after(floating: list[bool], plain: list[bool], start: int, end: int) -> bool:
    """Есть ли простой текстовый блок (текст статьи) после блока ``start`` до ``end`` (не включая).

    Args:
        floating: Плавающие ли блоки.
        plain: Простой ли текст.
        start: Блок, после которого смотрим.
        end: Граница (не включая).
    """
    return any(plain[i] and not floating[i] for i in range(start + 1, end))


def _at_page_top(blocks: list[str], floating: list[bool], page_start: int, index: int) -> bool:
    """Стоит ли блок сверху полосы: перед ним на полосе только плавающие блоки, рубрики, маркеры и комментарии.

    Args:
        blocks: Тексты блоков выпуска.
        floating: Плавающие ли блоки.
        page_start: Индекс первого блока полосы.
        index: Проверяемый блок.
    """
    return all(floating[i] or _BEFORE_HEADING.match(blocks[i]) for i in range(page_start, index))


def _choose_primary(candidates: list[_Candidate], pages: list[PageRefs], toc_page: int | None) -> _Candidate:
    """Настоящий заголовок среди кандидатов: на странице из оглавления, иначе ближайший к ней, иначе первый.

    Args:
        candidates: `#` статьи по порядку выпуска (непустой список).
        pages: Данные полос (номера).
        toc_page: Страница статьи по оглавлению; ``None`` — не указана.
    """
    if toc_page is None:
        return candidates[0]
    numbered = [
        (abs(pages[c.page].number - toc_page), i, c)
        for i, c in enumerate(candidates)
        if pages[c.page].number is not None
    ]
    if not numbered:
        return candidates[0]
    return min(numbered)[2]


def reconcile_headings(
    blocks: list[str],
    plain: list[bool],
    floating: list[bool],
    page_of: list[int],
    pages: list[PageRefs],
    toc: IssueToc | None,
    hints: dict[str, HeadingHint] | None = None,
) -> tuple[BlockEdits, ReconcileReport]:
    """Сверить `#` и `<rubric>` выпуска с оглавлением и подготовить правки.

    Args:
        blocks: Тексты абзацев выпуска по порядку (после склеек стыков).
        plain: Простой ли текст каждый блок (годится для сшивания — значит, это текст статьи).
        floating: Плавающий ли блок (сноска, таблица, иллюстрация, подпись).
        page_of: Индекс полосы для каждого блока.
        pages: Данные полос по порядку.
        toc: «Содержание» выпуска с id; ``None`` — сверять не с чем, правятся только полосы оглавления.
        hints: Подсказки вспомогательного запроса (``heading_check``): id статьи → блок с её названием
            и вердикт; применяются к статьям, у которых ни `#`, ни восстановления из тела и
            колонтитула. ``None`` — без подсказок.

    Returns:
        ``(правки, отчёт)``.
    """
    hints = hints or {}
    edits = BlockEdits()
    report = ReconcileReport()
    page_start: dict[int, int] = {}  # полоса → индекс её первого блока
    page_end: dict[int, int] = {}  # полоса → индекс за её последним блоком
    for index, page in enumerate(page_of):
        page_start.setdefault(page, index)
        page_end[page] = index + 1

    # Полосы оглавления: единственный законный `#` — заголовок списка, и только на первой из подряд
    # идущих полос одного вида (второй полосе «Содержания» модель дописывает `# СОДЕРЖАНИЕ`, которого
    # на ней нет, а `continues_previous` ставит нестрого); на слитой полосе-продолжении он лишний тем более.
    for index, text in enumerate(blocks):
        match = _H1.match(text)
        page_index = page_of[index]
        page = pages[page_index]
        if match is None or page.stage != Stage.TOC.value:
            continue
        previous = pages[page_index - 1] if page_index > 0 else None
        continuation = page.toc_merged or (
            previous is not None and previous.stage == Stage.TOC.value and previous.toc_kind == page.toc_kind
        )
        if continuation and _TOC_HEADING.search(match.group(1)):
            edits.delete.add(index)
            report.toc_page_headings.append({"page": page.file, "text": match.group(1), "action": "deleted"})
        elif not _TOC_HEADING.search(match.group(1)):
            edits.replace[index] = "#" + text
            report.toc_page_headings.append({"page": page.file, "text": match.group(1), "action": "demoted"})
    if toc is None:
        return edits, report

    articles = toc.articles
    titles = [a.title for a in articles]
    article_ids = [a.id for a in articles]
    rubric_entries = toc.rubric_entries
    rubric_titles = [r["title"] for r in rubric_entries]
    rubric_ids = [r["id"] for r in rubric_entries]

    # 1. Кандидаты: каждый уцелевший `#` → статья по id или названию. Заголовок годового указателя на
    #    полосе оглавления тоже кандидат («Указатель статей…» в «Содержании» — статья), но только если
    #    название начинается с него: иначе `# СОДЕРЖАНИЕ` уходил лекции «Планирование… Содержание…»
    #    по вхождению слова (1973/06).
    by_article: dict[str, list[_Candidate]] = {}
    for index, text in enumerate(blocks):
        match = _H1.match(text)
        if match is None or index in edits.delete or index in edits.replace:
            continue
        page_index = page_of[index]
        page = pages[page_index]
        article_id = _ref_id(match.group(1), page.headings, "article_id", titles, article_ids)
        if article_id is None:
            continue
        if page.stage == Stage.TOC.value:
            title = toc.article_by_id(article_id).title
            if not normalize_title(title).startswith(normalize_title(match.group(1))):
                continue
        by_article.setdefault(article_id, []).append(_Candidate(index, page_index, match.group(1)))

    # 2. По статье: настоящий `#` по странице оглавления, фантомы — удалить или понизить, нет — восстановить.
    primary_of: dict[str, int] = {}  # id статьи → индекс блока настоящего `#` (у восстановленного — куда вставлен)
    phantom_headings: list[tuple[int, dict]] = []  # (индекс блока, запись) — в отчёт по порядку выпуска
    phantom_rubrics: list[tuple[int, dict]] = []
    for article in articles:
        toc_page = _int_or_none(article.page)
        candidates = by_article.get(article.id, [])
        entry = {"id": article.id, "title": article.title, "toc_page": toc_page}
        if candidates:
            primary = _choose_primary(candidates, pages, toc_page)
            primary_of[article.id] = primary.block
            edits.anchors.append((("article", article.id), primary.block))
            page = pages[primary.page]
            entry.update(status="found", page=page.file, page_number=page.number, text=primary.text)
            for candidate in candidates:
                if candidate is primary:
                    continue
                page = pages[candidate.page]
                start, end = page_start[candidate.page], page_end[candidate.page]
                unprinted = header_key(candidate.text) in page.header_keys or (
                    _at_page_top(blocks, floating, start, candidate.block)
                    and page.number != toc_page
                    and _first_plain_after(floating, plain, candidate.block, end)
                )
                if unprinted:
                    edits.delete.add(candidate.block)
                    action = "deleted"
                else:
                    edits.replace[candidate.block] = "#" + blocks[candidate.block]
                    action = "demoted"
                phantom_headings.append(
                    (
                        candidate.block,
                        {"article_id": article.id, "page": page.file, "text": candidate.text, "action": action},
                    )
                )
        else:
            # Сначала по телу страницы статьи (название набрано жирным или `##`), потом по колонтитулу.
            restored = _restore_from_body(
                blocks, floating, page_of, pages, page_start, page_end, article.id, article.title, toc_page, edits
            )
            if restored is None:
                from_header = _restore_heading(
                    blocks, floating, page_of, pages, page_start, page_end, article.id, article.title, toc_page, edits
                )
                restored = (*from_header, "running_header") if from_header is not None else None
            hint = hints.get(article.id)
            if restored is None and hint is not None and hint.block is not None:
                # Подсказка модели: указанный абзац становится `#` (текст без разметки, маркер — текст тега).
                restored = _restore_from_hint(blocks, page_of, pages, article.id, hint.block, edits)
                entry["model"] = hint.verdict
            elif hint is not None:
                entry["model"] = hint.verdict
            if restored is not None:
                page_index, text, at, source = restored
                primary_of[article.id] = at
                page = pages[page_index]
                entry.update(status="restored", source=source, page=page.file, page_number=page.number, text=text)
            else:
                skip = skip_reason(article, articles)
                if skip is not None:
                    entry.update(status="skipped", reason=skip, page=None, page_number=None, text=None)
                else:
                    entry.update(status="missing", page=None, page_number=None, text=None)
        report.articles.append(entry)

    # 3. Рубрики: кандидаты по id, настоящий — перед `#` первой статьи раздела.
    by_rubric: dict[str, list[_Candidate]] = {}
    for index, text in enumerate(blocks):
        match = _RUBRIC.match(text)
        if match is None:
            continue
        page_index = page_of[index]
        rubric_id = _ref_id(match.group(1), pages[page_index].rubrics, "rubric_id", rubric_titles, rubric_ids)
        if rubric_id is not None:
            by_rubric.setdefault(rubric_id, []).append(_Candidate(index, page_index, match.group(1)))
    for rubric in rubric_entries:
        rubric_id = rubric["id"]
        # `#` первых статей всех разделов с этой рубрикой (рубрика «ИНФОРМАЦИЯ» может идти дважды).
        heads = [
            primary_of[section.articles[0].id]
            for section in toc.sections
            if section.id == rubric_id and section.articles and section.articles[0].id in primary_of
        ]
        candidates = by_rubric.get(rubric_id, [])
        primaries: list[_Candidate] = []
        for head in heads:
            before = [c for c in candidates if c.block < head and c not in primaries]
            if not before:
                continue
            # Непосредственно перед `#` — своя; иначе ближайшая выше, если между ней и `#` нет текста
            # статьи (шапка раздела над `#` через маркер или иллюстрацию).
            nearest = before[-1]
            if nearest.block == head - 1 or not _first_plain_after(floating, plain, nearest.block, head):
                primaries.append(nearest)
        entry = {"id": rubric_id, "title": rubric["title"]}
        if not primaries and candidates and not heads:
            primaries = [candidates[0]]  # статьи раздела без `#` — рубрике некуда встать, первый тег остаётся
        if primaries:
            for primary in primaries:
                edits.anchors.append((("rubric", rubric_id), primary.block))
            page = pages[primaries[0].page]
            entry.update(status="found", page=page.file, page_number=page.number, text=primaries[0].text)
        elif heads:
            restored = _rubric_from_marker(
                blocks, floating, page_of, page_start, heads[0], rubric_id, rubric["title"], edits
            )
            if restored is None:
                restored = _restore_rubric(pages, page_of, heads[0], rubric_id, rubric["title"], edits)
            if restored is not None:
                page = pages[page_of[heads[0]]]
                entry.update(status="restored", page=page.file, page_number=page.number, text=restored)
            else:
                entry.update(status="missing", page=None, page_number=None, text=None)
        else:
            entry.update(status="missing", page=None, page_number=None, text=None)
        for candidate in candidates:
            if candidate in primaries:
                continue
            page = pages[candidate.page]
            if header_key(candidate.text) in page.header_keys:
                edits.delete.add(candidate.block)
                action = "deleted"
            else:
                edits.replace[candidate.block] = f"<{StructureTag.MARKER}>*{candidate.text}*</{StructureTag.MARKER}>"
                action = "marker"
            phantom_rubrics.append(
                (candidate.block, {"rubric_id": rubric_id, "page": page.file, "text": candidate.text, "action": action})
            )
        report.rubrics.append(entry)
    report.phantom_headings = [item for _, item in sorted(phantom_headings, key=lambda pair: pair[0])]
    report.phantom_rubrics = [item for _, item in sorted(phantom_rubrics, key=lambda pair: pair[0])]

    # 4. Комментарии с id перед настоящими `#` и `<rubric>`, оставшимися на месте.
    for (kind, ref_id), index in edits.anchors:
        comment = article_comment(ref_id) if kind == "article" else rubric_comment(ref_id)
        edits.insert(index, comment)
    return edits, report


def _restore_heading(
    blocks: list[str],
    floating: list[bool],
    page_of: list[int],
    pages: list[PageRefs],
    page_start: dict[int, int],
    page_end: dict[int, int],
    article_id: str,
    title: str,
    toc_page: int | None,
    edits: BlockEdits,
) -> tuple[int, str, int] | None:
    """Статья без `#`: полоса, где модель увидела её название в колонтитуле, получает заголовок.

    Из полос с этой статьёй в колонтитуле берётся страница статьи по оглавлению, иначе ближайшая к
    ней по номеру, без номеров — первая (модель ставит id статьи и по колонтитулу с одной лишь
    рубрикой, так что первая попавшаяся полоса может быть далеко от начала статьи). На той полосе
    `#` без id (`#`, не отнесённый ни к одной статье) становится её заголовком; иначе заголовок
    вставляется из текста колонтитула сверху полосы (после плавающих блоков, рубрик и маркеров).

    Args:
        blocks: Тексты блоков.
        floating: Плавающие ли блоки.
        page_of: Полоса каждого блока.
        pages: Данные полос.
        page_start: Первый блок каждой полосы.
        page_end: Конец (не включая) блоков каждой полосы.
        article_id: Статья.
        title: Её название из оглавления — колонтитул должен быть на него похож.
        toc_page: Её страница по оглавлению; ``None`` — не указана.
        edits: Правки — сюда пишется вставка или якорь.

    Returns:
        ``(индекс полосы, текст заголовка, индекс блока)`` — блок самого `#` или блок, перед которым он
        вставлен; ``None``, если колонтитула с этой статьёй нет.
    """
    found = [
        (page_index, text)
        for page_index, page in enumerate(pages)
        if page_index in page_start and (text := page.header_text(title, article_id=article_id)) is not None
    ]
    if toc_page is not None:
        numbered = [(abs(pages[i].number - toc_page), i, text) for i, text in found if pages[i].number is not None]
        if numbered:
            found = [min(numbered)[1:]]
    for page_index, text in found[:1]:
        start, end = page_start[page_index], page_end[page_index]
        for index in range(start, end):
            match = _H1.match(blocks[index])
            if match is not None and index not in edits.delete and index not in edits.replace:
                edits.anchors.append((("article", article_id), index))
                return page_index, match.group(1), index
        # Первое место на полосе, где может стоять заголовок: после плавающих блоков, рубрик и маркеров.
        at = start
        while at < end and (floating[at] or _BEFORE_HEADING.match(blocks[at])):
            at += 1
        edits.insert(at, article_comment(article_id), None)
        edits.insert(at, f"# {text}", ("article", article_id))
        return page_index, text, at
    return None


# Название, набранное не заголовком первого уровня: `##`–`######`, жирным (`**…**`, `__…__`),
# курсивом (`*…*`, `_…_`), подчёркнутым (`<u>…</u>`) — часто с пометкой «Тема:» (лекции раздела
# «Экономическое образование кадров» в МТС: `**Тема: Ленинские принципы…**`).
_BODY_TITLE_MARKUP = re.compile(r"^(?:#{2,6}\s+|\*{1,2}|_{1,2}|<u>)|(?:\*{1,2}|_{1,2}|</u>)$", re.IGNORECASE)
_TOPIC_PREFIX = re.compile(r"^Тема\s*\d*\s*[:.]\s*", re.IGNORECASE)  # «Тема:», «Тема.», «Тема 11.»
BODY_TITLE_MAX_CHARS = 300
LINE_MAX_CHARS = 200  # строка-кандидат для вспомогательного запроса длиннее не бывает названием


def _body_title(text: str) -> str | None:
    """Абзац как возможное название статьи: без разметки заголовка/жирного/курсива/подчёркивания; ``None`` — не похоже.

    Пометка «Тема:» остаётся в тексте — заголовок в md должен быть как напечатан; для сопоставления
    с оглавлением её снимает :func:`_title_key`.

    Args:
        text: Абзац тела.
    """
    stripped = text.strip()
    # Абзац с тегом в начале (рубрика, автор, сноска, таблица, комментарий) — не название; `<u>` — можно.
    if "\n" in stripped or len(stripped) > BODY_TITLE_MAX_CHARS or re.match(r"<(?!u>)", stripped, re.IGNORECASE):
        return None
    return strip_title_markup(stripped) or None


def strip_title_markup(text: str) -> str:
    """Снять с абзаца разметку заголовка/жирного/курсива/подчёркивания — текст названия как напечатан.

    Args:
        text: Абзац тела (одной строкой).
    """
    cleaned = " ".join(text.split())
    # Разметка снимается по слою: `**Тема:** …` → «Тема: …»; вложенное (жирный курсив) — тоже.
    for _ in range(3):
        cleaned = _BODY_TITLE_MARKUP.sub("", cleaned).strip()
    # Пометка, обёрнутая отдельно (`**Тема 9.** Название`): «Тема 9.** Название» → снять хвост разметки.
    return re.sub(r"^(Тема\s*\d*\s*[:.])\s*(?:\*{1,2}|_{1,2})\s*", r"\1 ", cleaned, flags=re.IGNORECASE)


def _title_key(text: str) -> str:
    """Текст названия для сопоставления с оглавлением: без пометки «Тема:».

    Args:
        text: Название по :func:`_body_title`.
    """
    return _TOPIC_PREFIX.sub("", text).strip()


# Похожесть, с которой абзац в окне «страница статьи по оглавлению ±1» принимается за её название
# (после того как строгий `title_matches` — равенство, вхождение, ≥ 0.75 — не сработал): «НА ОДНОМ
# ИЗ ГЛАВНЫХ НАПРАВЛЕНИЙ» ↔ «На главном направлении» — 0.72 (1973/11). Ниже — только модели.
BODY_TITLE_MIN_RATIO = 0.6
# Второй по похожести кандидат на той же полосе должен отставать хотя бы на столько — иначе выбор
# неоднозначен и абзац оставляется вспомогательному запросу.
BODY_TITLE_MIN_GAP = 0.1


@dataclass(frozen=True)
class _BodyCandidate:
    """Абзац полосы, годный в название: индекс, текст без разметки, похожесть на название, из тега ли."""

    index: int
    text: str
    ratio: float
    from_tag: bool  # `<marker>`/`<rubric>` — раздел, который оглавление считает статьёй


def _body_candidates(
    blocks: list[str], floating: list[bool], start: int, end: int, title: str, edits: BlockEdits
) -> list[_BodyCandidate]:
    """Абзацы полосы, похожие на название статьи: заголовки, жирный/курсив и маркеры.

    Args:
        blocks: Тексты блоков.
        floating: Плавающие ли блоки.
        start: Первый блок полосы.
        end: Конец (не включая) блоков полосы.
        title: Название из оглавления.
        edits: Правки — блоки, уже удалённые или заменённые, не в счёт.

    Returns:
        Кандидаты по убыванию похожести; готовые `#` (кандидаты других статей) не входят.
    """
    key = normalize_title(title)
    found: list[_BodyCandidate] = []
    for index in range(start, end):
        if index in edits.delete or index in edits.replace or _H1.match(blocks[index]):
            continue
        # `<rubric>` не кандидат: тег уже принадлежит разделу перед `#` другой статьи; маркер — да
        # (шапка раздела, которую оглавление считает статьёй, а на полосе за ней нет `#`); маркер —
        # плавающий блок, остальные плавающие (сноски, таблицы, иллюстрации) в названия не годятся.
        tag = _MARKER_TEXT.match(blocks[index])
        if tag is None and floating[index]:
            continue
        text = tag.group(1) if tag is not None else _body_title(blocks[index])
        if text is None:
            continue
        candidate_key = normalize_title(_title_key(text))
        if not candidate_key:
            continue
        if title_matches(_title_key(text), [title]):
            ratio = 1.0
        else:
            ratio = difflib.SequenceMatcher(None, candidate_key, key).ratio()
        if ratio >= BODY_TITLE_MIN_RATIO:
            found.append(_BodyCandidate(index, text, ratio, tag is not None))
    return sorted(found, key=lambda c: -c.ratio)


def _restore_from_body(
    blocks: list[str],
    floating: list[bool],
    page_of: list[int],
    pages: list[PageRefs],
    page_start: dict[int, int],
    page_end: dict[int, int],
    article_id: str,
    title: str,
    toc_page: int | None,
    edits: BlockEdits,
) -> tuple[int, str, int, str] | None:
    """Статья без `#`: на её странице по оглавлению (или ±1) абзац с её названием становится `#`.

    Модель не делает `#` из названия, набранного не как заголовок (лекции «Экономического образования
    кадров»: «**Тема: …**» под названием курса), пишет его с другой формулировкой («НА ОДНОМ ИЗ ГЛАВНЫХ
    НАПРАВЛЕНИЙ» при «На главном направлении» в оглавлении) или маркером раздела («ИНФОРМАЦИЯ»,
    «Наш стол справок» — оглавление считает раздел статьёй). Ищется только в окне страницы статьи:
    сначала полосы с её номером, затем ±1 — иначе похожий абзац другой статьи стал бы заголовком.
    Строгое совпадение (:func:`title_matches`) берётся сразу; ослабленное (≥ ``BODY_TITLE_MIN_RATIO``)
    — только если второй кандидат на полосе отстаёт на ``BODY_TITLE_MIN_GAP``.

    Args:
        blocks: Тексты блоков.
        floating: Плавающие ли блоки.
        page_of: Полоса каждого блока.
        pages: Данные полос.
        page_start: Первый блок каждой полосы.
        page_end: Конец (не включая) блоков каждой полосы.
        article_id: Статья.
        title: Её название из оглавления.
        toc_page: Её страница по оглавлению; ``None`` — искать негде.
        edits: Правки — сюда пишется замена и якорь.

    Returns:
        ``(индекс полосы, текст заголовка, индекс блока, источник)`` — источник ``body`` (строгое
        совпадение), ``body_loose`` (ослабленное) или ``marker`` (из тега раздела); ``None`` — не найдено.
    """
    if toc_page is None:
        return None
    for distance in (0, 1):
        for page_index, page in enumerate(pages):
            if page.number is None or abs(page.number - toc_page) != distance or page_index not in page_start:
                continue
            candidates = _body_candidates(blocks, floating, page_start[page_index], page_end[page_index], title, edits)
            if not candidates:
                continue
            best = candidates[0]
            if best.ratio < 1.0 and len(candidates) > 1 and candidates[1].ratio > best.ratio - BODY_TITLE_MIN_GAP:
                continue  # два похожих абзаца — пусть решает вспомогательный запрос
            edits.replace[best.index] = f"# {best.text}"
            edits.anchors.append((("article", article_id), best.index))
            source = "marker" if best.from_tag else "body" if best.ratio >= 1.0 else "body_loose"
            return page_index, best.text, best.index, source
    return None


def _rubric_from_marker(
    blocks: list[str],
    floating: list[bool],
    page_of: list[int],
    page_start: dict[int, int],
    head: int,
    rubric_id: str,
    title: str,
    edits: BlockEdits,
) -> str | None:
    """Рубрика без `<rubric>`: `<marker>` с её названием выше `#` первой статьи раздела на той же полосе → `<rubric>`.

    Шапку раздела модель иногда пишет маркером, а перед восстановленным `#` (см. ``_restore_from_body``)
    доводка полосы её к заголовку не подтянула — `#` там ещё не было.

    Args:
        blocks: Тексты блоков.
        floating: Плавающие ли блоки.
        page_of: Полоса каждого блока.
        page_start: Первый блок каждой полосы.
        head: Индекс блока `#` первой статьи раздела.
        rubric_id: Рубрика.
        title: Её название из оглавления.
        edits: Правки — сюда пишется замена и якорь.

    Returns:
        Текст рубрики или ``None``, если маркера нет.
    """
    start = page_start.get(page_of[head], head)
    for index in range(head - 1, start - 1, -1):
        match = _MARKER_TEXT.match(blocks[index])
        if match is not None and title_matches(match.group(1), [title]) and index not in edits.replace:
            edits.replace[index] = f"<{StructureTag.RUBRIC}>*{match.group(1)}*</{StructureTag.RUBRIC}>"
            edits.anchors.append((("rubric", rubric_id), index))
            return match.group(1)
    return None


def _restore_rubric(
    pages: list[PageRefs], page_of: list[int], head: int, rubric_id: str, title: str, edits: BlockEdits
) -> str | None:
    """Рубрика без `<rubric>`: если модель видела её в колонтитуле какой-то полосы — тег перед `#` первой статьи раздела.

    Args:
        pages: Данные полос.
        page_of: Полоса каждого блока.
        head: Индекс блока `#` первой статьи раздела.
        rubric_id: Рубрика.
        title: Её название из оглавления — колонтитул должен быть на него похож.
        edits: Правки — сюда пишется вставка.

    Returns:
        Текст вставленной рубрики или ``None``, если её нет ни в одном колонтитуле.
    """
    for page in pages:
        text = page.header_text(title, rubric_id=rubric_id)
        if text is not None:
            # Рубрика — перед заголовком, в том числе перед восстановленным (он уже запланирован на это место).
            edits.insert(head, f"<{StructureTag.RUBRIC}>*{text}*</{StructureTag.RUBRIC}>", ("rubric", rubric_id), True)
            edits.insert(head, rubric_comment(rubric_id), None, True)
            return text
    return None


def _int_or_none(text: str | None) -> int | None:
    """Номер страницы статьи из оглавления числом; ``None`` — не указан или не число."""
    if not text:
        return None
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None
