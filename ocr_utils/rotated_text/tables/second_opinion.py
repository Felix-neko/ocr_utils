"""Второе мнение surya для каждой повёрнутой ячейки — с фильтром выдумок.

Два читателя, потому что они ошибаются на разном. Tesseract на наборном тексте даёт
CER 0.034 (37 ячеек с эталоном), но на чертёжном курсиве бланков (1968/03 IMG_0114:
«с 7 по 12») не читает ничего. Surya читает и курсив, но ЛАТИНИЦЕЙ («c 7 no 12»,
«NN n.n.» вместо «№№ п.п.»), а на короткой надписи в узкой графе дописывает текст, которого
нет, — вплоть до фраз на корейском (медианный CER 0, средний 2.36). Поэтому:

* уверенный tesseract (подменяемый текст с уверенностью от 0.8) остаётся; surya берётся
  только там, где tesseract не уверен или не прочёл;
* латинские двойники в ответе surya заменяются кириллицей (c → с, n → п, o → о …):
  журнал русский, латиница в нём — марки и формулы, где это ничего не портит;
* ЧУЖАЯ ПИСЬМЕННОСТЬ — любая буква не из кириллицы и латиницы отвергает ответ целиком;
* уверенность surya не ниже 0.5: «SI NO 6» вместо «с 1 по 6» пришло с 0.28;
* ДЛИНА — ответ не длиннее 120 знаков (выдумка удлиняет).

ЯЧЕЙКА ЧИТАЕТСЯ КАК БЛОК, БЕЗ ДЕТЕКТОРА СТРОК (задача ``block_without_boxes``, рамка —
вся вырезка). Замер: с детектором 1,3 с на ячейку (детектор гоняется на каждый кроп, его
постобработка на CPU держит родительский процесс на 100% одного ядра, GPU загружен на 8%),
блоком — 0,26 с, и текст тот же или лучше. Подача таблицы целиком проверена и отвергнута:
детектор режет боковой текст на слова в произвольном порядке («окончатель», «запаса»,
«плана», «U», «잍»), а курсивный бланк читает как «90013» вместо «с 7 по 12». Читаются
только ячейки, где tesseract ненадёжен. Модель живёт в родительском процессе —
видеопамять одна на всех (CLAUDE.md), в пул воркеров GPU не заворачивается.
"""

from __future__ import annotations

import re
import time
from typing import Sequence

import cv2
import numpy as np

from ocr_utils.rotated_text.tables.ocr import CellText, prepare

# От этой уверенности подменяемый ответ tesseract считается надёжным и surya не заменяет.
RELIABLE_CONFIDENCE = 0.8

# Ниже этой уверенности tesseract согласия с ним не требуется: при 0.52 он прочёл
# «потрё НОСТЬ Г.)», а surya — «Пятидневная потребность III», и требовать согласия значило
# отвергнуть верный ответ. Порог тот же, что у подмены (``pipeline.CONFIDENCE_REPLACE``).
TRUST_BELOW = 0.6

MAX_LENGTH = 400
MAX_DISAGREEMENT = 0.5

# Галлюцинация surya — зацикленный текст: «and the second second second…», «de la company
# de la company…», «Leys Leys Leys…» по 765 знаков с уверенностью 0.95–0.99. Признаки:
# среди словесных токенов (с буквами; ряды чисел «0 0 0 1 1 1» — не зацикливание) разных
# не больше половины (от шести токенов), серия одного знака от десяти подряд, либо больше
# десяти латинских букв без кириллических двойников — журнал русский, настоящая латиница в
# нём короткая (марки, формулы).
DISTINCT_SHARE = 0.5
REPEAT_MIN_TOKENS = 6
RUN_LENGTH = 10
LATIN_MAX = 10

# Порог уверенности surya для короткой надписи без слова от трёх букв («с 7 по 12»): свой,
# ниже tesseract-овского 0.9 — surya на курсивном бланке даёт 0.85–0.95 на верных чтениях.
SURYA_SHORT_CONFIDENCE = 0.8

# Ниже этой уверенности ответ surya не берётся.
SURYA_MIN_CONFIDENCE = 0.5

# Латинские двойники кириллических букв: по начертанию в прямом наборе (A → А, c → с …)
# и в курсивном (n → п, u → и). Буквы без двойника (i, j, l, f, s, v, w, z, d, g, q, r, t,
# b, h и заглавные I, J, L, S, U, V, W, Z, D, F, G, Q, R) в наборе нет намеренно.
LATIN_TWINS = "AaBCcEeHKkMmOoPpTXxyun"
HOMOGLYPHS = str.maketrans(LATIN_TWINS, "АаВСсЕеНКкМмОоРрТХхуип")
LATIN = re.compile(r"[A-Za-z]")
TAG = re.compile(r"</?(?!br)[a-z]+>")
BREAK = re.compile(r"<br\s*/?>")

DEFAULT_BATCH = 16
BATCH_MIN = 1

# Буква, которая не кириллица и не латиница.
FOREIGN_LETTER = re.compile(r"[^\W\d_]")
OWN_LETTER = re.compile(r"[а-яёА-ЯЁa-zA-Z]")

_SHARED = None


def surya_available() -> bool:
    try:
        import surya.recognition  # noqa: F401
    except Exception:  # noqa: BLE001 — любая причина означает одно: surya нет
        return False
    return True


def foreign_script(text: str) -> bool:
    return any(not OWN_LETTER.match(char) for char in FOREIGN_LETTER.findall(text))


def cyrillic(text: str) -> str:
    """Убрать теги surya (``<i>``) и заменить латинские двойники кириллицей — но только
    если ВСЕ латинские буквы ячейки имеют двойников.

    В ячейке бывает либо кириллица, либо латиница, либо цифры. Surya пишет курсивный
    бланк целиком латиницей («c 7 no 12»), и кириллического слова, по которому можно было
    бы понять, что это транслитерация, там нет — поэтому решает состав букв. Буква без
    двойника («i=n», «a_i», «III», индексы j, k) — признак, что латиница настоящая:
    формула, индекс, римское число, и ячейка остаётся как есть.
    """
    clean = BREAK.sub(" ", TAG.sub("", text))
    latin = LATIN.findall(clean)
    if not latin or any(letter not in LATIN_TWINS for letter in latin):
        return clean
    return clean.translate(HOMOGLYPHS)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("ё", "е").replace("Ё", "Е")).strip().lower()


def edit_distance(first: str, second: str) -> int:
    if first == second:
        return 0
    if not first or not second:
        return len(first) + len(second)
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, start=1):
        current = [i]
        for j, right in enumerate(second, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def disagreement(reference: str, hypothesis: str) -> float:
    """Расстояние Левенштейна к длине эталона."""
    a, b = normalize(reference), normalize(hypothesis)
    if not a:
        return 0.0 if not b else 1.0
    return edit_distance(a, b) / len(a)


def hallucinated(text: str) -> "str | None":
    """Почему ответ похож на выдумку, или None."""
    words = [token.lower() for token in text.split() if FOREIGN_LETTER.search(token)]
    if len(words) >= REPEAT_MIN_TOKENS and len(set(words)) <= DISTINCT_SHARE * len(words):
        return "зацикленный текст"
    if re.search(r"(.)\1{%d,}" % (RUN_LENGTH - 1), text):
        return "серия одного знака"
    if sum(1 for letter in LATIN.findall(text) if letter not in LATIN_TWINS) > LATIN_MAX:
        return "длинная латиница"
    return None


def accept(
    tesseract: CellText, surya: CellText, ours_reliable: bool = False, ours_plausible: bool = True
) -> tuple[bool, str]:
    """Брать ли ответ surya вместо ответа tesseract; второй элемент — причина.
    ``ours_reliable`` — ответ tesseract подменяем и уверен: тогда surya не нужна.
    ``ours_plausible`` — ответ tesseract хотя бы похож на текст: только тогда от surya
    требуется согласие с ним; с мусором («6710 [2» вместо «с 7 по 12») сверять нечего."""
    if ours_reliable:
        return False, "tesseract надёжен"
    theirs = cyrillic(surya.text)
    if not theirs.strip():
        return False, "surya ничего не прочла"
    if foreign_script(theirs):
        return False, "чужая письменность"
    if surya.confidence < SURYA_MIN_CONFIDENCE:
        return False, f"уверенность surya {surya.confidence:.2f} ниже {SURYA_MIN_CONFIDENCE}"
    if len(theirs) > MAX_LENGTH:
        return False, "ответ длиннее ожидаемого"
    why = hallucinated(theirs)
    if why:
        return False, why
    if (
        ours_plausible
        and tesseract.confidence >= TRUST_BELOW
        and disagreement(tesseract.text, theirs) > MAX_DISAGREEMENT
    ):
        return False, "расходится с уверенным tesseract"
    return True, "принято"


def _predictors():
    global _SHARED
    if _SHARED is None:
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        recognizer = RecognitionPredictor(FoundationPredictor())
        recognizer.disable_tqdm = True
        detector = DetectionPredictor()
        detector.disable_tqdm = True
        _SHARED = (recognizer, detector)
    return _SHARED


def _recognize(recognizer, pages, batch: int):
    """Партия блоков с откатом по видеопамяти: при нехватке делится пополам, пока не влезет."""
    import torch
    from surya.common.surya.schema import TaskNames

    while True:
        try:
            return recognizer(
                pages,
                task_names=[TaskNames.block_without_boxes] * len(pages),
                bboxes=[[[0, 0, page.width, page.height]] for page in pages],
                math_mode=False,
                recognition_batch_size=batch,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= BATCH_MIN:
                raise
            batch = max(BATCH_MIN, batch // 2)


def read_cells(crops: Sequence[np.ndarray], batch: int = DEFAULT_BATCH) -> list[CellText]:
    """Прочитать выпрямленные вырезки surya блоками; порядок ответов — порядок вырезок."""
    if not crops:
        return []
    from PIL import Image

    recognizer, _ = _predictors()
    results: list[CellText] = []
    for start in range(0, len(crops), batch):
        chunk = crops[start : start + batch]
        pages = [Image.fromarray(cv2.cvtColor(prepare(gray), cv2.COLOR_GRAY2RGB)) for gray in chunk]
        started = time.time()
        predictions = _recognize(recognizer, pages, batch)
        elapsed = (time.time() - started) / max(1, len(chunk))
        for prediction in predictions:
            # Блок приходит одной «строкой» с переводами ``<br>`` и тегами начертания.
            lines = [
                piece.strip()
                for line in prediction.text_lines
                for piece in BREAK.split(TAG.sub("", line.text))
                if piece.strip()
            ]
            scores = [float(getattr(line, "confidence", 0.0) or 0.0) for line in prediction.text_lines]
            results.append(
                CellText(
                    lines=lines, confidence=float(np.mean(scores)) if scores else 0.0, engine="surya", seconds=elapsed
                )
            )
    return results
