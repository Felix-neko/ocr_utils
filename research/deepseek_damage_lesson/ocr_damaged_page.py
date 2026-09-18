"""Одна повреждённая страница → DeepSeek V4.1 Flash через OpenRouter → JSON, Markdown и метаданные.

Учебная копия боевого хода ``external_ocr_models run --damage``: всё, что там разложено по
модулям (imaging, prompts, client, ocr, schema, render), здесь собрано в один файл и
откомментировано построчно. Логика та же самая, поэтому результат на той же странице
должен совпасть с боевым прогоном с точностью до случайности модели.

Ход работы, сверху вниз:

1. картинку читаем, переводим в серое, режем на две горизонтальные полосы с перекрытием,
   уменьшаем каждую до 2200 px по длинной стороне и кодируем в JPEG → base64;
2. собираем два сообщения — системный промпт (правила транскрипции и правило 6 про
   повреждения) и пользовательское (подсказка про корешок + обе картинки);
3. шлём POST на OpenRouter с тем же телом, что и боевой код: температура 0, JSON-режим
   ``json_object`` (родной эндпоинт DeepSeek строгую схему не умеет), рассуждения выключены,
   провайдер — сам DeepSeek;
4. из ответа вытаскиваем JSON (модели любят обернуть его в ```-ограждения), приводим к
   нашим полям, доставляем пометки ``<restored>``/``<fuzzy>``/``<unknown/>`` из списка
   ``edge_words`` туда, где модель их в тексте не поставила;
5. пишем три файла рядом: ``.json`` (поля страницы), ``.md`` (YAML-шапка + тело) и
   ``.meta.json`` (провайдер, токены, цена, счётчики пометок).

Запуск: ``uv run python -m research.deepseek_damage_lesson.ocr_damaged_page``.
Ключ OpenRouter берётся из переменной окружения ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import base64  # картинка едет в запросе как data-URL, то есть в base64
import io  # JPEG кодируется в память, а не на диск
import json  # тело запроса, ответ и выходные файлы — всё JSON
import os  # ключ API читаем из окружения
import re  # разбор ответа: ограждения, теги, разрядка
import time  # замер времени запроса
from datetime import datetime, timezone  # отметка времени в .meta.json
from pathlib import Path

import requests  # один POST-запрос к OpenRouter; SDK не нужен
from PIL import Image, ImageOps  # чтение скана, поворот по EXIF, серое, уменьшение

# ---------------------------------------------------------------------------------------
# ПУТИ И ПАРАМЕТРЫ (вместо CLI). Всё, что в боевом коде было опциями, здесь — константы.
# ---------------------------------------------------------------------------------------

# Папка этого скрипта: промпты лежат рядом, выход — в подпапке.
HERE = Path(__file__).resolve().parent

# Какую страницу распознаём. Левая страница разворота, правый край строк ушёл в корешок.
SCAN = (
    HERE.parent
    / "external_ocr_models"
    / "damaged"
    / "нарезанное по страницам"
    / "часть букв в словах совсем закрыта корешком"
    / "IMG_0006_L.jpg"
)

# Куда писать результат: три файла с именем страницы в подпапке «выход».
OUT_DIR = HERE / "выход"

# Подсказка модели про ЭТУ страницу — та же строка, что стоит для неё в
# run_scripts/external_ocr_models/damaged_hints.txt. Без подсказки модель достраивает буквы
# молча, а с завышенной — выдумывает повреждения, поэтому подсказка описывает ровно то,
# что видно глазами: левая страница, правые концы строк скрыты, 1–4 буквы.
HINT = (
    "This is the LEFT page of a tightly bound volume. The RIGHT ends of the lines disappear into the "
    "binding gutter: the last one to four letters of many lines are completely hidden, not visible at all."
)

# Язык промптов: "en" — боевые (system_prompt.md, user_prompt.md), "ru" — их перевод
# (system_prompt_ru.md, user_prompt_ru.md). Имена ключей JSON и тегов одинаковы в обоих.
PROMPT_LANG = "en"

# Модель на OpenRouter. У V4.1 Flash родное зрение с потолком ~1024 токенов на картинку
# (≈1300 px по стороне), поэтому плотную полосу режем на две части — см. STRIPS.
MODEL = "deepseek/deepseek-v4.1-flash"

# Адрес чата OpenRouter (OpenAI-совместимый API) и имя переменной окружения с ключом.
API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_KEY = "OPENROUTER_API_KEY"

# Нарезка картинки: на сколько горизонтальных полос резать и какое перекрытие между ними.
# 8 % высоты страницы — это 5–6 строк текста: модель точно увидит место стыка и не потеряет
# строку, попавшую на границу.
STRIPS = 2
STRIP_OVERLAP = 0.08

# Длинная сторона каждой полосы после уменьшения и качество JPEG. 2200 px ≈ 216 dpi для
# страницы 10 дюймов: строчные буквы основного текста ≈ 20 px, ниже VLM начинают путать буквы.
MAX_SIDE = 2200
JPEG_QUALITY = 85

# Потолок выходных токенов: страница текста — 2–4 тыс. токенов, 16 тыс. оставляет запас,
# но не даёт зациклившейся модели наговорить на доллар.
MAX_TOKENS = 16000

# Сколько раз повторять запрос при перегрузе (429/5xx) или сетевой ошибке и таймаут одного запроса.
ATTEMPTS = 5
TIMEOUT_S = 300.0
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}

# Версия промпта из боевого пакета — пишется в .meta.json, чтобы выходы разных формулировок не смешивались.
PROMPT_VERSION = 13


# ---------------------------------------------------------------------------------------
# ШАГ 1. КАРТИНКА: прочитать, обесцветить, нарезать, уменьшить, закодировать
# ---------------------------------------------------------------------------------------


def encode_jpeg(image: Image.Image) -> tuple[bytes, int, int]:
    """Серая уменьшенная JPEG-копия куска страницы: байты и её размер в пикселях."""
    # Серое: цвет модели для чтения букв не нужен, а файл втрое меньше.
    if image.mode != "L":
        image = ImageOps.grayscale(image)
    # Уменьшаем только если картинка крупнее потолка; мелкую не растягиваем.
    scale = MAX_SIDE / max(image.size)
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    # Кодируем в JPEG прямо в память.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue(), image.width, image.height


def prepare_strips(path: Path) -> list[tuple[bytes, int, int]]:
    """Страница → STRIPS горизонтальных полос сверху вниз с перекрытием, каждая — JPEG."""
    image = Image.open(path)
    # Фотоаппарат мог записать поворот в EXIF, а не в пиксели: применяем его, иначе
    # модель получит лежащую на боку страницу.
    image = ImageOps.exif_transpose(image)
    image.load()
    height = image.height
    # Высота одной полосы без перекрытия и половина перекрытия в пикселях.
    step = height / STRIPS
    margin = int(height * STRIP_OVERLAP / 2)
    pieces = []
    for index in range(STRIPS):
        # Границы куска: своя доля высоты плюс запас вверх и вниз, зажатый в кадр.
        top = max(0, int(index * step) - margin)
        bottom = min(height, int((index + 1) * step) + margin)
        # Режем по ИСХОДНОМУ разрешению и только потом уменьшаем: так каждой полосе
        # достаётся больше пикселей, чем досталось бы ей от уменьшенной целой страницы.
        pieces.append(encode_jpeg(image.crop((0, top, image.width, bottom))))
    return pieces


def data_url(jpeg: bytes) -> str:
    """JPEG-байты → строка data:image/jpeg;base64,… — так картинка вкладывается в JSON запроса."""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


# ---------------------------------------------------------------------------------------
# ШАГ 2. ПРОМПТЫ И ТЕЛО ЗАПРОСА
# ---------------------------------------------------------------------------------------


def load_prompts(hint: str, lang: str = PROMPT_LANG) -> tuple[str, str]:
    """Системный и пользовательский промпты из файлов рядом со скриптом.

    Английские — отрендеренные Jinja-шаблоны боевого пакета (режим json, повреждения
    включены, издание не названо, две полосы); русские — их перевод. В пользовательском —
    плейсхолдер ``{hint}`` под подсказку про страницу; системный подставлять нельзя: в нём
    фигурные скобки JSON.
    """
    suffix = "" if lang == "en" else f"_{lang}"
    system = (HERE / f"system_prompt{suffix}.md").read_text(encoding="utf-8")
    user = (HERE / f"user_prompt{suffix}.md").read_text(encoding="utf-8").replace("{hint}", hint)
    return system, user


def build_payload(system: str, user: str, strips: list[tuple[bytes, int, int]]) -> dict:
    """Тело запроса chat/completions — ровно то, что боевой код шлёт для deepseek-v41-flash."""
    # Пользовательское сообщение — список частей: сначала текст, затем картинки по порядку.
    content: list[dict] = [{"type": "text", "text": user}]
    for jpeg, _width, _height in strips:
        # detail=high: по low DeepSeek ужал бы картинку до 512×512, и буквы бы пропали.
        content.append({"type": "image_url", "image_url": {"url": data_url(jpeg), "detail": "high"}})
    return {
        "model": MODEL,
        "messages": [
            # Системное сообщение — правила транскрипции и разметки повреждений.
            {"role": "system", "content": system},
            # Пользовательское — подсказка про корешок, просьба транскрибировать, картинки.
            {"role": "user", "content": content},
        ],
        # Температура 0: транскрипция должна быть воспроизводимой, творчество тут вредно.
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        # Просим валидный JSON. Строгую схему (json_schema) родной эндпоинт DeepSeek не
        # умеет, поэтому поля описаны словами в системном промпте, а здесь — только формат.
        "response_format": {"type": "json_object"},
        # Рассуждения выключены: на OCR они не помогают, а стоят втрое (замер: 4238 токенов
        # рассуждений при побайтно том же тексте).
        "reasoning": {"enabled": False},
        # Маршрутизация OpenRouter: сначала родной DeepSeek (самый дешёвый и без
        # квантования), при его недоступности — любой другой, кроме Relace (fp4-копия).
        "provider": {"order": ["deepseek"], "allow_fallbacks": True, "ignore": ["relace"]},
    }


# ---------------------------------------------------------------------------------------
# ШАГ 3. ЗАПРОС К OPENROUTER С ПОВТОРАМИ
# ---------------------------------------------------------------------------------------


def chat(payload: dict) -> tuple[dict, int]:
    """POST на OpenRouter; при перегрузе или сетевом сбое — пауза и повтор. Возвращает тело и число попыток."""
    key = os.environ.get(ENV_KEY)
    if not key:
        raise SystemExit(f"нет ключа OpenRouter: задайте ${ENV_KEY}")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # Заголовки-визитка OpenRouter: попадают в их статистику, на маршрутизацию не влияют.
        "HTTP-Referer": "https://github.com/Felix-Neko/ocr_utils",
        "X-Title": "ocr_utils deepseek_damage_lesson",
    }
    last_error = "запрос не удался"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT_S)
        except requests.RequestException as error:
            last_error = f"сеть: {error}"
        else:
            if response.status_code in RETRY_STATUSES:
                # Перегруз провайдера — имеет смысл подождать и повторить.
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            elif response.status_code != 200:
                # 400/401/402/403 — наша ошибка или кончились деньги; повтор не вылечит.
                raise SystemExit(f"HTTP {response.status_code}: {response.text[:1000]}")
            else:
                body = response.json()
                # OpenRouter умеет отдать 200 с ошибкой провайдера внутри тела.
                if "error" in body and not body.get("choices"):
                    raise SystemExit(f"провайдер: {body['error']}")
                return body, attempt
        if attempt < ATTEMPTS:
            delay = min(60.0, 2.0 * 2**attempt)  # 4, 8, 16, 32 с — экспоненциальная пауза
            print(f"попытка {attempt}/{ATTEMPTS} не удалась ({last_error}), пауза {delay:.0f} с")
            time.sleep(delay)
    raise SystemExit(last_error)


def response_text(body: dict) -> str:
    """Текст ответа модели из первого choice; некоторые провайдеры отдают его списком частей."""
    message = body["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


# ---------------------------------------------------------------------------------------
# ШАГ 4. РАЗБОР ОТВЕТА
# ---------------------------------------------------------------------------------------

# Ограждение ```json … ```: модели ставят его, даже когда просили голый JSON.
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)

# Разрядка: «П р и м е ч а н и е» — три и более букв через одиночные пробелы.
_SPACED_WORD = re.compile(r"(?<![А-ЯЁа-яёA-Za-z])([А-ЯЁа-яёA-Za-z](?: [А-ЯЁа-яёA-Za-z]){2,})(?![А-ЯЁа-яёA-Za-z])")

# Теги повреждений — чтобы снимать их со слов из edge_words и считать в тексте.
_TAG_STRIP = re.compile(r"</?(restored|fuzzy)>|<unknown\s*/>")
_UNKNOWN = re.compile(r"<unknown\s*/>")


def parse_json_text(text: str) -> dict:
    """JSON из ответа: как есть, без ограждений или первый ``{...}`` в тексте."""
    if not text.strip():
        raise SystemExit("пустой ответ модели")
    # Кандидаты по убыванию строгости: весь текст, без ограждений, от первой { до последней }.
    candidates = [text]
    match = _FENCE.match(text)
    if match:
        candidates.append(match.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
        candidates.append(text[start:])  # первый объект, что бы ни шло следом
    decoder = json.JSONDecoder()
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            # raw_decode читает ОДИН объект и не ругается на хвост после него.
            payload, _ = decoder.raw_decode(candidate.lstrip())
            if isinstance(payload, dict) and isinstance(payload.get("content_markdown"), str):
                return payload
            last_error = ValueError("в JSON нет строкового content_markdown")
        except json.JSONDecodeError as error:
            last_error = error
    raise SystemExit(f"невалидный JSON: {last_error}")


def unspace_letters(text: str) -> str:
    """«П р и м е ч а н и е» → «*Примечание*»: разрядку склеиваем и выделяем курсивом."""

    def join(match: re.Match) -> str:
        word = match.group(1).replace(" ", "")
        before = text[max(0, match.start() - 2) : match.start()]
        # Слово уже стоит в курсиве или жирном — только склеиваем.
        return word if before.endswith(("*", "_")) else f"*{word}*"

    return _SPACED_WORD.sub(join, text)


def text_or_none(value: object) -> str | None:
    """Пустая строка и null — одно и то же: «нет значения»."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def to_page_result(payload: dict) -> dict:
    """Ответ модели → словарь с полями страницы в том виде, в каком его пишет боевой .json."""
    return {
        "content_markdown": unspace_letters(payload["content_markdown"]),
        "page_number": text_or_none(payload.get("page_number")),
        "running_header": text_or_none(payload.get("running_header")),
        "running_footer": text_or_none(payload.get("running_footer")),
        "is_toc": bool(payload.get("is_toc", False)),
        "notes": str(payload.get("notes") or ""),
        "rubric": text_or_none(payload.get("rubric")),
        "title": text_or_none(payload.get("title")),
        # Авторы — список {name, position}; записи без имени выбрасываем.
        "authors": [
            {"name": str(item.get("name")).strip(), "position": text_or_none(item.get("position"))}
            for item in (payload.get("authors") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ],
        # Поля режима повреждений.
        "restored": [str(item) for item in (payload.get("restored") or []) if str(item).strip()],
        "fuzzy": [str(item) for item in (payload.get("fuzzy") or []) if str(item).strip()],
        "unknown": int(payload.get("unknown") or 0),
        "damage": str(payload.get("damage") or ""),
        "edge_words": [
            item for item in (payload.get("edge_words") or []) if isinstance(item, dict) and item.get("full")
        ],
    }


def tags_from_edge_words(body: str, edge_words: list[dict]) -> tuple[str, int]:
    """Расставить теги по списку ``edge_words`` там, где модель в тексте их не поставила.

    Модель охотнее заполняет список повреждённых строк, чем ставит теги внутри текста.
    Для записи ``kind=hidden`` невидимая часть — это ``full`` минус видимый фрагмент ``seen``
    (с начала или с конца слова); первое вхождение голого ``full`` в тексте заменяется на слово
    с ``<restored>``. ``fuzzy`` — то же с ``<fuzzy>``; ``unknown`` — ``seen`` + ``<unknown/>``.
    Возвращает текст и число вставленных пометок.
    """
    inserted = 0
    for item in edge_words:
        # Слово «как написано» и «как видно», без тегов, если модель их всё же поставила.
        full = _TAG_STRIP.sub("", str(item.get("full") or "")).strip()
        seen = _TAG_STRIP.sub("", str(item.get("seen") or "")).strip()
        kind = str(item.get("kind") or "hidden")
        # «адми-» → «административные» — обычный перенос, а не срез: модель, которой сказали
        # про повреждённый край, охотно записывает переносы в «скрытые» буквы. Пропускаем.
        if not full or full not in body or seen.endswith(("-", "­")):
            continue
        tagged = None
        if kind == "unknown":
            tagged = (seen + "<unknown/>") if full.startswith(seen) else ("<unknown/>" + seen)
        elif seen and full.startswith(seen) and len(full) > len(seen):
            # Видно начало слова — скрыт хвост.
            tag = "restored" if kind == "hidden" else "fuzzy"
            tagged = f"{seen}<{tag}>{full[len(seen):]}</{tag}>"
        elif seen and full.endswith(seen) and len(full) > len(seen):
            # Видно окончание — скрыто начало.
            tag = "restored" if kind == "hidden" else "fuzzy"
            tagged = f"<{tag}>{full[: len(full) - len(seen)]}</{tag}>{seen}"
        elif kind == "fuzzy":
            tagged = f"<fuzzy>{full}</fuzzy>"
        if not tagged:
            continue
        # Ищем голое слово целиком: не внутри другого слова и не внутри уже стоящего тега.
        pattern = re.compile(r"(?<![\w<>/])" + re.escape(full) + r"(?![\w<])")
        match = pattern.search(body)
        if match is None:
            continue
        before = body[max(0, match.start() - 12) : match.start()]
        if "<restored>" in before or "<fuzzy>" in before or "<unknown" in before:
            continue  # слово уже помечено
        body = body[: match.start()] + tagged + body[match.end() :]
        inserted += 1
    return body, inserted


def tag_counts(text: str) -> dict[str, int]:
    """Сколько в тексте пар <restored>, <fuzzy> и маркеров <unknown/> — для сводки в .meta.json."""
    counts = {name: len(re.findall(rf"<{name}>.*?</{name}>", text, re.DOTALL)) for name in ("restored", "fuzzy")}
    counts["unknown"] = len(_UNKNOWN.findall(text))
    return counts


# ---------------------------------------------------------------------------------------
# ШАГ 5. ВЫХОДНЫЕ ФАЙЛЫ
# ---------------------------------------------------------------------------------------


def yaml_value(value: object) -> str:
    """Скаляр в YAML-шапке .md: null, true/false или строка в кавычках."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(str(value), ensure_ascii=False)


def to_markdown(result: dict) -> str:
    """Файл .md: YAML-шапка с полями страницы, затем тело в Markdown."""
    head = [
        "---",
        f"page_number: {yaml_value(result['page_number'])}",
        f"running_header: {yaml_value(result['running_header'])}",
        f"running_footer: {yaml_value(result['running_footer'])}",
        f"is_toc: {yaml_value(result['is_toc'])}",
        f"notes: {yaml_value(result['notes'])}",
        "---",
        "",
    ]
    return "\n".join(head) + result["content_markdown"].rstrip() + "\n"


def recognise(scan: Path, hint: str, out_dir: Path, lang: str = PROMPT_LANG) -> dict:
    """Одна страница от картинки до трёх файлов; возвращает .meta.json в виде словаря."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / scan.stem  # IMG_0006_L → IMG_0006_L.json / .md / .meta.json
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.monotonic()

    # 1. Картинка → две серые JPEG-полосы.
    strips = prepare_strips(scan)
    print("полосы:", [(width, height, f"{len(jpeg) // 1024} КБ") for jpeg, width, height in strips])

    # 2. Промпты и тело запроса.
    system, user = load_prompts(hint, lang)
    payload = build_payload(system, user, strips)

    # 3. Запрос.
    body, attempts = chat(payload)
    text = response_text(body)
    usage = body.get("usage") or {}
    print(
        f"ответ: {len(text)} знаков, провайдер {body.get('provider')}, токены {usage.get('prompt_tokens')}+"
        f"{usage.get('completion_tokens')}, цена ${usage.get('cost')}"
    )
    # Сырой ответ сохраняем всегда: по нему видно, что именно вернула модель.
    base.with_suffix(".raw.txt").write_text(text, encoding="utf-8")

    # 4. Разбор и доставка тегов из edge_words.
    result = to_page_result(parse_json_text(text))
    result["content_markdown"], inserted = tags_from_edge_words(result["content_markdown"], result["edge_words"])

    # 5. Три выходных файла — в той же форме, что у боевого прогона.
    base.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    meta = {
        "page": scan.name,
        "model": "deepseek-v41-flash",
        "openrouter_id": MODEL,
        "started_at": started_at,
        "prompt_version": PROMPT_VERSION,
        "prompt_lang": lang,
        "output_mode": "json",
        "strips": STRIPS,
        "max_side": MAX_SIDE,
        "reasoning": "off",
        "image_px": [[width, height] for _jpeg, width, height in strips],
        "image_bytes": sum(len(jpeg) for jpeg, _w, _h in strips),
        "json_mode_used": "json_object",
        "seconds": round(time.monotonic() - started, 2),
        "provider": body.get("provider"),
        "served_model": body.get("model"),
        "request_id": body.get("id"),
        "finish_reason": body["choices"][0].get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost"),
        "attempts": attempts,
        "response_chars": len(text),
        "tags_from_edge_words": inserted,
        "content_chars": len(result["content_markdown"]),
        "tags": tag_counts(result["content_markdown"]),
        "damage_seen": result["damage"],
    }
    base.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"пометок: {meta['tags']}, доставлено из edge_words: {inserted}")
    print(f"записано: {base.with_suffix('.json')}, .md, .meta.json")
    return meta


def main() -> None:
    """Точка входа: константы SCAN, HINT, OUT_DIR, PROMPT_LANG → одна страница."""
    recognise(SCAN, HINT, OUT_DIR, PROMPT_LANG)


if __name__ == "__main__":
    main()
