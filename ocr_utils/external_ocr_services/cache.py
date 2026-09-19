"""Кэш запросов к модели: каждый проход по полосе — своя папка с запросом и ответом, повтор без запроса.

Зачем. Полоса за прогон может уйти в модель несколько раз: этапом ``toc``, после понижения — этапом
``page``, на круге повтора — с новым списком статей, внутри одного прохода — в нескольких режимах JSON
(откат ``json_schema`` → ``json_object`` → без формата, эхо ``response_format``). В ``--debug-dir``
каждый следующий проход перезаписывает ``.prompt.txt``/``.raw.txt`` предыдущего, а ``--skip-done``
знает только про готовый ``.json``. Здесь каждый запрос лежит отдельно и находится по ключу.

Ключ — SHA-256 от payload запроса (``ocr.build_payload``), в котором каждая картинка заменена хэшем
её JPEG-байтов (:func:`request_key`). В ключ входит всё, что влияет на ответ: модель, системный и
пользовательский промпты (с версией промпта, списком статей, раскладкой тайлов, блоком второго
прохода), ``response_format``, ``reasoning``, ``temperature``, ``max_tokens`` и сами тайлы. Поэтому
разные этапы, понижение, круг повтора и режимы JSON получают разные ключи сами собой, а тот же запрос
(``temperature 0``) отдаёт сохранённый ответ без обращения к модели.

Раскладка: ``cache_dir/{год}/{выпуск}/{полоса}/{этап}[.pass2].{режим JSON}.{12 знаков ключа}/`` с
``request.json`` (что ушло, без байтов картинок) и ``response.json`` (ответ модели с вердиктом:
годный, эхо формата, цикл заполнителя). Сетевые ошибки не кэшируются — они временные, — но
дописываются в ``errors.jsonl`` папки полосы.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from ocr_utils.external_ocr_services.client import ChatResponse

logger = logging.getLogger(__name__)

REQUEST_FILE = "request.json"
RESPONSE_FILE = "response.json"
ERRORS_FILE = "errors.jsonl"
KEY_PREFIX_LEN = 12  # сколько знаков ключа идёт в имя папки; полный ключ — внутри файлов


class CacheVerdict(StrEnum):
    """Как ``recognize_page`` оценил ответ: годен ли он или цепочка откатов пошла дальше."""

    OK = "ok"  # ответ разбирается как страница
    FORMAT_ECHO = "format_echo"  # ``{"type": "json_object"}`` вместо страницы — запрос повторён без режима
    GAP_RUNAWAY = "gap_runaway"  # модель зациклилась на «▒» до потолка токенов — запрос повторён в другом режиме


@dataclass(frozen=True)
class CachedResponse:
    """Что нашлось в кэше по ключу: ответ модели и вердикт, вынесенный ему в тот раз."""

    response: ChatResponse
    verdict: CacheVerdict
    path: Path  # папка записи — для лога и meta


def tile_digest(data: bytes) -> str:
    """Хэш байтов тайла для ключа и ``request.json``.

    Args:
        data: JPEG тайла, как он уходит в data-URL.

    Returns:
        ``sha256:<hex>``.
    """
    return "sha256:" + hashlib.sha256(data).hexdigest()


def strip_images(payload: dict) -> tuple[dict, list[str]]:
    """Payload без байтов картинок: каждая часть ``image_url`` заменена хэшем данных.

    Args:
        payload: Тело запроса из ``ocr.build_payload`` (data-URL тайлов внутри ``messages``).

    Returns:
        ``(payload без картинок, хэши тайлов по порядку)``. Исходный словарь не меняется.
    """
    tiles: list[str] = []
    messages = []
    for message in payload.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            messages.append(message)
            continue
        parts = []
        for part in content:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                # data-URL: «data:image/jpeg;base64,<данные>» — хэшируем сами данные, а не строку,
                # чтобы ключ совпадал с хэшем файла тайла.
                data = url.split(",", 1)[1].encode("ascii") if url.startswith("data:") else url.encode("utf-8")
                digest = tile_digest(_b64_decode(data))
                tiles.append(digest)
                parts.append({"type": "image_url", "tile": digest, "detail": part["image_url"].get("detail")})
            else:
                parts.append(part)
        messages.append({**message, "content": parts})
    return {**payload, "messages": messages}, tiles


def _b64_decode(data: bytes) -> bytes:
    """Base64 → байты; не-base64 (внешний URL) возвращается как есть.

    Args:
        data: Байты после запятой data-URL или сам URL.
    """
    try:
        return base64.b64decode(data, validate=True)
    except Exception:
        return data


def request_key(payload: dict) -> str:
    """Ключ кэша: SHA-256 канонического JSON payload без байтов картинок.

    Args:
        payload: Тело запроса из ``ocr.build_payload``.

    Returns:
        64 hex-знака; одинаковый payload (промпты, параметры, тайлы) → одинаковый ключ независимо от
        порядка ключей словаря.
    """
    stripped, _ = strip_images(payload)
    canonical = json.dumps(stripped, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def entry_dir(root: Path, rel: Path, stage: str, json_mode: str, key: str, second_pass: bool = False) -> Path:
    """Папка записи кэша для одного запроса.

    Args:
        root: Корень кэша (``--cache-dir``).
        rel: Путь полосы относительно in-dir (``{год}/{выпуск}/{полоса}.jpg``).
        stage: Этап (``page`` / ``toc``).
        json_mode: Режим JSON попытки.
        key: Полный ключ запроса.
        second_pass: Второй проход — суффикс ``.pass2`` в имени.

    Returns:
        ``root / год / выпуск / полоса / {этап}[.pass2].{режим}.{key[:12]}``.
    """
    name = f"{stage}{'.pass2' if second_pass else ''}.{json_mode}.{key[:KEY_PREFIX_LEN]}"
    return root / rel.with_suffix("") / name


def _write_json(path: Path, data: Any) -> None:
    """JSON во временный файл рядом и ``os.replace`` — читатель не увидит недописанного файла.

    Args:
        path: Куда писать.
        data: Что писать (сериализуемое JSON).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


class RequestCache:
    """Кэш на диске: ``lookup`` по ключу перед запросом, ``store`` после ответа, ``record_error`` при сбое.

    Args:
        root: Корень кэша.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def lookup(self, path: Path) -> CachedResponse | None:
        """Есть ли в папке записи ответ.

        Args:
            path: Папка записи (:func:`entry_dir`).

        Returns:
            Ответ и вердикт или ``None``, если записи нет либо она битая (тогда запрос идёт заново).
        """
        file = path / RESPONSE_FILE
        if not file.is_file():
            return None
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            verdict = CacheVerdict(data.pop("verdict"))
            data.pop("received_at", None)
            response = ChatResponse(**data)
        except (ValueError, TypeError, KeyError) as error:
            logger.warning("кэш %s: запись не читается (%s), запрос уйдёт заново", path, error)
            return None
        return CachedResponse(response, verdict, path)

    def store(
        self, path: Path, request_info: dict, payload: dict, response: ChatResponse, verdict: CacheVerdict
    ) -> None:
        """Записать запрос и ответ.

        Args:
            path: Папка записи.
            request_info: Контекст запроса для ``request.json``: полоса, этап, ожидаемый вид оглавления,
                ``toc_hash``, версия промпта, модель, раскладка тайлов — всё, что помогает понять
                запись без meta полосы.
            payload: Тело запроса с картинками; сюда ложится без них, с хэшами тайлов.
            response: Ответ модели.
            verdict: Как ответ оценён.
        """
        stripped, tiles = strip_images(payload)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_json(path / REQUEST_FILE, {**request_info, "tiles": tiles, "created_at": now, "payload": stripped})
        _write_json(path / RESPONSE_FILE, {**asdict(response), "verdict": verdict.value, "received_at": now})

    def record_error(self, path: Path, error: str, body: str) -> None:
        """Дописать сетевую ошибку в ``errors.jsonl`` папки полосы (сама запись не создаётся).

        Args:
            path: Папка записи, к которой относился запрос; ошибка ложится на уровень выше, у полосы.
            error: Текст ошибки.
            body: Начало тела ответа сервера.
        """
        page_dir = path.parent
        page_dir.mkdir(parents=True, exist_ok=True)
        line = {
            "entry": path.name,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": error,
            "body": body[:300],
        }
        with (page_dir / ERRORS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


@lru_cache(maxsize=8)
def cache_for(root: Path) -> RequestCache:
    """Один объект кэша на корень (пул потоков делит его; состояния у него нет, только путь).

    Args:
        root: Корень кэша.
    """
    return RequestCache(root)
