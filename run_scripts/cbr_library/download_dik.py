"""Качалка выпусков «Деньги и кредит» по индексу, собранному scrape_dik_index.py.

Раскладывает PDF по папкам-годам: <dest>/<год>/<имя файла с сервера>.
Докачивает частично скачанное (HTTP Range), пропускает уже целые файлы.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

_print_lock = threading.Lock()


class Stale(IOError):
    """Скачанный кусок разошёлся с тем, что отдаёт сервер, — докачка бессмысленна."""


def log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr, flush=True)


def issue_slug(title: str) -> str:
    """«Выпуск Т.85 N 1 (2026)» -> «t85_n1» для запасного имени файла."""
    slug = title.replace("Выпуск", "").strip()
    slug = re.sub(r"\(\d{4}\)", "", slug)
    slug = re.sub(r"[^\w]+", "_", slug, flags=re.UNICODE).strip("_").lower()
    return slug or "issue"


def remote_size(url: str, timeout: int) -> int | None:
    """Свежий размер файла на сервере (HEAD)."""
    head = requests.head(url, headers={"User-Agent": UA}, timeout=timeout, verify=False, allow_redirects=True)
    return int(head.headers.get("Content-Length", 0)) or None


def declared_size(response, offset: int) -> int | None:
    """Сколько байт обещает сам ответ на GET: из Content-Range для 206, иначе из Content-Length."""
    if response.status_code == 206:
        match = re.search(r"/(\d+)\s*$", response.headers.get("Content-Range", ""))
        return int(match.group(1)) if match else None
    length = response.headers.get("Content-Length")
    return offset + int(length) if length else None


def looks_complete_pdf(path: str) -> bool:
    """Похоже ли на целый PDF: сигнатура в начале и маркер конца файла в хвосте."""
    try:
        with open(path, "rb") as f:
            if f.read(5) != b"%PDF-":
                return False
            f.seek(-2048, os.SEEK_END)
            return b"%%EOF" in f.read()
    except OSError:
        return False


def download_one(task: dict, attempts: int, timeout: int) -> dict:
    """Скачать один файл с докачкой; вернуть статус для сводки.

    Размер из индекса считаем подсказкой, а не истиной: HEAD у библиотеки иногда отдаёт
    закешированную длину, разъезжающуюся с телом ответа. Финальную проверку делаем по
    Content-Length/Content-Range того самого GET, которым качали.
    """
    url, path, hint = task["url"], task["dest"], task.get("size")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        got = os.path.getsize(path)
        if hint and got == hint:
            return {**task, "status": "skip"}
        try:
            if got == remote_size(url, timeout):
                return {**task, "status": "skip"}
        except Exception:
            pass  # не достучались — просто перекачаем

    tmp = path + ".part"
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout, verify=False) as r:
                if r.status_code == 416:  # частичный файл разъехался с сервером
                    raise Stale("сервер отверг Range")
                if have and r.status_code != 206:  # сервер не понял Range — качаем целиком
                    have = 0
                r.raise_for_status()
                expected = declared_size(r, have)
                with open(tmp, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            got = os.path.getsize(tmp)
            if expected and got != expected:
                # у библиотеки бывает кеш, отдающий старый Content-Length с новым телом:
                # если скачанное — целый PDF, верим файлу, а не заголовку
                if looks_complete_pdf(tmp):
                    log(f"  {os.path.basename(path)}: заголовок обещал {expected}, пришло {got}; PDF целый, принимаем")
                else:
                    raise Stale(f"размер {got} != обещанного в ответе {expected}")
            break
        except Exception as exc:
            # сетевой сбой — докачиваем дальше; расхождение размеров — начинаем файл с нуля
            if isinstance(exc, Stale) and os.path.exists(tmp):
                os.remove(tmp)
            if attempt == attempts:
                return {**task, "status": "error", "error": str(exc)}
            log(f"  ретрай {attempt}/{attempts} для {os.path.basename(path)}: {exc}")
            time.sleep(3 * attempt)

    os.replace(tmp, path)
    return {**task, "status": "ok", "bytes": os.path.getsize(path)}


def build_tasks(index: list[dict], dest_root: str, want: str) -> list[dict]:
    tasks = []
    for edition in index:
        year = edition.get("year") or 0
        for entry in edition.get("files", []):
            if want != "all" and entry.get("name") != want:
                continue
            if not entry.get("url"):
                continue
            name = os.path.basename(entry["path"])
            if not name.lower().endswith(".pdf"):
                name = f"dik_{year}_{issue_slug(edition['title'])}.pdf"
            tasks.append(
                {
                    "url": entry["url"],
                    "dest": os.path.join(dest_root, str(year), name),
                    "size": entry.get("size"),
                    "title": edition["title"],
                }
            )
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True, help="JSON от scrape_dik_index.py")
    ap.add_argument("--dest", required=True, help="корневая папка, внутри создаются папки по годам")
    ap.add_argument("--name", default="Полный текст", help="какой файл качать ('all' — все)")
    ap.add_argument("--jobs", type=int, default=4, help="сколько файлов тянуть одновременно")
    ap.add_argument("--attempts", type=int, default=4, help="попыток на файл")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--report", help="куда сохранить JSON-отчёт")
    args = ap.parse_args()

    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)

    tasks = build_tasks(index, args.dest, args.name)
    total = sum(t["size"] or 0 for t in tasks)
    log(f"к скачиванию: {len(tasks)} файлов, ~{total / 2**30:.1f} ГиБ")

    results, done, got_bytes = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for res in pool.map(lambda t: download_one(t, args.attempts, args.timeout), tasks):
            results.append(res)
            done += 1
            got_bytes += res.get("size") or 0
            log(f"[{done}/{len(tasks)}] {res['status']:5} {res['title']} -> {os.path.relpath(res['dest'], args.dest)}")

    ok = sum(r["status"] == "ok" for r in results)
    skip = sum(r["status"] == "skip" for r in results)
    err = [r for r in results if r["status"] == "error"]
    log(f"итог: скачано {ok}, пропущено {skip}, ошибок {len(err)}")
    for r in err:
        log(f"  ОШИБКА {r['title']}: {r['error']}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
