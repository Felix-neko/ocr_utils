"""Сбор индекса выпусков журнала «Деньги и кредит» из библиотеки Банка России.

Для каждого выпуска (страница /catalog/lib/mag/<id>/) вытаскиваем список
прикреплённых PDF-файлов («Содержание», «Полный текст», …), для каждого
резолвим реальный путь на сервере через ajax-эндпоинт pdf.viewer и узнаём
размер HEAD-запросом. Результат — JSON-индекс для последующей качалки.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://library.cbr.ru"
AJAX = BASE + "/bitrix/services/main/ajax.php?mode=ajax&c=sbbr:pdf.viewer&action=getFile"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# один <li> со ссылкой на просмотрщик и подписью файла
RE_ITEM = re.compile(
    r'<li class="book-card-2__files-item">.*?'
    r"/pdf-viewer/\?uid=([0-9a-f-]+).*?"
    r'<span class="file-link__name">\s*(.*?)\s*</span>',
    re.S,
)
RE_EDITION = re.compile(r'<a style="text-decoration:none" href="(/catalog/lib/mag/\d+/)">\s*(.*?)\s*</a>', re.S)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    s.verify = False
    return s


def fetch_editions(session: requests.Session, seed_url: str) -> list[dict]:
    """Список всех выпусков из модалки «Выберите выпуск» на странице любого выпуска."""
    html = session.get(seed_url, timeout=60).text
    start = html.find('id="select-edition"')
    end = html.find("</article>", start)
    out = []
    for url, title in RE_EDITION.findall(html[start:end]):
        title = re.sub(r"\s+", " ", title).strip()
        year = re.search(r"\((\d{4})\)", title)
        out.append(
            {
                "url": BASE + url,
                "mag_id": re.search(r"/(\d+)/$", url).group(1),
                "title": title,
                "year": int(year.group(1)) if year else None,
            }
        )
    return out


def probe_edition(edition: dict) -> dict:
    """Дособрать по выпуску: список файлов с путями и размерами."""
    session = make_session()
    files = []
    try:
        html = session.get(edition["url"], timeout=60).text
        seen = set()
        for uid, name in RE_ITEM.findall(html):
            if uid in seen:
                continue
            seen.add(uid)
            entry = {"uid": uid, "name": re.sub(r"\s+", " ", name).strip()}
            try:
                data = session.post(
                    AJAX, files={"documentUuid": (None, uid), "docType": (None, ""), "sessid": (None, "")}, timeout=60
                ).json()
                entry["path"] = data.get("data")
            except Exception as exc:  # сеть/парсинг — фиксируем и идём дальше
                entry["error"] = f"ajax: {exc}"
            if entry.get("path"):
                entry["url"] = BASE + entry["path"]
                try:
                    head = session.head(entry["url"], timeout=60, allow_redirects=True)
                    entry["http_status"] = head.status_code
                    entry["size"] = int(head.headers.get("Content-Length", 0)) or None
                    entry["content_type"] = head.headers.get("Content-Type")
                except Exception as exc:
                    entry["error"] = f"head: {exc}"
            files.append(entry)
    except Exception as exc:
        edition["error"] = str(exc)
    edition["files"] = files
    return edition


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", default=BASE + "/catalog/lib/mag/1053233/", help="страница любого выпуска журнала")
    ap.add_argument("--out", required=True, help="куда положить JSON-индекс")
    ap.add_argument("--jobs", type=int, default=8, help="сколько параллельных запросов к сайту")
    args = ap.parse_args()

    session = make_session()
    editions = fetch_editions(session, args.seed)
    print(f"выпусков в списке: {len(editions)}", file=sys.stderr, flush=True)

    done = 0
    result = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for edition in pool.map(probe_edition, editions):
            result.append(edition)
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(editions)}", file=sys.stderr, flush=True)

    result.sort(key=lambda e: int(e["mag_id"]))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"готово: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
