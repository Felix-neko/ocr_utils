"""Создаёт проект, метки и задачи в CVAT из смонтированной папки share.

Запускается в эфемерном контейнере python на сети cvat (см. up.sh):
подключается к cvat-server:8080 под учёткой admin, работает в контексте
организации ORG_SLUG.

Идемпотентность: проект ищется по имени, задачи — по имени; повторный
запуск не создаёт дубликатов, а лишь добавляет недостающее.

Разбиение: 1 папка-выпуск (каталог, где лежат .jpg) -> 1 задача (task),
внутри — 1 job. Задачи создаются «из share» (server_files), файлы через
браузер не загружаются.
"""

import os
import sys

from cvat_sdk import make_client, models
from cvat_sdk.core.proxies.tasks import ResourceType

SHARE_ROOT = "/home/django/share"  # тот же путь, что видит cvat_server
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Схему указываем ЯВНО: без неё cvat-sdk по умолчанию идёт по https и
# спотыкается об SSL (сервер слушает обычный http).
BASE_URL = os.environ.get("CVAT_URL", "http://cvat-server:8080")
ADMIN_USER = os.environ["ADMIN_USER"]
ADMIN_PASS = os.environ["ADMIN_PASS"]
ANN_USER = os.environ["ANN_USER"]
ORG_SLUG = os.environ["ORG_SLUG"]
PROJECT_NAME = os.environ["PROJECT_NAME"]

# Цвета задаём явно: сам CVAT раздаёт их подряд из палитры и выдал три почти
# неразличимых блёклых оттенка (#a3bea1 / #acc4aa / #b5cbb3).
# Логика выбора: сканы жёлто-бежевые, поэтому берём холодные насыщенные тона —
# на тёплой бумаге они видны лучше всего. Двум маскам достаются пурпур и голубой:
# они максимально далеки друг от друга и не сливаются при дальтонизме
# (в отличие от пары красный/зелёный).
LABELS = [
    {"name": "Растровое изображение", "type": "rectangle", "color": "#00E676"},  # ярко-зелёный
    {"name": "Библиотечная печать", "type": "mask", "color": "#D500F9"},  # пурпурный
    {"name": "Рукописная надпись", "type": "mask", "color": "#00B0FF"},  # голубой
]


def issue_dirs(root):
    """Возвращает [(rel_dir, [rel_file, ...]), ...] по каталогам с картинками."""
    result = []
    for cur_dir, _subdirs, files in os.walk(root):
        images = sorted(
            f for f in files if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")
        )
        if not images:
            continue
        rel_dir = os.path.relpath(cur_dir, root)
        rel_files = [os.path.join(rel_dir, f) for f in images]
        result.append((rel_dir, rel_files))
    result.sort(key=lambda item: item[0])
    return result


def task_name_for(rel_dir):
    """'1966 готово/01 готово' -> '1966 / 01'."""
    parts = [p.replace(" готово", "").strip() for p in rel_dir.split(os.sep)]
    parts = [p for p in parts if p]
    return " / ".join(parts) if parts else rel_dir


def main():
    issues = issue_dirs(SHARE_ROOT)
    if not issues:
        print(f"В {SHARE_ROOT} не найдено картинок — нечего заводить.", file=sys.stderr)
        sys.exit(1)
    total_images = sum(len(files) for _, files in issues)
    print(f"Найдено папок-выпусков: {len(issues)}, картинок всего: {total_images}")

    with make_client(host=BASE_URL, credentials=(ADMIN_USER, ADMIN_PASS)) as client:
        client.organization_slug = ORG_SLUG

        # --- проект ---
        existing = [p for p in client.projects.list() if p.name == PROJECT_NAME]
        if existing:
            project = existing[0]
            print(f"Проект {PROJECT_NAME!r} уже существует (id={project.id}), переиспользуем.")
        else:
            project = client.projects.create(
                spec=models.ProjectWriteRequest(
                    name=PROJECT_NAME,
                    labels=[models.PatchedLabelRequest(**lbl) for lbl in LABELS],
                )
            )
            print(f"Проект {PROJECT_NAME!r} создан (id={project.id}).")

        # --- id аннотатора ---
        (users, _) = client.api_client.users_api.list(search=ANN_USER, page_size=100)
        annotator_id = next((u.id for u in users.results if u.username == ANN_USER), None)
        if annotator_id is None:
            print(f"Пользователь {ANN_USER!r} не найден — задачи будут без назначения.")

        # --- какие задачи уже есть ---
        existing_names = {
            t.name for t in client.tasks.list() if t.project_id == project.id
        }

        created, skipped, failed = 0, 0, 0
        for idx, (rel_dir, rel_files) in enumerate(issues, start=1):
            name = task_name_for(rel_dir)
            prefix = f"[{idx}/{len(issues)}] {name} ({len(rel_files)} шт.)"
            if name in existing_names:
                print(f"{prefix}: уже есть, пропуск.")
                skipped += 1
                continue
            try:
                task = client.tasks.create_from_data(
                    spec=models.TaskWriteRequest(name=name, project_id=project.id),
                    resource_type=ResourceType.SHARE,
                    resources=rel_files,
                    data_params={"image_quality": 70},
                )
                if annotator_id is not None:
                    for job in task.get_jobs():
                        job.update(models.PatchedJobWriteRequest(assignee=annotator_id))
                print(f"{prefix}: создана (task id={task.id}), назначена {ANN_USER!r}.")
                created += 1
            except Exception as exc:  # noqa: BLE001
                print(f"{prefix}: ОШИБКА: {exc}", file=sys.stderr)
                failed += 1

        print(f"\nИтог: создано={created}, пропущено={skipped}, ошибок={failed}.")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
