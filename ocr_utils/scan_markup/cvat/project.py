"""Заведение проекта, задач и джобов CVAT по содержимому базы.

Иерархия::

    пак сканов      -> проект CVAT
    годовой комплект-> задача (task), ~1200 кадров
    выпуск          -> джоб (job), ~100 кадров
    полоса          -> кадр (frame)

Границы джобов задаются параметром ``job_file_mapping``: это единственный способ описать
их явно. Обычное разбиение по ``segment_size`` делит задачу на равные куски и границы
выпусков не соблюдает, а несовместимость проверяется сервером — ``job_file_mapping``
нельзя сочетать ни с ``segment_size``, ни с ``sorting_method``, ни с ``start_frame`` /
``stop_frame`` / ``frame_filter`` (см. ``cvat/apps/engine/task.py``,
``_validate_job_file_mapping``), поэтому ни одного из них мы не передаём.

Границы джобов, раз заданные, изменить нельзя, и отдельный джоб нельзя даже удалить:
``JobViewSet.perform_destroy`` отвечает ``"Only ground truth jobs can be removed"``.
Поэтому обновить часть уже размеченного года можно лишь пересозданием всей задачи — как
это делается без потери ручной разметки, описано в ``scan_markup.cvat.publish``.

Картинки не грузятся по сети: задачи заводятся из share-каталога
(``ResourceType.SHARE``), который смонтирован в ``cvat_server`` как ``/home/django/share``.
Отсюда требование к ``--share-root`` — он должен совпадать с ``IMAGES_DIR`` в
``docker/.env``.
"""

import logging

from tqdm import tqdm

from ocr_utils.scan_markup.db.models import (
    KIND_COLOR,
    KIND_COLOR_TEXT,
    KIND_GRAYSCALE,
    KIND_STAMP_SUSPECT,
    MASK_HANDWRITING,
    MASK_LIBRARY_STAMP,
    MASK_OTHER_REMOVAL,
    POINT_EXLIBRIS,
)
from ocr_utils.scan_markup.geometry import to_cvat_rect

logger = logging.getLogger(__name__)

# Метки проекта. Цвета выбраны так: сканы жёлто-бежевые,
# поэтому берём холодные насыщенные тона, на тёплой бумаге они видны лучше всего;
# пурпур и голубой максимально далеки друг от друга и не сливаются при дальтонизме.
#
# color/grayscale — ДВУМЯ МЕТКАМИ, а не атрибутом одной: разные цвета рамок видно на
# кадре сразу, всю полосу можно проверить одним взглядом, тогда как атрибут пришлось бы
# открывать по каждому объекту отдельно.
LABEL_RASTER_COLOR = "Растр цветной"
LABEL_RASTER_GRAY = "Растр серый"
LABEL_STAMP = "Библиотечная печать"
# Прямоугольник, а не маска: автодетектор находит только рамку. Разметчик по ней либо
# обводит печать кистью под метку «Библиотечная печать», либо снимает прямоугольник, если
# это оказался не оттиск.
LABEL_STAMP_SUSPECT = "Подозрение на печать"

# Прочее, что разметчик убирает с полосы кистью. Три отдельные метки, а не одна с атрибутом:
# закрашивать их будет LaMa, и по видам их разносят не ради статистики, а потому что у них
# разная цена ошибки. Печать стоит на пустом поле, и лишний захват безобиден; рукописная
# надпись лежит поверх текста, и лишний захват сожрёт строку.
LABEL_HANDWRITING = "Рукописная надпись"
LABEL_OTHER_REMOVAL = "Прочее под удаление"
# Точка, а не прямоугольник: разметчик указывает МЕСТО вставки своего экслибриса, а размер
# знака задаётся при вклейке, не здесь.
LABEL_EXLIBRIS = "Экслибрис"

# Цветной типографский набор: заголовок, поздравление, анонс — не растр и не рисунок.
# Прямоугольник, потому что это область страницы, а не объект со сложным контуром.
LABEL_COLOR_TEXT = "Цветной текст или штрих"

# Цвета разнесены по кругу и все насыщенные: сканы жёлто-бежевые, и бледное на них теряется.
# Занятые тона — зелёный 150°, голубой 200°, пурпур 290°, оранжевый 25°, жёлтый 55°,
# красный 350°; точке достался единственный свободный участок, сине-фиолетовый 250°. Белый
# для точки не годится: на бумаге его не видно вовсе.
#
# Круг к седьмой метке кончился, и «Цветной текст» взял тёмно-розовый 335° — соседний с
# красным. Путаницы не будет: различать надо ПРЯМОУГОЛЬНИКИ между собой, а их всего четыре
# (зелёный, голубой, оранжевый, тёмно-розовый), маски же рисуются заливкой и с рамкой не
# сливаются.
LABELS = [
    {"name": LABEL_RASTER_COLOR, "type": "rectangle", "color": "#00E676"},  # ярко-зелёный
    {"name": LABEL_RASTER_GRAY, "type": "rectangle", "color": "#00B0FF"},  # голубой
    {"name": LABEL_STAMP_SUSPECT, "type": "rectangle", "color": "#FF6D00"},  # оранжевый
    {"name": LABEL_STAMP, "type": "mask", "color": "#D500F9"},  # пурпурный
    {"name": LABEL_HANDWRITING, "type": "mask", "color": "#FFEA00"},  # жёлтый
    {"name": LABEL_OTHER_REMOVAL, "type": "mask", "color": "#FF1744"},  # красный
    {"name": LABEL_EXLIBRIS, "type": "points", "color": "#651FFF"},  # сине-фиолетовый
    {"name": LABEL_COLOR_TEXT, "type": "rectangle", "color": "#C51162"},  # тёмно-розовый
]

# Метка -> значение колонки kind в базе и обратно.
LABEL_BY_KIND = {
    KIND_COLOR: LABEL_RASTER_COLOR,
    KIND_GRAYSCALE: LABEL_RASTER_GRAY,
    KIND_STAMP_SUSPECT: LABEL_STAMP_SUSPECT,
    KIND_COLOR_TEXT: LABEL_COLOR_TEXT,
}
KIND_BY_LABEL = {
    LABEL_RASTER_COLOR: KIND_COLOR,
    LABEL_RASTER_GRAY: KIND_GRAYSCALE,
    LABEL_STAMP_SUSPECT: KIND_STAMP_SUSPECT,
    LABEL_COLOR_TEXT: KIND_COLOR_TEXT,
}
MASK_KIND_BY_LABEL = {
    LABEL_STAMP: MASK_LIBRARY_STAMP,
    LABEL_HANDWRITING: MASK_HANDWRITING,
    LABEL_OTHER_REMOVAL: MASK_OTHER_REMOVAL,
}
POINT_KIND_BY_LABEL = {LABEL_EXLIBRIS: POINT_EXLIBRIS}

# Качество JPEG, которым CVAT пережимает кадры уже у себя. Картинки и так уменьшены и
# сохранены с quality=95, так что это второе пережатие — единственное заметное.
IMAGE_QUALITY = 70


def ensure_project(client, name: str) -> object:
    """Проект по имени: находит существующий или создаёт с нужными метками.

    Идемпотентность по ИМЕНИ: повторный прогон не должен
    заводить второй проект с той же разметкой.

    У найденного проекта метки ДОСЫЛАЮТСЯ: новый класс разметки, появившийся в ``LABELS``
    позже, иначе не попал бы в него никогда — а пересоздавать проект ради метки нельзя, в
    нём уже лежит работа разметчика. Досылаются только недостающие ПО ИМЕНИ; существующие не
    трогаются вовсе, потому что смена типа или цвета метки обесценила бы нарисованное под ней.
    """
    from cvat_sdk import models

    for project in client.projects.list():
        if project.name == name:
            logger.info("Проект %r уже есть (id=%s), переиспользую", name, project.id)
            add_missing_labels(client, project.id)
            return project

    project = client.projects.create(
        spec=models.ProjectWriteRequest(name=name, labels=[models.PatchedLabelRequest(**label) for label in LABELS])
    )
    logger.info("Проект %r создан (id=%s)", name, project.id)
    return project


def add_missing_labels(client, project_id: int) -> list[str]:
    """Заводит в проекте метки из ``LABELS``, которых там ещё нет. Возвращает их имена.

    Метка заводится ПАТЧЕМ ПРОЕКТА, а не через ``labels_api``: там есть только ``list``,
    ``retrieve``, ``partial_update`` и ``destroy``, создания нет вовсе. Сервер сливает
    присланный список с имеющимся — запись без ``id`` создаётся как новая, поэтому шлём
    только недостающие и ничего не теряем.
    """
    from cvat_sdk import models

    present = set(project_label_ids(client, project_id))
    missing = [label for label in LABELS if label["name"] not in present]
    if not missing:
        return []

    client.api_client.projects_api.partial_update(
        id=project_id,
        patched_project_write_request=models.PatchedProjectWriteRequest(
            labels=[models.PatchedLabelRequest(**label) for label in missing]
        ),
    )
    names = [label["name"] for label in missing]
    logger.warning("В проект id=%s дописаны метки: %s", project_id, ", ".join(names))
    return names


def project_label_ids(client, project_id: int) -> dict[str, int]:
    """``{имя метки: id}`` для проекта."""
    labels, _ = client.api_client.labels_api.list(project_id=project_id, page_size=100)
    return {label.name: label.id for label in labels.results}


# Имя, под которым живёт задача, пока пересоздание не доведено до конца. Задача с таким
# суффиксом — заведомо недоделанная: её создали, а старую ещё не удалили.
TEMP_SUFFIX = " (пересоздание)"


def year_task_name(year_name: str, issues: int, pages: int) -> str:
    """Имя задачи-года: ``1966 · выпусков 6 · полос 582``.

    Существительное перед числом не случайно: так не приходится согласовывать окончание
    («582 полосы», но «1189 полос»), а сортировка по имени остаётся сортировкой по году.
    """
    return f"{year_name} · выпусков {issues} · полос {pages}"


def find_task(client, project_id: int, name: str):
    """Задача проекта по ТОЧНОМУ имени или ``None``."""
    for task in client.tasks.list():
        if task.project_id == project_id and task.name == name:
            return task
    return None


def find_year_task(client, project_id: int, year_name: str, task_id: int | None = None):
    """Задача года: сперва по id из базы, и только потом по имени.

    По id — потому что имя задачи не вечно. В него входят число выпусков и полос, а они
    меняются при обновлении пака; и разметчик вправе переименовать задачу руками. Поиск
    только по имени превращал бы любое такое изменение в ТИХИЙ ДУБЛЬ: прежняя задача не
    нашлась бы, рядом выросла бы вторая с теми же кадрами, и разметка осталась бы в первой.

    Запасной поиск по имени нужен для задач, заведённых до появления id в базе: годится и
    голое ``1966``, и любое имя, начинающееся с года. Недоделанный двойник от прерванного
    пересоздания при этом исключается явно — он тоже начинается с года.
    """
    tasks = [task for task in client.tasks.list() if task.project_id == project_id]
    if task_id is not None:
        for task in tasks:
            if task.id == task_id:
                return task
    for task in tasks:
        if task.name.endswith(TEMP_SUFFIX):
            continue
        if task.name == year_name or task.name.startswith(f"{year_name} "):
            return task
    return None


def create_year_task(client, project_id: int, name: str, job_files: list[list[str]]):
    """Задача-год из share-каталога, с джобами по границам выпусков.

    ``job_files`` — список списков относительных путей внутри share, по одному списку на
    выпуск. Порядок сохраняется: плоское объединение идёт в ``resources``, оно же задаёт
    нумерацию кадров.
    """
    from cvat_sdk import models
    from cvat_sdk.core.proxies.tasks import ResourceType

    resources = [path for files in job_files for path in files]
    logger.info("Создаю задачу %r: выпусков %d, кадров %d", name, len(job_files), len(resources))
    return client.tasks.create_from_data(
        spec=models.TaskWriteRequest(name=name, project_id=project_id),
        resource_type=ResourceType.SHARE,
        resources=resources,
        data_params={"image_quality": IMAGE_QUALITY, "job_file_mapping": job_files},
    )


def frame_index_by_name(task) -> dict[str, int]:
    """``{имя кадра: индекс}`` по данным СЕРВЕРА.

    Индекс берётся отсюда, а не из позиции файла в переданном списке: нумерацию кадров
    определяет сервер, и полагаться на совпадение с нашим порядком — лишнее допущение,
    которое сломается молча и криво (разметка окажется не на тех полосах).
    """
    return {frame.name: index for index, frame in enumerate(task.get_frames_info())}


def raster_shapes(pages_with_regions, frames: dict[str, int], label_ids: dict[str, int]) -> list:
    """Предразметка: прямоугольники из базы -> шейпы CVAT.

    ``pages_with_regions`` — последовательность ``(page, [RasterRegion, ...])``. Полосы,
    которых нет среди кадров задачи, пропускаются: так прогон по подмножеству лет не
    падает на чужих полосах.

    Маски печатей не предзаливаются — автодетектора печатей нет, разметчик рисует их с нуля.
    """
    from cvat_sdk import models

    shapes = []
    for page, regions in pages_with_regions:
        frame = frames.get(page.cvat_rel_path)
        if frame is None:
            continue
        for region in regions:
            label_id = label_ids.get(LABEL_BY_KIND[region.kind])
            if label_id is None:
                continue
            x1, y1, x2, y2 = to_cvat_rect(
                region.x1, region.y1, region.x2, region.y2, page.divisor, page.cvat_width, page.cvat_height
            )
            if x2 - x1 < 1 or y2 - y1 < 1:  # выродилось при делении — рисовать нечего
                continue
            shapes.append(
                models.LabeledShapeRequest(type="rectangle", frame=frame, label_id=label_id, points=[x1, y1, x2, y2])
            )
    return shapes


def upload_preannotations(task, shapes: list) -> int:
    """Заливает предразметку в задачу, ЗАМЕНЯЯ имеющуюся. Возвращает число шейпов.

    Замена, а не добавление: повторный прогон ``to-cvat`` по уже размеченной задаче иначе
    удвоил бы каждый прямоугольник. Поэтому команда и требует ``--force-annotations``,
    чтобы перезалить разметку в задачу, которая уже существует.
    """
    from cvat_sdk import models

    if not shapes:
        return 0
    task.set_annotations(models.LabeledDataRequest(shapes=shapes))
    return len(shapes)


def assign_jobs_to_issues(task, issues) -> int:
    """Проставляет ``issue.cvat_job_id`` по порядку джобов задачи.

    Джобы задачи идут в том же порядке, что группы ``job_file_mapping``, то есть в порядке
    выпусков. Сверяем длины: если сервер нарезал иначе, лучше честно ничего не проставить,
    чем связать выпуски с чужими джобами.
    """
    jobs = sorted(task.get_jobs(), key=lambda job: job.start_frame)
    if len(jobs) != len(issues):
        logger.warning(
            "Задача %r: джобов %d, выпусков %d — не связываю их, порядок ненадёжен", task.name, len(jobs), len(issues)
        )
        return 0
    for job, issue in zip(jobs, issues):
        issue.cvat_job_id = job.id
    return len(jobs)


def assign_annotator(client, task, username: str) -> None:
    """Назначает все джобы задачи на аннотатора; молча пропускает, если его нет."""
    from cvat_sdk import models

    users, _ = client.api_client.users_api.list(search=username, page_size=100)
    user_id = next((user.id for user in users.results if user.username == username), None)
    if user_id is None:
        logger.warning("Пользователь %r не найден — задача остаётся без назначения", username)
        return
    for job in task.get_jobs():
        job.update(models.PatchedJobWriteRequest(assignee=user_id))


def fetch_shapes_by_frame(task) -> dict[str, list]:
    """Разметка задачи, разложенная по ИМЕНАМ кадров: ``{имя кадра: [шейп, ...]}``.

    Именно по именам, а не по номерам: пересозданная задача нумерует кадры заново, и
    номера старой задачи в новой означают уже другие полосы. Имя кадра — путь внутри
    share — устойчиво к пересозданию.
    """
    names = {index: frame.name for index, frame in enumerate(task.get_frames_info())}
    by_frame: dict[str, list] = {}
    annotations = task.get_annotations()
    for shape in annotations.shapes:
        name = names.get(shape.frame)
        if name is None:
            continue
        by_frame.setdefault(name, []).append(shape)
    return by_frame


def shapes_to_requests(by_frame: dict[str, list], frames: dict[str, int], skip_names: set[str]) -> list:
    """Сохранённые шейпы -> запросы на заливку в НОВУЮ задачу.

    ``frames`` — нумерация кадров новой задачи, ``skip_names`` — кадры, чью разметку
    переносить нельзя (файл под ними изменился, и обведённое на старом кадре к новому
    отношения не имеет).

    Геометрию не трогаем вообще. Уменьшенная копия готовится тем же делителем, что и
    раньше, значит кадр в новой задаче попиксельно тот же, и координаты — что углы
    прямоугольников, что RLE масок — остаются верными без единого пересчёта.

    ПОЛОСА, ПЕРЕЕХАВШАЯ В ДРУГОЙ ВЫПУСК, ищется по имени файла. Имя кадра — это путь внутри
    share, и у переехавшей полосы оно другое: ``пак-1/1971/05/IMG_0053_1L.jpg`` стало
    ``пак-1/1971/04/IMG_0053_1L.jpg``. Сверка по полному пути такую разметку молча теряла
    (замер: 82 шейпа из 83 переехали, один пропал). Поэтому для не нашедшихся имён идёт
    вторая попытка — по basename, и только если он в новой задаче ЕДИНСТВЕННЫЙ: полоса могла
    и правда исчезнуть, а угадывать, на какой из двух одинаковых кадров лить чужую разметку,
    нельзя.
    """
    from cvat_sdk import models

    by_basename: dict[str, list[str]] = {}
    for name in frames:
        by_basename.setdefault(name.rsplit("/", 1)[-1], []).append(name)

    requests = []
    for name, shapes in by_frame.items():
        if name in skip_names:
            continue
        frame = frames.get(name)
        if frame is None:
            candidates = by_basename.get(name.rsplit("/", 1)[-1], [])
            if len(candidates) != 1:
                continue
            frame = frames[candidates[0]]
            logger.warning("Полоса переехала: разметка с %r перенесена на %r", name, candidates[0])
        for shape in shapes:
            requests.append(
                models.LabeledShapeRequest(
                    type=shape.type,
                    frame=frame,
                    label_id=shape.label_id,
                    points=list(shape.points),
                    occluded=shape.occluded,
                    outside=shape.outside,
                    z_order=shape.z_order,
                    group=shape.group,
                    rotation=shape.rotation,
                    attributes=[
                        models.AttributeValRequest(spec_id=attr.spec_id, value=attr.value) for attr in shape.attributes
                    ],
                )
            )
    return requests


def job_states(task) -> list[tuple[str, str]]:
    """``[(state, stage), ...]`` по джобам задачи В ПОРЯДКЕ КАДРОВ."""
    return [(str(job.state), str(job.stage)) for job in sorted(task.get_jobs(), key=lambda job: job.start_frame)]


def apply_job_states(task, states: list[tuple[str, str]], reset: set[int] | None = None) -> int:
    """Проставляет джобам новой задачи состояния старой; возвращает число изменённых.

    Состояние джоба — это ручная отметка «выпуск размечен», и разметка её не содержит:
    в выгрузке шейпов её нет, а пересоздание заводит джобы заново, то есть все в ``new``.
    Замер на паке-1: у года 1971 так потерялись пять ``completed`` и один ``in progress``,
    и восстанавливать их пришлось из дампа базы CVAT.

    Сопоставление ПОЗИЦИОННОЕ, а не по номерам кадров: пересоздают год как раз тогда, когда
    состав выпуска изменился, и границы кадров уже другие. Порядок выпусков при этом тот же.

    ``reset`` — позиции джобов, которым «завершён» больше не подходит: в выпуск добавилась
    полоса, которой разметчик там не видел. Такие получают «в работе».
    """
    from cvat_sdk import models
    from cvat_sdk.api_client.model.operation_status import OperationStatus

    jobs = sorted(task.get_jobs(), key=lambda job: job.start_frame)
    if len(jobs) != len(states):
        logger.warning(
            "Задача %r: джобов %d, сохранённых состояний %d — состояния не восстанавливаю",
            task.name,
            len(jobs),
            len(states),
        )
        return 0

    reset = reset or set()
    changed = 0
    for index, (job, (state, stage)) in enumerate(zip(jobs, states)):
        target = "in progress" if index in reset and state == "completed" else state
        if (str(job.state), str(job.stage)) == (target, stage):
            continue
        job.update(models.PatchedJobWriteRequest(state=OperationStatus(target), stage=models.JobStage(stage)))
        changed += 1
    return changed


def rename_task(task, name: str) -> None:
    """Переименовывает задачу."""
    from cvat_sdk import models

    task.update(models.PatchedTaskWriteRequest(name=name))
