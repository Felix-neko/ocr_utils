"""Метки проекта CVAT: досылка недостающих в уже заведённый проект.

Настоящий сервер не нужен — проверяется наша половина: что шлётся только недостающее и что
существующие метки не трогаются. Всё, что делает сервер, сведено к списку меток.
"""

from ocr_utils.scan_markup.cvat.project import LABELS, add_missing_labels, ensure_project


class _Label:
    def __init__(self, label_id: int, name: str) -> None:
        self.id = label_id
        self.name = name


class _Page:
    def __init__(self, results) -> None:
        self.results = results


class _LabelsApi:
    def __init__(self, labels) -> None:
        self.labels = labels

    def list(self, project_id: int, page_size: int = 100):  # noqa: ARG002 — сигнатура SDK
        return _Page(self.labels), None


class _ProjectsApi:
    def __init__(self, owner) -> None:
        self.owner = owner
        self.patches = []

    def partial_update(self, id: int, patched_project_write_request):  # noqa: A002 — имя из SDK
        self.patches.append((id, patched_project_write_request))
        for label in patched_project_write_request.labels:
            self.owner.labels.append(_Label(100 + len(self.owner.labels), label["name"]))


class _ApiClient:
    def __init__(self, labels) -> None:
        self.labels = [_Label(index, name) for index, name in enumerate(labels, start=1)]
        self.labels_api = _LabelsApi(self.labels)
        self.projects_api = _ProjectsApi(self)


class _Project:
    def __init__(self, project_id: int, name: str) -> None:
        self.id = project_id
        self.name = name


class _Projects:
    def __init__(self, projects) -> None:
        self._projects = projects
        self.created = []

    def list(self):
        return self._projects

    def create(self, spec):
        self.created.append(spec)
        return _Project(99, spec.name)


class _Client:
    def __init__(self, projects, labels) -> None:
        self.projects = _Projects(projects)
        self.api_client = _ApiClient(labels)


def test_missing_labels_are_added_to_an_existing_project() -> None:
    """В проекте с частью меток заводятся ровно недостающие."""
    have = [LABELS[0]["name"], LABELS[1]["name"]]
    client = _Client([_Project(1, "пак-1")], have)

    added = add_missing_labels(client, 1)

    assert added == [label["name"] for label in LABELS[2:]]
    sent = client.api_client.projects_api.patches[0][1].labels
    assert [label["name"] for label in sent] == added, "существующие метки слать не надо"


def test_full_project_is_left_alone() -> None:
    """Все метки на месте — сервер не трогаем вовсе."""
    client = _Client([_Project(1, "пак-1")], [label["name"] for label in LABELS])

    assert add_missing_labels(client, 1) == []
    assert client.api_client.projects_api.patches == []


def test_ensure_project_tops_up_labels_of_a_found_project() -> None:
    """Найденный по имени проект получает недостающие метки, а не создаётся заново."""
    client = _Client([_Project(1, "пак-1")], [LABELS[0]["name"]])

    project = ensure_project(client, "пак-1")

    assert project.id == 1 and client.projects.created == []
    assert client.api_client.projects_api.patches, "метки должны были дописаться"


def test_new_project_is_created_with_all_labels() -> None:
    """Проекта нет — создаётся сразу со всем набором меток."""
    client = _Client([], [])

    project = ensure_project(client, "пак-1")

    assert project.id == 99
    assert len(client.projects.created[0].labels) == len(LABELS)
