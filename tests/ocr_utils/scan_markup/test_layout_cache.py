"""Кэш разметки surya: попадание — модель не зовётся, промах и битый файл — пересчёт и перезапись."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from ocr_utils.scan_markup.detection import layout_cache
from ocr_utils.scan_markup.detection.layout_cache import CachedLayout, load, picture_boxes, save
from ocr_utils.scan_markup.detection.page import PageAnalysis, PageOptions, page_layout_for
from ocr_utils.page_layout.geometry import Box
from ocr_utils.scan_markup.table_detection.layout import Block, PageLayout

REL = "1966/01/IMG_0017_2R.tif"


def _layout() -> PageLayout:
    return PageLayout(
        (Block("Table", 0.9, Box(10, 20, 300, 400)), Block("Picture", 0.8, Box(0, 500, 200, 700))), 873, 1512
    )


class _ForeignThing:
    """Объект «чужого пакета»: при чтении кэша его класс не должен импортироваться."""

    def __init__(self) -> None:
        self.payload = [1, 2, 3]


@pytest.fixture
def foreign(monkeypatch):
    """Класс, который pickle запишет как ``surya.layout.schema._ForeignThing`` — как настоящий LayoutResult."""
    import surya.layout.schema as schema

    monkeypatch.setattr(_ForeignThing, "__module__", schema.__name__)
    monkeypatch.setattr(schema, "_ForeignThing", _ForeignThing, raising=False)
    return _ForeignThing


def test_round_trip_keeps_layout_and_drops_raw(tmp_path: Path, foreign) -> None:
    saved = save(tmp_path, REL, CachedLayout(_layout(), dpi=150, raw=foreign()))
    assert saved == tmp_path / "1966/01/IMG_0017_2R.pkl"
    got = load(tmp_path, REL)
    assert got is not None and got.layout == _layout() and got.dpi == 150
    assert got.raw is None, "сырой ответ surya в конвейере не нужен и не восстанавливается"


def test_foreign_classes_are_stubbed_not_imported(tmp_path: Path, foreign) -> None:
    """Кэш исследований хранит сырой LayoutResult; воркер читает его без импорта surya."""
    save(tmp_path, REL, CachedLayout(_layout(), raw=foreign()))
    # Модуль, которого нет вовсе, — самый строгий случай: обычный pickle упал бы на импорте.
    # Имя той же длины: в pickle строка лежит с префиксом длины.
    path = layout_cache.cache_path(tmp_path, REL)
    data = path.read_bytes().replace(b"surya.layout.schema", b"no.module.here.xyz1")
    assert data != path.read_bytes()
    path.write_bytes(data)
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(data)
    got = load(tmp_path, REL)
    assert got is not None and got.layout == _layout()


def test_missing_file_is_a_miss(tmp_path: Path) -> None:
    assert load(tmp_path, REL) is None
    assert load(None, REL) is None


@pytest.mark.parametrize(
    "spoil",
    [
        lambda path: path.write_bytes(b"\x80\x04garbage"),
        lambda path: path.write_bytes(pickle.dumps({"scan_rel_path": REL})),
        lambda path: path.write_bytes(
            pickle.dumps({"scan_rel_path": "1966/01/other.tif", "layout": _layout().to_json()})
        ),
        lambda path: path.write_bytes(pickle.dumps(["не словарь"])),
    ],
    ids=["битый pickle", "без layout", "чужой scan_rel_path", "не словарь"],
)
def test_corrupt_cache_is_a_miss_and_gets_overwritten(tmp_path: Path, spoil, caplog) -> None:
    path = layout_cache.cache_path(tmp_path, REL)
    path.parent.mkdir(parents=True)
    spoil(path)
    with caplog.at_level("WARNING"):
        assert load(tmp_path, REL) is None
    assert "разобрана заново" in caplog.text
    save(tmp_path, REL, CachedLayout(_layout()))
    assert load(tmp_path, REL) is not None
    assert not list(path.parent.glob("*.part")), "временный файл записи должен исчезнуть"


def test_picture_boxes_are_rescaled_to_the_work_copy() -> None:
    cached = CachedLayout(_layout())
    # Копия 1/4 на пиксель уже, чем картинка кэша: рамки масштабируются, а не берутся как есть.
    boxes = picture_boxes(cached, 872, 1510)
    assert boxes == [Box(0, 499, 200, 699)]
    assert picture_boxes(cached, 873, 1512) == [Box(0, 500, 200, 700)]


class _Detector:
    """Дублёр модели: при вызове падает — попадание в кэш обязано обойтись без него."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, bgr):
        self.calls += 1
        raise AssertionError("модель звать нельзя: разметка есть в кэше")


class _Predicting:
    """Дублёр модели, который отвечает: промах кэша обязан дойти до него и записать кэш."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, bgr):
        from types import SimpleNamespace

        self.calls += 1
        h, w = bgr.shape[:2]
        block = SimpleNamespace(bbox=[0.0, 0.0, w / 2, h / 2], label="Table", confidence=0.7, polygon=[])
        return SimpleNamespace(bboxes=[block]), 1.0


def _analysis(work_shape=(1512, 873)) -> PageAnalysis:
    analysis = PageAnalysis(REL, 3, width=work_shape[1] * 4, height=work_shape[0] * 4, dpi=600)
    analysis.work = np.full((*work_shape, 3), 245, np.uint8)
    analysis.work_gray = analysis.work[..., 0].copy()
    return analysis


def test_cache_hit_skips_the_model_and_miss_writes_the_cache(tmp_path: Path) -> None:
    options = PageOptions(layout_cache_dir=tmp_path)
    analysis = _analysis()
    analysis.layout = load(tmp_path, REL)  # промах: файла ещё нет
    assert analysis.layout is None

    model = _Predicting()
    got = page_layout_for(analysis, model, options)
    assert model.calls == 1 and got is not None
    assert got.layout.blocks[0].label == "Table" and got.dpi == 150
    assert load(tmp_path, REL) is not None, "промах обязан записать кэш"

    hit = _analysis()
    hit.layout = load(tmp_path, REL)
    assert page_layout_for(hit, _Detector(), options) is hit.layout
