"""Подключение стороннего ONNX-классификатора по описанию."""

from __future__ import annotations

import json

import pytest

from ocr_utils.page_layout.orientation.detectors import optional


@pytest.fixture
def model(tmp_path, monkeypatch):
    """Пара «веса + описание», как её кладёт тот, кто модель достал."""
    weights = tmp_path / "doc_ori.onnx"
    weights.write_bytes(b"")
    weights.with_suffix(".json").write_text(
        json.dumps(
            {
                "size": [224, 224],
                "layout": "NCHW",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "classes": [0, 90, 180, 270],
                "meaning": "state",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(optional.MODEL_ENV, str(weights))
    return weights


def test_detector_is_unavailable_without_the_environment_variable(monkeypatch):
    monkeypatch.delenv(optional.MODEL_ENV, raising=False)
    assert not optional.onnx_available()


def test_detector_is_unavailable_when_the_description_is_missing(tmp_path, monkeypatch):
    """Веса без описания — это неизвестная нормировка и неизвестный порядок классов.

    Запускать такую модель нельзя: она не упадёт, а тихо выдаст правдоподобную чушь.
    """
    weights = tmp_path / "doc_ori.onnx"
    weights.write_bytes(b"")
    monkeypatch.setenv(optional.MODEL_ENV, str(weights))
    assert not optional.onnx_available()


def test_missing_weights_do_not_crash_the_run(model, monkeypatch):
    """Сломанная сторонняя модель обязана превратиться в «сказать нечего», а не в падение."""
    monkeypatch.setenv(optional.MODEL_ENV, str(model.parent / "нет-такого.onnx"))
    verdicts = optional.OnnxOrientation()([object(), object()])
    assert len(verdicts) == 2
    assert all(verdict.confidence == 0.0 and verdict.note for verdict in verdicts)


class StubSession:
    """Подставная ONNX-сессия: отдаёт заданные логиты, ничего не считая."""

    def __init__(self, logits):
        self._logits = logits

    def get_inputs(self):
        return [type("Input", (), {"name": "x"})()]

    def run(self, _outputs, feed):
        self.fed = feed["x"]
        return [self._logits]


def run_with(monkeypatch, config, logits):
    import numpy as np

    predictor = optional.OnnxOrientation()
    session = StubSession(np.asarray(logits, dtype=np.float32))

    def fake_load(self=predictor):
        self._config = config
        self._session = session
        return session

    monkeypatch.setattr(predictor, "_load", fake_load)
    images = [np.zeros((300, 200, 3), dtype=np.uint8) for _ in logits]
    return predictor(images), session


BASE_CONFIG = {
    "size": [224, 224],
    "layout": "NCHW",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "classes": [0, 90, 180, 270],
}


def test_state_meaning_is_inverted_into_a_correction(monkeypatch):
    """«Угол, НА КОТОРЫЙ повёрнуто» — значит выпрямлять надо в обратную сторону."""
    verdicts, _ = run_with(monkeypatch, {**BASE_CONFIG, "meaning": "state"}, [[0.0, 9.0, 0.0, 0.0]])
    assert verdicts[0].rotate_cw == 270


def test_correction_meaning_is_taken_as_is(monkeypatch):
    verdicts, _ = run_with(monkeypatch, {**BASE_CONFIG, "meaning": "correction"}, [[0.0, 9.0, 0.0, 0.0]])
    assert verdicts[0].rotate_cw == 90


def test_input_is_shaped_and_normalized_by_the_description(monkeypatch):
    """Размер и раскладка берутся из описания: угадывать их нельзя, чужая нормировка
    не роняет модель, а тихо превращает её ответы в правдоподобную чушь."""
    _, session = run_with(monkeypatch, {**BASE_CONFIG, "meaning": "correction"}, [[9.0, 0.0, 0.0, 0.0]])
    assert session.fed.shape == (1, 3, 224, 224)
