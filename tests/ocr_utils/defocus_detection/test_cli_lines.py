"""Сквозные проверки режима --use-surya-lines.

Сама surya здесь не запускается: она требует GPU и весов, а проверять надо не её, а
обвязку вокруг неё. Поэтому детектор подменяется фейком, который отдаёт заранее известные
полигоны, — и тогда всё остальное (отбор, замер, зональная сетка, отчёты, кэш) становится
проверяемым обычным тестом.
"""

import csv

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from ocr_utils.defocus_detection import cli
from ocr_utils.defocus_detection.lines.detect import DetectCache, DetectParams
from ocr_utils.defocus_detection.lines.regions import LineRegion

from .pages import blur, draw_text_lines

PAGE = dict(height=1200, width=1200, stroke=4, line_height=26, columns=5)


class FakeDetector:
    """Детектор-заглушка: отдаёт полигоны, положенные рядом с картинкой.

    Attributes:
        calls: Сколько раз к нему обратились — по этому числу проверяется кэш.
    """

    def __init__(self, polygons_by_stem, params=None, cache=None):
        self._polygons = polygons_by_stem
        self._params = params or DetectParams()
        self._cache = cache
        self.calls = 0

    @property
    def params(self):
        """Параметры детекции."""
        return self._params

    def detect(self, path, gray):
        """Возвращает области строк для кадра, при наличии кэша — через него.

        Args:
            path: Путь к изображению.
            gray: Полутоновый кадр.

        Returns:
            Список ``LineRegion``.
        """
        if self._cache is not None:
            cached = self._cache.load(path, self._params)
            if cached is not None:
                return cached
        self.calls += 1
        regions = [LineRegion(polygon=p, confidence=0.9) for p in self._polygons[path.stem]]
        if self._cache is not None:
            self._cache.store(path, self._params, regions)
        return regions


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """Папка из шести полос (две заметно мягче) и подменённый детектор строк.

    Returns:
        Кортеж (папка, фейковый детектор).
    """
    directory = tmp_path / "scans"
    directory.mkdir()
    polygons = {}
    for index in range(4):
        page, polys = draw_text_lines(seed=index, **PAGE)
        stem = f"sharp_{index:02d}"
        cv2.imwrite(str(directory / f"{stem}.png"), blur(page, 0.7))
        polygons[stem] = polys
    for index in range(2):
        page, polys = draw_text_lines(seed=100 + index, **PAGE)
        stem = f"soft_{index:02d}"
        cv2.imwrite(str(directory / f"{stem}.png"), blur(page, 2.0))
        polygons[stem] = polys

    holder = {}

    def factory(params=None, cache=None):
        holder["detector"] = FakeDetector(polygons, params=params, cache=cache)
        return holder["detector"]

    monkeypatch.setattr(cli, "LineDetector", factory)
    return directory, holder


def run(*args):
    """Запускает CLI и возвращает результат.

    Args:
        *args: Аргументы командной строки.

    Returns:
        Объект ``click.testing.Result``.
    """
    result = CliRunner().invoke(cli.main, [*args, "--workers", "1", "--quiet"])
    assert result.exit_code == 0, result.output + str(result.exception)
    return result


def test_soft_pages_come_first(folder):
    """Мягкие кадры должны всплыть наверх и в режиме по строкам тоже."""
    directory, _ = folder
    output = run(str(directory), "--use-surya-lines").output
    overall = output.split("2. ЗОНАЛЬНЫЙ")[0]
    order = [line.split()[-1] for line in overall.splitlines() if line.strip().endswith(".png")]
    assert order[0].startswith("soft_") and order[1].startswith("soft_")


def test_report_gains_readability_and_line_columns(folder):
    """В режиме по строкам появляются колонки читаемости и числа строк."""
    directory, _ = folder
    output = run(str(directory), "--use-surya-lines").output
    assert "σ/высота" in output
    assert "строк" in output
    assert "перепад (тайлы)" in output


def test_without_the_flag_nothing_changes(folder):
    """Без флага отчёт остаётся прежним — новых колонок нет, детектор не вызывается."""
    directory, holder = folder
    output = run(str(directory)).output
    assert "σ/высота" not in output
    assert "перепад (тайлы)" not in output
    assert "detector" not in holder, "детектор не должен создаваться без флага"


def test_csv_gets_line_and_tile_columns(folder, tmp_path):
    """CSV пополняется колонками по строкам и развёрнутой картой тайлов."""
    directory, _ = folder
    csv_path = tmp_path / "out.csv"
    run(str(directory), "--use-surya-lines", "--csv", str(csv_path))

    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    for column in ("score_norm", "lines_measured", "lines_detected", "chunks", "tile_drop", "tile_zone"):
        assert column in rows[0]
    assert "tile_sharp_r1c1" in rows[0] and "tile_sharp_r3c3" in rows[0]
    assert float(rows[0]["lines_measured"]) > 0


@pytest.mark.parametrize("side", [3, 4])
def test_zonal_grid_side_is_configurable(folder, tmp_path, side):
    """--zonal-tiles задаёт сторону сетки, и CSV разворачивается под неё."""
    directory, _ = folder
    csv_path = tmp_path / "out.csv"
    run(str(directory), "--use-surya-lines", "--zonal-tiles", str(side), "--csv", str(csv_path))
    with csv_path.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert f"tile_sharp_r{side}c{side}" in header
    assert f"tile_sharp_r{side + 1}c{side + 1}" not in header


def test_detect_cache_spares_the_second_run(folder, tmp_path):
    """Повторный прогон той же папки не должен трогать детектор.

    Ради этого кэш и заведён: детекция — самая дорогая часть, а пороги метрик
    подбираются итеративно, прогон за прогоном по одним и тем же файлам.
    """
    directory, holder = folder
    cache_dir = tmp_path / "cache"

    run(str(directory), "--use-surya-lines", "--detect-cache", str(cache_dir))
    first = holder["detector"].calls
    assert first == 6

    run(str(directory), "--use-surya-lines", "--detect-cache", str(cache_dir))
    assert holder["detector"].calls == 0, "второй прогон обязан читать кэш"


def test_cache_is_invalidated_by_detection_params(folder, tmp_path):
    """Смена параметров детекции обязана делать старые записи невидимыми."""
    directory, holder = folder
    cache_dir = tmp_path / "cache"

    run(str(directory), "--use-surya-lines", "--detect-cache", str(cache_dir))
    run(str(directory), "--use-surya-lines", "--detect-cache", str(cache_dir), "--surya-tile-side", "900")
    assert holder["detector"].calls == 6, "другой тайл детекции — другой результат, кэш не подходит"


def test_debug_dir_writes_one_overlay_per_frame(folder, tmp_path):
    """--debug-dir выгружает наложение по каждому кадру."""
    directory, _ = folder
    debug_dir = tmp_path / "debug"
    run(str(directory), "--use-surya-lines", "--debug-dir", str(debug_dir))

    overlays = sorted(debug_dir.glob("*.jpg"))
    assert len(overlays) == 6
    image = cv2.imread(str(overlays[0]))
    assert image is not None and image.ndim == 3


def test_debug_dir_requires_the_flag(folder, tmp_path):
    """--debug-dir без --use-surya-lines — ошибка, а не молчаливо пустая папка."""
    directory, _ = folder
    result = CliRunner().invoke(cli.main, [str(directory), "--debug-dir", str(tmp_path / "d"), "--workers", "1", "-q"])
    assert result.exit_code != 0
    assert "--use-surya-lines" in result.output


def test_cache_survives_a_corrupted_entry(folder, tmp_path):
    """Битый файл кэша не должен ронять прогон — он просто считается заново."""
    directory, holder = folder
    cache_dir = tmp_path / "cache"
    run(str(directory), "--use-surya-lines", "--detect-cache", str(cache_dir))

    for entry in cache_dir.rglob("*.json"):
        entry.write_text("{это не json", encoding="utf-8")
    run(str(directory), "--use-surya-lines", "--detect-cache", str(cache_dir))
    assert holder["detector"].calls == 6


def test_cache_roundtrip_preserves_polygons(tmp_path):
    """Кэш обязан возвращать ровно те полигоны, что в него положили."""
    cache = DetectCache(tmp_path / "cache")
    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), np.zeros((32, 32), dtype=np.uint8))
    params = DetectParams()

    original = [LineRegion(polygon=np.array([[1.0, 2.0], [30.0, 3.0], [30.0, 20.0], [1.0, 19.0]]), confidence=0.75)]
    assert cache.load(image, params) is None
    cache.store(image, params, original)

    restored = cache.load(image, params)
    assert restored is not None and len(restored) == 1
    assert np.allclose(restored[0].polygon, original[0].polygon)
    assert restored[0].confidence == pytest.approx(0.75)
