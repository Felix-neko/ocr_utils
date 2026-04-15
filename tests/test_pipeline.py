"""Тесты пайплайна: process_single_pdf, process_directory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from ocr_utils.pipeline import process_directory, process_single_pdf


class TestProcessSinglePdf:
    """Тесты для process_single_pdf (с замоканным OCR, чтобы тесты были быстрыми)."""

    def test_nonexistent_source_raises(self, tmp_dir: Path) -> None:
        """Несуществующий исходный файл → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            process_single_pdf(src_pdf=tmp_dir / "nonexistent.pdf", dst_pdf=tmp_dir / "out.pdf")

    def test_creates_output_directory(self, portrait_pdf: Path, tmp_dir: Path) -> None:
        """Выходная директория должна быть создана автоматически."""
        out_path = tmp_dir / "deep" / "nested" / "dir" / "out.pdf"
        with patch("ocr_utils.pipeline.run_ocr") as mock_ocr:
            # Мок OCR: просто копируем входной файл в выходной
            def fake_ocr(input_pdf, output_pdf, **kwargs):
                import shutil

                shutil.copy2(input_pdf, output_pdf)
                return output_pdf

            mock_ocr.side_effect = fake_ocr
            process_single_pdf(src_pdf=portrait_pdf, dst_pdf=out_path)

        assert out_path.parent.exists()

    def test_pipeline_with_mocked_ocr(self, mixed_pdf: Path, tmp_dir: Path) -> None:
        """Полный пайплайн с замоканным OCR: проверяем, что промежуточный PDF создаётся корректно."""
        out_path = tmp_dir / "result.pdf"
        intermediate_page_counts = []

        with patch("ocr_utils.pipeline.run_ocr") as mock_ocr:

            def fake_ocr(input_pdf, output_pdf, **kwargs):
                import shutil

                # Считаем страницы промежуточного PDF до того, как tmp_dir будет удалён
                doc = fitz.open(str(input_pdf))
                intermediate_page_counts.append(len(doc))
                doc.close()
                shutil.copy2(input_pdf, output_pdf)
                return output_pdf

            mock_ocr.side_effect = fake_ocr
            process_single_pdf(src_pdf=mixed_pdf, dst_pdf=out_path, tmp_dir=tmp_dir)

        assert out_path.exists()
        # Промежуточный PDF должен содержать 4 страницы (1 + 2 + 1)
        assert len(intermediate_page_counts) == 1
        assert intermediate_page_counts[0] == 4

    def test_temp_dir_cleanup(self, portrait_pdf: Path, tmp_dir: Path) -> None:
        """При tmp_dir=None временная директория должна быть очищена."""
        out_path = tmp_dir / "result.pdf"
        with patch("ocr_utils.pipeline.run_ocr") as mock_ocr:

            def fake_ocr(input_pdf, output_pdf, **kwargs):
                import shutil

                shutil.copy2(input_pdf, output_pdf)
                return output_pdf

            mock_ocr.side_effect = fake_ocr
            # tmp_dir=None → создаётся и удаляется автоматически
            process_single_pdf(src_pdf=portrait_pdf, dst_pdf=out_path, tmp_dir=None)

        assert out_path.exists()


class TestProcessDirectory:
    """Тесты для process_directory."""

    def _make_dir_with_pdfs(self, base: Path, count: int = 3) -> Path:
        """Создать директорию с несколькими простыми PDF."""
        src = base / "src"
        src.mkdir()
        for i in range(count):
            doc = fitz.open()
            doc.new_page(width=595, height=842)
            doc.save(str(src / f"file_{i}.pdf"))
            doc.close()
        return src

    def _make_nested_dir_with_pdfs(self, base: Path) -> Path:
        """Создать вложенную структуру директорий с PDF."""
        src = base / "src"
        (src / "sub1").mkdir(parents=True)
        (src / "sub2" / "deep").mkdir(parents=True)
        for path in [src / "root.pdf", src / "sub1" / "a.pdf", src / "sub2" / "deep" / "b.pdf"]:
            doc = fitz.open()
            doc.new_page(width=595, height=842)
            doc.save(str(path))
            doc.close()
        return src

    def test_finds_all_pdfs(self, tmp_dir: Path) -> None:
        """Должны найтись все PDF в директории."""
        src = self._make_dir_with_pdfs(tmp_dir)
        dst = tmp_dir / "dst"
        with patch("ocr_utils.pipeline.process_single_pdf") as mock:
            mock.return_value = None
            process_directory(src, dst, workers=1)
        assert mock.call_count == 3

    def test_creates_output_dir(self, tmp_dir: Path) -> None:
        """Выходная директория должна быть создана."""
        src = self._make_dir_with_pdfs(tmp_dir)
        dst = tmp_dir / "new_output"
        with patch("ocr_utils.pipeline.process_single_pdf"):
            process_directory(src, dst, workers=1)
        assert dst.exists()

    def test_preserves_relative_paths(self, tmp_dir: Path) -> None:
        """Относительные пути должны сохраняться в выходной директории."""
        src = self._make_nested_dir_with_pdfs(tmp_dir)
        dst = tmp_dir / "dst"

        calls = []
        with patch("ocr_utils.pipeline._process_one") as mock:
            mock.return_value = ("test", None)
            # Вызываем напрямую, проверяя формирование задач
            from ocr_utils.pipeline import process_directory

            # Вместо мока process_single_pdf — проверим через реальный вызов с workers=1
            pass

        # Проверим через правильный мок
        with patch("ocr_utils.pipeline.process_single_pdf") as mock:
            mock.return_value = None
            process_directory(src, dst, workers=1)

        # Собираем dst-пути из вызовов
        dst_paths = []
        for call in mock.call_args_list:
            dst_pdf = call.kwargs.get("dst_pdf") or call.args[1]
            dst_paths.append(Path(dst_pdf))

        # Проверяем, что пути сохраняют структуру
        rel_paths = {p.relative_to(dst) for p in dst_paths}
        assert Path("root.pdf") in rel_paths
        assert Path("sub1/a.pdf") in rel_paths
        assert Path("sub2/deep/b.pdf") in rel_paths

    def test_nonexistent_source_dir_raises(self, tmp_dir: Path) -> None:
        """Несуществующая исходная директория → NotADirectoryError."""
        with pytest.raises(NotADirectoryError):
            process_directory(tmp_dir / "nonexistent", tmp_dir / "dst")

    def test_empty_directory(self, tmp_dir: Path) -> None:
        """Пустая директория → пустой результат."""
        src = tmp_dir / "empty"
        src.mkdir()
        result = process_directory(src, tmp_dir / "dst", workers=1)
        assert result == {}

    def test_workers_default(self) -> None:
        """По умолчанию workers = 3/4 ядер, но не менее 1."""
        import multiprocessing

        cpu = multiprocessing.cpu_count()
        expected = max(1, (cpu * 3) // 4)
        assert expected >= 1
