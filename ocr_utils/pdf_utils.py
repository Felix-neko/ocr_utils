from pathlib import Path
from typing import Iterable, Optional, List
import img2pdf


def files_to_pdf(source_paths: Iterable[Path], dest_path: Path) -> None:
    """
    Собирает PDF из пачки JPEG-файлов без переконвертации (lossless).
    
    Использует img2pdf для встраивания JPEG напрямую в PDF без декодирования/перекодирования.
    Размер PDF будет примерно равен сумме размеров исходных JPEG плюс небольшой overhead PDF-контейнера.
    
    Args:
        source_paths: Итерируемая коллекция путей к исходным изображениям
        dest_path: Путь к выходному PDF-файлу
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    source_list = sorted(source_paths)
    if not source_list:
        raise ValueError("Список исходных файлов пуст")
    
    with open(dest_path, "wb") as f:
        f.write(img2pdf.convert([str(p) for p in source_list]))


def dir_to_pdf(
    source_dir: Path,
    dest_path: Path,
    prefixes: Optional[List[str]] = None,
    suffixes: Optional[List[str]] = None,
    extensions: Optional[List[str]] = None,
) -> None:
    """
    Собирает PDF из изображений в директории с фильтрацией по префиксам и суффиксам.
    
    Args:
        source_dir: Путь к входной директории
        dest_path: Путь к выходному PDF-файлу
        prefixes: Список префиксов для фильтрации файлов (по умолчанию ["_"])
        suffixes: Список суффиксов перед расширением (по умолчанию ["-4x"])
        extensions: Список расширений файлов (по умолчанию ["jpg", "jpeg"])
    """
    if prefixes is None:
        prefixes = ["_"]
    if suffixes is None:
        suffixes = ["-4x"]
    if extensions is None:
        extensions = ["jpg", "jpeg"]
    
    extensions_lower = [ext.lower() for ext in extensions]
    
    matching_files = []
    for file_path in source_dir.iterdir():
        if not file_path.is_file():
            continue
        
        file_ext = file_path.suffix.lower().lstrip('.')
        if file_ext not in extensions_lower:
            continue
        
        filename = file_path.name
        
        prefix_match = any(filename.startswith(prefix) for prefix in prefixes)
        if not prefix_match:
            continue
        
        stem = file_path.stem
        suffix_match = any(stem.endswith(suffix) for suffix in suffixes)
        if not suffix_match:
            continue
        
        matching_files.append(file_path)
    
    if not matching_files:
        raise ValueError(f"Не найдено файлов в {source_dir} с заданными критериями")
    
    files_to_pdf(matching_files, dest_path)


if __name__ == "__main__":
    root_dir = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989) [pics_only]")
    
    for year_dir in sorted(root_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        
        print(f"Обрабатываю год: {year_dir.name}")
        
        for journal_dir in sorted(year_dir.iterdir()):
            if not journal_dir.is_dir():
                continue
            
            if not journal_dir.name.endswith('.page_pics'):
                continue
            
            pdf_name = journal_dir.name.replace('.page_pics', '.pdf')
            pdf_path = year_dir / pdf_name
            
            try:
                dir_to_pdf(journal_dir, pdf_path)
                print(f"  ✓ Создан: {pdf_name}")
            except ValueError as e:
                print(f"  ✗ Пропущен {journal_dir.name}: {e}")
            except Exception as e:
                print(f"  ✗ Ошибка при обработке {journal_dir.name}: {e}")
    
    print("Готово!")
