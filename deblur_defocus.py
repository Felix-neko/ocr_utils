#!/usr/bin/env python3
"""Defocus deblur для газетных сканов с использованием современных моделей."""

import io
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import rawpy
import torch
from PIL import Image, ImageOps
from tqdm import tqdm


DEFAULT_INPUT = Path(__file__).parent / "test_input"
DEFAULT_OUTPUT = Path(__file__).parent / "test_output"


def read_raf_image(path: Path) -> np.ndarray:
    """Читает RAF и возвращает RGB uint8."""
    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(bytes(thumb.data)))
                img = ImageOps.exif_transpose(img)
                return np.array(img.convert("RGB"))
        except Exception:
            pass
        return raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)


def compute_sharpness_laplacian(rgb: np.ndarray) -> float:
    """Глобальная оценка резкости через дисперсию лапласиана."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_blur_kernel_size(img_patch: np.ndarray) -> int:
    """Оценивает размер ядра размытия для патча изображения через анализ краёв."""
    gray = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY) if len(img_patch.shape) == 3 else img_patch
    
    # Метод 1: Анализ ширины краёв через градиенты
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Находим сильные края
    edge_threshold = np.percentile(gradient_magnitude, 90)
    strong_edges = gradient_magnitude > edge_threshold
    
    if strong_edges.sum() > 100:  # Достаточно краёв для анализа
        # Анализируем ширину краёв
        edge_widths = []
        for _ in range(min(50, strong_edges.sum() // 10)):
            y, x = np.where(strong_edges)
            if len(y) == 0:
                break
            idx = np.random.randint(len(y))
            py, px = y[idx], x[idx]
            
            # Смотрим профиль градиента вдоль нормали к краю
            angle = np.arctan2(sobely[py, px], sobelx[py, px])
            dy, dx = int(np.sin(angle)), int(np.cos(angle))
            
            # Измеряем ширину края
            width = 0
            for step in range(1, 20):
                ny, nx = py + dy * step, px + dx * step
                if 0 <= ny < gray.shape[0] and 0 <= nx < gray.shape[1]:
                    if gradient_magnitude[ny, nx] > edge_threshold * 0.3:
                        width += 1
                    else:
                        break
                else:
                    break
            
            if width > 0:
                edge_widths.append(width)
        
        if edge_widths:
            avg_width = np.median(edge_widths)
            # Размытые края шире
            kernel_size = max(5, min(31, int(avg_width * 3)))
            return kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Метод 2: Fallback через дисперсию лапласиана
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Чем меньше дисперсия, тем больше размытие
    if laplacian_var < 100:
        kernel_size = 21
    elif laplacian_var < 300:
        kernel_size = 15
    elif laplacian_var < 600:
        kernel_size = 11
    elif laplacian_var < 1000:
        kernel_size = 9
    else:
        kernel_size = 7
    
    return kernel_size


def estimate_blur_amount(img_patch: np.ndarray) -> float:
    """Оценивает степень размытия патча (0 = резкий, 1 = сильно размыт)."""
    gray = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY) if len(img_patch.shape) == 3 else img_patch
    
    # Используем дисперсию лапласиана
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Нормализуем (эмпирические пороги для газетного текста)
    # Очень резкий текст: > 2000, резкий: > 1000, средний: 500, размытый: < 200
    if laplacian_var > 2000:
        blur_score = 0.0  # Очень резкий - не трогаем
    elif laplacian_var > 1000:
        blur_score = 0.2  # Резкий - минимальная обработка
    elif laplacian_var > 500:
        blur_score = 0.5  # Средний - умеренная обработка
    elif laplacian_var > 200:
        blur_score = 0.7  # Размытый - активная обработка
    else:
        blur_score = 1.0  # Сильно размытый - максимальная обработка
    
    return blur_score


def unsharp_mask_deblur(img: np.ndarray, sigma: float = 1.0, amount: float = 1.5, threshold: int = 0) -> np.ndarray:
    """Классический unsharp mask для повышения резкости."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        sharpened = np.where(low_contrast_mask, img, sharpened)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def wiener_deconvolution(img: np.ndarray, kernel_size: int = 15, noise_var: float = 0.01) -> np.ndarray:
    """Винеровская деконволюция для defocus deblur."""
    from scipy import signal
    from scipy.ndimage import gaussian_filter
    
    result = np.zeros_like(img)
    
    psf = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    cv2.circle(psf, (center, center), kernel_size // 4, 1, -1)
    psf = psf / psf.sum()
    
    for i in range(3):
        channel = img[:, :, i].astype(np.float64) / 255.0
        
        psf_fft = np.fft.fft2(psf, s=channel.shape)
        img_fft = np.fft.fft2(channel)
        
        psf_conj = np.conj(psf_fft)
        psf_abs_sq = np.abs(psf_fft) ** 2
        
        wiener_filter = psf_conj / (psf_abs_sq + noise_var)
        restored_fft = wiener_filter * img_fft
        restored = np.fft.ifft2(restored_fft).real
        
        result[:, :, i] = np.clip(restored * 255, 0, 255)
    
    return result.astype(np.uint8)


def richardson_lucy_deblur(img: np.ndarray, iterations: int = 30, kernel_size: int = 15) -> np.ndarray:
    """Richardson-Lucy деконволюция для defocus blur."""
    result = np.zeros_like(img, dtype=np.float64)
    
    psf = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    cv2.circle(psf, (center, center), kernel_size // 4, 1, -1)
    psf = psf / psf.sum()
    
    psf_mirror = np.flip(psf)
    
    for i in range(3):
        channel = img[:, :, i].astype(np.float64) / 255.0
        estimate = channel.copy()
        
        for _ in range(iterations):
            conv = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)
            relative_blur = channel / (conv + 1e-10)
            estimate *= cv2.filter2D(relative_blur, -1, psf_mirror, borderType=cv2.BORDER_REFLECT)
        
        result[:, :, i] = np.clip(estimate * 255, 0, 255)
    
    return result.astype(np.uint8)


def blind_deconvolution(img: np.ndarray, psf_size: int = 15, iterations: int = 30) -> np.ndarray:
    """Blind deconvolution - автоматически оценивает PSF и восстанавливает изображение."""
    from skimage import restoration
    
    result = np.zeros_like(img, dtype=np.float64)
    
    # Инициализируем PSF как круглое размытие (defocus)
    psf_init = np.zeros((psf_size, psf_size))
    center = psf_size // 2
    cv2.circle(psf_init, (center, center), psf_size // 4, 1, -1)
    psf_init = psf_init / psf_init.sum()
    
    for i in range(3):
        channel = img[:, :, i].astype(np.float64) / 255.0
        
        # Применяем unsupervised Wiener-Hunt deconvolution
        # Автоматически оценивает PSF и восстанавливает изображение
        deconvolved, psf_estimated = restoration.unsupervised_wiener(
            channel,
            psf_init,
            max_num_iter=iterations,
            clip=False
        )
        
        result[:, :, i] = np.clip(deconvolved * 255, 0, 255)
    
    return result.astype(np.uint8)


def adaptive_deblur(img: np.ndarray, method: str = "rl", **kwargs) -> np.ndarray:
    """Адаптивный deblur с выбором метода."""
    if method == "unsharp":
        return unsharp_mask_deblur(img, **kwargs)
    elif method == "wiener":
        return wiener_deconvolution(img, **kwargs)
    elif method == "rl":
        return richardson_lucy_deblur(img, **kwargs)
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def adaptive_tiled_deblur(
    img: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
    method: str = "rl",
    iterations: int = 30,
    blur_threshold: float = 0.3,
) -> np.ndarray:
    """Тайловый адаптивный deblur с автоматическим определением параметров для каждого тайла."""
    h, w = img.shape[:2]
    result = np.zeros_like(img, dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)
    
    stride = tile_size - overlap
    
    # Создаём окно для плавного смешивания
    window = np.outer(
        np.hanning(tile_size),
        np.hanning(tile_size)
    )
    
    y_positions = list(range(0, h - tile_size + 1, stride))
    x_positions = list(range(0, w - tile_size + 1, stride))
    
    if not y_positions:
        y_positions = [0]
    if not x_positions:
        x_positions = [0]
    
    # Добавляем последние тайлы, если не покрыли всё изображение
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)
    
    total_tiles = len(y_positions) * len(x_positions)
    pbar = tqdm(total=total_tiles, desc="  Обработка тайлов", leave=False)
    
    for y in y_positions:
        for x in x_positions:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            tile = img[y:y_end, x:x_end].copy()
            
            # Оцениваем степень размытия тайла
            blur_amount = estimate_blur_amount(tile)
            
            # Обрабатываем только размытые тайлы
            if blur_amount > blur_threshold:
                # Автоматически определяем размер ядра
                kernel_size = estimate_blur_kernel_size(tile)
                
                # Адаптируем количество итераций к степени размытия
                adaptive_iterations = max(10, int(iterations * blur_amount))
                
                # Применяем deblur
                if method == "rl":
                    deblurred = richardson_lucy_deblur(tile, iterations=adaptive_iterations, kernel_size=kernel_size)
                elif method == "wiener":
                    deblurred = wiener_deconvolution(tile, kernel_size=kernel_size)
                elif method == "blind":
                    deblurred = blind_deconvolution(tile, psf_size=kernel_size, iterations=adaptive_iterations)
                else:
                    deblurred = unsharp_mask_deblur(tile, sigma=1.5, amount=2.0)
                
                # Смешиваем с оригиналом в зависимости от степени размытия
                alpha = blur_amount
                tile_result = (alpha * deblurred.astype(np.float64) + 
                             (1 - alpha) * tile.astype(np.float64))
            else:
                tile_result = tile.astype(np.float64)
            
            # Применяем окно для плавного смешивания
            tile_h, tile_w = tile.shape[:2]
            current_window = window[:tile_h, :tile_w]
            
            for c in range(3):
                result[y:y_end, x:x_end, c] += tile_result[:, :, c] * current_window
            weight[y:y_end, x:x_end] += current_window
            
            pbar.update(1)
    
    pbar.close()
    
    # Нормализуем по весам
    for c in range(3):
        result[:, :, c] /= np.maximum(weight, 1e-10)
    
    return np.clip(result, 0, 255).astype(np.uint8)


@click.command()
@click.argument("input_dir", default=str(DEFAULT_INPUT))
@click.argument("output_dir", default=str(DEFAULT_OUTPUT))
@click.option(
    "--method",
    type=click.Choice(["unsharp", "wiener", "rl", "blind"]),
    default="blind",
    show_default=True,
    help="Метод deblur: unsharp (быстрый), wiener (средний), rl (Richardson-Lucy), blind (автоматический)",
)
@click.option("--iterations", default=30, show_default=True, help="Количество итераций для RL")
@click.option("--kernel-size", default=15, show_default=True, help="Размер ядра размытия (0 = авто)")
@click.option("--sigma", default=1.5, show_default=True, help="Sigma для unsharp mask")
@click.option("--amount", default=2.0, show_default=True, help="Amount для unsharp mask")
@click.option("--tile-size", default=512, show_default=True, help="Размер тайла для адаптивной обработки")
@click.option("--overlap", default=64, show_default=True, help="Перекрытие тайлов")
@click.option("--blur-threshold", default=0.3, show_default=True, help="Порог для определения размытых зон (0-1)")
@click.option("--adaptive/--no-adaptive", default=True, show_default=True, help="Использовать адаптивную тайловую обработку")
def main(
    input_dir: str,
    output_dir: str,
    method: str,
    iterations: int,
    kernel_size: int,
    sigma: float,
    amount: float,
    tile_size: int,
    overlap: int,
    blur_threshold: float,
    adaptive: bool,
) -> None:
    """Восстанавливает чёткость RAF-сканов с defocus blur.

    Использует классические методы деконволюции, оптимизированные для defocus blur.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raf_files = sorted(in_path.glob("*.[Rr][Aa][Ff]"))
    if not raf_files:
        click.echo(f"RAF-файлы не найдены в {in_path}")
        return

    mode_str = "адаптивный тайловый" if adaptive else "глобальный"
    click.echo(f"Найдено {len(raf_files)} RAF-файлов | метод: {method} | режим: {mode_str}")

    sharpness_before = []
    sharpness_after = []

    for raf_path in tqdm(raf_files, desc="Обрабатываем"):
        try:
            rgb = read_raf_image(raf_path)
            
            sharp_before = compute_sharpness_laplacian(rgb)
            sharpness_before.append(sharp_before)

            if adaptive:
                # Адаптивная тайловая обработка с автоопределением параметров
                result = adaptive_tiled_deblur(
                    rgb,
                    tile_size=tile_size,
                    overlap=overlap,
                    method=method,
                    iterations=iterations,
                    blur_threshold=blur_threshold,
                )
            else:
                # Глобальная обработка всего изображения
                if method == "unsharp":
                    result = unsharp_mask_deblur(rgb, sigma=sigma, amount=amount)
                elif method == "wiener":
                    result = wiener_deconvolution(rgb, kernel_size=kernel_size)
                elif method == "blind":
                    result = blind_deconvolution(rgb, psf_size=kernel_size, iterations=iterations)
                else:
                    result = richardson_lucy_deblur(rgb, iterations=iterations, kernel_size=kernel_size)
            
            sharp_after = compute_sharpness_laplacian(result)
            sharpness_after.append(sharp_after)

            out_file = out_path / (raf_path.stem + ".png")
            Image.fromarray(result).save(out_file, optimize=False)
            
            tqdm.write(
                f"  {raf_path.name} → {out_file.name} | резкость: {sharp_before:.1f} → {sharp_after:.1f} "
                f"({sharp_after - sharp_before:+.1f})"
            )

        except Exception as e:
            tqdm.write(f"  Ошибка {raf_path.name}: {e}")
            import traceback
            tqdm.write(traceback.format_exc())

    click.echo(f"\nГотово. Результаты в {out_path}")
    
    if sharpness_before and sharpness_after:
        avg_before = np.mean(sharpness_before)
        avg_after = np.mean(sharpness_after)
        click.echo(f"\n=== Оценка резкости (Laplacian variance) ===")
        click.echo(f"Средняя резкость до:    {avg_before:.1f}")
        click.echo(f"Средняя резкость после: {avg_after:.1f}")
        click.echo(f"Изменение:              {avg_after - avg_before:+.1f} ({(avg_after / avg_before - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
