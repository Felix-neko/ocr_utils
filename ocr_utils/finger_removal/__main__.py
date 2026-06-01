"""CLI удаления пальцев со сканов: маска (нейросеть/кожа) → инпейнтинг (LaMa/SD).

Примеры:
    # только маски + отладочные оверлеи (проверить детекцию, ничего не инпейнтить)
    uv run python -m ocr_utils.finger_removal IN OUT --mask-only --debug-dir OUT/debug

    # удалить пальцы через LaMa
    uv run python -m ocr_utils.finger_removal IN OUT --inpaint-method lama

    # через Stable Diffusion с собственным промптом
    uv run python -m ocr_utils.finger_removal IN OUT --inpaint-method sd \\
        --sd-prompt "blank blue book cover, no text"
"""

import logging
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .inpainting import DEFAULT_SD_NEGATIVE, DEFAULT_SD_PROMPT, inpaint_image
from .masking import build_finger_mask, build_finger_mask_batch, overlay_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_rgb(path: Path) -> np.ndarray:
    """Загружает изображение как RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось загрузить: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_manual_mask(mask_dir: Path | None, stem: str, shape: tuple[int, int]) -> np.ndarray | None:
    """Ищет ручную маску mask_dir/<stem>.png; возвращает uint8 0/255 или None."""
    if mask_dir is None:
        return None
    for ext in (".png", ".PNG"):
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            if m.shape[:2] != shape:
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            return (m > 127).astype(np.uint8) * 255
    return None


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--mask-method",
    type=click.Choice(["auto", "neural", "skin"]),
    default="auto",
    show_default=True,
    help="Способ построения маски пальца",
)
@click.option(
    "--inpaint-method",
    type=click.Choice(["edge", "vertical", "horizontal", "lama", "sd"]),
    default="edge",
    show_default=True,
    help="Движок инпейнтинга: edge — продолжение кромки книги с авто-выбором оси "
    "(лучше для этих сканов); vertical/horizontal — принудительная ось; lama/sd — нейросети",
)
@click.option("--edge-frac", default=0.12, show_default=True, help="Ширина краевой рамки (доля кадра)")
@click.option("--dilate", "dilate_px", default=14, show_default=True, help="Дилатация маски, пикс.")
@click.option("--padding", default=96, show_default=True, help="Контекст вокруг маски для инпейнтинга, пикс.")
@click.option("--mask-only", is_flag=True, default=False, help="Только построить маски/оверлеи, без инпейнтинга")
@click.option(
    "--mask-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Каталог с ручными масками <stem>.png (приоритетнее автоматики)",
)
@click.option(
    "--debug-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Куда сохранять отладочные оверлеи маски",
)
@click.option("--sd-prompt", default=DEFAULT_SD_PROMPT, show_default=True, help="Промпт для SD-инпейнтинга")
@click.option("--sd-negative", default=DEFAULT_SD_NEGATIVE, show_default=True, help="Негативный промпт для SD")
@click.option("--sd-steps", default=40, show_default=True, help="Число шагов SD")
@click.option(
    "--sd-guidance", default=5.0, show_default=True, help="guidance_scale SD (ниже = меньше выдуманных деталей)"
)
@click.option(
    "--sd-smooth",
    default=0.6,
    show_default=True,
    help="Пост-сглаживание заполнения с сохранением границ, 0..1 (выше = ровнее бумага)",
)
@click.option(
    "--sd-model",
    default="stable-diffusion-v1-5/stable-diffusion-inpainting",
    show_default=True,
    help="HF-id модели SD inpainting",
)
@click.option("--device", default=None, help="cuda / cpu (по умолчанию авто)")
@click.option("--batch-size", default=8, show_default=True, help="Размер батча для обработки")
@click.option("--conf", default=0.03, show_default=True, help="Порог уверенности YOLO (ниже = больше детекций)")
def main(
    input_dir: Path,
    output_dir: Path,
    mask_method: str,
    inpaint_method: str,
    edge_frac: float,
    dilate_px: int,
    padding: int,
    mask_only: bool,
    mask_dir: Path | None,
    debug_dir: Path | None,
    sd_prompt: str,
    sd_negative: str,
    sd_steps: int,
    sd_guidance: float,
    sd_smooth: float,
    sd_model: str,
    device: str | None,
    batch_size: int,
    conf: float,
) -> None:
    """Убирает пальцы со сканов из INPUT_DIR и сохраняет PNG в OUTPUT_DIR."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    extensions = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    files = []
    for ext in extensions:
        files.extend(p for p in input_dir.glob(ext) if p.is_file())
    files = sorted(files)
    if not files:
        logger.warning("Изображения (PNG/JPG/JPEG) не найдены в %s", input_dir)
        return

    logger.info(
        "Файлов: %d | маска: %s | инпейнт: %s | устройство: %s | батч: %d%s",
        len(files),
        mask_method,
        inpaint_method,
        device,
        batch_size,
        " | mask-only" if mask_only else "",
    )

    for batch_start in tqdm(range(0, len(files), batch_size), desc="Батчи", unit="batch"):
        batch_files = files[batch_start : batch_start + batch_size]
        
        try:
            batch_data = []
            for path in batch_files:
                try:
                    rgb = load_rgb(path)
                    manual = load_manual_mask(mask_dir, path.stem, rgb.shape[:2])
                    orig_ext = path.suffix.lower()
                    batch_data.append((path, rgb, manual, orig_ext))
                except Exception as e:
                    tqdm.write(f"  Ошибка загрузки {path.name}: {e}")
            
            if not batch_data:
                continue
            
            paths, rgbs, manuals, orig_exts = zip(*batch_data)
            
            auto_indices = [i for i, m in enumerate(manuals) if m is None]
            auto_rgbs = [rgbs[i] for i in auto_indices]
            
            if auto_rgbs:
                auto_results = build_finger_mask_batch(
                    auto_rgbs, method=mask_method, edge_frac=edge_frac, dilate_px=dilate_px, device=device, conf=conf
                )
            else:
                auto_results = []
            
            masks_and_infos = []
            auto_idx = 0
            for manual in manuals:
                if manual is not None:
                    masks_and_infos.append((manual, "manual"))
                else:
                    masks_and_infos.append(auto_results[auto_idx])
                    auto_idx += 1
            
            for path, rgb, (mask, info), orig_ext in zip(paths, rgbs, masks_and_infos, orig_exts):
                try:
                    mask_px = int(np.count_nonzero(mask))
                    
                    if debug_dir is not None:
                        Image.fromarray(overlay_mask(rgb, mask)).save(debug_dir / f"{path.stem}_overlay.png")
                        Image.fromarray(mask).save(debug_dir / f"{path.stem}_mask.png")
                    
                    if mask_only:
                        tqdm.write(f"  {path.name} | маска={info} | пикселей={mask_px}")
                        continue
                    
                    if mask_px == 0:
                        result = rgb
                    else:
                        result = inpaint_image(
                            rgb,
                            mask,
                            method=inpaint_method,
                            device=device,
                            padding=padding,
                            sd_prompt=sd_prompt,
                            sd_negative=sd_negative,
                            sd_steps=sd_steps,
                            sd_guidance=sd_guidance,
                            sd_smooth=sd_smooth,
                            sd_model_id=sd_model,
                        )
                    
                    if orig_ext in (".jpg", ".jpeg"):
                        out_file = output_dir / f"{path.stem}{orig_ext}"
                        Image.fromarray(result).save(out_file, quality=95)
                    else:
                        out_file = output_dir / f"{path.stem}.png"
                        Image.fromarray(result).save(out_file)
                    
                    tqdm.write(f"  {path.name} → {out_file.name} | маска={info} | пикселей={mask_px}")
                
                except Exception as e:
                    tqdm.write(f"  Ошибка обработки {path.name}: {e}")
                    import traceback
                    tqdm.write(traceback.format_exc())
        
        except Exception as e:
            tqdm.write(f"  Ошибка батча: {e}")
            import traceback
            tqdm.write(traceback.format_exc())

    logger.info("Готово. Результаты в %s", output_dir)


if __name__ == "__main__":
    main()
