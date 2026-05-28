#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rawpy",
#     "numpy",
#     "Pillow",
#     "opencv-python-headless",
#     "click",
#     "tqdm",
#     "torch",
#     "simdeblur @ git+https://github.com/ljzycmd/SimDeblur.git",
#     "deepinv",
# ]
# ///
"""Восстанавливает чёткость газетных сканов из RAF-файлов."""

import io
from enum import Enum
from pathlib import Path

import click
import cv2
import numpy as np
import rawpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm


DEFAULT_INPUT = Path(__file__).parent / "test_input"
DEFAULT_OUTPUT = Path(__file__).parent / "test_output"
NAFNET_CACHE = Path.home() / ".cache" / "nafnet"

# Веса NAFNet на HuggingFace (зеркало megvii-research/NAFNet)
_NAFNET_HF = {
    "gopro-w32": "https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth",
    "gopro-w64": "https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width64.pth",
}

# Параметры архитектуры берём из официальных yml-конфигов репозитория NAFNet
_NAFNET_PARAMS = {
    "gopro-w32": dict(img_channel=3, width=32, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]),
    "gopro-w64": dict(img_channel=3, width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]),
}


class Method(str, Enum):
    NAFNET = "nafnet"
    RESTORMER = "restormer"


# ═══════════════════════════════════════════════════════════════════════════════
# Загрузка моделей
# ═══════════════════════════════════════════════════════════════════════════════

def _load_nafnet(variant: str, device: torch.device) -> nn.Module:
    import sys, types, importlib.util

    # Загружаем только нужные файлы напрямую, минуя сломанный __init__ simdeblur.
    if "simdeblur.model.backbone.nafnet.nafnet" not in sys.modules:
        _sd_spec = importlib.util.find_spec("simdeblur")
        _sd_dir = Path(_sd_spec.submodule_search_locations[0])
        _naf_dir = _sd_dir / "model" / "backbone" / "nafnet"

        # Заглушка для BACKBONE_REGISTRY (нужен только как декоратор)
        _build = types.ModuleType("simdeblur.model.build")
        _build.BACKBONE_REGISTRY = type("_Reg", (), {"register": lambda self: (lambda cls: cls)})()
        sys.modules["simdeblur.model.build"] = _build

        for _name, _file in [
            ("simdeblur.model.backbone.nafnet.arch_util", "arch_util.py"),
            ("simdeblur.model.backbone.nafnet.nafnet", "nafnet.py"),
        ]:
            _spec = importlib.util.spec_from_file_location(_name, _naf_dir / _file)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)

    NAFNet = sys.modules["simdeblur.model.backbone.nafnet.nafnet"].NAFNet

    url = _NAFNET_HF[variant]
    fname = url.split("/")[-1]
    weights_path = NAFNET_CACHE / fname

    if not weights_path.exists():
        NAFNET_CACHE.mkdir(parents=True, exist_ok=True)
        click.echo(f"Скачиваем {fname} с HuggingFace...")
        torch.hub.download_url_to_file(url, str(weights_path), progress=True)

    click.echo(f"Загружаем {fname}...")
    model = NAFNet(**_NAFNET_PARAMS[variant]).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    state = ckpt.get("params", ckpt.get("state_dict", ckpt))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def _load_restormer(device: torch.device) -> nn.Module:
    import deepinv

    click.echo("Загружаем Restormer (defocus_deblurring)...")
    return deepinv.models.Restormer(pretrained="defocus_deblurring", LayerNorm_type="WithBias", device=device)


# ═══════════════════════════════════════════════════════════════════════════════
# Обработка изображений
# ═══════════════════════════════════════════════════════════════════════════════

def read_raf_image(path: Path) -> np.ndarray:
    """Читает RAF и возвращает RGB uint8. Сначала JPEG-превью, потом rawpy."""
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


def compute_sharpness_map(rgb: np.ndarray, tile_size: int = 256) -> np.ndarray:
    """Тайловая карта резкости (дисперсия лапласиана). Возвращает float32 (rows, cols)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    rows = max(1, (h + tile_size - 1) // tile_size)
    cols = max(1, (w + tile_size - 1) // tile_size)
    sharpness = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            tile = gray[r * tile_size : (r + 1) * tile_size, c * tile_size : (c + 1) * tile_size]
            if tile.size > 0:
                sharpness[r, c] = float(cv2.Laplacian(tile, cv2.CV_64F).var())
    return sharpness


def inference_tiled(
    model: nn.Module,
    img: np.ndarray,
    tile_size: int,
    overlap: int,
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """Тайловый инференс с Hann-blend для сшивки без артефактов на границах."""
    h, w = img.shape[:2]
    stride = tile_size - overlap

    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    pad_h = max(0, tile_size - h) if h <= tile_size else (stride - (h - tile_size) % stride) % stride
    pad_w = max(0, tile_size - w) if w <= tile_size else (stride - (w - tile_size) % stride) % stride
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, H, W = tensor.shape
    output = torch.zeros(1, 3, H, W)
    weight = torch.zeros(1, 1, H, W)

    win1d = torch.hann_window(tile_size, periodic=False)
    win2d = (win1d.unsqueeze(0) * win1d.unsqueeze(1)).view(1, 1, tile_size, tile_size)

    ys = list(range(0, H - tile_size + 1, stride)) or [0]
    xs = list(range(0, W - tile_size + 1, stride)) or [0]
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)

    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]
        batch = torch.cat([tensor[:, :, y : y + tile_size, x : x + tile_size] for y, x in batch_coords]).to(device)
        with torch.no_grad():
            outs = model(batch).clamp(0, 1).cpu()
        for j, (y, x) in enumerate(batch_coords):
            output[:, :, y : y + tile_size, x : x + tile_size] += outs[j : j + 1] * win2d
            weight[:, :, y : y + tile_size, x : x + tile_size] += win2d

    output /= weight.clamp(min=1e-6)

    # Мягкий возврат к оригиналу на границах кадра — модель не обучена на фоне/переплёте.
    # Ramp: 0 (оригинал) на краю → 1 (восстановленное) через `overlap` пикселей.
    if overlap > 0 and h > overlap * 2 and w > overlap * 2:
        ramp = torch.linspace(0.0, 1.0, overlap)
        edge_mask = torch.ones(1, 1, H, W)
        edge_mask[:, :, :overlap, :] *= ramp.view(-1, 1)
        edge_mask[:, :, h - overlap : h, :] *= ramp.flip(0).view(-1, 1)
        edge_mask[:, :, :, :overlap] *= ramp.view(1, -1)
        edge_mask[:, :, :, w - overlap : w] *= ramp.flip(0).view(1, -1)
        output = edge_mask * output + (1.0 - edge_mask) * tensor

    result = output[0].permute(1, 2, 0).numpy()[:h, :w]
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def adaptive_blend(
    original: np.ndarray,
    restored: np.ndarray,
    sharpness_map: np.ndarray,
    threshold: float,
    scale: float = 100.0,
) -> np.ndarray:
    """Blend: alpha→1 (restored) там где расфокус, alpha→0 (original) где резко."""
    h, w = original.shape[:2]
    alpha_map = 1.0 / (1.0 + np.exp((sharpness_map - threshold) / scale))
    alpha_full = cv2.resize(alpha_map, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    result = alpha_full * restored.astype(np.float32) + (1.0 - alpha_full) * original.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

@click.command()
@click.argument("input_dir", default=str(DEFAULT_INPUT))
@click.argument("output_dir", default=str(DEFAULT_OUTPUT))
@click.option(
    "--method",
    type=click.Choice([m.value for m in Method]),
    default=Method.NAFNET.value,
    show_default=True,
    help="Метод восстановления чёткости",
)
@click.option(
    "--nafnet-variant",
    type=click.Choice(list(_NAFNET_HF)),
    default="gopro-w32",
    show_default=True,
    help="[nafnet] gopro-w32 (быстрее) / gopro-w64 (качественнее)",
)
@click.option("--tile-size", default=512, show_default=True, help="Размер тайла для инференса (пикс.)")
@click.option("--overlap", default=64, show_default=True, help="Перекрытие тайлов (пикс.)")
@click.option("--batch-size", default=4, show_default=True, help="Кол-во тайлов в батче")
@click.option(
    "--blend-threshold",
    default=300.0,
    show_default=True,
    help="Порог Laplacian variance для adaptive blend",
)
@click.option("--no-blend", is_flag=True, default=False, help="Деблюр без adaptive blend")
def main(
    input_dir: str,
    output_dir: str,
    method: str,
    nafnet_variant: str,
    tile_size: int,
    overlap: int,
    batch_size: int,
    blend_threshold: float,
    no_blend: bool,
) -> None:
    """Восстанавливает чёткость RAF-сканов и сохраняет как PNG.

    Читает RAF-файлы из INPUT_DIR, применяет нейросетевой деблюр,
    сохраняет PNG в OUTPUT_DIR (создаётся автоматически).
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raf_files = sorted(in_path.glob("*.[Rr][Aa][Ff]"))
    if not raf_files:
        click.echo(f"RAF-файлы не найдены в {in_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"Найдено {len(raf_files)} RAF-файлов | метод: {method} | устройство: {device}")

    m = Method(method)
    if m is Method.NAFNET:
        model = _load_nafnet(nafnet_variant, device)
    else:
        model = _load_restormer(device)
    model.eval()

    for raf_path in tqdm(raf_files, desc="Обрабатываем"):
        try:
            rgb = read_raf_image(raf_path)

            sharpness_map = None if no_blend else compute_sharpness_map(rgb, tile_size=256)

            restored = inference_tiled(model, rgb, tile_size=tile_size, overlap=overlap, device=device, batch_size=batch_size)

            result = restored if (no_blend or sharpness_map is None) else adaptive_blend(
                rgb, restored, sharpness_map, threshold=blend_threshold
            )

            out_file = out_path / (raf_path.stem + ".png")
            Image.fromarray(result).save(out_file, optimize=False)
            tqdm.write(f"  {raf_path.name} → {out_file.name}")

        except Exception as e:
            tqdm.write(f"  Ошибка {raf_path.name}: {e}")
            import traceback
            tqdm.write(traceback.format_exc())

    click.echo(f"\nГотово. Результаты в {out_path}")


if __name__ == "__main__":
    main()
