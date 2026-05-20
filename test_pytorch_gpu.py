#!/usr/bin/env python3
"""Проверка работоспособности PyTorch с GPU"""

import torch

print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"CUDA версия: {torch.version.cuda}")
print(f"Количество GPU: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Название GPU: {torch.cuda.get_device_name(0)}")
    print(f"cuDNN версия: {torch.backends.cudnn.version()}")
    print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Тестовое вычисление на GPU
    device = torch.device('cuda')
    print(f"\nТестирование вычислений на GPU...")
    
    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)
    
    import time
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"✓ Умножение матриц 5000x5000 на GPU: {elapsed:.3f} сек")
    print(f"✓ Результат shape: {z.shape}")
    print(f"✓ GPU работает корректно!")
else:
    print("✗ CUDA недоступна!")
