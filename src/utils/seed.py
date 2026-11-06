"""
src/utils/seed.py
-----------------
Reproducibility helpers. Call set_global_seed() at the top of every script.

Design decision: we set seeds for Python, NumPy, PyTorch (CPU + CUDA), and
CuDNN determinism. The slight performance cost of determinism is worth it for
reproducible research. We document the seed in every checkpoint filename.
"""

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Fix every RNG relevant to deep learning experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # multi-GPU safety

    # CuDNN: deterministic=True sacrifices ~5–10% speed; acceptable for research.
    # If you need raw speed, set deterministic=False and benchmark=True.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
