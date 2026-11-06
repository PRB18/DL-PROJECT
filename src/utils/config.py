"""
src/utils/config.py
-------------------
Loads and validates the YAML config with OmegaConf.

Why OmegaConf?  It gives us dot-access (cfg.training.lr), CLI overrides
(python train.py training.lr=1e-4), and struct-mode validation — all without
writing a dataclass schema by hand.
"""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(path: str | Path = "configs/config.yaml") -> DictConfig:
    """Load YAML config and return an OmegaConf DictConfig."""
    cfg = OmegaConf.load(str(path))
    # Merge with CLI overrides if called from a script
    # e.g.  python train.py training.stage2.lr=5e-5
    return cfg


def save_config(cfg: DictConfig, output_path: str | Path) -> None:
    """Persist the resolved config next to a checkpoint for reproducibility."""
    OmegaConf.save(cfg, str(output_path))
