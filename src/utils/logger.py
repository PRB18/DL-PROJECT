"""
src/utils/logger.py
-------------------
Unified logger: writes to console (Rich) + TensorBoard + a plain text log file.

Design decision: one logger object passed around avoids scattered print()
statements and makes it easy to swap in W&B later.
"""

import logging
from pathlib import Path
from datetime import datetime

from rich.logging import RichHandler
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Thin wrapper around Python logging + TensorBoard SummaryWriter."""

    def __init__(self, name: str, log_dir: str | Path, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ---- Python logger (console + file) ----
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        if not self._logger.handlers:
            # Rich console handler (colours, tracebacks)
            self._logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))

            # Plain-text file handler
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(self.log_dir / f"{name}_{ts}.log")
            fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
            self._logger.addHandler(fh)

        # ---- TensorBoard ----
        self.tb: SummaryWriter | None = None
        if use_tensorboard:
            self.tb = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

    # Delegate standard logging calls
    def info(self, msg: str) -> None:    self._logger.info(msg)
    def warning(self, msg: str) -> None: self._logger.warning(msg)
    def error(self, msg: str) -> None:   self._logger.error(msg)

    def scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar to TensorBoard."""
        if self.tb:
            self.tb.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.tb:
            self.tb.close()
