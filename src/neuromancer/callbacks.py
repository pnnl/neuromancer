"""
Callback classes for versatile behavior in the Trainer object at specified checkpoints.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import clear_output


class Callback:
    """
    Callback base class which allows for bare functionality of Trainer
    """

    def __init__(self):
        pass

    def begin_train(self, trainer):
        pass

    def begin_epoch(self, trainer, output):
        pass

    def begin_eval(self, trainer, output):
        pass

    def end_batch(self, trainer, output):
        pass

    def end_eval(self, trainer, output):
        pass

    def end_epoch(self, trainer, output):
        pass

    def end_train(self, trainer, output):
        pass

    def begin_test(self, trainer):
        pass

    def end_test(self, trainer, output):
        pass


class LossHistoryCallback(Callback):
    """
    Plot and save train (and optional dev) loss history at end of each epoch.

    Args:
        plots_dir: Directory to write PNGs into (created if missing).
        show: Whether to show the plot (for Jupyter notebooks).
    Behavior:
        - Runs on Trainer.end_epoch when `current_epoch` is a multiple of `epoch_verbose`.
        - Always plots train history; plots dev history only if present.
        - Saves semilogy plot to `loss_history_epoch_{epoch}.png`.
    """

    def __init__(self, plots_dir: Path | None = None, show: bool = True):
        super().__init__()
        self.plots_dir = Path(plots_dir) if plots_dir is not None else None
        self.show = show

    def _save_fig(self, name: str):
        if self.plots_dir is None:
            return
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.plots_dir / name)
        plt.close()

    def end_epoch(self, trainer, output):
        if trainer.current_epoch % trainer.epoch_verbose != 0:
            return

        train_loss_history = [
            l.detach().cpu().numpy() for l in trainer.loss_history["train"]
        ]

        clear_output(wait=True)
        plt.semilogy(train_loss_history, label="Train loss")

        if "dev" in trainer.loss_history and trainer.loss_history["dev"]:
            dev_loss_history = [
                l.detach().cpu().numpy() for l in trainer.loss_history["dev"]
            ]
            plt.semilogy(dev_loss_history, label="Dev loss")

        plt.xlabel("# Epochs")
        plt.legend()
        if self.show:
            plt.show()
        self._save_fig(f"loss_history_epoch_{trainer.current_epoch}.png")
