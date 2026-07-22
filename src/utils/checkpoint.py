from pathlib import Path
from typing import Any

import torch
from torch import nn


def build_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_dev_eer: float,
    best_dev_loss: float,
    best_epoch: int,
    history: dict[str, list[float | int]],
    config: dict[str, Any],
    wandb_run_id: str | None,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_dev_eer": best_dev_eer,
        "best_dev_loss": best_dev_loss,
        "best_epoch": best_epoch,
        "history": history,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "config": config,
        "wandb_run_id": wandb_run_id,
    }


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
