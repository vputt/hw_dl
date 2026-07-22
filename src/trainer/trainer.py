from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.logger.wandb_writer import WandbWriter
from src.metrics.eer import compute_eer_percent
from src.model.lcnn import LCNN
from src.utils.checkpoint import build_checkpoint, save_checkpoint


def is_better_dev_result(
    dev_eer: float,
    dev_loss: float,
    best_dev_eer: float,
    best_dev_loss: float,
) -> bool:
    """Сначала сравнивает dev EER, а при равенстве dev loss"""
    return (dev_eer, dev_loss) < (best_dev_eer, best_dev_loss)


class Trainer:
    """Обучение, проверка на dev/eval и сохранение checkpoint"""

    def __init__(
        self,
        model: LCNN,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        device: torch.device,
        writer: WandbWriter,
        output_dir: str | Path,
        epochs: int,
        amp: bool,
        eval_amp: bool,
        log_interval: int,
        save_every_epochs: int,
        config_snapshot: dict[str, Any],
        grad_clip_norm: float | None = None,
        artifact_name: str = "lcnn-checkpoint",
        log_model_artifacts: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = writer
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_every_epochs = save_every_epochs
        self.config_snapshot = config_snapshot
        self.grad_clip_norm = grad_clip_norm
        self.artifact_name = artifact_name
        self.log_model_artifacts = log_model_artifacts

        self.amp_enabled = amp and device.type == "cuda"
        self.eval_amp_enabled = eval_amp and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.start_epoch = 0
        self.global_step = 0
        self.best_dev_eer = float("inf")
        self.best_dev_loss = float("inf")
        self.best_epoch = -1
        self.history: dict[str, list[float | int]] = {
            "epoch": [],
            "train_loss": [],
            "dev_loss": [],
            "dev_eer_percent": [],
            "dev_accuracy_percent": [],
            "eval_loss": [],
            "eval_eer_percent": [],
            "eval_accuracy_percent": [],
            "learning_rate": [],
        }

    def restore(self, checkpoint: dict[str, Any]) -> None:
        """Продолжает обучение с той же эпохи и теми же состояниями"""
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_dev_eer = float(checkpoint.get("best_dev_eer", float("inf")))
        self.best_dev_loss = float(checkpoint.get("best_dev_loss", float("inf")))
        self.best_epoch = int(checkpoint.get("best_epoch", -1))
        self.history = checkpoint.get("history", self.history)

    def _move_batch(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        features = batch["features"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)
        return features, labels

    def train_one_epoch(self, loader: Iterable[dict[str, Any]], epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_examples = 0

        progress = tqdm(loader, desc=f"Train {epoch + 1}/{self.epochs}", leave=False)
        for batch in progress:
            features, labels = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            # AMP ускоряет train на GPU, но для dev/eval по умолчанию выключен.
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(features)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            self.global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if self.global_step % self.log_interval == 0:
                self.writer.log_step(loss.item(), self.global_step)

        return total_loss / total_examples

    @torch.inference_mode()
    def evaluate(
        self,
        loader: Iterable[dict[str, Any]],
        description: str,
    ) -> dict[str, float]:
        """Считает loss, accuracy и EER на полном разделе"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        scores: list[float] = []
        labels_list: list[int] = []

        for batch in tqdm(loader, desc=description, leave=False):
            features, labels = self._move_batch(batch)
            with torch.autocast(device_type=self.device.type, enabled=self.eval_amp_enabled):
                logits = self.model(features)
                loss = self.criterion(logits, labels)

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size
            scores.extend(self.model.score_from_logits(logits).float().cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

        eer, threshold = compute_eer_percent(scores, labels_list)
        return {
            "loss": total_loss / total_examples,
            "accuracy_percent": 100.0 * total_correct / total_examples,
            "eer_percent": eer,
            "threshold": threshold,
        }

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return build_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            global_step=self.global_step,
            best_dev_eer=self.best_dev_eer,
            best_dev_loss=self.best_dev_loss,
            best_epoch=self.best_epoch,
            history=self.history,
            config=self.config_snapshot,
            wandb_run_id=self.writer.run_id,
        )

    def _append_history(
        self,
        epoch: int,
        train_loss: float,
        dev: dict[str, float],
        eval_metrics: dict[str, float] | None,
        learning_rate: float,
    ) -> None:
        self.history["epoch"].append(epoch + 1)
        self.history["train_loss"].append(train_loss)
        self.history["dev_loss"].append(dev["loss"])
        self.history["dev_eer_percent"].append(dev["eer_percent"])
        self.history["dev_accuracy_percent"].append(dev["accuracy_percent"])
        self.history["learning_rate"].append(learning_rate)
        if eval_metrics is not None:
            self.history["eval_loss"].append(eval_metrics["loss"])
            self.history["eval_eer_percent"].append(eval_metrics["eer_percent"])
            self.history["eval_accuracy_percent"].append(eval_metrics["accuracy_percent"])

    def fit(
        self,
        train_loader: Iterable[dict[str, Any]],
        dev_loader: Iterable[dict[str, Any]],
        eval_loader: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.start_epoch, self.epochs):
            learning_rate = self.optimizer.param_groups[0]["lr"]
            train_loss = self.train_one_epoch(train_loader, epoch)
            dev = self.evaluate(dev_loader, f"Dev {epoch + 1}/{self.epochs}")
            eval_metrics = None
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader, f"Eval {epoch + 1}/{self.epochs}")

            improved = is_better_dev_result(
                dev["eer_percent"],
                dev["loss"],
                self.best_dev_eer,
                self.best_dev_loss,
            )
            if improved:
                self.best_dev_eer = dev["eer_percent"]
                self.best_dev_loss = dev["loss"]
                self.best_epoch = epoch

            # Eval логируется для графика, но не участвует в выборе best.pt
            self._append_history(epoch, train_loss, dev, eval_metrics, learning_rate)
            metrics = {
                "train/loss": train_loss,
                "dev/loss": dev["loss"],
                "dev/eer_percent": dev["eer_percent"],
                "dev/accuracy_percent": dev["accuracy_percent"],
                "dev/eer_threshold": dev["threshold"],
                "learning_rate": learning_rate,
            }
            if eval_metrics is not None:
                metrics.update(
                    {
                        "eval/loss": eval_metrics["loss"],
                        "eval/eer_percent": eval_metrics["eer_percent"],
                        "eval/accuracy_percent": eval_metrics["accuracy_percent"],
                        "eval/eer_threshold": eval_metrics["threshold"],
                    }
                )
            self.writer.log_epoch(metrics, epoch + 1, self.global_step)

            if self.scheduler is not None:
                self.scheduler.step()

            payload = self._checkpoint_payload(epoch)
            last_path = save_checkpoint(payload, self.output_dir / "last.pt")
            best_path = self.output_dir / "best.pt"
            if improved:
                save_checkpoint(payload, best_path)
                self.writer.set_summary("best/dev_eer_percent", self.best_dev_eer)
                self.writer.set_summary("best/dev_loss", self.best_dev_loss)
                self.writer.set_summary("best/epoch", epoch + 1)
                if eval_metrics is not None:
                    self.writer.set_summary(
                        "eval_at_best_dev/eer_percent",
                        eval_metrics["eer_percent"],
                    )

            periodic = self.save_every_epochs > 0 and (
                (epoch + 1) % self.save_every_epochs == 0 or epoch + 1 == self.epochs
            )
            if periodic:
                epoch_path = save_checkpoint(
                    payload,
                    self.output_dir / f"epoch_{epoch + 1:03d}.pt",
                )
                if self.log_model_artifacts:
                    self.writer.log_checkpoints(
                        [last_path, best_path, epoch_path],
                        artifact_name=self.artifact_name,
                        aliases=["latest", f"epoch-{epoch + 1:03d}"],
                        metadata={"epoch": epoch + 1, "best_epoch": self.best_epoch + 1},
                    )

            message = (
                f"Epoch {epoch + 1:03d}/{self.epochs} | "
                f"train CE={train_loss:.5f} | dev CE={dev['loss']:.5f} | "
                f"dev EER={dev['eer_percent']:.3f}%"
            )
            if eval_metrics is not None:
                message += f" | eval EER={eval_metrics['eer_percent']:.3f}%"
            message += (
                f" | best dev={self.best_dev_eer:.3f}% "
                f"(epoch {self.best_epoch + 1})"
            )
            print(message)
