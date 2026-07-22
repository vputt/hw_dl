from pathlib import Path
from typing import Any


class WandbWriter:
    """Небольшая обёртка над W&B для логов и checkpoint-артефактов"""

    def __init__(
        self,
        enabled: bool,
        project: str,
        entity: str | None,
        run_name: str,
        mode: str,
        config: dict[str, Any],
        run_id: str | None = None,
        resume: bool = False,
    ) -> None:
        self._wandb = None
        self._run = None
        if not enabled:
            return

        import wandb

        if run_id is None:
            run_id = wandb.util.generate_id()
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            id=run_id,
            resume="must" if resume else "never",
            mode=mode,
            config=config,
        )

        wandb.define_metric("train/global_step")
        wandb.define_metric("train/step_loss", step_metric="train/global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/loss", step_metric="epoch")
        wandb.define_metric("dev/*", step_metric="epoch")
        wandb.define_metric("eval/*", step_metric="epoch")
        wandb.define_metric("learning_rate", step_metric="epoch")

    @property
    def run_id(self) -> str | None:
        return self._run.id if self._run is not None else None

    def log_step(self, loss: float, global_step: int) -> None:
        if self._wandb is not None:
            self._wandb.log(
                {"train/step_loss": loss, "train/global_step": global_step}
            )

    def log_epoch(self, metrics: dict[str, float], epoch: int, global_step: int) -> None:
        """Логирует все метрики эпохи с единым шагом epoch."""
        if self._wandb is not None:
            self._wandb.log(
                {**metrics, "epoch": epoch, "train/global_step": global_step}
            )

    def set_summary(self, key: str, value: Any) -> None:
        if self._run is not None:
            self._run.summary[key] = value

    def log_checkpoints(
        self,
        paths: list[str | Path],
        artifact_name: str,
        aliases: list[str],
        metadata: dict[str, Any],
    ) -> None:
        if self._wandb is None or self._run is None:
            return

        artifact = self._wandb.Artifact(
            artifact_name,
            type="model",
            metadata=metadata,
        )
        for path_value in paths:
            path = Path(path_value)
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        self._run.log_artifact(artifact, aliases=aliases)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
