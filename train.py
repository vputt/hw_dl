from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.asvspoof import ASVspoofDataset
from src.logger.wandb_writer import WandbWriter
from src.model.lcnn import LCNN
from src.trainer.trainer import Trainer
from src.transforms.lfcc import LFCCFrontend
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import seed_worker, set_seed


def _absolute(path: str) -> Path:
    """Переводит путь из Hydra-конфига в обычный абсолютный путь"""
    return Path(to_absolute_path(path))


def _build_frontend(config: DictConfig, random_crop: bool) -> LFCCFrontend:
    """Создаёт LFCC с одинаковыми параметрами для всех частей датасета"""
    return LFCCFrontend(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        num_filters=config.num_filters,
        num_ceps=config.num_ceps,
        delta_window=config.delta_window,
        time_frames=config.time_frames,
        log_eps=config.log_eps,
        random_crop=random_crop,
    )


def _build_dataset(
    protocol_path: str,
    audio_dir: str,
    data_config: DictConfig,
    frontend: LFCCFrontend,
) -> ASVspoofDataset:
    return ASVspoofDataset(
        protocol_path=_absolute(protocol_path),
        audio_dir=_absolute(audio_dir),
        transform=frontend,
        extension=data_config.extension,
        expected_sample_rate=data_config.sample_rate,
    )


def _loader_options(config: DictConfig, device: torch.device) -> dict[str, Any]:
    """Общие настройки DataLoader для train, dev и eval"""
    num_workers = int(config.num_workers)
    return {
        "num_workers": num_workers,
        "pin_memory": bool(config.pin_memory and device.type == "cuda"),
        "persistent_workers": bool(config.persistent_workers and num_workers > 0),
        "worker_init_fn": seed_worker,
    }


def _take_balanced_batch(loader: DataLoader) -> dict[str, Any]:
    """Берёт один batch, в котором есть оба класса"""
    for batch in loader:
        if torch.unique(batch["label"]).numel() == 2:
            return batch
    raise RuntimeError("Для one-batch test нужен batch с двумя классами")


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config: DictConfig) -> None:
    resolved_config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.seed, deterministic=config.debug.deterministic)
    device = torch.device(
        config.device
        if config.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # На train используется случайный crop, а на dev и eval центральный
    train_dataset = _build_dataset(
        config.data.train_protocol,
        config.data.train_audio,
        config.data,
        _build_frontend(config.frontend, random_crop=True),
    )
    dev_dataset = _build_dataset(
        config.data.dev_protocol,
        config.data.dev_audio,
        config.data,
        _build_frontend(config.frontend, random_crop=False),
    )
    loader_options = _loader_options(config.data, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        **loader_options,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        **loader_options,
    )

    eval_loader = None
    if config.trainer.monitor_eval and not config.debug.overfit_one_batch:
        eval_dataset = _build_dataset(
            config.data.eval_protocol,
            config.data.eval_audio,
            config.data,
            _build_frontend(config.frontend, random_crop=False),
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.trainer.eval_batch_size,
            shuffle=False,
            **loader_options,
        )

    # До полного обучения проверяем, что модель может запомнить один batch
    if config.debug.overfit_one_batch:
        one_batch = _take_balanced_batch(train_loader)
        train_batches = [one_batch]
        dev_batches = [one_batch]
        print("One-batch test: один и тот же batch используется для train и dev")
    else:
        train_batches = train_loader
        dev_batches = dev_loader

    # Модель получает LFCC [B, 1, 60, 750] и возвращает два logit
    model = LCNN(
        feature_bins=config.model.feature_bins,
        time_frames=config.model.time_frames,
        num_classes=config.model.num_classes,
        dropout_p=config.model.dropout_p,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer.lr,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler.step_size,
        gamma=config.scheduler.gamma,
    )

    # Если указан checkpoint, восстанавливаются веса и состояние обучения
    resume_payload = None
    if config.trainer.resume_from:
        resume_path = _absolute(config.trainer.resume_from)
        resume_payload = load_checkpoint(resume_path)
        print("Продолжаем обучение из", resume_path)

    run_id = resume_payload.get("wandb_run_id") if resume_payload is not None else None
    writer = WandbWriter(
        enabled=config.wandb.enabled,
        project=config.wandb.project,
        entity=config.wandb.entity,
        run_name=config.wandb.run_name,
        mode=config.wandb.mode,
        config=resolved_config,
        run_id=run_id,
        resume=run_id is not None,
    )

    output_dir = _absolute(config.trainer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)

    # Trainer содержит сам цикл эпох, оценку EER и сохранение checkpoint.
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        writer=writer,
        output_dir=output_dir,
        epochs=config.trainer.epochs,
        amp=config.trainer.amp,
        eval_amp=config.trainer.eval_amp,
        log_interval=config.trainer.log_interval,
        save_every_epochs=config.trainer.save_every_epochs,
        config_snapshot=resolved_config,
        grad_clip_norm=config.trainer.grad_clip_norm,
        artifact_name=config.wandb.artifact_name,
        log_model_artifacts=config.wandb.log_model_artifacts,
    )
    if resume_payload is not None:
        trainer.restore(resume_payload)

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    writer.set_summary("model/trainable_parameters", trainable_parameters)
    print(
        f"Устройство: {device} | train: {len(train_dataset)} | "
        f"dev: {len(dev_dataset)} | параметров: {trainable_parameters:,}"
    )
    if eval_loader is not None:
        print("Eval используется только для W&B-графика, best.pt выбирается по dev")

    try:
        trainer.fit(train_batches, dev_batches, eval_loader)
    finally:
        writer.finish()


if __name__ == "__main__":
    main()
