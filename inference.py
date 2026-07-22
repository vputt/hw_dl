import csv
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.datasets.asvspoof import ASVspoofDataset
from src.metrics.eer import compute_eer_percent
from src.model.lcnn import LCNN
from src.trainer.inferencer import Inferencer
from src.transforms.lfcc import LFCCFrontend
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import seed_worker, set_seed
from src.utils.submission import write_submission


def _absolute(path: str) -> Path:
    """Переводит путь из Hydra-конфига в абсолютный"""
    return Path(to_absolute_path(path))


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config: DictConfig) -> None:
    set_seed(config.seed)
    device = torch.device(
        config.device
        if config.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Параметры LFCC и LCNN берём из checkpoint, чтобы не получить другую модель.
    checkpoint = load_checkpoint(_absolute(config.inference.checkpoint))
    frontend_config = checkpoint["config"]["frontend"]
    model_config = checkpoint["config"]["model"]
    frontend = LFCCFrontend(
        sample_rate=frontend_config["sample_rate"],
        n_fft=frontend_config["n_fft"],
        win_length=frontend_config["win_length"],
        hop_length=frontend_config["hop_length"],
        num_filters=frontend_config["num_filters"],
        num_ceps=frontend_config["num_ceps"],
        delta_window=frontend_config["delta_window"],
        time_frames=frontend_config["time_frames"],
        log_eps=frontend_config["log_eps"],
        random_crop=False,
    )
    dataset = ASVspoofDataset(
        protocol_path=_absolute(config.data.protocol),
        audio_dir=_absolute(config.data.audio),
        transform=frontend,
        extension=config.data.extension,
        expected_sample_rate=config.data.sample_rate,
    )
    num_workers = int(config.data.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(config.data.pin_memory and device.type == "cuda"),
        persistent_workers=bool(config.data.persistent_workers and num_workers > 0),
        worker_init_fn=seed_worker,
    )

    model = LCNN(
        feature_bins=model_config["feature_bins"],
        time_frames=model_config["time_frames"],
        num_classes=model_config["num_classes"],
        dropout_p=model_config["dropout_p"],
    )
    model.load_state_dict(checkpoint["state_dict"])

    # Для каждого файла сохраняется непрерывный score:
    # bonafide_logit - spoof_logit
    keys, scores, labels = Inferencer(
        model,
        device=device,
        amp=config.inference.amp,
    ).run(loader)

    output_path = write_submission(keys, scores, _absolute(config.inference.output_path))
    print(
        f"Сохранено {len(scores)} строк в {output_path}; "
        f"уникальных score: {len(set(scores))}"
    )
    if config.inference.compute_eer:
        eer, threshold = compute_eer_percent(scores, labels)
        print(f"Eval EER: {eer:.4f}% | threshold: {threshold:.6f}")

    # Отдельный EER помогает увидеть атаки, на которых модель ошибается чаще
    if config.inference.per_attack_output:
        attack_ids = [entry.attack_id for entry in dataset.entries]
        rows = []
        for attack_id in sorted(set(attack_ids) - {"-"}):
            indices = [
                index
                for index, current_attack in enumerate(attack_ids)
                if labels[index] == 1 or current_attack == attack_id
            ]
            attack_scores = [scores[index] for index in indices]
            attack_labels = [labels[index] for index in indices]
            attack_eer, _ = compute_eer_percent(attack_scores, attack_labels)
            rows.append((attack_id, attack_eer))

        attack_path = _absolute(config.inference.per_attack_output)
        attack_path.parent.mkdir(parents=True, exist_ok=True)
        with attack_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["attack_id", "eer_percent"])
            writer.writerows(rows)
        print("EER по атакам сохранён в", attack_path)


if __name__ == "__main__":
    main()
