import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.model.lcnn import LCNN


class Inferencer:
    def __init__(self, model: LCNN, device: torch.device, amp: bool = True) -> None:
        self.model = model.to(device)
        self.device = device
        self.amp_enabled = amp and device.type == "cuda"

    @torch.inference_mode()
    def run(self, loader: DataLoader) -> tuple[list[str], list[float], list[int]]:
        self.model.eval()
        keys: list[str] = []
        scores: list[float] = []
        labels: list[int] = []
        for batch in tqdm(loader, desc="inference", leave=False):
            features = batch["features"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(features)
            batch_scores = self.model.score_from_logits(logits).float().cpu().tolist()
            keys.extend(batch["key"])
            scores.extend(batch_scores)
            labels.extend(batch["label"].tolist())
        return keys, scores, labels

