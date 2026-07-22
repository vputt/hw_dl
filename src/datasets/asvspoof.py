from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.datasets.protocol import ProtocolEntry, read_protocol


class ASVspoofDataset(Dataset[dict[str, Any]]):
    """Читает ASVspoof 2019 LA в порядке официального протокола"""

    def __init__(
        self,
        protocol_path: str | Path,
        audio_dir: str | Path,
        transform: Callable[[Tensor], Tensor] | None = None,
        extension: str = ".flac",
        expected_sample_rate: int = 16_000,
    ) -> None:
        self.entries = read_protocol(protocol_path)
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        self.extension = extension if extension.startswith(".") else f".{extension}"
        self.expected_sample_rate = expected_sample_rate

    def __len__(self) -> int:
        return len(self.entries)

    def _audio_path(self, entry: ProtocolEntry) -> Path:
        return self.audio_dir / f"{entry.key}{self.extension}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Импорт внутри Dataset не загружает torchaudio до первого обращения к аудио
        import torchaudio

        entry = self.entries[index]
        audio_path = self._audio_path(entry)
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.expected_sample_rate:
            raise ValueError(
                f"{audio_path}: expected {self.expected_sample_rate} Hz, got {sample_rate} Hz"
            )
        if waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)
        features = self.transform(waveform) if self.transform is not None else waveform
        return {
            "features": features,
            "label": torch.tensor(entry.label_index, dtype=torch.long),
            "key": entry.key,
            "attack_id": entry.attack_id,
        }
