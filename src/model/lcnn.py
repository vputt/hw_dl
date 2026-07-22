import torch
from torch import Tensor, nn


class MaxFeatureMap(nn.Module):
    """Делит каналы пополам и оставляет поэлементный максимум"""

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        first, second = torch.chunk(inputs, chunks=2, dim=self.dim)
        return torch.maximum(first, second)


class LCNN(nn.Module):
    """Light CNN для классификации LFCC-признаков на spoof и bonafide"""

    def __init__(
        self,
        feature_bins: int = 60,
        time_frames: int = 750,
        num_classes: int = 2,
        dropout_p: float = 0.25,
    ) -> None:
        super().__init__()
        self.feature_bins = feature_bins
        self.time_frames = time_frames

        # После каждого MFM число каналов уменьшается в два раза.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, bias=False),
            MaxFeatureMap(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, padding=1, bias=False),
            MaxFeatureMap(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            MaxFeatureMap(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, padding=1, bias=False),
            MaxFeatureMap(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            MaxFeatureMap(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            MaxFeatureMap(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        pooled_frequency = self._pooled_size(feature_bins)
        pooled_time = self._pooled_size(time_frames)
        flattened_size = 32 * pooled_frequency * pooled_time

        # Dropout расположен перед финальным BatchNorm, как требуется в задании
        self.embedding = nn.Sequential(
            nn.Linear(flattened_size, 160),
            MaxFeatureMap(),
            nn.Dropout(p=dropout_p),
            nn.BatchNorm1d(80),
        )
        self.classifier = nn.Linear(80, num_classes)
        self.apply(self._initialize_weights)

    def _pooled_size(self, size: int) -> int:
        for _ in range(4):
            size //= 2
        return size

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, features: Tensor) -> Tensor:
        convolutional = self.features(features)
        embedding = self.embedding(torch.flatten(convolutional, start_dim=1))
        return self.classifier(embedding)

    def score_from_logits(self, logits: Tensor) -> Tensor:
        """Чем выше score, тем больше модель уверена в классе bonafide"""
        return logits[:, 1] - logits[:, 0]
