import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LFCCFrontend(nn.Module):
    """Строит LFCC, delta и delta-delta"""

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        win_length: int = 320,
        hop_length: int = 160,
        num_filters: int = 20,
        num_ceps: int = 20,
        delta_window: int = 2,
        time_frames: int = 750,
        log_eps: float = 1.0e-12,
        random_crop: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_filters = num_filters
        self.num_ceps = num_ceps
        self.delta_window = delta_window
        self.time_frames = time_frames
        self.log_eps = log_eps
        self.random_crop = random_crop

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer("filterbank", self._build_filterbank(), persistent=False)
        self.register_buffer("dct_matrix", self._build_dct_matrix(), persistent=False)

    def _build_filterbank(self) -> Tensor:
        """Треугольные фильтры поверх STFT"""
        frequencies = torch.linspace(0.0, self.sample_rate / 2, self.n_fft // 2 + 1)
        points = torch.linspace(0.0, self.sample_rate / 2, self.num_filters + 2)
        filters = []

        for index in range(self.num_filters):
            left, center, right = points[index : index + 3]
            rising = (frequencies - left) / (center - left)
            falling = (right - frequencies) / (right - center)
            filters.append(torch.minimum(rising, falling).clamp_min(0.0))

        return torch.stack(filters)

    def _build_dct_matrix(self) -> Tensor:
        """DCT переводит log-энергии фильтров в коэффициенты"""
        coefficient = torch.arange(self.num_ceps, dtype=torch.float32).unsqueeze(1)
        channel = torch.arange(self.num_filters, dtype=torch.float32).unsqueeze(0)
        matrix = torch.cos(
            math.pi / self.num_filters * (channel + 0.5) * coefficient
        )
        matrix[0] *= math.sqrt(1.0 / self.num_filters)
        matrix[1:] *= math.sqrt(2.0 / self.num_filters)
        return matrix

    def compute_delta(self, features: Tensor, window_length: int = 2) -> Tensor:
        """Считает изменение признаков по времени"""
        padded = F.pad(
            features.unsqueeze(0),
            (window_length, window_length),
            mode="replicate",
        ).squeeze(0)
        delta = torch.zeros_like(features)

        for offset in range(1, window_length + 1):
            right = padded[
                :, window_length + offset : window_length + offset + features.shape[1]
            ]
            left = padded[
                :, window_length - offset : window_length - offset + features.shape[1]
            ]
            delta += offset * (right - left)

        denominator = 2 * sum(offset**2 for offset in range(1, window_length + 1))
        return delta / denominator

    def compute_lfcc(self, waveform: Tensor) -> Tensor:
        if waveform.numel() < self.n_fft:
            waveform = F.pad(waveform, (0, self.n_fft - waveform.numel()))

        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=waveform.dtype, device=waveform.device),
            center=False,
            return_complex=True,
        )

        # STFT -> мощность -> линейный filter bank -> log -> DCT
        power = spectrum.abs().square()
        filtered_power = self.filterbank.to(power) @ power
        log_filterbank = torch.log(filtered_power.clamp_min(self.log_eps))

        static_lfcc = self.dct_matrix.to(log_filterbank) @ log_filterbank
        log_energy = torch.log(power.sum(dim=0).clamp_min(self.log_eps))
        static_lfcc = torch.cat([log_energy.unsqueeze(0), static_lfcc[1:]], dim=0)

        delta = self.compute_delta(static_lfcc, self.delta_window)
        delta_delta = self.compute_delta(delta, self.delta_window)
        return torch.cat([static_lfcc, delta, delta_delta], dim=0)

    def fix_time_length(self, features: Tensor) -> Tensor:
        """Приводит запись к 750 кадрам с помощью crop или zero-padding"""
        current_frames = features.shape[-1]

        if current_frames > self.time_frames:
            max_start = current_frames - self.time_frames
            start = (
                int(torch.randint(max_start + 1, size=(1,)).item())
                if self.random_crop
                else max_start // 2
            )
            return features[..., start : start + self.time_frames]

        if current_frames < self.time_frames:
            missing = self.time_frames - current_frames
            left = missing // 2
            return F.pad(features, (left, missing - left), value=0.0)

        return features

    def forward(self, waveform: Tensor) -> Tensor:
        lfcc = self.compute_lfcc(waveform)
        return self.fix_time_length(lfcc).unsqueeze(0)
