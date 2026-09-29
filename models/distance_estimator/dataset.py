"""
Dataset Generator for Distance Estimator.
Generates synthetic (RSSI, Frequency, Intensity) -> Distance training samples
using physical RF propagation channel physics.
"""

import math
import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from configs.config import ChannelConfig
from sim.channel import RFChannel


class RFDistanceDataset(Dataset):
    """PyTorch Dataset holding RF feature vectors and ground truth distances."""

    def __init__(self, samples: List[Tuple[List[float], float]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, distance = self.samples[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor([distance], dtype=torch.float32)
        )


def generate_rf_distance_data(
    num_samples: int = 5000,
    channel_config: ChannelConfig = None,
    min_dist_m: float = 5.0,
    max_dist_m: float = 600.0,
    seed: int = 42
) -> List[Tuple[List[float], float]]:
    """
    Synthesize dataset by sampling random emitter distances, transmit powers, and frequencies.
    """
    channel_cfg = channel_config or ChannelConfig()
    channel_model = RFChannel(channel_cfg)
    rng = random.Random(seed)

    rssi_min = -110.0
    rssi_max = 10.0
    freq_min = channel_cfg.min_freq_mhz
    freq_max = channel_cfg.max_freq_mhz

    dataset: List[Tuple[List[float], float]] = []

    for _ in range(num_samples):
        # Sample distance (log-uniform distribution across distance range)
        log_d = rng.uniform(math.log(min_dist_m), math.log(max_dist_m))
        distance_m = math.exp(log_d)

        # Sample transmit power (typical FHSS handhelds / vehicle radios: 15 dBm to 30 dBm)
        tx_power_dbm = rng.uniform(15.0, 28.0)

        # Sample frequency
        freq_mhz = rng.uniform(freq_min, freq_max)

        # Compute simulated RSSI and intensity with shadowing and multipath fading
        rssi, intensity = channel_model.compute_rssi(
            tx_power_dbm=tx_power_dbm,
            distance_m=distance_m,
            freq_mhz=freq_mhz,
            add_shadowing=True,
            add_fast_fading=True
        )

        # Feature normalization
        rssi_norm = (rssi - rssi_min) / (rssi_max - rssi_min)
        freq_norm = (freq_mhz - freq_min) / max(1.0, (freq_max - freq_min))
        intensity_clamped = max(0.0, min(1.0, intensity))

        features = [rssi_norm, freq_norm, intensity_clamped, rssi_norm ** 2]
        dataset.append((features, distance_m))

    return dataset
