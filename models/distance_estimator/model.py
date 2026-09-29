"""
Distance Estimator Deep Regression Network.
Predicts physical emitter distance (radius in meters) from RF features (RSSI, frequency, signal intensity).
"""

import math
import os
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import DistanceEstimatorConfig, ChannelConfig


class DistanceRegressionNet(nn.Module):
    """
    Multilayer Perceptron for estimating physical distance from RF features.
    Uses Softplus on final layer to enforce strictly positive physical distance predictions.
    """

    def __init__(self, input_dim: int = 4, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.05))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        # Ensure positive distance
        layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DistanceEstimator:
    """
    Inference and prediction wrapper for physical distance estimation.
    Includes feature normalization, uncertainty estimation, and checkpoint persistence.
    """

    def __init__(
        self,
        config: Optional[DistanceEstimatorConfig] = None,
        channel_config: Optional[ChannelConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or DistanceEstimatorConfig()
        self.channel_config = channel_config or ChannelConfig()
        self.device = device or torch.device("cpu")

        self.model = DistanceRegressionNet(
            input_dim=4,
            hidden_dims=self.config.hidden_dims
        ).to(self.device)

        # Feature normalization scales
        self.rssi_min = -110.0
        self.rssi_max = 10.0
        self.freq_min = self.channel_config.min_freq_mhz
        self.freq_max = self.channel_config.max_freq_mhz

    def extract_features(self, rssi: float, freq_mhz: float, intensity: float) -> List[float]:
        """
        Normalize and engineer input feature vector:
        [rssi_norm, freq_norm, intensity, rssi_norm_sq]
        """
        rssi_norm = (rssi - self.rssi_min) / (self.rssi_max - self.rssi_min)
        freq_norm = (freq_mhz - self.freq_min) / max(1.0, (self.freq_max - self.freq_min))
        intensity_clamped = max(0.0, min(1.0, intensity))
        return [rssi_norm, freq_norm, intensity_clamped, rssi_norm ** 2]

    def predict(self, rssi: float, freq_mhz: float, intensity: float) -> Tuple[float, float]:
        """
        Estimate physical distance (in meters) and standard error radius.
        
        Returns:
            (distance_m, radius_uncertainty_m)
        """
        self.model.eval()
        features = self.extract_features(rssi, freq_mhz, intensity)
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=self.device)
            pred_dist = self.model(x).item()

        # Uncertainty is proportional to distance and inversely proportional to intensity
        uncertainty_m = max(5.0, pred_dist * 0.12 * (1.5 - intensity))
        return float(pred_dist), float(uncertainty_m)

    def save(self, file_path: str) -> None:
        """Save model weights."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "rssi_min": self.rssi_min,
            "rssi_max": self.rssi_max
        }, file_path)

    def load(self, file_path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.rssi_min = checkpoint.get("rssi_min", -110.0)
        self.rssi_max = checkpoint.get("rssi_max", 10.0)
        self.model.eval()
