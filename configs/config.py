"""
Smart Scan Configuration
Central configuration parameters for simulation, models, tracker, and pipeline.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

@dataclass
class ChannelConfig:
    num_channels: int = 64
    min_freq_mhz: float = 400.0
    max_freq_mhz: float = 500.0
    ref_distance_m: float = 1.0
    ref_rssi_at_1m_dbm: float = -20.0
    path_loss_exponent: float = 2.8
    shadowing_std_db: float = 2.5
    noise_floor_dbm: float = -95.0

@dataclass
class HopPredictorConfig:
    history_len: int = 8
    hidden_dim: int = 128
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 64
    target_update_steps: int = 100

@dataclass
class DistanceEstimatorConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    distance_tolerance_m: float = 15.0

@dataclass
class ThreatTrackerConfig:
    association_dist_threshold_m: float = 45.0
    min_hops_for_classification: int = 4
    radius_variance_threshold_m2: float = 2000.0
    max_track_idle_seconds: float = 3.0
    threat_confidence_threshold: float = 0.75

@dataclass
class SystemConfig:
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    hop_predictor: HopPredictorConfig = field(default_factory=HopPredictorConfig)
    distance_estimator: DistanceEstimatorConfig = field(default_factory=DistanceEstimatorConfig)
    tracker: ThreatTrackerConfig = field(default_factory=ThreatTrackerConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SystemConfig":
        data = json.loads(json_str)
        return cls(
            channel=ChannelConfig(**data.get("channel", {})),
            hop_predictor=HopPredictorConfig(**data.get("hop_predictor", {})),
            distance_estimator=DistanceEstimatorConfig(**data.get("distance_estimator", {})),
            tracker=ThreatTrackerConfig(**data.get("tracker", {}))
        )
