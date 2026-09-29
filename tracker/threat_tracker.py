"""
Threat Tracker Module.
Maintains state for all detected RF emitters, performs spatial-spectral track association,
integrates DQN next-hop predictions and Distance Estimator regression, and runs continuous
threat assessment.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from configs.config import ChannelConfig, ThreatTrackerConfig
from sim.types import SignalReading
from sim.channel import RFChannel
from models.hop_predictor.agent import DQNHopPredictor
from models.distance_estimator.model import DistanceEstimator
from models.threat_classifier.rules import RuleBasedThreatClassifier, ThreatAssessment, ThreatLevel


@dataclass
class EmitterTrack:
    """Stateful representation of a tracked RF emitter entity."""
    track_id: str
    first_seen: float
    last_seen: float
    hop_channels: List[int] = field(default_factory=list)
    hop_frequencies: List[float] = field(default_factory=list)
    rssi_history: List[float] = field(default_factory=list)
    intensity_history: List[float] = field(default_factory=list)
    distance_history: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    # Hop prediction state
    predicted_next_channel: Optional[int] = None
    predicted_next_freq_mhz: Optional[float] = None
    prediction_confidence: float = 0.0
    prediction_hits: int = 0
    
    # Distance state
    estimated_radius_m: float = 0.0
    radius_uncertainty_m: float = 0.0
    
    # Threat assessment state
    threat_assessment: Optional[ThreatAssessment] = None
    
    # Ground truth (if available in simulation for diagnostic scoring)
    ground_truth_emitter_id: Optional[str] = None
    ground_truth_is_threat: Optional[bool] = None

    @property
    def hop_count(self) -> int:
        return len(self.hop_channels)

    @property
    def threat_level(self) -> str:
        return self.threat_assessment.level if self.threat_assessment else ThreatLevel.TRANSIENT

    @property
    def threat_confidence(self) -> float:
        return self.threat_assessment.confidence if self.threat_assessment else 0.0


class ThreatTracker:
    """
    Central tracking engine for the Smart Scan Electronic Warfare system.
    Ties together the Hop Predictor (DQN), Distance Estimator, and Threat Classifier.
    """

    def __init__(
        self,
        channel_config: Optional[ChannelConfig] = None,
        tracker_config: Optional[ThreatTrackerConfig] = None,
        hop_predictor: Optional[DQNHopPredictor] = None,
        distance_estimator: Optional[DistanceEstimator] = None,
        threat_classifier: Optional[RuleBasedThreatClassifier] = None
    ):
        self.channel_config = channel_config or ChannelConfig()
        self.tracker_config = tracker_config or ThreatTrackerConfig()
        self.channel_model = RFChannel(self.channel_config)

        self.hop_predictor = hop_predictor or DQNHopPredictor(channel_config=self.channel_config)
        self.distance_estimator = distance_estimator or DistanceEstimator(channel_config=self.channel_config)
        self.threat_classifier = threat_classifier or RuleBasedThreatClassifier()

        self.tracks: Dict[str, EmitterTrack] = {}
        self.track_counter = 0
        self.last_update_time = 0.0

    def _generate_track_id(self) -> str:
        self.track_counter += 1
        return f"TRK-{self.track_counter:03d}"

    def update(self, reading: SignalReading) -> EmitterTrack:
        """
        Process an incoming signal reading:
        1. Predict distance radius via Distance Estimator.
        2. Associate reading with best matching track or create new track.
        3. Predict next frequency hop via DQN Hop Predictor.
        4. Update threat assessment via Threat Classifier.
        """
        self.last_update_time = reading.timestamp

        # 1. Estimate distance
        pred_distance, pred_uncertainty = self.distance_estimator.predict(
            rssi=reading.rssi,
            freq_mhz=reading.frequency,
            intensity=reading.intensity
        )

        # 2. Track Association: Find best matching candidate track
        best_track_id = self._find_matching_track(reading, pred_distance)

        if best_track_id is None:
            # Create a new track
            track_id = self._generate_track_id()
            track = EmitterTrack(
                track_id=track_id,
                first_seen=reading.timestamp,
                last_seen=reading.timestamp,
                ground_truth_emitter_id=reading.ground_truth_emitter_id,
                ground_truth_is_threat=reading.ground_truth_is_threat
            )
            self.tracks[track_id] = track
        else:
            track = self.tracks[best_track_id]
            # Check if this reading verified a prior hop prediction
            if track.predicted_next_channel is not None and track.predicted_next_channel == reading.channel_idx:
                track.prediction_hits += 1

        # 3. Update track observation history
        track.last_seen = reading.timestamp
        track.hop_channels.append(reading.channel_idx)
        track.hop_frequencies.append(reading.frequency)
        track.rssi_history.append(reading.rssi)
        track.intensity_history.append(reading.intensity)
        track.distance_history.append(pred_distance)
        track.timestamps.append(reading.timestamp)

        # Maintain bounded history window (last 50 hops)
        max_history = 50
        if len(track.hop_channels) > max_history:
            track.hop_channels = track.hop_channels[-max_history:]
            track.hop_frequencies = track.hop_frequencies[-max_history:]
            track.rssi_history = track.rssi_history[-max_history:]
            track.intensity_history = track.intensity_history[-max_history:]
            track.distance_history = track.distance_history[-max_history:]
            track.timestamps = track.timestamps[-max_history:]

        # Update smoothed radius estimate
        track.estimated_radius_m = sum(track.distance_history) / len(track.distance_history)
        track.radius_uncertainty_m = pred_uncertainty

        # 4. Predict next hop using DQN
        next_channel, confidence = self.hop_predictor.predict_from_channel_history(track.hop_channels)
        track.predicted_next_channel = next_channel
        track.predicted_next_freq_mhz = self.channel_model.channel_idx_to_frequency(next_channel)
        track.prediction_confidence = confidence

        # 5. Evaluate threat status
        track.threat_assessment = self.threat_classifier.evaluate_track(
            distance_history=track.distance_history,
            timestamps=track.timestamps,
            hop_prediction_hits=track.prediction_hits
        )

        return track

    def _find_matching_track(self, reading: SignalReading, pred_distance: float) -> Optional[str]:
        """
        Spatial-spectral track association algorithm.
        Matches detection against candidate active tracks based on:
        - Hop prediction match (high affinity)
        - Distance proximity (must be within distance tolerance)
        - Temporal recency
        """
        best_track_id = None
        lowest_cost = float("inf")

        for track_id, track in self.tracks.items():
            dt = reading.timestamp - track.last_seen
            if dt > self.tracker_config.max_track_idle_seconds:
                continue

            # Spectral match bonus
            spectral_match = (track.predicted_next_channel == reading.channel_idx)

            # Distance deviation
            dist_diff = abs(pred_distance - track.estimated_radius_m)

            # Rejection threshold for distance
            if dist_diff > self.tracker_config.association_dist_threshold_m * 2.5:
                continue

            # Cost function
            cost = dist_diff
            if spectral_match:
                cost *= 0.2  # Significant cost reduction if DQN predicted this exact channel

            if cost < lowest_cost and cost < self.tracker_config.association_dist_threshold_m * 2.0:
                lowest_cost = cost
                best_track_id = track_id

        return best_track_id

    def prune_idle_tracks(self, current_time: Optional[float] = None, timeout_s: float = 4.0) -> List[str]:
        """Remove tracks that haven't hopped within timeout window."""
        now = self.last_update_time if current_time is None else current_time
        pruned_ids = []

        for track_id in list(self.tracks.keys()):
            if now - self.tracks[track_id].last_seen > timeout_s:
                del self.tracks[track_id]
                pruned_ids.append(track_id)

        return pruned_ids

    def get_active_tracks(self) -> List[EmitterTrack]:
        """Return list of all currently tracked emitters."""
        return list(self.tracks.values())

    def get_threats(self) -> List[EmitterTrack]:
        """Return list of confirmed threat emitters."""
        return [t for t in self.tracks.values() if t.threat_level == ThreatLevel.THREAT]

    def get_summary(self) -> Dict[str, Any]:
        """Generate a high-level operational summary."""
        tracks = list(self.tracks.values())
        threats = [t for t in tracks if t.threat_level == ThreatLevel.THREAT]
        suspicious = [t for t in tracks if t.threat_level == ThreatLevel.SUSPICIOUS]
        noise = [t for t in tracks if t.threat_level == ThreatLevel.NOISE]

        return {
            "total_active_tracks": len(tracks),
            "threat_count": len(threats),
            "suspicious_count": len(suspicious),
            "noise_count": len(noise),
            "threat_ids": [t.track_id for t in threats]
        }
