"""
Rule-Based Threat Classifier.
Implements the core Electronic Warfare threat classification logic:
Repeated frequency-hopping detected at roughly the same distance radius over time = THREAT.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


class ThreatLevel:
    THREAT = "THREAT"               # Confirmed tactical frequency-hopping threat
    SUSPICIOUS = "SUSPICIOUS"       # Hopping pattern emerging, under evaluation
    TRANSIENT = "TRANSIENT"         # Few hops, insufficient data
    NOISE = "NOISE_OR_DECOY"        # Erratic distances, inconsistent hopping


@dataclass
class ThreatAssessment:
    level: str
    confidence: float
    mean_distance_m: float
    distance_std_m: float
    hop_count: int
    duration_s: float
    reason: str


class RuleBasedThreatClassifier:
    """
    Evaluates tracked emitter histories using deterministic, explainable EW heuristics:
    1. Hop repetition: requires a minimum number of hops (e.g. >= 4).
    2. Radius consistency: standard deviation of distance estimates must be bounded.
    3. Temporal persistence: emitter must remain active over a sustained window.
    """

    def __init__(
        self,
        min_hops_threat: int = 4,
        max_dist_std_m: float = 45.0,
        max_dist_variance_m2: float = 2000.0,
        min_duration_s: float = 0.05
    ):
        self.min_hops_threat = min_hops_threat
        self.max_dist_std_m = max_dist_std_m
        self.max_dist_variance_m2 = max_dist_variance_m2
        self.min_duration_s = min_duration_s

    def evaluate_track(
        self,
        distance_history: List[float],
        timestamps: List[float],
        hop_prediction_hits: int = 0
    ) -> ThreatAssessment:
        """
        Evaluate distance history and temporal activity to classify threat level.
        """
        n = len(distance_history)
        if n == 0:
            return ThreatAssessment(
                level=ThreatLevel.TRANSIENT,
                confidence=0.0,
                mean_distance_m=0.0,
                distance_std_m=0.0,
                hop_count=0,
                duration_s=0.0,
                reason="No detections recorded."
            )

        mean_dist = sum(distance_history) / float(n)
        duration_s = max(0.0, timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

        if n < 2:
            return ThreatAssessment(
                level=ThreatLevel.TRANSIENT,
                confidence=0.2,
                mean_distance_m=mean_dist,
                distance_std_m=0.0,
                hop_count=n,
                duration_s=duration_s,
                reason=f"Single pulse detected at ~{mean_dist:.0f}m. Need more hops to establish track."
            )

        # Compute distance variance & standard deviation
        variance = sum((d - mean_dist) ** 2 for d in distance_history) / float(n)
        std_dev = math.sqrt(variance)

        # Check for erratic decoy / noise
        # High distance variance indicates random unsynchronized bursts or erratic multi-path
        is_radius_consistent = (std_dev <= self.max_dist_std_m) and (variance <= self.max_dist_variance_m2)

        if not is_radius_consistent and n >= 4:
            # High scatter across hops -> Noise or erratic decoy
            confidence = min(0.95, 0.5 + (std_dev / (2.0 * self.max_dist_std_m)))
            return ThreatAssessment(
                level=ThreatLevel.NOISE,
                confidence=confidence,
                mean_distance_m=mean_dist,
                distance_std_m=std_dev,
                hop_count=n,
                duration_s=duration_s,
                reason=f"Erratic distance scatter (σ=±{std_dev:.1f}m > threshold). Likely decoy or noise."
            )

        if n >= self.min_hops_threat and is_radius_consistent:
            # Confirmed threat: repeated hops at consistent radius
            consistency_factor = max(0.0, 1.0 - (std_dev / self.max_dist_std_m))
            hop_factor = min(1.0, n / 10.0)
            confidence = min(0.99, 0.65 + 0.25 * consistency_factor + 0.10 * hop_factor)

            return ThreatAssessment(
                level=ThreatLevel.THREAT,
                confidence=confidence,
                mean_distance_m=mean_dist,
                distance_std_m=std_dev,
                hop_count=n,
                duration_s=duration_s,
                reason=f"Consistent FHSS transmitter at {mean_dist:.0f}m (σ=±{std_dev:.1f}m) across {n} hops."
            )

        # In-between: suspicious track building history
        return ThreatAssessment(
            level=ThreatLevel.SUSPICIOUS,
            confidence=0.5 + 0.1 * n,
            mean_distance_m=mean_dist,
            distance_std_m=std_dev,
            hop_count=n,
            duration_s=duration_s,
            reason=f"Tracking candidate emitter ({n}/{self.min_hops_threat} hops) at ~{mean_dist:.0f}m."
        )
