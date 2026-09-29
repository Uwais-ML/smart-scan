"""Unit tests for Threat Classifier and Threat Tracker."""

import unittest
from configs.config import ChannelConfig, ThreatTrackerConfig
from models.threat_classifier.rules import RuleBasedThreatClassifier, ThreatLevel
from tracker.threat_tracker import ThreatTracker
from sim.types import SignalReading


class TestThreatTracker(unittest.TestCase):

    def setUp(self):
        self.channel_cfg = ChannelConfig()
        self.tracker_cfg = ThreatTrackerConfig(
            association_dist_threshold_m=45.0,
            min_hops_for_classification=4,
            radius_variance_threshold_m2=2000.0
        )
        self.classifier = RuleBasedThreatClassifier(
            min_hops_threat=4,
            max_dist_variance_m2=2000.0
        )
        self.tracker = ThreatTracker(
            channel_config=self.channel_cfg,
            tracker_config=self.tracker_cfg,
            threat_classifier=self.classifier
        )

    def test_threat_rule_consistent_vs_erratic(self):
        """Repeated hops at consistent radius should classify as THREAT, erratic as NOISE."""
        # Consistent distances (e.g. 100m +/- 5m)
        consistent_distances = [102.0, 98.0, 105.0, 96.0, 100.0]
        timestamps = [0.0, 0.02, 0.04, 0.06, 0.08]
        assessment_threat = self.classifier.evaluate_track(consistent_distances, timestamps)
        self.assertEqual(assessment_threat.level, ThreatLevel.THREAT)

        # Erratic distances (e.g. 30m, 450m, 90m, 600m)
        erratic_distances = [30.0, 450.0, 90.0, 600.0, 120.0]
        assessment_noise = self.classifier.evaluate_track(erratic_distances, timestamps)
        self.assertEqual(assessment_noise.level, ThreatLevel.NOISE)

    def test_tracker_ingestion_and_track_creation(self):
        """Tracker should create and update tracks on incoming signal readings."""
        r1 = SignalReading(
            timestamp=0.01,
            frequency=420.0,
            rssi=-48.0,
            intensity=0.9,
            channel_idx=5,
            ground_truth_emitter_id="TEST_EMITTER"
        )
        track = self.tracker.update(r1)
        self.assertIsNotNone(track)
        self.assertEqual(len(self.tracker.tracks), 1)
        self.assertEqual(track.hop_count, 1)

    def test_tracker_pruning(self):
        """Idle tracks beyond timeout threshold should be pruned."""
        r1 = SignalReading(timestamp=0.0, frequency=420.0, rssi=-50.0, intensity=0.9, channel_idx=2)
        self.tracker.update(r1)
        self.assertEqual(len(self.tracker.tracks), 1)

        # Prune at time t = 10.0s (timeout is 3.0s)
        pruned = self.tracker.prune_idle_tracks(current_time=10.0, timeout_s=3.0)
        self.assertEqual(len(pruned), 1)
        self.assertEqual(len(self.tracker.tracks), 0)


if __name__ == "__main__":
    unittest.main()
