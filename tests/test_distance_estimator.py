"""Unit tests for Distance Estimator regression model and dataset generation."""

import unittest
import torch
from configs.config import ChannelConfig, DistanceEstimatorConfig
from models.distance_estimator.model import DistanceEstimator, DistanceRegressionNet
from models.distance_estimator.dataset import generate_rf_distance_data


class TestDistanceEstimator(unittest.TestCase):

    def setUp(self):
        self.channel_cfg = ChannelConfig()
        self.dist_cfg = DistanceEstimatorConfig()
        self.estimator = DistanceEstimator(config=self.dist_cfg, channel_config=self.channel_cfg)

    def test_dataset_generation(self):
        """Dataset generator should produce correct number of valid samples."""
        samples = generate_rf_distance_data(num_samples=50, channel_config=self.channel_cfg)
        self.assertEqual(len(samples), 50)
        for feats, dist in samples:
            self.assertEqual(len(feats), 4)
            self.assertGreater(dist, 0.0)

    def test_model_forward_pass_and_positive_distance(self):
        """DistanceRegressionNet should always output strictly positive distance predictions."""
        net = DistanceRegressionNet(input_dim=4, hidden_dims=[32, 16])
        dummy_input = torch.randn(8, 4)
        output = net(dummy_input)
        self.assertEqual(output.shape, (8, 1))
        self.assertTrue((output >= 0.0).all())

    def test_estimator_prediction(self):
        """DistanceEstimator.predict should return positive distance and uncertainty radius in meters."""
        dist_m, unc_m = self.estimator.predict(rssi=-55.0, freq_mhz=450.0, intensity=0.85)
        self.assertGreater(dist_m, 0.0)
        self.assertGreater(unc_m, 0.0)


if __name__ == "__main__":
    unittest.main()
