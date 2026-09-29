"""Unit tests for Hop Predictor environment, DQN network, and agent."""

import unittest
import torch
from configs.config import ChannelConfig, HopPredictorConfig
from models.hop_predictor.env import HopSequenceEnv
from models.hop_predictor.agent import DQNHopPredictor, HopQNetwork


class TestHopPredictor(unittest.TestCase):

    def setUp(self):
        self.channel_cfg = ChannelConfig(num_channels=16)
        self.hop_cfg = HopPredictorConfig(history_len=4, hidden_dim=32, batch_size=8)
        self.env = HopSequenceEnv(
            channel_config=self.channel_cfg,
            hop_config=self.hop_cfg,
            seed=42
        )
        self.agent = DQNHopPredictor(
            channel_config=self.channel_cfg,
            config=self.hop_cfg
        )

    def test_env_reset_and_step(self):
        """Environment state must have length equal to history_len and values in [0, 1]."""
        state = self.env.reset()
        self.assertEqual(len(state), self.hop_cfg.history_len)
        for val in state:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

        next_state, reward, done, info = self.env.step(action_channel_idx=0)
        self.assertEqual(len(next_state), self.hop_cfg.history_len)
        self.assertIn("hit", info)
        self.assertIn("actual_channel", info)

    def test_q_network_forward(self):
        """HopQNetwork should output Q-values tensor of shape [batch_size, num_channels]."""
        net = HopQNetwork(input_dim=4, hidden_dim=32, output_dim=16)
        dummy_input = torch.randn(4, 4)
        output = net(dummy_input)
        self.assertEqual(output.shape, (4, 16))

    def test_agent_prediction_and_replay(self):
        """Agent should store transitions and predict next hop channel within valid range."""
        state = self.env.reset()
        action, conf = self.agent.select_action(state, evaluate=True)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.channel_cfg.num_channels)
        self.assertGreaterEqual(conf, 0.0)

        # Push to replay buffer
        self.agent.replay_buffer.push(state, action, 1.0, state, False)
        self.assertEqual(len(self.agent.replay_buffer), 1)


if __name__ == "__main__":
    unittest.main()
