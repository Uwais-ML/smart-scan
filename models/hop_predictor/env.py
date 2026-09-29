"""
Hop Prediction Reinforcement Learning Environment.
Provides a Gym-style interface for training a Deep Q-Network to predict
the next frequency hop channel in an FHSS sequence.
"""

import random
from typing import List, Tuple, Dict, Any, Optional
from configs.config import ChannelConfig, HopPredictorConfig
from sim.emitter import FrequencyHoppingEmitter


class HopSequenceEnv:
    """
    RL Environment for learning FHSS frequency hopping sequences.
    
    State: Rolling history of the last K channel indices (normalized to [0, 1]).
    Action: Discrete index of the predicted next channel (0 to num_channels - 1).
    Reward: +1.0 for correct prediction, -0.1 for adjacent channel, -0.3 for miss.
    """

    def __init__(
        self,
        channel_config: Optional[ChannelConfig] = None,
        hop_config: Optional[HopPredictorConfig] = None,
        emitter: Optional[FrequencyHoppingEmitter] = None,
        pattern_type: str = "lcg",
        seed: int = 42
    ):
        self.channel_config = channel_config or ChannelConfig()
        self.hop_config = hop_config or HopPredictorConfig()
        self.num_channels = self.channel_config.num_channels
        self.history_len = self.hop_config.history_len
        self.seed = seed
        self.pattern_type = pattern_type

        if emitter is None:
            self.emitter = FrequencyHoppingEmitter(
                emitter_id="ENV_EMITTER",
                channel_config=self.channel_config,
                seed=self.seed,
                pattern_type=self.pattern_type
            )
        else:
            self.emitter = emitter

        self.history: List[int] = []
        self.current_step = 0
        self.max_steps_per_episode = 1000
        self.reset()

    def _get_normalized_state(self) -> List[float]:
        """Normalize discrete channel indices to [0.0, 1.0] for the neural network."""
        return [ch / float(self.num_channels - 1) for ch in self.history]

    def reset(self, seed: Optional[int] = None) -> List[float]:
        """Reset the environment and generate initial history buffer."""
        if seed is not None:
            self.seed = seed
            self.emitter = FrequencyHoppingEmitter(
                emitter_id="ENV_EMITTER",
                channel_config=self.channel_config,
                seed=self.seed,
                pattern_type=self.pattern_type
            )
        else:
            self.emitter.reset()

        self.history = []
        for _ in range(self.history_len):
            _, _, _, _, ch_idx = self.emitter.next_hop()
            self.history.append(ch_idx)

        self.current_step = 0
        return self._get_normalized_state()

    def step(self, action_channel_idx: int) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        """
        Execute one action step.
        
        Args:
            action_channel_idx: Predicted channel index (0 to num_channels - 1)
            
        Returns:
            (next_state, reward, done, info)
        """
        self.current_step += 1

        # Advance emitter to obtain ground truth next hop
        _, _, _, _, ground_truth_next_ch = self.emitter.next_hop()

        # Compute reward
        is_hit = (action_channel_idx == ground_truth_next_ch)
        if is_hit:
            reward = 1.0
        elif abs(action_channel_idx - ground_truth_next_ch) == 1:
            reward = -0.1
        else:
            reward = -0.3

        # Update history window
        self.history.pop(0)
        self.history.append(ground_truth_next_ch)

        done = (self.current_step >= self.max_steps_per_episode)
        info = {
            "hit": is_hit,
            "predicted_channel": action_channel_idx,
            "actual_channel": ground_truth_next_ch,
            "step": self.current_step
        }

        return self._get_normalized_state(), reward, done, info
