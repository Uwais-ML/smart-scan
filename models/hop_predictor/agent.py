"""
Deep Q-Network (DQN) Agent for FHSS Next-Hop Prediction.
Implements Q-Network, Replay Buffer, Epsilon-Greedy Exploration, Target Network,
and high-performance batch training in PyTorch.
"""

import math
import random
import os
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from configs.config import HopPredictorConfig, ChannelConfig


class HopQNetwork(nn.Module):
    """
    Feedforward Neural Network for estimating Q-values of each frequency channel.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for off-policy DQN training."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
        done_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor

    def __len__(self) -> int:
        return len(self.buffer)


class DQNHopPredictor:
    """
    DQN Agent that learns and predicts frequency hops.
    """

    def __init__(
        self,
        channel_config: Optional[ChannelConfig] = None,
        config: Optional[HopPredictorConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.channel_config = channel_config or ChannelConfig()
        self.config = config or HopPredictorConfig()
        self.num_channels = self.channel_config.num_channels
        self.history_len = self.config.history_len

        self.device = device or torch.device("cpu")

        # Networks
        self.policy_net = HopQNetwork(
            input_dim=self.history_len,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.num_channels
        ).to(self.device)

        self.target_net = HopQNetwork(
            input_dim=self.history_len,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.num_channels
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        self.epsilon = self.config.epsilon_start
        self.step_count = 0

    def select_action(self, state: List[float], evaluate: bool = False) -> Tuple[int, float]:
        """
        Select an action given state.
        
        Returns:
            (channel_idx, confidence_score)
        """
        if not evaluate and random.random() < self.epsilon:
            action = random.randint(0, self.num_channels - 1)
            confidence = 1.0 / self.num_channels
            return action, confidence

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action = int(q_values.argmax(dim=1).item())
            
            # Confidence via softmax over Q-values
            probs = F.softmax(q_values, dim=1)
            confidence = float(probs[0, action].item())
            
            return action, confidence

    def predict_from_channel_history(self, history: List[int]) -> Tuple[int, float]:
        """
        Inference helper: accepts list of raw channel indices (last K hops),
        normalizes them, and returns (predicted_next_channel, confidence).
        """
        if len(history) < self.history_len:
            # Pad with most recent or zeroes if history is short
            pad_val = history[-1] if history else 0
            padded = [pad_val] * (self.history_len - len(history)) + list(history)
        else:
            padded = list(history[-self.history_len:])

        norm_state = [ch / float(self.num_channels - 1) for ch in padded]
        return self.select_action(norm_state, evaluate=True)

    def train_step(self) -> Optional[float]:
        """Performs one step of Q-learning batch optimization."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        self.step_count += 1
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size, self.device
        )

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        # Target Q = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            expected_q = rewards + (self.config.gamma * next_q_values * (1.0 - dones))

        loss = F.smooth_l1_loss(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

        # Update target network
        if self.step_count % self.config.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, file_path: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count
        }, file_path)

    def load(self, file_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(file_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.config.epsilon_min)
        self.step_count = checkpoint.get("step_count", 0)
        self.policy_net.eval()
