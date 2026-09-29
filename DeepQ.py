"""
Convenience entry point for Deep Q-Network Hop Predictor.
Exposes the DQN agent, environment, and training functions.
"""

from models.hop_predictor import (
    DQNHopPredictor,
    HopSequenceEnv,
    HopQNetwork,
    train_hop_predictor,
    evaluate_agent
)

__all__ = [
    "DQNHopPredictor",
    "HopSequenceEnv",
    "HopQNetwork",
    "train_hop_predictor",
    "evaluate_agent"
]

if __name__ == "__main__":
    train_hop_predictor()
