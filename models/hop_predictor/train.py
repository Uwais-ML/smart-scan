"""
Training and Evaluation script for Hop Predictor DQN.
Trains the DQN on frequency hopping sequences and validates performance against
naive baselines (uniform random guess and most-recent-channel repeat).
"""

import os
import random
from typing import Dict, Tuple

import torch
from configs.config import ChannelConfig, HopPredictorConfig
from models.hop_predictor.env import HopSequenceEnv
from models.hop_predictor.agent import DQNHopPredictor
from sim.emitter import FrequencyHoppingEmitter


def evaluate_agent(
    agent: DQNHopPredictor,
    eval_env: HopSequenceEnv,
    num_eval_steps: int = 500
) -> Dict[str, float]:
    """
    Evaluates the trained DQN predictor against naive baselines.
    """
    state = eval_env.reset()
    dqn_hits = 0
    repeat_baseline_hits = 0
    random_baseline_hits = 0

    for _ in range(num_eval_steps):
        # 1. DQN prediction
        action, _ = agent.select_action(state, evaluate=True)

        # 2. Naive Repeat Baseline (predicts same channel as the latest hop)
        repeat_action = eval_env.history[-1]

        # 3. Uniform Random Baseline
        random_action = random.randint(0, eval_env.num_channels - 1)

        next_state, _, _, info = eval_env.step(action)
        actual = info["actual_channel"]

        if action == actual:
            dqn_hits += 1
        if repeat_action == actual:
            repeat_baseline_hits += 1
        if random_action == actual:
            random_baseline_hits += 1

        state = next_state

    return {
        "dqn_accuracy": dqn_hits / float(num_eval_steps),
        "repeat_baseline_accuracy": repeat_baseline_hits / float(num_eval_steps),
        "random_baseline_accuracy": random_baseline_hits / float(num_eval_steps),
        "total_steps": float(num_eval_steps)
    }


def train_hop_predictor(
    num_episodes: int = 60,
    steps_per_episode: int = 250,
    save_path: str = "models/checkpoints/hop_dqn.pt"
) -> Tuple[DQNHopPredictor, Dict[str, float]]:
    """
    Train DQN Hop Predictor across multiple hopping sequences and evaluate.
    """
    channel_cfg = ChannelConfig(num_channels=64)
    hop_cfg = HopPredictorConfig(
        history_len=8,
        hidden_dim=128,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.998,
        batch_size=64
    )

    env = HopSequenceEnv(
        channel_config=channel_cfg,
        hop_config=hop_cfg,
        pattern_type="lcg",
        seed=42
    )

    agent = DQNHopPredictor(channel_config=channel_cfg, config=hop_cfg)

    print("=== Training DQN Hop Predictor ===")
    for episode in range(1, num_episodes + 1):
        state = env.reset(seed=42 + (episode % 5))
        episode_reward = 0.0
        episode_hits = 0

        for step in range(steps_per_episode):
            action, _ = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            episode_reward += reward
            if info["hit"]:
                episode_hits += 1

        if episode % 10 == 0 or episode == num_episodes:
            acc = episode_hits / float(steps_per_episode)
            print(f"Episode {episode:03d}/{num_episodes:03d} | Reward: {episode_reward:+6.1f} | Acc: {acc*100:5.1f}% | Epsilon: {agent.epsilon:.3f}")

    # Evaluate on held-out seed sequence
    eval_emitter = FrequencyHoppingEmitter(
        emitter_id="EVAL_EMITTER",
        channel_config=channel_cfg,
        seed=999,
        pattern_type="lcg"
    )
    eval_env = HopSequenceEnv(
        channel_config=channel_cfg,
        hop_config=hop_cfg,
        emitter=eval_emitter
    )

    metrics = evaluate_agent(agent, eval_env, num_eval_steps=500)
    print("\n=== Evaluation Results on Held-out Sequence ===")
    print(f"DQN Accuracy:             {metrics['dqn_accuracy']*100:.2f}%")
    print(f"Repeat Baseline Accuracy: {metrics['repeat_baseline_accuracy']*100:.2f}%")
    print(f"Random Baseline Accuracy: {metrics['random_baseline_accuracy']*100:.2f}%")

    if save_path:
        agent.save(save_path)
        print(f"Model saved to {save_path}")

    return agent, metrics


if __name__ == "__main__":
    train_hop_predictor()
