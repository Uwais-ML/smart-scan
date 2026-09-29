"""
Training script for the Distance Estimator regression model.
Trains on RF propagation features and reports error in interpretable radius metrics (± meters).
"""

import math
import os
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.config import ChannelConfig, DistanceEstimatorConfig
from models.distance_estimator.model import DistanceEstimator, DistanceRegressionNet
from models.distance_estimator.dataset import generate_rf_distance_data, RFDistanceDataset


def evaluate_distance_estimator(
    model: DistanceRegressionNet,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates regression model and reports radius metrics in meters.
    """
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    within_15m_count = 0
    within_30m_count = 0
    total_samples = 0

    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)

            preds = model(features)
            errors = (preds - targets).abs()

            total_mae += errors.sum().item()
            total_mse += ((preds - targets) ** 2).sum().item()

            within_15m_count += (errors <= 15.0).sum().item()
            within_30m_count += (errors <= 30.0).sum().item()
            total_samples += targets.size(0)

    mae = total_mae / max(1, total_samples)
    rmse = math.sqrt(total_mse / max(1, total_samples))
    acc_15m = within_15m_count / max(1, total_samples)
    acc_30m = within_30m_count / max(1, total_samples)

    return {
        "mae_meters": mae,
        "rmse_meters": rmse,
        "accuracy_within_15m": acc_15m,
        "accuracy_within_30m": acc_30m,
        "total_samples": float(total_samples)
    }


def train_distance_estimator(
    epochs: int = 35,
    batch_size: int = 64,
    save_path: str = "models/checkpoints/distance_estimator.pt"
) -> Tuple[DistanceEstimator, Dict[str, float]]:
    """
    Train Distance Estimator and save best checkpoint.
    """
    channel_cfg = ChannelConfig()
    dist_cfg = DistanceEstimatorConfig(epochs=epochs, batch_size=batch_size)
    device = torch.device("cpu")

    # Generate synthetic training and validation splits
    print("Generating synthetic RF distance propagation dataset...")
    train_data = generate_rf_distance_data(num_samples=6000, channel_config=channel_cfg, seed=42)
    val_data = generate_rf_distance_data(num_samples=1500, channel_config=channel_cfg, seed=999)

    train_loader = DataLoader(RFDistanceDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RFDistanceDataset(val_data), batch_size=batch_size, shuffle=False)

    estimator = DistanceEstimator(config=dist_cfg, channel_config=channel_cfg, device=device)
    model = estimator.model

    # Huber Loss / Smooth L1 loss is robust against fading outliers
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print("=== Training Distance Estimator Regression Model ===")
    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)

        train_loss /= len(train_data)
        metrics = evaluate_distance_estimator(model, val_loader, device)
        scheduler.step(metrics["mae_meters"])

        if metrics["mae_meters"] < best_mae:
            best_mae = metrics["mae_meters"]
            estimator.save(save_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:6.2f} | "
                f"Val MAE: ±{metrics['mae_meters']:4.1f}m | "
                f"RMSE: {metrics['rmse_meters']:4.1f}m | "
                f"Within 30m: {metrics['accuracy_within_30m']*100:4.1f}%"
            )

    # Final evaluation using loaded best checkpoint
    estimator.load(save_path)
    final_metrics = evaluate_distance_estimator(estimator.model, val_loader, device)

    print("\n=== Final Distance Estimator Evaluation ===")
    print(f"Mean Absolute Error (MAE): ±{final_metrics['mae_meters']:.2f} meters")
    print(f"Root Mean Squared Error:   {final_metrics['rmse_meters']:.2f} meters")
    print(f"Accuracy within ±15m:      {final_metrics['accuracy_within_15m']*100:.2f}%")
    print(f"Accuracy within ±30m:      {final_metrics['accuracy_within_30m']*100:.2f}%")
    print(f"Saved checkpoint:          {save_path}")

    return estimator, final_metrics


if __name__ == "__main__":
    train_distance_estimator()
