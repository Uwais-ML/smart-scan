"""
Live Smart Scan Pipeline.
Wires signal source (simulated or real SDR) -> DQN Hop Predictor -> Distance Estimator -> Threat Tracker -> Live Tactical Display.
"""

import argparse
import os
import sys
import time
from typing import Optional

import torch
from configs.config import SystemConfig
from sim.types import SignalSource
from sim.signal_source import SimulatedRFEnvironment
from hardware.sdr_interface import SDRSignalSource
from models.hop_predictor.agent import DQNHopPredictor
from models.hop_predictor.train import train_hop_predictor
from models.distance_estimator.model import DistanceEstimator
from models.distance_estimator.train import train_distance_estimator
from models.threat_classifier.rules import RuleBasedThreatClassifier, ThreatLevel
from tracker.threat_tracker import ThreatTracker
from dashboard.terminal_ui import TacticalDashboard, Colors


def load_or_train_models(config: SystemConfig):
    """Load model checkpoints or trigger training if checkpoints do not exist."""
    hop_ckpt = "models/checkpoints/hop_dqn.pt"
    dist_ckpt = "models/checkpoints/distance_estimator.pt"

    # 1. DQN Hop Predictor
    hop_agent = DQNHopPredictor(channel_config=config.channel, config=config.hop_predictor)
    if os.path.exists(hop_ckpt):
        print(f"[SmartScan] Loading DQN Hop Predictor from {hop_ckpt}...")
        hop_agent.load(hop_ckpt)
    else:
        print("[SmartScan] Checkpoint not found. Training DQN Hop Predictor...")
        hop_agent, _ = train_hop_predictor(save_path=hop_ckpt)

    # 2. Distance Estimator
    dist_estimator = DistanceEstimator(config=config.distance_estimator, channel_config=config.channel)
    if os.path.exists(dist_ckpt):
        print(f"[SmartScan] Loading Distance Estimator from {dist_ckpt}...")
        dist_estimator.load(dist_ckpt)
    else:
        print("[SmartScan] Checkpoint not found. Training Distance Estimator...")
        dist_estimator, _ = train_distance_estimator(save_path=dist_ckpt)

    # 3. Threat Classifier
    classifier = RuleBasedThreatClassifier(
        min_hops_threat=config.tracker.min_hops_for_classification,
        max_dist_variance_m2=config.tracker.radius_variance_threshold_m2
    )

    return hop_agent, dist_estimator, classifier


def run_pipeline(
    source_type: str = "sim",
    total_steps: int = 100,
    render_interval: int = 10,
    delay_s: float = 0.0,
    config: Optional[SystemConfig] = None
):
    """
    Main Electronic Warfare real-time scanning loop.
    """
    cfg = config or SystemConfig()

    print(f"{Colors.BOLD}{Colors.CYAN}Initializing Smart Scan Electronic Warfare System...{Colors.RESET}")
    hop_agent, dist_estimator, classifier = load_or_train_models(cfg)

    # Initialize Signal Source (Hardware SDR or Simulator)
    if source_type.lower() == "sdr":
        print(f"[SmartScan] Initializing Hardware SDR Interface...")
        signal_source: SignalSource = SDRSignalSource(config=cfg.channel)
        source_label = "Real SDR / Hardware Energy Detector"
    else:
        print(f"[SmartScan] Initializing High-Fidelity RF Simulator...")
        signal_source: SignalSource = SimulatedRFEnvironment(config=cfg.channel)
        source_label = "Synthetic RF Environment"

    tracker = ThreatTracker(
        channel_config=cfg.channel,
        tracker_config=cfg.tracker,
        hop_predictor=hop_agent,
        distance_estimator=dist_estimator,
        threat_classifier=classifier
    )

    dashboard = TacticalDashboard()
    start_wall_time = time.time()
    confirmed_threat_ids = set()

    print(f"\n{Colors.GREEN}>> Smart Scan pipeline active. Ingesting RF stream...{Colors.RESET}\n")

    for step in range(1, total_steps + 1):
        # 1. Ingest next RF detection from signal provider
        reading = signal_source.get_next_reading()

        # 2. Update threat tracker (runs Distance Estimator, Association, DQN Hop Predictor & Threat Classifier)
        updated_track = tracker.update(reading)

        # Check for newly confirmed threats
        if updated_track.threat_level == ThreatLevel.THREAT and updated_track.track_id not in confirmed_threat_ids:
            confirmed_threat_ids.add(updated_track.track_id)
            alert = (
                f"{Colors.RED}{Colors.BOLD}[THREAT CONFIRMED]{Colors.RESET} "
                f"Track {updated_track.track_id} at {updated_track.estimated_radius_m:.1f}m radius "
                f"(Targeting: Ch {updated_track.predicted_next_channel:02d} @ {updated_track.predicted_next_freq_mhz:.1f}MHz)"
            )
            dashboard.log_alert(alert)

        # 3. Periodically prune stale tracks
        tracker.prune_idle_tracks(current_time=reading.timestamp, timeout_s=cfg.tracker.max_track_idle_seconds)

        # 4. Render live tactical dashboard
        if step % render_interval == 0 or step == total_steps:
            elapsed = time.time() - start_wall_time
            display_str = dashboard.render(tracker, source_label, step, elapsed)
            print(display_str)

        if delay_s > 0:
            time.sleep(delay_s)

    # Final summary report
    print(f"\n{Colors.BOLD}{Colors.GREEN}========================================================================================{Colors.RESET}")
    print(f"{Colors.BOLD} SMART SCAN MISSION SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}========================================================================================{Colors.RESET}")
    summary = tracker.get_summary()
    print(f"Total RF Events Processed: {total_steps}")
    print(f"Active Emitter Tracks:     {summary['total_active_tracks']}")
    print(f"Confirmed Hostile Threats: {summary['threat_count']} ({', '.join(summary['threat_ids']) if summary['threat_ids'] else 'None'})")
    print(f"Suspicious Emitters:       {summary['suspicious_count']}")
    print(f"Filtered Noise/Decoys:     {summary['noise_count']}")

    threat_tracks = tracker.get_threats()
    for t in threat_tracks:
        print(f"\n  • Threat {Colors.RED}{Colors.BOLD}{t.track_id}{Colors.RESET}:")
        print(f"    - Estimated Radius:     {t.estimated_radius_m:.1f} ± {t.radius_uncertainty_m:.1f} meters")
        print(f"    - Total Hops Logged:    {t.hop_count}")
        print(f"    - Next Hop Prediction:  Channel {t.predicted_next_channel:02d} ({t.predicted_next_freq_mhz:.2f} MHz) [Confidence: {t.prediction_confidence*100:.1f}%]")
        print(f"    - Threat Assessment:    {t.threat_assessment.reason if t.threat_assessment else ''}")


def main():
    parser = argparse.ArgumentParser(description="Smart Scan: Electronic Warfare Frequency-Hopping Detection System")
    parser.add_argument("--mode", choices=["sim", "sdr"], default="sim", help="Signal source provider: sim or sdr")
    parser.add_argument("--steps", type=int, default=150, help="Number of RF signal events to process")
    parser.add_argument("--interval", type=int, default=15, help="Dashboard refresh interval in steps")
    parser.add_argument("--delay", type=float, default=0.0, help="Artificial delay in seconds per step (for live demo)")
    args = parser.parse_args()

    run_pipeline(
        source_type=args.mode,
        total_steps=args.steps,
        render_interval=args.interval,
        delay_s=args.delay
    )


if __name__ == "__main__":
    main()
