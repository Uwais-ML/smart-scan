# Smart Scan — Electronic Warfare FHSS Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/ML-Deep%20Q--Network-green" />
  <img src="https://img.shields.io/badge/Domain-Electronic%20Warfare-red" />
  <img src="https://img.shields.io/badge/Tests-15%20Passing-brightgreen" />
</p>

> **Smart Scan** is an autonomous Electronic Warfare (EW) radio detection and tracking system designed to detect and track hostile Frequency-Hopping Spread Spectrum (FHSS) radios using deep reinforcement learning, neural distance regression, and rule-based EW threat classification — all without requiring physical SDR hardware.

---

## 🎯 What It Does

Instead of slowly sweeping across the full radio spectrum, Smart Scan uses three ML models working in tandem:

1. **Hop Predictor (DQN)** — A Deep Q-Network that *learns* the pseudo-random frequency hopping pattern of a transmitter and pre-tunes the receiver to the next channel *before* the transmission happens. Achieved **64.2% prediction accuracy** across 64 frequency channels (vs. 1.6% uniform random baseline and 0% repeat baseline).

2. **Distance Estimator** — A deep regression MLP that takes RF features `(RSSI, Frequency, Signal Intensity)` and outputs a physical distance estimate `(radius ± uncertainty in meters)`.

3. **Threat Classifier** — An explainable, rule-based electronic warfare classifier implementing the core tactical principle:
   > *Repeated frequency-hopping detected at roughly the same distance radius over time = **THREAT**.*

These feed into a **Threat Tracker** that maintains per-emitter state across hops, performs spatial-spectral track association, and exposes results to a **live web dashboard** and a **terminal tactical radar**.

---

## 🏗️ Architecture

```
SignalSource (Simulated RF / SDR Hardware)
         │
         ▼ (timestamp, freq_mhz, rssi_dbm, intensity)
┌────────────────────┐
│  Distance Estimator│──► Estimated Radius (meters)
│  (MLP Regression)  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐    Spatial-Spectral    ┌────────────────────┐
│  DQN Hop Predictor │◄── Track Association ──│   Threat Tracker   │
│  (Next-Hop RL)     │──► Next Freq Bin       │  (Stateful Tracks) │
└────────────────────┘                        └────────┬───────────┘
                                                       │
                                                       ▼
                                            ┌────────────────────┐
                                            │  Threat Classifier │
                                            │  (Radius Consisten.)│
                                            └────────┬───────────┘
                                                     │
                                                     ▼
                                            ┌────────────────────┐
                                            │  Web Dashboard     │
                                            │  Terminal Radar    │
                                            └────────────────────┘
```

---

## 📁 Project Structure

```
smart-scan/
├── configs/                        # Central system configuration
│   └── config.py
├── sim/                            # RF Physical Layer Simulator
│   ├── types.py                    # SignalSource & SignalReading interface contract
│   ├── channel.py                  # Log-distance path loss + shadowing + multipath
│   ├── emitter.py                  # FHSS transmitters: LCG / Markov / Costas patterns
│   ├── noise.py                    # Background noise & erratic decoy generator
│   └── signal_source.py            # Discrete-event RF environment stream
├── hardware/
│   └── sdr_interface.py            # SDR hardware stub (same API as sim, swap-in ready)
├── models/
│   ├── hop_predictor/
│   │   ├── env.py                  # Gym-style RL environment for FHSS hopping
│   │   ├── agent.py                # DQN: Q-network, replay buffer, target network
│   │   └── train.py                # Training + baseline benchmark script
│   ├── distance_estimator/
│   │   ├── model.py                # MLP regression: RSSI + Freq + Intensity → Distance
│   │   ├── dataset.py              # RF propagation synthetic dataset generator
│   │   └── train.py                # Training + radius MAE evaluation script
│   └── threat_classifier/
│       └── rules.py                # Rule-based EW: radius consistency threat classifier
├── tracker/
│   └── threat_tracker.py           # Stateful multi-emitter tracker & association engine
├── pipeline/
│   └── run_live.py                 # Live scanning pipeline (entry point)
├── web/
│   ├── app.py                      # HTTP server + real-time SSE API backend
│   └── index.html                  # Live web dashboard (canvas radar + spectrogram)
├── dashboard/
│   └── terminal_ui.py              # ANSI terminal tactical radar display
├── tests/                          # Unit & integration tests (15/15 passing)
├── DeepQ.py                        # Convenience entry point for DQN training
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/smart-scan.git
cd smart-scan
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Live Terminal Radar Pipeline
```bash
python -m pipeline.run_live --steps 150 --interval 15
```

### 3. Launch the Web Dashboard
```bash
python -m web.app 8080
# Open http://localhost:8080 in your browser
```

### 4. Train the Models
```bash
# Train DQN Hop Predictor
python -m models.hop_predictor.train

# Train Distance Estimator
python -m models.distance_estimator.train
```

### 5. Run All Tests
```bash
python -m unittest discover -s tests
# → Ran 15 tests in ~1s — OK
```

---

## 📊 Model Performance

| Model | Metric | Value |
|---|---|---|
| DQN Hop Predictor | Next-hop accuracy (64 channels) | **64.2%** |
| DQN Hop Predictor | Uniform random baseline | 1.6% |
| DQN Hop Predictor | Repeat-last baseline | 0.0% |
| Distance Estimator | Mean Absolute Error (MAE) | **±44.1 meters** |
| Distance Estimator | RMSE | 75.1 meters |
| Distance Estimator | Accuracy within ±30m | 61.7% |

---

## ⚙️ Configuration

All system parameters are defined in [`configs/config.py`](configs/config.py):

```python
ChannelConfig       # Frequency band, num channels, path-loss model
HopPredictorConfig  # DQN hyperparameters (history window, hidden dim, epsilon)
DistanceEstimatorConfig  # MLP architecture, learning rate, distance tolerance
ThreatTrackerConfig # Association thresholds, hop classification rules
```

---

## 🔌 Hardware SDR Integration

The simulator and real SDR hardware share an **identical interface contract** (`SignalSource` ABC):

```python
def get_next_reading() -> SignalReading:
    # Returns: (timestamp, frequency_mhz, rssi_dbm, intensity, channel_idx)
```

To swap in real hardware (HackRF, RTL-SDR, LimeSDR, USRP):
1. Implement `hardware/sdr_interface.py` using your SDR driver (`pyrtlsdr`, `SoapySDR`, `uhd`).
2. Change `--mode sim` to `--mode sdr` when running `pipeline/run_live.py`.
3. No other code changes needed.

---

## ⚠️ Known Limitations & Future Work

- **Real FHSS crypto patterns**: Military FHSS radios often use cryptographic pseudo-random sequences (e.g. SINCGARS). The DQN performs well on learnable synthetic patterns. Against truly unpredictable crypto-PRNG sequences, the model's role shifts from "learn the sequence" to "detect statistical timing leakage" — a significantly harder research problem flagged for future work.
- **Multi-emitter disambiguation**: Track association uses a simple cost function. Dense RF environments with many co-channel emitters at similar distances would require more sophisticated multi-hypothesis tracking.
- **Distance estimator accuracy**: The ±44m MAE reflects fundamental limits of RSSI-based ranging due to multipath fading. Sensor fusion with AoA (Angle of Arrival) or TDOA (Time Difference of Arrival) would substantially improve this.
- **Learned threat classifier**: A gradient-boosted or MLP classifier trained on logged tracking sessions is deferred as a Phase 2 stretch goal (see `models/threat_classifier/` for stub).

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
