# Smart Scan — Implementation Plan
### For the build agent (Claude Code / equivalent)

**Project:** Frequency-hopping radio detection system for electronic warfare.
**Goal of this doc:** a phase-by-phase build order an autonomous coding agent can execute with minimal ambiguity, including what to fake/simulate until hardware decisions land.

---

## 0. System Summary (read this first)

Three ML components work together:

1. **Hop Predictor** — a Deep Q-Network (DQN) that predicts which frequency a frequency-hopping transmitter will jump to next, so the scanner can pre-tune instead of sweeping sequentially.
2. **Distance Estimator** — a deep learning regression model that takes RSSI + frequency + signal intensity and outputs a distance estimate (radius).
3. **Threat Classifier** — a lightweight decision model that separates noise from real threats, using the rule: **repeated hopping detected at roughly the same distance radius over time = threat**.

These three feed a **Threat Tracker** that maintains state per detected emitter (hop history, distance history, threat status) and exposes results to a UI/dashboard.

**Hardware is not yet decided.** This plan assumes a **simulation-first** approach: build and validate the entire ML + tracking pipeline against a synthetic RF signal simulator, with a clean interface boundary so real hardware (SDR — e.g. HackRF, USRP, LimeSDR) can be swapped in later without touching the ML code.

---

## 1. Repo Structure

```
smart-scan/
├── sim/                    # synthetic signal generator (stands in for hardware)
│   ├── emitter.py          # simulates a frequency-hopping transmitter
│   ├── noise.py            # background/interference generator
│   └── channel.py          # RSSI falloff model, multipath option
├── hardware/                # real SDR interface (built later, same API as sim/)
│   └── sdr_interface.py
├── models/
│   ├── hop_predictor/       # DQN
│   │   ├── env.py           # Gym-style environment wrapping the hop sequence
│   │   ├── agent.py         # DQN agent (network + replay buffer + training loop)
│   │   └── train.py
│   ├── distance_estimator/
│   │   ├── model.py         # regression network
│   │   ├── dataset.py
│   │   └── train.py
│   └── threat_classifier/
│       ├── rules.py         # rule-based first pass (radius consistency check)
│       └── model.py         # optional learned classifier, phase 2
├── tracker/
│   └── threat_tracker.py    # stateful per-emitter tracking, ties all 3 models together
├── pipeline/
│   └── run_live.py          # main loop: signal in -> predictions -> tracker -> output
├── dashboard/                # optional UI (phase 4)
├── tests/
├── configs/
│   └── default.yaml
└── README.md
```

**Build this structure first, empty files with docstrings stating intent, before writing logic.** This keeps every module's contract explicit before implementation.

---

## 2. Phase Plan

### Phase 1 — Simulator (do this before any ML)

You cannot train or test anything without data, and hardware isn't chosen yet. Build `sim/` first.

**`sim/emitter.py`**
- Simulates one frequency-hopping transmitter: a hop sequence (can be pseudo-random, following a seeded PRNG to mimic real FHSS patterns), a hop dwell time, and a transmit power.
- Output per timestep: `(timestamp, frequency, tx_power)`

**`sim/channel.py`**
- Converts `tx_power` + `distance` (ground truth, since this is simulated) into a realistic `RSSI` using a standard path-loss model (e.g. log-distance path loss). Add configurable noise/multipath jitter.
- This gives you **ground-truth distance** to train and evaluate the Distance Estimator against — critical, since you won't have ground truth once real hardware is in the loop.

**`sim/noise.py`**
- Generates background noise / decoy signals that do NOT follow a hopping pattern, and some that hop randomly without distance consistency (non-threats) — this is what the Threat Classifier needs to learn to reject.

**Deliverable:** running `sim/` produces a stream of `(timestamp, frequency, rssi, intensity)` tuples for multiple simulated emitters (some threats, some noise), with ground-truth labels stored alongside for training/eval — labels are never fed to the models, only used to score them.

---

### Phase 2 — Hop Predictor (DQN)

**Framing:** state = recent hop history (last N frequencies + timing), action = predicted next frequency (discretized frequency bins), reward = +1 if prediction matches actual next hop, small negative reward otherwise (encourages confident correct predictions over hedging).

**`models/hop_predictor/env.py`**
- Gym-style `step()`/`reset()` wrapping the simulator's hop sequence so the DQN can be trained with standard RL libraries (Stable-Baselines3 recommended over hand-rolling DQN — faster to a working baseline).

**`models/hop_predictor/agent.py`**
- Standard DQN: Q-network (small MLP is enough — state space is a short hop-history window, not raw signal), replay buffer, target network, epsilon-greedy exploration.

**Success criterion for this phase:** prediction accuracy on next-hop frequency meaningfully beats a naive baseline (most-recent-frequency-repeats, or uniform-random-guess over the frequency band) on held-out simulated hop sequences with different seeds than training.

**Known hard part to flag, not solve yet:** real FHSS patterns may be pseudo-random with a cryptographic-ish seed (military radios often use this specifically to be unpredictable) — if the actual target hardware uses this, the DQN's job becomes "detect statistical bias/timing leakage" rather than "learn the sequence," which is a much harder and more research-y problem. Build against a learnable synthetic pattern first; flag this risk to the team early rather than discovering it late.

---

### Phase 3 — Distance Estimator

**Framing:** supervised regression. Input features: `RSSI`, `frequency`, `signal intensity` (and derived features: RSSI variance over a short window, signal-to-noise ratio if available). Output: distance estimate (single scalar) or a distance *bucket/radius range* if precise regression proves noisy.

**`models/distance_estimator/model.py`**
- Start with a simple feedforward network (2-3 hidden layers) — this is a small-feature-count regression problem, not one that needs a large model. Only escalate complexity if a simple model underperforms.

**`models/distance_estimator/train.py`**
- Train on simulator-generated `(rssi, frequency, intensity) -> distance` pairs across many simulated distances/positions.
- **Report error as a radius (± meters), not just MSE** — the threat rule downstream needs an interpretable radius consistency check, so the model's practical output should already be in those terms.

**Success criterion:** mean absolute error on held-out simulated positions within a tolerance the team defines as "good enough for radius-consistency threat detection" (this tolerance directly determines how tight/loose the threat rule's "roughly the same radius" threshold can be — decide the tolerance here, then reuse it in Phase 4, don't set it twice independently).

---

### Phase 4 — Threat Classifier + Tracker

**`models/threat_classifier/rules.py`** (build this before any learned model)
- Rule-based first pass: for each tracked emitter, check whether its estimated distance radius stays within a tolerance band across repeated hop-detections over a time window. If yes → threat. If distance is erratic/inconsistent → noise.
- **This rule-based version should be your Phase 4 deliverable for the hackathon demo** — it's explainable (important for judges), doesn't need training data you don't have yet, and directly implements the stated threat rule.

**`tracker/threat_tracker.py`**
- Maintains a dict of `emitter_id -> {hop_history, distance_history, first_seen, last_seen, threat_status}`.
- `emitter_id` assignment: since you can't rely on a literal ID from the RF signal, cluster detections by hop-pattern similarity + distance consistency (a simple approach: if a new detection's predicted-next-hop matches an existing tracked emitter's DQN prediction and distance is consistent, associate it with that emitter; otherwise spawn a new tracked entity).
- Runs the rule-based threat check on each update.

**`models/threat_classifier/model.py`** — **phase 2 stretch goal, not MVP.** Only build a learned classifier (e.g. small gradient-boosted tree or MLP on tracked-history features) once the rule-based version works end-to-end and you have logged real tracking sessions to train on. Don't build this before the rule-based version has a working demo — it's the one component genuinely optional for a working prototype.

---

### Phase 5 — Pipeline Integration

**`pipeline/run_live.py`**
- Wires: signal source (sim/ initially, hardware/ later, same interface) → Hop Predictor (running continuously, feeding predictions to help pre-tune the "scan") → Distance Estimator (runs per detected hop) → Threat Tracker (ingests both, maintains state, outputs threat list).
- This is the file to actually run for a live demo.

**Interface contract to lock early:** both `sim/` and the future `hardware/sdr_interface.py` must expose the *identical* function signature (e.g. `get_next_reading() -> (timestamp, frequency, rssi, intensity)`), so swapping from simulated to real hardware later is a one-line change in `run_live.py`, not a rewrite. Define this interface in Phase 1, before writing the simulator's internals.

---

### Phase 6 — Dashboard (optional, time-permitting)

A simple live view: list of tracked emitters, their current distance estimate, threat status (color-coded), and hop-prediction confidence. Not required for a working system — build only after Phases 1-5 are solid, since judges score the working pipeline over the UI.

---

## 3. What's Explicitly Deferred (don't build yet)

- Real hardware integration — blocked on hardware choice; `hardware/sdr_interface.py` stays a stub matching the sim interface until decided.
- Learned threat classifier — rule-based version is the MVP; only revisit if time remains after Phase 5 works end-to-end.
- Multi-emitter disambiguation beyond simple clustering — fine for a demo with a handful of simulated emitters; a crowded real RF environment is a harder tracking problem to defer past the hackathon.

## 4. Immediate Next Action for the Agent

Start at Phase 1. Do not start DQN or distance-model code before the simulator produces a working, ground-truth-labeled data stream — every downstream phase depends on it for both training and evaluation.
