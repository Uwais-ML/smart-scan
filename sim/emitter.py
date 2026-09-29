"""
Frequency-Hopping Emitter Simulator.
Models frequency-hopping spread spectrum (FHSS) radios with configurable hopping sequences,
dwell times, transmit powers, and spatial trajectories.
"""

import math
import random
from typing import List, Optional, Tuple
from configs.config import ChannelConfig
from sim.channel import RFChannel


class FrequencyHoppingEmitter:
    """
    Simulates a single frequency-hopping RF emitter.
    Uses a seeded PRNG hopping algorithm (e.g. Linear Congruential Generator or Permutation Table)
    to mimic realistic FHSS radio patterns.
    """

    def __init__(
        self,
        emitter_id: str,
        channel_config: ChannelConfig,
        hop_dwell_time: float = 0.02,    # 20ms per hop (50 hops/sec)
        tx_power_dbm: float = 20.0,       # 100mW transmit power
        initial_distance_m: float = 100.0,
        radial_velocity_mps: float = 0.0, # meters per second (radial drift)
        seed: int = 42,
        is_threat: bool = True,
        pattern_type: str = "lcg"         # "lcg", "markov", "costas", "random"
    ):
        self.emitter_id = emitter_id
        self.config = channel_config
        self.channel_model = RFChannel(channel_config)
        self.hop_dwell_time = hop_dwell_time
        self.tx_power_dbm = tx_power_dbm
        self.distance_m = initial_distance_m
        self.initial_distance_m = initial_distance_m
        self.radial_velocity_mps = radial_velocity_mps
        self.seed = seed
        self.is_threat = is_threat
        self.pattern_type = pattern_type

        self.rng = random.Random(seed)
        self.current_time = 0.0
        self.current_channel_idx = 0
        self._init_pattern()

    def _init_pattern(self) -> None:
        """Initialize the frequency hopping sequence generator."""
        N = self.config.num_channels
        if self.pattern_type == "lcg":
            # Linear Congruential Sequence: X_{n+1} = (a * X_n + c) mod M
            # Standard PRNG hopping sequence used in FHSS tactical radios
            self.lcg_a = 17
            self.lcg_c = 13
            self.lcg_state = self.rng.randint(0, N - 1)
            self.current_channel_idx = self.lcg_state
        elif self.pattern_type == "markov":
            # Transition matrix with structured hop deltas
            self.hop_deltas = [3, -5, 7, -11, 13, -17, 19, -23]
            self.current_channel_idx = self.rng.randint(0, N - 1)
        elif self.pattern_type == "costas":
            # Costas frequency hopping array permutation
            self.perm = list(range(N))
            self.rng.shuffle(self.perm)
            self.costas_idx = 0
            self.current_channel_idx = self.perm[0]
        else: # "random"
            self.current_channel_idx = self.rng.randint(0, N - 1)

    def next_hop(self, elapsed_time: Optional[float] = None) -> Tuple[float, float, float, float, int]:
        """
        Advance the emitter state to the next hop.

        Returns:
            (timestamp, frequency_mhz, rssi_dbm, intensity, channel_idx)
        """
        dt = self.hop_dwell_time if elapsed_time is None else elapsed_time
        self.current_time += dt

        # Update physical distance if moving
        self.distance_m = max(1.0, self.distance_m + self.radial_velocity_mps * dt)

        # Advance frequency hop sequence
        N = self.config.num_channels
        if self.pattern_type == "lcg":
            self.lcg_state = (self.lcg_a * self.lcg_state + self.lcg_c) % N
            self.current_channel_idx = self.lcg_state
        elif self.pattern_type == "markov":
            delta = self.rng.choice(self.hop_deltas)
            self.current_channel_idx = (self.current_channel_idx + delta) % N
        elif self.pattern_type == "costas":
            self.costas_idx = (self.costas_idx + 1) % len(self.perm)
            self.current_channel_idx = self.perm[self.costas_idx]
        else:
            self.current_channel_idx = self.rng.randint(0, N - 1)

        freq_mhz = self.channel_model.channel_idx_to_frequency(self.current_channel_idx)
        rssi, intensity = self.channel_model.compute_rssi(
            self.tx_power_dbm,
            self.distance_m,
            freq_mhz
        )

        return self.current_time, freq_mhz, rssi, intensity, self.current_channel_idx

    def reset(self) -> None:
        """Reset emitter state."""
        self.rng = random.Random(self.seed)
        self.current_time = 0.0
        self.distance_m = self.initial_distance_m
        self._init_pattern()
