"""
RF Noise and Non-Threat Interferer Generator.
Simulates non-hopping background RF emissions, random transients, and erratic decoys
for the Threat Classifier to distinguish from genuine threats.
"""

import math
import random
from typing import Optional, Tuple
from configs.config import ChannelConfig
from sim.channel import RFChannel


class NoiseGenerator:
    """
    Generates background RF noise, transient pulses, and erratic non-threat transmissions.
    """

    def __init__(self, config: ChannelConfig, seed: int = 1337):
        self.config = config
        self.channel_model = RFChannel(config)
        self.rng = random.Random(seed)
        self.current_time = 0.0

    def generate_transient_noise(self, current_time: float) -> Tuple[float, float, float, float, int]:
        """
        Generates a transient RF burst (e.g., motor ignition, sporadic Wi-Fi/Bluetooth packet,
        or atmospheric pulse).
        """
        channel_idx = self.rng.randint(0, self.config.num_channels - 1)
        freq_mhz = self.channel_model.channel_idx_to_frequency(channel_idx)

        # Noise bursts have erratic, generally low RSSI near or slightly above noise floor
        rssi = self.config.noise_floor_dbm + self.rng.uniform(3.0, 18.0)
        # Low intensity / SNR
        intensity = self.rng.uniform(0.05, 0.35)

        return current_time, freq_mhz, rssi, intensity, channel_idx

    def generate_erratic_decoy(self, current_time: float) -> Tuple[float, float, float, float, int, float]:
        """
        Generates an erratic decoy signal that hops randomly without distance consistency
        (random erratic virtual distances).
        """
        channel_idx = self.rng.randint(0, self.config.num_channels - 1)
        freq_mhz = self.channel_model.channel_idx_to_frequency(channel_idx)

        # Erratic random distance between 20m and 800m
        erratic_distance = self.rng.uniform(20.0, 800.0)
        tx_power = self.rng.uniform(10.0, 30.0)

        rssi, intensity = self.channel_model.compute_rssi(tx_power, erratic_distance, freq_mhz)
        return current_time, freq_mhz, rssi, intensity, channel_idx, erratic_distance

    def reset(self) -> None:
        self.current_time = 0.0
