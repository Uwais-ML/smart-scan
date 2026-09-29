"""
RF Channel Model.
Implements log-distance path-loss with log-normal shadowing and multipath fast fading.
"""

import math
import random
from typing import Tuple
from configs.config import ChannelConfig


class RFChannel:
    """
    Simulates radio frequency propagation channel physics:
    - Free space / log-distance path loss: PL(d) = PL(d0) + 10 * n * log10(d / d0)
    - Shadow fading: X_sigma ~ Gaussian(0, sigma^2)
    - Fast fading (Rayleigh/multipath)
    - Thermal noise floor
    """

    def __init__(self, config: ChannelConfig = None):
        self.config = config or ChannelConfig()

    def compute_rssi(
        self,
        tx_power_dbm: float,
        distance_m: float,
        freq_mhz: float,
        add_shadowing: bool = True,
        add_fast_fading: bool = True
    ) -> Tuple[float, float]:
        """
        Compute RSSI (dBm) and Signal Intensity (SNR metric) for a given distance and frequency.

        Returns:
            (rssi_dbm, intensity)
        """
        # Clamp distance to minimum reference distance
        d = max(distance_m, self.config.ref_distance_m)

        # Log-distance path loss relative to reference distance (1m)
        path_loss_db = 10.0 * self.config.path_loss_exponent * math.log10(d / self.config.ref_distance_m)

        # Frequency-dependent attenuation adjustment (higher frequency suffers slightly more loss)
        center_freq = (self.config.min_freq_mhz + self.config.max_freq_mhz) / 2.0
        freq_factor = 20.0 * math.log10(freq_mhz / center_freq) if freq_mhz > 0 else 0.0

        # Log-normal shadowing (large-scale fading)
        shadowing_db = random.gauss(0.0, self.config.shadowing_std_db) if add_shadowing else 0.0

        # Small-scale multipath fast fading (Rayleigh amplitude corresponds to exponential power)
        multipath_db = 0.0
        if add_fast_fading:
            # Random rayleigh power sample in dB
            fade = -math.log(max(random.random(), 1e-6))
            multipath_db = 10.0 * math.log10(max(fade, 0.05))

        # Received power at 1m is (tx_power + ref_rssi_at_1m)
        rx_power_dbm = tx_power_dbm + self.config.ref_rssi_at_1m_dbm - path_loss_db - freq_factor + shadowing_db + multipath_db

        # Received power cannot drop arbitrarily below noise floor
        rx_power_dbm = max(rx_power_dbm, self.config.noise_floor_dbm - 5.0)

        # Compute signal intensity (normalized SNR relative to noise floor)
        snr_db = max(0.0, rx_power_dbm - self.config.noise_floor_dbm)
        # Sigmoidal mapping to [0.0, 1.0]
        intensity = 1.0 / (1.0 + math.exp(-0.1 * (snr_db - 10.0)))

        return rx_power_dbm, intensity

    def frequency_to_channel_idx(self, freq_mhz: float) -> int:
        """Map a frequency in MHz to a discrete channel index."""
        span = self.config.max_freq_mhz - self.config.min_freq_mhz
        step = span / self.config.num_channels
        idx = int((freq_mhz - self.config.min_freq_mhz) / step)
        return max(0, min(self.config.num_channels - 1, idx))

    def channel_idx_to_frequency(self, idx: int) -> float:
        """Map a discrete channel index to center frequency in MHz."""
        span = self.config.max_freq_mhz - self.config.min_freq_mhz
        step = span / self.config.num_channels
        return self.config.min_freq_mhz + (idx + 0.5) * step
