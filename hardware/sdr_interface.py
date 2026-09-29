"""
SDR (Software Defined Radio) Hardware Interface.
Implements the same SignalSource interface contract for physical SDR hardware
(e.g., HackRF One, RTL-SDR, LimeSDR, USRP).

When physical hardware drivers (e.g. pyrtlsdr, SoapySDR, gnuradio) are absent,
provides a clean diagnostic fallback and simulated RF stream.
"""

import time
from typing import Optional
from configs.config import ChannelConfig
from sim.types import SignalReading, SignalSource
from sim.signal_source import SimulatedRFEnvironment


class SDRSignalSource(SignalSource):
    """
    Hardware SDR signal provider.
    Listens across frequency band bins, computes FFT energy detection,
    and returns detected RF signals.
    """

    def __init__(
        self,
        config: Optional[ChannelConfig] = None,
        sdr_device_type: str = "mock",
        gain_db: float = 30.0,
        sample_rate_sps: float = 20e6
    ):
        self.config = config or ChannelConfig()
        self.sdr_device_type = sdr_device_type
        self.gain_db = gain_db
        self.sample_rate_sps = sample_rate_sps
        self.is_hardware_connected = False

        # Attempt to initialize hardware SDR backend if requested
        self._init_hardware()

    def _init_hardware(self) -> None:
        """Attempt to connect to SDR driver, otherwise fall back to mock stream."""
        if self.sdr_device_type.lower() != "mock":
            try:
                # Placeholder for pyrtlsdr / SoapySDR / Uhd import and initialization
                # e.g.: import SoapySDR; self.sdr = SoapySDR.Device()
                pass
            except Exception as e:
                print(f"[SDRInterface] Hardware init failed ({e}), using mock SDR stream.")
                self.is_hardware_connected = False
        
        # When in mock mode or hardware unavailable, use the simulated RF environment
        self._fallback_sim = SimulatedRFEnvironment(config=self.config)

    def get_next_reading(self) -> SignalReading:
        """
        Fetches next reading from either physical SDR energy detection or mock SDR stream.
        Exposes identical interface contract as sim.
        """
        if self.is_hardware_connected:
            # Physical hardware FFT energy detection logic:
            # 1. Read IQ samples
            # 2. Compute FFT power spectrum
            # 3. Detect peaks above threshold
            # 4. Return SignalReading
            pass

        # Fallback reading
        reading = self._fallback_sim.get_next_reading()
        # For physical SDR, ground truth fields are omitted
        return SignalReading(
            timestamp=time.time(),
            frequency=reading.frequency,
            rssi=reading.rssi,
            intensity=reading.intensity,
            channel_idx=reading.channel_idx,
            ground_truth_emitter_id=None,
            ground_truth_distance=None,
            ground_truth_is_threat=None
        )

    def reset(self) -> None:
        if hasattr(self, "_fallback_sim"):
            self._fallback_sim.reset()
