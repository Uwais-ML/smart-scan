"""
Simulated RF Environment.
Aggregates multiple simulated emitters, noise, and channel falloff into an event stream
that conforms to the SignalSource interface contract.
"""

import heapq
import random
from typing import List, Optional, Tuple
from configs.config import ChannelConfig
from sim.types import SignalReading, SignalSource
from sim.channel import RFChannel
from sim.emitter import FrequencyHoppingEmitter
from sim.noise import NoiseGenerator


class SimulatedRFEnvironment(SignalSource):
    """
    Full RF spectrum simulation environment.
    Maintains a priority queue (discrete event simulation) of signal detections
    arriving from various active emitters and background noise sources.
    """

    def __init__(
        self,
        config: ChannelConfig = None,
        emitters: Optional[List[FrequencyHoppingEmitter]] = None,
        noise_rate_hz: float = 15.0,
        decoy_rate_hz: float = 5.0,
        seed: int = 100
    ):
        self.config = config or ChannelConfig()
        self.channel_model = RFChannel(self.config)
        self.noise_rate_hz = noise_rate_hz
        self.decoy_rate_hz = decoy_rate_hz
        self.seed = seed
        self.rng = random.Random(seed)
        self.noise_gen = NoiseGenerator(self.config, seed=seed + 1)

        # Default emitters if none provided:
        # Threat 1: Tactical FHSS at 80m (LCG pattern)
        # Threat 2: Hostile FHSS at 250m (Markov pattern)
        # Civilian 1: Non-threat static transmitter at 450m (Costas pattern, low power)
        if emitters is None:
            self.emitters = [
                FrequencyHoppingEmitter(
                    emitter_id="THREAT_ALPHA",
                    channel_config=self.config,
                    hop_dwell_time=0.02,   # 50 hops/sec
                    tx_power_dbm=23.0,
                    initial_distance_m=85.0,
                    seed=101,
                    is_threat=True,
                    pattern_type="lcg"
                ),
                FrequencyHoppingEmitter(
                    emitter_id="THREAT_BRAVO",
                    channel_config=self.config,
                    hop_dwell_time=0.035,  # ~28 hops/sec
                    tx_power_dbm=26.0,
                    initial_distance_m=220.0,
                    seed=202,
                    is_threat=True,
                    pattern_type="markov"
                ),
                FrequencyHoppingEmitter(
                    emitter_id="CIVILIAN_COMMS",
                    channel_config=self.config,
                    hop_dwell_time=0.05,
                    tx_power_dbm=14.0,
                    initial_distance_m=500.0,
                    seed=303,
                    is_threat=False,
                    pattern_type="costas"
                )
            ]
        else:
            self.emitters = emitters

        # Event queue: list of tuples (timestamp, event_type, emitter_obj_or_none)
        self.event_queue: List[Tuple[float, str, Optional[FrequencyHoppingEmitter]]] = []
        self.current_time = 0.0
        self._init_queue()

    def _init_queue(self) -> None:
        """Initialize the event queue with the first hop of each emitter and noise."""
        self.event_queue.clear()
        self.current_time = 0.0

        for emitter in self.emitters:
            emitter.reset()
            first_time = emitter.hop_dwell_time * self.rng.uniform(0.1, 1.0)
            heapq.heappush(self.event_queue, (first_time, "emitter", emitter))

        # Schedule first noise event
        if self.noise_rate_hz > 0:
            first_noise_dt = self.rng.expovariate(self.noise_rate_hz)
            heapq.heappush(self.event_queue, (first_noise_dt, "noise", None))

        # Schedule first decoy event
        if self.decoy_rate_hz > 0:
            first_decoy_dt = self.rng.expovariate(self.decoy_rate_hz)
            heapq.heappush(self.event_queue, (first_decoy_dt, "decoy", None))

    def reset(self) -> None:
        """Reset the simulated RF environment."""
        self.rng = random.Random(self.seed)
        self.noise_gen.reset()
        self._init_queue()

    def get_next_reading(self) -> SignalReading:
        """
        Pulls the next signal detection event in chronological order.
        Conforms strictly to the SignalSource interface contract.
        """
        if not self.event_queue:
            self._init_queue()

        t, event_type, emitter = heapq.heappop(self.event_queue)
        self.current_time = t

        if event_type == "emitter" and emitter is not None:
            # Advance emitter and schedule next hop
            t_hop, freq_mhz, rssi, intensity, ch_idx = emitter.next_hop()
            next_t = t + emitter.hop_dwell_time
            heapq.heappush(self.event_queue, (next_t, "emitter", emitter))

            return SignalReading(
                timestamp=t,
                frequency=freq_mhz,
                rssi=rssi,
                intensity=intensity,
                channel_idx=ch_idx,
                ground_truth_emitter_id=emitter.emitter_id,
                ground_truth_distance=emitter.distance_m,
                ground_truth_is_threat=emitter.is_threat
            )

        elif event_type == "noise":
            # Generate transient noise reading
            _, freq_mhz, rssi, intensity, ch_idx = self.noise_gen.generate_transient_noise(t)
            next_noise_dt = self.rng.expovariate(self.noise_rate_hz)
            heapq.heappush(self.event_queue, (t + next_noise_dt, "noise", None))

            return SignalReading(
                timestamp=t,
                frequency=freq_mhz,
                rssi=rssi,
                intensity=intensity,
                channel_idx=ch_idx,
                ground_truth_emitter_id="NOISE_BURST",
                ground_truth_distance=None,
                ground_truth_is_threat=False
            )

        elif event_type == "decoy":
            # Generate erratic decoy reading
            _, freq_mhz, rssi, intensity, ch_idx, d = self.noise_gen.generate_erratic_decoy(t)
            next_decoy_dt = self.rng.expovariate(self.decoy_rate_hz)
            heapq.heappush(self.event_queue, (t + next_decoy_dt, "decoy", None))

            return SignalReading(
                timestamp=t,
                frequency=freq_mhz,
                rssi=rssi,
                intensity=intensity,
                channel_idx=ch_idx,
                ground_truth_emitter_id="DECOY_ERRATIC",
                ground_truth_distance=d,
                ground_truth_is_threat=False
            )

        # Fallback
        return SignalReading(
            timestamp=self.current_time,
            frequency=self.config.min_freq_mhz,
            rssi=self.config.noise_floor_dbm,
            intensity=0.0,
            channel_idx=0
        )
