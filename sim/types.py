"""
Data structures and interface definitions for RF Signal sources.
"""

from dataclasses import dataclass
from typing import Optional, Iterator
from abc import ABC, abstractmethod


@dataclass
class SignalReading:
    """
    Standard RF signal detection reading emitted by either
    the simulator or the physical SDR interface.
    """
    timestamp: float          # Time in seconds
    frequency: float          # Center frequency in MHz
    rssi: float               # Received Signal Strength Indicator in dBm
    intensity: float          # Normalized signal intensity / SNR metric (0.0 to 1.0+)
    channel_idx: int = 0      # Discretized channel index
    
    # Ground truth metadata (populated by simulator for training & evaluation, None for real SDR)
    ground_truth_emitter_id: Optional[str] = None
    ground_truth_distance: Optional[float] = None
    ground_truth_is_threat: Optional[bool] = None

    def as_tuple(self) -> tuple:
        """Returns standard 4-tuple (timestamp, frequency, rssi, intensity)."""
        return (self.timestamp, self.frequency, self.rssi, self.intensity)


class SignalSource(ABC):
    """
    Abstract Base Class for all signal input providers (Simulated or Real Hardware).
    """

    @abstractmethod
    def get_next_reading(self) -> SignalReading:
        """Fetch the next RF detection event."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the signal source state."""
        pass

    def stream(self, max_readings: Optional[int] = None) -> Iterator[SignalReading]:
        """Yield a stream of signal readings."""
        count = 0
        while max_readings is None or count < max_readings:
            reading = self.get_next_reading()
            if reading is None:
                break
            yield reading
            count += 1
