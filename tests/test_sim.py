"""Unit tests for RF Channel, Emitters, Noise, and Simulated RF Environment."""

import unittest
from configs.config import ChannelConfig
from sim.channel import RFChannel
from sim.emitter import FrequencyHoppingEmitter
from sim.noise import NoiseGenerator
from sim.signal_source import SimulatedRFEnvironment
from sim.types import SignalReading


class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.config = ChannelConfig(num_channels=32, min_freq_mhz=400.0, max_freq_mhz=500.0)
        self.channel = RFChannel(self.config)

    def test_channel_path_loss_monotonicity(self):
        """RSSI should be significantly lower at 500m than at 10m on average."""
        rssi_close, _ = self.channel.compute_rssi(tx_power_dbm=20.0, distance_m=10.0, freq_mhz=450.0, add_shadowing=False, add_fast_fading=False)
        rssi_far, _ = self.channel.compute_rssi(tx_power_dbm=20.0, distance_m=500.0, freq_mhz=450.0, add_shadowing=False, add_fast_fading=False)
        self.assertGreater(rssi_close, rssi_far)

    def test_frequency_channel_mapping(self):
        """Test round-trip mapping between frequencies and channel indices."""
        ch_idx = 10
        freq = self.channel.channel_idx_to_frequency(ch_idx)
        recovered_idx = self.channel.frequency_to_channel_idx(freq)
        self.assertEqual(ch_idx, recovered_idx)

    def test_emitter_hopping(self):
        """Emitter should produce consecutive hops within frequency band."""
        emitter = FrequencyHoppingEmitter(
            emitter_id="TEST_EMITTER",
            channel_config=self.config,
            hop_dwell_time=0.02,
            initial_distance_m=100.0,
            seed=42
        )
        hops = []
        for _ in range(20):
            t, freq, rssi, intensity, ch_idx = emitter.next_hop()
            self.assertGreaterEqual(freq, self.config.min_freq_mhz)
            self.assertLessEqual(freq, self.config.max_freq_mhz)
            self.assertGreaterEqual(ch_idx, 0)
            self.assertLess(ch_idx, self.config.num_channels)
            hops.append(ch_idx)
        self.assertEqual(len(hops), 20)

    def test_simulated_rf_environment_stream(self):
        """SimulatedRFEnvironment should output valid SignalReading objects chronologically."""
        env = SimulatedRFEnvironment(config=self.config, seed=123)
        readings = []
        last_t = -1.0
        for _ in range(30):
            reading = env.get_next_reading()
            self.assertIsInstance(reading, SignalReading)
            self.assertGreaterEqual(reading.timestamp, last_t)
            last_t = reading.timestamp
            readings.append(reading)
        self.assertEqual(len(readings), 30)


if __name__ == "__main__":
    unittest.main()
