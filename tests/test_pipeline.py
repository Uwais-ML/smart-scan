"""End-to-End integration tests for live Smart Scan pipeline."""

import unittest
from pipeline.run_live import run_pipeline


class TestPipeline(unittest.TestCase):

    def test_pipeline_sim_mode(self):
        """Test running end-to-end pipeline in simulation mode."""
        # Run 20 steps
        try:
            run_pipeline(source_type="sim", total_steps=20, render_interval=20, delay_s=0.0)
            success = True
        except Exception as e:
            success = False
            print(f"Pipeline sim mode error: {e}")
        self.assertTrue(success)

    def test_pipeline_sdr_mode(self):
        """Test running end-to-end pipeline in SDR hardware interface mode."""
        try:
            run_pipeline(source_type="sdr", total_steps=15, render_interval=15, delay_s=0.0)
            success = True
        except Exception as e:
            success = False
            print(f"Pipeline sdr mode error: {e}")
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
