"""
Smart Scan Web Server & Real-time API.
Provides REST and Server-Sent Events (SSE) stream endpoints for the Smart Scan
Electronic Warfare tactical radar frontend. Uses Python's built-in http.server (no external web framework required).
"""

import json
import time
import os
import sys
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List

from configs.config import SystemConfig
from sim.signal_source import SimulatedRFEnvironment
from sim.emitter import FrequencyHoppingEmitter
from hardware.sdr_interface import SDRSignalSource
from models.hop_predictor.agent import DQNHopPredictor
from models.distance_estimator.model import DistanceEstimator
from models.threat_classifier.rules import RuleBasedThreatClassifier, ThreatLevel
from tracker.threat_tracker import ThreatTracker


class SmartScanState:
    """Manages the backend state, models, and real-time scanning loop."""

    def __init__(self):
        self.config = SystemConfig()
        self.is_running = True
        self.scan_speed = 1.0  # Speed multiplier
        self.source_mode = "sim"
        self.lock = threading.Lock()

        # Load models
        self.hop_agent = DQNHopPredictor(channel_config=self.config.channel, config=self.config.hop_predictor)
        hop_ckpt = "models/checkpoints/hop_dqn.pt"
        if os.path.exists(hop_ckpt):
            self.hop_agent.load(hop_ckpt)

        self.dist_estimator = DistanceEstimator(config=self.config.distance_estimator, channel_config=self.config.channel)
        dist_ckpt = "models/checkpoints/distance_estimator.pt"
        if os.path.exists(dist_ckpt):
            self.dist_estimator.load(dist_ckpt)

        self.classifier = RuleBasedThreatClassifier(
            min_hops_threat=self.config.tracker.min_hops_for_classification,
            max_dist_variance_m2=self.config.tracker.radius_variance_threshold_m2
        )

        self.signal_source = SimulatedRFEnvironment(config=self.config.channel)
        self.tracker = ThreatTracker(
            channel_config=self.config.channel,
            tracker_config=self.config.tracker,
            hop_predictor=self.hop_agent,
            distance_estimator=self.dist_estimator,
            threat_classifier=self.classifier
        )

        self.spectrogram_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        self.confirmed_threat_ids = set()
        self.total_readings = 0
        self.start_time = time.time()

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.worker_thread.start()

    def _scan_loop(self):
        """Continuous background signal ingestion and tracking loop."""
        while True:
            if not self.is_running:
                time.sleep(0.1)
                continue

            with self.lock:
                reading = self.signal_source.get_next_reading()
                track = self.tracker.update(reading)
                self.total_readings += 1

                # Record waterfall spectrogram data
                self.spectrogram_history.append({
                    "timestamp": reading.timestamp,
                    "frequency": reading.frequency,
                    "channel": reading.channel_idx,
                    "rssi": reading.rssi,
                    "intensity": reading.intensity,
                    "track_id": track.track_id,
                    "is_threat": (track.threat_level == ThreatLevel.THREAT)
                })
                if len(self.spectrogram_history) > 120:
                    self.spectrogram_history.pop(0)

                # Check for newly confirmed threat
                if track.threat_level == ThreatLevel.THREAT and track.track_id not in self.confirmed_threat_ids:
                    self.confirmed_threat_ids.add(track.track_id)
                    self.alerts.append({
                        "id": len(self.alerts) + 1,
                        "timestamp": round(reading.timestamp, 2),
                        "track_id": track.track_id,
                        "type": "THREAT_CONFIRMED",
                        "radius": round(track.estimated_radius_m, 1),
                        "next_channel": track.predicted_next_channel,
                        "next_freq": round(track.predicted_next_freq_mhz or 0.0, 1),
                        "confidence": round(track.threat_confidence * 100, 1),
                        "message": f"Confirmed hostile hopping emitter {track.track_id} at {round(track.estimated_radius_m, 1)}m radius"
                    })
                    if len(self.alerts) > 20:
                        self.alerts.pop(0)

                # Prune old tracks
                self.tracker.prune_idle_tracks(current_time=reading.timestamp, timeout_s=3.5)

            # Sleep interval outside lock
            sleep_time = max(0.01, 0.04 / max(0.1, self.scan_speed))
            time.sleep(sleep_time)

    def get_snapshot(self) -> Dict[str, Any]:
        """Return full current system state."""
        with self.lock:
            tracks_data = []
            for t in self.tracker.get_active_tracks():
                tracks_data.append({
                    "track_id": t.track_id,
                    "threat_level": t.threat_level,
                    "threat_confidence": round(t.threat_confidence * 100, 1),
                    "estimated_radius_m": round(t.estimated_radius_m, 1),
                    "radius_uncertainty_m": round(t.radius_uncertainty_m, 1),
                    "hop_count": t.hop_count,
                    "last_frequency": round(t.hop_frequencies[-1], 2) if t.hop_frequencies else 0.0,
                    "last_channel": t.hop_channels[-1] if t.hop_channels else 0,
                    "predicted_next_channel": t.predicted_next_channel,
                    "predicted_next_freq_mhz": round(t.predicted_next_freq_mhz, 2) if t.predicted_next_freq_mhz else None,
                    "prediction_confidence": round(t.prediction_confidence * 100, 1),
                    "prediction_hits": t.prediction_hits,
                    "reasoning": t.threat_assessment.reason if t.threat_assessment else "",
                    "recent_rssi": [round(r, 1) for r in t.rssi_history[-15:]],
                    "recent_distances": [round(d, 1) for d in t.distance_history[-15:]]
                })

            summary = self.tracker.get_summary()

            return {
                "system": {
                    "is_running": self.is_running,
                    "source_mode": self.source_mode,
                    "scan_speed": self.scan_speed,
                    "uptime_seconds": round(time.time() - self.start_time, 1),
                    "total_readings": self.total_readings,
                    "num_channels": self.config.channel.num_channels,
                    "min_freq_mhz": self.config.channel.min_freq_mhz,
                    "max_freq_mhz": self.config.channel.max_freq_mhz
                },
                "summary": summary,
                "tracks": tracks_data,
                "spectrogram": self.spectrogram_history[-40:],
                "alerts": self.alerts[-10:]
            }

    def set_running(self, running: bool):
        with self.lock:
            self.is_running = running

    def set_speed(self, speed: float):
        with self.lock:
            self.scan_speed = max(0.1, min(10.0, speed))

    def reset_system(self):
        with self.lock:
            self.signal_source.reset()
            self.tracker = ThreatTracker(
                channel_config=self.config.channel,
                tracker_config=self.config.tracker,
                hop_predictor=self.hop_agent,
                distance_estimator=self.dist_estimator,
                threat_classifier=self.classifier
            )
            self.spectrogram_history.clear()
            self.alerts.clear()
            self.confirmed_threat_ids.clear()
            self.total_readings = 0
            self.start_time = time.time()


# Global server state instance
STATE = SmartScanState()


class SmartScanRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for Smart Scan Web Dashboard."""

    def __init__(self, *args, **kwargs):
        # Serve static files from web/ directory
        web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/api/snapshot":
            data = STATE.get_snapshot()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif parsed.path == "/api/stream":
            # Server-Sent Events (SSE) stream
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                while True:
                    snapshot = STATE.get_snapshot()
                    payload = f"data: {json.dumps(snapshot)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # Default fallback to static file handler (index.html)
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len).decode('utf-8') if content_len > 0 else "{}"
        try:
            params = json.loads(body)
        except Exception:
            params = {}

        if parsed.path == "/api/control/toggle":
            STATE.set_running(params.get("running", not STATE.is_running))
            res = {"status": "ok", "is_running": STATE.is_running}
        elif parsed.path == "/api/control/speed":
            STATE.set_speed(float(params.get("speed", 1.0)))
            res = {"status": "ok", "scan_speed": STATE.scan_speed}
        elif parsed.path == "/api/control/reset":
            STATE.reset_system()
            res = {"status": "ok", "message": "System reset successfully"}
        else:
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(res).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence routine static polling logs
        pass


def start_server(port: int = 8080):
    server_address = ('127.0.0.1', port)
    httpd = ThreadingHTTPServer(server_address, SmartScanRequestHandler)
    print(f"\n========================================================")
    print(f"  SMART SCAN TACTICAL WEB DASHBOARD RUNNING")
    print(f"  URL: http://localhost:{port}")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Smart Scan web server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_server(port)
