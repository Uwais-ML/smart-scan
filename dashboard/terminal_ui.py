"""
Terminal UI & Electronic Warfare Tactical Dashboard.
Renders real-time visual displays of detected emitters, distance radius bands,
predicted next frequency hops, and confirmed EW threats using clean ANSI formatting.
"""

import os
import sys
import time
from typing import List, Dict, Any
from tracker.threat_tracker import EmitterTrack, ThreatTracker
from models.threat_classifier.rules import ThreatLevel


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"


class TacticalDashboard:
    """
    Electronic Warfare Tactical Command Dashboard for Smart Scan.
    """

    def __init__(self, title: str = "SMART SCAN — ELECTRONIC WARFARE TACTICAL RADAR"):
        self.title = title
        self.alert_history: List[str] = []

    def log_alert(self, alert_msg: str) -> None:
        """Add an EW alert to recent log."""
        self.alert_history.append(alert_msg)
        if len(self.alert_history) > 6:
            self.alert_history.pop(0)

    def format_threat_badge(self, level: str) -> str:
        """Format colorized badge for threat level."""
        if level == ThreatLevel.THREAT:
            return f"{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD} THREAT {Colors.RESET}"
        elif level == ThreatLevel.SUSPICIOUS:
            return f"{Colors.YELLOW}{Colors.BOLD}SUSPICIOUS{Colors.RESET}"
        elif level == ThreatLevel.TRANSIENT:
            return f"{Colors.CYAN}TRANSIENT{Colors.RESET}"
        else:
            return f"{Colors.DIM}NOISE/DECOY{Colors.RESET}"

    def render_ascii_radar(self, tracks: List[EmitterTrack], max_radius_m: float = 600.0) -> str:
        """
        Renders an ASCII concentric range ring display showing emitter distance radii.
        """
        lines = []
        rings = [100, 250, 400, 550]
        radar_width = 54

        lines.append(f"{Colors.BOLD}┌─ TACTICAL RADIUS SPECTRUM (Range: 0 - {max_radius_m:.0f}m) ─────────────────┐{Colors.RESET}")

        # Track markers plotted along range axis
        range_axis = ["─"] * radar_width
        # Mark range rings
        for ring in rings:
            pos = int((ring / max_radius_m) * (radar_width - 1))
            if 0 <= pos < radar_width:
                range_axis[pos] = "┼"

        for track in tracks:
            pos = int((track.estimated_radius_m / max_radius_m) * (radar_width - 1))
            pos = max(0, min(radar_width - 1, pos))
            if track.threat_level == ThreatLevel.THREAT:
                range_axis[pos] = f"{Colors.RED}{Colors.BOLD}▲{Colors.RESET}"
            elif track.threat_level == ThreatLevel.SUSPICIOUS:
                range_axis[pos] = f"{Colors.YELLOW}◆{Colors.RESET}"
            else:
                range_axis[pos] = f"{Colors.CYAN}○{Colors.RESET}"

        lines.append(f"│ 0m [{' '.join(range_axis[:25])} ...] {max_radius_m:.0f}m │")
        lines.append(f"{Colors.BOLD}└────────────────────────────────────────────────────────┘{Colors.RESET}")
        return "\n".join(lines)

    def render(self, tracker: ThreatTracker, source_name: str, step_count: int, elapsed_time: float) -> str:
        """
        Build complete dashboard string.
        """
        tracks = tracker.get_active_tracks()
        summary = tracker.get_summary()

        out = []
        out.append(f"\n{Colors.BOLD}{Colors.CYAN}========================================================================================{Colors.RESET}")
        out.append(f"{Colors.BOLD}{Colors.WHITE} {self.title} {Colors.RESET} [{source_name.upper()}]")
        out.append(f" Time: {elapsed_time:6.2f}s | Processed Readings: {step_count:05d} | Active Tracks: {summary['total_active_tracks']} | {Colors.RED}{Colors.BOLD}Threats: {summary['threat_count']}{Colors.RESET}")
        out.append(f"{Colors.BOLD}{Colors.CYAN}========================================================================================{Colors.RESET}")

        # Active Track Table
        out.append(f"{Colors.BOLD}{'ID':<8} {'STATUS':<14} {'EST. RADIUS':<14} {'HOPS':<6} {'DQN NEXT HOP':<18} {'CONF':<8} {'REASONING'}{Colors.RESET}")
        out.append("─" * 92)

        if not tracks:
            out.append(f"  {Colors.DIM}No RF signals detected yet. Scanning spectrum...{Colors.RESET}")
        else:
            for track in sorted(tracks, key=lambda t: (t.threat_level != ThreatLevel.THREAT, t.estimated_radius_m)):
                badge = self.format_threat_badge(track.threat_level)
                radius_str = f"{track.estimated_radius_m:5.1f}m (±{track.radius_uncertainty_m:3.0f}m)"
                
                next_hop_str = f"Ch {track.predicted_next_channel:02d} ({track.predicted_next_freq_mhz:5.1f}MHz)" if track.predicted_next_channel is not None else "N/A"
                conf_str = f"{track.threat_confidence*100:4.1f}%"
                reason = track.threat_assessment.reason if track.threat_assessment else ""
                if len(reason) > 34:
                    reason = reason[:31] + "..."

                out.append(f"{track.track_id:<8} {badge:<23} {radius_str:<14} {track.hop_count:<6} {next_hop_str:<18} {conf_str:<8} {Colors.DIM}{reason}{Colors.RESET}")

        out.append("─" * 92)

        # Tactical Range Radar
        out.append(self.render_ascii_radar(tracks))

        # Recent Alerts
        if self.alert_history:
            out.append(f"\n{Colors.BOLD}{Colors.YELLOW}Recent EW Alerts:{Colors.RESET}")
            for alert in self.alert_history[-4:]:
                out.append(f"  {alert}")

        return "\n".join(out)
