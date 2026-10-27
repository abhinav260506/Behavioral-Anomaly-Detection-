"""
Cold-Start Handler & Peer-Group Benchmarking Module.
Addresses Deliverable #2 & Objective #5 by providing peer population baselines
for brand-new entities (users, service accounts, edge devices) with insufficient history (< 10 events).
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from src.schema import AccessLogEvent

class ColdStartEngine:
    def __init__(self, cold_start_threshold: int = 10):
        self.cold_start_threshold = cold_start_threshold
        # Population peer profile means & standard deviations by entity_type
        self.peer_profiles: Dict[str, Dict[str, Dict[str, float]]] = {
            "user": {
                "avg_session_duration": 1800.0,
                "std_session_duration": 600.0,
                "typical_off_hours_rate": 0.05,
                "typical_resource_novelty": 0.1,
                "max_expected_velocity_kmh": 600.0,
            },
            "service_account": {
                "avg_session_duration": 300.0,
                "std_session_duration": 100.0,
                "typical_off_hours_rate": 0.50,
                "typical_resource_novelty": 0.02,
                "max_expected_velocity_kmh": 100.0,
            },
            "edge_device": {
                "avg_session_duration": 60.0,
                "std_session_duration": 15.0,
                "typical_off_hours_rate": 0.50,
                "typical_resource_novelty": 0.01,
                "max_expected_velocity_kmh": 50.0,
            }
        }

    def compute_cold_start_weight(self, history_count: float) -> Tuple[float, float]:
        """
        Returns (individual_weight, peer_weight).
        For history_count == 0: (0.0, 1.0)
        For history_count >= 10: (1.0, 0.0)
        Linear transition in between.
        """
        w_ind = min(1.0, max(0.0, history_count / self.cold_start_threshold))
        w_peer = 1.0 - w_ind
        return w_ind, w_peer

    def evaluate_peer_deviation(self, event: AccessLogEvent, features: Dict[str, float]) -> float:
        """
        Computes deviation score (0.0 to 1.0) against peer population baseline.
        """
        etype = event.entity_type if event.entity_type in self.peer_profiles else "user"
        peer = self.peer_profiles[etype]

        score = 0.0

        # Geo-velocity relative to max expected for entity type (Continuous physical impossibility scaling)
        vel = features.get("geo_velocity_kmh", 0.0)
        max_exp = peer["max_expected_velocity_kmh"]
        if vel > max_exp:
            # Scale continuously from 0.35 up to 1.0 for extreme physically impossible speeds (>3,000 km/h)
            score += min(1.0, 0.35 + (vel - max_exp) / 3000.0)

        # Off hours penalty
        if features.get("is_off_hours", 0.0) > 0 and peer["typical_off_hours_rate"] < 0.1:
            score += 0.20

        # Auth failures
        if features.get("rolling_failures_5m", 0.0) > 3:
            score += 0.30

        # Fingerprint mismatch
        if features.get("fingerprint_mismatch", 0.0) > 0 or features.get("fp_mismatch", 0.0) > 0:
            score += 0.25

        return min(1.0, score)
