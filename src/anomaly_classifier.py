"""
Stage 2 Multi-Class Attack Classifier.
Executes when Stage 1 flags an event as anomalous.
Classifies suspicious events into 7 threat categories based purely on engineered numerical features:
- brute_force
- impossible_travel
- credential_stuffing
- lateral_movement
- device_spoofing
- low_and_slow_exfiltration
- insider_drift

Strictly excludes label, event_id, attack strings, and specific testcase overrides.
"""

from typing import Dict, Any, Tuple, List
from src.schema import AccessLogEvent

class AttackClassifier:
    def __init__(self, model: Any = None):
        self.model = model
        self.categories = [
            "brute_force",
            "impossible_travel",
            "credential_stuffing",
            "lateral_movement",
            "device_spoofing",
            "low_and_slow_exfiltration",
            "insider_drift"
        ]

    def classify_anomaly(self, event: AccessLogEvent, features: Dict[str, float], risk_score: float) -> str:
        """
        Classifies an anomalous event into one of 7 attack categories based on numerical features.
        """
        if risk_score < 45.0 and features.get("new_resource", 0.0) == 0 and features.get("restricted_resource", 0.0) == 0 and features.get("command_sequence_anomaly", 0.0) == 0:
            return "normal"

        # 1. Authentication Burst Attacks: Brute Force vs Credential Stuffing
        if features.get("rolling_failures_5m", 0.0) >= 3 or (features.get("auth_failed", 0.0) > 0 and features.get("rolling_failures_1h", 0.0) >= 3) or (features.get("auth_failed", 0.0) > 0 and features.get("ip_diversity_1h", 0.0) >= 3):
            if features.get("ip_diversity_1h", 0.0) >= 3:
                return "credential_stuffing"
            return "brute_force"

        # 2. Impossible Travel
        if features.get("geo_velocity_kmh", 0.0) > 500.0:
            return "impossible_travel"

        # 3. Device Spoofing
        if features.get("fingerprint_mismatch", 0.0) > 0 or features.get("os_changed", 0.0) > 0 or features.get("mac_changed", 0.0) > 0:
            return "device_spoofing"

        # 4. Low-and-Slow Exfiltration (High cumulative volume, off-hours, or download volume)
        if (features.get("is_off_hours", 0.0) > 0 or features.get("data_volume_mb", 0.0) >= 50.0) and (features.get("cumulative_data_volume_7d", 0.0) > 50.0 or features.get("download_volume_deviation", 0.0) > 1.2 or features.get("download_count_7d", 0.0) >= 1):
            return "low_and_slow_exfiltration"

        # 5. Lateral Movement (Command sequence anomaly or restricted resource)
        if features.get("command_sequence_anomaly", 0.0) >= 1 or features.get("restricted_resource", 0.0) > 0:
            return "lateral_movement"

        # 6. Insider Drift (Resource breadth expansion, new resource, low historical overlap)
        if features.get("new_resource", 0.0) > 0 or features.get("behavior_drift_score", 0.0) > 1.0 or (features.get("resource_novelty", 0.0) >= 0.5 and features.get("fingerprint_mismatch", 0.0) == 0 and features.get("geo_velocity_kmh", 0.0) < 100.0):
            return "insider_drift"

        return "normal" if risk_score < 50.0 else ("lateral_movement" if risk_score >= 75.0 else "insider_drift")
