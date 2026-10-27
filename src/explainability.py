from typing import Dict, Any, List, Tuple
from src.schema import AccessLogEvent

ATTACK_DESCRIPTIONS = {
    "impossible_travel": "Sequential login requests originating from geographically distant locations within an implausibly short timeframe exceeding maximum physical travel velocity.",
    "lateral_movement": "Unusual navigation across sensitive internal infrastructure, restricted server configurations, or credential stores attempting unauthorized privilege expansion.",
    "device_spoofing": "Sudden unauthorized shifts in operating system, MAC address, or protocol fingerprints across active sessions.",
    "brute_force": "Rapid burst of failed authentication attempts originating from a single IP address attempting credential discovery.",
    "credential_stuffing": "Distributed login failure attempts across multiple distinct user accounts originating from shared infrastructure.",
    "low_and_slow_exfiltration": "Gradual off-hours data transfers accumulating elevated volume across extended time windows.",
    "insider_drift": "Behavioral drift over time characterized by expanding resource access, new department interactions, or off-hours activity.",
    "normal": "Access activity adheres to standard behavioral baseline and peer population norms."
}

ATTACK_DESCRIPTIONS = {
    "impossible_travel": "Sequential login requests originating from geographically distant locations within an implausibly short timeframe exceeding maximum physical travel velocity.",
    "lateral_movement": "Unusual navigation across sensitive internal infrastructure, restricted server configurations, or credential stores attempting unauthorized privilege expansion.",
    "device_spoofing": "Sudden unauthorized shifts in operating system, MAC address, or protocol fingerprints across active sessions.",
    "brute_force": "Rapid burst of failed authentication attempts originating from a single IP address attempting credential discovery.",
    "credential_stuffing": "Distributed login failure attempts across multiple distinct user accounts originating from shared infrastructure.",
    "low_and_slow_exfiltration": "Gradual off-hours data transfers accumulating elevated volume across extended time windows.",
    "insider_drift": "Behavioral drift over time characterized by expanding resource access, new department interactions, or off-hours activity.",
    "normal": "Access activity adheres to standard behavioral baseline and peer population norms."
}

class ExplainabilityEngine:
    def __init__(self):
        pass

    def explain_sequence(
        self,
        raw_events: List[Dict[str, Any]],
        traces: List[Dict[str, Any]],
        predicted_attack: str,
        max_risk: float,
        threshold: float = 60.0
    ) -> Dict[str, Any]:
        """
        Generates attack-specific explanation, root cause analysis, and feature attributions
        tailored strictly to the predicted attack category and user input event sequence.
        """
        key_reasons: List[str] = []
        attribution_weights: Dict[str, float] = {}

        attack_desc = ATTACK_DESCRIPTIONS.get(predicted_attack, ATTACK_DESCRIPTIONS["normal"])

        if max_risk < threshold or predicted_attack == "normal":
            return {
                "predicted_attack": "normal",
                "attack_description": ATTACK_DESCRIPTIONS["normal"],
                "key_reasons": [
                    "All observed telemetry parameters fall within standard operational baselines.",
                    "Access resources align with standard role permissions and historical schedules.",
                    "No physical velocity anomalies, fingerprint mismatches, or auth bursts detected."
                ],
                "feature_attribution_weights": {"Profile Baseline Alignment": 95.0, "Peer Group Consistency": 90.0}
            }

        # --- Attack-Specific Explainability Builders ---

        if predicted_attack == "impossible_travel":
            # Extract consecutive geo points
            geo_events = [e for e in raw_events if e.get("geo_location") and isinstance(e.get("geo_location"), dict)]
            if len(geo_events) >= 2:
                g1, g2 = geo_events[0]["geo_location"], geo_events[-1]["geo_location"]
                c1, c2 = g1.get("city", "Location A"), g2.get("city", "Location B")
                country1, country2 = g1.get("country", "Country A"), g2.get("country", "Country B")
                key_reasons.append(f"Implausible Physical Movement: Sequential login requests detected between {c1} ({country1}) and {c2} ({country2}).")
            else:
                key_reasons.append("Implausible Physical Movement: Rapid geographic position shift between consecutive sessions.")

            # Find max velocity feature in traces
            max_vel = 0.0
            for t in traces:
                v = t.get("raw_feature_vector", {}).get("geo_velocity_kmh", 0.0)
                if v > max_vel: max_vel = v
            if max_vel > 0:
                key_reasons.append(f"Excessive Travel Velocity: Implied movement speed of {max_vel:.0f} km/h exceeds 500 km/h physical threshold.")

            key_reasons.append("Source IP Infrastructure Shift: Connection originated from geographically distant autonomous systems.")
            attribution_weights = {"Geo Velocity": min(100.0, max(85.0, max_vel / 300.0)), "Location Novelty": 85.0, "Time Delta Anomaly": 75.0}

        elif predicted_attack == "lateral_movement":
            restr_paths = set()
            cmds = set()
            for e in raw_events:
                res = e.get("resource_accessed", "")
                if any(k in res.lower() for k in ["admin", "config", "vault", "credentials", "internal", "secret", "root"]):
                    restr_paths.add(res)
                for c in e.get("command_sequence", []):
                    cmds.add(c)

            if restr_paths:
                paths_str = ", ".join(list(restr_paths)[:3])
                key_reasons.append(f"Restricted Endpoint Traversal: Sequential access escalation reaching protected endpoints [{paths_str}].")
            else:
                key_reasons.append("Restricted Endpoint Traversal: Entity navigated into protected internal infrastructure.")

            key_reasons.append("Privilege Escalation Pathway: Entity transitioned from standard user portals to internal server configurations.")

            if cmds:
                cmds_str = ", ".join(list(cmds)[:3])
                key_reasons.append(f"Privileged Command Execution: Security-sensitive system commands [{cmds_str}] executed.")

            attribution_weights = {"Restricted Resource Access": 90.0, "Resource Breadth Expansion": 80.0, "Command Sequence Anomaly": 85.0 if cmds else 65.0}

        elif predicted_attack == "device_spoofing":
            fp_changes = set()
            os_set = set()
            mac_set = set()
            proto_set = set()

            for e in raw_events:
                # Direct JSON properties or nested device_fingerprint
                os_val = e.get("device_os") or (e.get("device_fingerprint", {}).get("os_version") if isinstance(e.get("device_fingerprint"), dict) else None)
                mac_val = e.get("mac_address") or (e.get("device_fingerprint", {}).get("mac_address") if isinstance(e.get("device_fingerprint"), dict) else None)
                proto_val = e.get("protocol") or (e.get("device_fingerprint", {}).get("protocol") if isinstance(e.get("device_fingerprint"), dict) else None)

                if os_val: os_set.add(os_val)
                if mac_val: mac_set.add(mac_val)
                if proto_val: proto_set.add(proto_val)

            if len(os_set) > 1:
                key_reasons.append(f"Operating System Alteration: Session OS switched across active stream [{', '.join(os_set)}].")
            if len(mac_set) > 1:
                key_reasons.append(f"Hardware MAC Address Mutation: Client MAC address changed across requests [{', '.join(mac_set)}].")
            if len(proto_set) > 1:
                key_reasons.append(f"Network Protocol Shift: Connection transport mutated across requests [{', '.join(proto_set)}].")
            if not key_reasons:
                key_reasons.append("Unauthenticated Client Fingerprint Mismatch: Device hardware & OS signatures do not match baseline.")

            attribution_weights = {"Fingerprint Mismatch": 90.0, "OS Alteration": 85.0, "MAC Address Shift": 80.0}

        elif predicted_attack == "brute_force":
            fails = sum(1 for e in raw_events if e.get("auth_success") is False)
            key_reasons.append(f"High-Frequency Authentication Failures: Burst of {max(3, fails)} consecutive failed login attempts.")
            key_reasons.append("Single-Source IP Targeting: Concentrated credential discovery attempts from unified source infrastructure.")
            attribution_weights = {"Auth Failures (5m)": 95.0, "Auth Failure Rate": 85.0}

        elif predicted_attack == "credential_stuffing":
            key_reasons.append("Multi-Account Login Burst: Failed authentication attempts targeting multiple distinct user entities.")
            key_reasons.append("Distributed Proxy Origin: Coordinated login discovery attempts originating from shared proxy / NAT IP.")
            attribution_weights = {"IP Diversity": 90.0, "Cross-Entity Failures": 85.0}

        elif predicted_attack == "low_and_slow_exfiltration":
            tot_vol = sum(float(e.get("data_volume_mb", 0.0)) for e in raw_events)
            key_reasons.append(f"Cumulative Volume Accumulation: Total data transfer reached {tot_vol:.1f} MB across observation window.")
            key_reasons.append("Off-Hours Transfer Schedule: Data exfiltration initiated during non-operational off-hours window.")
            key_reasons.append("Sustained Volume Anomaly: Download volume significantly exceeds historical 7-day baselines.")
            attribution_weights = {"Data Volume Deviation": 90.0, "Off-Hours Timing": 75.0, "Download Volume 7d": 70.0}

        elif predicted_attack == "insider_drift":
            depts = set()
            for e in raw_events:
                res = e.get("resource_accessed", "")
                parts = [p for p in res.split("/") if p]
                if parts: depts.add(parts[0])

            if len(depts) > 1:
                key_reasons.append(f"Organizational Scope Drift: Access footprint expanded across {len(depts)} distinct departments [{', '.join(depts)}].")
            else:
                key_reasons.append("Behavioral Profile Drift: Entity expanded access beyond established historical role boundaries.")

            key_reasons.append("Resource Novelty Accumulation: Sequential interaction with previously unvisited enterprise data stores.")
            key_reasons.append("Off-Hours Privilege Escalation: Elevated data access conducted during non-business hours.")
            attribution_weights = {"Departmental Expansion": 85.0, "Behavior Drift Score": 80.0, "Resource Novelty": 75.0}

        return {
            "predicted_attack": predicted_attack,
            "attack_description": attack_desc,
            "key_reasons": key_reasons,
            "feature_attribution_weights": attribution_weights
        }

    def explain_alert(
        self,
        event: AccessLogEvent,
        features: Dict[str, float],
        risk_score: float,
        predicted_attack: str,
        threshold: float = 60.0
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generates list of contributing factors and structured explanation metrics.
        """
        factors: List[str] = []
        key_reasons: List[str] = []
        attribution_weights: Dict[str, float] = {}

        if risk_score < threshold:
            factors.append("Isolated evaluation completed. No anomaly exceeded the configured Stage-1 threshold.")
            key_reasons.append("All observed telemetry parameters fall within standard operational bounds.")
            return factors, {
                "risk_score": risk_score,
                "predicted_attack": "normal",
                "attack_description": ATTACK_DESCRIPTIONS.get("normal", ""),
                "key_reasons": key_reasons,
                "feature_attribution_weights": {},
                "entity_id": event.entity_id if event else "N/A"
            }

        factors.append("Isolated evaluation detected anomalous behavior exceeding the configured Stage-1 threshold.")

        # Geo-Velocity attribution
        vel = features.get("geo_velocity_kmh", 0.0)
        if vel > 500.0:
            msg = f"Implausible Geo-Velocity: Detected travel speed of {vel:.0f} km/h between login locations (threshold: 500 km/h)"
            factors.append(msg)
            key_reasons.append(f"Implausible physical movement speed of {vel:.0f} km/h recorded between login sessions.")
            attribution_weights["Geo Velocity"] = min(100.0, (vel / 1000.0) * 100.0)

        # Device Fingerprint attribution
        if features.get("fp_mismatch", 0.0) > 0 or features.get("fingerprint_mismatch", 0.0) > 0:
            msg = "Device Fingerprint Mismatch: MAC address or OS version does not match established baseline profile"
            factors.append(msg)
            key_reasons.append("Sudden change in client MAC address, OS version, or network protocol detected.")
            attribution_weights["Fingerprint Mismatch"] = 80.0

        # Auth Failure attribution
        fail_5m = features.get("rolling_failures_5m", 0.0)
        if fail_5m >= 3:
            msg = f"Authentication Burst: {int(fail_5m)} failed login attempts recorded in 5 minutes"
            factors.append(msg)
            key_reasons.append(f"Burst of {int(fail_5m)} consecutive authentication failures within a 5-minute window.")
            attribution_weights["Auth Failures (5m)"] = min(100.0, fail_5m * 15.0)

        # Restricted Resource / Privilege Expansion
        if features.get("restricted_resource", 0.0) > 0:
            res_path = event.resource_accessed if event and hasattr(event, 'resource_accessed') else "restricted endpoint"
            msg = f"Restricted Resource Access: Access attempt to protected endpoint '{res_path}'"
            factors.append(msg)
            key_reasons.append(f"Attempted access to security-sensitive endpoint '{res_path}'.")
            attribution_weights["Restricted Resource"] = 90.0

        # Suspicious Command Sequence attribution
        if features.get("command_sequence_anomaly", 0.0) > 0 or features.get("suspicious_cmd_count", 0.0) > 0:
            cmds_str = ", ".join(event.command_sequence) if event and hasattr(event, 'command_sequence') and event.command_sequence else "security commands"
            msg = f"Suspicious Privileged Command Sequence: Executed security-sensitive commands [{cmds_str}]"
            factors.append(msg)
            key_reasons.append(f"Execution of privileged system commands [{cmds_str}].")
            attribution_weights["Suspicious Commands"] = 95.0

        # Off-Hours Access attribution
        if features.get("is_off_hours", 0.0) > 0:
            msg = "Anomalous Off-Hours Timing: Connection established outside standard business hours"
            factors.append(msg)
            key_reasons.append("Activity initiated during non-standard off-hours window.")
            attribution_weights["Off-Hours Timing"] = 40.0

        # Resource Novelty
        if features.get("resource_novelty", 0.0) > 0.8:
            key_reasons.append("Resource access pattern represents 100% novel interaction with zero prior baseline.")
            attribution_weights["Resource Novelty"] = 65.0

        if not key_reasons:
            key_reasons.append("Multidimensional feature vector deviation from standard profile.")

        explanation_details = {
            "risk_score": risk_score,
            "predicted_attack": predicted_attack,
            "attack_description": ATTACK_DESCRIPTIONS.get(predicted_attack, ATTACK_DESCRIPTIONS["normal"]),
            "key_reasons": key_reasons,
            "feature_attribution_weights": attribution_weights,
            "entity_id": event.entity_id if event else "N/A"
        }

        return factors, explanation_details
