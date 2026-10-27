"""
Causal Feature Engineering Engine for Behavioral Anomaly Detection.
Extracts rich temporal, behavioral, fingerprint, resource edge, exfiltration, and drift features.
Strictly causal: uses only information available at or before the current event.
EXCLUDES all labels, event_ids, attack-name strings, and ground truth to prevent data leakage.
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.generator import haversine_distance

SUSPICIOUS_COMMAND_KEYWORDS = {"mimikatz", "vault", "kerberos", "dump", "netstat", "whoami", "shadow", "cat /data", "curl http"}
RESTRICTED_RESOURCE_KEYWORDS = {"vault", "credentials", "kerberos", "admin", "master_keys", "finance_prod", "exfil", "bulk_export", "internal", "server-config", "config", "secret", "prod", "root", "auth", "sysconfig"}

STAGE1_FEATURE_NAMES = [
    "hour_sin", "hour_cos", "is_weekend", "is_off_hours", "geo_velocity_kmh",
    "rolling_failures_5m", "rolling_failures_1h", "ip_diversity_1h", "auth_failed",
    "session_duration", "history_count", "cold_start",
    "fingerprint_known", "fingerprint_mismatch", "os_changed", "mac_changed",
    "protocol_changed", "fingerprint_change_count_24h", "time_since_last_fingerprint_change",
    "new_resource", "resource_novelty", "unique_resources_10m", "unique_resources_1h",
    "resource_breadth_change", "restricted_resource", "new_entity_resource_edge",
    "rare_entity_resource_edge", "command_sequence_anomaly", "data_volume_mb",
    "download_volume_1h", "download_volume_24h", "download_volume_7d",
    "off_hours_access_count_7d", "download_count_7d", "sensitive_resource_access_7d",
    "cumulative_data_volume_7d", "download_volume_deviation", "resource_breadth_7d",
    "resource_breadth_30d", "new_resources_7d", "new_departments_7d",
    "sensitive_resource_ratio", "historical_resource_overlap", "privilege_expansion_rate",
    "behavior_drift_score"
]

class FeatureExtractor:
    def __init__(self):
        # In-memory entity & IP causal state trackers
        self.entity_history: Dict[str, List[AccessLogEvent]] = {}
        self.ip_history: Dict[str, List[AccessLogEvent]] = {}
        self.entity_resource_sets: Dict[str, Set[str]] = {}
        self.entity_primary_fp: Dict[str, DeviceFingerprint] = {}
        self.entity_fp_change_timestamps: Dict[str, List[datetime]] = {}
        self.global_edge_counts: Dict[Tuple[str, str], int] = {}
        self.global_resource_counts: Dict[str, int] = {}
        self.global_total_events: int = 0

    def extract_features(self, event: AccessLogEvent) -> Dict[str, float]:
        """
        Extracts engineered causal numerical feature vector from a single event.
        Asserts zero data leakage (excludes label, event_id, attack names, ground truth).
        """
        eid = event.entity_id
        geo_location_missing = 1.0 if event.geo_location is None else 0.0
        device_fingerprint_missing = 1.0 if event.device_fingerprint is None else 0.0
        ip_addr = event.geo_location.ip if event.geo_location is not None else "0.0.0.0"
        res = event.resource_accessed if event.resource_accessed is not None else "/api/v1/unknown"
        dt = datetime.fromisoformat(event.timestamp)
        history = self.entity_history.get(eid, [])

        # ----------------------------------------------------
        # 1. TIME & CYCLICAL FEATURES
        # ----------------------------------------------------
        hour = dt.hour + dt.minute / 60.0
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        is_weekend = 1.0 if dt.weekday() >= 5 else 0.0
        is_off_hours = 1.0 if (dt.hour < 7 or dt.hour > 19) else 0.0

        # ----------------------------------------------------
        # 2. GEO-VELOCITY FEATURE
        # ----------------------------------------------------
        geo_velocity_kmh = 0.0
        if len(history) > 0 and event.geo_location is not None:
            last_evt = history[-1]
            if last_evt.geo_location is not None:
                last_dt = datetime.fromisoformat(last_evt.timestamp)
                time_delta_hours = max((dt - last_dt).total_seconds() / 3600.0, 0.0001)
                dist_km = haversine_distance(
                    event.geo_location.latitude, event.geo_location.longitude,
                    last_evt.geo_location.latitude, last_evt.geo_location.longitude
                )
                geo_velocity_kmh = dist_km / time_delta_hours

        # ----------------------------------------------------
        # 3. DEVICE SPOOFING FEATURES (Causal Fingerprint Tracking)
        # ----------------------------------------------------
        fp_known = 1.0 if len(history) > 0 and event.device_fingerprint is not None else 0.0
        primary_fp = self.entity_primary_fp.get(eid, None)

        os_changed = 0.0
        mac_changed = 0.0
        protocol_changed = 0.0
        fp_mismatch = 0.0

        if fp_known == 1.0 and primary_fp is not None and event.device_fingerprint is not None:
            if event.device_fingerprint.os_version != primary_fp.os_version:
                os_changed = 1.0
            if event.device_fingerprint.mac_address != primary_fp.mac_address:
                mac_changed = 1.0
            if event.device_fingerprint.protocol != primary_fp.protocol:
                protocol_changed = 1.0
            if os_changed or mac_changed:
                fp_mismatch = 1.0

        # Fingerprint change velocity (24h count & time since last change)
        fp_changes_24h = 0
        time_since_last_change = 999.0
        fp_stamps = self.entity_fp_change_timestamps.get(eid, [])
        for stamp in reversed(fp_stamps):
            delta_sec = (dt - stamp).total_seconds()
            if delta_sec <= 86400:
                fp_changes_24h += 1
            if time_since_last_change == 999.0:
                time_since_last_change = delta_sec / 3600.0

        # ----------------------------------------------------
        # 4. ROLLING AUTHENTICATION & IP DIVERSITY FEATURES
        # ----------------------------------------------------
        rolling_failures_5m = 0
        rolling_failures_1h = 0
        ip_set_1h = set()

        for past_evt in reversed(history):
            past_dt = datetime.fromisoformat(past_evt.timestamp)
            delta_sec = (dt - past_dt).total_seconds()
            if delta_sec > 3600:
                break
            if past_evt.geo_location is not None:
                ip_set_1h.add(past_evt.geo_location.ip)
            if not past_evt.auth_success:
                rolling_failures_1h += 1
                if delta_sec <= 300:
                    rolling_failures_5m += 1

        ip_history = self.ip_history.get(ip_addr, [])
        targeted_entities_1h = set([eid])
        for past_evt in reversed(ip_history):
            past_dt = datetime.fromisoformat(past_evt.timestamp)
            delta_sec = (dt - past_dt).total_seconds()
            if delta_sec > 3600:
                break
            targeted_entities_1h.add(past_evt.entity_id)
            if not past_evt.auth_success:
                rolling_failures_1h += 1
                if delta_sec <= 300:
                    rolling_failures_5m += 1

        ip_diversity_1h = float(max(len(ip_set_1h), len(targeted_entities_1h)))

        # ----------------------------------------------------
        # 5. LATERAL MOVEMENT FEATURES (Behavioral & Resource Edge)
        # ----------------------------------------------------
        known_res_set = self.entity_resource_sets.get(eid, set())
        new_resource = 1.0 if res not in known_res_set else 0.0
        
        if len(history) == 0:
            resource_novelty = 0.5  # Cold start default
        else:
            resource_novelty = 1.0 if new_resource == 1.0 else 0.0

        res_10m = set()
        res_1h = set()
        for past_evt in reversed(history):
            past_dt = datetime.fromisoformat(past_evt.timestamp)
            delta_sec = (dt - past_dt).total_seconds()
            if delta_sec > 3600:
                break
            res_1h.add(past_evt.resource_accessed)
            if delta_sec <= 600:
                res_10m.add(past_evt.resource_accessed)

        unique_resources_10m = float(len(res_10m))
        unique_resources_1h = float(len(res_1h))
        resource_breadth_change = unique_resources_1h / max(1.0, len(known_res_set) / max(1.0, len(history)/10.0))

        restricted_resource = 1.0 if any(k in res.lower() for k in RESTRICTED_RESOURCE_KEYWORDS) else 0.0
        
        edge_key = (eid, res)
        edge_count = self.global_edge_counts.get(edge_key, 0)
        new_entity_resource_edge = 1.0 if edge_count == 0 else 0.0

        res_global_cnt = self.global_resource_counts.get(res, 0)
        total_global = max(1, self.global_total_events)
        rare_entity_resource_edge = res_global_cnt / float(total_global)

        command_sequence_anomaly = 0.0
        for cmd in event.command_sequence:
            for kw in SUSPICIOUS_COMMAND_KEYWORDS:
                if kw in cmd.lower():
                    command_sequence_anomaly += 1.0
                    break

        # ----------------------------------------------------
        # 6. LOW-AND-SLOW EXFILTRATION FEATURES (Accumulated Windows)
        # ----------------------------------------------------
        try:
            data_vol = float(getattr(event, 'data_volume_mb', 0.0))
        except (ValueError, TypeError):
            data_vol = 0.0

        download_volume_1h = data_vol
        download_volume_24h = data_vol
        download_volume_7d = data_vol
        off_hours_access_count_7d = 1.0 if is_off_hours == 1.0 else 0.0
        download_count_7d = 1.0 if data_vol > 0 else 0.0
        sensitive_resource_access_7d = 1.0 if restricted_resource == 1.0 else 0.0

        for past_evt in reversed(history):
            past_dt = datetime.fromisoformat(past_evt.timestamp)
            delta_sec = (dt - past_dt).total_seconds()
            try:
                p_vol = float(getattr(past_evt, 'data_volume_mb', 0.0))
            except (ValueError, TypeError):
                p_vol = 0.0
            p_off = 1.0 if (past_dt.hour < 7 or past_dt.hour > 19) else 0.0
            p_rest = 1.0 if any(k in past_evt.resource_accessed.lower() for k in RESTRICTED_RESOURCE_KEYWORDS) else 0.0

            if delta_sec <= 3600:
                download_volume_1h += p_vol
            if delta_sec <= 86400:
                download_volume_24h += p_vol
            if delta_sec <= 7 * 86400:
                download_volume_7d += p_vol
                off_hours_access_count_7d += p_off
                if p_vol > 0:
                    download_count_7d += 1.0
                sensitive_resource_access_7d += p_rest

        cumulative_data_volume_7d = download_volume_7d
        # Historical mean 24h volume
        mean_24h_vol = max(1.0, download_volume_7d / 7.0)
        download_volume_deviation = download_volume_24h / mean_24h_vol

        # ----------------------------------------------------
        # 7. INSIDER DRIFT FEATURES (Gradual Temporal Expansion)
        # ----------------------------------------------------
        res_7d = set([res])
        res_30d = set([res])
        res_prior_30d = set()
        dept_7d = set([res.split('/')[1] if '/' in res and len(res.split('/')) > 1 else res])

        for past_evt in reversed(history):
            past_dt = datetime.fromisoformat(past_evt.timestamp)
            delta_sec = (dt - past_dt).total_seconds()
            p_res = past_evt.resource_accessed
            p_dept = p_res.split('/')[1] if '/' in p_res and len(p_res.split('/')) > 1 else p_res

            if delta_sec <= 7 * 86400:
                res_7d.add(p_res)
                dept_7d.add(p_dept)
            if delta_sec <= 30 * 86400:
                res_30d.add(p_res)
            else:
                res_prior_30d.add(p_res)

        resource_breadth_7d = float(len(res_7d))
        resource_breadth_30d = float(len(res_30d))
        new_resources_7d = float(len(res_7d - res_prior_30d)) if len(res_prior_30d) > 0 else float(len(res_7d))
        new_departments_7d = float(len(dept_7d))
        sensitive_resource_ratio = sensitive_resource_access_7d / max(1.0, float(len(history) + 1))

        # Jaccard similarity between 7d set and prior 30d set
        if len(res_prior_30d) > 0:
            inter = len(res_7d.intersection(res_prior_30d))
            union = len(res_7d.union(res_prior_30d))
            historical_resource_overlap = float(inter) / float(union) if union > 0 else 1.0
        else:
            historical_resource_overlap = 1.0  # Cold start baseline

        privilege_expansion_rate = sensitive_resource_access_7d / 7.0
        behavior_drift_score = (1.0 - historical_resource_overlap) * (new_resources_7d + 1.0)

        # ----------------------------------------------------
        # 8. GENERAL COLD START & STATE UPDATE
        # ----------------------------------------------------
        auth_failed = 0.0 if event.auth_success else 1.0
        session_duration = float(event.session_duration)
        history_count = float(len(history))
        cold_start = 1.0 if history_count < 10 else 0.0

        # CAUSAL STATE UPDATE (AFTER feature extraction for current event)
        if primary_fp is None:
            self.entity_primary_fp[eid] = event.device_fingerprint
        elif fp_mismatch == 1.0:
            if eid not in self.entity_fp_change_timestamps:
                self.entity_fp_change_timestamps[eid] = []
            self.entity_fp_change_timestamps[eid].append(dt)

        if eid not in self.entity_history:
            self.entity_history[eid] = []
            self.entity_resource_sets[eid] = set()
        if ip_addr not in self.ip_history:
            self.ip_history[ip_addr] = []

        self.entity_history[eid].append(event)
        self.entity_resource_sets[eid].add(res)
        self.ip_history[ip_addr].append(event)

        self.global_edge_counts[edge_key] = edge_count + 1
        self.global_resource_counts[res] = res_global_cnt + 1
        self.global_total_events += 1

        # Final Feature Vector Dictionary (NO labels, NO event_ids, NO attack strings)
        return {
            # Cyclical Time & Geo
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "is_weekend": is_weekend,
            "is_off_hours": is_off_hours,
            "geo_velocity_kmh": geo_velocity_kmh,
            # Auth & Failure
            "rolling_failures_5m": float(rolling_failures_5m),
            "rolling_failures_1h": float(rolling_failures_1h),
            "ip_diversity_1h": ip_diversity_1h,
            "auth_failed": auth_failed,
            "session_duration": session_duration,
            "history_count": history_count,
            "cold_start": cold_start,
            # Device Spoofing
            "fingerprint_known": fp_known,
            "fingerprint_mismatch": fp_mismatch,
            "fp_mismatch": fp_mismatch,
            "os_changed": os_changed,
            "mac_changed": mac_changed,
            "protocol_changed": protocol_changed,
            "fingerprint_change_count_24h": float(fp_changes_24h),
            "time_since_last_fingerprint_change": time_since_last_change,
            # Lateral Movement
            "new_resource": new_resource,
            "resource_novelty": resource_novelty,
            "unique_resources_10m": unique_resources_10m,
            "unique_resources_1h": unique_resources_1h,
            "resource_breadth_change": resource_breadth_change,
            "restricted_resource": restricted_resource,
            "new_entity_resource_edge": new_entity_resource_edge,
            "rare_entity_resource_edge": rare_entity_resource_edge,
            "command_sequence_anomaly": command_sequence_anomaly,
            "suspicious_cmd_count": command_sequence_anomaly,
            # Low-and-Slow Exfiltration
            "data_volume_mb": data_vol,
            "download_volume_1h": download_volume_1h,
            "download_volume_24h": download_volume_24h,
            "download_volume_7d": download_volume_7d,
            "off_hours_access_count_7d": off_hours_access_count_7d,
            "download_count_7d": download_count_7d,
            "sensitive_resource_access_7d": sensitive_resource_access_7d,
            "cumulative_data_volume_7d": cumulative_data_volume_7d,
            "download_volume_deviation": download_volume_deviation,
            # Insider Drift
            "resource_breadth_7d": resource_breadth_7d,
            "resource_breadth_30d": resource_breadth_30d,
            "new_resources_7d": new_resources_7d,
            "new_departments_7d": new_departments_7d,
            "sensitive_resource_ratio": sensitive_resource_ratio,
            "historical_resource_overlap": historical_resource_overlap,
            "privilege_expansion_rate": privilege_expansion_rate,
            "behavior_drift_score": behavior_drift_score
        }

    def get_feature_names(self) -> List[str]:
        return list(self.extract_features(AccessLogEvent(
            event_id="dummy", entity_id="dummy", entity_type="user", timestamp="2026-07-26T00:00:00Z",
            geo_location=GeoLocation("1.1.1.1", "US", "NY", 40.0, -74.0), resource_accessed="/dummy",
            auth_method="password", auth_success=True, session_duration=10.0, command_sequence=[],
            device_fingerprint=DeviceFingerprint("Win11", "00:00:00:00:00:00", "TLS", "Agent")
        )).keys())
