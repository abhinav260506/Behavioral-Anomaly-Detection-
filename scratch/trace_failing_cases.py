"""
Diagnostic Script for Tasks 1, 2, 3, 4:
Traces numerical component breakdown for Impossible Travel and Lateral Movement failing testcases.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES, RESTRICTED_RESOURCE_KEYWORDS
from src.cold_start import ColdStartEngine
from src.anomaly_detector import SequentialAnomalyDetector
from src.config import STAGE1_THRESHOLD

CONTINUOUS_FEATURE_NAMES = [
    "hour_sin", "hour_cos", "geo_velocity_kmh", "session_duration",
    "history_count", "data_volume_mb", "download_volume_1h",
    "download_volume_24h", "download_volume_7d", "download_volume_deviation"
]

def run_diagnostics():
    print("="*100)
    print("TASK 1 — IMPOSSIBLE TRAVEL NUMERICAL TRACE")
    print("="*100)

    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)

    total_len = len(df_raw)
    val_len = 4200
    test_len = 4200
    train_end = total_len - val_len - test_len
    val_end = total_len - test_len

    train_df = df_raw.iloc[:train_end].copy()
    val_df = df_raw.iloc[train_end:val_end].copy()

    # Fit feature extractor and scaler on Train normal events only
    fe_tr = FeatureExtractor()
    tr_full_list = []
    for _, r in train_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=GeoLocation(ip=r["ip"], country=r["country"], city=r["city"], latitude=float(r["latitude"]), longitude=float(r["longitude"])),
            resource_accessed=r["resource_accessed"], auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=DeviceFingerprint(os_version=r["os_version"], mac_address=r["mac_address"], protocol=r["protocol"], browser_or_agent=r["browser_agent"]),
            data_volume_mb=vol, label=r["label"]
        )
        f_dict = fe_tr.extract_features(evt)
        f_dict["label"] = r["label"]
        tr_full_list.append(f_dict)

    df_tr_feats = pd.DataFrame(tr_full_list)
    df_tr_matrix = df_tr_feats[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    scaled_tr_matrix = scaler.fit_transform(df_tr_matrix.values)

    df_tr_cont = df_tr_feats[CONTINUOUS_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler_cont = StandardScaler()
    scaled_tr_cont = scaler_cont.fit_transform(df_tr_cont.values)

    normal_scaled_tr = scaled_tr_matrix[df_tr_feats["label"] == "normal"]
    normal_scaled_cont = scaled_tr_cont[df_tr_feats["label"] == "normal"]

    cov_matrix = np.cov(normal_scaled_cont, rowvar=False) + 1e-2 * np.eye(normal_scaled_cont.shape[1])
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    iso_model = IsolationForest(contamination=0.03, random_state=42)
    iso_model.fit(normal_scaled_tr)

    val_norm_scores = iso_model.score_samples(normal_scaled_tr)
    mu_val_if = float(np.mean(val_norm_scores))
    std_val_if = float(np.std(val_norm_scores))

    mahal_tr_dists = [float(np.sqrt(np.dot(np.dot(v, inv_cov_matrix), v))) for v in normal_scaled_cont]
    mu_dist = float(np.mean(mahal_tr_dists))
    std_dist = float(np.std(mahal_tr_dists))

    # --- TASK 1: IMPOSSIBLE TRAVEL PAYLOAD ---
    it_payload = [
        AccessLogEvent("it_1", "u_it", "user", "2026-07-25T08:00:00Z", GeoLocation("1.1.1.1", "India", "Chennai", 13.0827, 80.2707), "/doc", "password", True, 120.0, [], DeviceFingerprint("Win11", "AA:BB:CC:DD:EE:FF", "TLS", "Chrome")),
        AccessLogEvent("it_2", "u_it", "user", "2026-07-25T08:25:00Z", GeoLocation("2.2.2.2", "USA", "New York", 40.7128, -74.0060), "/doc", "password", True, 120.0, [], DeviceFingerprint("Win11", "AA:BB:CC:DD:EE:FF", "TLS", "Chrome"))
    ]

    fe_it = FeatureExtractor()
    det_it = SequentialAnomalyDetector()
    det_it.scaler = scaler
    det_it.model = iso_model
    det_it.is_fitted = True

    cs_engine = ColdStartEngine()

    for idx, evt in enumerate(it_payload):
        print(f"\n--- Impossible Travel Event {idx+1} ({evt.geo_location.city}) ---")
        feats = fe_it.extract_features(evt)
        f_vals = np.array([feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)
        scaled_v = scaler.transform(f_vals.reshape(1, -1))[0]

        f_cont_vals = np.array([feats.get(k, 0.0) for k in CONTINUOUS_FEATURE_NAMES], dtype=float)
        scaled_v_cont = scaler_cont.transform(f_cont_vals.reshape(1, -1))[0]

        raw_if = float(iso_model.score_samples(scaled_v.reshape(1, -1))[0])
        z_if = (mu_val_if - raw_if) / std_val_if
        g_if = float(1.0 - np.exp(-max(0.0, z_if - 1.5) / 2.0))

        dist = float(np.sqrt(np.dot(np.dot(scaled_v_cont, inv_cov_matrix), scaled_v_cont)))
        z_dist = max(0.0, (dist - mu_dist) / std_dist)
        d_score = float(1.0 - np.exp(-max(0.0, z_dist - 1.5) / 2.0))

        peer = cs_engine.evaluate_peer_deviation(evt, feats)

        h_cnt = int(feats.get("history_count", 0))
        w_ind = min(1.0, h_cnt / 10.0)
        w_peer = 1.0 - w_ind

        nov = float(feats.get("resource_novelty", 0.0))
        drift = float(feats.get("behavior_drift_score", 0.0))
        new_res = float(feats.get("new_resource", 0.0))
        new_res_anom = 1.0 if (new_res == 1.0 and h_cnt > 0) else 0.0
        struct_anom = max(
            float(feats.get("restricted_resource", 0.0)),
            float(feats.get("command_sequence_anomaly", 0.0)),
            new_res_anom,
            1.0 if nov > 0.50 else 0.0,
            drift if drift >= 0.50 else 0.0
        )

        ip_div = float(feats.get("ip_diversity_1h", 1.0))
        ip_fails = float(feats.get("rolling_failures_5m", 0.0))
        cross_entity_score = float(min(1.0, (ip_fails * 0.20) + (ip_div * 0.10))) if (ip_fails >= 3.0) else 0.0

        context_score = max(cross_entity_score, struct_anom * 0.85, (w_ind * d_score) + (w_peer * peer))
        final_risk = round(((0.40 * g_if) + (0.60 * context_score)) * 100.0, 1)

        print(f"  geo_velocity_kmh:             {feats.get('geo_velocity_kmh'):.2f}")
        print(f"  history_count:                {h_cnt}")
        print(f"  cold_start:                   {feats.get('cold_start')}")
        print(f"  Isolation Forest raw_score:   {raw_if:.4f}")
        print(f"  calibrated S_global:          {g_if:.4f}")
        print(f"  S_cross_entity:               {cross_entity_score:.4f}")
        print(f"  S_structural:                 {struct_anom:.4f}")
        print(f"  S_behavior (Mahalanobis):     {d_score:.4f} (Mahal Dist={dist:.2f}, Z_dist={z_dist:.2f})")
        print(f"  S_peer:                       {peer:.4f}")
        print(f"  Fusion Weights:               w_ind={w_ind:.2f}, w_peer={w_peer:.2f}")
        print(f"  Context Score:                {context_score:.4f}")
        print(f"  Final Risk Score:             {final_risk:.1f}")

    # --- TASK 2: LATERAL MOVEMENT PAYLOAD ---
    print("\n" + "="*100)
    print("TASK 2 — LATERAL MOVEMENT NUMERICAL TRACE")
    print("="*100)

    lm_payload = [
        AccessLogEvent("lm_1", "u_lm", "user", "2026-07-25T14:00:00Z", GeoLocation("10.0.0.5", "US", "Chicago", 41.8, -87.6), "/engineering/dashboard", "password", True, 120.0, [], DeviceFingerprint("Win11", "AA:BB:CC:DD:EE:FF", "TLS", "Chrome")),
        AccessLogEvent("lm_2", "u_lm", "user", "2026-07-25T14:05:00Z", GeoLocation("10.0.0.5", "US", "Chicago", 41.8, -87.6), "/engineering/git", "password", True, 120.0, [], DeviceFingerprint("Win11", "AA:BB:CC:DD:EE:FF", "TLS", "Chrome")),
        AccessLogEvent("lm_3", "u_lm", "user", "2026-07-25T14:10:00Z", GeoLocation("10.0.0.5", "US", "Chicago", 41.8, -87.6), "/internal/server-config", "password", True, 120.0, [], DeviceFingerprint("Win11", "AA:BB:CC:DD:EE:FF", "TLS", "Chrome")),
        AccessLogEvent("lm_4", "u_lm", "user", "2026-07-25T14:15:00Z", GeoLocation("10.0.0.5", "US", "Chicago", 41.8, -87.6), "/admin/credentials", "password", True, 120.0, [], DeviceFingerprint("Win11", "AA:BB:CC:DD:EE:FF", "TLS", "Chrome")),
        AccessLogEvent("lm_5", "u_lm", "user", "2026-07-25T14:20:00Z", GeoLocation("10.0.0.5", "US", "Chicago", 41.8, -87.6), "/admin/vault_keys", "password", True, 300.0, ["mimikatz.exe"], DeviceFingerprint("Win11", "AA:BB:CC:DD:EE:FF", "TLS", "Chrome"))
    ]

    fe_lm = FeatureExtractor()
    det_lm = SequentialAnomalyDetector()
    det_lm.scaler = scaler
    det_lm.model = iso_model
    det_lm.is_fitted = True

    for idx, evt in enumerate(lm_payload):
        print(f"\n--- Lateral Movement Event {idx+1} ({evt.resource_accessed}) ---")
        feats = fe_lm.extract_features(evt)
        f_vals = np.array([feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)
        scaled_v = scaler.transform(f_vals.reshape(1, -1))[0]

        f_cont_vals = np.array([feats.get(k, 0.0) for k in CONTINUOUS_FEATURE_NAMES], dtype=float)
        scaled_v_cont = scaler_cont.transform(f_cont_vals.reshape(1, -1))[0]

        raw_if = float(iso_model.score_samples(scaled_v.reshape(1, -1))[0])
        z_if = (mu_val_if - raw_if) / std_val_if
        g_if = float(1.0 - np.exp(-max(0.0, z_if - 1.5) / 2.0))

        dist = float(np.sqrt(np.dot(np.dot(scaled_v_cont, inv_cov_matrix), scaled_v_cont)))
        z_dist = max(0.0, (dist - mu_dist) / std_dist)
        d_score = float(1.0 - np.exp(-max(0.0, z_dist - 1.5) / 2.0))

        peer = cs_engine.evaluate_peer_deviation(evt, feats)

        h_cnt = int(feats.get("history_count", 0))
        w_ind = min(1.0, h_cnt / 10.0)
        w_peer = 1.0 - w_ind

        nov = float(feats.get("resource_novelty", 0.0))
        drift = float(feats.get("behavior_drift_score", 0.0))
        new_res = float(feats.get("new_resource", 0.0))
        new_res_anom = 1.0 if (new_res == 1.0 and h_cnt > 0) else 0.0
        struct_anom = max(
            float(feats.get("restricted_resource", 0.0)),
            float(feats.get("command_sequence_anomaly", 0.0)),
            new_res_anom,
            1.0 if nov > 0.50 else 0.0,
            drift if drift >= 0.50 else 0.0
        )

        ip_div = float(feats.get("ip_diversity_1h", 1.0))
        ip_fails = float(feats.get("rolling_failures_5m", 0.0))
        cross_entity_score = float(min(1.0, (ip_fails * 0.20) + (ip_div * 0.10))) if (ip_fails >= 3.0) else 0.0

        context_score = max(cross_entity_score, struct_anom * 0.85, (w_ind * d_score) + (w_peer * peer))
        final_risk = round(((0.40 * g_if) + (0.60 * context_score)) * 100.0, 1)

        print(f"  new_resource:                 {feats.get('new_resource')}")
        print(f"  resource_novelty:             {feats.get('resource_novelty')}")
        print(f"  restricted_resource:          {feats.get('restricted_resource')}")
        print(f"  command_sequence_anomaly:     {feats.get('command_sequence_anomaly')}")
        print(f"  history_count:                {h_cnt}")
        print(f"  Isolation Forest raw_score:   {raw_if:.4f}")
        print(f"  calibrated S_global:          {g_if:.4f}")
        print(f"  S_structural:                 {struct_anom:.4f}")
        print(f"  S_behavior:                   {d_score:.4f}")
        print(f"  Context Score:                {context_score:.4f}")
        print(f"  Final Risk Score:             {final_risk:.1f}")

    # --- TASK 3: RESOURCE SEMANTICS AUDIT ---
    print("\n" + "="*100)
    print("TASK 3 — RESOURCE SEMANTICS VERIFICATION")
    print("="*100)
    test_paths = ["/internal/server-config", "/admin/credentials", "/admin/vault_keys"]
    for path in test_paths:
        is_restr = any(k in path.lower() for k in RESTRICTED_RESOURCE_KEYWORDS)
        print(f"Path: {path:<30} | Is Restricted Resource: {is_restr}")

    # --- TASK 4: FEATURE CALIBRATION Z-SCORES ---
    print("\n" + "="*100)
    print("TASK 4 — FEATURE CALIBRATION Z-SCORES")
    print("="*100)
    check_feats = ["geo_velocity_kmh", "resource_novelty", "restricted_resource", "command_sequence_anomaly", "resource_breadth_change", "privilege_expansion_rate"]
    for fname in check_feats:
        vals = df_tr_feats[fname].values
        p50 = np.percentile(vals, 50)
        p90 = np.percentile(vals, 90)
        p95 = np.percentile(vals, 95)
        p99 = np.percentile(vals, 99)
        mu = np.mean(vals)
        std = np.std(vals) if np.std(vals) > 0 else 1.0
        print(f"Feature: {fname:<30} | P50={p50:.2f}, P90={p90:.2f}, P95={p95:.2f}, P99={p99:.2f}, Mean={mu:.2f}, Std={std:.2f}")

if __name__ == "__main__":
    run_diagnostics()
