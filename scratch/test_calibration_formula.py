"""
Test script to verify statistical calibration of Stage-1 Isolation Forest and Mahalanobis Distance.
Evaluates on Train and Validation splits to achieve Stage-1 FPR <= 10% and Stage-1 Recall >= 80%.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.cold_start import ColdStartEngine

def test_calibration():
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
    CONTINUOUS_FEATURE_NAMES = [
        "hour_sin", "hour_cos", "geo_velocity_kmh", "session_duration",
        "history_count", "data_volume_mb", "download_volume_1h",
        "download_volume_24h", "download_volume_7d", "download_volume_deviation"
    ]

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

    print(f"Normal Train IF Scores: Mean={mu_val_if:.4f}, Std={std_val_if:.4f}")
    print(f"Normal Train Mahalanobis Dists: Mean={mu_dist:.4f}, Std={std_dist:.4f}")

    # Evaluate on Validation split
    fe_val = FeatureExtractor()
    cs_engine = ColdStartEngine()

    val_norm_risks = []
    val_att_risks = []

    for _, r in val_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        geo = GeoLocation(ip=r["ip"], country=r["country"], city=r["city"], latitude=float(r["latitude"]), longitude=float(r["longitude"]))
        fp = DeviceFingerprint(os_version=r["os_version"], mac_address=r["mac_address"], protocol=r["protocol"], browser_or_agent=r["browser_agent"])
        vol = float(r.get("data_volume_mb", 0.0))

        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=geo, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=fp, data_volume_mb=vol, label=r["label"]
        )

        feats = fe_val.extract_features(evt)
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

        ip_div = float(feats.get("ip_diversity_1h", 1.0))
        ip_fails = float(feats.get("rolling_failures_5m", 0.0))
        nov = float(feats.get("resource_novelty", 0.0))
        drift = float(feats.get("behavior_drift_score", 0.0))
        new_res = float(feats.get("new_resource", 0.0))
        h_cnt = int(feats.get("history_count", 0))

        new_res_anom = 1.0 if (new_res == 1.0 and h_cnt > 0) else 0.0
        struct_anom = max(
            float(feats.get("restricted_resource", 0.0)),
            float(feats.get("command_sequence_anomaly", 0.0)),
            new_res_anom,
            1.0 if nov > 0.50 else 0.0,
            drift if drift >= 0.50 else 0.0
        )

        cross_entity_score = float(min(1.0, (ip_fails * 0.20) + (ip_div * 0.10))) if (ip_fails >= 3.0) else 0.0

        w_ind = min(1.0, h_cnt / 10.0)
        w_peer = 1.0 - w_ind

        context_score = max(cross_entity_score, struct_anom * 0.85, (w_ind * d_score) + (w_peer * peer))
        final_risk = round(((0.40 * g_if) + (0.60 * context_score)) * 100.0, 1)

        if r["label"] == "normal":
            val_norm_risks.append(final_risk)
        else:
            val_att_risks.append(final_risk)

    val_norm_risks = np.array(val_norm_risks)
    val_att_risks = np.array(val_att_risks)

    print("\n--- Validation Risk Percentiles ---")
    print(f"Normal Val Risk P50={np.percentile(val_norm_risks, 50):.1f}, P90={np.percentile(val_norm_risks, 90):.1f}, P95={np.percentile(val_norm_risks, 95):.1f}, Max={np.max(val_norm_risks):.1f}")
    print(f"Attack Val Risk P50={np.percentile(val_att_risks, 50):.1f}, P90={np.percentile(val_att_risks, 90):.1f}, P95={np.percentile(val_att_risks, 95):.1f}, Min={np.min(val_att_risks):.1f}")

    print("\n--- Normal Validation Component Breakdown ---")
    val_norm_g_if = []
    val_norm_d_score = []
    val_norm_peer = []
    val_norm_struct = []
    val_norm_cross = []

    for _, r in val_df[val_df["label"] == "normal"].iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        geo = GeoLocation(ip=r["ip"], country=r["country"], city=r["city"], latitude=float(r["latitude"]), longitude=float(r["longitude"]))
        fp = DeviceFingerprint(os_version=r["os_version"], mac_address=r["mac_address"], protocol=r["protocol"], browser_or_agent=r["browser_agent"])
        vol = float(r.get("data_volume_mb", 0.0))

        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=geo, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=fp, data_volume_mb=vol, label=r["label"]
        )

        feats = fe_val.extract_features(evt)
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

        nov = float(feats.get("resource_novelty", 0.0))
        drift = float(feats.get("behavior_drift_score", 0.0))
        new_res = float(feats.get("new_resource", 0.0))
        h_cnt = int(feats.get("history_count", 0))

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

        val_norm_g_if.append(g_if)
        val_norm_d_score.append(d_score)
        val_norm_peer.append(peer)
        val_norm_struct.append(struct_anom)
        val_norm_cross.append(cross_entity_score)

    print(f"Normal Val S_global (IF):     Mean={np.mean(val_norm_g_if):.4f}, Max={np.max(val_norm_g_if):.4f}")
    print(f"Normal Val S_behavior (Dist): Mean={np.mean(val_norm_d_score):.4f}, Max={np.max(val_norm_d_score):.4f}")
    print(f"Normal Val S_peer:            Mean={np.mean(val_norm_peer):.4f}, Max={np.max(val_norm_peer):.4f}")
    print(f"Normal Val S_structural:      Mean={np.mean(val_norm_struct):.4f}, Max={np.max(val_norm_struct):.4f}")
    print(f"Normal Val S_cross_entity:    Mean={np.mean(val_norm_cross):.4f}, Max={np.max(val_norm_cross):.4f}")
    print(f"{'Threshold T':<12} | {'Val Normal FPR %':<18} | {'Val Attack Recall %':<20} | {'F1 Score %':<12}")
    print("-" * 70)
    for T in [20.0, 30.0, 40.0, 45.0, 50.0, 55.0, 60.0]:
        fpr = (np.sum(val_norm_risks >= T) / float(len(val_norm_risks))) * 100.0
        rec = (np.sum(val_att_risks >= T) / float(len(val_att_risks))) * 100.0
        prec = (np.sum(val_att_risks >= T) / float(np.sum(val_att_risks >= T) + np.sum(val_norm_risks >= T))) * 100.0 if (np.sum(val_att_risks >= T) + np.sum(val_norm_risks >= T)) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        print(f"{T:<12.1f} | {fpr:>17.2f}% | {rec:>19.2f}% | {f1:>11.2f}%")

if __name__ == "__main__":
    test_calibration()
