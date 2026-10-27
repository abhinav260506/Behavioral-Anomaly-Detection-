"""
Diagnostic script to inspect 100 normal test events and trace why normal events received high Stage-1 risk scores.
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
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.cold_start import ColdStartEngine
from src.anomaly_detector import SequentialAnomalyDetector
from src.config import STAGE1_THRESHOLD

def diagnose():
    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)

    total_len = len(df_raw)
    val_len = 4200
    test_len = 4200
    train_end = total_len - val_len - test_len
    val_end = total_len - test_len

    train_df = df_raw.iloc[:train_end].copy()
    test_df = df_raw.iloc[val_end:].copy()

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

    normal_scaled_tr = scaled_tr_matrix[df_tr_feats["label"] == "normal"]
    cov_matrix = np.cov(normal_scaled_tr, rowvar=False) + 1e-4 * np.eye(normal_scaled_tr.shape[1])
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    iso_model = IsolationForest(contamination=0.03, random_state=42)
    iso_model.fit(normal_scaled_tr)

    val_norm_scores = iso_model.score_samples(normal_scaled_tr)
    mu_val_if = float(np.mean(val_norm_scores))
    std_val_if = float(np.std(val_norm_scores))

    print(f"Training Normal Isolation Forest score_samples mean: {mu_val_if:.4f}, std: {std_val_if:.4f}")

    fe_test = FeatureExtractor()
    cs_engine = ColdStartEngine()
    detector = SequentialAnomalyDetector()
    detector.scaler = scaler
    detector.model = iso_model
    detector.is_fitted = True

    normal_test_samples = test_df[test_df["label"] == "normal"].head(100)

    g_if_list = []
    d_score_list = []
    struct_list = []
    peer_list = []
    risk_list = []

    for idx, r in normal_test_samples.iterrows():
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

        feats = fe_test.extract_features(evt)
        f_vals = np.array([feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)
        scaled_v = scaler.transform(f_vals.reshape(1, -1))[0]

        raw_if = float(iso_model.score_samples(scaled_v.reshape(1, -1))[0])
        z_if = (mu_val_if - raw_if) / (std_val_if * 2.0)
        g_if = float(1.0 / (1.0 + np.exp(-z_if)))

        dist = float(np.sqrt(np.dot(np.dot(scaled_v, inv_cov_matrix), scaled_v)))
        d_score = float(1.0 - np.exp(-dist / 7.8839))

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

        w_ind = min(1.0, h_cnt / 10.0)
        w_peer = 1.0 - w_ind

        context_score = max(struct_anom * 0.85, (w_ind * d_score) + (w_peer * peer))
        final_risk = round(((0.40 * g_if) + (0.60 * context_score)) * 100.0, 1)

        g_if_list.append(g_if)
        d_score_list.append(d_score)
        struct_list.append(struct_anom)
        peer_list.append(peer)
        risk_list.append(final_risk)

    print("\n--- Diagnostic Component Breakdown for First 100 Normal Test Events ---")
    print(f"S_global (IF):     Mean={np.mean(g_if_list):.4f}, Min={np.min(g_if_list):.4f}, Max={np.max(g_if_list):.4f}")
    print(f"S_behavior (Dist): Mean={np.mean(d_score_list):.4f}, Min={np.min(d_score_list):.4f}, Max={np.max(d_score_list):.4f}")
    print(f"S_structural:      Mean={np.mean(struct_list):.4f}, Min={np.min(struct_list):.4f}, Max={np.max(struct_list):.4f}")
    print(f"S_peer:            Mean={np.mean(peer_list):.4f}, Min={np.min(peer_list):.4f}, Max={np.max(peer_list):.4f}")
    print(f"Final Risk:        Mean={np.mean(risk_list):.1f}, Min={np.min(risk_list):.1f}, Max={np.max(risk_list):.1f}")

if __name__ == "__main__":
    diagnose()
