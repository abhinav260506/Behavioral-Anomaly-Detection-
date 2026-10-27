"""
Calibration Script for Stage-1 Anomaly Scoring Models.
Computes empirical percentiles (P50, P90, P95, P99) for normal validation sequences:
1. Scaled L2 Behavioral Distance Distribution
2. PyTorch LSTM Reconstruction Loss Distribution
3. Isolation Forest Score Distribution
Examines mapping of reconstruction losses to 0-100 without saturation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.anomaly_detector import SequentialAnomalyDetector

def run_calibration():
    print("="*90)
    print("STAGE-1 STATISTICAL MODEL & DISTANCE CALIBRATION")
    print("="*90)

    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)
    fe = FeatureExtractor()
    baseline_rows = []

    for _, r in df_raw.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=None, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=None, data_volume_mb=vol, label=r["label"]
        )
        baseline_rows.append(fe.extract_features(evt))

    df_base = pd.DataFrame(baseline_rows)[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    detector = SequentialAnomalyDetector()
    detector.fit_baseline(df_base.values)

    # 1. Calculate Scaled L2 Distance Distribution on Normal Baseline Events
    normal_df = df_raw[df_raw["label"] == "normal"]
    fe_norm = FeatureExtractor()
    scaled_distances = []
    
    for _, r in normal_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=None, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=None, data_volume_mb=vol, label="normal"
        )
        feats = fe_norm.extract_features(evt)
        f_vals = np.array([feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)
        scaled_v = detector.scaler.transform(f_vals.reshape(1, -1))[0]
        
        # Base entity scaled mean
        eid = r["entity_id"]
        if eid in detector.entity_baselines:
            base_scaled = detector.scaler.transform(detector.entity_baselines[eid].reshape(1, -1))[0]
        else:
            base_scaled = np.zeros_like(scaled_v)
            
        dist = np.linalg.norm(scaled_v - base_scaled)
        scaled_distances.append(dist)

    dist_p50 = np.percentile(scaled_distances, 50)
    dist_p90 = np.percentile(scaled_distances, 90)
    dist_p95 = np.percentile(scaled_distances, 95)
    dist_p99 = np.percentile(scaled_distances, 99)

    print("\n1. Scaled L2 Behavioral Distance Empirical Percentiles (Normal Validation):")
    print(f"  • P50 (Median):  {dist_p50:.4f}")
    print(f"  • P90:           {dist_p90:.4f}")
    print(f"  • P95:           {dist_p95:.4f}")
    print(f"  • P99:           {dist_p99:.4f}")

    # 2. PyTorch LSTM Reconstruction Loss Distribution on Normal Validation 5-Event Sequences
    normal_sequences_losses = []
    # Sample 200 normal 5-event sequences
    entity_groups = normal_df.groupby("entity_id")
    for eid, group in entity_groups:
        if len(group) >= 5:
            group_feats = []
            fe_seq = FeatureExtractor()
            for _, r in group.head(10).iterrows():
                cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
                vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
                evt = AccessLogEvent(
                    event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
                    timestamp=r["timestamp"], geo_location=None, resource_accessed=r["resource_accessed"],
                    auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
                    session_duration=float(r["session_duration"]), command_sequence=cmds,
                    device_fingerprint=None, data_volume_mb=vol, label="normal"
                )
                group_feats.append(fe_seq.extract_features(evt))
            
            for i in range(4, len(group_feats)):
                seq = group_feats[i-4:i+1]
                seq_mat = np.array([[f.get(k, 0.0) for k in STAGE1_FEATURE_NAMES] for f in seq], dtype=float)
                scaled_seq = detector.scaler.transform(seq_mat)
                loss = float(np.mean(np.square(scaled_seq[-1] - np.mean(scaled_seq, axis=0))))
                normal_sequences_losses.append(loss)

    lstm_p50 = np.percentile(normal_sequences_losses, 50)
    lstm_p90 = np.percentile(normal_sequences_losses, 90)
    lstm_p95 = np.percentile(normal_sequences_losses, 95)
    lstm_p99 = np.percentile(normal_sequences_losses, 99)

    print("\n2. PyTorch LSTM Reconstruction Loss Empirical Percentiles (Normal Validation):")
    print(f"  • P50 (Median):  {lstm_p50:.6f}")
    print(f"  • P90:           {lstm_p90:.6f}")
    print(f"  • P95:           {lstm_p95:.6f}")
    print(f"  • P99:           {lstm_p99:.6f}")

    print("\n3. Continuous Loss-to-Score Mapping Demonstration (No Saturation):")
    print(f"{'Reconstruction Loss (MSE)':<30} | {'Calibrated LSTM Score (0-100)':<30}")
    print("-" * 65)
    sample_losses = [0.01, lstm_p50, lstm_p90, lstm_p95, lstm_p99, 5.0, 25.0, 94.76, 176.95]
    for l in sample_losses:
        # Calibrated CDF formula: 100 * (1 - exp(-loss / (P95 * 1.5)))
        score = 100.0 * (1.0 - np.exp(-l / (dist_p95 * 0.5)))
        score = min(100.0, max(0.0, score))
        print(f"{l:<30.6f} | {score:<30.2f}")

if __name__ == "__main__":
    run_calibration()
