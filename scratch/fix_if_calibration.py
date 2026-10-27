"""
Diagnostic Script for Isolation Forest Score Calibration.
Computes empirical distribution of score_samples() on Normal Validation vs Attack data.
Fixes 88.0 false positive mapping on normal inliers.
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

def inspect_if_distribution():
    print("="*90)
    print("ISOLATION FOREST RAW SCORE DISTRIBUTION & CALIBRATION FIX")
    print("="*90)

    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)
    fe = FeatureExtractor()

    # Train / Val Split
    total = len(df_raw)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_df = df_raw.iloc[:train_end]
    val_df = df_raw.iloc[train_end:val_end]
    test_df = df_raw.iloc[val_end:]

    # Extract Train features
    train_rows = []
    for _, r in train_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=None, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=None, data_volume_mb=vol, label=r["label"]
        )
        train_rows.append(fe.extract_features(evt))

    df_tr_matrix = pd.DataFrame(train_rows)[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    scaled_tr = scaler.fit_transform(df_tr_matrix.values)

    iso_model = IsolationForest(contamination=0.03, random_state=42)
    iso_model.fit(scaled_tr[train_df["label"] == "normal"])

    # Extract Normal Validation score_samples()
    val_normal_df = val_df[val_df["label"] == "normal"]
    val_norm_rows = []
    fe_val = FeatureExtractor()
    for _, r in val_normal_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=None, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=None, data_volume_mb=vol, label="normal"
        )
        val_norm_rows.append(fe_val.extract_features(evt))

    df_val_matrix = pd.DataFrame(val_norm_rows)[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled_val_norm = scaler.transform(df_val_matrix.values)
    val_scores = iso_model.score_samples(scaled_val_norm)
    val_dec = iso_model.decision_function(scaled_val_norm)

    # Extract Attack Test score_samples()
    test_attack_df = test_df[test_df["label"] != "normal"]
    test_att_rows = []
    fe_att = FeatureExtractor()
    for _, r in test_attack_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=None, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=None, data_volume_mb=vol, label=r["label"]
        )
        test_att_rows.append(fe_att.extract_features(evt))

    df_att_matrix = pd.DataFrame(test_att_rows)[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled_att = scaler.transform(df_att_matrix.values)
    att_scores = iso_model.score_samples(scaled_att)
    att_dec = iso_model.decision_function(scaled_att)

    print("\n1. Isolation Forest raw score_samples() Statistics:")
    print(f"  • Normal Validation  - Mean: {np.mean(val_scores):.6f}, Std: {np.std(val_scores):.6f}, Min: {np.min(val_scores):.6f}, Max: {np.max(val_scores):.6f}")
    print(f"  • Attack Test Set    - Mean: {np.mean(att_scores):.6f}, Std: {np.std(att_scores):.6f}, Min: {np.min(att_scores):.6f}, Max: {np.max(att_scores):.6f}")

    print("\n2. Normal Validation Percentiles for score_samples():")
    print(f"  • P50 (Median): {np.percentile(val_scores, 50):.6f}")
    print(f"  • P10:          {np.percentile(val_scores, 10):.6f}")
    print(f"  • P5:           {np.percentile(val_scores, 5):.6f}")
    print(f"  • P1:           {np.percentile(val_scores, 1):.6f}")

    mu_val = np.mean(val_scores)
    std_val = np.std(val_scores)

    def calibrate_if_score(raw_score):
        # Calibrated Sigmoid CDF mapping centered on normal mean mu_val
        z = (mu_val - raw_score) / (std_val * 2.0)
        return float(1.0 / (1.0 + np.exp(-z)))

    print("\n3. Calibrated IF Score Mapping Demonstration (0.0 to 1.0):")
    print(f"{'Data Group':<25} | {'Raw score_samples()':<22} | {'Old Flawed Formula':<20} | {'New Calibrated IF Score':<22}")
    print("-" * 95)
    for name, score in [("Normal Median (P50)", np.percentile(val_scores, 50)),
                        ("Normal P5", np.percentile(val_scores, 5)),
                        ("Normal P1", np.percentile(val_scores, 1)),
                        ("Attack Median", np.percentile(att_scores, 50)),
                        ("Severe Attack Min", np.min(att_scores))]:
        old_val = float(1.0 / (1.0 + np.exp(score * 4.0))) * 100.0
        new_val = calibrate_if_score(score) * 100.0
        print(f"{name:<25} | {score:<22.6f} | {old_val:<20.2f} | {new_val:<22.2f}")

if __name__ == "__main__":
    inspect_if_distribution()
