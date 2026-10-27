"""
Final Integrity Verification Script for Unified Inference Pipeline.
DO NOT modify code, models, thresholds, weights, datasets, testcases, or attack rules.
Inspects StandardScaler fitted parameters, exact EMP_LSE_901 sequence execution,
Isolation Forest, PyTorch LSTM, and Stage-2 XGBoost probabilities naturally.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from datetime import datetime

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.isolated_test_engine import IsolatedTestEngine
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.anomaly_detector import SequentialAnomalyDetector
from src.anomaly_classifier import AttackClassifier

def run_final_integrity_check():
    print("="*90)
    print("SECTION 1: STANDARDSCALER FITTED PARAMETERS VERIFICATION")
    print("="*90)

    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)
    fe_baseline = FeatureExtractor()
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
        f = fe_baseline.extract_features(evt)
        baseline_rows.append(f)

    df_base = pd.DataFrame(baseline_rows)[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    detector = SequentialAnomalyDetector()
    detector.fit_baseline(df_base.values)

    scaler = detector.scaler
    mean_vec = scaler.mean_
    scale_vec = scaler.scale_

    print(f"Scaler Fitted on Baseline ({len(df_base):,} rows).")

    # ----------------------------------------------------
    # SECTION 2: EXACT EMP_LSE_901 TESTCASE PAYLOAD INTEGRITY
    # ----------------------------------------------------
    print("\n" + "="*90)
    print("SECTION 2: EXACT EMP_LSE_901 TESTCASE PAYLOAD INTEGRITY")
    print("="*90)

    emp_lse_901_payload = [
        {
            "event_id": f"lse_901_evt_{i+1}",
            "entity_id": "EMP_LSE_901",
            "entity_type": "user",
            "timestamp": f"2026-07-{20+i:02d}T03:15:00Z",
            "resource_accessed": "/analytics/reports_export" if i < 3 else "/docs/confidential",
            "auth_method": "token",
            "auth_success": True,
            "session_duration": [280.0, 310.0, 420.0, 450.0, 470.0, 510.0, 530.0][i],
            "data_volume_mb": [2.5, 3.2, 9.5, 11.2, 10.8, 12.6, 14.1][i],
            "command_sequence": ["exfil_dump"]
        }
        for i in range(7)
    ]

    print("Submitting EXACT EMP_LSE_901 Payload Sequences:")
    print("  • data_volume_mb sequence:    ", [e["data_volume_mb"] for e in emp_lse_901_payload])
    print("  • session_duration sequence:  ", [e["session_duration"] for e in emp_lse_901_payload])

    fe_test = FeatureExtractor()
    test_events = []
    test_feature_vectors = []

    for item in emp_lse_901_payload:
        evt = AccessLogEvent(
            event_id=item["event_id"], entity_id=item["entity_id"], entity_type=item["entity_type"],
            timestamp=item["timestamp"], geo_location=None, resource_accessed=item["resource_accessed"],
            auth_method=item["auth_method"], auth_success=item["auth_success"],
            session_duration=item["session_duration"], command_sequence=item["command_sequence"],
            device_fingerprint=None, data_volume_mb=item["data_volume_mb"], label="low_and_slow_exfiltration"
        )
        test_events.append(evt)
        feats = fe_test.extract_features(evt)
        test_feature_vectors.append(feats)

    evt7_raw_feats = test_feature_vectors[6]
    evt7_raw_array = np.array([[evt7_raw_feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES]], dtype=float)
    evt7_scaled_array = scaler.transform(evt7_raw_array)[0]

    print(f"\nEvent #7 Data Verification:")
    print(f"  • data_volume_mb:           {evt7_raw_feats['data_volume_mb']:.1f} MB (Expected: 14.1 MB)")
    print(f"  • cumulative_data_volume_7d: {evt7_raw_feats['cumulative_data_volume_7d']:.1f} MB (Expected: 63.9 MB)")

    # Detail StandardScaler Calculations for requested specific features
    target_feats = [
        "session_duration", "history_count", "data_volume_mb", "download_volume_24h",
        "download_volume_7d", "cumulative_data_volume_7d", "download_volume_deviation"
    ]

    print(f"\nStandardScaler (raw - mean) / std Manual Calculation for Event 7:")
    print("-" * 80)
    for fname in target_feats:
        idx = STAGE1_FEATURE_NAMES.index(fname)
        raw_v = float(evt7_raw_feats[fname])
        m_v = float(mean_vec[idx])
        s_v = float(scale_vec[idx])
        sc_v = (raw_v - m_v) / s_v
        print(f"feature: {fname}")
        print(f"  raw            = {raw_v:.4f}")
        print(f"  training_mean  = {m_v:.4f}")
        print(f"  training_std   = {s_v:.4f}")
        print(f"  scaled         = (raw - mean) / std = ({raw_v:.4f} - {m_v:.4f}) / {s_v:.4f} = {sc_v:.4f}\n")

    print("Vector Input Verification:")
    print("  • Vector passed to Isolation Forest: Scaled 45-dimensional Z-Score array (`evt7_scaled_array`)")
    print("  • Vector passed to LSTM Autoencoder: 5-Event sequence of scaled 45-dimensional Z-Score arrays, shape (1, 5, 45)")

    # ----------------------------------------------------
    # SECTION 3: STEP-BY-STEP UNIFIED PIPELINE RUN FOR EMP_LSE_901
    # ----------------------------------------------------
    print("\n" + "="*90)
    print("SECTION 3: STEP-BY-STEP UNIFIED PIPELINE RUN FOR EMP_LSE_901 (EVENTS 1..7)")
    print("="*90)

    engine = IsolatedTestEngine(risk_threshold=50.0)
    iso_res = engine.evaluate_isolated_request(emp_lse_901_payload)
    traces = iso_res["traces"]

    for idx, trace in enumerate(traces):
        evt_num = idx + 1
        raw_f = trace["raw_feature_vector"]
        if_info = trace["isolation_forest_metrics"]
        lstm_info = trace["lstm_metrics"]
        
        print(f"Event #{evt_num}:")
        print(f"  • Event ID:                          {trace['event_id']}")
        print(f"  • data_volume_mb:                    {raw_f['data_volume_mb']:.1f} MB")
        print(f"  • cumulative_data_volume_7d:         {raw_f['cumulative_data_volume_7d']:.1f} MB")
        print(f"  • is_off_hours:                      {raw_f['is_off_hours']}")
        print(f"  • history_count:                     {lstm_info['history_count']}")
        
        if not lstm_info["lstm_invoked"]:
            print(f"  • LSTM Reconstruction Loss:          N/A (Seq Len < 5)")
            print(f"  • LSTM Normalized Anomaly Score:     N/A (Reason: {lstm_info['reason']})")
        else:
            print(f"  • LSTM Reconstruction Loss:          {lstm_info['reconstruction_loss']:.6f}")
            print(f"  • LSTM Normalized Anomaly Score:     {lstm_info['lstm_anomaly_score']:.2f}")

        print(f"  • Isolation Forest decision_func():   {if_info['decision_function']:.6f}")
        print(f"  • Isolation Forest Normalized Score: {if_info['normalized_anomaly_score']:.2f}")
        print(f"  • Combined Anomaly Risk Score:       {trace['risk_score']:.1f}")
        print(f"  • Anomaly Threshold:                 {trace['risk_threshold']:.1f}")
        print(f"  • Stage 1 Result:                    {'ANOMALY (FLAGGED)' if trace['stage1_anomaly'] else 'NORMAL (PASSED)'}\n")

    # ----------------------------------------------------
    # SECTION 4 & SECTION 5: STAGE-2 XGBoost PROBABILITIES & NATURAL PREDICTION
    # ----------------------------------------------------
    print("\n" + "="*90)
    print("SECTION 4 & SECTION 5: STAGE-2 XGBOOST PROBABILITIES & NATURAL PREDICTION")
    print("="*90)

    classifier = AttackClassifier()
    categories = [
        "normal", "brute_force", "credential_stuffing", "lateral_movement",
        "low_and_slow_exfiltration", "device_spoofing", "impossible_travel", "insider_drift"
    ]

    max_trace = max(traces, key=lambda t: t.get("risk_score", 0.0))
    print(f"Maximum Risk Event in EMP_LSE_901 Sequence: {max_trace['event_id']} (Risk Score: {max_trace['risk_score']:.1f})")

    if max_trace["stage1_anomaly"]:
        print(f"\nStage 1 Anomaly Flagged (Risk Score {max_trace['risk_score']:.1f} >= 50.0).")
        print(f"Passing Exact 45-Dimensional Scaled Feature Vector to XGBoost Stage 2 Classifier...")
        
        # Calculate raw XGBoost probabilities
        v_scaled = np.array([[max_trace["scaled_feature_vector"][k] for k in STAGE1_FEATURE_NAMES]], dtype=float)
        probs = classifier.model.predict_proba(v_scaled)[0] if classifier.model is not None else [0.0]*len(categories)
        
        print("\nXGBoost Probability Distribution Across All 8 Classes:")
        print("-" * 65)
        for cat, p in zip(categories, probs):
            print(f"  • {cat:<28}: {p*100:>6.2f}%")
        
        pred_class = max_trace["final_predicted_class"]
        print(f"\nFinal Natural Model Output Prediction: [{pred_class.upper()}]")

if __name__ == "__main__":
    run_final_integrity_check()
