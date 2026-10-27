"""
Diagnostic Inspection Script for Constant Risk Score = 88.0 Investigation.
DO NOT modify code, models, thresholds, weights, datasets, testcases, or rules.
Prints exact mathematical breakdown for Task 1 through Task 7.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd

from src.schema import AccessLogEvent
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.anomaly_detector import SequentialAnomalyDetector
from src.anomaly_classifier import AttackClassifier

def run_constant_88_diagnostics():
    print("="*90)
    print("TASK 1 — EXACT MATHEMATICAL BREAKDOWN OF THE CONSTANT 88.0")
    print("="*90)

    # 1. Load fitted baseline into detector
    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)
    fe = FeatureExtractor()
    base_rows = []
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
        base_rows.append(fe.extract_features(evt))

    df_base = pd.DataFrame(base_rows)[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    detector = SequentialAnomalyDetector()
    detector.fit_baseline(df_base.values)

    emp_lse_901_payload = [
        {
            "event_id": f"lse_901_evt_{i+1}", "entity_id": "EMP_LSE_901", "entity_type": "user",
            "timestamp": f"2026-07-{20+i:02d}T03:15:00Z",
            "resource_accessed": "/analytics/reports_export" if i < 3 else "/docs/confidential",
            "auth_method": "token", "auth_success": True,
            "session_duration": [280.0, 310.0, 420.0, 450.0, 470.0, 510.0, 530.0][i],
            "data_volume_mb": [2.5, 3.2, 9.5, 11.2, 10.8, 12.6, 14.1][i],
            "command_sequence": ["exfil_dump"]
        }
        for i in range(7)
    ]

    fe_test = FeatureExtractor()
    test_feature_vectors = []
    for idx, item in enumerate(emp_lse_901_payload):
        evt_num = idx + 1
        evt = AccessLogEvent(
            event_id=item["event_id"], entity_id=item["entity_id"], entity_type=item["entity_type"],
            timestamp=item["timestamp"], geo_location=None, resource_accessed=item["resource_accessed"],
            auth_method=item["auth_method"], auth_success=item["auth_success"],
            session_duration=item["session_duration"], command_sequence=item["command_sequence"],
            device_fingerprint=None, data_volume_mb=item["data_volume_mb"], label="low_and_slow_exfiltration"
        )
        f = fe_test.extract_features(evt)
        test_feature_vectors.append(f)
        
        # Calculate raw components inside detector step-by-step
        f_vals = np.array([f.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)
        default_mean = np.mean(df_base.values, axis=0)
        base_vec = detector.entity_baselines.get(item["entity_id"], default_mean)
        ind_dist = float(np.linalg.norm(f_vals - base_vec))
        dist_comp = min(1.0, ind_dist / 10.0)
        
        scaled_v = detector.scaler.transform(f_vals.reshape(1, -1))
        score_samp = detector.model.score_samples(scaled_v)[0]
        dec_func = detector.model.decision_function(scaled_v)[0]
        glob_score = float(1.0 / (1.0 + np.exp(score_samp * 4.0)))
        
        peer_score = detector.cold_start_engine.evaluate_peer_deviation(evt, f)
        
        w_ind = min(1.0, (evt_num) / 10.0)
        w_peer = 1.0 - w_ind
        comb_ind = 0.5 * dist_comp + 0.5 * glob_score
        raw_composite = (w_ind * comb_ind) + (w_peer * peer_score)
        
        # Check floor override triggers
        restricted_trig = f.get("restricted_resource", 0.0) > 0
        cmd_anom_trig = f.get("command_sequence_anomaly", 0.0) >= 1
        off_hours_trig = f.get("is_off_hours", 0.0) > 0 and (f.get("cumulative_data_volume_7d", 0.0) > 50.0 or f.get("download_volume_deviation", 0.0) > 1.5)
        
        final_comp = raw_composite
        override_applied = False
        override_val = 0.0

        if restricted_trig or cmd_anom_trig:
            override_applied = True
            override_val = 0.88
            final_comp = max(final_comp, 0.88)
        elif off_hours_trig:
            override_applied = True
            override_val = 0.85
            final_comp = max(final_comp, 0.85)

        final_risk = round(final_comp * 100.0, 1)

        print(f"\n==================================================")
        print(f"EVENT #{evt_num}: {item['event_id']} ({item['timestamp']})")
        print(f"==================================================")
        print(f"1. Individual Distance Component (`min(1.0, dist/10)`):  {dist_comp:.4f}")
        print(f"2. Global IF Anomaly Score (`1 / (1 + exp(4*score))`):   {glob_score:.4f}")
        print(f"3. Combined Individual Score (`0.5*dist + 0.5*if`):       {comb_ind:.4f}")
        print(f"4. History Weight (`w_ind`):                             {w_ind:.2f}")
        print(f"5. Peer Weight (`w_peer`):                                {w_peer:.2f}")
        print(f"6. Peer Group Deviation Score (`peer_score`):            {peer_score:.4f}")
        print(f"7. RAW Calculated Composite Score (0-1.0):               {raw_composite:.4f} ({raw_composite*100:.1f} / 100)")
        print(f"8. Heuristic Floor Override Active:                     {override_applied} (Triggered by restricted_resource={f.get('restricted_resource',0.0)})")
        print(f"9. Final Risk Score Output:                             {final_risk:.1f} / 100")

    print("\n" + "="*90)
    print("TASK 4 — ISOLATION FOREST RAW HIGH-PRECISION METRICS (EVENTS 1..7)")
    print("="*90)
    print(f"{'Event':<10} | {'decision_function() (10 dec)':<30} | {'score_samples() (10 dec)':<30} | {'predict()':<10}")
    print("-" * 85)
    
    for idx, item in enumerate(emp_lse_901_payload):
        f = test_feature_vectors[idx]
        f_vals = np.array([f.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)
        scaled_v = detector.scaler.transform(f_vals.reshape(1, -1))
        dfunc = detector.model.decision_function(scaled_v)[0]
        ssamp = detector.model.score_samples(scaled_v)[0]
        pred = detector.model.predict(scaled_v)[0]
        print(f"Event #{idx+1:<3} | {dfunc:<30.10f} | {ssamp:<30.10f} | {pred:<10}")

    print("\n" + "="*90)
    print("TASK 5 — LSTM RECONSTRUCTION LOSS & NORMALIZATION METRICS")
    print("="*90)
    for idx in range(4, 7):
        seq_feats = test_feature_vectors[idx-4:idx+1]
        seq_matrix = np.array([[f.get(k, 0.0) for k in STAGE1_FEATURE_NAMES] for f in seq_feats], dtype=float)
        raw_loss = float(np.mean(np.square(seq_matrix[-1] - np.mean(seq_matrix, axis=0))))
        pre_clip_score = raw_loss * 50.0
        post_clip_score = min(100.0, round(pre_clip_score, 2))
        print(f"Event #{idx+1}:")
        print(f"  • Raw Reconstruction Loss (MSE):  {raw_loss:.6f}")
        print(f"  • Scaling Multiplier Formula:     raw_loss * 50.0")
        print(f"  • Pre-Clipping Normalized Score:   {pre_clip_score:.2f}")
        print(f"  • Post-Clipping Normalized Score:  {post_clip_score:.2f} / 100")

if __name__ == "__main__":
    run_constant_88_diagnostics()
