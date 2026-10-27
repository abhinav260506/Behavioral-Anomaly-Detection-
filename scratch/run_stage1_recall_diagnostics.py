"""
Comprehensive Stage-1 Recall & End-to-End Diagnostic Suite.
DO NOT modify code, thresholds, rules, testcases, or retrain.
Evaluates Tasks 1 through 8 with exact empirical metrics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, precision_recall_curve, auc
import xgboost as xgb

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.isolated_test_engine import IsolatedTestEngine
from src.anomaly_detector import SequentialAnomalyDetector
from src.anomaly_classifier import AttackClassifier

def run_diagnostics():
    print("="*90)
    print("TASK 1 & TASK 2 — RECREATING 8 REGRESSION TEST CASES WITH FULL FEATURE TRACES")
    print("="*90)

    test_cases = [
        ("Test 1: Brute Force Burst", [
            {"event_id": f"bf_{i}", "entity_id": "u_bf", "timestamp": f"2026-07-25T10:00:{i*5:02d}Z",
             "geo_location": {"ip": "1.1.1.1", "city": "NY", "country": "US", "latitude": 40.7, "longitude": -74.0},
             "resource_accessed": "/api/v1/auth/login", "auth_success": False, "session_duration": 1}
            for i in range(5)
        ], "brute_force", ["rolling_failures_5m", "rolling_failures_1h", "auth_failed", "ip_diversity_1h"]),
        
        ("Test 2: Impossible Travel", [
            {"event_id": "it_1", "entity_id": "u_it", "timestamp": "2026-07-25T10:00:00Z",
             "geo_location": {"ip": "1.1.1.1", "city": "New York", "country": "US", "latitude": 40.71, "longitude": -74.00},
             "resource_accessed": "/doc", "auth_success": True, "session_duration": 120},
            {"event_id": "it_2", "entity_id": "u_it", "timestamp": "2026-07-25T10:15:00Z",
             "geo_location": {"ip": "2.2.2.2", "city": "Tokyo", "country": "Japan", "latitude": 35.67, "longitude": 139.65},
             "resource_accessed": "/doc", "auth_success": True, "session_duration": 120}
        ], "impossible_travel", ["geo_velocity_kmh", "ip_diversity_1h", "session_duration"]),

        ("Test 3: Credential Stuffing", [
            {"event_id": f"cs_{i}", "entity_id": f"user_cs_{i}", "timestamp": f"2026-07-25T12:00:{i*5:02d}Z",
             "geo_location": {"ip": "198.51.100.44", "city": "London", "country": "UK", "latitude": 51.5, "longitude": -0.1},
             "resource_accessed": "/api/v1/auth/login", "auth_success": False, "session_duration": 1}
            for i in range(4)
        ], "credential_stuffing", ["rolling_failures_5m", "rolling_failures_1h", "ip_diversity_1h", "auth_failed"]),

        ("Test 4: Lateral Movement", [
            {"event_id": "lm_1", "entity_id": "u_lm", "timestamp": "2026-07-25T14:00:00Z",
             "geo_location": {"ip": "10.0.0.5", "city": "Chicago", "country": "US", "latitude": 41.8, "longitude": -87.6},
             "resource_accessed": "/admin/vault_keys", "auth_success": True, "command_sequence": ["mimikatz.exe"], "session_duration": 300}
        ], "lateral_movement", ["new_resource", "resource_novelty", "resource_breadth_change", "restricted_resource", "new_entity_resource_edge", "rare_entity_resource_edge", "command_sequence_anomaly"]),

        ("Test 5: Device Spoofing", [
            {"event_id": "ds_1", "entity_id": "u_ds", "timestamp": "2026-07-25T15:00:00Z",
             "geo_location": {"ip": "10.0.0.6", "city": "Dallas", "country": "US", "latitude": 32.7, "longitude": -96.8},
             "resource_accessed": "/doc", "auth_success": True, "device_fingerprint": {"os_version": "Win11", "mac_address": "AA:BB:CC:DD:EE:FF", "protocol": "TLSv1.3", "browser_or_agent": "Chrome/120"}},
            {"event_id": "ds_2", "entity_id": "u_ds", "timestamp": "2026-07-25T15:10:00Z",
             "geo_location": {"ip": "10.0.0.6", "city": "Dallas", "country": "US", "latitude": 32.7, "longitude": -96.8},
             "resource_accessed": "/doc", "auth_success": True, "device_fingerprint": {"os_version": "Linux", "mac_address": "FF:EE:DD:CC:BB:AA", "protocol": "SSHv2", "browser_or_agent": "Agent"}}
        ], "device_spoofing", ["fingerprint_known", "fingerprint_mismatch", "os_changed", "mac_changed", "protocol_changed", "fingerprint_change_count_24h"]),

        ("Test 6: Low-and-Slow Exfiltration", [
            {"event_id": f"ex_{i}", "entity_id": "u_ex", "timestamp": f"2026-07-25T03:15:{i*5:02d}Z",
             "geo_location": {"ip": "10.0.0.7", "city": "Denver", "country": "US", "latitude": 39.7, "longitude": -104.9},
             "resource_accessed": "/exfil/bulk_export", "auth_success": True, "data_volume_mb": 250.0, "command_sequence": ["cat /data/part0"], "session_duration": 45}
            for i in range(3)
        ], "low_and_slow_exfiltration", ["is_off_hours", "data_volume_mb", "download_volume_24h", "download_volume_7d", "cumulative_data_volume_7d", "download_volume_deviation", "off_hours_access_count_7d", "resource_breadth_7d"]),

        ("Test 7: Insider Drift", [
            {"event_id": "id_1", "entity_id": "u_id", "timestamp": "2026-07-25T16:00:00Z",
             "geo_location": {"ip": "10.0.0.8", "city": "Seattle", "country": "US", "latitude": 47.6, "longitude": -122.3},
             "resource_accessed": "/api/v1/new_project_alpha", "auth_success": True, "command_sequence": ["git clone alpha"], "session_duration": 600}
        ], "insider_drift", ["new_resource", "resource_novelty", "resource_breadth_7d", "resource_breadth_30d", "new_resources_7d", "new_departments_7d", "sensitive_resource_ratio", "historical_resource_overlap", "privilege_expansion_rate", "behavior_drift_score"]),

        ("Test 8: Benign Normal Access", [
            {"event_id": "norm_1", "entity_id": "u_norm", "timestamp": "2026-07-25T11:00:00Z",
             "geo_location": {"ip": "10.0.0.9", "city": "Boston", "country": "US", "latitude": 42.3, "longitude": -71.0},
             "resource_accessed": "/dashboard", "auth_success": True, "command_sequence": [], "session_duration": 120}
        ], "normal", ["session_duration", "data_volume_mb", "is_off_hours"])
    ]

    engine = IsolatedTestEngine(risk_threshold=50.0)

    for name, payload, expected, special_keys in test_cases:
        print(f"\n==================================================")
        print(f"CASE: {name} (Expected: {expected})")
        print(f"==================================================")
        res = engine.evaluate_isolated_request(payload)
        traces = res["traces"]
        last_t = traces[-1]
        raw_f = last_t["raw_feature_vector"]
        mscores = last_t["model_scores"]

        print(f"Relevant Feature Values (Final Event):")
        for k in special_keys:
            print(f"  • {k:<35}: {raw_f.get(k, 'N/A')}")

        print(f"\nScore Components breakdown for Final Event:")
        print(f"  • Behavioral L2 Dist Score:      {mscores['behavioral_distance_score']:.2f}")
        print(f"  • Isolation Forest Score:        {mscores['isolation_forest_score']:.2f}")
        print(f"  • PyTorch LSTM Score:            {mscores['lstm_sequence_score']}")
        print(f"  • Peer Group Score:              {mscores['peer_group_score']:.2f}")
        print(f"  • History Count & Weights:       history_count={last_t['lstm_metrics']['history_count']}, w_ind={(last_t['lstm_metrics']['history_count'])/10.0:.2f}, w_peer={1.0 - (last_t['lstm_metrics']['history_count'])/10.0:.2f}")
        print(f"  • Final Stage-1 Risk Score:      {last_t['risk_score']:.1f} / 100.0 (Threshold = 50.0)")
        print(f"  • Final Predicted Class:         {last_t['final_predicted_class']} (Status: {'PASS' if last_t['final_predicted_class'] == expected else 'FAIL'})")

    print("\n" + "="*90)
    print("TASK 4 — CREDENTIAL STUFFING CROSS-ENTITY STATE CHECK")
    print("="*90)
    fe_check = FeatureExtractor()
    print(f"• fe_check.ip_history tracked: {hasattr(fe_check, 'ip_history')} (Global IP history dictionary initialized)")
    print(f"• fe_check.entity_history tracked: {hasattr(fe_check, 'entity_history')}")

    # Test multi-entity single IP failures
    cs_events = [
        AccessLogEvent(event_id=f"cs_evt_{i}", entity_id=f"user_cs_{i}", entity_type="user", timestamp=f"2026-07-25T12:00:{i*5:02d}Z", geo_location=GeoLocation(ip="198.51.100.44", country="UK", city="London", latitude=51.5, longitude=-0.1), resource_accessed="/login", auth_method="password", auth_success=False, session_duration=1.0, command_sequence=[], device_fingerprint=None, data_volume_mb=0.1, label="credential_stuffing")
        for i in range(4)
    ]

    print("\nTracing `ip_diversity_1h` across 4 distinct users from single IP (198.51.100.44):")
    for i, e in enumerate(cs_events):
        f_cs = fe_check.extract_features(e)
        print(f"  • Event #{i+1} ({e.entity_id}): rolling_failures_5m={f_cs['rolling_failures_5m']}, ip_diversity_1h={f_cs['ip_diversity_1h']}, auth_failed={f_cs['auth_failed']}")

    print("\n" + "="*90)
    print("TASK 5 & TASK 6 — COMPREHENSIVE END-TO-END 2-STAGE METRICS & CLASS SUPPORT")
    print("="*90)

    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)
    
    total_events = len(df_raw)
    train_end = int(total_events * 0.70)
    val_end = int(total_events * 0.85)

    train_df = df_raw.iloc[:train_end]
    val_df = df_raw.iloc[train_end:val_end]
    test_df = df_raw.iloc[val_end:]

    categories = [
        "normal", "brute_force", "credential_stuffing", "lateral_movement",
        "low_and_slow_exfiltration", "device_spoofing", "impossible_travel", "insider_drift"
    ]
    label2id = {c: i for i, c in enumerate(categories)}

    print(f"{'Attack Category':<30} | {'Train (70%)':<12} | {'Val (15%)':<12} | {'Test (15%)':<12} | {'Support Status':<20}")
    print("-" * 90)
    for cat in categories:
        tr_c = (train_df["label"] == cat).sum()
        va_c = (val_df["label"] == cat).sum()
        te_c = (test_df["label"] == cat).sum()
        status = "WEAK (<30 examples)" if (va_c < 30 or te_c < 30) else "ROBUST (>=30 examples)"
        print(f"{cat:<30} | {tr_c:<12} | {va_c:<12} | {te_c:<12} | {status:<20}")

    # Run complete 2-stage pipeline on untouched test set
    fe_prod = FeatureExtractor()
    detector_prod = SequentialAnomalyDetector()
    classifier_prod = AttackClassifier()

    # Fit detector baseline on train set
    tr_feats = [fe_prod.extract_features(
        AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=GeoLocation(ip=r["ip"], country=r["country"], city=r["city"], latitude=float(r["latitude"]), longitude=float(r["longitude"])),
            resource_accessed=r["resource_accessed"], auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=[c.strip() for c in str(r["commands"]).split(";") if c.strip()],
            device_fingerprint=DeviceFingerprint(os_version=r["os_version"], mac_address=r["mac_address"], protocol=r["protocol"], browser_or_agent=r["browser_agent"]),
            data_volume_mb=float(r.get("data_volume_mb", 0.0)), label=r["label"]
        )
    ) for _, r in train_df.iterrows()]

    df_tr_matrix = pd.DataFrame(tr_feats)[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    detector_prod.fit_baseline(df_tr_matrix.values)

    # Evaluate on Untouched Test Set
    stage1_binary_true = [] # 1 if attack, 0 if normal
    stage1_binary_pred = [] # 1 if risk >= 50.0, 0 if risk < 50.0
    end_to_end_preds = []
    end_to_end_true = []

    print("\nRunning COMPLETE 2-Stage Pipeline on Untouched Test Set (4,176 events)...")
    fe_test_eval = FeatureExtractor()

    for _, r in test_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        geo = GeoLocation(ip=r["ip"], country=r["country"], city=r["city"], latitude=float(r["latitude"]), longitude=float(r["longitude"]))
        fp = DeviceFingerprint(os_version=r["os_version"], mac_address=r["mac_address"], protocol=r["protocol"], browser_or_agent=r["browser_agent"])
        vol = float(r.get("data_volume_mb", 0.0))
        true_label = r["label"]

        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=geo, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=fp, data_volume_mb=vol, label=true_label
        )

        feats = fe_test_eval.extract_features(evt)
        risk_score, is_anom = detector_prod.predict_event(r["entity_id"], feats, evt)

        # Stage 1 binary tracking
        is_attack_true = 0 if true_label == "normal" else 1
        is_anom_pred = 1 if is_anom else 0
        stage1_binary_true.append(is_attack_true)
        stage1_binary_pred.append(is_anom_pred)

        # End-to-end decision: IF Stage 1 says normal, final prediction MUST be normal!
        if is_anom:
            pred_class = classifier_prod.classify_anomaly(evt, feats, risk_score)
        else:
            pred_class = "normal"

        end_to_end_preds.append(pred_class)
        end_to_end_true.append(true_label)

    # Calculate True Stage 1 Binary Metrics
    p_s1, r_s1, f1_s1, _ = precision_recall_fscore_support(stage1_binary_true, stage1_binary_pred, average="binary", zero_division=0)
    cm_s1 = confusion_matrix(stage1_binary_true, stage1_binary_pred)
    tn_s1, fp_s1, fn_s1, tp_s1 = cm_s1.ravel()
    fpr_s1 = (fp_s1 / float(fp_s1 + tn_s1)) * 100.0 if (fp_s1 + tn_s1) > 0 else 0.0

    print("\n" + "="*80)
    print("STAGE-1 ANOMALY DETECTOR BINARY METRICS (REAL UNTOUCHED TEST SET)")
    print("="*80)
    print(f"• Stage-1 Binary Precision: {p_s1*100:.2f}%")
    print(f"• Stage-1 Binary Recall:    {r_s1*100:.2f}%")
    print(f"• Stage-1 Binary F1-Score:  {f1_s1*100:.2f}%")
    print(f"• Stage-1 False Positive Rate (Benign Normal): {fpr_s1:.2f}% ({fp_s1} false alarms out of {fp_s1+tn_s1})")
    print(f"• Stage-1 True Attack Recall:                   {tp_s1 / float(tp_s1 + fn_s1) * 100:.2f}% ({tp_s1} attacks flagged out of {tp_s1+fn_s1})")

    # End-to-End Metrics
    report_e2e = classification_report(end_to_end_true, end_to_end_preds, target_names=categories, output_dict=True, zero_division=0)
    cm_e2e = confusion_matrix([label2id[c] for c in end_to_end_true], [label2id[c] for c in end_to_end_preds])

    print("\n" + "="*80)
    print("END-TO-END COMPLETE 2-STAGE PIPELINE METRICS (REAL UNTOUCHED TEST SET)")
    print("="*80)
    print(f"• End-to-End Accuracy:   {report_e2e['accuracy']*100:.2f}%")
    print(f"• End-to-End Macro Precision: {report_e2e['macro avg']['precision']*100:.2f}%")
    print(f"• End-to-End Macro Recall:    {report_e2e['macro avg']['recall']*100:.2f}%")
    print(f"• End-to-End Macro F1-Score:  {report_e2e['macro avg']['f1-score']*100:.2f}%")

    print("\nPer-Class End-to-End Precision / Recall / F1-Score / Support:")
    print(f"{'Category':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)
    for cat in categories:
        m = report_e2e[cat]
        print(f"{cat:<28} | {m['precision']*100:>9.2f}% | {m['recall']*100:>9.2f}% | {m['f1-score']*100:>9.2f}% | {m['support']:<8}")

    print("\nEnd-to-End Confusion Matrix:")
    print("Rows: True Label | Columns: Predicted Label")
    print(f"{'':<25}" + "".join([f"{c[:8]:>10}" for c in categories]))
    for i, cat in enumerate(categories):
        print(f"{cat:<25}" + "".join([f"{cm_e2e[i, j]:>10}" for j in range(len(categories))]))

    # ----------------------------------------------------
    # TASK 7 — LOW-AND-SLOW DIFFERENCE (EMP_LSE_901 vs Test 6)
    # ----------------------------------------------------
    print("\n" + "="*90)
    print("TASK 7 — LOW-AND-SLOW FEATURE COMPARISON (EMP_LSE_901 vs REGRESSION TEST 6)")
    print("="*90)
    lse_901_payload = [
        {"event_id": f"lse_901_evt_{i+1}", "entity_id": "EMP_LSE_901", "timestamp": f"2026-07-{20+i:02d}T03:15:00Z",
         "resource_accessed": "/analytics/reports_export" if i < 3 else "/docs/confidential",
         "auth_method": "token", "auth_success": True, "session_duration": 450.0,
         "data_volume_mb": [2.5, 3.2, 9.5, 11.2, 10.8, 12.6, 14.1][i], "command_sequence": ["exfil_dump"]}
        for i in range(7)
    ]
    reg_test6_payload = test_cases[5][1]

    res_901_comp = engine.evaluate_isolated_request(lse_901_payload)
    res_reg6_comp = engine.evaluate_isolated_request(reg_test6_payload)

    t901_last = res_901_comp["traces"][-1]["raw_feature_vector"]
    treg6_last = res_reg6_comp["traces"][-1]["raw_feature_vector"]

    print(f"{'Feature Name':<35} | {'EMP_LSE_901 (7 Days)':<22} | {'Regression Test 6 (15 Secs)':<25}")
    print("-" * 85)
    for k in ["is_off_hours", "data_volume_mb", "download_volume_24h", "download_volume_7d", "cumulative_data_volume_7d", "download_volume_deviation", "off_hours_access_count_7d", "resource_breadth_7d", "history_count"]:
        v1 = t901_last.get(k, "N/A")
        v2 = treg6_last.get(k, "N/A")
        print(f"{k:<35} | {str(v1):<22} | {str(v2):<25}")

    print(f"\nResulting Risk Scores:")
    print(f"  • EMP_LSE_901 Final Risk Score:      {res_901_comp['traces'][-1]['risk_score']:.1f} / 100.0  (7 multi-day events, cum_vol=63.9MB)")
    print(f"  • Regression Test 6 Final Risk Score: {res_reg6_comp['traces'][-1]['risk_score']:.1f} / 100.0  (3 burst events in 15 seconds, cum_vol=750MB)")

if __name__ == "__main__":
    run_diagnostics()
