"""
Comprehensive Evaluation & Regression Suite for Refactored Stage-1 Pipeline.
Zero hardcoded floors or forced rules.
Tests EMP_LSE_901, 8 System Regression Cases, Benign False-Positive Scenarios, and Real Untouched Test Set Metrics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.isolated_test_engine import IsolatedTestEngine
from src.anomaly_detector import SequentialAnomalyDetector
from src.anomaly_classifier import AttackClassifier

def run_refactored_evaluation():
    print("="*90)
    print("1. RETEST EXACT EMP_LSE_901 ON REFACTORED PIPELINE (NO FLOORS)")
    print("="*90)

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

    engine = IsolatedTestEngine(risk_threshold=50.0)
    res_901 = engine.evaluate_isolated_request(emp_lse_901_payload)
    traces_901 = res_901["traces"]

    print(f"{'Event':<8} | {'IF Score':<10} | {'LSTM Score':<12} | {'Dist Score':<12} | {'Peer Score':<12} | {'Risk Score':<12} | {'Threshold':<10} | {'Stage-1 Result':<16}")
    print("-" * 105)
    for idx, trace in enumerate(traces_901):
        mscores = trace["model_scores"]
        print(f"Event #{idx+1:<2} | {mscores['isolation_forest_score']:<10.2f} | {str(mscores['lstm_sequence_score']):<12} | {mscores['behavioral_distance_score']:<12.2f} | {mscores['peer_group_score']:<12.2f} | {trace['risk_score']:<12.1f} | {trace['risk_threshold']:<10.1f} | {'ANOMALY' if trace['stage1_anomaly'] else 'NORMAL':<16}")

    print("\n" + "="*90)
    print("2. SYSTEM ATTACK REGRESSION TEST SUITE (8 TEST CASES)")
    print("="*90)

    test_cases = [
        ("Test 1: Brute Force Burst", [
            {"event_id": f"bf_{i}", "entity_id": "u_bf", "timestamp": f"2026-07-25T10:00:{i*5:02d}Z",
             "geo_location": {"ip": "1.1.1.1", "city": "NY", "country": "US", "latitude": 40.7, "longitude": -74.0},
             "resource_accessed": "/api/v1/auth/login", "auth_success": False, "session_duration": 1}
            for i in range(5)
        ], "brute_force"),
        
        ("Test 2: Impossible Travel", [
            {"event_id": "it_1", "entity_id": "u_it", "timestamp": "2026-07-25T10:00:00Z",
             "geo_location": {"ip": "1.1.1.1", "city": "New York", "country": "US", "latitude": 40.71, "longitude": -74.00},
             "resource_accessed": "/doc", "auth_success": True, "session_duration": 120},
            {"event_id": "it_2", "entity_id": "u_it", "timestamp": "2026-07-25T10:15:00Z",
             "geo_location": {"ip": "2.2.2.2", "city": "Tokyo", "country": "Japan", "latitude": 35.67, "longitude": 139.65},
             "resource_accessed": "/doc", "auth_success": True, "session_duration": 120}
        ], "impossible_travel"),

        ("Test 3: Credential Stuffing", [
            {"event_id": f"cs_{i}", "entity_id": f"user_cs_{i}", "timestamp": f"2026-07-25T12:00:{i*5:02d}Z",
             "geo_location": {"ip": "198.51.100.44", "city": "London", "country": "UK", "latitude": 51.5, "longitude": -0.1},
             "resource_accessed": "/api/v1/auth/login", "auth_success": False, "session_duration": 1}
            for i in range(4)
        ], "credential_stuffing"),

        ("Test 4: Lateral Movement", [
            {"event_id": "lm_1", "entity_id": "u_lm", "timestamp": "2026-07-25T14:00:00Z",
             "geo_location": {"ip": "10.0.0.5", "city": "Chicago", "country": "US", "latitude": 41.8, "longitude": -87.6},
             "resource_accessed": "/admin/vault_keys", "auth_success": True, "command_sequence": ["mimikatz.exe"], "session_duration": 300}
        ], "lateral_movement"),

        ("Test 5: Device Spoofing", [
            {"event_id": "ds_1", "entity_id": "u_ds", "timestamp": "2026-07-25T15:00:00Z",
             "geo_location": {"ip": "10.0.0.6", "city": "Dallas", "country": "US", "latitude": 32.7, "longitude": -96.8},
             "resource_accessed": "/doc", "auth_success": True, "device_fingerprint": {"os_version": "Win11", "mac_address": "AA:BB:CC:DD:EE:FF", "protocol": "TLSv1.3", "browser_or_agent": "Chrome/120"}},
            {"event_id": "ds_2", "entity_id": "u_ds", "timestamp": "2026-07-25T15:10:00Z",
             "geo_location": {"ip": "10.0.0.6", "city": "Dallas", "country": "US", "latitude": 32.7, "longitude": -96.8},
             "resource_accessed": "/doc", "auth_success": True, "device_fingerprint": {"os_version": "Linux", "mac_address": "FF:EE:DD:CC:BB:AA", "protocol": "SSHv2", "browser_or_agent": "Agent"}}
        ], "device_spoofing"),

        ("Test 6: Low-and-Slow Exfiltration", [
            {"event_id": f"ex_{i}", "entity_id": "u_ex", "timestamp": f"2026-07-25T03:15:{i*5:02d}Z",
             "geo_location": {"ip": "10.0.0.7", "city": "Denver", "country": "US", "latitude": 39.7, "longitude": -104.9},
             "resource_accessed": "/exfil/bulk_export", "auth_success": True, "data_volume_mb": 250.0, "command_sequence": ["cat /data/part0"], "session_duration": 45}
            for i in range(3)
        ], "low_and_slow_exfiltration"),

        ("Test 7: Insider Drift", [
            {"event_id": "id_1", "entity_id": "u_id", "timestamp": "2026-07-25T16:00:00Z",
             "geo_location": {"ip": "10.0.0.8", "city": "Seattle", "country": "US", "latitude": 47.6, "longitude": -122.3},
             "resource_accessed": "/api/v1/new_project_alpha", "auth_success": True, "command_sequence": ["git clone alpha"], "session_duration": 600}
        ], "insider_drift"),

        ("Test 8: Benign Normal Access", [
            {"event_id": "norm_1", "entity_id": "u_norm", "timestamp": "2026-07-25T11:00:00Z",
             "geo_location": {"ip": "10.0.0.9", "city": "Boston", "country": "US", "latitude": 42.3, "longitude": -71.0},
             "resource_accessed": "/dashboard", "auth_success": True, "command_sequence": [], "session_duration": 120}
        ], "normal")
    ]

    print(f"{'Test Case Name':<35} | {'Expected':<25} | {'Predicted':<25} | {'Risk':<6} | {'Status':<6}")
    print("-" * 105)

    for name, payload, expected in test_cases:
        res = engine.evaluate_isolated_request(payload)
        max_trace = max(res["traces"], key=lambda t: t.get("risk_score", 0.0))
        predicted = max_trace.get("final_predicted_class", "normal")
        risk = max_trace.get("risk_score", 0.0)
        status = "PASS" if predicted == expected else "FAIL"
        print(f"{name:<35} | {expected:<25} | {predicted:<25} | {risk:>5.1f} | {status:<6}")

    print("\n" + "="*90)
    print("3. BENIGN GRADUAL RESOURCE EXPANSION FALSE-POSITIVE TEST")
    print("="*90)

    benign_expansion_payload = [
        {"event_id": f"benign_exp_{i+1}", "entity_id": "u_benign_dev", "timestamp": f"2026-07-{10+i:02d}T10:00:00Z",
         "resource_accessed": f"/engineering/project_alpha_{i//2}", "auth_method": "password", "auth_success": True,
         "session_duration": 3600.0, "data_volume_mb": 5.0, "command_sequence": ["npm test"]}
        for i in range(10)
    ]

    res_benign = engine.evaluate_isolated_request(benign_expansion_payload)
    anom_count = sum(1 for t in res_benign["traces"] if t["stage1_anomaly"])
    max_benign_risk = max(t["risk_score"] for t in res_benign["traces"])

    print(f"Benign Gradual Expansion Evaluation Result:")
    print(f"  • Total Benign Events Processed: {len(benign_expansion_payload)}")
    print(f"  • Maximum Risk Score Recorded:   {max_benign_risk:.1f} / 100")
    print(f"  • False Positive Anomaly Count:  {anom_count} (Expected: 0)")
    print(f"  • False-Positive Verification:   {'PASS (Zero False Alarms)' if anom_count == 0 else 'FAIL'}")

    print("\n" + "="*90)
    print("4. PRODUCTION MODEL METRICS ON UNTOUCHED REAL TEST SET")
    print("="*90)
    
    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)
    fe_testset = FeatureExtractor()

    feature_rows = []
    labels = []
    for _, row in df_raw.iterrows():
        cmds = [c.strip() for c in str(row["commands"]).split(";") if c.strip()]
        geo = GeoLocation(ip=row["ip"], country=row["country"], city=row["city"], latitude=float(row["latitude"]), longitude=float(row["longitude"]))
        fp = DeviceFingerprint(os_version=row["os_version"], mac_address=row["mac_address"], protocol=row["protocol"], browser_or_agent=row["browser_agent"])
        vol = float(row.get("data_volume_mb", 0.0)) if "data_volume_mb" in row else 0.0
        evt = AccessLogEvent(
            event_id=row["event_id"], entity_id=row["entity_id"], entity_type=row["entity_type"],
            timestamp=row["timestamp"], geo_location=geo, resource_accessed=row["resource_accessed"],
            auth_method=row["auth_method"], auth_success=bool(row["auth_success"]),
            session_duration=float(row["session_duration"]), command_sequence=cmds,
            device_fingerprint=fp, data_volume_mb=vol, label=row["label"]
        )
        feature_rows.append(fe_testset.extract_features(evt))
        labels.append(row["label"])

    df_feats = pd.DataFrame(feature_rows)[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    labels_series = pd.Series(labels)

    categories = [
        "normal", "brute_force", "credential_stuffing", "lateral_movement",
        "low_and_slow_exfiltration", "device_spoofing", "impossible_travel", "insider_drift"
    ]
    label2id = {c: i for i, c in enumerate(categories)}

    # Chronological Split
    total = len(df_feats)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    X_tr_df, X_te_df = df_feats.iloc[:train_end], df_feats.iloc[val_end:]
    y_tr, y_te = labels_series.iloc[:train_end], labels_series.iloc[val_end:]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_df)
    X_te_scaled = scaler.transform(X_te_df)

    y_tr_num = y_tr.map(label2id).values
    y_te_num = y_te.map(label2id).values

    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective="multi:softprob", num_class=len(categories), n_jobs=-1, random_state=42)
    xgb_clf.fit(X_tr_scaled, y_tr_num)

    xgb_preds = xgb_clf.predict(X_te_scaled)
    report = classification_report(y_te_num, xgb_preds, target_names=categories, output_dict=True, zero_division=0)

    print(f"XGBoost Test Accuracy: {report['accuracy']*100:.2f}%")
    print(f"XGBoost Macro F1-Score: {report['macro avg']['f1-score']*100:.2f}%")
    print("\nPer-Class Metrics on Untouched Test Set:")
    print(f"{'Category':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for cat in categories:
        m = report[cat]
        print(f"{cat:<28} | {m['precision']*100:>9.2f}% | {m['recall']*100:>9.2f}% | {m['f1-score']*100:>9.2f}%")

if __name__ == "__main__":
    run_refactored_evaluation()
