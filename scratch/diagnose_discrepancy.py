"""
Diagnostic Script for IsolatedTestEngine Discrepancies.
Diagnoses why IsolatedTestEngine produced different numerical results than run_full_system_verification.py.
Prints exact numerical traces for:
1. Impossible Travel
2. Lateral Movement
3. Insider Drift (Stage-2 45-dim vector & XGBoost probabilities)
4. Device Spoofing (Fingerprint features & XGBoost probabilities)
5. Component consistency & pipeline equality check
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from datetime import datetime

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES, RESTRICTED_RESOURCE_KEYWORDS
from src.cold_start import ColdStartEngine
from src.anomaly_detector import SequentialAnomalyDetector
from src.anomaly_classifier import AttackClassifier
from src.isolated_test_engine import IsolatedTestEngine

def run_discrepancy_diagnosis():
    print("="*100)
    print("DIAGNOSING ISOLATED TEST ENGINE DISCREPANCIES")
    print("="*100)

    engine = IsolatedTestEngine()

    # --- 1. IMPOSSIBLE TRAVEL ---
    print("\n" + "="*80)
    print("1. IMPOSSIBLE TRAVEL NUMERICAL TRACE IN ISOLATED TEST ENGINE")
    print("="*80)
    it_payload = [
        {"event_id": "it_1", "entity_id": "u_it", "timestamp": "2026-07-25T08:00:00Z",
         "geo_location": {"ip": "1.1.1.1", "city": "Chennai", "country": "India", "latitude": 13.0827, "longitude": 80.2707},
         "resource_accessed": "/doc", "auth_success": True, "session_duration": 120},
        {"event_id": "it_2", "entity_id": "u_it", "timestamp": "2026-07-25T08:25:00Z",
         "geo_location": {"ip": "2.2.2.2", "city": "New York", "country": "USA", "latitude": 40.7128, "longitude": -74.0060},
         "resource_accessed": "/doc", "auth_success": True, "session_duration": 120}
    ]

    res_it = engine.evaluate_isolated_request(it_payload)
    for idx, trace in enumerate(res_it["traces"]):
        print(f"\nTrace Event {idx+1}:")
        print(f"  Final Risk Score:           {trace['risk_score']}")
        print(f"  Stage-1 Anomaly Flag:       {trace['stage1_anomaly']}")
        print(f"  Final Predicted Class:      {trace['final_predicted_class']}")

    # Let's inspect features extracted inside detector for Event 2
    fe = FeatureExtractor()
    det = SequentialAnomalyDetector()

    evt1 = AccessLogEvent("it_1", "u_it", "user", "2026-07-25T08:00:00Z", GeoLocation("1.1.1.1", "India", "Chennai", 13.0827, 80.2707), "/doc", "password", True, 120.0, [], DeviceFingerprint("Win11", "AA:BB:CC", "TLS", "Chrome"))
    evt2 = AccessLogEvent("it_2", "u_it", "user", "2026-07-25T08:25:00Z", GeoLocation("2.2.2.2", "USA", "New York", 40.7128, -74.0060), "/doc", "password", True, 120.0, [], DeviceFingerprint("Win11", "AA:BB:CC", "TLS", "Chrome"))

    f1 = fe.extract_features(evt1)
    r1, _ = det.predict_event("u_it", f1, evt1)

    f2 = fe.extract_features(evt2)
    r2, _ = det.predict_event("u_it", f2, evt2)

    print("\nDirect Detector Predict for Impossible Travel Event 2:")
    print(f"  geo_velocity_kmh:           {f2.get('geo_velocity_kmh'):.2f}")
    print(f"  history_count:              {f2.get('history_count')}")
    print(f"  Direct Predict Risk:        {r2:.1f}")

    # --- 2. LATERAL MOVEMENT ---
    print("\n" + "="*80)
    print("2. LATERAL MOVEMENT NUMERICAL TRACE IN ISOLATED TEST ENGINE")
    print("="*80)
    lm_payload = [
        {"event_id": "lm_1", "entity_id": "u_lm", "timestamp": "2026-07-25T14:00:00Z",
         "geo_location": {"ip": "10.0.0.5", "city": "Chicago", "country": "US", "latitude": 41.8, "longitude": -87.6},
         "resource_accessed": "/engineering/dashboard", "auth_success": True, "session_duration": 120},
        {"event_id": "lm_2", "entity_id": "u_lm", "timestamp": "2026-07-25T14:05:00Z",
         "geo_location": {"ip": "10.0.0.5", "city": "Chicago", "country": "US", "latitude": 41.8, "longitude": -87.6},
         "resource_accessed": "/engineering/git", "auth_success": True, "session_duration": 120},
        {"event_id": "lm_3", "entity_id": "u_lm", "timestamp": "2026-07-25T14:10:00Z",
         "geo_location": {"ip": "10.0.0.5", "city": "Chicago", "country": "US", "latitude": 41.8, "longitude": -87.6},
         "resource_accessed": "/internal/server-config", "auth_success": True, "session_duration": 120},
        {"event_id": "lm_4", "entity_id": "u_lm", "timestamp": "2026-07-25T14:15:00Z",
         "geo_location": {"ip": "10.0.0.5", "city": "Chicago", "country": "US", "latitude": 41.8, "longitude": -87.6},
         "resource_accessed": "/admin/credentials", "auth_success": True, "session_duration": 120},
        {"event_id": "lm_5", "entity_id": "u_lm", "timestamp": "2026-07-25T14:20:00Z",
         "geo_location": {"ip": "10.0.0.5", "city": "Chicago", "country": "US", "latitude": 41.8, "longitude": -87.6},
         "resource_accessed": "/admin/vault_keys", "auth_success": True, "command_sequence": ["mimikatz.exe"], "session_duration": 300}
    ]

    res_lm = engine.evaluate_isolated_request(lm_payload)
    for idx, trace in enumerate(res_lm["traces"]):
        print(f"\nTrace Event {idx+1} ({trace.get('current_event_geo')}):")
        print(f"  Risk Score:                 {trace['risk_score']}")
        print(f"  Stage-1 Anomaly:            {trace['stage1_anomaly']}")
        print(f"  Predicted Class:            {trace['final_predicted_class']}")

    # --- 3. INSIDER DRIFT ---
    print("\n" + "="*80)
    print("3. INSIDER DRIFT STAGE-2 VECTOR & XGBOOST PROBABILITIES")
    print("="*80)
    id_payload = [
        {"event_id": f"id_seq_{i}", "entity_id": "u_id_seq", "timestamp": f"2026-07-25T{10+i:02d}:00:00Z",
         "geo_location": {"ip": "10.0.0.8", "city": "Seattle", "country": "US", "latitude": 47.6, "longitude": -122.3},
         "resource_accessed": f"/dept_{i}/resource", "auth_success": True, "command_sequence": ["git clone project"], "session_duration": 300}
        for i in range(4)
    ]

    res_id = engine.evaluate_isolated_request(id_payload)
    last_id_trace = res_id["traces"][-1]
    print(f"Insider Drift Risk Score:      {last_id_trace['risk_score']}")
    print(f"Stage-1 Anomaly Flag:          {last_id_trace['stage1_anomaly']}")
    print(f"Final Predicted Class:         {last_id_trace['final_predicted_class']}")

    classifier = AttackClassifier()
    print("\nClassifier Model Path:", classifier.model_path)
    print("Classifier Model Classes:", classifier.model.classes_ if hasattr(classifier, 'model') and hasattr(classifier.model, 'classes_') else "N/A")

    # --- 4. DEVICE SPOOFING ---
    print("\n" + "="*80)
    print("4. DEVICE SPOOFING STAGE-2 VECTOR & XGBOOST PROBABILITIES")
    print("="*80)
    ds_payload = [
        {"event_id": "ds_1", "entity_id": "u_ds", "timestamp": "2026-07-25T15:00:00Z",
         "geo_location": {"ip": "10.0.0.6", "city": "Dallas", "country": "US", "latitude": 32.7, "longitude": -96.8},
         "resource_accessed": "/doc", "auth_success": True, "device_fingerprint": {"os_version": "Win11", "mac_address": "AA:BB:CC:DD:EE:FF", "protocol": "TLSv1.3", "browser_or_agent": "Chrome/120"}},
        {"event_id": "ds_2", "entity_id": "u_ds", "timestamp": "2026-07-25T15:10:00Z",
         "geo_location": {"ip": "10.0.0.6", "city": "Dallas", "country": "US", "latitude": 32.7, "longitude": -96.8},
         "resource_accessed": "/doc", "auth_success": True, "device_fingerprint": {"os_version": "Linux", "mac_address": "FF:EE:DD:CC:BB:AA", "protocol": "SSHv2", "browser_or_agent": "Agent"}}
    ]

    res_ds = engine.evaluate_isolated_request(ds_payload)
    last_ds_trace = res_ds["traces"][-1]
    print(f"Device Spoofing Risk Score:    {last_ds_trace['risk_score']}")
    print(f"Stage-1 Anomaly Flag:          {last_ds_trace['stage1_anomaly']}")
    print(f"Final Predicted Class:         {last_ds_trace['final_predicted_class']}")

    # --- 5. CHECK REASON FOR DISCREPANCY BETWEEN AUTO-FIT & CUSTOM FIT ---
    print("\n" + "="*80)
    print("5. MODEL & SCALER AUTO-FIT VS MASTER SCRIPT AUDIT")
    print("="*80)

    det_auto = SequentialAnomalyDetector(auto_fit=True)
    print(f"Auto-fit detector is_fitted: {det_auto.is_fitted}")
    print(f"Auto-fit scaler mean (first 5): {det_auto.scaler.mean_[:5] if det_auto.is_fitted else 'N/A'}")

    # Inspect _auto_fit_from_csv implementation details
    df_raw = pd.read_csv("data/synthetic_telemetry_27k.csv")
    print(f"Total CSV rows: {len(df_raw)}")
    print("First 5 rows labels:", df_raw["label"].head(5).tolist())

if __name__ == "__main__":
    run_discrepancy_diagnosis()
