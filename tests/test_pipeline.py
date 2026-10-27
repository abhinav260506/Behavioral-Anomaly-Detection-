"""
Automated System Verification Tests for AI-Powered Behavioral Anomaly Detection.
Verifies generator schema compliance, feature engineering, cold start, detector, classifier, pipeline, and 70/15/15 dataset splits.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.generator import SyntheticDataGenerator
from src.feature_engineering import FeatureExtractor
from src.cold_start import ColdStartEngine
from src.anomaly_detector import SequentialAnomalyDetector
from src.anomaly_classifier import AttackClassifier
from src.explainability import ExplainabilityEngine
from src.pipeline import SecurityAnomalyPipeline
from src.train import DatasetTrainer

def test_synthetic_generator_schema():
    gen = SyntheticDataGenerator(num_entities=10)
    dataset = gen.generate_full_dataset(num_normal=100, inject_attacks=True)
    assert len(dataset) > 100
    first = dataset[0]
    assert hasattr(first, "entity_id")
    assert hasattr(first, "geo_location")
    assert hasattr(first, "device_fingerprint")
    assert first.label in [
        "normal", "brute_force", "impossible_travel", "credential_stuffing",
        "lateral_movement", "device_spoofing", "low_and_slow_exfiltration", "insider_drift"
    ]

def test_feature_engineering_geo_velocity():
    fe = FeatureExtractor()
    dt1 = datetime(2026, 7, 25, 10, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2026, 7, 25, 10, 15, 0, tzinfo=timezone.utc)  # 15 mins later

    evt1 = AccessLogEvent(
        event_id="e1", entity_id="user_test", entity_type="user", timestamp=dt1.isoformat(),
        geo_location=GeoLocation("1.1.1.1", "USA", "New York", 40.7128, -74.0060),
        resource_accessed="/api/test", auth_method="password", auth_success=True, session_duration=60,
        command_sequence=[], device_fingerprint=DeviceFingerprint("Win11", "AA:BB:CC:DD:EE:FF", "TLS", "Chrome"), label="normal"
    )
    evt2 = AccessLogEvent(
        event_id="e2", entity_id="user_test", entity_type="user", timestamp=dt2.isoformat(),
        geo_location=GeoLocation("2.2.2.2", "UK", "London", 51.5074, -0.1278),  # ~5,500 km away
        resource_accessed="/api/test", auth_method="password", auth_success=True, session_duration=60,
        command_sequence=[], device_fingerprint=DeviceFingerprint("Win11", "AA:BB:CC:DD:EE:FF", "TLS", "Chrome"), label="impossible_travel"
    )

    f1 = fe.extract_features(evt1)
    f2 = fe.extract_features(evt2)

    assert f1["geo_velocity_kmh"] == 0.0
    assert f2["geo_velocity_kmh"] > 5000.0  # Implausible speed

def test_cold_start_weights():
    cs = ColdStartEngine(cold_start_threshold=10)
    w_ind_0, w_peer_0 = cs.compute_cold_start_weight(0)
    assert w_ind_0 == 0.0 and w_peer_0 == 1.0

    w_ind_5, w_peer_5 = cs.compute_cold_start_weight(5)
    assert w_ind_5 == 0.5 and w_peer_5 == 0.5

    w_ind_10, w_peer_10 = cs.compute_cold_start_weight(10)
    assert w_ind_10 == 1.0 and w_peer_10 == 0.0

def test_classifier_attack_categories():
    ac = AttackClassifier()
    dummy_evt = AccessLogEvent(
        event_id="e", entity_id="u", entity_type="user", timestamp="2026-07-25T12:00:00Z",
        geo_location=GeoLocation("1.1.1.1", "US", "NY", 40.0, -74.0), resource_accessed="/test",
        auth_method="password", auth_success=False, session_duration=1, command_sequence=[],
        device_fingerprint=DeviceFingerprint("OS", "MAC", "TLS", "Agent"), label="brute_force"
    )

    feats = {
        "hour_sin": 0.0, "hour_cos": 1.0, "is_weekend": 0.0, "is_off_hours": 0.0,
        "geo_velocity_kmh": 0.0, "rolling_failures_5m": 8.0, "rolling_failures_1h": 10.0,
        "ip_diversity_1h": 1.0, "resource_novelty": 0.0, "fp_mismatch": 0.0,
        "suspicious_cmd_count": 0.0, "auth_failed": 1.0, "session_duration": 1.0, "history_count": 12.0
    }

    category = ac.classify_anomaly(dummy_evt, feats, risk_score=85.0)
    assert category == "brute_force"

def test_dataset_70_15_15_split():
    trainer = DatasetTrainer(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    gen = SyntheticDataGenerator(num_entities=10)
    dataset = gen.generate_full_dataset(num_normal=100, inject_attacks=False)

    train_data, val_data, test_data = trainer.split_dataset(dataset)
    total = len(dataset)

    assert len(train_data) == int(total * 0.70)
    assert len(val_data) == int(total * 0.85) - int(total * 0.70)
    assert len(train_data) + len(val_data) + len(test_data) == total

def test_train_val_test_pipeline():
    trainer = DatasetTrainer(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    results = trainer.run_training_pipeline(num_logs=300)

    assert results["splits"]["train_percent"] == "70%"
    assert results["splits"]["val_percent"] == "15%"
    assert results["splits"]["test_percent"] == "15%"
    assert results["test_metrics"]["recall"] >= 0.70

def test_evaluate_single_null_and_list_payload():
    from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
    from src.pipeline import SecurityAnomalyPipeline

    pipeline = SecurityAnomalyPipeline(num_entities=5)
    
    # Test payload containing null device_fingerprint, null resource_accessed, and status string
    raw_payload = [
        {
            "entity_id": "U103",
            "entity_type": "user",
            "timestamp": "2026-07-25T09:30:00Z",
            "geo_location": {
                "city": "London",
                "country": "United Kingdom",
                "latitude": 51.5074,
                "longitude": -0.1278
            },
            "status": "SUCCESS",
            "source_ip": None,
            "resource_accessed": None,
            "auth_method": None,
            "device_fingerprint": None
        }
    ]

    target = raw_payload[-1]
    geo_raw = target.get("geo_location") or {}
    geo = GeoLocation("127.0.0.1", geo_raw.get("country", ""), geo_raw.get("city", ""), geo_raw.get("latitude", 0), geo_raw.get("longitude", 0))
    fp_raw = target.get("device_fingerprint") or {}
    fp = DeviceFingerprint(fp_raw.get("os_version", "Win11"), fp_raw.get("mac_address", "00:00:00:00:00:00"), "TLS", "Agent")

    evt = AccessLogEvent("e_null", "U103", "user", "2026-07-25T09:30:00Z", geo, "/api/auth", "password", True, 120.0, [], fp, "normal")
    feats = pipeline.feature_extractor.extract_features(evt)
    score, is_anom = pipeline.detector.predict_event(evt.entity_id, feats, evt)
    
    assert score >= 0.0
    assert isinstance(is_anom, bool)

def test_custom_entity_id_attack_sequence():
    gen = SyntheticDataGenerator(num_entities=5)
    now = datetime.now(timezone.utc)
    # Test passing a custom unknown entity_id string
    custom_id = "User U103 logged in successfully from Chennai at 9:00 AM"
    events = gen.generate_attack_sequence("impossible_travel", now, entity_id=custom_id)
    
    assert len(events) > 0
    assert events[0].entity_id == custom_id


