import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from src.isolated_test_engine import IsolatedTestEngine

# 1. Device Spoofing Test Case
device_spoofing_payload = [
    {
        "event_id": "ds_evt_1",
        "entity_id": "user_device_test",
        "timestamp": "2026-07-25T10:00:00Z",
        "geo_location": {"city": "New York", "country": "USA", "latitude": 40.7128, "longitude": -74.0060},
        "resource_accessed": "/api/v1/payroll",
        "device_fingerprint": {"os_version": "Windows 11 Enterprise", "mac_address": "AA:BB:CC:DD:EE:FF", "protocol": "TLSv1.3", "browser_or_agent": "Chrome/120.0"}
    },
    {
        "event_id": "ds_evt_2",
        "entity_id": "user_device_test",
        "timestamp": "2026-07-25T10:05:00Z",
        "geo_location": {"city": "New York", "country": "USA", "latitude": 40.7128, "longitude": -74.0060},
        "resource_accessed": "/api/v1/payroll",
        "device_fingerprint": {"os_version": "Kali Linux 2024.1", "mac_address": "DE:AD:BE:EF:CA:FE", "protocol": "SSHv2", "browser_or_agent": "Mozilla/5.0"}
    }
]

# 2. Lateral Movement Test Case
lateral_movement_payload = [
    {
        "event_id": "lm_evt_1",
        "entity_id": "user_lat_test",
        "timestamp": "2026-07-25T11:00:00Z",
        "geo_location": {"city": "New York", "country": "USA", "latitude": 40.7128, "longitude": -74.0060},
        "resource_accessed": "/admin/vault_keys",
        "command_sequence": ["mimikatz.exe", "curl http://internal.vault/dump"]
    }
]

# 3. Low-and-Slow Exfiltration Test Case
exfiltration_payload = [
    {
        "event_id": "ex_evt_1",
        "entity_id": "user_exfil_test",
        "timestamp": "2026-07-25T03:15:00Z", # 03:15 AM off-hours
        "geo_location": {"city": "New York", "country": "USA", "latitude": 40.7128, "longitude": -74.0060},
        "resource_accessed": "/exfil/bulk_export",
        "command_sequence": ["cat /data/part0"]
    }
]

# 4. Insider Drift Test Case
insider_drift_payload = [
    {
        "event_id": "id_evt_1",
        "entity_id": "user_drift_test",
        "timestamp": "2026-07-25T14:00:00Z",
        "geo_location": {"city": "New York", "country": "USA", "latitude": 40.7128, "longitude": -74.0060},
        "resource_accessed": "/api/v1/new_project_alpha",
        "command_sequence": ["git clone alpha"]
    }
]

engine = IsolatedTestEngine(risk_threshold=50.0)

print("--- 1. DEVICE SPOOFING ---")
res1 = engine.evaluate_isolated_request(device_spoofing_payload)
print(f"Predicted: {res1['traces'][-1]['final_predicted_class']} | Risk: {res1['traces'][-1]['risk_score']}")

print("\n--- 2. LATERAL MOVEMENT ---")
res2 = engine.evaluate_isolated_request(lateral_movement_payload)
print(f"Predicted: {res2['traces'][-1]['final_predicted_class']} | Risk: {res2['traces'][-1]['risk_score']}")

print("\n--- 3. LOW-AND-SLOW EXFILTRATION ---")
res3 = engine.evaluate_isolated_request(exfiltration_payload)
print(f"Predicted: {res3['traces'][-1]['final_predicted_class']} | Risk: {res3['traces'][-1]['risk_score']}")

print("\n--- 4. INSIDER DRIFT ---")
res4 = engine.evaluate_isolated_request(insider_drift_payload)
print(f"Predicted: {res4['traces'][-1]['final_predicted_class']} | Risk: {res4['traces'][-1]['risk_score']}")
