"""
Verify that each attack category generates distinct, attack-specific XAI explainability outputs.
"""

from src.isolated_test_engine import IsolatedTestEngine
from src.explainability import ExplainabilityEngine
from src.config import STAGE1_THRESHOLD
import json

iso_engine = IsolatedTestEngine(risk_threshold=STAGE1_THRESHOLD)
explainer = ExplainabilityEngine()

# Test payloads for 4 different attack types + normal
payloads = {
    "Impossible Travel": [
        {"event_id": "IT_1", "entity_id": "USER_IT", "timestamp": "2026-10-26T08:00:00Z", "source_ip": "203.0.113.31", "resource_accessed": "/employee/dashboard", "geo_location": {"city": "Chennai", "country": "India", "latitude": 13.0827, "longitude": 80.2707}},
        {"event_id": "IT_2", "entity_id": "USER_IT", "timestamp": "2026-10-26T08:20:00Z", "source_ip": "198.51.100.92", "resource_accessed": "/employee/dashboard", "geo_location": {"city": "London", "country": "United Kingdom", "latitude": 51.5074, "longitude": -0.1278}}
    ],
    "Lateral Movement": [
        {"event_id": "LM_1", "entity_id": "USER_LM", "timestamp": "2026-10-26T10:00:00Z", "resource_accessed": "/employee/dashboard"},
        {"event_id": "LM_2", "entity_id": "USER_LM", "timestamp": "2026-10-26T10:02:00Z", "resource_accessed": "/internal/server-config"},
        {"event_id": "LM_3", "entity_id": "USER_LM", "timestamp": "2026-10-26T10:05:00Z", "resource_accessed": "/admin/vault_keys"}
    ],
    "Device Spoofing": [
        {"event_id": "DS_1", "entity_id": "USER_DS", "timestamp": "2026-10-26T12:00:00Z", "device_fingerprint": {"os_version": "Ubuntu-22.04", "mac_address": "00:11:22:33:44:55", "protocol": "HTTPS"}},
        {"event_id": "DS_2", "entity_id": "USER_DS", "timestamp": "2026-10-26T12:05:00Z", "device_fingerprint": {"os_version": "Kali-Linux", "mac_address": "AA:BB:CC:DD:EE:92", "protocol": "SSH"}},
        {"event_id": "DS_3", "entity_id": "USER_DS", "timestamp": "2026-10-26T12:08:00Z", "device_fingerprint": {"os_version": "Unknown-OS", "mac_address": "FF:EE:DD:CC:BB:AA", "protocol": "TELNET"}}
    ],
    "Insider Drift": [
        {"event_id": "ID_1", "entity_id": "USER_ID", "timestamp": "2026-10-26T14:00:00Z", "resource_accessed": "/engineering/repo"},
        {"event_id": "ID_2", "entity_id": "USER_ID", "timestamp": "2026-10-26T14:15:00Z", "resource_accessed": "/sales/leads"},
        {"event_id": "ID_3", "entity_id": "USER_ID", "timestamp": "2026-10-26T14:30:00Z", "resource_accessed": "/finance/payroll"},
        {"event_id": "ID_4", "entity_id": "USER_ID", "timestamp": "2026-10-26T14:45:00Z", "resource_accessed": "/hr/salaries"},
        {"event_id": "ID_5", "entity_id": "USER_ID", "timestamp": "2026-10-26T15:00:00Z", "resource_accessed": "/admin/master_keys"}
    ],
    "Normal Baseline": [
        {"event_id": "N_1", "entity_id": "USER_N", "timestamp": "2026-11-10T09:30:00Z", "resource_accessed": "/employee/dashboard"},
        {"event_id": "N_2", "entity_id": "USER_N", "timestamp": "2026-11-11T09:40:00Z", "resource_accessed": "/employee/profile"}
    ]
}

print("="*90)
print("TESTING DYNAMIC ATTACK-SPECIFIC EXPLAINABILITY FOR DIFFERENT TEST CASES")
print("="*90)

for name, raw_list in payloads.items():
    iso_res = iso_engine.evaluate_isolated_request(raw_list)
    traces = iso_res.get("traces", [])
    max_trace = max(traces, key=lambda t: t.get("risk_score", 0.0)) if traces else {}
    pred_class = max_trace.get("final_predicted_class", "normal")
    max_risk = max_trace.get("risk_score", 0.0)

    seq_expl = explainer.explain_sequence(raw_list, traces, pred_class, max_risk, STAGE1_THRESHOLD)

    print(f"\n>>> TESTPAYLOAD: {name.upper()}")
    print(f"    Predicted Category: {pred_class}")
    print(f"    Max Risk Score:     {max_risk} / 100")
    print(f"    Attack Description: {seq_expl['attack_description']}")
    print("    Key Reasons Why:")
    for r in seq_expl['key_reasons']:
        print(f"      • {r}")
    print(f"    Feature Attributions: {json.dumps(seq_expl['feature_attribution_weights'])}")

print("\n" + "="*90)
