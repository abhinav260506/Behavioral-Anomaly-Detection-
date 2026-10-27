"""
Evaluation Script for the 5 User Payload Sequences.
Executes IsolatedTestEngine.evaluate_isolated_request() on each payload and outputs results.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from src.isolated_test_engine import IsolatedTestEngine

payload_it = {
  "events": [
    {
      "event_id": "FINAL_IT_001",
      "entity_id": "FINAL_USER_IT_01",
      "entity_type": "user",
      "timestamp": "2026-10-26T08:00:00Z",
      "source_ip": "203.0.113.31",
      "resource_accessed": "/employee/dashboard",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 180,
      "data_volume_mb": 1.2,
      "geo_location": {
        "city": "Chennai",
        "country": "India",
        "latitude": 13.0827,
        "longitude": 80.2707
      }
    },
    {
      "event_id": "FINAL_IT_002",
      "entity_id": "FINAL_USER_IT_01",
      "entity_type": "user",
      "timestamp": "2026-10-26T08:20:00Z",
      "source_ip": "198.51.100.92",
      "resource_accessed": "/employee/dashboard",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 160,
      "data_volume_mb": 1.0,
      "geo_location": {
        "city": "London",
        "country": "United Kingdom",
        "latitude": 51.5074,
        "longitude": -0.1278
      }
    }
  ]
}

payload_lm = {
  "events": [
    {
      "event_id": "FINAL_LM_001",
      "entity_id": "FINAL_USER_LM_01",
      "entity_type": "user",
      "timestamp": "2026-10-27T09:00:00Z",
      "source_ip": "10.20.30.41",
      "resource_accessed": "/employee/dashboard",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 180,
      "data_volume_mb": 1.0
    },
    {
      "event_id": "FINAL_LM_002",
      "entity_id": "FINAL_USER_LM_01",
      "entity_type": "user",
      "timestamp": "2026-10-27T09:06:00Z",
      "source_ip": "10.20.30.41",
      "resource_accessed": "/engineering/wiki",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 160,
      "data_volume_mb": 1.5
    },
    {
      "event_id": "FINAL_LM_003",
      "entity_id": "FINAL_USER_LM_01",
      "entity_type": "user",
      "timestamp": "2026-10-27T09:12:00Z",
      "source_ip": "10.20.30.41",
      "resource_accessed": "/internal/server-config",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 120,
      "data_volume_mb": 2.0
    },
    {
      "event_id": "FINAL_LM_004",
      "entity_id": "FINAL_USER_LM_01",
      "entity_type": "user",
      "timestamp": "2026-10-27T09:18:00Z",
      "source_ip": "10.20.30.41",
      "resource_accessed": "/admin/credentials",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 100,
      "data_volume_mb": 2.5
    },
    {
      "event_id": "FINAL_LM_005",
      "entity_id": "FINAL_USER_LM_01",
      "entity_type": "user",
      "timestamp": "2026-10-27T09:24:00Z",
      "source_ip": "10.20.30.41",
      "resource_accessed": "/admin/vault_keys",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 90,
      "data_volume_mb": 3.0
    }
  ]
}

payload_ds = {
  "events": [
    {
      "event_id": "FINAL_DS_001",
      "entity_id": "FINAL_DEVICE_01",
      "entity_type": "service",
      "timestamp": "2026-10-28T09:00:00Z",
      "source_ip": "10.30.40.50",
      "resource_accessed": "/service/dashboard",
      "auth_method": "certificate",
      "auth_success": True,
      "session_duration": 180,
      "data_volume_mb": 2.0,
      "device_os": "Ubuntu-22.04",
      "mac_address": "00:11:22:33:44:55",
      "protocol": "HTTPS"
    },
    {
      "event_id": "FINAL_DS_002",
      "entity_id": "FINAL_DEVICE_01",
      "entity_type": "service",
      "timestamp": "2026-10-28T09:10:00Z",
      "source_ip": "10.30.40.50",
      "resource_accessed": "/service/dashboard",
      "auth_method": "certificate",
      "auth_success": True,
      "session_duration": 170,
      "data_volume_mb": 2.1,
      "device_os": "Ubuntu-22.04",
      "mac_address": "00:11:22:33:44:55",
      "protocol": "HTTPS"
    },
    {
      "event_id": "FINAL_DS_003",
      "entity_id": "FINAL_DEVICE_01",
      "entity_type": "service",
      "timestamp": "2026-10-28T09:20:00Z",
      "source_ip": "10.30.40.50",
      "resource_accessed": "/service/dashboard",
      "auth_method": "certificate",
      "auth_success": True,
      "session_duration": 175,
      "data_volume_mb": 2.0,
      "device_os": "Ubuntu-22.04",
      "mac_address": "00:11:22:33:44:55",
      "protocol": "HTTPS"
    },
    {
      "event_id": "FINAL_DS_004",
      "entity_id": "FINAL_DEVICE_01",
      "entity_type": "service",
      "timestamp": "2026-10-28T09:30:00Z",
      "source_ip": "10.30.40.50",
      "resource_accessed": "/service/dashboard",
      "auth_method": "certificate",
      "auth_success": True,
      "session_duration": 180,
      "data_volume_mb": 2.2,
      "device_os": "Ubuntu-22.04",
      "mac_address": "00:11:22:33:44:55",
      "protocol": "HTTPS"
    },
    {
      "event_id": "FINAL_DS_005",
      "entity_id": "FINAL_DEVICE_01",
      "entity_type": "service",
      "timestamp": "2026-10-28T09:40:00Z",
      "source_ip": "10.30.40.50",
      "resource_accessed": "/service/dashboard",
      "auth_method": "certificate",
      "auth_success": True,
      "session_duration": 165,
      "data_volume_mb": 2.0,
      "device_os": "Ubuntu-22.04",
      "mac_address": "00:11:22:33:44:55",
      "protocol": "HTTPS"
    },
    {
      "event_id": "FINAL_DS_006",
      "entity_id": "FINAL_DEVICE_01",
      "entity_type": "service",
      "timestamp": "2026-10-28T09:43:00Z",
      "source_ip": "10.30.40.50",
      "resource_accessed": "/internal/server-config",
      "auth_method": "certificate",
      "auth_success": True,
      "session_duration": 120,
      "data_volume_mb": 4.0,
      "device_os": "Windows-Server-2025",
      "mac_address": "AA:BB:CC:DD:EE:91",
      "protocol": "SSH"
    },
    {
      "event_id": "FINAL_DS_007",
      "entity_id": "FINAL_DEVICE_01",
      "entity_type": "service",
      "timestamp": "2026-10-28T09:46:00Z",
      "source_ip": "10.30.40.50",
      "resource_accessed": "/admin/credentials",
      "auth_method": "certificate",
      "auth_success": True,
      "session_duration": 100,
      "data_volume_mb": 5.0,
      "device_os": "Kali-Linux",
      "mac_address": "AA:BB:CC:DD:EE:92",
      "protocol": "SSH"
    }
  ]
}

payload_id = {
  "events": [
    {
      "event_id": "FINAL_ID_001",
      "entity_id": "FINAL_EMP_ID_01",
      "entity_type": "user",
      "timestamp": "2026-10-29T10:00:00Z",
      "source_ip": "10.40.20.15",
      "resource_accessed": "/engineering/jira",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 300,
      "data_volume_mb": 2.0
    },
    {
      "event_id": "FINAL_ID_002",
      "entity_id": "FINAL_EMP_ID_01",
      "entity_type": "user",
      "timestamp": "2026-10-30T10:15:00Z",
      "source_ip": "10.40.20.15",
      "resource_accessed": "/engineering/git-core",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 320,
      "data_volume_mb": 3.0
    },
    {
      "event_id": "FINAL_ID_003",
      "entity_id": "FINAL_EMP_ID_01",
      "entity_type": "user",
      "timestamp": "2026-10-31T10:30:00Z",
      "source_ip": "10.40.20.15",
      "resource_accessed": "/engineering/jira",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 280,
      "data_volume_mb": 2.5
    },
    {
      "event_id": "FINAL_ID_004",
      "entity_id": "FINAL_EMP_ID_01",
      "entity_type": "user",
      "timestamp": "2026-11-01T11:00:00Z",
      "source_ip": "10.40.20.15",
      "resource_accessed": "/engineering/deploy-pipeline",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 300,
      "data_volume_mb": 3.5
    },
    {
      "event_id": "FINAL_ID_005",
      "entity_id": "FINAL_EMP_ID_01",
      "entity_type": "user",
      "timestamp": "2026-11-02T11:30:00Z",
      "source_ip": "10.40.20.15",
      "resource_accessed": "/engineering/git-core",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 310,
      "data_volume_mb": 3.0
    },
    {
      "event_id": "FINAL_ID_006",
      "entity_id": "FINAL_EMP_ID_01",
      "entity_type": "user",
      "timestamp": "2026-11-03T12:00:00Z",
      "source_ip": "10.40.20.15",
      "resource_accessed": "/finance/payroll",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 250,
      "data_volume_mb": 5.0
    },
    {
      "event_id": "FINAL_ID_007",
      "entity_id": "FINAL_EMP_ID_01",
      "entity_type": "user",
      "timestamp": "2026-11-04T12:30:00Z",
      "source_ip": "10.40.20.15",
      "resource_accessed": "/hr/performance_reviews",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 260,
      "data_volume_mb": 6.0
    },
    {
      "event_id": "FINAL_ID_008",
      "entity_id": "FINAL_EMP_ID_01",
      "entity_type": "user",
      "timestamp": "2026-11-05T19:30:00Z",
      "source_ip": "10.40.20.15",
      "resource_accessed": "/sales/contracts",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 300,
      "data_volume_mb": 8.0
    },
    {
      "event_id": "FINAL_ID_009",
      "entity_id": "FINAL_EMP_ID_01",
      "entity_type": "user",
      "timestamp": "2026-11-06T21:00:00Z",
      "source_ip": "10.40.20.15",
      "resource_accessed": "/admin/vault_keys",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 350,
      "data_volume_mb": 15.0
    },
    {
      "event_id": "FINAL_ID_010",
      "entity_id": "FINAL_EMP_ID_01",
      "entity_type": "user",
      "timestamp": "2026-11-07T22:15:00Z",
      "source_ip": "10.40.20.15",
      "resource_accessed": "/exfil/bulk_export",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 420,
      "data_volume_mb": 80.0
    }
  ]
}

payload_normal = {
  "events": [
    {
      "event_id": "FINAL_NORMAL_001",
      "entity_id": "FINAL_NORMAL_USER_01",
      "entity_type": "user",
      "timestamp": "2026-11-10T09:30:00Z",
      "source_ip": "10.50.10.20",
      "resource_accessed": "/employee/dashboard",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 240,
      "data_volume_mb": 1.0
    },
    {
      "event_id": "FINAL_NORMAL_002",
      "entity_id": "FINAL_NORMAL_USER_01",
      "entity_type": "user",
      "timestamp": "2026-11-11T09:40:00Z",
      "source_ip": "10.50.10.20",
      "resource_accessed": "/employee/dashboard",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 250,
      "data_volume_mb": 1.2
    },
    {
      "event_id": "FINAL_NORMAL_003",
      "entity_id": "FINAL_NORMAL_USER_01",
      "entity_type": "user",
      "timestamp": "2026-11-12T10:00:00Z",
      "source_ip": "10.50.10.20",
      "resource_accessed": "/employee/profile",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 230,
      "data_volume_mb": 0.8
    },
    {
      "event_id": "FINAL_NORMAL_004",
      "entity_id": "FINAL_NORMAL_USER_01",
      "entity_type": "user",
      "timestamp": "2026-11-13T09:50:00Z",
      "source_ip": "10.50.10.20",
      "resource_accessed": "/employee/dashboard",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 260,
      "data_volume_mb": 1.1
    },
    {
      "event_id": "FINAL_NORMAL_005",
      "entity_id": "FINAL_NORMAL_USER_01",
      "entity_type": "user",
      "timestamp": "2026-11-14T10:10:00Z",
      "source_ip": "10.50.10.20",
      "resource_accessed": "/employee/profile",
      "auth_method": "password",
      "auth_success": True,
      "session_duration": 245,
      "data_volume_mb": 0.9
    }
  ]
}

def main():
    engine = IsolatedTestEngine()

    testcases = [
        ("1. Impossible Travel", payload_it["events"]),
        ("2. Lateral Movement", payload_lm["events"]),
        ("3. Device Spoofing", payload_ds["events"]),
        ("4. Insider Drift", payload_id["events"]),
        ("5. Benign Normal Control", payload_normal["events"])
    ]

    all_results = {}

    for name, events in testcases:
        res = engine.evaluate_isolated_request(events)
        all_results[name] = res

        print("="*80)
        print(f"PAYLOAD EVALUATION: {name}")
        print("="*80)
        for t in res["traces"]:
            idx = t["event_index"]
            eid = t["event_id"]
            risk = t["risk_score"]
            anom = t["stage1_anomaly"]
            cls_name = t["final_predicted_class"]
            print(f"Event {idx} ({eid}): Risk={risk:>5.1f} | Stage1_Flag={str(anom):<5} | Class={cls_name}")

        last_t = res["traces"][-1]
        print(f"\n---> FINAL SUMMARY FOR {name}: Risk = {last_t['risk_score']} | Class = {last_t['final_predicted_class']}\n")

    # Print JSON output for each testcase
    print("\n" + "="*80)
    print("STRUCTURED JSON EVALUATION OUTPUT")
    print("="*80)
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()
