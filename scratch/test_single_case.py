import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from src.isolated_test_engine import IsolatedTestEngine

payload = [
    {
      "event_id": "evt_cs_001",
      "entity_id": "TEST_USER_101",
      "entity_type": "user",
      "timestamp": "2026-07-25T15:00:00Z",
      "source_ip": "203.0.113.50",
      "geo_location": {
        "country": "India",
        "city": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707
      },
      "resource_accessed": "/api/v1/auth/login",
      "auth_method": "password",
      "auth_success": False,
      "session_duration": 1
    },
    {
      "event_id": "evt_cs_002",
      "entity_id": "TEST_USER_102",
      "entity_type": "user",
      "timestamp": "2026-07-25T15:00:10Z",
      "source_ip": "203.0.113.50",
      "geo_location": {
        "country": "India",
        "city": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707
      },
      "resource_accessed": "/api/v1/auth/login",
      "auth_method": "password",
      "auth_success": False,
      "session_duration": 1
    },
    {
      "event_id": "evt_cs_003",
      "entity_id": "TEST_USER_103",
      "entity_type": "user",
      "timestamp": "2026-07-25T15:00:20Z",
      "source_ip": "203.0.113.50",
      "geo_location": {
        "country": "India",
        "city": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707
      },
      "resource_accessed": "/api/v1/auth/login",
      "auth_method": "password",
      "auth_success": False,
      "session_duration": 1
    },
    {
      "event_id": "evt_cs_004",
      "entity_id": "TEST_USER_104",
      "entity_type": "user",
      "timestamp": "2026-07-25T15:00:30Z",
      "source_ip": "203.0.113.50",
      "geo_location": {
        "country": "India",
        "city": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707
      },
      "resource_accessed": "/api/v1/auth/login",
      "auth_method": "password",
      "auth_success": False,
      "session_duration": 1
    },
    {
      "event_id": "evt_cs_005",
      "entity_id": "TEST_USER_105",
      "entity_type": "user",
      "timestamp": "2026-07-25T15:00:40Z",
      "source_ip": "203.0.113.50",
      "geo_location": {
        "country": "India",
        "city": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707
      },
      "resource_accessed": "/api/v1/auth/login",
      "auth_method": "password",
      "auth_success": False,
      "session_duration": 1
    },
    {
      "event_id": "evt_cs_006",
      "entity_id": "TEST_USER_106",
      "entity_type": "user",
      "timestamp": "2026-07-25T15:00:50Z",
      "source_ip": "203.0.113.50",
      "geo_location": {
        "country": "India",
        "city": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707
      },
      "resource_accessed": "/api/v1/auth/login",
      "auth_method": "password",
      "auth_success": False,
      "session_duration": 1
    }
]

engine = IsolatedTestEngine(risk_threshold=50.0)
res = engine.evaluate_isolated_request(payload)
print(json.dumps(res, indent=2))
