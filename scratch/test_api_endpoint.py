import urllib.request
import json

url = "http://localhost:5000/api/evaluate_single"

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

data_bytes = json.dumps(payload_it).encode("utf-8")
req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req) as response:
        res_text = response.read().decode("utf-8")
        res_json = json.loads(res_text)
        print("API Response for Impossible Travel:")
        print(f"Predicted Attack: {res_json.get('predicted_attack_type')}")
        print(f"Risk Score:       {res_json.get('risk_score')} / 100")
        print(f"Description:      {res_json.get('attack_description')}")
        print("Key Reasons:")
        for r in res_json.get("key_reasons", []):
            print(f"  • {r}")
        print("Feature Attributions:", json.dumps(res_json.get("feature_attribution_weights")))
except Exception as e:
    print(f"Error: {e}")
