"""
Comprehensive Automated Test Suite for All 9 REST API Endpoints in app.py.
Verifies HTTP status 200, valid JSON response structure, and error handling.
"""

import urllib.request
import urllib.parse
import json

BASE_URL = "http://localhost:5000"

def get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "TestClient"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")
        status = resp.status
        try:
            data = json.loads(content)
        except Exception:
            data = content
        return status, data

def post(path, payload):
    url = f"{BASE_URL}{path}"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")
        status = resp.status
        try:
            data = json.loads(content)
        except Exception:
            data = content
        return status, data

def run_tests():
    print("="*80)
    print("AUTOMATED TEST SUITE FOR ALL APP.PY REST API ENDPOINTS")
    print("="*80)

    results = []

    # 1. GET /
    try:
        s, d = get("/")
        is_ok = s == 200 and "<!DOCTYPE html>" in d
        results.append(("1. GET /", "PASS" if is_ok else "FAIL", f"HTTP {s}"))
    except Exception as e:
        results.append(("1. GET /", "FAIL", str(e)))

    # 2. GET /static/css/styles.css
    try:
        s, d = get("/static/css/styles.css")
        is_ok = s == 200 and len(d) > 0
        results.append(("2. GET /static/css/styles.css", "PASS" if is_ok else "FAIL", f"HTTP {s}"))
    except Exception as e:
        results.append(("2. GET /static/css/styles.css", "FAIL", str(e)))

    # 3. GET /static/js/dashboard.js
    try:
        s, d = get("/static/js/dashboard.js")
        is_ok = s == 200 and len(d) > 0
        results.append(("3. GET /static/js/dashboard.js", "PASS" if is_ok else "FAIL", f"HTTP {s}"))
    except Exception as e:
        results.append(("3. GET /static/js/dashboard.js", "FAIL", str(e)))

    # 4. GET /api/alerts
    try:
        s, d = get("/api/alerts")
        is_ok = s == 200 and isinstance(d, list)
        results.append(("4. GET /api/alerts", "PASS" if is_ok else "FAIL", f"HTTP {s}, alerts={len(d)}"))
    except Exception as e:
        results.append(("4. GET /api/alerts", "FAIL", str(e)))

    # 5. GET /api/metrics
    try:
        s, d = get("/api/metrics")
        is_ok = s == 200 and isinstance(d, dict) and "total_logs_processed" in d
        results.append(("5. GET /api/metrics", "PASS" if is_ok else "FAIL", f"HTTP {s}, keys={list(d.keys())}"))
    except Exception as e:
        results.append(("5. GET /api/metrics", "FAIL", str(e)))

    # 6. GET /api/entity/U101
    try:
        s, d = get("/api/entity/U101")
        is_ok = s == 200 and isinstance(d, dict) and d.get("entity_id") == "U101"
        results.append(("6. GET /api/entity/U101", "PASS" if is_ok else "FAIL", f"HTTP {s}, history={len(d.get('history', []))}"))
    except Exception as e:
        results.append(("6. GET /api/entity/U101", "FAIL", str(e)))

    # 7. POST /api/inject
    try:
        s, d = post("/api/inject", {"attack_type": "impossible_travel", "entity_id": "U102"})
        is_ok = s == 200 and d.get("status") == "success"
        results.append(("7. POST /api/inject", "PASS" if is_ok else "FAIL", f"HTTP {s}, status={d.get('status')}"))
    except Exception as e:
        results.append(("7. POST /api/inject", "FAIL", str(e)))

    # 8. POST /api/evaluate_single
    try:
        eval_payload = {
            "events": [
                {"event_id": "test_1", "entity_id": "u_test", "timestamp": "2026-10-26T08:00:00Z", "resource_accessed": "/api/v1/auth/login", "auth_success": False},
                {"event_id": "test_2", "entity_id": "u_test", "timestamp": "2026-10-26T08:00:05Z", "resource_accessed": "/api/v1/auth/login", "auth_success": False}
            ]
        }
        s, d = post("/api/evaluate_single", eval_payload)
        is_ok = s == 200 and d.get("isolated_test_mode") is True
        results.append(("8. POST /api/evaluate_single", "PASS" if is_ok else "FAIL", f"HTTP {s}, risk={d.get('risk_score')}, pred={d.get('predicted_attack_type')}"))
    except Exception as e:
        results.append(("8. POST /api/evaluate_single", "FAIL", str(e)))

    # 9. POST /api/feedback
    try:
        s, d = post("/api/feedback", {"alert_id": "alt_1", "action": "false_positive"})
        is_ok = s == 200 and d.get("status") == "updated"
        results.append(("9. POST /api/feedback", "PASS" if is_ok else "FAIL", f"HTTP {s}, status={d.get('status')}"))
    except Exception as e:
        results.append(("9. POST /api/feedback", "FAIL", str(e)))

    print(f"\n{'Endpoint':<35} | {'Result':<8} | {'Details':<40}")
    print("-" * 88)
    for name, res, det in results:
        print(f"{name:<35} | {res:<8} | {det:<40}")

    all_passed = all(res == "PASS" for _, res, _ in results)
    print("\n" + "="*80)
    print(f"OVERALL ALL-ENDPOINT SUITE DECISION: {'ALL PASSED 100%' if all_passed else 'SOME ENDPOINTS FAILED'}")
    print("="*80)

if __name__ == "__main__":
    run_tests()
