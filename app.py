"""
Web Application Backend & REST API Server for SOC Analyst Dashboard.
Runs on http://localhost:5000 without requiring third-party web frameworks.
"""

import json
import os
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any
from src.pipeline import SecurityAnomalyPipeline

# Initialize global pipeline instance and generate initial batch of telemetry
GLOBAL_PIPELINE = SecurityAnomalyPipeline(num_entities=25)
GLOBAL_PIPELINE.run_batch_simulation(num_logs=1200)

class SOCDashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    def _send_file(self, filepath: str, content_type: str):
        if not os.path.exists(filepath):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File Not Found")
            return
        
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.end_headers()
        with open(filepath, 'rb') as f:
            self.wfile.write(f.read())

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_file("templates/index.html", "text/html")

        elif path.startswith("/static/css/"):
            rel_path = path[1:]
            self._send_file(rel_path, "text/css")

        elif path.startswith("/static/js/"):
            rel_path = path[1:]
            self._send_file(rel_path, "application/javascript")

        elif path == "/api/alerts":
            alerts_data = [a.to_dict() for a in GLOBAL_PIPELINE.alerts]
            # Sort by risk_score descending
            alerts_data.sort(key=lambda x: x["risk_score"], reverse=True)
            self._send_json(alerts_data)

        elif path == "/api/metrics":
            metrics = GLOBAL_PIPELINE.get_summary_metrics()
            
            # Attach verified frozen model evaluation scores
            metrics["frozen_scores"] = {
                "stage1_roc_auc": 0.9734,
                "stage1_pr_auc": 0.7266,
                "stage1_fpr_percent": 2.45,
                "stage1_precision_percent": 71.47,
                "stage1_recall_percent": 70.47,
                "stage1_f1_percent": 70.97,
                "end2end_accuracy_percent": 95.36,
                "macro_precision_percent": 71.63,
                "macro_recall_percent": 66.58,
                "macro_f1_percent": 67.91,
                "entity_overlap": 0,
                "calibrated_threshold": 32.0,
                "per_class_table": [
                    {"class": "normal", "support": 4122, "precision": 97.43, "recall": 97.55, "f1_score": 97.49},
                    {"class": "lateral_movement", "support": 96, "precision": 89.72, "recall": 100.00, "f1_score": 94.58},
                    {"class": "brute_force", "support": 45, "precision": 100.00, "recall": 82.22, "f1_score": 90.24},
                    {"class": "credential_stuffing", "support": 61, "precision": 100.00, "recall": 78.69, "f1_score": 88.07},
                    {"class": "device_spoofing", "support": 21, "precision": 58.33, "recall": 66.67, "f1_score": 62.22},
                    {"class": "insider_drift", "support": 70, "precision": 43.33, "recall": 55.71, "f1_score": 48.75},
                    {"class": "low_and_slow_exfiltration", "support": 42, "precision": 34.21, "recall": 30.95, "f1_score": 32.50},
                    {"class": "impossible_travel", "support": 24, "precision": 50.00, "recall": 20.83, "f1_score": 29.41}
                ],
                "confusion_matrix_8x8": [
                    [4021, 0, 0, 10, 25, 10, 5, 51],
                    [8, 37, 0, 0, 0, 0, 0, 0],
                    [13, 0, 48, 0, 0, 0, 0, 0],
                    [0, 0, 0, 96, 0, 0, 0, 0],
                    [28, 0, 0, 1, 13, 0, 0, 0],
                    [7, 0, 0, 0, 0, 14, 0, 0],
                    [19, 0, 0, 0, 0, 0, 5, 0],
                    [31, 0, 0, 0, 0, 0, 0, 39]
                ]
            }

            # Also load artifact hashes if metadata.json exists
            meta_path = os.path.abspath("models/metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    metrics["metadata"] = meta
                except Exception:
                    pass

            self._send_json(metrics)

        elif path.startswith("/api/entity/"):
            entity_id = path.split("/api/entity/")[1]
            history = GLOBAL_PIPELINE.entity_history_map.get(entity_id, [])
            baseline = GLOBAL_PIPELINE.detector.entity_baselines.get(entity_id, [])
            profile_info = GLOBAL_PIPELINE.generator.entities.get(entity_id, {})

            fp_dict = {}
            if "device_fingerprint" in profile_info and hasattr(profile_info["device_fingerprint"], "to_dict"):
                fp_dict = profile_info["device_fingerprint"].to_dict()

            res = {
                "entity_id": entity_id,
                "entity_type": profile_info.get("entity_type", "user"),
                "role": profile_info.get("role", "N/A"),
                "primary_city": profile_info.get("primary_city", {}).get("city", "N/A"),
                "fingerprint": fp_dict,
                "history": history
            }
            self._send_json(res)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get('Content-Length', 0))
        post_body = self.rfile.read(content_len).decode('utf-8') if content_len > 0 else "{}"
        try:
            body = json.loads(post_body)
        except Exception:
            body = {}

        if path == "/api/inject":
            attack_type = body.get("attack_type", "brute_force")
            entity_id = body.get("entity_id", None)
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            events = GLOBAL_PIPELINE.generator.generate_attack_sequence(attack_type, now, entity_id)
            new_alerts = []
            for evt in events:
                _, alt = GLOBAL_PIPELINE.process_event(evt)
                if alt:
                    new_alerts.append(alt.to_dict())

            self._send_json({
                "status": "success",
                "injected_attack": attack_type,
                "events_generated": len(events),
                "new_alerts_triggered": len(new_alerts)
            })

        elif path == "/api/evaluate_single":
            try:
                from src.isolated_test_engine import IsolatedTestEngine
                from src.config import STAGE1_THRESHOLD
                from src.explainability import ExplainabilityEngine

                # Check if request specifies isolated mode or list input
                raw_list = body
                if isinstance(body, dict):
                    if "events" in body and isinstance(body["events"], list):
                        raw_list = body["events"]
                    elif "events" not in body:
                        raw_list = [body]

                if not isinstance(raw_list, list):
                    raw_list = [body]

                # Run Isolated Test Engine (no state contamination, zero storage/persistence)
                iso_engine = IsolatedTestEngine(risk_threshold=STAGE1_THRESHOLD)
                iso_res = iso_engine.evaluate_isolated_request(raw_list)

                traces = iso_res.get("traces", [])
                
                # Find event trace with maximum risk score in the sequence
                max_trace = max(traces, key=lambda t: t.get("risk_score", 0.0)) if len(traces) > 0 else {}
                anom_traces = [t for t in traces if t.get("stage1_anomaly", False)]

                pred_class = max_trace.get("final_predicted_class", "normal")
                max_risk = max_trace.get("risk_score", 0.0)

                explainer = ExplainabilityEngine()
                seq_expl = explainer.explain_sequence(raw_list, traces, pred_class, max_risk, STAGE1_THRESHOLD)

                # Extract contributing factors across all event traces
                factors = []
                for t in traces:
                    phrasing = t.get("explanation_phrasing")
                    if phrasing and phrasing not in factors:
                        factors.append(phrasing)
                if len(factors) == 0:
                    if max_risk < STAGE1_THRESHOLD:
                        factors = ["Isolated evaluation completed. No anomaly exceeded the configured Stage-1 threshold."]
                    else:
                        factors = ["Isolated evaluation detected anomalous behavior exceeding the configured Stage-1 threshold."]

                res = {
                    "isolated_test_mode": True,
                    "event_id": max_trace.get("event_id", "evt_iso"),
                    "entity_id": max_trace.get("entity_id", "U103"),
                    "risk_score": max_risk,
                    "is_anomaly": len(anom_traces) > 0,
                    "predicted_attack_type": pred_class,
                    "attack_description": seq_expl.get("attack_description", ""),
                    "key_reasons": seq_expl.get("key_reasons", []),
                    "ground_truth_label": "N/A",  # Stateless display
                    "contributing_factors": factors,
                    "feature_attribution_weights": seq_expl.get("feature_attribution_weights", {}),
                    "diagnostic_trace": iso_res
                }
                self._send_json(res)
            except Exception as e:
                import traceback
                print(f"[!] Evaluate Single Exception: {traceback.format_exc()}")
                self._send_json({"error": f"Evaluation Error: {str(e)}"}, status=400)

        elif path == "/api/feedback":
            alert_id = body.get("alert_id")
            action = body.get("action")  # 'confirm' or 'false_positive'
            
            # Find alert
            target_alert = next((a for a in GLOBAL_PIPELINE.alerts if a.alert_id == alert_id), None)
            if target_alert:
                if action == "false_positive":
                    target_alert.is_anomaly = False
                    target_alert.risk_score = max(10.0, target_alert.risk_score - 40.0)
                    target_alert.contributing_factors.append("Marked as False Positive by SOC Analyst (Drift Tuning Applied)")
                elif action == "confirm":
                    target_alert.risk_score = 100.0
                    target_alert.contributing_factors.append("Threat Confirmed by SOC Analyst")

            self._send_json({"status": "updated", "alert_id": alert_id, "action": action})

        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=5000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SOCDashboardHandler)
    print(f"[*] SOC Analyst Dashboard Server running on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server(5000)
