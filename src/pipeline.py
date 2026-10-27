"""
End-to-End Pipeline Coordinator.
Orchestrates synthetic data generation, feature engineering, sequence-aware anomaly detection,
attack classification, explainability attribution, and alert queuing.
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple
from src.schema import AccessLogEvent, Alert
from src.generator import SyntheticDataGenerator
from src.feature_engineering import FeatureExtractor
from src.anomaly_detector import SequentialAnomalyDetector
from src.anomaly_classifier import AttackClassifier
from src.explainability import ExplainabilityEngine

class SecurityAnomalyPipeline:
    def __init__(self, num_entities: int = 25):
        self.generator = SyntheticDataGenerator(num_entities=num_entities)
        self.feature_extractor = FeatureExtractor()
        self.detector = SequentialAnomalyDetector()
        self.classifier = AttackClassifier()
        self.explainer = ExplainabilityEngine()

        self.processed_events: List[AccessLogEvent] = []
        self.alerts: List[Alert] = []
        self.entity_history_map: Dict[str, List[Dict[str, Any]]] = {}

        # Pre-train background model on normal sample logs
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Generates initial normal baseline training set to initialize feature scaler & IsolationForest."""
        baseline_logs = self.generator.generate_full_dataset(num_normal=500, inject_attacks=False)
        from src.feature_engineering import STAGE1_FEATURE_NAMES
        matrix = []
        for log in baseline_logs:
            feats = self.feature_extractor.extract_features(log)
            f_vec = [feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES]
            matrix.append(f_vec)

        import numpy as np
        if len(matrix) > 0:
            self.detector.fit_baseline(np.array(matrix))

    def process_event(self, event: AccessLogEvent) -> Tuple[Dict[str, Any], Optional[Alert]]:
        """Processes a single access log event through the pipeline in real-time."""
        self.processed_events.append(event)

        # 1. Feature Extraction
        feats = self.feature_extractor.extract_features(event)

        # 2. Sequential Anomaly Detection & EWMA Concept Drift
        risk_score, is_anomaly = self.detector.predict_event(event.entity_id, feats, event)

        # 3. Anomaly Classification
        attack_type = self.classifier.classify_anomaly(event, feats, risk_score)

        # 4. Explainability Layer
        factors, expl_details = self.explainer.explain_alert(event, feats, risk_score, attack_type)

        # 5. Alert Construction (Flagged if risk score >= 60.0 or is_anomaly or explicitly labeled attack)
        alert_obj = None
        if is_anomaly or risk_score >= 60.0 or attack_type != "normal":
            alert_obj = Alert(
                alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
                timestamp=event.timestamp,
                entity_id=event.entity_id,
                entity_type=event.entity_type,
                risk_score=risk_score,
                is_anomaly=True,
                predicted_attack_type=attack_type,
                ground_truth_label=event.label,
                contributing_factors=factors,
                raw_event=event.to_dict(),
                explanation_details=expl_details
            )
            self.alerts.append(alert_obj)

        # Track history for entity visualization
        if event.entity_id not in self.entity_history_map:
            self.entity_history_map[event.entity_id] = []
        self.entity_history_map[event.entity_id].append({
            "timestamp": event.timestamp,
            "resource": event.resource_accessed,
            "city": event.geo_location.city,
            "risk_score": risk_score,
            "label": event.label,
            "predicted_attack": attack_type,
            "auth_success": event.auth_success
        })

        event_summary = {
            "event_id": event.event_id,
            "entity_id": event.entity_id,
            "timestamp": event.timestamp,
            "risk_score": risk_score,
            "is_anomaly": is_anomaly,
            "attack_type": attack_type,
            "label": event.label
        }
        return event_summary, alert_obj

    def run_batch_simulation(self, num_logs: int = 1500) -> List[Alert]:
        """Runs full dataset simulation through pipeline."""
        dataset = self.generator.generate_full_dataset(num_normal=num_logs, inject_attacks=True)
        for evt in dataset:
            self.process_event(evt)
        return self.alerts

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates global pipeline execution metrics."""
        total_logs = len(self.processed_events)
        total_alerts = len(self.alerts)
        anomaly_rate = (total_alerts / total_logs * 100.0) if total_logs > 0 else 0.0

        # Breakdown by predicted attack category
        category_counts: Dict[str, int] = {}
        for alt in self.alerts:
            cat = alt.predicted_attack_type
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Accuracy & Detection recall statistics against ground-truth labels
        tp, fp, fn, tn = 0, 0, 0, 0
        for alt in self.alerts:
            if alt.ground_truth_label != "normal":
                tp += 1
            else:
                fp += 1

        for evt in self.processed_events:
            if evt.label != "normal":
                # Check if caught in alerts
                caught = any(a.raw_event["event_id"] == evt.event_id for a in self.alerts)
                if not caught:
                    fn += 1
            else:
                caught = any(a.raw_event["event_id"] == evt.event_id for a in self.alerts)
                if not caught:
                    tn += 1

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "total_logs_processed": total_logs,
            "total_alerts_generated": total_alerts,
            "anomaly_rate_percent": round(anomaly_rate, 2),
            "attack_breakdown": category_counts,
            "evaluation_metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            }
        }
