"""
Model Training, Validation & Testing Pipeline.
Splits synthetic sequential dataset into:
- 70% Training set (used for baseline profiling & IsolationForest fitting)
- 15% Validation set (used for threshold calibration & hyperparameter tuning)
- 15% Testing set (used for blind final model evaluation)
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from src.generator import SyntheticDataGenerator
from src.feature_engineering import FeatureExtractor
from src.anomaly_detector import SequentialAnomalyDetector
from src.anomaly_classifier import AttackClassifier
from src.explainability import ExplainabilityEngine
from src.schema import AccessLogEvent

class DatasetTrainer:
    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.feature_extractor = FeatureExtractor()
        self.detector = SequentialAnomalyDetector()
        self.classifier = AttackClassifier()
        self.explainer = ExplainabilityEngine()

    def split_dataset(self, dataset: List[AccessLogEvent]) -> Tuple[List[AccessLogEvent], List[AccessLogEvent], List[AccessLogEvent]]:
        """
        Splits dataset chronologically into 70% Train, 15% Validation, and 15% Test.
        Sequential temporal ordering is preserved to prevent data leakage in behavioral logs.
        """
        total = len(dataset)
        train_end = int(total * self.train_ratio)
        val_end = int(total * (self.train_ratio + self.val_ratio))

        train_data = dataset[:train_end]
        val_data = dataset[train_end:val_end]
        test_data = dataset[val_end:]

        return train_data, val_data, test_data

    def run_training_pipeline(self, num_logs: int = 2000) -> Dict[str, Any]:
        """Runs complete train (70%) / validation (15%) / test (15%) pipeline."""
        print(f"[*] Generating {num_logs} sequential logs for experiment...")
        generator = SyntheticDataGenerator(num_entities=30)
        full_dataset = generator.generate_full_dataset(num_normal=num_logs, inject_attacks=True)

        train_set, val_set, test_set = self.split_dataset(full_dataset)

        print(f"[*] Dataset Split: Train={len(train_set)} (70%), Val={len(val_set)} (15%), Test={len(test_set)} (15%)")

        # 1. TRAIN PHASE (70%): Extract features and fit baseline profiler
        from src.feature_engineering import STAGE1_FEATURE_NAMES
        train_matrix = []
        for evt in train_set:
            feats = self.feature_extractor.extract_features(evt)
            if evt.label == "normal":
                f_vec = [feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES]
                train_matrix.append(f_vec)

        if len(train_matrix) > 0:
            self.detector.fit_baseline(np.array(train_matrix))

        # 2. VALIDATION PHASE (15%): Evaluate thresholds & calibrate detection cutoff
        val_scores = []
        val_labels = []
        for evt in val_set:
            feats = self.feature_extractor.extract_features(evt)
            score, is_anom = self.detector.predict_event(evt.entity_id, feats, evt)
            val_scores.append(score)
            val_labels.append(1 if evt.label != "normal" else 0)

        # Calibrate optimal risk threshold on validation set
        best_threshold = 60.0
        best_f1 = 0.0
        for th in range(40, 85, 5):
            preds = [1 if s >= th else 0 for s in val_scores]
            tp = sum((p == 1 and l == 1) for p, l in zip(preds, val_labels))
            fp = sum((p == 1 and l == 0) for p, l in zip(preds, val_labels))
            fn = sum((p == 0 and l == 1) for p, l in zip(preds, val_labels))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(th)

        print(f"[*] Validation Phase Completed: Calibrated Optimal Risk Threshold = {best_threshold} (F1 = {best_f1:.4f})")

        # 3. TESTING PHASE (15%): Blind final evaluation on held-out test data
        test_tp, test_fp, test_fn, test_tn = 0, 0, 0, 0
        test_alerts = []

        for evt in test_set:
            feats = self.feature_extractor.extract_features(evt)
            score, is_anom = self.detector.predict_event(evt.entity_id, feats, evt)
            predicted_anom = (score >= best_threshold)
            actual_anom = (evt.label != "normal")

            if predicted_anom and actual_anom:
                test_tp += 1
            elif predicted_anom and not actual_anom:
                test_fp += 1
            elif not predicted_anom and actual_anom:
                test_fn += 1
            else:
                test_tn += 1

            if predicted_anom:
                atk_type = self.classifier.classify_anomaly(evt, feats, score)
                factors, _ = self.explainer.explain_alert(evt, feats, score, atk_type)
                test_alerts.append({
                    "entity_id": evt.entity_id,
                    "risk_score": score,
                    "predicted_attack": atk_type,
                    "ground_truth": evt.label,
                    "factors": factors
                })

        prec = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 1.0
        rec = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 1.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        results = {
            "splits": {
                "train_count": len(train_set),
                "val_count": len(val_set),
                "test_count": len(test_set),
                "train_percent": "70%",
                "val_percent": "15%",
                "test_percent": "15%"
            },
            "calibrated_threshold": best_threshold,
            "test_metrics": {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "true_positives": test_tp,
                "false_positives": test_fp,
                "false_negatives": test_fn,
                "true_negatives": test_tn,
                "total_test_alerts": len(test_alerts)
            }
        }

        return results

if __name__ == "__main__":
    trainer = DatasetTrainer(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    res = trainer.run_training_pipeline(num_logs=2000)
    print("\n[+] Train / Val / Test Evaluation Results:")
    import json
    print(json.dumps(res, indent=2))
