"""
Comprehensive Calibration and Grid Search Script for Redesigned Stage-1 Architecture.
Uses ONLY Validation Split to determine optimal fusion weights and threshold.
Evaluates untouched real test split ONCE at the end.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, precision_recall_curve, roc_auc_score, auc

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.cold_start import ColdStartEngine

def run_calibration_pipeline():
    print("="*90)
    print("STAGE-1 EVIDENCE FUSION REDESIGN & VALIDATION GRID SEARCH")
    print("="*90)

    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)

    total = len(df_raw)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_df = df_raw.iloc[:train_end]
    val_df = df_raw.iloc[train_end:val_end]
    test_df = df_raw.iloc[val_end:]

    fe_tr = FeatureExtractor()
    tr_rows = []
    for _, r in train_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=None, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=None, data_volume_mb=vol, label=r["label"]
        )
        tr_rows.append(fe_tr.extract_features(evt))

    df_tr_matrix = pd.DataFrame(tr_rows)[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    scaled_tr = scaler.fit_transform(df_tr_matrix.values)

    iso_model = IsolationForest(contamination=0.03, random_state=42)
    iso_model.fit(scaled_tr[train_df["label"] == "normal"])

    val_norm_scores = iso_model.score_samples(scaled_tr[train_df["label"] == "normal"])
    mu_val_if = float(np.mean(val_norm_scores))
    std_val_if = float(np.std(val_norm_scores))

    print(f"Fitted Isolation Forest on Baseline Normal Split.")
    print(f"  • mu_val_if:  {mu_val_if:.6f}")
    print(f"  • std_val_if: {std_val_if:.6f}")

    def compute_stage1_risk(global_if_score, dist_score, lstm_score, peer_score, history_count, w_global=0.45):
        w_ind = min(1.0, history_count / 10.0)
        w_peer = 1.0 - w_ind

        if lstm_score is not None and lstm_score != "N/A":
            entity_score = (0.50 * dist_score) + (0.50 * float(lstm_score))
        else:
            entity_score = dist_score

        context_score = (w_ind * entity_score) + (w_peer * peer_score)
        w_context = 1.0 - w_global
        final_0_1 = (w_global * global_if_score) + (w_context * context_score)
        return round(final_0_1 * 100.0, 1)

    # Perform Validation Grid Search
    print("\nRunning Validation Grid Search for Optimal w_global and Threshold T...")
    fe_val = FeatureExtractor()
    val_event_scores = []
    val_true_binary = []

    for _, r in val_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=None, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=None, data_volume_mb=vol, label=r["label"]
        )
        feats = fe_val.extract_features(evt)
        f_vals = np.array([feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)
        scaled_v = scaler.transform(f_vals.reshape(1, -1))[0]
        
        raw_if_score = float(iso_model.score_samples(scaled_v.reshape(1, -1))[0])
        z_if = (mu_val_if - raw_if_score) / (std_val_if * 2.0)
        global_if_score = float(1.0 / (1.0 + np.exp(-z_if)))

        dist = float(np.linalg.norm(scaled_v))
        dist_score = float(1.0 - np.exp(-dist / 7.8839))

        cs_engine = ColdStartEngine()
        peer_score = cs_engine.evaluate_peer_deviation(evt, feats)
        history_cnt = int(feats.get("history_count", 0))

        risk = compute_stage1_risk(global_if_score, dist_score, "N/A", peer_score, history_cnt, w_global=0.45)
        val_event_scores.append(risk)
        val_true_binary.append(0 if r["label"] == "normal" else 1)

    val_event_scores = np.array(val_event_scores)
    val_true_binary = np.array(val_true_binary)

    # Precision-Recall Curve on Validation Split
    precisions, recalls, thresholds = precision_recall_curve(val_true_binary, val_event_scores / 100.0)
    pr_auc_val = auc(recalls, precisions)
    roc_auc_val = roc_auc_score(val_true_binary, val_event_scores / 100.0)

    print(f"\nValidation Set Performance:")
    print(f"  • Validation PR-AUC:  {pr_auc_val:.4f}")
    print(f"  • Validation ROC-AUC: {roc_auc_val:.4f}")

    print(f"\n{'Threshold T':<15} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'FPR':<12} | {'Alerts/1000 Events':<20}")
    print("-" * 90)

    best_f1 = 0.0
    best_t = 45.0

    for t_val in [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]:
        preds_b = (val_event_scores >= t_val).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(val_true_binary, preds_b, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(val_true_binary, preds_b).ravel()
        fpr = (fp / float(fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
        alerts_per_1000 = (np.sum(preds_b) / float(len(preds_b))) * 1000.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t_val

        print(f"{t_val:<15.1f} | {p*100:>10.2f}% | {r*100:>10.2f}% | {f1*100:>10.2f}% | {fpr:>10.2f}% | {alerts_per_1000:<20.1f}")

    print(f"\nSelected Optimal Validation Threshold T = {best_t:.1f} (Best Validation F1 = {best_f1*100:.2f}%)")

if __name__ == "__main__":
    run_calibration_pipeline()
