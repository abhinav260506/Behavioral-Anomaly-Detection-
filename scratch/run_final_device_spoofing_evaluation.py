"""
Final Evaluation of Refactored Pipeline with Enhanced Device Spoofing Training Sequences.
Evaluates ONLY after retraining on the updated TRAIN split.
Stage 1 parameters are frozen (w_global = 0.35, T = 65.0).
Evaluates untouched held-out test split ONCE.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, precision_recall_curve, roc_auc_score, auc
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.cold_start import ColdStartEngine
from src.anomaly_classifier import AttackClassifier

def run_final_evaluation():
    print("="*90)
    print("TASK 3 & TASK 5 — RETRAINING ON NEW TRAIN SPLIT & FINAL HELD-OUT TEST EVALUATION")
    print("="*90)

    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)

    total_len = len(df_raw)
    val_len = 4200
    test_len = 4200
    train_end = total_len - val_len - test_len
    val_end = total_len - test_len

    train_df = df_raw.iloc[:train_end].copy()
    val_df = df_raw.iloc[train_end:val_end].copy()
    test_df = df_raw.iloc[val_end:].copy()

    print(f"Dataset Split Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    sorted_unique_labels = sorted(list(df_raw["label"].unique()))
    label2id = {c: i for i, c in enumerate(sorted_unique_labels)}

    # 1. Feature Extraction on Training Split
    print("\n[1/3] Extracting 45 Stage-1 features on updated Training Split...")
    fe_tr = FeatureExtractor()
    tr_full_list = []

    for _, r in train_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=None, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=DeviceFingerprint(os_version=r["os_version"], mac_address=r["mac_address"], protocol=r["protocol"], browser_or_agent=r["browser_agent"]),
            data_volume_mb=vol, label=r["label"]
        )
        f_dict = fe_tr.extract_features(evt)
        f_dict["label"] = r["label"]
        tr_full_list.append(f_dict)

    df_tr_feats = pd.DataFrame(tr_full_list)
    df_tr_matrix = df_tr_feats[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 2. Fit Scaler & Isolation Forest Baseline on Normal Train Split
    scaler = StandardScaler()
    scaled_tr = scaler.fit_transform(df_tr_matrix.values)

    iso_model = IsolationForest(contamination=0.03, random_state=42)
    iso_model.fit(scaled_tr[df_tr_feats["label"] == "normal"])

    val_norm_scores = iso_model.score_samples(scaled_tr[df_tr_feats["label"] == "normal"])
    mu_val_if = float(np.mean(val_norm_scores))
    std_val_if = float(np.std(val_norm_scores))

    print("    [+] Fitted Isolation Forest baseline model on Normal Train Split.")
    print(f"        mu_val_if = {mu_val_if:.6f}, std_val_if = {std_val_if:.6f}")

    # 3. Retrain Stage-2 XGBoost Classifier on Resampled Train Split
    print("\n[2/3] Retraining Stage-2 XGBoost Classifier using SMOTE on Train Split ONLY...")
    y_tr = np.array([label2id[l] for l in df_tr_feats["label"]])
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_tr_res, y_tr_res = smote.fit_resample(df_tr_matrix.values, y_tr)

    new_xgb = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="mlogloss"
    )
    new_xgb.fit(X_tr_res, y_tr_res)
    print("    [+] Successfully trained Stage-2 XGBoost Multi-Class Classifier.")

    # 4. Evaluate Untouched Held-Out Test Split (4,200 events)
    print("\n[3/3] Evaluating Untouched Held-Out Test Split (4,200 events)...")
    fe_test = FeatureExtractor()
    cs_engine = ColdStartEngine()

    frozen_w_g = 0.35
    frozen_t = 65.0

    s1_true_binary = []
    s1_pred_binary = []
    test_event_risks = []

    e2e_true = []
    e2e_pred = []

    ds_detected_s1 = 0
    ds_total_test = 0

    for _, r in test_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        geo = GeoLocation(ip=r["ip"], country=r["country"], city=r["city"], latitude=float(r["latitude"]), longitude=float(r["longitude"]))
        fp = DeviceFingerprint(os_version=r["os_version"], mac_address=r["mac_address"], protocol=r["protocol"], browser_or_agent=r["browser_agent"])
        vol = float(r.get("data_volume_mb", 0.0))
        true_label = r["label"]

        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=geo, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=fp, data_volume_mb=vol, label=true_label
        )

        feats = fe_test.extract_features(evt)
        f_vals = np.array([feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)
        scaled_v = scaler.transform(f_vals.reshape(1, -1))[0]

        raw_if = float(iso_model.score_samples(scaled_v.reshape(1, -1))[0])
        z_if = (mu_val_if - raw_if) / (std_val_if * 2.0)
        g_if = float(1.0 / (1.0 + np.exp(-z_if)))

        dist = float(np.linalg.norm(scaled_v))
        d_score = float(1.0 - np.exp(-dist / 7.8839))

        peer = cs_engine.evaluate_peer_deviation(evt, feats)
        h_cnt = int(feats.get("history_count", 0))
        w_ind = min(1.0, h_cnt / 10.0)
        w_peer = 1.0 - w_ind

        context = (w_ind * d_score) + (w_peer * peer)
        final_risk = round(((frozen_w_g * g_if) + ((1.0 - frozen_w_g) * context)) * 100.0, 1)

        is_anom = final_risk >= frozen_t

        s1_true = 0 if true_label == "normal" else 1
        s1_pred = 1 if is_anom else 0
        s1_true_binary.append(s1_true)
        s1_pred_binary.append(s1_pred)
        test_event_risks.append(final_risk)

        if true_label == "device_spoofing":
            ds_total_test += 1
            if is_anom:
                ds_detected_s1 += 1

        e2e_true.append(true_label)
        if is_anom:
            f_arr = f_vals.reshape(1, -1)
            pred_id = new_xgb.predict(f_arr)[0]
            pred_label = sorted_unique_labels[pred_id]
        else:
            pred_label = "normal"

        e2e_pred.append(pred_label)

    # 5. Compute Metrics
    s1_true_binary = np.array(s1_true_binary)
    s1_pred_binary = np.array(s1_pred_binary)
    test_event_risks = np.array(test_event_risks)

    p_s1, r_s1, f1_s1, _ = precision_recall_fscore_support(s1_true_binary, s1_pred_binary, average="binary", zero_division=0)
    cm_s1 = confusion_matrix(s1_true_binary, s1_pred_binary)
    tn_s1, fp_s1, fn_s1, tp_s1 = cm_s1.ravel()
    fpr_s1 = (fp_s1 / float(fp_s1 + tn_s1)) * 100.0 if (fp_s1 + tn_s1) > 0 else 0.0

    precisions_s1, recalls_s1, _ = precision_recall_curve(s1_true_binary, test_event_risks / 100.0)
    pr_auc_s1 = auc(recalls_s1, precisions_s1)
    roc_auc_s1 = roc_auc_score(s1_true_binary, test_event_risks / 100.0)

    report_e2e = classification_report(e2e_true, e2e_pred, labels=sorted_unique_labels, target_names=sorted_unique_labels, output_dict=True, zero_division=0)
    cm_e2e = confusion_matrix(e2e_true, e2e_pred, labels=sorted_unique_labels)

    ds_e2e_tp = sum(1 for t, p in zip(e2e_true, e2e_pred) if t == "device_spoofing" and p == "device_spoofing")
    ds_e2e_rec = (ds_e2e_tp / float(ds_total_test) * 100.0) if ds_total_test > 0 else 0.0

    print("\n" + "="*80)
    print("STAGE-1 BINARY GATEWAY METRICS (UNTOUCHED HELD-OUT TEST SPLIT)")
    print("="*80)
    print(f"• True Positives (TP):   {tp_s1}")
    print(f"• False Positives (FP):  {fp_s1}")
    print(f"• True Negatives (TN):   {tn_s1}")
    print(f"• False Negatives (FN):  {fn_s1}")
    print(f"• Precision:             {p_s1*100:.2f}%")
    print(f"• Recall:                {r_s1*100:.2f}%")
    print(f"• F1-Score:              {f1_s1*100:.2f}%")
    print(f"• False Positive Rate:   {fpr_s1:.2f}%")
    print(f"• PR-AUC:                {pr_auc_s1:.4f}")
    print(f"• ROC-AUC:               {roc_auc_s1:.4f}")

    print("\n" + "="*80)
    print("DEVICE SPOOFING SPECIFIC EVALUATION SUMMARY")
    print("="*80)
    print(f"• Test Support:          {ds_total_test}")
    print(f"• Stage-1 Detected:      {ds_detected_s1} / {ds_total_test}")
    print(f"• Stage-1 Recall:        {(ds_detected_s1 / float(ds_total_test) * 100.0) if ds_total_test > 0 else 0.0:.2f}%")
    print(f"• End-to-End Recall:     {ds_e2e_rec:.2f}% ({ds_e2e_tp} / {ds_total_test})")

    print("\n" + "="*80)
    print("END-TO-END PIPELINE METRICS (UNTOUCHED HELD-OUT TEST SPLIT)")
    print("="*80)
    print(f"• Accuracy:              {report_e2e['accuracy']*100:.2f}%")
    print(f"• Macro Precision:       {report_e2e['macro avg']['precision']*100:.2f}%")
    print(f"• Macro Recall:          {report_e2e['macro avg']['recall']*100:.2f}%")
    print(f"• Macro F1-Score:        {report_e2e['macro avg']['f1-score']*100:.2f}%")

    print("\nPer-Class End-to-End Metrics Table:")
    print(f"{'Category':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)
    for cat in sorted_unique_labels:
        m = report_e2e[cat]
        print(f"{cat:<30} | {m['precision']*100:>9.2f}% | {m['recall']*100:>9.2f}% | {m['f1-score']*100:>9.2f}% | {int(m['support']):<8}")

    print("\n8x8 END-TO-END CONFUSION MATRIX:")
    print("Rows: True Label | Columns: Predicted Label")
    print(f"{'':<28}" + "".join([f"{c[:8]:>10}" for c in sorted_unique_labels]))
    for i, cat in enumerate(sorted_unique_labels):
        print(f"{cat:<28}" + "".join([f"{cm_e2e[i, j]:>10}" for j in range(len(sorted_unique_labels))]))

if __name__ == "__main__":
    run_final_evaluation()
