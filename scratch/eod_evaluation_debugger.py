"""
End-of-Day Evaluation Debugger & Label Alignment Fix.
Fast execution using 1,000-event training sample for baseline.
Verifies exact XGBoost class ordering, fixes evaluation label mapping,
evaluates Validation threshold grid [60..75], performs false-positive analysis,
and runs untouched test set ONCE with fixed label alignment.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, precision_recall_curve, roc_auc_score, auc

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint
from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.cold_start import ColdStartEngine
from src.anomaly_classifier import AttackClassifier

def run_eod_debugging():
    print("="*90)
    print("TASK 1 & TASK 2 — CLASS LABEL DECODING & TRUE TEST SUPPORT VERIFICATION")
    print("="*90)

    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)

    total = len(df_raw)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_df = df_raw.iloc[:train_end]
    val_df = df_raw.iloc[train_end:val_end]
    test_df = df_raw.iloc[val_end:]

    # Inspect AttackClassifier class ordering
    classifier = AttackClassifier()
    if classifier.model is not None:
        print(f"XGBoost model.classes_: {classifier.model.classes_}")
    else:
        print("XGBoost model: (Instantiated on-demand)")
    
    # Verify exact LabelEncoder / Sorted String Classes
    sorted_unique_labels = sorted(list(df_raw["label"].unique()))
    print("\nSorted Unique String Labels in Dataset:")
    for idx, lname in enumerate(sorted_unique_labels):
        print(f"  Numeric ID {idx} -> decoded class name: '{lname}'")

    print("\nTrue Test Support Breakdown (Total Test Events = len(test_df)):")
    test_counts = dict(Counter(test_df["label"]))
    total_test_support = 0
    print(f"{'Class Name':<30} | {'Exact Test Support':<20}")
    print("-" * 55)
    for lname in sorted_unique_labels:
        cnt = test_counts.get(lname, 0)
        total_test_support += cnt
        print(f"{lname:<30} | {cnt:<20}")
    print("-" * 55)
    print(f"{'TOTAL SUM':<30} | {total_test_support:<20} (Expected: {len(test_df)})")

    # ----------------------------------------------------
    # TASK 6 & 7: VALIDATION GRID SEARCH (THRESHOLDS 60..75 & FUSION WEIGHTS)
    # ----------------------------------------------------
    print("\n" + "="*90)
    print("TASK 6 & TASK 7 — VALIDATION GRID SEARCH (THRESHOLDS 60..75 & FUSION WEIGHTS)")
    print("="*90)

    fe_tr = FeatureExtractor()
    tr_rows = []
    # Sample 1000 events for fast baseline fitting
    for _, r in train_df.head(1000).iterrows():
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
    iso_model.fit(scaled_tr[train_df.head(1000)["label"] == "normal"])

    val_norm_scores = iso_model.score_samples(scaled_tr[train_df.head(1000)["label"] == "normal"])
    mu_val_if = float(np.mean(val_norm_scores))
    std_val_if = float(np.std(val_norm_scores))

    # Evaluate Validation Grid Search across weights [0.25..0.45] and thresholds [60..75]
    fe_val = FeatureExtractor()
    val_event_feats = []
    val_true_binary = []
    val_events = []

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
        val_event_feats.append(feats)
        val_events.append(evt)
        val_true_binary.append(0 if r["label"] == "normal" else 1)

    val_true_binary = np.array(val_true_binary)
    cs_engine = ColdStartEngine()

    print(f"\nEvaluating Fusion Weight Grid & High Thresholds on Validation Split (4,051 events):")
    print(f"{'w_global':<10} | {'Threshold T':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FPR':<10} | {'Alerts/1000':<15} | {'Recall@Top1%':<15}")
    print("-" * 105)

    best_grid_f1 = 0.0
    best_grid_w = 0.35
    best_grid_t = 65.0

    for w_g in [0.25, 0.30, 0.35, 0.40, 0.45]:
        v_risks = []
        for idx, feats in enumerate(val_event_feats):
            f_vals = np.array([feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)
            scaled_v = scaler.transform(f_vals.reshape(1, -1))[0]
            raw_if = float(iso_model.score_samples(scaled_v.reshape(1, -1))[0])
            z_if = (mu_val_if - raw_if) / (std_val_if * 2.0)
            g_if = float(1.0 / (1.0 + np.exp(-z_if)))

            dist = float(np.linalg.norm(scaled_v))
            d_score = float(1.0 - np.exp(-dist / 7.8839))

            peer = cs_engine.evaluate_peer_deviation(val_events[idx], feats)
            h_cnt = int(feats.get("history_count", 0))
            w_ind = min(1.0, h_cnt / 10.0)
            w_peer = 1.0 - w_ind

            context = (w_ind * d_score) + (w_peer * peer)
            r_val = round(((w_g * g_if) + ((1.0 - w_g) * context)) * 100.0, 1)
            v_risks.append(r_val)

        v_risks = np.array(v_risks)

        for t_val in [60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0]:
            preds_b = (v_risks >= t_val).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(val_true_binary, preds_b, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(val_true_binary, preds_b).ravel()
            fpr = (fp / float(fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
            alerts_per_1000 = (np.sum(preds_b) / float(len(preds_b))) * 1000.0

            top1_cutoff = np.percentile(v_risks, 99.0)
            top1_preds = (v_risks >= top1_cutoff).astype(int)
            tp_top1 = np.sum((top1_preds == 1) & (val_true_binary == 1))
            recall_top1 = (tp_top1 / float(np.sum(val_true_binary))) * 100.0 if np.sum(val_true_binary) > 0 else 0.0

            if f1 > best_grid_f1 and fpr <= 15.0:
                best_grid_f1 = f1
                best_grid_w = w_g
                best_grid_t = t_val

            print(f"{w_g:<10.2f} | {t_val:<12.1f} | {p*100:>9.2f}% | {r*100:>9.2f}% | {f1*100:>9.2f}% | {fpr:>9.2f}% | {alerts_per_1000:<15.1f} | {recall_top1:<15.2f}%")

    print(f"\nSelected Optimal Validation Parameters (FPR <= 15.0% Constraint):")
    print(f"  • Global Weight w_global: {best_grid_w:.2f}")
    print(f"  • Context Weight:         {1.0 - best_grid_w:.2f}")
    print(f"  • Threshold T:            {best_grid_t:.1f}")
    print(f"  • Validation F1-Score:    {best_grid_f1*100:.2f}%")

    # ----------------------------------------------------
    # TASK 5: STAGE-1 FALSE POSITIVE DIAGNOSTIC DISTRIBUTION ANALYSIS
    # ----------------------------------------------------
    print("\n" + "="*90)
    print("TASK 5 — STAGE-1 FALSE POSITIVE DIAGNOSTIC ANALYSIS (AT THRESHOLD T=65.0)")
    print("="*90)

    w_g_opt = best_grid_w
    t_opt = best_grid_t

    fp_if_list, fp_dist_list, fp_peer_list, fp_h_list, fp_risk_list = [], [], [], [], []
    tn_if_list, tn_dist_list, tn_peer_list, tn_h_list, tn_risk_list = [], [], [], [], []
    tp_if_list, tp_dist_list, tp_peer_list, tp_h_list, tp_risk_list = [], [], [], [], []

    for idx, feats in enumerate(val_event_feats):
        f_vals = np.array([feats.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)
        scaled_v = scaler.transform(f_vals.reshape(1, -1))[0]
        raw_if = float(iso_model.score_samples(scaled_v.reshape(1, -1))[0])
        z_if = (mu_val_if - raw_if) / (std_val_if * 2.0)
        g_if = float(1.0 / (1.0 + np.exp(-z_if))) * 100.0

        dist = float(np.linalg.norm(scaled_v))
        d_score = float(1.0 - np.exp(-dist / 7.8839)) * 100.0

        peer = cs_engine.evaluate_peer_deviation(val_events[idx], feats) * 100.0
        h_cnt = int(feats.get("history_count", 0))
        w_ind = min(1.0, h_cnt / 10.0)
        w_peer = 1.0 - w_ind

        context = (w_ind * (d_score/100.0)) + (w_peer * (peer/100.0))
        r_val = round(((w_g_opt * (g_if/100.0)) + ((1.0 - w_g_opt) * context)) * 100.0, 1)

        is_true_att = (val_true_binary[idx] == 1)
        is_flagged = (r_val >= t_opt)

        if not is_true_att and is_flagged:
            fp_if_list.append(g_if)
            fp_dist_list.append(d_score)
            fp_peer_list.append(peer)
            fp_h_list.append(h_cnt)
            fp_risk_list.append(r_val)
        elif not is_true_att and not is_flagged:
            tn_if_list.append(g_if)
            tn_dist_list.append(d_score)
            tn_peer_list.append(peer)
            tn_h_list.append(h_cnt)
            tn_risk_list.append(r_val)
        elif is_true_att and is_flagged:
            tp_if_list.append(g_if)
            tp_dist_list.append(d_score)
            tp_peer_list.append(peer)
            tp_h_list.append(h_cnt)
            tp_risk_list.append(r_val)

    print(f"Component Distributions (Mean Values):")
    print(f"{'Group':<30} | {'Count':<8} | {'IF Score':<12} | {'Dist Score':<12} | {'Peer Score':<12} | {'History Cnt':<12} | {'Final Risk':<12}")
    print("-" * 105)
    print(f"{'False Positive Normals':<30} | {len(fp_if_list):<8} | {np.mean(fp_if_list):<12.2f} | {np.mean(fp_dist_list):<12.2f} | {np.mean(fp_peer_list):<12.2f} | {np.mean(fp_h_list):<12.1f} | {np.mean(fp_risk_list):<12.1f}")
    print(f"{'Correct Normals (TN)':<30} | {len(tn_if_list):<8} | {np.mean(tn_if_list):<12.2f} | {np.mean(tn_dist_list):<12.2f} | {np.mean(tn_peer_list):<12.2f} | {np.mean(tn_h_list):<12.1f} | {np.mean(tn_risk_list):<12.1f}")
    print(f"{'True Attacks (TP)':<30} | {len(tp_if_list):<8} | {np.mean(tp_if_list):<12.2f} | {np.mean(tp_dist_list):<12.2f} | {np.mean(tp_peer_list):<12.2f} | {np.mean(tp_h_list):<12.1f} | {np.mean(tp_risk_list):<12.1f}")

    # ----------------------------------------------------
    # TASK 8: SINGLE FINAL RUN ON UNTOUCHED REAL TEST SPLIT
    # ----------------------------------------------------
    print("\n" + "="*90)
    print("TASK 8 — SINGLE FINAL RUN ON UNTOUCHED REAL TEST SPLIT (4,176 EVENTS)")
    print("="*90)
    print(f"Freezing Selected Parameters: w_global = {best_grid_w:.2f}, Threshold T = {best_grid_t:.1f}")

    fe_te = FeatureExtractor()
    test_event_scores = []
    test_true_binary = []

    e2e_true_decoded = []
    e2e_pred_decoded = []

    for _, r in test_df.iterrows():
        cmds = [c.strip() for c in str(r["commands"]).split(";") if c.strip()]
        geo = GeoLocation(ip=r["ip"], country=r["country"], city=r["city"], latitude=float(r["latitude"]), longitude=float(r["longitude"]))
        fp = DeviceFingerprint(os_version=r["os_version"], mac_address=r["mac_address"], protocol=r["protocol"], browser_or_agent=r["browser_agent"])
        vol = float(r.get("data_volume_mb", 0.0))
        true_label_str = r["label"]

        evt = AccessLogEvent(
            event_id=r["event_id"], entity_id=r["entity_id"], entity_type=r["entity_type"],
            timestamp=r["timestamp"], geo_location=geo, resource_accessed=r["resource_accessed"],
            auth_method=r["auth_method"], auth_success=bool(r["auth_success"]),
            session_duration=float(r["session_duration"]), command_sequence=cmds,
            device_fingerprint=fp, data_volume_mb=vol, label=true_label_str
        )

        feats = fe_te.extract_features(evt)
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
        final_risk = round(((best_grid_w * g_if) + ((1.0 - best_grid_w) * context)) * 100.0, 1)

        is_anom = final_risk >= best_grid_t

        test_event_scores.append(final_risk)
        test_true_binary.append(0 if true_label_str == "normal" else 1)

        e2e_true_decoded.append(true_label_str)

        if is_anom:
            pred_label = classifier.classify_anomaly(evt, feats, final_risk)
        else:
            pred_label = "normal"

        e2e_pred_decoded.append(pred_label)

    test_event_scores = np.array(test_event_scores)
    test_true_binary = np.array(test_true_binary)
    test_preds_binary = (test_event_scores >= best_grid_t).astype(int)

    # Metrics
    precisions_te, recalls_te, _ = precision_recall_curve(test_true_binary, test_event_scores / 100.0)
    pr_auc_te = auc(recalls_te, precisions_te)
    roc_auc_te = roc_auc_score(test_true_binary, test_event_scores / 100.0)

    p_s1, r_s1, f1_s1, _ = precision_recall_fscore_support(test_true_binary, test_preds_binary, average="binary", zero_division=0)
    tn_s1, fp_s1, fn_s1, tp_s1 = confusion_matrix(test_true_binary, test_preds_binary).ravel()
    fpr_s1 = (fp_s1 / float(fp_s1 + tn_s1)) * 100.0

    top_1_cutoff = np.percentile(test_event_scores, 99.0)
    top_1_preds = (test_event_scores >= top_1_cutoff).astype(int)
    tp_top1 = np.sum((top_1_preds == 1) & (test_true_binary == 1))
    recall_top1 = (tp_top1 / float(np.sum(test_true_binary))) * 100.0

    alerts_per_1000 = (np.sum(test_preds_binary) / float(len(test_preds_binary))) * 1000.0

    print("\n" + "="*80)
    print("FINAL REFACTORED STAGE-1 METRICS (UNTOUCHED REAL TEST SET)")
    print("="*80)
    print(f"• PR-AUC:                             {pr_auc_te:.4f}")
    print(f"• ROC-AUC:                            {roc_auc_te:.4f}")
    print(f"• Stage-1 Binary Precision:           {p_s1*100:.2f}%")
    print(f"• Stage-1 Binary Recall:              {r_s1*100:.2f}%")
    print(f"• Stage-1 Binary F1-Score:            {f1_s1*100:.2f}%")
    print(f"• Stage-1 False Positive Rate (FPR):  {fpr_s1:.2f}%")
    print(f"• Recall at Top 1% SOC Alert Budget:  {recall_top1:.2f}%")
    print(f"• Alerts per 1000 Events:             {alerts_per_1000:.1f}")

    # TASK 4: CORRECT END-TO-END METRICS WITH 100% PERFECT LABEL ALIGNMENT
    print("\n" + "="*80)
    print("TASK 4 — FINAL END-TO-END METRICS (100% PERFECT LABEL ALIGNMENT)")
    print("="*80)

    report_e2e = classification_report(
        e2e_true_decoded, e2e_pred_decoded,
        labels=sorted_unique_labels, target_names=sorted_unique_labels,
        output_dict=True, zero_division=0
    )
    cm_e2e = confusion_matrix(e2e_true_decoded, e2e_pred_decoded, labels=sorted_unique_labels)

    print(f"• End-to-End Accuracy:                {report_e2e['accuracy']*100:.2f}%")
    print(f"• End-to-End Macro Precision:         {report_e2e['macro avg']['precision']*100:.2f}%")
    print(f"• End-to-End Macro Recall:            {report_e2e['macro avg']['recall']*100:.2f}%")
    print(f"• End-to-End Macro F1-Score:          {report_e2e['macro avg']['f1-score']*100:.2f}%")

    print("\nPer-Class End-to-End Metrics Table (Matching True Test Support):")
    print(f"{'Category':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)
    for cat in sorted_unique_labels:
        m = report_e2e[cat]
        print(f"{cat:<30} | {m['precision']*100:>9.2f}% | {m['recall']*100:>9.2f}% | {m['f1-score']*100:>9.2f}% | {int(m['support']):<8}")

    print("\nTASK 3 — 8x8 END-TO-END CONFUSION MATRIX:")
    print("Rows: True Label | Columns: Predicted Label")
    print(f"{'':<28}" + "".join([f"{c[:8]:>10}" for c in sorted_unique_labels]))
    for i, cat in enumerate(sorted_unique_labels):
        print(f"{cat:<28}" + "".join([f"{cm_e2e[i, j]:>10}" for j in range(len(sorted_unique_labels))]))

if __name__ == "__main__":
    run_eod_debugging()
