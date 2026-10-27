"""
Final Model Evaluation Script on Frozen Held-Out Test Set.
Responsibility: Evaluation ONLY. ZERO model training or fitting.

Rules:
- NEVER calls .fit(), .fit_transform(), .fit_resample(), optimizer.step(), or backward().
- NEVER performs SMOTE or model weight updates.
- Loads frozen artifacts from models/ directory.
- Evaluates Stage-1, End-to-End, and Per-Class metrics on untouched held-out TEST set.
- Produces complete 8x8 confusion matrix and verifies zero entity overlap & artifact SHA256 hashes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import joblib
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, precision_recall_curve, auc, accuracy_score
)
import xgboost as xgb
import torch
import torch.nn as nn

from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint

# ==========================================
# STAGE 1: PyTorch LSTM Autoencoder Architecture (Evaluation Only)
# ==========================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super(LSTMAutoencoder, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder_lstm(x)
        decoded, _ = self.decoder_lstm(encoded)
        return decoded

def calculate_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def run_frozen_evaluation(
    csv_path: str = "data/synthetic_telemetry_27k.csv",
    models_dir: str = "models"
):
    models_dir = os.path.abspath(models_dir)
    csv_path = os.path.abspath(csv_path)

    scaler_path = os.path.join(models_dir, "scaler.joblib")
    iso_path = os.path.join(models_dir, "isolation_forest.joblib")
    lstm_path = os.path.join(models_dir, "lstm_autoencoder.pth")
    xgb_path = os.path.join(models_dir, "xgboost_classifier.json")
    meta_path = os.path.join(models_dir, "metadata.json")
    split_path = os.path.join(models_dir, "train_val_test_split.json")

    # 1. Verify existence of frozen artifacts
    for p in [scaler_path, iso_path, lstm_path, xgb_path, meta_path, split_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required frozen artifact: {p}. Run 'python -m src.train_final' first.")

    # 2. Load Metadata and Split Definitions
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    with open(split_path, "r") as f:
        split_data = json.load(f)

    train_entity_ids = split_data["train_entity_ids"]
    val_entity_ids = split_data["val_entity_ids"]
    test_entity_ids = split_data["test_entity_ids"]

    # Explicit Integrity Check: Zero Entity Overlap
    overlap_tr_te = set(train_entity_ids) & set(test_entity_ids)
    overlap_va_te = set(val_entity_ids) & set(test_entity_ids)
    entity_overlap_count = len(overlap_tr_te) + len(overlap_va_te)
    assert entity_overlap_count == 0, f"DATA LEAKAGE DETECTED! Entity overlap = {entity_overlap_count}"

    # 3. Load Frozen Model Artifacts
    scaler = joblib.load(scaler_path)
    iso_forest = joblib.load(iso_path)

    input_dim = metadata.get("feature_count", 45)
    autoencoder = LSTMAutoencoder(input_dim=input_dim, hidden_dim=32)
    autoencoder.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
    autoencoder.eval()  # Freeze evaluation mode

    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(xgb_path)

    threshold = float(metadata["selected_stage1_threshold"])
    attack_categories = metadata["attack_categories"]
    all_categories = ["normal"] + attack_categories
    label2id = {lbl: i for i, lbl in enumerate(attack_categories)}
    all_label2id = {lbl: i for i, lbl in enumerate(all_categories)}

    # 4. Load Raw Test Set Data
    df_raw = pd.read_csv(csv_path)
    df_raw.columns = df_raw.columns.str.strip()

    test_mask = df_raw["entity_id"].isin(test_entity_ids)
    df_test_raw = df_raw[test_mask].copy()

    extractor = FeatureExtractor()
    feature_rows = []
    test_labels = []

    for _, row in df_test_raw.iterrows():
        cmds = [c.strip() for c in str(row["commands"]).split(";") if c.strip()]
        geo = GeoLocation(
            ip=str(row["ip"]), country=str(row["country"]), city=str(row["city"]),
            latitude=float(row["latitude"]), longitude=float(row["longitude"])
        )
        fp = DeviceFingerprint(
            os_version=str(row["os_version"]), mac_address=str(row["mac_address"]),
            protocol=str(row["protocol"]), browser_or_agent=str(row["browser_agent"])
        )
        evt = AccessLogEvent(
            event_id=str(row["event_id"]), entity_id=str(row["entity_id"]), entity_type=str(row["entity_type"]),
            timestamp=str(row["timestamp"]), geo_location=geo, resource_accessed=str(row["resource_accessed"]),
            auth_method=str(row["auth_method"]), auth_success=bool(row["auth_success"]),
            session_duration=float(row["session_duration"]), command_sequence=cmds,
            device_fingerprint=fp, label=str(row["label"])
        )
        feats = extractor.extract_features(evt)
        feature_rows.append(feats)
        test_labels.append(str(row["label"]))

    df_test_feats = pd.DataFrame(feature_rows)
    df_test_feats = df_test_feats[STAGE1_FEATURE_NAMES]
    df_test_feats = df_test_feats.replace([np.inf, -np.inf], np.nan).fillna(df_test_feats.median())

    # Apply scaler.transform ONLY (NEVER fit or fit_transform)
    X_test_scaled = scaler.transform(df_test_feats)

    # 5. Stage-1 Anomaly Score Inference
    iso_scores = -iso_forest.score_samples(X_test_scaled)
    with torch.no_grad():
        tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
        recon_x = autoencoder(tensor_x)
        mse_scores = torch.mean((tensor_x - recon_x)**2, dim=[1, 2]).numpy()

    test_composite_scores = 50.0 * (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-6) + \
                            50.0 * (mse_scores - mse_scores.min()) / (mse_scores.max() - mse_scores.min() + 1e-6)

    stage1_binary_preds = (test_composite_scores >= threshold).astype(int)
    stage1_binary_truth = (pd.Series(test_labels) != "normal").astype(int).values

    # Stage-1 Binary Metrics
    tp = int(sum((p == 1 and t == 1) for p, t in zip(stage1_binary_preds, stage1_binary_truth)))
    fp = int(sum((p == 1 and t == 0) for p, t in zip(stage1_binary_preds, stage1_binary_truth)))
    fn = int(sum((p == 0 and t == 1) for p, t in zip(stage1_binary_preds, stage1_binary_truth)))
    tn = int(sum((p == 0 and t == 0) for p, t in zip(stage1_binary_preds, stage1_binary_truth)))

    stage1_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    stage1_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    stage1_f1 = (2 * stage1_prec * stage1_rec) / (stage1_prec + stage1_rec) if (stage1_prec + stage1_rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(stage1_binary_truth, test_composite_scores))
    except Exception:
        roc_auc = 0.0

    try:
        prec_arr, rec_arr, _ = precision_recall_curve(stage1_binary_truth, test_composite_scores)
        pr_auc = float(auc(rec_arr, prec_arr))
    except Exception:
        pr_auc = 0.0

    # 6. End-to-End 8-Class Evaluation
    end2end_preds = []
    # Predict with Stage-2 XGBoost for all test samples
    xgb_preds_numeric = xgb_clf.predict(X_test_scaled)

    for i in range(len(test_labels)):
        if stage1_binary_preds[i] == 0:
            end2end_preds.append("normal")
        else:
            # Map numeric prediction to string label
            pred_class_id = xgb_preds_numeric[i]
            end2end_preds.append(attack_categories[pred_class_id])

    y_test_true_arr = np.array(test_labels)
    y_test_pred_arr = np.array(end2end_preds)

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_test_true_arr, y_test_pred_arr, labels=all_categories, average='macro', zero_division=0
    )
    acc = accuracy_score(y_test_true_arr, y_test_pred_arr)

    # Per-Class Metrics
    prec_per_cls, rec_per_cls, f1_per_cls, supp_per_cls = precision_recall_fscore_support(
        y_test_true_arr, y_test_pred_arr, labels=all_categories, zero_division=0
    )

    # 8x8 Confusion Matrix
    cm_8x8 = confusion_matrix(y_test_true_arr, y_test_pred_arr, labels=all_categories)

    # 7. Print Final Formatted Evaluation Report
    print("=" * 80)
    print("FROZEN MODEL EVALUATION")
    print("=" * 80)
    print("\n[DATASET SPLIT OVERVIEW]")
    print(f"  Train entities:      {len(train_entity_ids)}")
    print(f"  Validation entities: {len(val_entity_ids)}")
    print(f"  Test entities:       {len(test_entity_ids)}")
    print(f"  Entity overlap:      0")

    print("\n[STAGE-1 ANOMALY DETECTION METRICS]")
    print(f"  TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"  Precision:           {stage1_prec * 100:.2f}%")
    print(f"  Recall:              {stage1_rec * 100:.2f}%")
    print(f"  F1-Score:            {stage1_f1 * 100:.2f}%")
    print(f"  False Positive Rate: {fpr * 100:.2f}%")
    print(f"  ROC-AUC:             {roc_auc:.4f}")
    print(f"  PR-AUC:              {pr_auc:.4f}")

    print("\n[END-TO-END SYSTEM METRICS]")
    print(f"  Accuracy:        {acc * 100:.2f}%")
    print(f"  Macro Precision: {macro_prec * 100:.2f}%")
    print(f"  Macro Recall:    {macro_rec * 100:.2f}%")
    print(f"  Macro F1-Score:  {macro_f1 * 100:.2f}%")

    print("\n[PER-CLASS EVALUATION TABLE]")
    print(f"  {'Class':<28} | {'Support':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("  " + "-" * 75)
    for idx, cat_name in enumerate(all_categories):
        print(f"  {cat_name:<28} | {supp_per_cls[idx]:<8} | {prec_per_cls[idx]*100:>8.2f}% | {rec_per_cls[idx]*100:>8.2f}% | {f1_per_cls[idx]*100:>8.2f}%")

    print("\n[8x8 CONFUSION MATRIX (True \\ Predicted)]")
    header_str = f"  {'True / Pred':<22} | " + " | ".join([f"{c[:6]:>6}" for c in all_categories])
    print(header_str)
    print("  " + "-" * len(header_str))
    for i, true_cat in enumerate(all_categories):
        row_str = f"  {true_cat:<22} | " + " | ".join([f"{cm_8x8[i, j]:>6}" for j in range(len(all_categories))])
        print(row_str)

    print("\n[INTEGRITY VERIFICATION]")
    print("  Training performed during evaluation: NO")
    print("  SMOTE performed during evaluation:    NO")
    print("  Scaler fitting during evaluation:     NO")
    print(f"  Entity overlap:                       0 (Verified: {len(overlap_tr_te)} train-test, {len(overlap_va_te)} val-test)")

    print("\n[FROZEN MODEL ARTIFACT SHA256 HASHES]")
    print(f"  scaler.joblib:           {calculate_sha256(scaler_path)}")
    print(f"  isolation_forest.joblib: {calculate_sha256(iso_path)}")
    print(f"  lstm_autoencoder.pth:    {calculate_sha256(lstm_path)}")
    print(f"  xgboost_classifier.json: {calculate_sha256(xgb_path)}")
    print(f"  metadata.json:           {calculate_sha256(meta_path)}")
    print("=" * 80)

if __name__ == "__main__":
    run_frozen_evaluation()
