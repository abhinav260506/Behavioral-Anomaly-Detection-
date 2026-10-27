"""
Final Model Training & Serialization Script.
Responsibility: ONLY script allowed to fit models and save frozen artifacts.

Pipeline:
1. Load raw dataset (data/synthetic_telemetry_27k.csv).
2. Entity-grouped Train / Validation / Test split with ZERO entity overlap.
3. Fit StandardScaler on TRAIN split ONLY.
4. Fit IsolationForest on normal TRAIN samples ONLY.
5. Train PyTorch LSTM Autoencoder on normal TRAIN samples ONLY.
6. Apply SMOTE ONLY to Stage-2 TRAIN attack samples.
7. Train XGBoost Stage-2 multi-class attack classifier.
8. Use VALIDATION split ONLY for Stage-1 risk threshold calibration.
9. NEVER use TEST split for fitting, scaling, SMOTE, or threshold calibration.
10. Save frozen model artifacts and metadata to models/ directory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import joblib
import datetime
import hashlib
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Any, Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.feature_engineering import FeatureExtractor, STAGE1_FEATURE_NAMES
from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# STAGE 1: PyTorch LSTM Autoencoder
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
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def train_and_freeze_models(
    csv_path: str = "data/synthetic_telemetry_27k.csv",
    models_dir: str = "models",
    seed: int = SEED
):
    print("=" * 80)
    print("TRAIN_FINAL.PY — MODEL FITTING & SERIALIZATION PIPELINE")
    print("=" * 80)

    os.makedirs(models_dir, exist_ok=True)
    csv_path = os.path.abspath(csv_path)

    # 1. Load Raw Dataset & Perform Causal Feature Extraction
    print(f"[*] Step 1: Loading raw dataset from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    df_raw.columns = df_raw.columns.str.strip()

    extractor = FeatureExtractor()
    feature_rows = []
    labels = []
    entity_ids = []

    for _, row in df_raw.iterrows():
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
        labels.append(str(row["label"]))
        entity_ids.append(str(row["entity_id"]))

    df_feats = pd.DataFrame(feature_rows)
    df_feats = df_feats[STAGE1_FEATURE_NAMES]  # Ensure exact feature ordering
    df_feats = df_feats.replace([np.inf, -np.inf], np.nan).fillna(df_feats.median())
    labels_series = pd.Series(labels)
    entity_series = pd.Series(entity_ids)

    # 2. Entity-Grouped Train / Validation / Test Split (Zero Entity Overlap)
    print("\n[*] Step 2: Performing Entity-Grouped Train / Val / Test Split (Zero Entity Overlap)...")
    unique_entities = np.array(sorted(df_raw["entity_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_entities)

    num_entities = len(unique_entities)
    n_train = int(num_entities * 0.70)
    n_val = int(num_entities * 0.15)

    train_entities = sorted(list(unique_entities[:n_train]))
    val_entities = sorted(list(unique_entities[n_train:n_train + n_val]))
    test_entities = sorted(list(unique_entities[n_train + n_val:]))

    # Verify zero entity overlap
    overlap_tr_va = set(train_entities) & set(val_entities)
    overlap_tr_te = set(train_entities) & set(test_entities)
    overlap_va_te = set(val_entities) & set(test_entities)
    assert len(overlap_tr_va) == 0 and len(overlap_tr_te) == 0 and len(overlap_va_te) == 0, "Entity overlap detected!"

    print(f"    Train Entities ({len(train_entities)}): {train_entities[:5]}...")
    print(f"    Val Entities   ({len(val_entities)}): {val_entities[:5]}...")
    print(f"    Test Entities  ({len(test_entities)}): {test_entities[:5]}...")
    print(f"    Entity Overlap: 0 (Verified)")

    train_mask = entity_series.isin(train_entities)
    val_mask = entity_series.isin(val_entities)
    test_mask = entity_series.isin(test_entities)

    X_train_df = df_feats[train_mask].copy()
    y_train = labels_series[train_mask].copy()

    X_val_df = df_feats[val_mask].copy()
    y_val = labels_series[val_mask].copy()

    X_test_df = df_feats[test_mask].copy()
    y_test = labels_series[test_mask].copy()

    print(f"    Train Split: {len(X_train_df)} logs | Normal={sum(y_train=='normal')}, Attacks={sum(y_train!='normal')}")
    print(f"    Val Split:   {len(X_val_df)} logs | Normal={sum(y_val=='normal')}, Attacks={sum(y_val!='normal')}")
    print(f"    Test Split:  {len(X_test_df)} logs | Normal={sum(y_test=='normal')}, Attacks={sum(y_test!='normal')} [HELD-OUT]")

    # 3. Fit StandardScaler on TRAIN split ONLY
    print("\n[*] Step 3: Fitting StandardScaler on TRAIN split ONLY...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test_df)

    # 4. Fit IsolationForest on Normal TRAIN Samples ONLY
    print("\n[*] Step 4: Fitting IsolationForest on normal TRAIN samples ONLY...")
    train_normal_mask = (y_train == "normal")
    X_train_normal = X_train_scaled[train_normal_mask]

    iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=seed)
    iso_forest.fit(X_train_normal)

    # 5. Train PyTorch LSTM Autoencoder on Normal TRAIN Samples ONLY
    print("\n[*] Step 5: Training PyTorch LSTM Autoencoder on normal TRAIN samples ONLY...")
    input_dim = len(STAGE1_FEATURE_NAMES)
    autoencoder = LSTMAutoencoder(input_dim=input_dim, hidden_dim=32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.005)

    train_tensor = torch.tensor(X_train_normal, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)

    autoencoder.train()
    for epoch in range(10):
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon = autoencoder(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
    autoencoder.eval()

    # 6. Calibrate Stage-1 Threshold on VALIDATION Split ONLY
    print("\n[*] Step 6: Calibrating Stage-1 risk threshold on VALIDATION split ONLY...")
    def compute_stage1_scores(X_matrix):
        iso_scores = -iso_forest.score_samples(X_matrix)
        with torch.no_grad():
            tensor_x = torch.tensor(X_matrix, dtype=torch.float32).unsqueeze(1)
            recon_x = autoencoder(tensor_x)
            mse_scores = torch.mean((tensor_x - recon_x)**2, dim=[1, 2]).numpy()
        
        composite = 50.0 * (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-6) + \
                    50.0 * (mse_scores - mse_scores.min()) / (mse_scores.max() - mse_scores.min() + 1e-6)
        return composite

    val_scores = compute_stage1_scores(X_val_scaled)
    val_binary_truth = (y_val != "normal").astype(int).values

    best_th = 50.0
    best_f1 = 0.0
    for th in range(30, 85, 2):
        preds = (val_scores >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(val_binary_truth, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)

    print(f"    [+] Calibrated Validation Threshold: Cutoff = {best_th:.1f} (Val F1 = {best_f1:.4f})")

    # 7. Apply SMOTE ONLY to Stage-2 TRAIN Attack Samples & Fit XGBoost
    print("\n[*] Step 7: Applying SMOTE ONLY to Stage-2 TRAIN attack samples & training XGBoost...")
    attack_categories = [
        "brute_force", "credential_stuffing", "lateral_movement",
        "low_and_slow_exfiltration", "device_spoofing", "impossible_travel", "insider_drift"
    ]
    label2id = {label: i for i, label in enumerate(attack_categories)}
    id2label = {i: label for i, label in enumerate(attack_categories)}

    train_atk_mask = (y_train != "normal")
    X_train_atk = X_train_scaled[train_atk_mask]
    y_train_atk_labels = y_train[train_atk_mask]
    y_train_atk_numeric = y_train_atk_labels.map(label2id).values

    # Handle singletons before SMOTE
    X_aug, y_aug = list(X_train_atk), list(y_train_atk_numeric)
    for class_id, count in Counter(y_train_atk_numeric).items():
        if count < 6:
            indices = [i for i, c in enumerate(y_train_atk_numeric) if c == class_id]
            for _ in range(6 - count):
                idx = np.random.choice(indices)
                noise = np.random.normal(0, 0.01, size=X_train_atk.shape[1])
                X_aug.append(X_train_atk[idx] + noise)
                y_aug.append(class_id)

    smote = SMOTE(k_neighbors=3, random_state=seed)
    X_train_atk_res, y_train_atk_res = smote.fit_resample(np.array(X_aug), np.array(y_aug))

    print(f"    Stage 2 Attack Samples: Before SMOTE={len(X_train_atk)}, After SMOTE={len(X_train_atk_res)}")

    xgb_clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(attack_categories),
        random_state=seed
    )
    xgb_clf.fit(X_train_atk_res, y_train_atk_res)

    # 8. Save All Frozen Model Artifacts & Metadata to models/
    print("\n[*] Step 8: Saving frozen model artifacts and metadata to models/...")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    iso_path = os.path.join(models_dir, "isolation_forest.joblib")
    lstm_path = os.path.join(models_dir, "lstm_autoencoder.pth")
    xgb_path = os.path.join(models_dir, "xgboost_classifier.json")
    meta_path = os.path.join(models_dir, "metadata.json")
    split_path = os.path.join(models_dir, "train_val_test_split.json")

    joblib.dump(scaler, scaler_path)
    joblib.dump(iso_forest, iso_path)
    torch.save(autoencoder.state_dict(), lstm_path)
    xgb_clf.save_model(xgb_path)

    split_data = {
        "train_entity_ids": train_entities,
        "val_entity_ids": val_entities,
        "test_entity_ids": test_entities,
        "train_indices": [int(i) for i in X_train_df.index],
        "val_indices": [int(i) for i in X_val_df.index],
        "test_indices": [int(i) for i in X_test_df.index]
    }
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)

    import sklearn
    metadata = {
        "model_name": "Aegis AI Two-Stage Cybersecurity Anomaly Detector",
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": seed,
        "selected_stage1_threshold": best_th,
        "feature_count": len(STAGE1_FEATURE_NAMES),
        "feature_names": STAGE1_FEATURE_NAMES,
        "attack_categories": attack_categories,
        "label2id": label2id,
        "id2label": id2label,
        "train_entity_count": len(train_entities),
        "val_entity_count": len(val_entities),
        "test_entity_count": len(test_entities),
        "train_sample_count": len(X_train_df),
        "val_sample_count": len(X_val_df),
        "test_sample_count": len(X_test_df),
        "train_counts_by_label": dict(Counter(y_train)),
        "val_counts_by_label": dict(Counter(y_val)),
        "test_counts_by_label": dict(Counter(y_test)),
        "package_versions": {
            "joblib": joblib.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__
        }
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print SHA256 hashes of saved artifacts
    print("\n[+] FROZEN MODEL ARTIFACTS SERIALIZED SUCCESSFULLY:")
    print(f"    • {scaler_path} (SHA256: {calculate_sha256(scaler_path)})")
    print(f"    • {iso_path} (SHA256: {calculate_sha256(iso_path)})")
    print(f"    • {lstm_path} (SHA256: {calculate_sha256(lstm_path)})")
    print(f"    • {xgb_path} (SHA256: {calculate_sha256(xgb_path)})")
    print(f"    • {meta_path} (SHA256: {calculate_sha256(meta_path)})")
    print(f"    • {split_path}")
    print("=" * 80)

if __name__ == "__main__":
    train_and_freeze_models()
