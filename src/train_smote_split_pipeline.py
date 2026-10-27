"""
Complete Model Training, Validation & Testing Pipeline on SMOTE-Resampled Dataset.
Splits 176,103 SMOTE-balanced dataset into:
- 70% Training Set (~123,272 samples)
- 15% Validation Set (~26,415 samples)
- 15% Testing Set (~26,416 samples)

Trains Random Forest, XGBoost, and PyTorch LSTM models on the new dataset.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, Any, Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.feature_engineering import FeatureExtractor
from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint

np.random.seed(42)
torch.manual_seed(42)

class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 8):
        super(PyTorchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class SMOTEFullSplitTrainer:
    def __init__(self, csv_path: str = "data/synthetic_telemetry_27k.csv", seed: int = 42):
        self.csv_path = os.path.abspath(csv_path)
        self.seed = seed
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.categories = [
            "normal",
            "brute_force",
            "credential_stuffing",
            "lateral_movement",
            "low_and_slow_exfiltration",
            "device_spoofing",
            "impossible_travel",
            "insider_drift"
        ]
        self.label2id = {cat: i for i, cat in enumerate(self.categories)}
        self.id2label = {i: cat for i, cat in enumerate(self.categories)}

    def step1_prepare_smote_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads raw dataset and applies SMOTE to generate the 176k balanced dataset."""
        print("[1] Loading raw CSV & generating 176k SMOTE balanced dataset (15% Normal / 85% Attacks)...")
        from imblearn.over_sampling import SMOTE

        df_raw = pd.read_csv(self.csv_path)
        df_raw.columns = df_raw.columns.str.strip()

        feature_rows = []
        labels = []

        for _, row in df_raw.iterrows():
            cmds = [c.strip() for c in str(row["commands"]).split(";") if c.strip()]
            geo = GeoLocation(ip=row["ip"], country=row["country"], city=row["city"], latitude=float(row["latitude"]), longitude=float(row["longitude"]))
            fp = DeviceFingerprint(os_version=row["os_version"], mac_address=row["mac_address"], protocol=row["protocol"], browser_or_agent=row["browser_agent"])
            evt = AccessLogEvent(
                event_id=row["event_id"], entity_id=row["entity_id"], entity_type=row["entity_type"],
                timestamp=row["timestamp"], geo_location=geo, resource_accessed=row["resource_accessed"],
                auth_method=row["auth_method"], auth_success=bool(row["auth_success"]),
                session_duration=float(row["session_duration"]), command_sequence=cmds,
                device_fingerprint=fp, label=row["label"]
            )
            feats = self.feature_extractor.extract_features(evt)
            feature_rows.append(feats)
            labels.append(row["label"])

        df_feats = pd.DataFrame(feature_rows)
        df_feats = df_feats.replace([np.inf, -np.inf], np.nan).fillna(df_feats.median())
        labels_series = pd.Series(labels)

        # Map labels
        y_numeric = labels_series.map(lambda l: self.label2id.get(l, self.label2id["lateral_movement"])).values

        # Scale features
        X_scaled = self.scaler.fit_transform(df_feats)

        # Apply SMOTE resampling
        counts_before = Counter(y_numeric)
        n_normal = counts_before[self.label2id["normal"]]
        N_total = int(round(n_normal / 0.15))
        target_attack_total = N_total - n_normal
        target_per_attack = int(round(target_attack_total / 7.0))

        strategy = {self.label2id["normal"]: n_normal}
        for cat, idx in self.label2id.items():
            if cat != "normal":
                strategy[idx] = target_per_attack

        smote = SMOTE(sampling_strategy=strategy, k_neighbors=5, random_state=self.seed)
        X_res, y_res = smote.fit_resample(X_scaled, y_numeric)

        print(f"    [+] SMOTE dataset created: {len(X_res)} total samples.")
        return X_res, y_res

    def step2_split_70_15_15(self, X_res: np.ndarray, y_res: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits SMOTE Dataset into:
        - 70% Training Set
        - 15% Validation Set
        - 15% Testing Set
        """
        print("[2] Partitioning SMOTE Dataset into 70% Train, 15% Val, and 15% Test...")
        # First split into 70% Train and 30% Temp (Val + Test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_res, y_res, test_size=0.30, random_state=self.seed, stratify=y_res
        )

        # Split 30% Temp into equal 15% Val and 15% Test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.seed, stratify=y_temp
        )

        print(f"    Train Split (70%): {len(X_train):,} samples")
        print(f"    Val Split   (15%): {len(X_val):,} samples")
        print(f"    Test Split  (15%): {len(X_test):,} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def step3_train_and_evaluate_models(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Trains Random Forest, XGBoost, and PyTorch LSTM models on 70% train split and evaluates on 15% test split."""
        print("\n[3] Training Machine Learning Models on 70% SMOTE Training Split...")

        # 1. Random Forest Classifier
        print("    [A] Training Random Forest (100 Trees)...")
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=self.seed)
        rf.fit(X_train, y_train)
        rf_val_preds = rf.predict(X_val)
        rf_test_preds = rf.predict(X_test)
        rf_rep = classification_report(y_test, rf_test_preds, target_names=self.categories, output_dict=True)

        # 2. XGBoost Classifier
        print("    [B] Training XGBoost Multi-Class Classifier...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(self.categories),
            n_jobs=-1,
            random_state=self.seed
        )
        xgb_clf.fit(X_train, y_train)
        xgb_val_preds = xgb_clf.predict(X_val)
        xgb_test_preds = xgb_clf.predict(X_test)
        xgb_rep = classification_report(y_test, xgb_test_preds, target_names=self.categories, output_dict=True)

        # 3. PyTorch LSTM Classifier
        print("    [C] Training PyTorch LSTM Sequence Model...")
        X_train_seq = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_seq = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
        X_test_seq = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(TensorDataset(X_train_seq, y_train_tensor), batch_size=512, shuffle=True)

        lstm_model = PyTorchLSTM(input_dim=X_train.shape[1], hidden_dim=64, num_classes=len(self.categories))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)

        lstm_model.train()
        for epoch in range(10):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = lstm_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        lstm_model.eval()
        with torch.no_grad():
            lstm_test_preds = torch.argmax(lstm_model(X_test_seq), dim=1).numpy()

        lstm_rep = classification_report(y_test, lstm_test_preds, target_names=self.categories, output_dict=True)

        print("\n[+] Model Training Complete!")
        print(f"    Random Forest  - Test Accuracy: {rf_rep['accuracy']*100:.2f}% | Macro F1: {rf_rep['macro avg']['f1-score']*100:.2f}%")
        print(f"    XGBoost        - Test Accuracy: {xgb_rep['accuracy']*100:.2f}% | Macro F1: {xgb_rep['macro avg']['f1-score']*100:.2f}%")
        print(f"    PyTorch LSTM   - Test Accuracy: {lstm_rep['accuracy']*100:.2f}% | Macro F1: {lstm_rep['macro avg']['f1-score']*100:.2f}%")

        return {
            "rf_rep": rf_rep,
            "xgb_rep": xgb_rep,
            "lstm_rep": lstm_rep,
            "cm_rf": confusion_matrix(y_test, rf_test_preds),
            "cm_xgb": confusion_matrix(y_test, xgb_test_preds),
            "cm_lstm": confusion_matrix(y_test, lstm_test_preds)
        }

    def step4_visualize_results(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, results: Dict):
        """Generates summary visualization charts."""
        os.makedirs("static", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # 1. Split Distribution Chart
        train_counts = Counter([self.id2label[i] for i in y_train])
        val_counts = Counter([self.id2label[i] for i in y_val])
        test_counts = Counter([self.id2label[i] for i in y_test])

        cats = self.categories
        x = np.arange(len(cats))
        w = 0.25

        axes[0].bar(x - w, [train_counts[c] for c in cats], w, label='70% Train Set', color='#3b82f6')
        axes[0].bar(x, [val_counts[c] for c in cats], w, label='15% Val Set', color='#f59e0b')
        axes[0].bar(x + w, [test_counts[c] for c in cats], w, label='15% Test Set', color='#10b981')

        axes[0].set_ylabel('Sample Count', fontsize=11)
        axes[0].set_title('70% Train / 15% Val / 15% Test Split Distribution\n(on 176k SMOTE Dataset)', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cats, rotation=45, ha='right', fontsize=8, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # 2. XGBoost Confusion Matrix on 15% Test Set
        cm = results["cm_xgb"]
        im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1].set_title('XGBoost Confusion Matrix on 15% Test Set', fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=axes[1])
        tick_marks = np.arange(len(cats))
        axes[1].set_xticks(tick_marks)
        axes[1].set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
        axes[1].set_yticks(tick_marks)
        axes[1].set_yticklabels(cats, fontsize=8)
        axes[1].set_ylabel('True Category Label', fontsize=10)
        axes[1].set_xlabel('Predicted Category Label', fontsize=10)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

        plt.tight_layout()
        plot_path = os.path.abspath("static/smote_full_70_15_15_eval.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[+] Saved evaluation plot to: {plot_path}")
        return plot_path

    def run_full_pipeline(self) -> Dict[str, Any]:
        X_res, y_res = self.step1_prepare_smote_dataset()
        X_tr, X_va, X_te, y_tr, y_va, y_te = self.step2_split_70_15_15(X_res, y_res)
        results = self.step3_train_and_evaluate_models(X_tr, X_va, X_te, y_tr, y_va, y_te)
        plot_path = self.step4_visualize_results(X_tr, X_va, X_te, y_tr, y_va, y_te, results)

        return {
            "total_smote_samples": len(y_res),
            "train_count": len(y_tr),
            "val_count": len(y_va),
            "test_count": len(y_te),
            "results": results,
            "plot_path": plot_path
        }

if __name__ == "__main__":
    trainer = SMOTEFullSplitTrainer(csv_path="data/synthetic_telemetry_27k.csv", seed=42)
    output = trainer.run_full_pipeline()
    print("\n" + "="*80)
    print("70% TRAIN / 15% VAL / 15% TEST PIPELINE ON SMOTE DATASET COMPLETED.")
    print("="*80)
