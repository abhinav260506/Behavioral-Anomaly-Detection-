"""
SMOTE / SMOTETomek Resampling & Training Pipeline on the 27,000 Synthetic Telemetry Dataset.
Rebalances 27,008 raw logs to:
- Normal (Benign): 15.00% (26,415 samples)
- 7 Attack Categories: 85.00% total (21,383 samples each = 12.14% per attack category)
Total resampled dataset size: ~176,100 samples.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import math
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, Any, Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.feature_engineering import FeatureExtractor
from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint

np.random.seed(42)
torch.manual_seed(42)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 8):
        super(LSTMClassifier, self).__init__()
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

class SMOTE27KPipeline:
    def __init__(self, csv_path: str = "data/synthetic_telemetry_27k.csv", seed: int = 42):
        self.csv_path = os.path.abspath(csv_path)
        self.seed = seed
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.attack_categories = [
            "normal",
            "brute_force",
            "credential_stuffing",
            "lateral_movement",
            "low_and_slow_exfiltration",
            "device_spoofing",
            "impossible_travel",
            "insider_drift"
        ]
        self.label2id = {label: i for i, label in enumerate(self.attack_categories)}
        self.id2label = {i: label for i, label in enumerate(self.attack_categories)}

    def step1_load_and_clean_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Step 1: Load raw CSV, clean headers, strip spaces, extract numerical features."""
        print(f"[1] Loading & Cleaning CSV Dataset from: {self.csv_path}...")
        df_raw = pd.read_csv(self.csv_path)

        # Strip whitespace from column names and string values
        df_raw.columns = df_raw.columns.str.strip()
        for col in df_raw.select_dtypes(include=['object']).columns:
            df_raw[col] = df_raw[col].astype(str).str.strip()

        feature_rows = []
        labels = []

        print("    Extracting engineered sequential feature vectors...")
        for _, row in df_raw.iterrows():
            # Reconstruct AccessLogEvent for feature extractor
            cmds = [c.strip() for c in str(row["commands"]).split(";") if c.strip()]
            geo = GeoLocation(
                ip=row["ip"],
                country=row["country"],
                city=row["city"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"])
            )
            fp = DeviceFingerprint(
                os_version=row["os_version"],
                mac_address=row["mac_address"],
                protocol=row["protocol"],
                browser_or_agent=row["browser_agent"]
            )
            evt = AccessLogEvent(
                event_id=row["event_id"],
                entity_id=row["entity_id"],
                entity_type=row["entity_type"],
                timestamp=row["timestamp"],
                geo_location=geo,
                resource_accessed=row["resource_accessed"],
                auth_method=row["auth_method"],
                auth_success=bool(row["auth_success"]),
                session_duration=float(row["session_duration"]),
                command_sequence=cmds,
                device_fingerprint=fp,
                label=row["label"]
            )

            feats = self.feature_extractor.extract_features(evt)
            feature_rows.append(feats)
            labels.append(row["label"])

        df_feats = pd.DataFrame(feature_rows)
        labels_series = pd.Series(labels)

        # Drop NaN/Inf if any
        df_feats = df_feats.replace([np.inf, -np.inf], np.nan)
        df_feats = df_feats.fillna(df_feats.median())

        print(f"    [+] Successfully loaded & cleaned {len(df_feats)} rows with {len(df_feats.columns)} numerical features.")
        return df_feats, labels_series

    def step2_label_grouping(self, labels: pd.Series) -> pd.Series:
        """Step 2: Label Merging into unified categories."""
        print("[2] Grouping & Merging Subtype Labels into unified categories...")
        mapped = labels.map(lambda l: l if l in self.attack_categories else "lateral_movement")
        return mapped

    def step3_feature_scaling(self, X: pd.DataFrame) -> np.ndarray:
        """Step 3: Feature Scaling using StandardScaler."""
        print("[3] Feature Scaling (StandardScaler)...")
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def step4_apply_smote(self, X_scaled: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Step 4: SMOTE Resampling to achieve:
        - Normal: 15.00% (26,415 samples)
        - 7 Attack Categories: 85.00% total (21,383 samples each = 12.14% per attack category)
        Total resampled size: 176,096 samples.
        """
        print("[4] Applying SMOTE Resampling (Target: 15% Normal, 85% Attacks [12.14% each])...")
        counts_before = Counter(y)

        n_normal = counts_before[self.label2id["normal"]]
        # Target total size: N_total = n_normal / 0.15 = 176,100
        N_total = int(round(n_normal / 0.15))
        target_attack_total = N_total - n_normal
        target_per_attack = int(round(target_attack_total / 7.0))

        strategy = {self.label2id["normal"]: n_normal}
        for label, idx in self.label2id.items():
            if label != "normal":
                strategy[idx] = target_per_attack

        smote = SMOTE(sampling_strategy=strategy, k_neighbors=5, random_state=self.seed)
        X_res, y_res = smote.fit_resample(X_scaled, y)

        counts_after = Counter(y_res)
        total_res = len(y_res)

        before_stats = {}
        after_stats = {}

        for idx, name in self.id2label.items():
            cb = counts_before.get(idx, 0)
            ca = counts_after.get(idx, 0)
            before_stats[name] = {
                "count": cb,
                "pct": round(cb / len(y) * 100, 2)
            }
            after_stats[name] = {
                "count": ca,
                "pct": round(ca / total_res * 100, 2)
            }

        print(f"    [+] Resampling complete! Resampled dataset size: {total_res} rows.")
        return X_res, y_res, before_stats, after_stats

    def step5_train_models(self, X_res: np.ndarray, y_res: np.ndarray) -> Dict[str, Any]:
        """Step 5: Train Random Forest & PyTorch LSTM Classifiers."""
        print("[5] Training Machine Learning Models (Random Forest & PyTorch LSTM) on resampled dataset...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=self.seed, stratify=y_res)

        print("    Training Random Forest (100 Trees)...")
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=self.seed)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_rep = classification_report(y_test, rf_preds, target_names=self.attack_categories, output_dict=True)

        print("    Training PyTorch LSTM Classifier...")
        X_train_seq = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_seq = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_train_seq, y_train_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        lstm_model = LSTMClassifier(input_dim=X_train.shape[1], hidden_dim=64, num_classes=len(self.attack_categories))
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
            test_out = lstm_model(X_test_seq)
            lstm_preds = torch.argmax(test_out, dim=1).numpy()

        lstm_rep = classification_report(y_test, lstm_preds, target_names=self.attack_categories, output_dict=True)

        return {
            "rf_report": rf_rep,
            "lstm_report": lstm_rep,
            "cm_rf": confusion_matrix(y_test, rf_preds),
            "cm_lstm": confusion_matrix(y_test, lstm_preds),
            "y_test": y_test,
            "rf_preds": rf_preds,
            "lstm_preds": lstm_preds
        }

    def step6_visualize_results(self, before_stats: Dict, after_stats: Dict, eval_results: Dict):
        """Step 6: Generate Summary Charts & Plots."""
        print("[6] Generating Evaluation & Resampling Visualizations...")
        os.makedirs("static", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        categories = self.attack_categories
        before_pcts = [before_stats[cat]["pct"] for cat in categories]
        after_pcts = [after_stats[cat]["pct"] for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        axes[0].bar(x - width/2, before_pcts, width, label='Before SMOTE (27k Raw)', color='#ef4444')
        axes[0].bar(x + width/2, after_pcts, width, label='After SMOTE (15% Norm / 12.14% Atk)', color='#10b981')
        axes[0].set_ylabel('Percentage of Dataset (%)', fontsize=12)
        axes[0].set_title('27k Dataset Class Distribution Before vs After SMOTE', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        axes[0].axhline(y=15.0, color='blue', linestyle='--', label='Normal Target (15%)')
        axes[0].axhline(y=12.14, color='orange', linestyle=':', label='Attack Target (12.14%)')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        cm = eval_results["cm_rf"]
        im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1].set_title('Random Forest Confusion Matrix (Resampled Test Set)', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=axes[1])
        tick_marks = np.arange(len(categories))
        axes[1].set_xticks(tick_marks)
        axes[1].set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        axes[1].set_yticks(tick_marks)
        axes[1].set_yticklabels(categories, fontsize=9)
        axes[1].set_ylabel('True Label', fontsize=11)
        axes[1].set_xlabel('Predicted Label', fontsize=11)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

        plt.tight_layout()
        plot_path = os.path.abspath("static/smote_27k_evaluation_chart.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"    [+] Saved visualization plot to: {plot_path}")
        return plot_path

    def run_full_pipeline(self) -> Dict[str, Any]:
        df_feats, labels_raw = self.step1_load_and_clean_data()
        labels_grouped = self.step2_label_grouping(labels_raw)
        X_scaled = self.step3_feature_scaling(df_feats)

        y_numeric = labels_grouped.map(self.label2id).values

        X_res, y_res, before_stats, after_stats = self.step4_apply_smote(X_scaled, y_numeric)
        eval_results = self.step5_train_models(X_res, y_res)
        plot_path = self.step6_visualize_results(before_stats, after_stats, eval_results)

        return {
            "before_stats": before_stats,
            "after_stats": after_stats,
            "rf_macro_f1": round(eval_results["rf_report"]["macro avg"]["f1-score"], 4),
            "lstm_macro_f1": round(eval_results["lstm_report"]["macro avg"]["f1-score"], 4),
            "total_resampled_samples": len(y_res),
            "plot_path": plot_path
        }

if __name__ == "__main__":
    pipeline = SMOTE27KPipeline(csv_path="data/synthetic_telemetry_27k.csv", seed=42)
    output = pipeline.run_full_pipeline()
    print("\n" + "="*80)
    print("27,000 LOGS DATASET - SMOTE RESAMPLING SUMMARY TABLE:")
    print("="*80)
    print(f"{'Category':<28} | {'Before Count':<12} | {'Before %':<10} | {'After Count':<12} | {'After %':<10}")
    print("-" * 80)
    for cat in pipeline.attack_categories:
        b_cnt = output["before_stats"][cat]["count"]
        b_pct = output["before_stats"][cat]["pct"]
        a_cnt = output["after_stats"][cat]["count"]
        a_pct = output["after_stats"][cat]["pct"]
        print(f"{cat:<28} | {b_cnt:<12} | {b_pct:<9.2f}% | {a_cnt:<12} | {a_pct:<9.2f}%")
    print("-" * 80)
    print(f"Total Resampled Dataset Size: {output['total_resampled_samples']} rows")
    print(f"Random Forest Macro F1-Score: {output['rf_macro_f1']}")
    print(f"PyTorch LSTM Macro F1-Score:  {output['lstm_macro_f1']}")
    print("="*80)
