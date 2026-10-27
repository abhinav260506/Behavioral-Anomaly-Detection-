"""
SMOTE / SMOTETomek Resampling & Model Training Pipeline.
Rebalances security access log dataset according to user specification:
- Normal (Benign): 15.00% of total dataset
- 7 Attack Categories: 85.00% total (12.14% per attack category)
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

from src.generator import SyntheticDataGenerator
from src.feature_engineering import FeatureExtractor

# Set random seeds for reproducibility
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

class SMOTESecurityPipeline:
    def __init__(self, num_logs: int = 2000, seed: int = 42):
        self.num_logs = num_logs
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
        """Step 1: Generate & Clean raw telemetry data."""
        print("[1] Loading & Cleaning Telemetry Data...")
        generator = SyntheticDataGenerator(num_entities=30, seed=self.seed)
        events = generator.generate_full_dataset(num_normal=self.num_logs, inject_attacks=True)

        rows = []
        labels = []
        for evt in events:
            feats = self.feature_extractor.extract_features(evt)
            rows.append(feats)
            labels.append(evt.label)

        df = pd.DataFrame(rows)
        labels_series = pd.Series(labels)

        # Cleaning: Strip string whitespace if any, replace infinity / NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())

        print(f"    Raw Dataset Loaded: {len(df)} total rows across {len(df.columns)} numerical features.")
        return df, labels_series

    def step2_label_grouping(self, labels: pd.Series) -> pd.Series:
        """Step 2: Label Merging into 8 unified categories."""
        print("[2] Grouping & Merging Subtype Labels...")
        mapped = labels.map(lambda l: l if l in self.attack_categories else "lateral_movement")
        return mapped

    def step3_feature_scaling(self, X: pd.DataFrame) -> np.ndarray:
        """Step 3: Feature Scaling using StandardScaler."""
        print("[3] Feature Scaling (StandardScaler)...")
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def step4_apply_smote_tomek(self, X_scaled: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Step 4: SMOTE / SMOTETomek with custom target percentage:
        - Normal: 15.00%
        - 7 Attacks: 85.00% total (12.14% per attack type)
        """
        print("[4] Applying SMOTETomek Resampling (Target: 15% Normal, 85% Attacks [12.14% each])...")
        counts_before = Counter(y)

        # Before SMOTE: handle extreme singletons by duplicating rare classes so n_samples >= 4
        X_aug, y_aug = list(X_scaled), list(y)
        for class_id, count in counts_before.items():
            if count < 6:
                indices = [i for i, label in enumerate(y) if label == class_id]
                for _ in range(6 - count):
                    idx = np.random.choice(indices)
                    noise = np.random.normal(0, 0.01, size=X_scaled.shape[1])
                    X_aug.append(X_scaled[idx] + noise)
                    y_aug.append(class_id)

        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)

        n_normal = counts_before[self.label2id["normal"]]
        # Normal is 15% -> N_total = n_normal / 0.15
        N_total = int(round(n_normal / 0.15))
        target_attack_total = N_total - n_normal
        target_per_attack = int(round(target_attack_total / 7.0))

        strategy = {self.label2id["normal"]: n_normal}
        for label, idx in self.label2id.items():
            if label != "normal":
                strategy[idx] = target_per_attack

        smote = SMOTE(sampling_strategy=strategy, k_neighbors=3, random_state=self.seed)
        X_res, y_res = smote.fit_resample(X_aug, y_aug)

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

        return X_res, y_res, before_stats, after_stats

    def step5_train_models(self, X_res: np.ndarray, y_res: np.ndarray) -> Dict[str, Any]:
        """Step 5: Train Random Forest & PyTorch LSTM Classifiers."""
        print("[5] Training Machine Learning Models (Random Forest & PyTorch LSTM)...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=self.seed, stratify=y_res)

        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_rep = classification_report(y_test, rf_preds, target_names=self.attack_categories, output_dict=True)

        # 2. PyTorch LSTM Classifier
        X_train_seq = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_seq = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_train_seq, y_train_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        lstm_model = LSTMClassifier(input_dim=X_train.shape[1], hidden_dim=64, num_classes=len(self.attack_categories))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)

        lstm_model.train()
        for epoch in range(15):
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

        # Bar chart comparison
        axes[0].bar(x - width/2, before_pcts, width, label='Before SMOTE (Raw)', color='#ef4444')
        axes[0].bar(x + width/2, after_pcts, width, label='After SMOTE (15% Norm / 12.14% Atk)', color='#10b981')
        axes[0].set_ylabel('Percentage of Dataset (%)', fontsize=12)
        axes[0].set_title('Class Distribution Before vs After SMOTE', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        axes[0].axhline(y=15.0, color='blue', linestyle='--', label='Normal Target (15%)')
        axes[0].axhline(y=12.14, color='orange', linestyle=':', label='Attack Target (12.14%)')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Confusion Matrix Heatmap for Random Forest
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
        plot_path = os.path.abspath("static/smote_evaluation_chart.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"    [+] Saved visualization plot to: {plot_path}")
        return plot_path

    def run_full_pipeline(self) -> Dict[str, Any]:
        df, labels_raw = self.step1_load_and_clean_data()
        labels_grouped = self.step2_label_grouping(labels_raw)
        X_scaled = self.step3_feature_scaling(df)

        y_numeric = labels_grouped.map(self.label2id).values

        X_res, y_res, before_stats, after_stats = self.step4_apply_smote_tomek(X_scaled, y_numeric)
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
    pipeline = SMOTESecurityPipeline(num_logs=2000, seed=42)
    output = pipeline.run_full_pipeline()
    print("\n" + "="*80)
    print("SMOTE RESAMPLING SUMMARY TABLE:")
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
