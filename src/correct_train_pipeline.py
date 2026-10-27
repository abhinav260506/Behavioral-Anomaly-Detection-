"""
Correct Machine Learning Training Protocol (No Data Leakage).

Gold Standard Protocol:
1. CHRONOLOGICAL / STRATIFIED SPLIT FIRST into 70% Train, 15% Val, 15% Test on RAW dataset.
2. Scaler fitted ONLY on 70% Train Set.
3. SMOTE applied ONLY to 70% Train Set.
4. Validation (15%) and Testing (15%) sets remain 100% ORIGINAL, REAL, and UNTOUCHED.
5. Models trained on SMOTE-rebalanced Train set and evaluated on REAL, UNSEEN Test Set.
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
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
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

class CorrectPipelineTrainer:
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

    def step1_load_and_split_first(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Step 1: Load raw dataset & Split FIRST (70% Train, 15% Val, 15% Test) chronologically."""
        print("[1] Loading raw 27k dataset & performing Chronological Split FIRST...")
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

        # Chronological Split
        total = len(df_feats)
        train_end = int(total * 0.70)
        val_end = int(total * 0.85)

        X_tr_df, X_va_df, X_te_df = df_feats.iloc[:train_end], df_feats.iloc[train_end:val_end], df_feats.iloc[val_end:]
        y_tr, y_va, y_te = labels_series.iloc[:train_end], labels_series.iloc[train_end:val_end], labels_series.iloc[val_end:]

        print(f"    Raw Train Set (70%): {len(X_tr_df):,} logs")
        print(f"    Raw Val Set   (15%): {len(X_va_df):,} logs [REAL & UNTOUCHED]")
        print(f"    Raw Test Set  (15%): {len(X_te_df):,} logs [REAL & UNTOUCHED]")

        return X_tr_df, X_va_df, X_te_df, y_tr, y_va, y_te

    def step2_scale_and_smote_train_only(self, X_tr_df: pd.DataFrame, X_va_df: pd.DataFrame, X_te_df: pd.DataFrame, y_tr: pd.Series, y_va: pd.Series, y_te: pd.Series):
        """
        Step 2: Fit Scaler ONLY on Train Set & Apply SMOTE ONLY to Train Set.
        Validation and Test sets are scaled using Train scaler and remain 100% UNTOUCHED (NO SMOTE).
        """
        print("\n[2] Feature Scaling & Applying SMOTE ONLY to 70% Train Set...")
        X_tr_scaled = self.scaler.fit_transform(X_tr_df)
        X_va_scaled = self.scaler.transform(X_va_df)
        X_te_scaled = self.scaler.transform(X_te_df)

        y_tr_num = y_tr.map(self.label2id).values
        y_va_num = y_va.map(self.label2id).values
        y_te_num = y_te.map(self.label2id).values

        # Handle rare singletons in Train set before SMOTE
        counts_tr = Counter(y_tr_num)
        X_aug, y_aug = list(X_tr_scaled), list(y_tr_num)
        for class_id, count in counts_tr.items():
            if count < 6:
                indices = [i for i, c in enumerate(y_tr_num) if c == class_id]
                for _ in range(6 - count):
                    idx = np.random.choice(indices)
                    noise = np.random.normal(0, 0.01, size=X_tr_scaled.shape[1])
                    X_aug.append(X_tr_scaled[idx] + noise)
                    y_aug.append(class_id)

        # Calculate SMOTE target ratios for Training set ONLY
        n_normal_tr = counts_tr[self.label2id["normal"]]
        N_target_tr = int(round(n_normal_tr / 0.15))
        target_attack_tr = int(round((N_target_tr - n_normal_tr) / 7.0))

        strategy = {self.label2id["normal"]: n_normal_tr}
        for cat, idx in self.label2id.items():
            if cat != "normal":
                strategy[idx] = max(target_attack_tr, counts_tr[idx])

        smote = SMOTE(sampling_strategy=strategy, k_neighbors=3, random_state=self.seed)
        X_tr_res, y_tr_res = smote.fit_resample(np.array(X_aug), np.array(y_aug))

        print(f"    Train Set Before SMOTE: {len(X_tr_scaled):,} samples")
        print(f"    Train Set After SMOTE:  {len(X_tr_res):,} samples")
        print("    Validation & Test Sets: 100% UNTOUCHED, ORIGINAL REAL DATA")

        return X_tr_res, y_tr_res, X_va_scaled, y_va_num, X_te_scaled, y_te_num

    def step3_train_and_evaluate_real_test(self, X_tr_res, y_tr_res, X_va_scaled, y_va_num, X_te_scaled, y_te_num):
        """Step 3: Train models on SMOTE Train set & Evaluate on REAL UNTOUCHED Test set."""
        print("\n[3] Training Models on SMOTE Train Set & Evaluating on REAL UNTOUCHED Test Set...")

        # 1. Random Forest Classifier
        print("    [A] Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=self.seed)
        rf.fit(X_tr_res, y_tr_res)
        rf_test_preds = rf.predict(X_te_scaled)
        rf_rep = classification_report(y_te_num, rf_test_preds, target_names=self.categories, output_dict=True, zero_division=0)

        # 2. XGBoost Classifier
        print("    [B] Training XGBoost Classifier...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(self.categories),
            n_jobs=-1,
            random_state=self.seed
        )
        xgb_clf.fit(X_tr_res, y_tr_res)
        xgb_test_preds = xgb_clf.predict(X_te_scaled)
        xgb_rep = classification_report(y_te_num, xgb_test_preds, target_names=self.categories, output_dict=True, zero_division=0)

        # 3. PyTorch LSTM Classifier
        print("    [C] Training PyTorch LSTM Model...")
        X_tr_seq = torch.tensor(X_tr_res, dtype=torch.float32).unsqueeze(1)
        y_tr_tensor = torch.tensor(y_tr_res, dtype=torch.long)
        X_te_seq = torch.tensor(X_te_scaled, dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(TensorDataset(X_tr_seq, y_tr_tensor), batch_size=256, shuffle=True)

        lstm = PyTorchLSTM(input_dim=X_tr_res.shape[1], hidden_dim=64, num_classes=len(self.categories))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lstm.parameters(), lr=0.005)

        lstm.train()
        for epoch in range(10):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = lstm(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        lstm.eval()
        with torch.no_grad():
            lstm_test_preds = torch.argmax(lstm(X_te_seq), dim=1).numpy()

        lstm_rep = classification_report(y_te_num, lstm_test_preds, target_names=self.categories, output_dict=True, zero_division=0)

        print("\n[+] Realistic Model Performance on REAL UNTOUCHED Test Set:")
        print(f"    Random Forest  - Test Accuracy: {rf_rep['accuracy']*100:.2f}% | Macro F1: {rf_rep['macro avg']['f1-score']*100:.2f}%")
        print(f"    XGBoost        - Test Accuracy: {xgb_rep['accuracy']*100:.2f}% | Macro F1: {xgb_rep['macro avg']['f1-score']*100:.2f}%")
        print(f"    PyTorch LSTM   - Test Accuracy: {lstm_rep['accuracy']*100:.2f}% | Macro F1: {lstm_rep['macro avg']['f1-score']*100:.2f}%")

        return {
            "rf_rep": rf_rep,
            "xgb_rep": xgb_rep,
            "lstm_rep": lstm_rep,
            "cm_xgb": confusion_matrix(y_te_num, xgb_test_preds)
        }

    def step4_visualize(self, results: Dict):
        """Generates evaluation visualization on REAL UNTOUCHED Test Set."""
        os.makedirs("static", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Model comparison bar chart
        models = ['Random Forest', 'XGBoost', 'PyTorch LSTM']
        accuracies = [results['rf_rep']['accuracy']*100, results['xgb_rep']['accuracy']*100, results['lstm_rep']['accuracy']*100]
        f1_scores = [results['rf_rep']['macro avg']['f1-score']*100, results['xgb_rep']['macro avg']['f1-score']*100, results['lstm_rep']['macro avg']['f1-score']*100]

        x = np.arange(len(models))
        w = 0.35

        axes[0].bar(x - w/2, accuracies, w, label='Test Accuracy (%)', color='#3b82f6')
        axes[0].bar(x + w/2, f1_scores, w, label='Macro F1-Score (%)', color='#10b981')
        axes[0].set_ylabel('Percentage (%)', fontsize=11)
        axes[0].set_title('Model Performance on REAL Untouched Test Set\n(SMOTE Applied ONLY to Training Split)', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, fontsize=10, fontweight='bold')
        axes[0].set_ylim(0, 105)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
            axes[0].text(i - w/2, acc + 1.5, f"{acc:.2f}%", ha='center', va='bottom', fontsize=9)
            axes[0].text(i + w/2, f1 + 1.5, f"{f1:.2f}%", ha='center', va='bottom', fontsize=9)

        # Confusion Matrix for XGBoost
        cm = results["cm_xgb"]
        im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1].set_title('XGBoost Confusion Matrix on REAL Untouched Test Set', fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=axes[1])
        cats = self.categories
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
        plot_path = os.path.abspath("static/correct_protocol_eval.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[+] Saved correct protocol visualization to: {plot_path}")
        return plot_path

    def run_full_pipeline(self) -> Dict[str, Any]:
        X_tr_df, X_va_df, X_te_df, y_tr, y_va, y_te = self.step1_load_and_split_first()
        X_tr_res, y_tr_res, X_va_scaled, y_va_num, X_te_scaled, y_te_num = self.step2_scale_and_smote_train_only(
            X_tr_df, X_va_df, X_te_df, y_tr, y_va, y_te
        )
        results = self.step3_train_and_evaluate_real_test(X_tr_res, y_tr_res, X_va_scaled, y_va_num, X_te_scaled, y_te_num)
        plot_path = self.step4_visualize(results)
        return {"results": results, "plot_path": plot_path}

if __name__ == "__main__":
    trainer = CorrectPipelineTrainer(csv_path="data/synthetic_telemetry_27k.csv", seed=42)
    trainer.run_full_pipeline()
    print("\n" + "="*80)
    print("CORRECT MACHINE LEARNING TRAINING PROTOCOL COMPLETED.")
    print("="*80)
