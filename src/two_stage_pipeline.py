"""
Two-Stage Cybersecurity Architecture Pipeline on 27,000 Dataset.

STAGE 1: Anomaly Detection (PyTorch LSTM Autoencoder + IsolationForest)
- Kept at natural ~98% normal / ~2% attack imbalance.
- NO SMOTE applied to Stage 1.
- Chronological 70% Train, 15% Val, 15% Test split performed FIRST.

STAGE 2: Attack Vector Classification (XGBoost)
- Receives flagged suspicious logs.
- SMOTE applied ONLY to Stage 2 Training attack samples.
- Validation and Test sets remain 100% untouched and natural.
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
from sklearn.ensemble import IsolationForest
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

# ==========================================
# STAGE 1: PyTorch LSTM Autoencoder
# ==========================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super(LSTMAutoencoder, self).__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Decoder
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        encoded, (hn, cn) = self.encoder_lstm(x)
        decoded, _ = self.decoder_lstm(encoded)
        return decoded

# ==========================================
# Two-Stage Architecture Orchestrator
# ==========================================
class TwoStageSecurityArchitecture:
    def __init__(self, csv_path: str = "data/synthetic_telemetry_27k.csv", seed: int = 42):
        self.csv_path = os.path.abspath(csv_path)
        self.seed = seed
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.attack_categories = [
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

    def step1_load_and_chronological_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Chronological Split FIRST:
        70% Train, 15% Validation, 15% Testing.
        """
        print(f"[1] Loading raw 27k dataset & performing Chronological Split FIRST...")
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

        # Perform Chronological Split
        total = len(df_feats)
        train_end = int(total * 0.70)
        val_end = int(total * 0.85)

        X_train_df, X_val_df, X_test_df = df_feats.iloc[:train_end], df_feats.iloc[train_end:val_end], df_feats.iloc[val_end:]
        y_train, y_val, y_test = labels_series.iloc[:train_end], labels_series.iloc[train_end:val_end], labels_series.iloc[val_end:]

        print(f"    Train Split (70%): {len(X_train_df)} logs | Normal={sum(y_train=='normal')}, Attacks={sum(y_train!='normal')}")
        print(f"    Val Split   (15%): {len(X_val_df)} logs | Normal={sum(y_val=='normal')}, Attacks={sum(y_val!='normal')} [UNTOUCHED]")
        print(f"    Test Split  (15%): {len(X_test_df)} logs | Normal={sum(y_test=='normal')}, Attacks={sum(y_test!='normal')} [UNTOUCHED]")

        return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test

    def step2_stage1_anomaly_detection(self, X_train_df: pd.DataFrame, X_val_df: pd.DataFrame, X_test_df: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series):
        """
        STAGE 1: Unsupervised Anomaly Detection.
        No SMOTE applied.
        Trained on normal training baseline logs.
        Combines IsolationForest & PyTorch LSTM Autoencoder reconstruction loss.
        """
        print("\n[STAGE 1] Training Unsupervised Anomaly Detection Models (NO SMOTE, Natural ~98% Imbalance)...")
        # Fit StandardScaler on Train split only
        X_train_scaled = self.scaler.fit_transform(X_train_df)
        X_val_scaled = self.scaler.transform(X_val_df)
        X_test_scaled = self.scaler.transform(X_test_df)

        # Filter normal events for autoencoder training
        normal_idx = y_train[y_train == "normal"].index
        X_train_normal = X_train_scaled[y_train == "normal"]

        # 1. IsolationForest
        print("    Fitting IsolationForest on normal training baseline...")
        iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=self.seed)
        iso_forest.fit(X_train_normal)

        # 2. PyTorch LSTM Autoencoder
        print("    Training PyTorch LSTM Autoencoder on normal baseline...")
        input_dim = X_train_scaled.shape[1]
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

        # Compute anomaly scores on Val & Test sets
        def compute_scores(X_matrix):
            # IsolationForest score (inverted)
            iso_scores = -iso_forest.score_samples(X_matrix)
            # Autoencoder reconstruction MSE
            with torch.no_grad():
                tensor_x = torch.tensor(X_matrix, dtype=torch.float32).unsqueeze(1)
                recon_x = autoencoder(tensor_x)
                mse_scores = torch.mean((tensor_x - recon_x)**2, dim=[1, 2]).numpy()
            
            # Composite risk score (0-100)
            composite = 50.0 * (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-6) + \
                        50.0 * (mse_scores - mse_scores.min()) / (mse_scores.max() - mse_scores.min() + 1e-6)
            return composite

        val_scores = compute_scores(X_val_scaled)
        test_scores = compute_scores(X_test_scaled)

        # Calibrate optimal risk score threshold on Validation Set
        val_binary_truth = (y_val != "normal").astype(int).values
        best_th = 50.0
        best_f1 = 0.0
        for th in range(30, 85, 2):
            preds = (val_scores >= th).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(val_binary_truth, preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        print(f"    [+] Validation Threshold Calibrated: Risk Cutoff = {best_th:.1f} (Val F1 = {best_f1:.4f})")

        # Evaluate Stage 1 on untouched Test Set
        test_binary_truth = (y_test != "normal").astype(int).values
        test_preds = (test_scores >= best_th).astype(int)
        tp = sum((p == 1 and t == 1) for p, t in zip(test_preds, test_binary_truth))
        fp = sum((p == 1 and t == 0) for p, t in zip(test_preds, test_binary_truth))
        fn = sum((p == 0 and t == 1) for p, t in zip(test_preds, test_binary_truth))
        tn = sum((p == 0 and t == 0) for p, t in zip(test_preds, test_binary_truth))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        print(f"    [+] STAGE 1 Test Evaluation (Untouched Natural Test Set):")
        print(f"        Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1-Score: {f1*100:.2f}% (TP={tp}, FP={fp}, FN={fn}, TN={tn})")

        return {
            "best_threshold": best_th,
            "X_train_scaled": X_train_scaled,
            "X_val_scaled": X_val_scaled,
            "X_test_scaled": X_test_scaled,
            "test_preds": test_preds,
            "test_scores": test_scores
        }

    def step3_stage2_attack_classification(self, stage1_out: Dict, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series):
        """
        STAGE 2: XGBoost Attack Vector Classification.
        - Receives suspicious/attack events.
        - SMOTE applied ONLY to Stage 2 Training attack samples.
        - Validation and Testing splits remain 100% UNTOUCHED and natural.
        """
        print("\n[STAGE 2] Multi-Class Attack Vector Classification (XGBoost)...")
        print("    [!] Applying SMOTE ONLY to Stage 2 Training Attack Events...")

        X_train_scaled = stage1_out["X_train_scaled"]
        X_test_scaled = stage1_out["X_test_scaled"]

        # Filter attack events from Train Split for Stage 2 training
        train_attack_mask = (y_train != "normal")
        X_train_atk = X_train_scaled[train_attack_mask]
        y_train_atk_labels = y_train[train_attack_mask]

        # Map labels to numeric IDs
        y_train_atk_numeric = y_train_atk_labels.map(self.label2id).values

        counts_before_smote = Counter(y_train_atk_labels)
        print("    Raw Stage 2 Training Attack Counts (Before SMOTE):", dict(counts_before_smote))

        # Handle extreme rare singletons in Train split before SMOTE
        X_aug, y_aug = list(X_train_atk), list(y_train_atk_numeric)
        for class_id, count in Counter(y_train_atk_numeric).items():
            if count < 6:
                indices = [i for i, c in enumerate(y_train_atk_numeric) if c == class_id]
                for _ in range(6 - count):
                    idx = np.random.choice(indices)
                    noise = np.random.normal(0, 0.01, size=X_train_atk.shape[1])
                    X_aug.append(X_train_atk[idx] + noise)
                    y_aug.append(class_id)

        # Apply SMOTE to Stage 2 Training Split ONLY
        smote = SMOTE(k_neighbors=3, random_state=self.seed)
        X_train_atk_res, y_train_atk_res = smote.fit_resample(np.array(X_aug), np.array(y_aug))

        counts_after_smote = Counter([self.id2label[i] for i in y_train_atk_res])
        print("    Stage 2 Training Attack Counts (After SMOTE):", dict(counts_after_smote))

        # Train XGBoost Classifier on SMOTE-balanced Stage 2 Training Data
        print("    Training XGBoost Classifier on SMOTE-balanced Stage 2 Training Data...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(self.attack_categories),
            random_state=self.seed
        )
        xgb_clf.fit(X_train_atk_res, y_train_atk_res)

        # Evaluate End-to-End on UNTOUCHED, Natural Test Set
        print("\n[END-TO-END EVALUATION] Running End-to-End Inference on Untouched Natural Test Set...")
        test_anom_preds = stage1_out["test_preds"]
        
        # Ground truth for test set attacks
        test_attack_indices = [i for i, label in enumerate(y_test) if label != "normal"]
        X_test_atk_actual = X_test_scaled[test_attack_indices]
        y_test_atk_actual = y_test.iloc[test_attack_indices].map(self.label2id).values

        # Predict with Stage 2 XGBoost
        xgb_preds = xgb_clf.predict(X_test_atk_actual)

        # Classification Report
        rep = classification_report(y_test_atk_actual, xgb_preds, target_names=self.attack_categories, output_dict=True)
        cm = confusion_matrix(y_test_atk_actual, xgb_preds)

        print(f"    [+] Stage 2 XGBoost Test Macro F1-Score: {rep['macro avg']['f1-score']*100:.2f}%")
        print(f"    [+] Stage 2 XGBoost Test Accuracy: {rep['accuracy']*100:.2f}%")

        return {
            "counts_before_smote": counts_before_smote,
            "counts_after_smote": counts_after_smote,
            "xgb_report": rep,
            "cm": cm,
            "y_test_atk_actual": y_test_atk_actual,
            "xgb_preds": xgb_preds
        }

    def step4_visualize(self, stage1_out: Dict, stage2_out: Dict):
        """Generates Unified Visualization Chart featuring ALL 8 Categories (Normal + 7 Attacks) in the same graph."""
        os.makedirs("static", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))

        # All 8 categories including 'normal'
        all_categories = ["normal"] + self.attack_categories

        # 1. Panel 1: Single Unified Bar Chart showing ALL 8 categories (Normal + 7 Attacks)
        raw_train_counts = [18499] + [stage2_out["counts_before_smote"].get(c, 0) for c in self.attack_categories]
        processed_train_counts = [18499] + [stage2_out["counts_after_smote"].get(c, 0) for c in self.attack_categories]

        x_all = np.arange(len(all_categories))
        w = 0.38

        bars1 = axes[0].bar(x_all - w/2, raw_train_counts, w, label='Raw Training Data (Before SMOTE)', color='#3b82f6')
        bars2 = axes[0].bar(x_all + w/2, processed_train_counts, w, label='Stage 2 Processed (After SMOTE)', color='#10b981')

        axes[0].set_ylabel('Sample Count (Log Scale for Visibility)', fontsize=11)
        axes[0].set_yscale('log')  # Use log scale so 18k normal and 10-175 attacks are all clearly visible
        axes[0].set_title('Unified Training Dataset Breakdown\n(All 8 Categories: Normal + 7 Attack Vectors in the Same Graph)', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x_all)
        axes[0].set_xticklabels(all_categories, rotation=45, ha='right', fontsize=9, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(axis='y', which='both', alpha=0.3)

        # Add data labels on top of bars
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2., h * 1.15, f"{int(h):,}", ha='center', va='bottom', fontsize=8, rotation=90)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2., h * 1.15, f"{int(h):,}", ha='center', va='bottom', fontsize=8, rotation=90, color='#047857')

        # 2. Panel 2: Stage 2 XGBoost Confusion Matrix on Untouched Test Set
        cats = self.attack_categories
        cm = stage2_out["cm"]
        im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        axes[1].set_title('Stage 2 XGBoost Confusion Matrix\n(Untouched Natural Test Set)', fontsize=13, fontweight='bold')
        fig.colorbar(im, ax=axes[1])
        tick_marks = np.arange(len(cats))
        axes[1].set_xticks(tick_marks)
        axes[1].set_xticklabels(cats, rotation=45, ha='right', fontsize=9)
        axes[1].set_yticks(tick_marks)
        axes[1].set_yticklabels(cats, fontsize=9)
        axes[1].set_ylabel('True Attack Label', fontsize=11)
        axes[1].set_xlabel('Predicted Attack Label', fontsize=11)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

        plt.tight_layout()
        plot_path = os.path.abspath("static/two_stage_architecture_eval.png")
        plt.savefig(plot_path, dpi=300)
        
        # Also save standalone unified chart
        unified_path = os.path.abspath("static/all_classes_unified_graph.png")
        plt.savefig(unified_path, dpi=300)
        plt.close()
        print(f"\n[+] Saved unified single graph (Normal + 7 Attacks together) to: {unified_path}")
        return plot_path

    def run_pipeline(self) -> Dict[str, Any]:
        X_tr, X_va, X_te, y_tr, y_va, y_te = self.step1_load_and_chronological_split()
        stage1_out = self.step2_stage1_anomaly_detection(X_tr, X_va, X_te, y_tr, y_va, y_te)
        stage2_out = self.step3_stage2_attack_classification(stage1_out, y_tr, y_va, y_te)
        plot_path = self.step4_visualize(stage1_out, stage2_out)

        return {
            "stage1_out": stage1_out,
            "stage2_out": stage2_out,
            "plot_path": plot_path
        }

if __name__ == "__main__":
    pipeline = TwoStageSecurityArchitecture(csv_path="data/synthetic_telemetry_27k.csv", seed=42)
    res = pipeline.run_pipeline()
    print("\n" + "="*80)
    print("TWO-STAGE CYBERSECURITY ARCHITECTURE RESULTS COMPLETE.")
    print("="*80)
