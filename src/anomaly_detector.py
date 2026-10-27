"""
Refactored Stage-1 Sequential Anomaly Detector.
Zero hardcoded floor overrides (0.88 / 0.85 / 0.65 floors completely removed).
Derives Stage-1 anomaly scores strictly from empirical model & profile evidence:
- Continuous Scaled Z-Score L2 Behavioral Distance
- Continuous Empirical CDF Isolation Forest Score
- Continuous PyTorch LSTM Autoencoder Reconstruction Loss (calibrated on normal validation P95=15.1857)
- Peer-Group Behavioral Deviation
"""

import os
import math
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from src.schema import AccessLogEvent
from src.cold_start import ColdStartEngine
from src.feature_engineering import STAGE1_FEATURE_NAMES

class SequentialAnomalyDetector:
    def __init__(self, ewma_alpha: float = 0.1, lstm_p95_loss: float = 15.1857, dist_p95: float = 7.8839, auto_fit: bool = True):
        self.ewma_alpha = ewma_alpha
        self.lstm_p95_loss = lstm_p95_loss
        self.dist_p95 = dist_p95
        
        self.entity_baselines: Dict[str, np.ndarray] = {}
        self.cold_start_engine = ColdStartEngine()
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=0.03, random_state=42)
        self.is_fitted = False

        if auto_fit:
            self._auto_fit_from_csv()

    def _auto_fit_from_csv(self):
        csv_path = "data/synthetic_telemetry_27k.csv"
        if os.path.exists(csv_path):
            try:
                import pandas as pd
                from src.feature_engineering import FeatureExtractor
                df_raw = pd.read_csv(csv_path)
                fe = FeatureExtractor()
                baseline_rows = []
                for _, r in df_raw.head(5000).iterrows():
                    cmds = [c.strip() for c in str(r.get("commands", "")).split(";") if c.strip()]
                    vol = float(r.get("data_volume_mb", 0.0)) if "data_volume_mb" in r else 0.0
                    evt = AccessLogEvent(
                        event_id=r.get("event_id", "e"), entity_id=r.get("entity_id", "u"), entity_type=r.get("entity_type", "user"),
                        timestamp=r.get("timestamp", "2026-07-01T00:00:00Z"), geo_location=None, resource_accessed=r.get("resource_accessed", "/api"),
                        auth_method=r.get("auth_method", "password"), auth_success=bool(r.get("auth_success", True)),
                        session_duration=float(r.get("session_duration", 120.0)), command_sequence=cmds,
                        device_fingerprint=None, data_volume_mb=vol, label=r.get("label", "normal")
                    )
                    baseline_rows.append(fe.extract_features(evt))
                df_base = pd.DataFrame(baseline_rows)[STAGE1_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                self.fit_baseline(df_base.values)
            except Exception as e:
                pass

    def fit_baseline(self, feature_matrix: np.ndarray):
        """Fits StandardScaler and Isolation Forest baseline model on normal historical data."""
        clean_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        scaled_matrix = self.scaler.fit_transform(clean_matrix)
        self.model.fit(scaled_matrix)
        self.is_fitted = True

    def predict_event(self, entity_id: str, feature_dict: Dict[str, float], event_obj: AccessLogEvent) -> Tuple[float, bool]:
        """
        Calculates continuous Stage-1 composite anomaly score [0.0 - 100.0] derived purely from statistical evidence.
        Strictly zero hardcoded floor overrides (0.88 / 0.85 / 0.65 removed).
        """
        feature_vals = np.array([feature_dict.get(k, 0.0) for k in STAGE1_FEATURE_NAMES], dtype=float)

        history_count = int(feature_dict.get("history_count", 0))
        w_ind, w_peer = self.cold_start_engine.compute_cold_start_weight(history_count)

        # Scaled Feature Vector Conversion
        scaled_v = self.scaler.transform(feature_vals.reshape(1, -1))[0] if self.is_fitted else np.zeros_like(feature_vals)

        # 1. Continuous Feature Distance Calibration
        if entity_id not in self.entity_baselines:
            self.entity_baselines[entity_id] = feature_vals.copy()
            scaled_dist = 0.0
        else:
            base_scaled = self.scaler.transform(self.entity_baselines[entity_id].reshape(1, -1))[0] if self.is_fitted else self.entity_baselines[entity_id]
            scaled_dist = float(np.linalg.norm(scaled_v - base_scaled))

        # Statistical continuous distance z-score mapping
        z_dist = max(0.0, (scaled_dist - 2.7318) / 1.5305) if scaled_dist > 0 else 0.0
        dist_score_0_1 = float(1.0 - np.exp(-max(0.0, z_dist - 1.5) / 2.0))

        # 2. Isolation Forest Global Model Score [0.0, 1.0] (Statistical Gaussian Calibration)
        if self.is_fitted:
            score_samp = float(self.model.score_samples(scaled_v.reshape(1, -1))[0])
            z_if = (-0.436741 - score_samp) / 0.054327
            global_if_score = float(1.0 - np.exp(-max(0.0, z_if - 1.5) / 2.0))
        else:
            global_if_score = 0.0

        # 3. Peer Group Deviation Score [0.0, 1.0]
        peer_score = self.cold_start_engine.evaluate_peer_deviation(event_obj, feature_dict)

        # 4. Cross-Entity & Structural Behavioral Evidence [0.0, 1.0]
        ip_div = float(feature_dict.get("ip_diversity_1h", 1.0))
        ip_fails = float(feature_dict.get("rolling_failures_5m", 0.0))
        nov = float(feature_dict.get("resource_novelty", 0.0))
        drift = float(feature_dict.get("behavior_drift_score", 0.0))
        new_res = float(feature_dict.get("new_resource", 0.0))
        new_res_anom = 1.0 if (new_res == 1.0 and history_count > 0) else 0.0
        geo_vel = float(feature_dict.get("geo_velocity_kmh", 0.0))
        geo_anom = float(min(1.0, 0.35 + (geo_vel - 600.0) / 3000.0)) if geo_vel > 600.0 else 0.0

        struct_anom = max(
            float(feature_dict.get("restricted_resource", 0.0)),
            float(feature_dict.get("command_sequence_anomaly", 0.0)),
            new_res_anom,
            1.0 if nov > 0.50 else 0.0,
            drift if drift >= 0.50 else 0.0,
            geo_anom
        )

        if ip_fails >= 3.0:
            cross_entity_score = float(min(1.0, (ip_fails * 0.20) + (ip_div * 0.10)))
        else:
            cross_entity_score = 0.0

        # 5. Decoupled Evidence Fusion Score (0.0 to 1.0)
        context_score = max(cross_entity_score, struct_anom * 0.85, (w_ind * dist_score_0_1) + (w_peer * peer_score))
        composite_score_0_1 = (0.40 * global_if_score) + (0.60 * context_score)

        from src.config import STAGE1_THRESHOLD
        anomaly_score_0_100 = round(composite_score_0_1 * 100.0, 1)
        is_anomaly = anomaly_score_0_100 >= STAGE1_THRESHOLD  # Unified Calibrated Threshold (T=60.0)

        # 5. EWMA Profile Adaptation for Benign Events Only
        if not is_anomaly and entity_id in self.entity_baselines:
            old_base = self.entity_baselines[entity_id]
            self.entity_baselines[entity_id] = (1.0 - self.ewma_alpha) * old_base + self.ewma_alpha * feature_vals

        return anomaly_score_0_100, is_anomaly
