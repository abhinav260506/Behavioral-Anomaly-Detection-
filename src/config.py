"""
Centralized Configuration for Anti-Gravity Cybersecurity Anomaly Detection System.
Provides unified production constants for Stage-1 and Stage-2 pipelines.
"""

# Production Stage-1 Composite Anomaly Risk Threshold (0.0 to 100.0)
# Calibrated on Normal Validation Split to ensure FPR <= 5% with maximum attack recall
STAGE1_THRESHOLD: float = 60.0

# EWMA Alpha for Feature History Smoothing
EWMA_ALPHA: float = 0.1

# Scaled Distance & LSTM Calibration P95 Loss Thresholds
BEHAVIORAL_DIST_P95: float = 7.8839
LSTM_LOSS_P95: float = 15.1857
