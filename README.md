# Aegis AI: AI-Powered Behavioral Anomaly Detection & Threat Classification Engine

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-20B2AA?style=for-the-badge)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.style=for-the-badge)](LICENSE)

An enterprise-grade, two-stage machine learning system designed to detect cybersecurity intrusions and compromised credential activity in near real-time. **Aegis AI** models entity behavioral baselines across users, service accounts, and edge devices, flags anomalies without ground-truth label leakage, and categorizes threats into 7 distinct attack vectors with an Explainable AI (XAI) risk score.

---

## 🌟 Key Performance Highlights

- **End-to-End System Accuracy**: **`95.36%`** on an untouched, entity-grouped held-out test split ($26$ test entities, $4,481$ logs).
- **Stage-1 Anomaly Detection ROC-AUC**: **`0.9734`** with a **`2.45%` False Positive Rate (FPR)** *(well below realistic SOC analyst budget thresholds)*.
- **Stage-2 Threat Classification**: **`94.58%` F1-Score** on *Lateral Movement*, **`90.24%` F1-Score** on *Brute Force*, and **`88.07%` F1-Score** on *Credential Stuffing*.
- **Zero Entity Overlap & Zero Data Leakage**: Strict split separation ($115$ Train entities, $24$ Validation entities, $26$ Test entities; **`0` entity overlap**).
- **Model Persistence**: Clean separation between model fitting ([src/train_final.py](file:///c:/Users/abhin/Desktop/honey/src/train_final.py)) and frozen inference ([src/evaluate_final.py](file:///c:/Users/abhin/Desktop/honey/src/evaluate_final.py)) using SHA256-verified artifacts in `models/`.
- **Analyst Usability**: Interactive web dashboard ([app.py](file:///c:/Users/abhin/Desktop/honey/app.py)) featuring dynamic XAI Threat Diagnosis cards, root cause bullet points, and percentage feature attributions.

---

## 📐 System Architecture

```
                          [ Incoming Raw Access Log Event ]
                                          │
                                          ▼
                      [ Causal Feature Extractor (45 Features) ]
                                          │
                                          ▼
                      [ STAGE 1: Unsupervised Anomaly Detection ]
                      ┌────────────────────────────────────────┐
                      │ • Isolation Forest (Trees: 100)        │
                      │ • PyTorch LSTM Autoencoder (Dim: 45->32)│
                      │ • Mahalanobis & L2 Distance Engine     │
                      │ • Continuous Geo-Velocity Scaling S_geo│
                      │ • Peer Group Cold-Start Profiler       │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                         Risk Score < T (32.0) ?
                        ┌──────────────────┴──────────────────┐
                        │                                     │
                     [ YES ]                                [ NO ]
                        │                                     │
                        ▼                                     ▼
                Flagged as NORMAL                   Anomalous Event Stream
              (Risk: 0-31.9 / 100)                            │
                                                              ▼
                                            [ STAGE 2: Supervised Classification ]
                                            ┌───────────────────────────────────┐
                                            │ XGBoost Multi-Class Classifier    │
                                            │ (150 Trees, Softprob, 7 Classes)  │
                                            └─────────────────┬─────────────────┘
                                                              │
                                                              ▼
                                                   Predicted Attack Vector &
                                                   XAI Root Cause Attribution
```

---

## 🚀 Quickstart & Usage

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed. Install required packages:
```bash
pip install torch xgboost scikit-learn imbalanced-learn joblib numpy pandas matplotlib faker pytest python-docx
```

### 2. Run Pytest Unit Test Suite
Verify pipeline integrity and component schema constraints:
```bash
python -m pytest tests/test_pipeline.py -v
```
*Expected Output*: `8 passed in 3.86s (100% PASS)`

### 3. Fit & Serialize Models (Training Pipeline)
Fit models on Train split, calibrate Stage-1 threshold ($T = 32.0$) on Validation split, and serialize frozen artifacts:
```bash
python -m src.train_final
```
*Artifacts Saved*: `scaler.joblib`, `isolation_forest.joblib`, `lstm_autoencoder.pth`, `xgboost_classifier.json`, `metadata.json`, `train_val_test_split.json` under `models/`.

### 4. Evaluate Frozen Models on Held-Out Test Set
Run pure inference on the untouched held-out test split (**ZERO** training code executed):
```bash
python -m src.evaluate_final
```

### 5. Launch SOC Analyst Operations Center Dashboard
Start the real-time REST API web server on `http://localhost:5000`:
```bash
python app.py
```
Open `http://localhost:5000` in your web browser to access the **Event Evaluator**, **Alert Queue**, and **Entity Profiler**.

---

## 🎯 Injected Attack Taxonomy & Performance

| Category | Description / Injected Pattern | Support | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`normal`** | Habitual baseline logins adhering to entity profile | 4,122 | 97.43% | 97.55% | **97.49%** |
| **`brute_force`** | Rapid repeated failed-auth attempts ($N \ge 3$ in 5m) from single IP | 45 | 100.00% | 82.22% | **90.24%** |
| **`credential_stuffing`** | Multi-account login failure bursts from shared proxy/NAT IP | 61 | 100.00% | 78.69% | **88.07%** |
| **`lateral_movement`** | Traversal into protected endpoints (`/internal/server-config`, `/admin/vault_keys`) | 96 | 89.72% | 100.00% | **94.58%** |
| **`low_and_slow_exfiltration`** | Off-hours small volume transfers accumulating beyond 7-day baselines | 42 | 34.21% | 30.95% | **32.50%** |
| **`device_spoofing`** | Mid-session device fingerprint alterations (OS `Ubuntu` $\to$ `Kali`, MAC shifts) | 21 | 58.33% | 66.67% | **62.22%** |
| **`impossible_travel`** | Logins from distant locations (Chennai to London) in 20m ($24,633\text{ km/h}$) | 24 | 50.00% | 20.83% | **29.41%** |
| **`insider_drift`** | Gradual resource access expansion across multiple organizational departments | 70 | 43.33% | 55.71% | **48.75%** |

---

## 📊 Complete 8x8 Confusion Matrix (True \ Predicted)

```text
True \ Pred               normal  brute_  creden  latera  low_an  device  imposs  inside
-----------------------------------------------------------------------------------------
normal                     4021       0       0      10      25      10       5      51
brute_force                   8      37       0       0       0       0       0       0
credential_stuffing          13       0      48       0       0       0       0       0
lateral_movement              0       0       0      96       0       0       0       0
low_and_slow_exfiltration    28       0       0       1      13       0       0       0
device_spoofing               7       0       0       0       0      14       0       0
impossible_travel            19       0       0       0       0       0       5       0
insider_drift                31       0       0       0       0       0       0      39
```

---

## 🔌 REST API Endpoints Overview

| HTTP Method | Path | Description | Response / Status |
| :---: | :--- | :--- | :---: |
| `GET` | `/` | Serves main SOC Analyst Dashboard UI | `HTTP 200` (HTML) |
| `GET` | `/api/alerts` | Returns active alert queue sorted by risk score descending | `HTTP 200` (JSON Array) |
| `GET` | `/api/metrics` | Returns overall system summary metrics & detection rates | `HTTP 200` (JSON Object) |
| `GET` | `/api/entity/<entity_id>` | Returns entity baseline profile and historical log trail | `HTTP 200` (JSON Object) |
| `POST` | `/api/evaluate_single` | Evaluates arbitrary access log events in Isolated Test Mode | `HTTP 200` (JSON Object) |
| `POST` | `/api/inject` | Simulates attack sequence injection into live stream | `HTTP 200` (JSON Object) |
| `POST` | `/api/feedback` | Processes analyst feedback (false positive tuning / confirmation) | `HTTP 200` (JSON Object) |

---

## 🔒 Serialized Artifact Checksums (SHA256)

- `models/scaler.joblib`: `8b9fa235d88b2e9c95a3c46931d81636e67bc05428930c4e43fad7f4e9f18e8c`
- `models/isolation_forest.joblib`: `cf58d4b477e45f6ba38a80edded24164c58df25e7359ece9165c49a32e1b9ed1`
- `models/lstm_autoencoder.pth`: `2b574c700f706363a9255113919b46c3a72e03c91c35a18f1f60dbd3db51b5df`
- `models/xgboost_classifier.json`: `c652c7214effdbfb6159864054d97166a60d2864c82149c12e91598d6a836d19`
- `models/metadata.json`: `d610c739d78633fdd5ab5cd02d672429cb7b03f4dd458d4c94a98ebf9b359c80`

---

## 📂 Project Repository Structure

```text
c:\Users\abhin\Desktop\honey\
├── app.py                         # SOC Analyst Operations Center Web Server (port 5000)
├── Master_Cybersecurity_Project_Report.docx # Formatted Word Project Report
├── README.md                      # Primary project documentation & quickstart guide
├── data/                          # Telemetry datasets
│   ├── synthetic_telemetry_27k.csv
│   └── synthetic_telemetry_27k.json
├── models/                        # Serialized Frozen Model Artifacts
│   ├── scaler.joblib
│   ├── isolation_forest.joblib
│   ├── lstm_autoencoder.pth
│   ├── xgboost_classifier.json
│   ├── metadata.json
│   └── train_val_test_split.json
├── src/                           # Core Source Code Modules
│   ├── train_final.py             # Fitting script responsible for model training & serialization
│   ├── evaluate_final.py           # Frozen evaluation script (ZERO training code)
│   ├── feature_engineering.py     # 45-dimensional causal feature extraction engine
│   ├── anomaly_detector.py        # Stage-1 anomaly detector & distance matrix engine
│   ├── anomaly_classifier.py      # Stage-2 threat taxonomy classification engine
│   ├── cold_start.py              # Cold-start & peer group deviation profiler
│   ├── explainability.py          # XAI root cause & feature attribution layer
│   ├── isolated_test_engine.py    # Stateless test evaluation engine for Web UI
│   ├── generator.py               # Synthetic access log telemetry generator
│   ├── pipeline.py                # Live telemetry stream orchestrator
│   └── schema.py                  # Pydantic telemetry event dataclasses
└── tests/
    └── test_pipeline.py           # Pytest unit test suite (8/8 PASSED)
```

---

## 📜 License & Citation

Licensed under the **MIT License**. Created for Advanced Agentic Behavioral Anomaly Detection in Cybersecurity.
