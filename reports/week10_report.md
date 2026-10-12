# Week 10 Engineering Report: Full-Stack Fraud Detection Deployment & Integration

## 1. Executive Summary

This report documents the completion of **Week 10**, finalizing the capstone integration for the **Credit Card Fraud Detection System**. Building on the tuned XGBoost classifier and production FastAPI REST service from Week 8, Week 10 delivers a modern, decoupled **Streamlit web application** (`streamlit_app.py`).

The solution provides fraud compliance teams and risk investigators with real-time single-transaction evaluation, multi-row batch CSV ingestion, customizable decision threshold tuning, and automated risk tier categorization.

### Key Deliverables Completed
- **Streamlit Application (`streamlit_app.py`)**: Real-time scoring, CSV batch scoring, analytics dashboard, dynamic threshold slider, and live health diagnostics.
- **Frontend Utilities (`frontend/api_client.py`, `frontend/validators.py`)**: Robust HTTP client, chunked batch communication, and pre-flight schema validators.
- **Scope Documentation (`docs/week9_scope.md`)**: Complete architecture and integration blueprint.
- **Comprehensive Test Suite (`tests/`)**: 24 tests spanning schema validation, API client fault tolerance, and live in-process end-to-end integration.
- **Root Documentation (`README.md`)**: Step-by-step setup and operational guide.
- **Repository Isolation**: Outer `.gitignore` explicitly preserving `EDP-Internship-Week-8/` untouched without retraining or modifying model artifacts.

---

## 2. System Architecture & Component Interaction

```
+-----------------------------------------------------------------------------------------+
|                                    PRESENTATION TIER                                    |
|                                                                                         |
|                            Streamlit UI (streamlit_app.py)                              |
|   - Real-time Transaction Scoring Form        - Dynamic Threshold Tuning (0.05 - 0.95)   |
|   - CSV Batch File Ingestion & Validation     - Risk Tiers: LOW, MEDIUM, HIGH           |
|   - Live Health & Microservice Ping           - Enriched CSV Export                     |
+--------------------------------------------+--------------------------------------------+
                                             | HTTP Requests (REST / JSON)
                                             v
+-----------------------------------------------------------------------------------------+
|                                      API CLIENT                                         |
|                                                                                         |
|                       frontend/api_client.py & frontend/validators.py                   |
|   - Connection retry & timeout protection     - Pre-flight 30-feature schema enforcement|
|   - Automatic batch chunking (500 items/req)  - Sanitization of non-numeric / NaN values|
+--------------------------------------------+--------------------------------------------+
                                             | HTTP Network Calls
                                             v
+-----------------------------------------------------------------------------------------+
|                                     MIDDLEWARE TIER                                     |
|                                                                                         |
|                        FastAPI Microservice (EDP-Internship-Week-8)                     |
|   - GET  /health           : Readiness & artifact verification                          |
|   - GET  /sample           : Reference legitimate and fraudulent payloads               |
|   - POST /predict          : Single transaction scoring with threshold param            |
|   - POST /predict/batch    : Multi-transaction array processing                         |
+--------------------------------------------+--------------------------------------------+
                                             | In-Memory DataFrames
                                             v
+-----------------------------------------------------------------------------------------+
|                                     INFERENCE TIER                                      |
|                                                                                         |
|   - Preprocessing: Scaling ONLY 'Time' and 'Amount' (models/scaler.pkl)                 |
|   - Feature Invariance: V1 through V28 preserved unscaled in canonical order            |
|   - Core Classifier: Tuned XGBoost Classifier (models/tuned_xgboost.pkl)                |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Features Implemented in Frontend

### 3.1 Live Service Health Monitoring
- Continuous polling of `GET /health` with millisecond response latency reporting.
- Visual badge indicator in the sidebar:
  - 🟢 **Online** (with latency and model details: `XGBClassifier`, 30 expected features).
  - 🔴 **Backend Offline** with troubleshooting hints and exact terminal commands to start the backend.

### 3.2 Single Transaction Prediction
- **Interactive Form**: Inputs for `Amount ($)`, `Time (seconds)`, and a responsive 4-column expandable grid for PCA features `V1` to `V28`.
- **Pre-set Demonstrators**:
  - `Load Legitimate Sample`: Pre-populates verified legitimate transaction ($149.97).
  - `Load Fraud Sample`: Pre-populates verified fraudulent transaction ($738.54).
  - `Reset Inputs to Zero`: Clears all fields.
- **Dynamic Decision Threshold Slider**: Ranges from `0.05` to `0.95` (default `0.50`), giving analysts immediate control over sensitivity vs false-positive tolerance.
- **Visual Result Card**:
  - Color-coded classification banner: Red (`🚨 FRAUDULENT`) or Green (`✅ LEGITIMATE`).
  - Risk Level Badge: `LOW` (< 40%), `MEDIUM` (40% - 75%), or `HIGH` (≥ 75%).
  - Probability meter with progress bar.
  - Raw JSON inspection expander.

### 3.3 CSV Batch Processing & Analytics
- **Drag-and-Drop Ingestion**: Accepts standard `.csv` files.
- **Built-in Sample Template**: "Download Sample Batch CSV" button produces a verified 5-row CSV for testing immediately.
- **Pre-Flight Schema Validation**:
  - Validates presence of all 30 features before transmitting across the network.
  - Strips optional ground-truth `Class` column safely if present in benchmark datasets.
  - Detects empty rows, non-numeric strings, negative amounts, and NaNs.
- **Chunked Inference**: `APIClient.predict_batch` partitions large transaction sets into 500-item chunks to prevent network timeouts.
- **Summary Metrics**: Metric cards showing Total Transactions, Flagged Fraud, Approved Legitimate, and Fraud Rate (%).
- **Interactive Charts**:
  - Risk Tier Breakdown (Bar Chart).
  - Probability Distribution Bins (0–20%, 20–40%, 40–60%, 60–80%, 80–100%).
- **Filterable Results Table**: Multi-select filter for risk tiers with formatted amounts and probabilities.
- **Export**: One-click download of the complete enriched CSV with `Prediction`, `Status`, `Fraud_Probability`, `Risk_Level`, and `Threshold_Used`.

---

## 4. Verification & Testing Results

A dedicated test suite was executed covering unit, mock, and live in-process integration scenarios.

### Test Execution Summary
Command: `python -m pytest tests/ -v`

```text
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\IntershipProject\EDP-AI-ML-Week9-10
collected 24 items

tests/test_api_client.py::TestAPIClient::test_01_health_check_healthy PASSED
tests/test_api_client.py::TestAPIClient::test_02_health_check_unhealthy PASSED
tests/test_api_client.py::TestAPIClient::test_03_health_check_connection_error PASSED
tests/test_api_client.py::TestAPIClient::test_04_get_sample_success PASSED
tests/test_api_client.py::TestAPIClient::test_05_predict_single_success PASSED
tests/test_api_client.py::TestAPIClient::test_06_predict_single_validation_error PASSED
tests/test_api_client.py::TestAPIClient::test_07_predict_single_timeout PASSED
tests/test_api_client.py::TestAPIClient::test_08_predict_batch_success PASSED
tests/test_api_client.py::TestAPIClient::test_09_predict_batch_chunking PASSED
tests/test_api_client.py::TestAPIClient::test_10_predict_batch_empty_list PASSED
tests/test_live_integration.py::TestLiveIntegration::test_01_live_health_endpoint PASSED
tests/test_live_integration.py::TestLiveIntegration::test_02_live_single_legitimate_prediction PASSED
tests/test_live_integration.py::TestLiveIntegration::test_03_live_single_fraud_prediction PASSED
tests/test_live_integration.py::TestLiveIntegration::test_04_live_batch_prediction_flow PASSED
tests/test_validators.py::TestValidators::test_01_valid_single_transaction PASSED
tests/test_validators.py::TestValidators::test_02_single_transaction_missing_feature PASSED
tests/test_validators.py::TestValidators::test_03_single_transaction_negative_amount PASSED
tests/test_validators.py::TestValidators::test_04_single_transaction_nan_infinite PASSED
tests/test_validators.py::TestValidators::test_05_valid_batch_dataframe PASSED
tests/test_validators.py::TestValidators::test_06_batch_dataframe_with_class_column PASSED
tests/test_validators.py::TestValidators::test_07_batch_dataframe_missing_column PASSED
tests/test_validators.py::TestValidators::test_08_batch_dataframe_empty PASSED
tests/test_validators.py::TestValidators::test_09_batch_dataframe_non_numeric PASSED
tests/test_validators.py::TestValidators::test_10_sample_template_integrity PASSED

======================= 24 passed, 33 warnings in 4.82s =======================
```

### Baseline Week 8 Verification
To confirm that Week 8 was not broken, Week 8 tests were also run independently:
Command: `python -m pytest tests/` in `EDP-Internship-Week-8`
- **Result:** `11 passed in 9.04s`
- **Integrity Status:** Unchanged and intact.

---

## 5. Artifact & Repository Preservation

In strict compliance with requirements:
1. `EDP-Internship-Week-8/` has been added to the root `.gitignore`.
2. No files inside `EDP-Internship-Week-8/` were modified, renamed, or deleted.
3. No model retraining was performed; all inferences leverage the pre-trained `models/tuned_xgboost.pkl` and `models/scaler.pkl`.
4. Git commits and pushes have been deferred as requested.
