# Week 9 Project Scope: End-to-End Fraud Detection System Integration

## 1. Executive Summary & Objectives

The primary objective of **Week 9 & 10** is to design and implement an interactive, production-grade frontend interface that seamlessly integrates with the **FastAPI REST service** developed in Week 8. This capstone integration completes the end-to-end Machine Learning lifecycle for credit card fraud detection without retraining or altering the trained model and feature scaler.

### Key Goals
1. **Interactive Real-Time Fraud Assessment**: Enable risk officers, fraud analysts, and automated systems to score individual credit card transactions with immediate visual feedback.
2. **Bulk Transaction Ingestion & Scoring**: Support CSV batch uploads for high-throughput batch evaluation with summary statistics, risk breakdowns, and downloadable enriched datasets.
3. **Seamless API Integration & Decoupling**: Build a robust, decoupled Streamlit frontend consuming the existing FastAPI backend (`/predict`, `/predict/batch`, `/health`, `/sample`).
4. **Resilience & Defensive Engineering**: Implement comprehensive input validation, clear error boundaries, connection fault tolerance, and schema verification.
5. **Quality Assurance**: Develop unit and integration tests verifying API client interactions, validation logic, and batch data processing pipelines.

---

## 2. System Architecture & Component Interaction

```
+-----------------------------------------------------------------------------------+
|                            USER INTERFACE (Streamlit)                             |
|                                                                                   |
|  +----------------------------+  +--------------------------+  +---------------+  |
|  | Single Transaction Predict |  |   Batch CSV Evaluation   |  | API Health &  |  |
|  | - 30 Numerical Features    |  | - CSV File Ingestion     |  |   Diagnostics |  |
|  | - Sample Presets (Fraud/   |  | - Strict Schema Valid.   |  | - Ping status |  |
|  |   Legit)                   |  | - Aggregated Analytics   |  | - Model info  |  |
|  | - Dynamic Threshold Slider|  | - Enriched CSV Export    |  | - Latency check| |
|  +--------------+-------------+  +------------+-------------+  +-------+-------+  |
|                 |                             |                        |          |
+-----------------|-----------------------------|------------------------|----------+
                  |                             |                        |
                  v                             v                        v
+-----------------------------------------------------------------------------------+
|                     FRONTEND CLIENT & UTILITIES (frontend/)                       |
|  - APIClient: HTTP calls with timeout & graceful error handling                   |
|  - Validators: Input ranges, column requirements (Time, V1..V28, Amount)          |
|  - UI Helpers: Formatting, risk badge rendering, chart generators                 |
+------------------------------------------+----------------------------------------+
                                           | HTTP Requests (REST / JSON)
                                           v
+-----------------------------------------------------------------------------------+
|                            BACKEND API SERVICE (FastAPI)                          |
|                                                                                   |
|  - GET  /health         : Readiness & model artifact verification                 |
|  - GET  /sample         : Reference transactions for demonstration                |
|  - POST /predict        : Single transaction inference & risk scoring             |
|  - POST /predict/batch  : Multi-transaction array inference                       |
+------------------------------------------+----------------------------------------+
                                           | In-Memory DataFrames
                                           v
+-----------------------------------------------------------------------------------+
|                       ML INFERENCE PIPELINE (EDP-Internship-Week-8)               |
|                                                                                   |
|  - Preprocessing: Scaling ONLY 'Time' & 'Amount' with RobustScaler / StandardScaler|
|  - Column Alignment: Strict 30-feature sequence: Time, V1..V28, Amount            |
|  - Model Inference: Tuned XGBoost Classifier (models/tuned_xgboost.pkl)           |
|  - Decision Logic: Dynamic threshold classification & Risk Tier categorization   |
+-----------------------------------------------------------------------------------+
```

---

## 3. Detailed Functional Requirements

### 3.1 Single Transaction Scoring
- **Input Fields**:
  - `Amount`: Monetary transaction value (≥ $0.00).
  - `Time`: Elapsed seconds from benchmark transaction.
  - `V1` to `V28`: PCA-transformed numerical components.
- **Preset Buttons**:
  - **Load Legitimate Sample**: Pre-populates the input form with a verified benign transaction from `/sample?fraud=false`.
  - **Load Fraudulent Sample**: Pre-populates the input form with a verified fraudulent transaction from `/sample?fraud=true`.
  - **Clear/Reset**: Returns all inputs to zero or default baseline.
- **Decision Controls**:
  - Threshold Slider (0.00 to 1.00, default `0.50`). Allows fraud analysts to tighten sensitivity (lower threshold to catch subtle fraud) or optimize precision (higher threshold).
- **Output Presentation**:
  - Classification banner (e.g., `LEGITIMATE` in green vs `FRAUDULENT` in red).
  - Estimated Fraud Probability with animated gauge / progress bar.
  - Risk Level indicator:
    - **LOW**: Probability < 40%
    - **MEDIUM**: Probability between 40% and 75%
    - **HIGH**: Probability ≥ 75%
  - Applied Decision Threshold confirmation.

### 3.2 Batch CSV Prediction
- **File Ingestion**: Accepts standard `.csv` files via drag-and-drop or file selector.
- **Sample Template**: Generates an instant downloadable sample CSV template containing expected headers and example rows.
- **Schema Validation**:
  - Must contain all 30 canonical features: `Time`, `V1` through `V28`, `Amount`.
  - Tolerates presence of optional target column `Class` by ignoring/stripping it safely prior to inference.
  - Checks for non-numeric data types and empty datasets, presenting helpful remediation messages.
- **Batch Processing**:
  - Formats records as a `BatchTransactionPayload` and invokes `POST /predict/batch`.
  - Handles batch chunking if payload exceeds maximum recommended limits.
- **Visual Analytics & Metrics**:
  - Key Performance Indicators: Total Processed, Total Flagged Fraud, Total Legitimate, Fraud Rate Percentage, Mean Fraud Probability.
  - Distribution breakdown (pie/bar chart of Risk Levels).
  - Probability distribution histogram.
  - High-Risk Transactions table with sorting and filtering options.
- **Exporting**:
  - Downloadable enriched CSV containing original rows with attached columns: `prediction`, `label`, `fraud_probability`, `risk_level`, and `threshold_used`.

### 3.3 API Health & Connectivity Management
- **Live Status Indicator**: Sidebar badge showing whether the FastAPI server is **Online (Healthy)** or **Offline / Unreachable**.
- **Configurable Backend URL**: Allows pointing to `http://127.0.0.1:8000`, `http://localhost:8000`, or remote container endpoints without modifying code.
- **Diagnostics Dashboard**: Inspects model type (`XGBClassifier`), feature count (30), and API response latency.

---

## 4. API Integration Contract

The frontend connects to the FastAPI backend using standard HTTP/REST semantics:

| Endpoint | Method | Query Params | Request Payload | Response Model | Description |
|---|---|---|---|---|---|
| `/health` | GET | None | None | `{"status": "healthy", "model_loaded": bool, "scaler_loaded": bool, "model_type": str, "expected_features_count": int}` | Health check & model availability |
| `/sample` | GET | `fraud: bool` | None | `{"type": str, "payload": Dict[str, float]}` | Sample transaction data for quick tests |
| `/predict` | POST | `threshold: float` | `TransactionPayload` (30 features) | `PredictionResponse` | Single transaction inference |
| `/predict/batch` | POST | `threshold: float` | `BatchTransactionPayload` (list of 30-feature objects) | `BatchPredictionResponse` | Bulk transaction inference |

### Canonical Feature Sequence (30 Features)
```
Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount
```

---

## 5. Error Handling & Defensive Design

1. **Backend Unavailability**:
   - If FastAPI is down or the port is blocked, the frontend catches `requests.exceptions.ConnectionError` and `requests.exceptions.Timeout`.
   - Displays a clean error banner: *"Cannot connect to FastAPI backend at `http://127.0.0.1:8000`"*, accompanied by actionable terminal start instructions (`uvicorn app:app --port 8000`).
2. **Missing Feature Headers**:
   - If an uploaded CSV lacks one or more required features, the validator reports the exact missing columns before sending data across the network.
3. **Invalid Data Formats**:
   - Non-numeric values in numeric columns are detected and flagged with row numbers.
4. **Empty Payloads**:
   - Empty files or empty transaction lists trigger friendly guidance without causing unhandled exceptions.

---

## 6. Testing Strategy

1. **Unit Tests**:
   - Validation suite: Tests feature column validation, type checks, missing column reporting, and sample template generation.
   - API Client suite: Tests mocked API communication for success cases, HTTP 400 errors, HTTP 503 errors, and connection timeouts.
   - Batch processor suite: Tests metric calculations, dataframe enrichment, and edge cases (0 fraud, 100% fraud, empty dataset).
2. **Integration Verification**:
   - Live endpoint calls to the active FastAPI server verifying end-to-end data pipeline integrity.
3. **Regression Safeguards**:
   - Verify that all existing Week 8 tests remain operational and that Week 8 files are strictly unmodified.

---

## 7. Deliverables & Acceptance Checklist

- [x] Outer `.gitignore` excluding `EDP-Internship-Week-8/` to protect prior deliverables.
- [x] Detailed Scope Document (`docs/week9_scope.md`).
- [x] Streamlit Application (`streamlit_app.py`) with Single and Batch prediction modules.
- [x] Modular Helper Libraries (`src/api_client.py`, `src/validators.py`).
- [x] Test Suite (`tests/`) covering API client, validation, and batch workflows.
- [x] Comprehensive Report (`reports/week10_report.md`).
- [x] Root `README.md` with operational instructions.
