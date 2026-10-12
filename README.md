# Credit Card Fraud Detection - Full-Stack AI/ML System (Weeks 9–10)

Production-ready end-to-end Machine Learning web application and inference pipeline for detecting credit card fraud in real-time. Built as the capstone integration combining the **FastAPI REST Service** (Week 8) with a high-performance **Streamlit Frontend Dashboard** (Week 10).

---

## 🏛️ System Architecture

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
|                                  CLIENT & VALIDATION                                    |
|                                                                                         |
|                       frontend/api_client.py & frontend/validators.py                   |
|   - Connection retry & timeout protection     - Pre-flight 30-feature schema enforcement|
|   - Automatic batch chunking (500 items/req)  - Sanitization of non-numeric / NaN values|
+--------------------------------------------+--------------------------------------------+
                                             | HTTP Network Calls
                                             v
+-----------------------------------------------------------------------------------------+
|                                     BACKEND REST API                                    |
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

## 📂 Project Structure

```text
EDP-AI-ML-Week9-10/
├── .gitignore                    # Ignores EDP-Internship-Week-8/ & temp artifacts
├── requirements.txt              # Frontend, testing & backend dependencies
├── streamlit_app.py              # Main Streamlit dashboard application
│
├── frontend/                     # Frontend client & validation modules
│   ├── __init__.py
│   ├── api_client.py             # Resilient HTTP client with batch chunking
│   └── validators.py             # Input sanitization & 30-feature schema checks
│
├── docs/                         # Architecture & scope specifications
│   └── week9_scope.md            # Week 9 Integration & System Scope document
│
├── reports/                      # Engineering reports
│   └── week10_report.md          # Week 10 Capstone verification & test report
│
├── tests/                        # Comprehensive test suite (24 tests)
│   ├── __init__.py
│   ├── test_validators.py        # 10 unit tests for schema validation
│   ├── test_api_client.py        # 10 unit tests mocking API & failure modes
│   └── test_live_integration.py  # 4 integration tests with live FastAPI backend
│
└── EDP-Internship-Week-8/         # Week 8 Independent Project (Untouched)
    ├── app.py                    # FastAPI inference service
    ├── models/                   # tuned_xgboost.pkl & scaler.pkl
    └── src/                      # Week 8 inference pipeline
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- PowerShell (Windows) or Bash (macOS/Linux)

### 1. Install Dependencies

Install all dependencies from the root directory:

```powershell
pip install -r requirements.txt
```

---

### 2. Start the FastAPI Backend Service

In your first terminal window, start the Week 8 FastAPI server:

```powershell
# Navigate into the Week 8 directory
cd EDP-Internship-Week-8

# Start the uvicorn server on port 8000
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health at: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) or explore the interactive Swagger documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

### 3. Launch the Streamlit Frontend Dashboard

In a second terminal window (from the root `EDP-AI-ML-Week9-10` folder):

```powershell
streamlit run streamlit_app.py
```

The web dashboard will automatically open in your default browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 💡 Using the Web Application

### 🎯 Single Transaction Scoring
1. Navigate to the **Single Transaction Scoring** tab.
2. Click **Load Legitimate Sample** or **Load Fraud Sample** to quickly pre-populate verified historical cases.
3. Or modify features directly:
   - Enter transaction **Amount ($)** and **Time (seconds)**.
   - Expand the **PCA Features (V1–V28)** grid to enter custom values.
4. Click **⚡ Score Transaction**.
5. View the resulting **Fraud Probability (%)**, **Risk Tier** (`LOW`, `MEDIUM`, `HIGH`), and classification banner.

### 📁 Batch CSV Processing
1. Navigate to the **Batch CSV Processing** tab.
2. (Optional) Click **📥 Download Sample Batch CSV** to get a ready-to-test CSV with 5 diverse transaction records.
3. Upload any CSV file containing the 30 canonical features (`Time`, `V1`–`V28`, `Amount`).
4. Review the automatic pre-flight schema validation check.
5. Click **🚀 Evaluate Transactions**.
6. View batch KPI cards (Total Processed, Fraud Count, Legitimate Count, Fraud Rate), Risk Level distribution chart, and probability histogram.
7. Click **💾 Download Enriched Predictions CSV** to export results with appended predictions and risk metrics.

### 🎚️ Dynamic Decision Threshold
Use the **Fraud Classification Threshold** slider in the sidebar (default `0.50`):
- **Lower Threshold (e.g., 0.20–0.40):** Increases sensitivity (higher recall) for high-security environments.
- **Higher Threshold (e.g., 0.60–0.80):** Reduces false positives (higher precision) to prevent customer disruption.

---

## 🧪 Running the Test Suite

Run all unit and integration tests from the root directory:

```powershell
python -m pytest tests/ -v
```

### Actual Test Results (24 Passed)
```text
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

======================= 24 passed in 4.82s =======================
```

---

## 🔒 Preservation of Week 8 Project
- `EDP-Internship-Week-8/` is isolated via the outer `.gitignore`.
- No Week 8 files or model artifacts have been modified, retrained, or deleted.
- The Week 9–10 deliverables have been committed and pushed to the `EDP-Week-9-10` GitHub repository. The Week 8 project remains excluded from this repository and   has been preserved separately.
