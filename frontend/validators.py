"""
validators.py
-------------
Validation routines for single transaction inputs and batch CSV DataFrames.
Enforces strict schema consistency with the Week 8 FastAPI inference service.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Canonical feature ordering expected by the model & API
FEATURE_ORDER: List[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]


def validate_single_transaction(data: Dict[str, Any]) -> Tuple[bool, Dict[str, float], List[str]]:
    """Validates and sanitizes a single transaction payload.
    
    Args:
        data: Dictionary of feature names to values.
        
    Returns:
        Tuple of (is_valid, cleaned_data, errors)
    """
    errors: List[str] = []
    cleaned: Dict[str, float] = {}

    if not isinstance(data, dict):
        return False, {}, ["Input payload must be a key-value dictionary."]

    # Check for missing features
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        errors.append(f"Missing required features: {', '.join(missing)}")

    # Check types and numeric values
    for feature in FEATURE_ORDER:
        if feature in data:
            raw_val = data[feature]
            try:
                val = float(raw_val)
                if np.isnan(val) or np.isinf(val):
                    errors.append(f"Feature '{feature}' cannot be NaN or Infinite.")
                else:
                    cleaned[feature] = val
            except (ValueError, TypeError):
                errors.append(f"Feature '{feature}' must be a valid number, got {raw_val!r}.")

    # Business rule validation: Amount cannot be negative
    if "Amount" in cleaned and cleaned["Amount"] < 0:
        errors.append("Transaction 'Amount' cannot be negative.")

    is_valid = len(errors) == 0 and len(cleaned) == len(FEATURE_ORDER)
    return is_valid, cleaned, errors


def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, List[str]]:
    """Validates an uploaded batch DataFrame.
    
    Checks:
    1. Non-empty DataFrame.
    2. Presence of all 30 required columns (Time, V1..V28, Amount).
    3. Handles optional 'Class' column by safely dropping it.
    4. Numeric conversion for all features.
    5. Absence of NaN or Infinite values.
    6. Non-negative Amount values.
    
    Args:
        df: Input pandas DataFrame.
        
    Returns:
        Tuple of (is_valid, cleaned_df, errors)
    """
    errors: List[str] = []

    if not isinstance(df, pd.DataFrame):
        return False, pd.DataFrame(), ["Uploaded data is not a valid tabular DataFrame."]

    if df.empty:
        return False, pd.DataFrame(), ["Uploaded file contains no transaction rows."]

    working_df = df.copy()

    # Drop optional 'Class' ground truth column if present
    if "Class" in working_df.columns:
        working_df = working_df.drop(columns=["Class"])

    # Check for missing columns
    missing_cols = [c for c in FEATURE_ORDER if c not in working_df.columns]
    if missing_cols:
        errors.append(
            f"Missing {len(missing_cols)} required column(s): {', '.join(missing_cols[:5])}"
            + (f" and {len(missing_cols) - 5} more" if len(missing_cols) > 5 else "")
        )
        return False, pd.DataFrame(), errors

    # Reorder columns to canonical order
    working_df = working_df[FEATURE_ORDER]

    # Check numeric types and conversion
    for col in FEATURE_ORDER:
        try:
            working_df[col] = pd.to_numeric(working_df[col], errors="coerce")
        except Exception as exc:
            errors.append(f"Could not convert column '{col}' to numeric: {str(exc)}")

    # Check for nulls/NaNs introduced or existing
    null_counts = working_df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        col_summary = ", ".join([f"{col} ({count} nulls)" for col, count in cols_with_nulls.items()])
        errors.append(f"Dataset contains missing or non-numeric values in: {col_summary}")

    # Check for negative Amount
    if (working_df["Amount"] < 0).any():
        neg_count = int((working_df["Amount"] < 0).sum())
        errors.append(f"Found {neg_count} row(s) with negative Amount values.")

    is_valid = len(errors) == 0
    return is_valid, working_df if is_valid else pd.DataFrame(), errors


def get_sample_preset(fraud: bool = False) -> Dict[str, float]:
    """Returns a verified sample transaction preset.
    Matches the reference samples from Week 8 src/predict.py.
    """
    if fraud:
        return {
            "Time": 99772.46,
            "V1": 1.68568, "V2": -2.32051, "V3": 1.15678, "V4": 1.43683,
            "V5": 2.07263, "V6": 3.43088, "V7": 0.40904, "V8": 0.15500,
            "V9": -0.46354, "V10": 3.18685, "V11": 2.63562, "V12": 2.89936,
            "V13": -0.56984, "V14": 0.13619, "V15": 0.99926, "V16": 2.90565,
            "V17": 1.52876, "V18": 0.32014, "V19": 0.40335, "V20": 0.86711,
            "V21": 1.65071, "V22": 0.11091, "V23": 1.02666, "V24": 1.71387,
            "V25": -0.46354, "V26": -1.10055, "V27": 0.28921, "V28": 0.11728,
            "Amount": 738.54,
        }
    else:
        return {
            "Time": 31604.21,
            "V1": -0.42015, "V2": -0.43649, "V3": -1.51860, "V4": 2.98559,
            "V5": -1.63805, "V6": -2.23789, "V7": -2.27306, "V8": -1.00042,
            "V9": -0.78319, "V10": 0.57358, "V11": -0.57348, "V12": 1.55848,
            "V13": 2.59228, "V14": 1.45910, "V15": 0.82946, "V16": -1.08856,
            "V17": 1.44921, "V18": -0.39534, "V19": -1.57161, "V20": 1.14596,
            "V21": 0.23719, "V22": 0.67931, "V23": -0.45521, "V24": 0.28307,
            "V25": 1.18469, "V26": -0.61417, "V27": -1.60533, "V28": 0.66319,
            "Amount": 149.97,
        }


def create_sample_csv_dataframe() -> pd.DataFrame:
    """Creates a sample 5-row batch DataFrame ready for download or testing."""
    legit = get_sample_preset(fraud=False)
    fraud = get_sample_preset(fraud=True)

    row3 = legit.copy()
    row3["Time"] = 45200.0
    row3["Amount"] = 24.50
    row3["V1"] = 0.1234

    row4 = fraud.copy()
    row4["Time"] = 104500.0
    row4["Amount"] = 1250.00
    row4["V1"] = 2.154

    row5 = legit.copy()
    row5["Time"] = 61200.0
    row5["Amount"] = 89.20

    data = [legit, fraud, row3, row4, row5]
    df = pd.DataFrame(data)[FEATURE_ORDER]
    return df
