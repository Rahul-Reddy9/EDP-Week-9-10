"""
test_validators.py
------------------
Unit tests for single transaction validation and batch CSV DataFrame validation.
"""

import unittest
import numpy as np
import pandas as pd

from frontend.validators import (
    FEATURE_ORDER,
    validate_single_transaction,
    validate_batch_dataframe,
    get_sample_preset,
    create_sample_csv_dataframe,
)


class TestValidators(unittest.TestCase):
    def setUp(self):
        self.valid_single = get_sample_preset(fraud=False)

    def test_01_valid_single_transaction(self):
        """Valid transaction with all 30 features should pass cleanly."""
        is_valid, cleaned, errors = validate_single_transaction(self.valid_single)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(cleaned), len(FEATURE_ORDER))
        self.assertAlmostEqual(cleaned["Amount"], 149.97)

    def test_02_single_transaction_missing_feature(self):
        """Single transaction missing features should be flagged."""
        incomplete = self.valid_single.copy()
        del incomplete["V1"]
        del incomplete["Amount"]

        is_valid, cleaned, errors = validate_single_transaction(incomplete)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Missing required features" in err for err in errors))
        self.assertTrue(any("V1" in err for err in errors))

    def test_03_single_transaction_negative_amount(self):
        """Negative amounts violate business validation rules."""
        invalid = self.valid_single.copy()
        invalid["Amount"] = -25.50

        is_valid, cleaned, errors = validate_single_transaction(invalid)
        self.assertFalse(is_valid)
        self.assertTrue(any("Amount' cannot be negative" in err for err in errors))

    def test_04_single_transaction_nan_infinite(self):
        """NaN or Inf feature values should be rejected."""
        invalid_nan = self.valid_single.copy()
        invalid_nan["V5"] = float("nan")

        is_valid, cleaned, errors = validate_single_transaction(invalid_nan)
        self.assertFalse(is_valid)
        self.assertTrue(any("cannot be NaN or Infinite" in err for err in errors))

    def test_05_valid_batch_dataframe(self):
        """Valid batch dataframe should pass and maintain canonical 30 columns."""
        df = create_sample_csv_dataframe()
        self.assertEqual(len(df), 5)

        is_valid, cleaned_df, errors = validate_batch_dataframe(df)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertEqual(list(cleaned_df.columns), FEATURE_ORDER)

    def test_06_batch_dataframe_with_class_column(self):
        """Batch dataframe containing a ground-truth 'Class' column should drop it safely."""
        df = create_sample_csv_dataframe()
        df["Class"] = [0, 1, 0, 1, 0]

        is_valid, cleaned_df, errors = validate_batch_dataframe(df)
        self.assertTrue(is_valid)
        self.assertNotIn("Class", cleaned_df.columns)
        self.assertEqual(list(cleaned_df.columns), FEATURE_ORDER)

    def test_07_batch_dataframe_missing_column(self):
        """Batch dataframe missing required columns should fail pre-flight validation."""
        df = create_sample_csv_dataframe()
        df = df.drop(columns=["V12", "Amount"])

        is_valid, cleaned_df, errors = validate_batch_dataframe(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing 2 required column(s)" in err for err in errors))

    def test_08_batch_dataframe_empty(self):
        """Empty dataframe should be rejected."""
        empty_df = pd.DataFrame()
        is_valid, cleaned_df, errors = validate_batch_dataframe(empty_df)
        self.assertFalse(is_valid)
        self.assertTrue(any("contains no transaction rows" in err for err in errors))

    def test_09_batch_dataframe_non_numeric(self):
        """DataFrame with non-numeric / string values in feature columns should be flagged."""
        df = create_sample_csv_dataframe()
        df.loc[0, "V1"] = "corrupted_text"

        is_valid, cleaned_df, errors = validate_batch_dataframe(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("missing or non-numeric" in err for err in errors))

    def test_10_sample_template_integrity(self):
        """Sample template generation must provide 5 rows with exact 30 canonical columns."""
        sample_df = create_sample_csv_dataframe()
        self.assertEqual(sample_df.shape, (5, 30))
        self.assertEqual(list(sample_df.columns), FEATURE_ORDER)
        self.assertFalse(sample_df.isnull().any().any())


if __name__ == "__main__":
    unittest.main()
