"""
test_api_client.py
------------------
Unit tests for APIClient using unittest.mock to simulate FastAPI server responses
and failure modes without requiring a running network daemon.
"""

import unittest
from unittest.mock import patch, MagicMock
from requests.exceptions import ConnectionError, Timeout

from frontend.api_client import APIClient
from frontend.validators import get_sample_preset


class TestAPIClient(unittest.TestCase):
    def setUp(self):
        self.client = APIClient(base_url="http://127.0.0.1:8000", timeout=2.0)
        self.sample = get_sample_preset(fraud=False)

    @patch("frontend.api_client.requests.get")
    def test_01_health_check_healthy(self, mock_get):
        """Mock successful health check response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "healthy",
            "model_loaded": True,
            "scaler_loaded": True,
            "model_type": "XGBClassifier",
            "expected_features_count": 30,
        }
        mock_get.return_value = mock_resp

        is_healthy, data, err = self.client.check_health()
        self.assertTrue(is_healthy)
        self.assertIsNone(err)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("latency_ms", data)

    @patch("frontend.api_client.requests.get")
    def test_02_health_check_unhealthy(self, mock_get):
        """Mock 503 Service Unavailable health check response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service unavailable: model artifacts could not be loaded"
        mock_get.return_value = mock_resp

        is_healthy, data, err = self.client.check_health()
        self.assertFalse(is_healthy)
        self.assertIn("503", err)

    @patch("frontend.api_client.requests.get")
    def test_03_health_check_connection_error(self, mock_get):
        """Mock connection error when FastAPI is offline."""
        mock_get.side_effect = ConnectionError("Connection refused")

        is_healthy, data, err = self.client.check_health()
        self.assertFalse(is_healthy)
        self.assertIn("Unable to connect", err)

    @patch("frontend.api_client.requests.get")
    def test_04_get_sample_success(self, mock_get):
        """Mock retrieving sample transaction from API."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "type": "Legitimate Example",
            "payload": self.sample,
        }
        mock_get.return_value = mock_resp

        sample, err = self.client.get_sample(fraud=False)
        self.assertIsNotNone(sample)
        self.assertIsNone(err)
        self.assertEqual(sample["Amount"], self.sample["Amount"])

    @patch("frontend.api_client.requests.post")
    def test_05_predict_single_success(self, mock_post):
        """Mock successful single transaction prediction."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "prediction": 0,
            "label": "Legitimate",
            "fraud_probability": 0.0125,
            "threshold_used": 0.5,
            "risk_level": "LOW",
        }
        mock_post.return_value = mock_resp

        result, err = self.client.predict_single(self.sample, threshold=0.5)
        self.assertIsNotNone(result)
        self.assertIsNone(err)
        self.assertEqual(result["prediction"], 0)
        self.assertEqual(result["label"], "Legitimate")
        self.assertEqual(result["risk_level"], "LOW")

    @patch("frontend.api_client.requests.post")
    def test_06_predict_single_validation_error(self, mock_post):
        """Mock 422 Unprocessable Entity error from backend schema validator."""
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {"detail": "Field 'Time' required"}
        mock_post.return_value = mock_resp

        result, err = self.client.predict_single(self.sample)
        self.assertIsNone(result)
        self.assertIn("API Validation Error", err)

    @patch("frontend.api_client.requests.post")
    def test_07_predict_single_timeout(self, mock_post):
        """Mock request timeout during prediction."""
        mock_post.side_effect = Timeout("Timed out")

        result, err = self.client.predict_single(self.sample)
        self.assertIsNone(result)
        self.assertIn("timed out", err)

    @patch("frontend.api_client.requests.post")
    def test_08_predict_batch_success(self, mock_post):
        """Mock successful batch prediction."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "total_processed": 2,
            "fraud_count": 1,
            "legitimate_count": 1,
            "predictions": [
                {
                    "prediction": 0,
                    "label": "Legitimate",
                    "fraud_probability": 0.05,
                    "threshold_used": 0.5,
                    "risk_level": "LOW",
                },
                {
                    "prediction": 1,
                    "label": "Fraudulent",
                    "fraud_probability": 0.95,
                    "threshold_used": 0.5,
                    "risk_level": "HIGH",
                },
            ],
        }
        mock_post.return_value = mock_resp

        batch_result, err = self.client.predict_batch([self.sample, self.sample], threshold=0.5)
        self.assertIsNotNone(batch_result)
        self.assertIsNone(err)
        self.assertEqual(batch_result["total_processed"], 2)
        self.assertEqual(batch_result["fraud_count"], 1)
        self.assertEqual(batch_result["legitimate_count"], 1)
        self.assertAlmostEqual(batch_result["fraud_rate"], 0.5)

    @patch("frontend.api_client.requests.post")
    def test_09_predict_batch_chunking(self, mock_post):
        """Mock batch processing chunking logic for large payloads."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "total_processed": 2,
            "fraud_count": 0,
            "legitimate_count": 2,
            "predictions": [
                {"prediction": 0, "label": "Legitimate", "fraud_probability": 0.01, "threshold_used": 0.5, "risk_level": "LOW"},
                {"prediction": 0, "label": "Legitimate", "fraud_probability": 0.02, "threshold_used": 0.5, "risk_level": "LOW"},
            ],
        }
        mock_post.return_value = mock_resp

        # Send 4 items with chunk_size=2 -> should result in 2 post calls
        items = [self.sample] * 4
        batch_result, err = self.client.predict_batch(items, threshold=0.5, chunk_size=2)
        self.assertIsNotNone(batch_result)
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(batch_result["total_processed"], 4)

    def test_10_predict_batch_empty_list(self):
        """Empty transaction list should return an error immediately."""
        result, err = self.client.predict_batch([])
        self.assertIsNone(result)
        self.assertIn("empty", err)


if __name__ == "__main__":
    unittest.main()
