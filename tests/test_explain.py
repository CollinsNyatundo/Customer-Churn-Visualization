"""tests/test_explain.py — SHAP explainability endpoint tests."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_MOCK_FEATURES = [
    {"feature": "nps_score", "raw_value": -80, "shap_value": 0.142, "direction": "increases_churn"},
    {"feature": "num_late_payments_12m", "raw_value": 5, "shap_value": 0.098, "direction": "increases_churn"},
    {"feature": "satisfaction_score", "raw_value": 1.5, "shap_value": -0.065, "direction": "decreases_churn"},
]


@pytest.fixture(scope="module")
def client():
    mock_pred = MagicMock()
    mock_pred.model_version = "Staging-v1"
    mock_pred.predict.return_value = (0.72, True, "high")
    mock_pred._pipeline = MagicMock()

    with patch("api.main.get_predictor", return_value=mock_pred):
        # Patch where it is *used* in main.py, not where defined
        with patch("api.main.explain_prediction", return_value=_MOCK_FEATURES):
            from api.main import app

            yield TestClient(app)


class TestExplainEndpoint:
    def test_explain_returns_200(self, client):
        assert client.post("/explain", json={}).status_code == 200

    def test_explain_has_prediction_fields(self, client):
        data = client.post("/explain", json={}).json()
        for field in ("churn_probability", "churn_prediction", "risk_tier", "model_version"):
            assert field in data

    def test_explain_has_top_features(self, client):
        data = client.post("/explain", json={}).json()
        assert "top_features" in data
        assert isinstance(data["top_features"], list)
        assert len(data["top_features"]) > 0

    def test_explain_feature_schema(self, client):
        data = client.post("/explain", json={}).json()
        f = data["top_features"][0]
        assert "feature" in f
        assert "shap_value" in f
        assert "direction" in f
        assert f["direction"] in ("increases_churn", "decreases_churn")

    def test_explain_top_n_param(self, client):
        assert client.post("/explain?top_n=3", json={}).status_code == 200

    def test_probability_in_range(self, client):
        data = client.post("/explain", json={}).json()
        assert 0 <= data["churn_probability"] <= 1
