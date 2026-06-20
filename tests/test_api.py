"""tests/test_api.py — FastAPI endpoint tests using TestClient."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the model loader mocked out."""
    mock_predictor = MagicMock()
    mock_predictor.model_version = "Staging-v1"
    mock_predictor.predict.return_value = (0.72, True, "high")

    with patch("api.main.get_predictor", return_value=mock_predictor):
        from api.main import app

        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "mlflow_uri" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 200

    def test_predict_schema(self, client):
        data = client.post("/predict", json={}).json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_tier" in data
        assert "model_version" in data

    def test_predict_probability_range(self, client):
        data = client.post("/predict", json={}).json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_risk_tier_valid(self, client):
        data = client.post("/predict", json={}).json()
        assert data["risk_tier"] in ("low", "medium", "high")

    def test_nps_validation(self, client):
        """nps_score outside [-100, 100] should return 422."""
        resp = client.post("/predict", json={"nps_score": 999})
        assert resp.status_code == 422

    def test_satisfaction_validation(self, client):
        resp = client.post("/predict", json={"satisfaction_score": 10})
        assert resp.status_code == 422


class TestBatchPredictEndpoint:
    @pytest.fixture()
    def batch_client(self):
        """Isolated client with predict_batch mocked (not module-cached)."""
        import importlib
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        import api.main as api_main

        importlib.reload(api_main)

        mock_pred = MagicMock()
        mock_pred.model_version = "Staging-v1"
        mock_pred.predict.return_value = (0.45, False, "medium")
        mock_pred.predict_batch.return_value = [(0.72, True, "high")] * 3
        mock_pred.get_pipeline.return_value = MagicMock()

        with patch("api.main.get_predictor", return_value=mock_pred):
            yield TestClient(api_main.app)

    def test_batch_returns_list(self, batch_client):
        payload = [{} for _ in range(3)]
        resp = batch_client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_batch_size_limit(self, batch_client):
        payload = [{} for _ in range(501)]
        resp = batch_client.post("/predict/batch", json=payload)
        assert resp.status_code == 400
