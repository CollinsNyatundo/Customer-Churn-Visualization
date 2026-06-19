"""tests/test_auth.py — API key authentication tests."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client_no_auth():
    """Client with no API_KEYS set → dev bypass mode."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("API_KEYS", None)
        from api.auth import _load_valid_keys

        _load_valid_keys.cache_clear()
        from unittest.mock import MagicMock

        mock_pred = MagicMock()
        mock_pred.model_version = "Staging-v1"
        mock_pred.predict.return_value = (0.45, False, "medium")

        with patch("api.main.get_predictor", return_value=mock_pred):
            from api.main import app

            yield TestClient(app)
        _load_valid_keys.cache_clear()


@pytest.fixture(scope="module")
def client_with_auth():
    """Client with API_KEYS configured."""
    with patch.dict(os.environ, {"API_KEYS": "valid-key-abc,valid-key-xyz"}):
        from api.auth import _load_valid_keys

        _load_valid_keys.cache_clear()
        from unittest.mock import MagicMock

        mock_pred = MagicMock()
        mock_pred.model_version = "Staging-v1"
        mock_pred.predict.return_value = (0.75, True, "high")

        with patch("api.main.get_predictor", return_value=mock_pred):
            from api.main import app

            yield TestClient(app, raise_server_exceptions=True)
        _load_valid_keys.cache_clear()


class TestDevBypassMode:
    def test_predict_without_key_allowed(self, client_no_auth):
        resp = client_no_auth.post("/predict", json={})
        assert resp.status_code == 200

    def test_health_always_accessible(self, client_no_auth):
        assert client_no_auth.get("/health").status_code == 200

    def test_auth_enabled_false_in_health(self, client_no_auth):
        data = client_no_auth.get("/health").json()
        assert data["auth_enabled"] is False


class TestApiKeyEnforcement:
    def test_missing_key_returns_401(self, client_with_auth):
        resp = client_with_auth.post("/predict", json={})
        assert resp.status_code == 401

    def test_invalid_key_returns_403(self, client_with_auth):
        resp = client_with_auth.post("/predict", json={}, headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    def test_valid_key_returns_200(self, client_with_auth):
        resp = client_with_auth.post("/predict", json={}, headers={"X-API-Key": "valid-key-abc"})
        assert resp.status_code == 200

    def test_second_valid_key_also_works(self, client_with_auth):
        resp = client_with_auth.post("/predict", json={}, headers={"X-API-Key": "valid-key-xyz"})
        assert resp.status_code == 200
