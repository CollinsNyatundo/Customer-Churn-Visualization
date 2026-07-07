"""
tests/test_alerts.py
----------------------
Tests for the alert dispatch layer (Slack + email), and specifically for
the generic AlertManager.send() method used by src.monitoring.performance.
This module previously had zero test coverage — a real bug (missing
send() method, causing AttributeError whenever a performance-degradation
alert tried to fire) went undetected as a result.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.alerts import AlertManager, EmailAlerter, SlackAlerter


class TestSlackAlerter:
    def test_returns_false_without_webhook_url(self):
        alerter = SlackAlerter(webhook_url="")
        assert alerter.send("title", "message") is False

    def test_sends_when_webhook_configured(self):
        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/fake")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("src.monitoring.alerts.urlopen", return_value=mock_resp):
            assert alerter.send("title", "message") is True

    def test_returns_false_on_url_error(self):
        from urllib.error import URLError

        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/fake")
        with patch("src.monitoring.alerts.urlopen", side_effect=URLError("unreachable")):
            assert alerter.send("title", "message") is False


class TestEmailAlerter:
    def test_returns_false_without_credentials(self):
        alerter = EmailAlerter()
        alerter.from_addr = ""
        assert alerter.send("subject", "body") is False

    def test_sends_when_configured(self):
        alerter = EmailAlerter()
        alerter.from_addr = "a@x.com"
        alerter.to_addr = "b@x.com"
        alerter.smtp_password = "secret"
        with patch("src.monitoring.alerts.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            assert alerter.send("subject", "body") is True
            mock_server.sendmail.assert_called_once()


class TestAlertManager:
    @pytest.fixture()
    def manager_with_mocked_dispatchers(self):
        manager = AlertManager()
        manager.slack = MagicMock()
        manager.email = MagicMock()
        return manager

    def test_send_calls_both_dispatchers(self, manager_with_mocked_dispatchers):
        """
        This is the exact call pattern used by
        src.monitoring.performance.PerformanceMonitor.alert_if_degraded().
        Its absence previously caused an AttributeError at the first real
        performance-degradation event.
        """
        manager = manager_with_mocked_dispatchers
        manager.send(title="Model Performance Degraded", body="AUC dropped below threshold")

        manager.slack.send.assert_called_once()
        manager.email.send.assert_called_once()

    def test_send_drift_alert_calls_both_dispatchers(self, manager_with_mocked_dispatchers):
        manager = manager_with_mocked_dispatchers
        manager.send_drift_alert(n_drifted=3, share=0.15, report_path="/tmp/report.html")
        manager.slack.send.assert_called_once()
        manager.email.send.assert_called_once()

    def test_send_pipeline_failure_calls_both_dispatchers(self, manager_with_mocked_dispatchers):
        manager = manager_with_mocked_dispatchers
        manager.send_pipeline_failure(flow_name="churn-full-pipeline", error="Connection refused")
        manager.slack.send.assert_called_once()
        manager.email.send.assert_called_once()

    def test_send_model_registered_calls_slack_only(self, manager_with_mocked_dispatchers):
        """send_model_registered is informational — Slack only, no email."""
        manager = manager_with_mocked_dispatchers
        manager.send_model_registered(version="3", auc=0.97, stage="Production")
        manager.slack.send.assert_called_once()
        manager.email.send.assert_not_called()
