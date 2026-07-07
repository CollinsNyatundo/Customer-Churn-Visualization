"""
src/monitoring/alerts.py
-------------------------
Alert dispatchers for drift detection and pipeline failures.
Supports Slack webhooks and SMTP email out of the box.

Configuration (via .env):
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
    ALERT_EMAIL_FROM=alerts@yourcompany.com
    ALERT_EMAIL_TO=team@yourcompany.com
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=alerts@yourcompany.com
    SMTP_PASSWORD=your-app-password
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class SlackAlerter:
    """Sends formatted alerts to a Slack channel via incoming webhook."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")

    def send(self, title: str, message: str, level: str = "warning") -> bool:
        if not self.webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not set — Slack alert skipped.")
            return False

        emoji = {"info": ":information_source:", "warning": ":warning:", "error": ":red_circle:"}.get(level, ":bell:")
        colour = {"info": "#36a64f", "warning": "#ff9800", "error": "#e53935"}.get(level, "#888888")

        payload = {
            "attachments": [
                {
                    "color": colour,
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": f"{emoji} {title}"}},
                        {"type": "section", "text": {"type": "mrkdwn", "text": message}},
                        {
                            "type": "context",
                            "elements": [{"type": "mrkdwn", "text": "*Source:* Churn ML System"}],
                        },
                    ],
                }
            ]
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(self.webhook_url, data=data, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=10) as resp:
                ok = resp.status == 200
                if ok:
                    logger.info("Slack alert sent: %s", title)
                return ok
        except URLError as e:
            logger.error("Slack alert failed: %s", e)
            return False


class EmailAlerter:
    """Sends plain-text/HTML alerts via SMTP."""

    def __init__(self):
        self.from_addr = os.getenv("ALERT_EMAIL_FROM", "")
        self.to_addr = os.getenv("ALERT_EMAIL_TO", "")
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")

    def send(self, subject: str, body: str) -> bool:
        if not all([self.from_addr, self.to_addr, self.smtp_password]):
            logger.warning("Email credentials not configured — email alert skipped.")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = self.to_addr
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_addr, self.to_addr, msg.as_string())
            logger.info("Email alert sent to %s: %s", self.to_addr, subject)
            return True
        except Exception as e:
            logger.error("Email alert failed: %s", e)
            return False


class AlertManager:
    """
    Unified alert dispatcher. Tries Slack first, falls back to email.
    Use send_drift_alert() and send_pipeline_alert() for typed alerts.
    """

    def __init__(self):
        self.slack = SlackAlerter()
        self.email = EmailAlerter()

    def send(self, title: str, body: str, level: str = "warning") -> None:
        """
        Generic alert dispatch — tries Slack, then email, with the given
        title/body/level. Used by callers (e.g. src.monitoring.performance)
        that don't fit one of the typed alert methods below.
        """
        self.slack.send(title, body, level=level)
        self.email.send(subject=f"[Churn ML] {title}", body=body)

    def send_drift_alert(self, n_drifted: int, share: float, report_path: str) -> None:
        title = "Data Drift Detected"
        message = (
            f"*{n_drifted} features* have drifted ({share:.1%} of tracked features).\n"
            f"Report saved to `{report_path}`.\n"
            f"*Action required*: review and consider model retraining."
        )
        self.slack.send(title, message, level="warning")
        self.email.send(
            subject=f"[Churn ML] {title} — {n_drifted} features",
            body=f"{title}\n\n{message.replace('*', '')}",
        )

    def send_pipeline_failure(self, flow_name: str, error: str) -> None:
        title = f"Pipeline Failure: {flow_name}"
        message = f"*Flow*: `{flow_name}`\n*Error*: ```{error[:500]}```"
        self.slack.send(title, message, level="error")
        self.email.send(
            subject=f"[Churn ML] Pipeline Failed: {flow_name}",
            body=f"Flow: {flow_name}\n\nError:\n{error}",
        )

    def send_model_registered(self, version: str, auc: float, stage: str) -> None:
        title = "New Model Registered"
        message = f"*Version*: {version} | *AUC*: {auc:.4f} | *Stage*: {stage}"
        self.slack.send(title, message, level="info")
