"""
locustfile.py
--------------
Locust load test + demo data generator for the Churn Prediction API.

Modes
-----
1. Load test against a running API:
   locust -f locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 -t 60s

2. Generate demo CSV of synthetic predictions (no server needed):
   python locustfile.py --generate --rows 1000 --out demo_predictions.csv

3. Locust web UI:
   locust -f locustfile.py --host http://localhost:8000
   → open http://localhost:8089
"""

from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
from locust import HttpUser, between, task, events
from locust.runners import MasterRunner


# ── Synthetic customer generator ─────────────────────────────────────────────

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
PAYMENT_METHODS = ["bank_transfer", "credit_card", "direct_debit", "check"]
TICKET_CATS = ["billing", "technical", "contract", "outage", "pricing", "other"]
CHANNELS = ["online", "phone", "agent"]
ACTIVITIES = ["a", "b", "c", "d"]
ORIGINS = ["campaign_a", "campaign_b", "campaign_c"]


def make_customer(
    high_risk: bool = False,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Generate one synthetic customer payload for the /predict endpoint.
    high_risk=True skews parameters toward churned profiles for demo purposes.
    """
    if rng is None:
        rng = np.random.default_rng()

    # High-risk profile: low NPS, many tickets, late payments, month-to-month contract
    if high_risk:
        return {
            "cons_12m": float(rng.exponential(5000)),
            "cons_gas_12m": float(rng.exponential(1000)),
            "cons_last_month": float(rng.exponential(600)),
            "imp_cons": float(rng.exponential(300)),
            "net_margin": float(rng.normal(80, 40)),
            "margin_gross_pow_ele": float(rng.normal(60, 30)),
            "num_years_antig": float(rng.integers(1, 3)),
            "pow_max": float(rng.exponential(20)),
            "nb_prod_act": int(rng.integers(1, 2)),
            "forecast_discount_energy": 0.0,
            "channel_sales": rng.choice(CHANNELS),
            "activity_new": rng.choice(ACTIVITIES),
            "origin_up": rng.choice(ORIGINS),
            "nps_score": int(rng.integers(-100, -20)),
            "satisfaction_score": round(float(rng.uniform(1.0, 2.5)), 1),
            "num_contacts_6m": int(rng.integers(8, 15)),
            "last_contact_days_ago": int(rng.integers(200, 365)),
            "contract_type": "month-to-month",
            "num_tickets_6m": int(rng.integers(8, 20)),
            "avg_resolution_hours": round(float(rng.exponential(48)), 1),
            "escalations_6m": int(rng.integers(2, 5)),
            "open_tickets": int(rng.integers(1, 4)),
            "top_ticket_category": rng.choice(["billing", "contract", "pricing"]),
            "num_late_payments_12m": int(rng.integers(3, 8)),
            "avg_days_late": round(float(rng.uniform(15, 60)), 1),
            "payment_method": rng.choice(["check", "credit_card"]),
            "total_outstanding": round(float(rng.exponential(500)), 2),
            "discount_pct": 0,
        }
    else:
        return {
            "cons_12m": float(rng.exponential(15000)),
            "cons_gas_12m": float(rng.exponential(3000)),
            "cons_last_month": float(rng.exponential(1200)),
            "imp_cons": float(rng.exponential(700)),
            "net_margin": float(rng.normal(300, 80)),
            "margin_gross_pow_ele": float(rng.normal(220, 60)),
            "num_years_antig": float(rng.integers(5, 15)),
            "pow_max": float(rng.exponential(60)),
            "nb_prod_act": int(rng.integers(2, 5)),
            "forecast_discount_energy": round(float(rng.uniform(0.05, 0.25)), 2),
            "channel_sales": rng.choice(CHANNELS),
            "activity_new": rng.choice(ACTIVITIES),
            "origin_up": rng.choice(ORIGINS),
            "nps_score": int(rng.integers(20, 100)),
            "satisfaction_score": round(float(rng.uniform(3.5, 5.0)), 1),
            "num_contacts_6m": int(rng.integers(0, 4)),
            "last_contact_days_ago": int(rng.integers(7, 60)),
            "contract_type": rng.choice(["one-year", "two-year"], p=[0.5, 0.5]),
            "num_tickets_6m": int(rng.integers(0, 3)),
            "avg_resolution_hours": round(float(rng.exponential(12)), 1),
            "escalations_6m": 0,
            "open_tickets": 0,
            "top_ticket_category": rng.choice(TICKET_CATS),
            "num_late_payments_12m": int(rng.integers(0, 1)),
            "avg_days_late": 0.0,
            "payment_method": rng.choice(["bank_transfer", "direct_debit"], p=[0.6, 0.4]),
            "total_outstanding": round(float(rng.exponential(50)), 2),
            "discount_pct": int(rng.choice([0, 10, 15, 20])),
        }


# ── Locust user behaviours ────────────────────────────────────────────────────


class ChurnAPIUser(HttpUser):
    """
    Simulates a realistic mix of API consumers:
    - 60% single predictions (typical CRM integration)
    - 25% health checks (monitoring / load balancers)
    - 15% batch predictions (nightly scoring jobs)
    """

    wait_time = between(0.5, 2.0)
    _rng = np.random.default_rng(int(time.time()))

    @task(6)
    def predict_single(self):
        high_risk = random.random() < 0.20  # 20% high-risk profiles
        payload = make_customer(high_risk=high_risk, rng=self._rng)
        with self.client.post(
            "/predict",
            json=payload,
            name="/predict (single)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not (0 <= data["churn_probability"] <= 1):
                    resp.failure("Invalid probability range")
                else:
                    resp.success()
            elif resp.status_code == 503:
                resp.failure("Model not loaded")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(2)
    def health_check(self):
        self.client.get("/health", name="/health")

    @task(1)
    def predict_batch(self):
        batch_size = random.randint(5, 30)
        payload = [make_customer(high_risk=(random.random() < 0.20), rng=self._rng) for _ in range(batch_size)]
        with self.client.post(
            "/predict/batch",
            json=payload,
            name=f"/predict/batch",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                results = resp.json()
                if len(results) != batch_size:
                    resp.failure(f"Expected {batch_size} results, got {len(results)}")
                else:
                    resp.success()
            else:
                resp.failure(f"Batch failed: {resp.status_code}")


# ── Standalone demo data generator ───────────────────────────────────────────


def generate_demo_csv(rows: int, out: str, high_risk_pct: float = 0.20) -> None:
    """
    Generate a CSV of synthetic customer payloads + predicted risk tier.
    Used to demo the system without a running API server.
    """
    rng = np.random.default_rng(42)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(rows):
        high_risk = rng.random() < high_risk_pct
        customer = make_customer(high_risk=high_risk, rng=rng)
        # Simulate a prediction score (for demo only — not the real model)
        base_prob = (0.65 if high_risk else 0.10) + float(rng.normal(0, 0.08))
        prob = float(np.clip(base_prob, 0.01, 0.99))
        tier = "high" if prob >= 0.6 else ("medium" if prob >= 0.3 else "low")
        records.append(
            {
                "customer_id": f"DEMO_{i:05d}",
                **customer,
                "churn_probability": round(prob, 4),
                "churn_prediction": prob >= 0.5,
                "risk_tier": tier,
                "simulated": True,
            }
        )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    churned = sum(1 for r in records if r["churn_prediction"])
    high_ct = sum(1 for r in records if r["risk_tier"] == "high")
    medium_ct = sum(1 for r in records if r["risk_tier"] == "medium")
    low_ct = sum(1 for r in records if r["risk_tier"] == "low")

    print(f"Generated {rows} demo predictions → {path}")
    print(f"  Predicted churners : {churned:>5} ({churned/rows:.1%})")
    print(f"  High risk          : {high_ct:>5} ({high_ct/rows:.1%})")
    print(f"  Medium risk        : {medium_ct:>5} ({medium_ct/rows:.1%})")
    print(f"  Low risk           : {low_ct:>5} ({low_ct/rows:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo prediction data")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--out", type=str, default="reports/demo_predictions.csv")
    parser.add_argument("--high-risk-pct", type=float, default=0.20)
    args = parser.parse_args()

    if args.generate:
        generate_demo_csv(args.rows, args.out, args.high_risk_pct)
    else:
        print("Run with --generate to produce demo data, or use with locust CLI.")
