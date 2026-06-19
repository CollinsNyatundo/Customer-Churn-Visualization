"""
src/visualizations/eda.py
--------------------------
Reusable Plotly figure factories. Every function returns a go.Figure
usable in both Jupyter (fig.show()) and Dash (dcc.Graph(figure=...)).
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

_C = {"churned": "#E24B4A", "retained": "#1D9E75"}
_T = "plotly_white"


def churn_overview(df: pd.DataFrame) -> go.Figure:
    counts = df["churn"].value_counts().reset_index()
    counts.columns = ["churn", "count"]
    counts["label"] = counts["churn"].map({True: "Churned", False: "Retained"})
    return px.pie(counts, names="label", values="count",
                  color="label", color_discrete_map={"Churned": _C["churned"], "Retained": _C["retained"]},
                  title="Overall churn rate", template=_T, hole=0.45)


def churn_by_tenure(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["tenure_band"] = pd.cut(df["num_years_antig"],
                               bins=[0, 1, 3, 5, 10, 100],
                               labels=["<1yr", "1-3yr", "3-5yr", "5-10yr", "10+yr"])
    rates = df.groupby("tenure_band", observed=True)["churn"].mean().mul(100).round(1).reset_index()
    return px.bar(rates, x="tenure_band", y="churn", title="Churn rate by tenure",
                  labels={"tenure_band": "Tenure", "churn": "Churn rate (%)"},
                  color="churn", color_continuous_scale=["#1D9E75", "#E24B4A"], template=_T)


def churn_by_consumption(df: pd.DataFrame, bins: int = 6) -> go.Figure:
    df = df.copy()
    df["cons_band"] = pd.qcut(df["cons_12m"], q=bins, duplicates="drop").astype(str)
    rates = df.groupby("cons_band", observed=True)["churn"].mean().mul(100).round(1).reset_index()
    return px.bar(rates, x="cons_band", y="churn",
                  title="Churn rate by 12-month consumption",
                  labels={"cons_band": "Consumption band (kWh)", "churn": "Churn rate (%)"},
                  color="churn", color_continuous_scale=["#1D9E75", "#E24B4A"], template=_T)


def churn_by_category(df: pd.DataFrame, col: str, title: str | None = None) -> go.Figure:
    rates = (df.groupby(col)["churn"].mean().mul(100).round(1)
             .sort_values().reset_index())
    rates.columns = [col, "churn_rate"]
    return px.bar(rates, x="churn_rate", y=col, orientation="h",
                  title=title or f"Churn rate by {col}",
                  color="churn_rate", color_continuous_scale=["#1D9E75", "#E24B4A"], template=_T)


def churn_by_nps(df: pd.DataFrame) -> go.Figure:
    if "nps_score" not in df.columns:
        return go.Figure()
    df = df.copy()
    df["nps_band"] = pd.cut(df["nps_score"], bins=[-101, -30, 0, 30, 101],
                            labels=["Detractor", "Passive-", "Passive+", "Promoter"])
    rates = df.groupby("nps_band", observed=True)["churn"].mean().mul(100).round(1).reset_index()
    return px.bar(rates, x="nps_band", y="churn", title="Churn rate by NPS band",
                  labels={"nps_band": "NPS Band", "churn": "Churn rate (%)"},
                  color="churn", color_continuous_scale=["#1D9E75", "#E24B4A"], template=_T)


def churn_by_support_tickets(df: pd.DataFrame) -> go.Figure:
    if "num_tickets_6m" not in df.columns:
        return go.Figure()
    df = df.copy()
    df["ticket_band"] = pd.cut(df["num_tickets_6m"], bins=[-1, 0, 2, 5, 100],
                               labels=["None", "1-2", "3-5", "6+"])
    rates = df.groupby("ticket_band", observed=True)["churn"].mean().mul(100).round(1).reset_index()
    return px.bar(rates, x="ticket_band", y="churn", title="Churn rate by support ticket volume",
                  labels={"ticket_band": "Tickets (6m)", "churn": "Churn rate (%)"},
                  color="churn", color_continuous_scale=["#1D9E75", "#E24B4A"], template=_T)


def margin_vs_churn(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["churn_label"] = df["churn"].map({True: "Churned", False: "Retained"})
    return px.scatter(df.sample(min(3000, len(df)), random_state=42),
                      x="cons_12m", y="net_margin", color="churn_label",
                      color_discrete_map={"Churned": _C["churned"], "Retained": _C["retained"]},
                      opacity=0.5, title="Net margin vs consumption by churn",
                      labels={"cons_12m": "12m consumption (kWh)", "net_margin": "Net margin"},
                      template=_T)


def correlation_heatmap(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    numeric = df.select_dtypes(include="number").copy()
    if "churn" not in numeric.columns:
        numeric["churn"] = df["churn"].astype(int)
    corr_with_churn = numeric.corr()["churn"].drop("churn").abs().nlargest(top_n)
    top_cols = corr_with_churn.index.tolist() + ["churn"]
    corr_matrix = numeric[top_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns,
                               y=corr_matrix.index, colorscale="RdBu", zmid=0,
                               text=corr_matrix.values, texttemplate="%{text}"))
    fig.update_layout(title=f"Correlation heatmap (top {top_n} vs churn)",
                      template=_T, height=520)
    return fig
