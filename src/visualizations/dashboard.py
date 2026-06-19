"""
src/visualizations/dashboard.py
---------------------------------
Dash layout + callbacks. Kept free of app instantiation for testability.
Imports create_layout() and register_callbacks() from app.py.
"""
from __future__ import annotations

import pandas as pd
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

from src.visualizations.eda import (
    churn_overview, churn_by_tenure, churn_by_consumption,
    churn_by_category, churn_by_nps, churn_by_support_tickets,
    margin_vs_churn, correlation_heatmap,
)


def create_layout(df: pd.DataFrame) -> html.Div:
    cat_options = [
        {"label": "Sales channel",  "value": "channel_sales"},
        {"label": "Activity type",  "value": "activity_new"},
        {"label": "Campaign origin","value": "origin_up"},
        {"label": "Contract type",  "value": "contract_type"},
        {"label": "Payment method", "value": "payment_method"},
    ]

    return html.Div(
        style={"fontFamily": "Arial, sans-serif", "maxWidth": "1280px",
               "margin": "0 auto", "padding": "24px"},
        children=[
            html.H1("Customer Churn Dashboard"),
            html.P(
                f"{len(df):,} customers · {df['churn'].mean()*100:.1f}% churn rate · "
                "Sources: BCG · CRM · Support · Billing",
                style={"color": "#666", "marginTop": "4px"},
            ),
            html.Hr(),

            # Row 1
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                            "gap": "24px", "marginBottom": "24px"},
                     children=[
                         dcc.Graph(figure=churn_overview(df)),
                         dcc.Graph(figure=churn_by_tenure(df)),
                     ]),

            # Row 2
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                            "gap": "24px", "marginBottom": "24px"},
                     children=[
                         dcc.Graph(figure=churn_by_nps(df)),
                         dcc.Graph(figure=churn_by_support_tickets(df)),
                     ]),

            # Row 3
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                            "gap": "24px", "marginBottom": "24px"},
                     children=[
                         dcc.Graph(figure=churn_by_consumption(df)),
                         dcc.Graph(figure=margin_vs_churn(df)),
                     ]),

            # Dynamic category chart
            html.Div(style={"marginBottom": "24px"}, children=[
                html.Label("Churn rate by category:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="cat-dropdown", options=cat_options,
                             value="channel_sales", clearable=False,
                             style={"width": "320px", "margin": "8px 0"}),
                dcc.Graph(id="cat-bar"),
            ]),

            # Correlation heatmap
            dcc.Graph(figure=correlation_heatmap(df)),

            # Hidden store
            dcc.Store(id="store", data=df.to_json(date_format="iso")),
        ],
    )


def register_callbacks(app) -> None:
    @app.callback(Output("cat-bar", "figure"),
                  Input("cat-dropdown", "value"),
                  Input("store", "data"))
    def update_cat(col: str, json_data: str) -> go.Figure:
        df = pd.read_json(json_data)
        df["churn"] = df["churn"].astype(bool)
        return churn_by_category(df, col)
