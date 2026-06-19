"""app.py — Production Dash entry point using MultiSourcePipeline."""
import os
import dash
from src.logging_config import configure_logging
from src.config import settings
from src.data.pipeline import MultiSourcePipeline
from src.features.engineering import build_feature_set
from src.visualizations.dashboard import create_layout, register_callbacks

configure_logging(level=os.getenv("LOG_LEVEL","INFO"), json=os.getenv("LOG_FORMAT","text")=="json")
settings.ensure_dirs()

pipeline = MultiSourcePipeline()
df, _ = pipeline.run()
df = build_feature_set(df)

app = dash.Dash(__name__, title="Customer Churn Dashboard", assets_folder="assets")
server = app.server
app.layout = create_layout(df)
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=settings.dash_debug, host=settings.dash_host, port=settings.dash_port)
