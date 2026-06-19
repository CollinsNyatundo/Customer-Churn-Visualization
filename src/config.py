from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", case_sensitive=False, extra="ignore")
    data_raw_dir: Path = ROOT / "data" / "raw"
    data_processed_dir: Path = ROOT / "data" / "processed"
    reports_dir: Path = ROOT / "reports"
    models_dir: Path = ROOT / "models"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_pipeline: str = "churn-data-pipeline"
    mlflow_experiment_model: str = "churn-model"
    mlflow_registered_model_name: str = "ChurnClassifier"
    synthetic_seed: int = 42
    model_test_size: float = 0.2
    model_cv_folds: int = 3
    model_random_state: int = 42
    dash_host: str = "0.0.0.0"
    dash_port: int = 8050
    dash_debug: bool = False

    def ensure_dirs(self):
        for d in [self.data_raw_dir, self.data_processed_dir, self.reports_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
