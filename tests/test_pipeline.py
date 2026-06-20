"""
tests/test_pipeline.py — Integration tests for the full pipeline and model.
These tests use in-memory fixtures (no disk I/O, no MLflow server).
"""

from unittest.mock import MagicMock, patch

from src.data.pipeline import MultiSourcePipeline
from src.data.sources.bcg_source import BCGClientSource, BCGPriceSource
from src.data.sources.billing_source import BillingSource
from src.data.sources.crm_source import CRMSource
from src.data.sources.support_source import SupportSource


class TestMultiSourcePipeline:
    def test_pipeline_runs_end_to_end(self, sample_client_df, sample_price_df, client_ids):
        """Pipeline should merge all 4 sources without error."""
        client_src = MagicMock(spec=BCGClientSource)
        client_src.name = "bcg_client"
        client_src.load.return_value = (
            sample_client_df,
            MagicMock(as_dict=lambda: {"bcg_client_rows": len(sample_client_df)}),
        )
        price_src = MagicMock(spec=BCGPriceSource)
        price_src.name = "bcg_price"
        price_src.load.return_value = (
            sample_price_df,
            MagicMock(as_dict=lambda: {"bcg_price_rows": len(sample_price_df)}),
        )

        pipeline = MultiSourcePipeline(
            client_source=client_src,
            price_source=price_src,
            crm_source=CRMSource(client_ids=client_ids),
            support_source=SupportSource(client_ids=client_ids),
            billing_source=BillingSource(client_ids=client_ids),
        )
        df, meta = pipeline.run()

        assert len(df) == len(sample_client_df)
        assert "nps_score" in df.columns
        assert "num_tickets_6m" in df.columns
        assert "num_late_payments_12m" in df.columns
        assert "merged_rows" in meta

    def test_metadata_contains_all_sources(self, sample_client_df, sample_price_df, client_ids):
        client_src = MagicMock(spec=BCGClientSource)
        client_src.name = "bcg_client"
        client_src.load.return_value = (
            sample_client_df,
            MagicMock(as_dict=lambda: {"bcg_client_rows": 100}),
        )
        price_src = MagicMock(spec=BCGPriceSource)
        price_src.name = "bcg_price"
        price_src.load.return_value = (
            sample_price_df,
            MagicMock(as_dict=lambda: {"bcg_price_rows": 1200}),
        )
        pipeline = MultiSourcePipeline(
            client_source=client_src,
            price_source=price_src,
            crm_source=CRMSource(client_ids=client_ids),
            support_source=SupportSource(client_ids=client_ids),
            billing_source=BillingSource(client_ids=client_ids),
        )
        _, meta = pipeline.run()

        assert "crm_rows" in meta
        assert "support_rows" in meta
        assert "billing_rows" in meta


class TestModelTraining:
    def test_train_returns_run_id(self, merged_df):
        """Model training should return a non-empty MLflow run ID."""
        with (
            patch("src.models.churn_model.model_run") as mock_run,
            patch("src.models.churn_model.mlflow") as mock_mlflow,
            patch("src.models.churn_model.register_model"),
            patch("src.models.churn_model.log_metrics"),
            patch("src.models.churn_model.log_params"),
        ):
            from contextlib import contextmanager

            from src.tracking.mlflow_tracker import TrackingResult

            @contextmanager
            def fake_model_run(*args, **kwargs):
                yield TrackingResult(run_id="test-run-id-123")

            mock_run.side_effect = fake_model_run
            mock_mlflow.sklearn.log_model = MagicMock()
            mock_mlflow.log_figure = MagicMock()

            from src.models.churn_model import train

            run_id = train(merged_df)

        assert run_id == "test-run-id-123"

    def test_feature_set_sufficient_for_training(self, merged_df):
        """Merged + featured df should have enough columns to not error on sklearn pipeline."""
        from src.models.churn_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES

        available_num = [c for c in NUMERIC_FEATURES if c in merged_df.columns]
        available_cat = [c for c in CATEGORICAL_FEATURES if c in merged_df.columns]
        assert len(available_num) >= 5, "Too few numeric features available"
        assert len(available_cat) >= 1, "Too few categorical features available"
