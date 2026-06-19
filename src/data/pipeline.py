from __future__ import annotations
import logging
import pandas as pd
from src.data.preprocessing import clean_client_data, aggregate_price_data, null_report
from src.data.sources.base import SourceMetadata
from src.data.sources.bcg_source import BCGClientSource, BCGPriceSource
from src.data.sources.billing_source import BillingSource
from src.data.sources.crm_source import CRMSource
from src.data.sources.support_source import SupportSource

logger = logging.getLogger(__name__)

class MultiSourcePipeline:
    def __init__(self, client_source=None, price_source=None,
                 crm_source=None, support_source=None, billing_source=None):
        self._client_src = client_source or BCGClientSource()
        self._price_src = price_source or BCGPriceSource()
        self._crm_src = crm_source
        self._support_src = support_source
        self._billing_src = billing_source

    def run(self):
        all_meta = {}
        client_df, meta = self._client_src.load()
        all_meta.update(meta.as_dict())
        client_df = clean_client_data(client_df)
        client_ids = client_df["id"].tolist()

        price_df, meta = self._price_src.load()
        all_meta.update(meta.as_dict())
        price_agg = aggregate_price_data(price_df)

        crm_src = self._crm_src or CRMSource(client_ids=client_ids)
        if not crm_src.client_ids: crm_src.client_ids = client_ids
        crm_df, meta = crm_src.load()
        all_meta.update(meta.as_dict())

        support_src = self._support_src or SupportSource(client_ids=client_ids)
        if not support_src.client_ids: support_src.client_ids = client_ids
        support_df, meta = support_src.load()
        all_meta.update(meta.as_dict())

        billing_src = self._billing_src or BillingSource(client_ids=client_ids)
        if not billing_src.client_ids: billing_src.client_ids = client_ids
        billing_df, meta = billing_src.load()
        all_meta.update(meta.as_dict())

        df = (client_df
              .merge(price_agg, on="id", how="left")
              .merge(crm_df, on="id", how="left")
              .merge(support_df, on="id", how="left")
              .merge(billing_df, on="id", how="left"))

        all_meta.update({"merged_rows": len(df), "merged_cols": len(df.columns),
                         "merged_null_rate": round(df.isnull().sum().sum()/max(df.size,1),4)})
        return df, all_meta
