from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class SourceMetadata:
    source_name: str
    row_count: int
    column_count: int
    null_rate: float
    duplicate_count: int
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            f"{self.source_name}_rows": self.row_count,
            f"{self.source_name}_cols": self.column_count,
            f"{self.source_name}_null_rate": round(self.null_rate, 4),
            f"{self.source_name}_duplicates": self.duplicate_count,
            **{f"{self.source_name}_{k}": v for k, v in self.extra.items()},
        }

class BaseDataSource(ABC):
    name: str = "base"

    def load(self):
        logger.info("Loading source: %s", self.name)
        df = self.extract()
        self.validate(df)
        meta = self._build_metadata(df)
        return df, meta

    @abstractmethod
    def extract(self) -> pd.DataFrame: ...

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> None: ...

    def _build_metadata(self, df):
        total = df.size or 1
        return SourceMetadata(
            source_name=self.name, row_count=len(df), column_count=len(df.columns),
            null_rate=df.isnull().sum().sum()/total, duplicate_count=int(df.duplicated().sum()),
        )
