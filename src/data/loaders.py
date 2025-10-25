"""Data loading logic for financial time series."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass
class HistoricalDataLoader:
    """Fetch and cache historical price data for a set of symbols."""

    symbols: Iterable[str]
    start_date: str
    end_date: Optional[str] = None
    cache_dir: Path = Path("data/processed")

    def load(self) -> pd.DataFrame:
        """Return a placeholder DataFrame until data access is implemented."""
        columns = ["symbol", "date", "close"]
        return pd.DataFrame(columns=columns)
