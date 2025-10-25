"""Project configuration dataclasses and defaults."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    """Configuration for data ingestion and preprocessing."""

    source_symbols: List[str] = field(
        default_factory=lambda: ["^GSPC", "SPY", "^IXIC"]
    )
    lookback_window: int = 60
    train_start: str = "2005-01-01"
    validation_start: str = "2018-01-01"
    test_start: str = "2021-01-01"
    cache_dir: Path = Path("data/processed")


@dataclass
class ModelConfig:
    """Configuration for the market regime transformer architecture."""

    d_model: int = 256
    n_heads: int = 8
    num_encoder_layers: int = 4
    dropout: float = 0.1
    regression_head_hidden: int = 128
    classification_head_hidden: int = 128
    num_regimes: int = 3


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    regression_loss_weight: float = 0.5
    classification_loss_weight: float = 0.5
    device: str = "cuda"
