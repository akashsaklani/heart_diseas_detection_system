"""Data loading, validation, cleaning and splitting (workflow steps 1, 2, 4)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


@dataclass
class CleaningReport:
    rows_raw: int
    duplicates_removed: int
    rows_clean: int
    missing_total: int
    n_columns: int


def load_raw(path: Path | str = config.DATA_RAW) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() and path == config.DATA_RAW and config.DATA_DEMO.exists():
        warnings.warn(
            "\n*** data/raw/heart.csv not found - using SYNTHETIC DEMO data. "
            "Metrics are NOT real. Put the real UCI heart.csv in data/raw/. ***\n",
            stacklevel=2,
        )
        path = config.DATA_DEMO
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the UCI heart.csv there, or run "
            "`python scripts/make_synthetic_data.py` to generate a demo dataset."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    expected = set(config.FEATURES + [config.TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Check missing values, drop duplicates (slide 9)."""
    validate_schema(df)
    # keep only the columns used by the project (11 features + target)
    df = df[config.FEATURES + [config.TARGET]]
    rows_raw = len(df)
    missing_total = int(df.isnull().sum().sum())
    if missing_total:
        df = df.dropna()
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    return df, CleaningReport(
        rows_raw=rows_raw,
        duplicates_removed=dups,
        rows_clean=len(df),
        missing_total=missing_total,
        n_columns=df.shape[1],
    )


def split_xy(df: pd.DataFrame):
    X = df.drop(config.TARGET, axis=1)
    y = df[config.TARGET]
    return X, y


def train_test(df: pd.DataFrame):
    """80:20 split with random_state=42 (241 train / 61 test for 302 rows)."""
    X, y = split_xy(df)
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )


def load_clean(path: Path | str = config.DATA_RAW):
    return clean(load_raw(path))
