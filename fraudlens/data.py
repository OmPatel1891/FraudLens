"""Loading and joining the two IEEE-CIS source tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR, ID_COL


def normalise_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename id-01..id-38 to id_01..id_38.

    Kaggle ships test_identity.csv with hyphens while train_identity.csv uses
    underscores. Merging without this rename silently produces a test frame
    whose identity features cannot align with the training schema.
    """
    return df.rename(columns=lambda c: c.replace("-", "_") if c.startswith("id-") else c)


def load_split(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and left-join the transaction and identity tables for one split."""
    data_dir = Path(data_dir)
    txn_path = data_dir / f"{split}_transaction.csv"
    id_path = data_dir / f"{split}_identity.csv"

    if not txn_path.exists():
        raise FileNotFoundError(
            f"{txn_path} not found. Run `python scripts/download_data.py` for the real "
            f"Kaggle data, or `python scripts/make_synthetic_data.py` for a runnable stand-in."
        )

    transactions = pd.read_csv(txn_path)
    identity = normalise_identity_columns(pd.read_csv(id_path))

    merged = transactions.merge(identity, on=ID_COL, how="left")
    return merged


def load_train_test(data_dir: Path = DATA_DIR):
    train = load_split("train", data_dir)
    test = load_split("test", data_dir)

    # Fail loudly if the two frames still disagree on identity columns; a
    # silent mismatch surfaces much later as an imputer shape error.
    train_ids = {c for c in train.columns if c.startswith("id_")}
    test_ids = {c for c in test.columns if c.startswith("id_")}
    missing = train_ids - test_ids - {"isFraud"}
    if missing:
        raise ValueError(
            f"train has identity columns absent from test: {sorted(missing)[:5]}... "
            "identity column normalisation failed"
        )

    return train, test
