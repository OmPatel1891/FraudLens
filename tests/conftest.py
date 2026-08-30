"""Shared fixtures. Builds a small in-memory dataset shaped like IEEE-CIS."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudlens.features import FeatureEngineer
from fraudlens.preprocessing import Preprocessor

N_V, N_C, N_D = 12, 14, 15


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 400

    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "isFraud": rng.integers(0, 2, n),
            "TransactionDT": np.sort(rng.integers(86400, 86400 * 120, n)).astype(float),
            "TransactionAmt": np.round(rng.uniform(1, 900, n), 2),
            "ProductCD": rng.choice(["W", "H", "C"], n),
            "card1": rng.integers(1000, 1100, n).astype(float),
            "card2": rng.integers(100, 130, n).astype(float),
            "card3": rng.choice([150.0, 185.0], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "card5": rng.integers(100, 120, n).astype(float),
            "card6": rng.choice(["credit", "debit"], n),
            "addr1": rng.integers(100, 400, n).astype(float),
            "addr2": np.full(n, 87.0),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
            "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", None], n),
            "DeviceType": rng.choice(["desktop", "mobile", None], n),
            "DeviceInfo": rng.choice(["Windows", "iOS Device", None], n),
        }
    )

    for i in range(1, N_C + 1):
        df[f"C{i}"] = rng.poisson(2, n).astype(float)
    for i in range(1, N_D + 1):
        values = rng.integers(0, 400, n).astype(float)
        values[rng.random(n) < 0.2] = np.nan
        df[f"D{i}"] = values
    for i in range(1, N_V + 1):
        values = rng.normal(0, 1, n)
        values[rng.random(n) < 0.15] = np.nan
        df[f"V{i}"] = values

    return df


@pytest.fixture(scope="session")
def fitted(raw_frame):
    """Engineer and preprocessor fitted on the first 70% by time."""
    cutoff = raw_frame["TransactionDT"].quantile(0.7)
    train = raw_frame[raw_frame["TransactionDT"] <= cutoff]

    engineer = FeatureEngineer().fit(train)
    preprocessor = Preprocessor().fit(engineer.transform(train))
    return engineer, preprocessor
