"""Preprocessing contract tests.

Guards the invariant that any input - a five-field API payload or a full
training row - produces the identical column set the imputer was fitted on.
Re-deriving the numeric/categorical split from dtypes breaks this and makes
every /predict call fail with an imputer shape error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import EXCLUDE_COLS, UNKNOWN_CATEGORY
from fraudlens.preprocessing import Preprocessor


def test_single_row_matches_batch(raw_frame, fitted):
    engineer, preprocessor = fitted
    engineered = engineer.transform(raw_frame)
    batch = preprocessor.transform(engineered)

    for position in (0, 42, len(raw_frame) - 1):
        single = preprocessor.transform(engineered.iloc[[position]])
        pd.testing.assert_frame_equal(
            single.reset_index(drop=True),
            batch.iloc[[position]].reset_index(drop=True),
            check_dtype=False,
        )


def test_sparse_request_produces_full_matrix(fitted):
    """The API's realistic payload must yield the exact training schema."""
    engineer, preprocessor = fitted
    payload = {
        "TransactionAmt": 299.99,
        "ProductCD": "W",
        "card4": "visa",
        "P_emaildomain": "gmail.com",
        "TransactionDT": 86400.0,
    }
    X = preprocessor.transform(engineer.transform(pd.DataFrame([payload])))

    assert list(X.columns) == preprocessor.feature_cols_
    assert X.shape == (1, len(preprocessor.feature_cols_))
    assert not X.isna().any().any(), "imputation must leave no NaNs for the model"


def test_column_order_is_stable(raw_frame, fitted):
    engineer, preprocessor = fitted
    engineered = engineer.transform(raw_frame)
    shuffled = engineered[list(engineered.columns)[::-1]]
    pd.testing.assert_frame_equal(
        preprocessor.transform(engineered), preprocessor.transform(shuffled)
    )


def test_unknown_category_is_isolated(fitted):
    """An unseen category must not impersonate a real one."""
    engineer, preprocessor = fitted
    known = preprocessor.transform(
        engineer.transform(pd.DataFrame([{"TransactionAmt": 50.0, "card4": "visa"}]))
    )
    unknown = preprocessor.transform(
        engineer.transform(pd.DataFrame([{"TransactionAmt": 50.0, "card4": "brand-new-network"}]))
    )
    assert known["card4"].iloc[0] != unknown["card4"].iloc[0]

    encoder = preprocessor.label_encoders_["card4"]
    assert unknown["card4"].iloc[0] == encoder.transform([UNKNOWN_CATEGORY])[0]


def test_extra_columns_are_dropped(fitted):
    engineer, preprocessor = fitted
    frame = engineer.transform(pd.DataFrame([{"TransactionAmt": 20.0}]))
    frame["a_column_training_never_saw"] = 1.0
    assert list(preprocessor.transform(frame).columns) == preprocessor.feature_cols_


def test_target_and_ids_never_enter_the_matrix(raw_frame, fitted):
    _, preprocessor = fitted
    for col in EXCLUDE_COLS:
        assert col not in preprocessor.feature_cols_


def test_high_missingness_columns_dropped():
    frame = pd.DataFrame(
        {
            "TransactionAmt": np.arange(100, dtype=float),
            "mostly_missing": [1.0] + [np.nan] * 99,
            "dense": np.arange(100, dtype=float),
        }
    )
    preprocessor = Preprocessor(max_missing_fraction=0.8).fit(frame)
    assert "mostly_missing" in preprocessor.dropped_cols_
    assert "dense" in preprocessor.feature_cols_


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit must be called"):
        Preprocessor().transform(pd.DataFrame([{"TransactionAmt": 1.0}]))
