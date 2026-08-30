"""Feature engineering tests.

The parity tests are the ones that matter. Computing an aggregate with
df.value_counts() inside transform would make a single-row request produce
different values from the same row inside a batch, and both paths would still
run without raising - so only an explicit equality check catches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.features import FeatureEngineer


def test_batch_and_single_row_agree(raw_frame, fitted):
    """A row scored alone must equal that row scored inside a batch."""
    engineer, _ = fitted
    batch = engineer.transform(raw_frame)

    for position in (0, 5, 137, len(raw_frame) - 1):
        single = engineer.transform(raw_frame.iloc[[position]])
        expected = batch.iloc[[position]][single.columns]
        pd.testing.assert_frame_equal(
            single.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            obj=f"row {position}",
        )


def test_output_independent_of_batch_composition(raw_frame, fitted):
    """Splitting the input must not change any row's features."""
    engineer, _ = fitted
    full = engineer.transform(raw_frame)
    halves = pd.concat(
        [engineer.transform(raw_frame.iloc[:150]), engineer.transform(raw_frame.iloc[150:])]
    )
    pd.testing.assert_frame_equal(full, halves[full.columns], check_dtype=False)


def test_output_independent_of_row_order(raw_frame, fitted):
    engineer, _ = fitted
    straight = engineer.transform(raw_frame)
    shuffled = engineer.transform(raw_frame.iloc[::-1]).iloc[::-1]
    pd.testing.assert_frame_equal(straight, shuffled[straight.columns], check_dtype=False)


def test_aggregates_come_from_training_only(raw_frame):
    """Lookup tables must reflect fit data, not whatever transform is handed."""
    train = raw_frame.iloc[:200]
    engineer = FeatureEngineer().fit(train)

    expected = int((train["card1"] == raw_frame["card1"].iloc[0]).sum())
    scored = engineer.transform(raw_frame.iloc[[0]])
    assert scored["card1_freq"].iloc[0] == expected


def test_unseen_card_gets_zero_frequency(raw_frame, fitted):
    engineer, _ = fitted
    row = raw_frame.iloc[[0]].copy()
    row["card1"] = 999999.0
    out = engineer.transform(row)
    assert out["card1_freq"].iloc[0] == 0.0
    # The ratio must stay finite by falling back to the population mean.
    assert np.isfinite(out["card1_amt_ratio"].iloc[0])


def test_missing_v_columns_do_not_rescale_summaries(raw_frame, fitted):
    """A request omitting most V columns must not shift V_sum's scale."""
    engineer, _ = fitted
    row = raw_frame.iloc[[3]]
    full = engineer.transform(row)

    sparse = row.drop(columns=[c for c in row.columns if c.startswith("V")][3:])
    partial = engineer.transform(sparse)

    assert partial["V_null_count"].iloc[0] > full["V_null_count"].iloc[0]
    assert set(full.columns) <= set(partial.columns) | {c for c in row.columns if c.startswith("V")}


def test_minimal_request_survives(fitted):
    """An amount-only payload is the realistic worst case from the API."""
    engineer, _ = fitted
    out = engineer.transform(pd.DataFrame([{"TransactionAmt": 199.99}]))
    assert len(out) == 1
    assert np.isclose(out["TransactionAmt_log"].iloc[0], np.log1p(199.99))


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit must be called"):
        FeatureEngineer().transform(pd.DataFrame([{"TransactionAmt": 1.0}]))


def test_temporal_features(fitted):
    engineer, _ = fitted
    # 23:00 on a Saturday relative to the dataset epoch.
    out = engineer.transform(
        pd.DataFrame([{"TransactionAmt": 10.0, "TransactionDT": 6 * 86400 + 23 * 3600}])
    )
    assert out["hour"].iloc[0] == 23
    assert out["is_night"].iloc[0] == 1.0
    assert out["is_weekend"].iloc[0] == 1.0


def test_round_amount_flag(fitted):
    engineer, _ = fitted
    out = engineer.transform(pd.DataFrame([{"TransactionAmt": 100.0}, {"TransactionAmt": 100.37}]))
    assert out["is_round_amount"].tolist() == [1.0, 0.0]
