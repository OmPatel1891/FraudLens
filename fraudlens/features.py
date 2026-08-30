"""Feature engineering shared by training and serving.

The rule this module exists to enforce: `transform` must produce identical
output for a row whether that row arrives alone or inside a million-row batch.
Any statistic computed across rows is therefore learned once in `fit` (on
training data only) and frozen into lookup tables that ship with the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CARD_COLS, DEVICE_COLS, EMAIL_COLS


def _digit_suffixed(columns, prefix: str) -> list:
    """Columns like V1/C14 - prefix followed only by digits."""
    return sorted(
        (c for c in columns if c.startswith(prefix) and c[len(prefix):].isdigit()),
        key=lambda c: int(c[len(prefix):]),
    )


class FeatureEngineer:
    """Builds fraud features from raw merged transaction+identity rows.

    Aggregate features (per-card frequency, per-card mean amount, domain
    frequency) are genuinely useful but cannot be derived from a single
    incoming transaction. `fit` precomputes them into lookup tables, which
    makes them available at serving time and simultaneously stops validation
    rows from leaking into training-time statistics.
    """

    def __init__(self) -> None:
        self.fitted_ = False

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        """Learn lookup tables. Pass training rows only."""
        # Pin the column groups so row-wise aggregates stay comparable even
        # when a request omits most V/C/D columns.
        self.v_cols_ = _digit_suffixed(df.columns, "V")
        self.c_cols_ = _digit_suffixed(df.columns, "C")
        self.d_cols_ = _digit_suffixed(df.columns, "D")

        amt = df["TransactionAmt"]
        self.global_amt_mean_ = float(amt.mean())
        self.global_amt_median_ = float(amt.median())
        self.global_amt_std_ = float(amt.std())

        self.card_freq_, self.card_mean_amt_, self.card_std_amt_ = {}, {}, {}
        for col in CARD_COLS:
            if col not in df.columns:
                continue
            self.card_freq_[col] = df[col].value_counts()
            grouped = df.groupby(col)["TransactionAmt"]
            self.card_mean_amt_[col] = grouped.mean()
            self.card_std_amt_[col] = grouped.std()

        self.domain_freq_ = {}
        for col in EMAIL_COLS + DEVICE_COLS:
            if col in df.columns:
                self.domain_freq_[col] = df[col].value_counts()

        self.n_train_rows_ = len(df)
        self.fitted_ = True
        return self

    # ── transform ────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply engineering. Safe for a single row or a full dataset."""
        if not self.fitted_:
            raise RuntimeError("FeatureEngineer.fit must be called before transform")

        # Each block reads only raw input columns, never another block's
        # output, so all of them can be accumulated and attached in one concat.
        # Adding ~100 columns one at a time reallocates the frame repeatedly.
        new: dict = {}
        self._add_temporal(df, new)
        self._add_amount(df, new)
        self._add_d_normalised(df, new)
        self._add_card_aggregates(df, new)
        self._add_email(df, new)
        self._add_domain_frequency(df, new)
        self._add_group_summaries(df, new)

        # Engineered values win where a name collides with a raw column.
        keep = [c for c in df.columns if c not in new]
        return pd.concat(
            [df[keep], pd.DataFrame(new, index=df.index)], axis=1, copy=False
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ── individual feature blocks ────────────────────────────────────────────
    @staticmethod
    def _column(df: pd.DataFrame, col: str) -> pd.Series:
        """Fetch a column, or an all-NaN stand-in when the caller omitted it."""
        if col in df.columns:
            return df[col]
        return pd.Series(np.nan, index=df.index, dtype="float64")

    def _numeric(self, df: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_numeric(self._column(df, col), errors="coerce")

    def _add_temporal(self, df: pd.DataFrame, new: dict) -> None:
        dt = self._numeric(df, "TransactionDT")
        hour = (dt // 3600) % 24
        dow = (dt // 86400) % 7

        new["hour"] = hour
        new["day"] = dow
        # Keep NaN as NaN rather than silently collapsing to 0, so the imputer
        # (not a comparison against NaN) decides the fill value.
        new["is_night"] = np.where(hour.isna(), np.nan, ((hour >= 22) | (hour <= 5)).astype(float))
        new["is_weekend"] = np.where(dow.isna(), np.nan, (dow >= 5).astype(float))

    def _add_amount(self, df: pd.DataFrame, new: dict) -> None:
        amt = self._numeric(df, "TransactionAmt")
        new["TransactionAmt_log"] = np.log1p(amt)
        new["TransactionAmt_decimal"] = amt - np.floor(amt)
        new["is_round_amount"] = np.where(
            amt.isna(), np.nan, (amt - np.floor(amt) == 0).astype(float)
        )
        # How unusual is this amount against the training population?
        new["TransactionAmt_zscore"] = (amt - self.global_amt_mean_) / (
            self.global_amt_std_ + 1e-9
        )

    def _add_d_normalised(self, df: pd.DataFrame, new: dict) -> None:
        if "D1" not in self.d_cols_:
            return
        d1 = self._numeric(df, "D1")
        for col in self.d_cols_:
            if col != "D1":
                new[f"{col}_norm"] = self._numeric(df, col) - d1

    def _add_card_aggregates(self, df: pd.DataFrame, new: dict) -> None:
        amt = self._numeric(df, "TransactionAmt")
        for col in CARD_COLS:
            if col not in self.card_freq_:
                continue
            keys = self._column(df, col)

            # A card never seen in training gets frequency 0, which is itself
            # a meaningful fraud signal rather than a missing value.
            new[f"{col}_freq"] = keys.map(self.card_freq_[col]).fillna(0.0).astype(float)

            mean_amt = keys.map(self.card_mean_amt_[col]).astype(float)
            new[f"{col}_mean_amt"] = mean_amt
            new[f"{col}_std_amt"] = keys.map(self.card_std_amt_[col]).astype(float)
            # Unseen cards fall back to the population mean so the ratio stays
            # interpretable instead of becoming NaN.
            new[f"{col}_amt_ratio"] = amt / (mean_amt.fillna(self.global_amt_mean_) + 1e-5)

    def _add_email(self, df: pd.DataFrame, new: dict) -> None:
        for col in EMAIL_COLS:
            domain = self._column(df, col)
            new[f"{col}_suffix"] = domain.astype("object").map(
                lambda x: str(x).split(".")[-1] if isinstance(x, str) else np.nan
            )

        p, r = self._column(df, EMAIL_COLS[0]), self._column(df, EMAIL_COLS[1])
        both_present = p.notna() & r.notna()
        new["email_domain_match"] = np.where(both_present, (p == r).astype(float), np.nan)

    def _add_domain_frequency(self, df: pd.DataFrame, new: dict) -> None:
        for col, table in self.domain_freq_.items():
            new[f"{col}_freq"] = self._column(df, col).map(table).fillna(0.0).astype(float)

    def _add_group_summaries(self, df: pd.DataFrame, new: dict) -> None:
        # Reindex to the training column set first. Without this, a request
        # carrying 3 of 339 V columns would produce a V_sum on a totally
        # different scale from the one the model was trained on.
        for prefix, cols in (("C", self.c_cols_), ("V", self.v_cols_)):
            if not cols:
                continue
            block = df.reindex(columns=cols)
            try:
                block = block.astype(float)
            except (TypeError, ValueError):
                block = block.apply(pd.to_numeric, errors="coerce")

            new[f"{prefix}_sum"] = block.sum(axis=1, min_count=1)
            new[f"{prefix}_mean"] = block.mean(axis=1)
            new[f"{prefix}_std"] = block.std(axis=1)
            new[f"{prefix}_null_count"] = block.isna().sum(axis=1).astype(float)
