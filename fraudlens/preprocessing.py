"""Encoding and imputation with an explicit, frozen column contract.

The numeric/categorical split is decided once during `fit` and stored. It must
never be re-derived from dtypes at transform time: label-encoded categoricals
look numeric, and any column the caller omits arrives as a float NaN, so the
split would come out different for a sparse request than for a training row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from .config import EXCLUDE_COLS, MAX_MISSING_FRACTION, MISSING_CATEGORY, UNKNOWN_CATEGORY


class Preprocessor:
    """Selects columns, label-encodes categoricals, then imputes numerics."""

    def __init__(self, max_missing_fraction: float = MAX_MISSING_FRACTION) -> None:
        self.max_missing_fraction = max_missing_fraction
        self.fitted_ = False

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> Preprocessor:
        candidates = [c for c in df.columns if c not in EXCLUDE_COLS]

        missing = df[candidates].isna().mean()
        self.dropped_cols_ = missing[missing > self.max_missing_fraction].index.tolist()
        kept = [c for c in candidates if c not in self.dropped_cols_]

        frame = df[kept]
        self.cat_cols_ = frame.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        self.num_cols_ = [c for c in kept if c not in self.cat_cols_]
        # Frozen order. Every transform reindexes to exactly this.
        self.feature_cols_ = self.num_cols_ + self.cat_cols_

        self.label_encoders_ = {}
        for col in self.cat_cols_:
            values = self._as_category_strings(frame[col])
            classes = sorted(set(values.unique()) | {MISSING_CATEGORY, UNKNOWN_CATEGORY})
            le = LabelEncoder().fit(np.array(classes, dtype=object))
            self.label_encoders_[col] = le

        self.num_imputer_ = SimpleImputer(strategy="median")
        if self.num_cols_:
            self.num_imputer_.fit(self._numeric_block(frame))
            # Cached so transform can fill in place rather than paying
            # scikit-learn's per-call validation on every request. The column
            # contract that validation would check is already guaranteed by
            # the reindex in _numeric_block.
            statistics = np.asarray(self.num_imputer_.statistics_, dtype=float)
            self.num_fill_ = np.where(np.isfinite(statistics), statistics, 0.0)
        else:
            self.num_fill_ = np.empty(0, dtype=float)

        self.fitted_ = True
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ── transform ────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Preprocessor.fit must be called before transform")

        # Each block is reindexed straight out of the input and the two are
        # joined once, in feature_cols_ order. Missing columns become NaN and
        # unexpected extras are dropped, which is what lets a five-field API
        # request line up with a full training row.
        blocks = []

        if self.num_cols_:
            values = self._numeric_block(df).to_numpy(dtype=float, copy=True)
            missing = np.isnan(values)
            if missing.any():
                values[missing] = np.take(self.num_fill_, np.nonzero(missing)[1])
            blocks.append(pd.DataFrame(values, index=df.index, columns=self.num_cols_))

        if self.cat_cols_:
            blocks.append(self._categorical_block(df))

        return blocks[0] if len(blocks) == 1 else pd.concat(blocks, axis=1, copy=False)

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _as_category_strings(series: pd.Series) -> pd.Series:
        """Normalise a categorical column to strings with an explicit missing token."""
        out = series.astype("object")
        return out.where(out.notna(), MISSING_CATEGORY).astype(str)

    def _numeric_block(self, df: pd.DataFrame) -> pd.DataFrame:
        block = df.reindex(columns=self.num_cols_)
        try:
            return block.astype(float)
        except (TypeError, ValueError):
            return block.apply(pd.to_numeric, errors="coerce")

    def _categorical_block(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded = {}
        for col in self.cat_cols_:
            le = self.label_encoders_[col]
            known = set(le.classes_)
            source = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
            values = self._as_category_strings(source)
            # Categories absent from training map to a dedicated token rather
            # than to classes_[0], which would silently impersonate a real value.
            values = values.where(values.isin(known), UNKNOWN_CATEGORY)
            encoded[col] = le.transform(values.to_numpy(dtype=object)).astype(float)

        return pd.DataFrame(encoded, index=df.index, columns=self.cat_cols_)
