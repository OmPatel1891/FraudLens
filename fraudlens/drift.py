"""Population Stability Index drift monitoring.

Evidently produces the rich offline report, but it is too heavy to call on a
live request path and its report object is not serialisable into an API
response. This module computes PSI from a small frozen reference summary, so
the running service can answer "has my input distribution moved?" on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PSI_MAJOR, PSI_MINOR


def build_reference(df: pd.DataFrame, columns: list, n_bins: int = 10) -> dict:
    """Freeze quantile bin edges and expected proportions from training data."""
    reference = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty or series.nunique() < 2:
            continue

        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(series, quantiles))
        if len(edges) < 3:
            continue
        # Open the outer edges so unseen extremes still land in a bin.
        edges[0], edges[-1] = -np.inf, np.inf

        counts, _ = np.histogram(series, bins=edges)
        proportions = counts / max(counts.sum(), 1)

        reference[col] = {
            "edges": edges.tolist(),
            "proportions": proportions.tolist(),
            "mean": float(series.mean()),
            "std": float(series.std()),
        }
    return reference


def _psi_for_column(values: np.ndarray, spec: dict, epsilon: float = 1e-6) -> float:
    edges = np.array(spec["edges"], dtype=float)
    expected = np.array(spec["proportions"], dtype=float)

    counts, _ = np.histogram(values, bins=edges)
    actual = counts / max(counts.sum(), 1)

    # Epsilon guards the log against empty bins on either side.
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def classify(psi: float) -> str:
    if psi >= PSI_MAJOR:
        return "major"
    if psi >= PSI_MINOR:
        return "minor"
    return "stable"


def compute_drift(current: pd.DataFrame, reference: dict, min_rows: int = 50) -> dict:
    """Compare a window of live traffic against the frozen reference."""
    if len(current) < min_rows:
        return {
            "status": "insufficient_data",
            "rows_observed": len(current),
            "rows_required": min_rows,
            "features": {},
        }

    per_feature = {}
    for col, spec in reference.items():
        if col not in current.columns:
            continue
        values = pd.to_numeric(current[col], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue
        psi = _psi_for_column(values, spec)
        per_feature[col] = {"psi": round(psi, 4), "severity": classify(psi)}

    drifted = [c for c, v in per_feature.items() if v["severity"] != "stable"]
    major = [c for c, v in per_feature.items() if v["severity"] == "major"]
    share_drifted = len(drifted) / max(len(per_feature), 1)

    # Breadth alone is not enough to page someone. A feed that starts reporting
    # amounts in cents moves one or two columns out of twenty and would sit
    # under any share-based threshold, so a single major shift alerts on its own.
    alert = bool(major) or share_drifted > 0.2

    return {
        "status": "ok",
        "rows_observed": len(current),
        "features_monitored": len(per_feature),
        "features_drifted": len(drifted),
        "share_drifted": round(share_drifted, 4),
        "major_drift": major,
        "alert": alert,
        "alert_reason": (
            f"major drift in {', '.join(major)}"
            if major
            else f"{share_drifted:.0%} of monitored features drifted"
            if share_drifted > 0.2
            else None
        ),
        "features": dict(
            sorted(per_feature.items(), key=lambda kv: kv[1]["psi"], reverse=True)
        ),
    }


def save_reference(reference: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, indent=2), encoding="utf-8")


def load_reference(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
