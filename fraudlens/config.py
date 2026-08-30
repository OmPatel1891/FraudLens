"""Central paths and constants for FraudLens."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# Columns never used as model inputs.
EXCLUDE_COLS = [ID_COL, TIME_COL, TARGET]

# Grouping keys for the fit-time aggregate lookup tables.
CARD_COLS = ["card1", "card2", "card3", "card5"]
EMAIL_COLS = ["P_emaildomain", "R_emaildomain"]
DEVICE_COLS = ["DeviceType", "DeviceInfo"]

# Sentinels so categorical handling is identical in training and serving.
MISSING_CATEGORY = "__missing__"
UNKNOWN_CATEGORY = "__unknown__"

# Fraction of rows a column may be missing before it is dropped.
MAX_MISSING_FRACTION = 0.8

# Temporal split boundaries as quantiles of TransactionDT.
# Three-way so that early stopping / threshold tuning and final reporting
# never touch the same rows.
TRAIN_END_Q = 0.70
VAL_END_Q = 0.85

# Default business cost matrix used for threshold selection.
# A missed fraud is charged at the disputed transaction amount when amounts are
# supplied, otherwise at this flat rate. A false positive covers manual review
# plus an allowance for the churn risk of blocking a real customer - which is
# why it is far above a pure review cost. Lowering it makes the model flag
# aggressively; a processor that blocks 10% of traffic will not stay in business.
COST_FALSE_NEGATIVE = 100.0
COST_FALSE_POSITIVE = 25.0

# Population Stability Index bands.
PSI_MINOR = 0.10
PSI_MAJOR = 0.25

# Columns the drift monitor always watches, on top of the model's top features
# by SHAP. Ranking by SHAP alone tends to select anonymous V columns and leave
# amount and time unmonitored, yet those are exactly where a broken upstream
# feed shows up first (currency switched to cents, a clock skew, a stalled
# batch). Names absent from a given run are skipped by `drift.build_reference`.
MONITOR_ALWAYS = [
    "TransactionAmt",
    "TransactionAmt_log",
    "TransactionAmt_zscore",
    "hour",
    "is_night",
]

# Single source of truth for MLflow, so the training run, `make mlflow` and the
# Compose service all read and write the same store.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
)
MLFLOW_EXPERIMENT = "FraudLens"


MLRUNS_DIR = ROOT / "mlruns"


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (DATA_DIR, MODEL_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR, MLRUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
