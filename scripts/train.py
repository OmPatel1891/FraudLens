"""Train FraudLens end to end and export every artifact the API needs.

Ordering is load-bearing: the temporal split happens *first*, and only then are
the feature lookup tables, encoders and imputers fitted - on the training slice
alone. Fitting any of them beforehand leaks validation-period statistics
backwards into training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudlens import drift
from fraudlens.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MONITOR_ALWAYS,
    PLOTS_DIR,
    TARGET,
    TIME_COL,
    TRAIN_END_Q,
    VAL_END_Q,
    ensure_dirs,
)
from fraudlens.data import load_split
from fraudlens.features import FeatureEngineer
from fraudlens.modeling import (
    CalibratedModel,
    ScaledModel,
    evaluate,
    pick_threshold,
    precision_at_recall,
    sweep_thresholds,
)
from fraudlens.preprocessing import Preprocessor

warnings.filterwarnings("ignore")


def temporal_split(df: pd.DataFrame):
    """Three-way chronological split.

    A third slice is what lets early stopping and threshold tuning use `val`
    while `test` stays untouched for reporting. With only two slices every
    headline number would be measured on data the model was tuned against.
    """
    t = df[TIME_COL]
    train_end = t.quantile(TRAIN_END_Q)
    val_end = t.quantile(VAL_END_Q)

    train_mask = t <= train_end
    val_mask = (t > train_end) & (t <= val_end)
    test_mask = t > val_end
    return train_mask, val_mask, test_mask


def train_models(X_train, y_train, X_val, y_val, seed=42, fast=False):
    """Fit the baseline plus both gradient-boosted models."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pos = max(int((y_train == 1).sum()), 1)
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos

    models, timings = {}, {}

    print("\n[1/3] Logistic Regression (baseline)")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    t0 = time.time()
    # lbfgs converges far faster than saga on this width; saga on 400 columns
    # and 400k rows took hours without improving the baseline.
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs", C=0.1, random_state=seed, n_jobs=-1
    )
    lr.fit(Xs, y_train)
    timings["Logistic Regression"] = time.time() - t0
    models["Logistic Regression"] = ScaledModel(lr, scaler)

    print("[2/3] XGBoost")
    import xgboost as xgb

    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200 if fast else 800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    timings["XGBoost"] = time.time() - t0
    models["XGBoost"] = xgb_model

    print("[3/3] LightGBM")
    import lightgbm as lgb

    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300 if fast else 1500,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    timings["LightGBM"] = time.time() - t0
    models["LightGBM"] = lgb_model

    return models, timings, scale_pos_weight


def log_to_mlflow(results, params_by_model, best_name):
    """Record every run. Failure here must not lose a trained model."""
    try:
        import mlflow
    except ImportError:
        print("  mlflow not installed; skipping experiment logging")
        return {}

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except Exception as exc:
        print(f"  mlflow unavailable ({exc}); skipping experiment logging")
        return {}

    run_ids = {}
    for name, metrics in results.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(
                {k: v for k, v in params_by_model.get(name, {}).items()
                 if isinstance(v, (int, float, str, bool))}
            )
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.set_tag("selected", str(name == best_name))
            run_ids[name] = run.info.run_id
    print(f"  logged {len(run_ids)} runs to {MLFLOW_TRACKING_URI}")
    return run_ids


def compute_shap(model, X_sample, plots_dir: Path):
    """Mean |SHAP| ranking, plus summary plots when shap is available."""
    try:
        import shap
    except ImportError:
        print("  shap not installed; skipping explainability")
        return [], None

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_sample)
    if isinstance(values, list):
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, 1]

    importance = (
        pd.Series(np.abs(values).mean(axis=0), index=X_sample.columns)
        .sort_values(ascending=False)
    )

    # The ranking feeds the drift monitor's watch list, so it has to survive a
    # serving-sized install with no plotting stack. Only the pictures are
    # optional.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; skipping SHAP plots")
        return importance.index.tolist(), values

    for kind, fname in [("dot", "08_shap_summary.png"), ("bar", "09_shap_bar.png")]:
        plt.figure(figsize=(10, 7))
        shap.summary_plot(values, X_sample, max_display=20, show=False, plot_type=kind)
        plt.tight_layout()
        plt.savefig(plots_dir / fname, dpi=140, bbox_inches="tight")
        plt.close()

    return importance.index.tolist(), values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="fewer trees, for smoke tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shap-sample", type=int, default=2000)
    parser.add_argument(
        "--threshold-strategy",
        default="min_cost",
        choices=["min_cost", "max_f1", "min_precision"],
    )
    parser.add_argument("--min-precision", type=float, default=0.9)
    args = parser.parse_args()

    ensure_dirs()

    print("=" * 68)
    print("FraudLens training")
    print("=" * 68)

    raw = load_split("train")
    print(f"Loaded {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    print(f"Fraud rate: {raw[TARGET].mean():.3%}")

    train_mask, val_mask, test_mask = temporal_split(raw)
    print(
        f"\nTemporal split -> train {train_mask.sum():,} | "
        f"val {val_mask.sum():,} | test {test_mask.sum():,}"
    )

    # Everything below is fitted on the training slice only.
    engineer = FeatureEngineer().fit(raw.loc[train_mask])
    engineered = engineer.transform(raw)
    print(f"Engineered to {engineered.shape[1]} columns")

    preprocessor = Preprocessor().fit(engineered.loc[train_mask])
    X = preprocessor.transform(engineered)
    y = raw[TARGET].astype(int)
    print(
        f"Model matrix: {X.shape[1]} features "
        f"({len(preprocessor.num_cols_)} numeric, {len(preprocessor.cat_cols_)} categorical); "
        f"dropped {len(preprocessor.dropped_cols_)} above the missingness cap"
    )

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    models, timings, spw = train_models(X_train, y_train, X_val, y_val, args.seed, args.fast)

    # Model selection uses val. test stays sealed until the very end.
    print("\nValidation results")
    val_results = {}
    for name, model in models.items():
        proba = model.predict_proba(X_val)[:, 1]
        metrics = evaluate(y_val, proba)
        metrics["train_time_s"] = round(timings[name], 1)
        val_results[name] = metrics
        print(
            f"  {name:<22} ROC-AUC={metrics['roc_auc']:.4f}  "
            f"PR-AUC={metrics['pr_auc']:.4f}  Brier={metrics['brier']:.4f}"
        )

    best_name = max(val_results, key=lambda k: val_results[k]["pr_auc"])
    best_raw = models[best_name]
    # PR-AUC, not ROC-AUC, decides: at a 3.5% positive rate ROC-AUC is
    # dominated by the majority class and barely separates candidates.
    print(f"\nSelected on PR-AUC: {best_name}")

    print("\nCalibrating on the validation slice ...")
    calibrated = CalibratedModel.fit(best_raw, X_val, y_val)
    val_proba = calibrated.predict_proba(X_val)[:, 1]
    print(
        f"  Brier {val_results[best_name]['brier']:.4f} -> "
        f"{evaluate(y_val, val_proba)['brier']:.4f}"
    )

    print("\nTuning the decision threshold on validation ...")
    amounts_val = raw.loc[val_mask, "TransactionAmt"].to_numpy()
    sweep = sweep_thresholds(y_val, val_proba, amounts=amounts_val)
    choice = pick_threshold(
        sweep, strategy=args.threshold_strategy, min_precision=args.min_precision
    )
    f1_choice = pick_threshold(sweep, strategy="max_f1")
    print(
        f"  {args.threshold_strategy}: t={choice.threshold:.3f}  "
        f"precision={choice.precision:.3f}  recall={choice.recall:.3f}  "
        f"flag_rate={choice.flag_rate:.3%}"
    )
    print(
        f"  max_f1 (for reference): t={f1_choice.threshold:.3f}  "
        f"F1={f1_choice.f1:.3f}  flag_rate={f1_choice.flag_rate:.3%}"
    )

    print("\nFinal evaluation on the sealed test slice")
    test_proba = calibrated.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_proba, threshold=choice.threshold)
    test_metrics["precision_at_80_recall"] = precision_at_recall(y_test, test_proba, 0.8)
    for k, v in test_metrics.items():
        print(f"  {k:<24} {v:.4f}")

    sample_n = min(args.shap_sample, len(X_val))
    print(f"\nComputing SHAP on {sample_n:,} validation rows ...")
    X_shap = X_val.sample(sample_n, random_state=args.seed)
    top_features, _ = compute_shap(best_raw, X_shap, PLOTS_DIR)

    params_by_model = {
        "Logistic Regression": {"solver": "lbfgs", "C": 0.1, "class_weight": "balanced"},
        "XGBoost": {"max_depth": 6, "learning_rate": 0.05, "scale_pos_weight": spw},
        "LightGBM": {"num_leaves": 63, "learning_rate": 0.05, "scale_pos_weight": spw},
    }
    print("\nLogging to MLflow ...")
    run_ids = log_to_mlflow(val_results, params_by_model, best_name)

    print("\nExporting artifacts ...")
    joblib.dump(engineer, MODEL_DIR / "feature_engineer.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(calibrated, MODEL_DIR / "model.joblib")
    joblib.dump(best_raw, MODEL_DIR / "base_model.joblib")

    ranked = top_features[:20] if top_features else preprocessor.num_cols_[:20]
    monitored = list(dict.fromkeys([*MONITOR_ALWAYS, *ranked]))
    reference = drift.build_reference(X_train, monitored)
    drift.save_reference(reference, MODEL_DIR / "drift_reference.json")
    # Only columns that survived reference building can ever report PSI, so the
    # API's capture list must match them exactly or it buffers dead weight.
    monitored = list(reference)

    meta = {
        "model_name": best_name,
        "calibrated": True,
        "threshold_strategy": args.threshold_strategy,
        "optimal_threshold": round(choice.threshold, 4),
        "f1_optimal_threshold": round(f1_choice.threshold, 4),
        "validation": {k: round(v, 4) for k, v in val_results[best_name].items()},
        "test": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "roc_auc": round(test_metrics["roc_auc"], 4),
        "pr_auc": round(test_metrics["pr_auc"], 4),
        "f1": round(test_metrics["f1"], 4),
        "brier": round(test_metrics["brier"], 4),
        "expected_cost_val": round(choice.expected_cost, 2),
        "flag_rate": round(choice.flag_rate, 4),
        "train_size": int(train_mask.sum()),
        "val_size": int(val_mask.sum()),
        "test_size": int(test_mask.sum()),
        "feature_count": int(X.shape[1]),
        "fraud_rate_train": round(float(y_train.mean()), 4),
        "top_features": top_features[:10],
        "monitored_features": monitored,
        "all_model_results": {
            name: {metric: round(value, 4) for metric, value in scores.items()}
            for name, scores in val_results.items()
        },
        "mlflow_run_id": run_ids.get(best_name, ""),
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sweep.to_csv(MODEL_DIR / "threshold_sweep.csv", index=False)

    for f in sorted(MODEL_DIR.iterdir()):
        print(f"  {f.name:<28} {f.stat().st_size/1024:>8.0f} KB")

    print(f"\nDone. {best_name} | test PR-AUC={test_metrics['pr_auc']:.4f} | "
          f"threshold={choice.threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
