"""FraudLens - Streamlit dashboard.

Tabs: Score Transaction | Batch Upload | Model Performance | Threshold | Drift
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
MODEL_DIR = Path("models")
PLOTS_DIR = Path("plots")
REPORTS_DIR = Path("reports")

st.set_page_config(page_title="FraudLens", page_icon="*", layout="wide")

RISK_COLORS = {
    "CRITICAL": "#c0392b",
    "HIGH": "#e67e22",
    "MEDIUM": "#f1c40f",
    "LOW": "#27ae60",
}


# ── data access ───────────────────────────────────────────────────────────────
# Short TTL so a restarted API is picked up without clearing the whole cache.
@st.cache_data(ttl=30)
def fetch_metrics() -> tuple[dict, str]:
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.ok:
            return response.json(), "api"
    except requests.RequestException:
        pass

    local = MODEL_DIR / "model_meta.json"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8")), "disk"
    return {}, "none"


@st.cache_data(ttl=30)
def fetch_health() -> dict:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return {}


@st.cache_data(ttl=30)
def load_threshold_sweep() -> pd.DataFrame | None:
    path = MODEL_DIR / "threshold_sweep.csv"
    return pd.read_csv(path) if path.exists() else None


meta, source = fetch_metrics()
health = fetch_health()


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("FraudLens")
    st.caption("Production ML fraud detection")

    if health.get("model_loaded"):
        st.success(f"API online - {health.get('model')}")
    elif health:
        st.error(f"API degraded: {health.get('error', 'unknown')}")
    else:
        st.warning("API unreachable")
        st.code("make api", language="bash")

    if source == "disk":
        st.info("Reading metrics from models/ on disk; the API is not responding.")

    if meta:
        st.divider()
        st.caption("Held-out test performance")
        st.metric("ROC-AUC", meta.get("roc_auc", "-"))
        st.metric("PR-AUC", meta.get("pr_auc", "-"))
        st.metric("Brier score", meta.get("brier", "-"))
        st.metric(
            "Threshold",
            meta.get("optimal_threshold", "-"),
            help=f"Selected by the {meta.get('threshold_strategy', 'n/a')} strategy",
        )

    st.divider()
    st.caption("Dataset: IEEE-CIS Fraud Detection")
    st.caption("590K transactions, ~3.5% fraud")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


if not meta:
    st.error("No model metadata available. Train a model first:")
    st.code("python scripts/make_synthetic_data.py\npython scripts/train.py", language="bash")
    st.stop()


tab_score, tab_batch, tab_perf, tab_thresh, tab_drift = st.tabs(
    ["Score Transaction", "Batch Upload", "Model Performance", "Threshold", "Drift"]
)


# ── tab 1: single transaction ─────────────────────────────────────────────────
with tab_score:
    st.header("Score a transaction")
    st.caption("Only the amount is required. Everything else sharpens the estimate.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Transaction")
        amount = st.number_input("Amount ($)", min_value=0.01, value=150.00, step=10.0)
        product = st.selectbox("Product code", ["W", "H", "C", "S", "R"])
        card4 = st.selectbox("Card network", ["visa", "mastercard", "discover", "american express"])
        card6 = st.selectbox("Card type", ["credit", "debit"])

    with col2:
        st.subheader("Card and address")
        card1 = st.number_input("Card ID (card1)", 1000, 20000, 9500)
        card2 = st.number_input("Card BIN (card2)", 100.0, 600.0, 360.0)
        addr1 = st.number_input("Billing region (addr1)", 100.0, 540.0, 325.0)
        addr2 = st.number_input("Country code (addr2)", 1.0, 200.0, 87.0)

    with col3:
        st.subheader("Identity and timing")
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com"]
        p_email = st.selectbox("Purchaser email domain", domains)
        r_email = st.selectbox("Recipient email domain", domains, index=4)
        device = st.selectbox("Device type", ["desktop", "mobile"])
        hour = st.slider("Hour of day", 0, 23, 14)
        day = st.slider("Day of week (0=Mon)", 0, 6, 2)

    explain = st.checkbox("Compute SHAP explanation", value=True)

    if st.button("Score transaction", type="primary", use_container_width=True):
        payload = {
            "TransactionAmt": amount,
            "ProductCD": product,
            "card1": card1,
            "card2": card2,
            "card4": card4,
            "card6": card6,
            "addr1": addr1,
            "addr2": addr2,
            "P_emaildomain": p_email,
            "R_emaildomain": r_email,
            "DeviceType": device,
            "TransactionDT": float(day * 86400 + hour * 3600),
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                params={"explain": str(explain).lower()},
                timeout=30,
            )
            if not response.ok:
                st.error(f"API returned {response.status_code}: {response.text}")
                st.stop()

            result = response.json()
            probability = result["fraud_probability"]
            colour = RISK_COLORS.get(result["risk_level"], "#888")

            st.markdown(
                f"""<div style='padding:18px;background:{colour}22;
                border-left:6px solid {colour};border-radius:6px;margin:12px 0'>
                <h3 style='color:{colour};margin:0'>Risk level: {result['risk_level']}</h3>
                <p style='margin:6px 0 0 0;font-size:1.05em'>
                Fraud probability <strong>{probability:.4f}</strong> &nbsp;|&nbsp;
                Decision <strong>{'BLOCK' if result['is_fraud'] else 'ALLOW'}</strong> &nbsp;|&nbsp;
                Threshold {result['threshold_used']:.4f} &nbsp;|&nbsp;
                {result['latency_ms']:.0f} ms</p></div>""",
                unsafe_allow_html=True,
            )

            if result.get("calibrated"):
                st.caption(
                    "Probabilities are isotonic-calibrated, so 0.30 means roughly a "
                    "30% chance of fraud rather than an arbitrary score."
                )

            figure, axis = plt.subplots(figsize=(9, 1.4))
            axis.barh(0, probability, color=colour, height=0.5)
            axis.barh(0, 1 - probability, left=probability, color="#ecf0f1", height=0.5)
            axis.axvline(
                result["threshold_used"], color="#2c3e50", lw=2,
                label=f"Threshold {result['threshold_used']:.3f}",
            )
            axis.set_xlim(0, 1)
            axis.set_yticks([])
            axis.set_xlabel("Fraud probability")
            axis.legend(loc="upper right", fontsize=8)
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)

            drivers = result.get("top_shap_drivers", [])
            if drivers:
                st.subheader("What drove this decision")
                frame = pd.DataFrame(drivers)
                frame["impact"] = frame["shap_value"].map(
                    lambda v: f"{'+' if v > 0 else ''}{v:.4f}"
                )
                st.dataframe(
                    frame[["feature", "feature_value", "impact", "direction"]],
                    use_container_width=True,
                    hide_index=True,
                )
            elif explain:
                st.info("No SHAP explanation was returned for this transaction.")

        except requests.exceptions.ConnectionError:
            st.error("API not reachable. Start it with:")
            st.code("make api    # or: python -m uvicorn api.main:app --reload --port 8000")


# ── tab 2: batch ──────────────────────────────────────────────────────────────
with tab_batch:
    st.header("Batch scoring")
    st.caption("Upload a CSV. A TransactionAmt column is required; all others are optional.")

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded:
        try:
            response = requests.post(
                f"{API_URL}/batch",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                timeout=300,
            )
            if not response.ok:
                st.error(f"API returned {response.status_code}: {response.text}")
            else:
                body = response.json()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Transactions", f"{body['total_transactions']:,}")
                col2.metric("Flagged", f"{body['fraud_count']:,}")
                col3.metric("Flag rate", f"{body['fraud_rate'] * 100:.2f}%")
                col4.metric("Latency", f"{body['latency_ms']:.0f} ms")

                results = pd.DataFrame(body["results"])
                st.dataframe(results, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results CSV",
                    results.to_csv(index=False).encode(),
                    "fraudlens_results.csv",
                    "text/csv",
                )

                counts = results["risk_level"].value_counts()
                figure, axis = plt.subplots(figsize=(7, 3))
                axis.bar(
                    counts.index,
                    counts.to_numpy(),
                    color=[RISK_COLORS.get(k, "#888") for k in counts.index],
                )
                axis.set_ylabel("Transactions")
                axis.set_title("Risk distribution")
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)

        except requests.exceptions.ConnectionError:
            st.error("API not reachable.")


# ── tab 3: performance ────────────────────────────────────────────────────────
with tab_perf:
    st.header("Model performance")
    st.caption(
        "All headline figures come from the sealed test slice, which was never "
        "used for early stopping, model selection or threshold tuning."
    )

    test_metrics = meta.get("test", {})
    if test_metrics:
        cols = st.columns(5)
        for col, key, label in zip(
            cols,
            ["roc_auc", "pr_auc", "precision", "recall", "flag_rate"],
            ["ROC-AUC", "PR-AUC", "Precision", "Recall", "Flag rate"],
            strict=True,
        ):
            value = test_metrics.get(key)
            col.metric(label, f"{value:.4f}" if isinstance(value, (int, float)) else "-")

    comparison = meta.get("all_model_results", {})
    if comparison:
        st.subheader("Candidate models (validation slice)")
        frame = pd.DataFrame(comparison).T
        preferred = [c for c in ["roc_auc", "pr_auc", "brier", "f1", "precision", "recall",
                                 "train_time_s"] if c in frame.columns]
        st.dataframe(frame[preferred], use_container_width=True)
        st.caption(
            "Selection is on PR-AUC, not ROC-AUC: at a 3.5% positive rate ROC-AUC is "
            "dominated by the majority class and barely separates candidates."
        )

    top_features = meta.get("top_features", [])
    if top_features:
        st.subheader("Top fraud signals by mean |SHAP|")
        st.dataframe(
            pd.DataFrame({"rank": range(1, len(top_features) + 1), "feature": top_features}),
            use_container_width=True,
            hide_index=True,
        )

    available = {
        name: PLOTS_DIR / filename
        for name, filename in {
            "SHAP summary (beeswarm)": "08_shap_summary.png",
            "SHAP importance (bar)": "09_shap_bar.png",
            "ROC and PR curves": "05_roc_pr_curves.png",
            "Confusion matrix": "06_confusion_matrix.png",
            "Threshold tuning": "07_threshold_tuning.png",
            "Class distribution": "01_class_distribution.png",
            "Missingness": "02_missingness.png",
            "Temporal fraud rate": "03_temporal_fraud.png",
        }.items()
        if (PLOTS_DIR / filename).exists()
    }
    if available:
        st.subheader("Diagnostic plots")
        selected = st.selectbox("Plot", list(available))
        st.image(str(available[selected]), use_container_width=True)
    else:
        st.info("No plots yet. Run `python scripts/train.py` or the notebook to generate them.")


# ── tab 4: threshold explorer ─────────────────────────────────────────────────
with tab_thresh:
    st.header("Threshold and business cost")
    st.caption(
        "0.5 is almost never the right cut-off for fraud. This is the full sweep "
        "computed on the validation slice during training."
    )

    sweep = load_threshold_sweep()
    if sweep is None:
        st.info("No threshold_sweep.csv found. Retrain to generate it.")
    else:
        chosen = meta.get("optimal_threshold")

        figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4))
        left.plot(sweep["threshold"], sweep["precision"], label="Precision", color="#3498db")
        left.plot(sweep["threshold"], sweep["recall"], label="Recall", color="#e74c3c")
        left.plot(sweep["threshold"], sweep["f1"], label="F1", color="#9b59b6", lw=2)
        if chosen:
            left.axvline(chosen, color="#2c3e50", ls="--", label=f"Chosen {chosen:.3f}")
        left.set_xlabel("Threshold")
        left.set_ylabel("Score")
        left.set_title("Precision / recall trade-off")
        left.legend(fontsize=8)

        right.plot(sweep["threshold"], sweep["expected_cost"], color="#16a085", lw=2)
        best = sweep.loc[sweep["expected_cost"].idxmin()]
        right.axvline(best["threshold"], color="#c0392b", ls="--",
                      label=f"Min cost {best['threshold']:.3f}")
        right.set_xlabel("Threshold")
        right.set_ylabel("Expected cost")
        right.set_title("Expected business cost")
        right.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

        st.caption(
            "A missed fraud is charged at the disputed transaction amount; a false "
            "positive is charged at a flat rate covering manual review plus the "
            "churn risk of blocking a real customer. Both are set in fraudlens/config.py."
        )

        explore = st.slider("Inspect a threshold", 0.0, 1.0, float(chosen or 0.5), 0.005)
        nearest = sweep.iloc[(sweep["threshold"] - explore).abs().argmin()]
        cols = st.columns(5)
        cols[0].metric("Precision", f"{nearest['precision']:.3f}")
        cols[1].metric("Recall", f"{nearest['recall']:.3f}")
        cols[2].metric("Flag rate", f"{nearest['flag_rate'] * 100:.2f}%")
        cols[3].metric("False positives", f"{int(nearest['fp']):,}")
        cols[4].metric("Missed frauds", f"{int(nearest['fn']):,}")


# ── tab 5: drift ──────────────────────────────────────────────────────────────
with tab_drift:
    st.header("Data drift")

    st.subheader("Live traffic (PSI)")
    st.caption(
        "Population Stability Index of transactions scored since the API started, "
        "measured against the frozen training reference."
    )

    try:
        response = requests.get(f"{API_URL}/drift", timeout=10)
        if response.status_code == 503:
            st.info(response.json().get("detail", "Drift monitoring unavailable."))
        elif response.ok:
            report = response.json()
            if report["status"] == "insufficient_data":
                st.info(
                    f"Observed {report['rows_observed']} of the "
                    f"{report['rows_required']} rows needed. Score more transactions "
                    "to populate the window."
                )
            else:
                cols = st.columns(4)
                cols[0].metric("Rows in window", f"{report['rows_observed']:,}")
                cols[1].metric("Features monitored", report["features_monitored"])
                cols[2].metric("Drifted", report["features_drifted"])
                cols[3].metric("Share drifted", f"{report['share_drifted'] * 100:.1f}%")

                if report["alert"]:
                    reason = report.get("alert_reason") or "input distribution moved"
                    st.error(f"Drift alert: {reason}")
                elif report["features_drifted"]:
                    st.warning(
                        f"{report['features_drifted']} feature(s) drifting, "
                        "below the alert threshold."
                    )
                else:
                    st.success("Input distribution is stable.")

                frame = pd.DataFrame(
                    [{"feature": k, **v} for k, v in report["features"].items()]
                )
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.caption("PSI below 0.10 is stable, 0.10-0.25 minor, above 0.25 major.")
        else:
            st.warning(f"Drift endpoint returned {response.status_code}.")
    except requests.exceptions.ConnectionError:
        st.warning("API not reachable, so live drift cannot be computed.")

    st.divider()
    st.subheader("Offline Evidently report")
    report_path = REPORTS_DIR / "drift_report.html"
    if report_path.exists():
        components.html(report_path.read_text(encoding="utf-8"), height=700, scrolling=True)
    else:
        st.info("No Evidently report found. Run section 10 of the notebook to generate one.")
