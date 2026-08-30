"""Generate IEEE-CIS-shaped synthetic data so the pipeline is runnable offline.

This mirrors the real dataset's structure: the same column families, a similar
fraud rate, heavy missingness in the V/D/id blocks, a temporal axis, and - on
purpose - the hyphenated `id-01` naming that Kaggle uses in test_identity.csv.
Reproducing that quirk keeps the schema guard in fraudlens.data honest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudlens.config import DATA_DIR

N_V = 339
N_C = 14
N_D = 15
N_M = 9
N_ID = 38

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com", "aol.com"]
PRODUCTS = ["W", "H", "C", "S", "R"]
CARD_NETWORKS = ["visa", "mastercard", "discover", "american express"]
CARD_TYPES = ["credit", "debit"]
DEVICE_TYPES = ["desktop", "mobile"]
DEVICE_INFO = ["Windows", "iOS Device", "MacOS", "Trident/7.0", "SAMSUNG SM-G892A", "rv:11.0"]


def _mask(rng, values, missing_rate):
    """Blank out a fraction of values to imitate the real sparsity."""
    values = pd.Series(values, dtype="object")
    drop = rng.random(len(values)) < missing_rate
    values[drop] = np.nan
    return values


def _solve_intercept(latent, target_rate, lo=-14.0, hi=6.0, iters=60):
    """Bisect for the intercept that puts the mean event rate on target."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(latent + mid)))))
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def build_transactions(rng, n, start_id, start_dt, with_label, fraud_rate=0.035):
    """Construct the transaction table with a learnable fraud signal."""
    # Latent risk drives both the features and the label, so a model has
    # something real to find rather than pure noise.
    risk = rng.beta(1.6, 9.0, n)
    risk_z = (risk - risk.mean()) / (risk.std() + 1e-9)

    dt = np.sort(start_dt + rng.integers(0, 182 * 86400, n))
    hour = (dt // 3600) % 24

    # Fraud skews nocturnal, higher-value, and toward anonymous email.
    night = np.isin(hour, [0, 1, 2, 3, 4, 5, 22, 23]).astype(float)
    amount = np.round(np.exp(rng.normal(3.9 + risk * 1.1, 1.0, n)), 2)
    amount = np.clip(amount, 0.5, 32000.0)

    p_email = rng.choice(EMAIL_DOMAINS, n, p=[0.42, 0.18, 0.13, 0.09, 0.12, 0.06])
    anon = (p_email == "anonymous.com").astype(float)

    # A logistic link keeps the signal strong enough to be learnable while the
    # solved intercept independently pins the base rate near the real 3.5%.
    latent = 2.6 * risk_z + 0.9 * night + 1.1 * anon
    fraud_prob = 1.0 / (1.0 + np.exp(-(latent + _solve_intercept(latent, fraud_rate))))

    df = pd.DataFrame({"TransactionID": np.arange(start_id, start_id + n)})
    if with_label:
        df["isFraud"] = (rng.random(n) < fraud_prob).astype(int)
    df["TransactionDT"] = dt
    df["TransactionAmt"] = amount
    df["ProductCD"] = rng.choice(PRODUCTS, n, p=[0.74, 0.09, 0.07, 0.06, 0.04])

    df["card1"] = rng.integers(1000, 18400, n)
    df["card2"] = _mask(rng, rng.integers(100, 600, n).astype(float), 0.015)
    df["card3"] = _mask(rng, rng.choice([150.0, 185.0], n, p=[0.88, 0.12]), 0.003)
    df["card4"] = _mask(rng, rng.choice(CARD_NETWORKS, n, p=[0.65, 0.31, 0.02, 0.02]), 0.003)
    df["card5"] = _mask(rng, rng.integers(100, 240, n).astype(float), 0.007)
    df["card6"] = _mask(rng, rng.choice(CARD_TYPES, n, p=[0.75, 0.25]), 0.003)

    df["addr1"] = _mask(rng, rng.integers(100, 540, n).astype(float), 0.11)
    df["addr2"] = _mask(rng, rng.choice([87.0, 60.0, 96.0], n, p=[0.96, 0.02, 0.02]), 0.11)
    df["dist1"] = _mask(rng, rng.exponential(60, n).round(0), 0.60)
    df["dist2"] = _mask(rng, rng.exponential(220, n).round(0), 0.93)

    df["P_emaildomain"] = _mask(rng, p_email, 0.16)
    df["R_emaildomain"] = _mask(rng, rng.choice(EMAIL_DOMAINS, n), 0.77)

    # C columns: counting features, mildly elevated for risky transactions.
    for i in range(1, N_C + 1):
        df[f"C{i}"] = np.round(rng.poisson(1.4 + risk * 2.6, n) * 1.0, 1)

    # D columns: day-deltas with the real dataset's rising missingness.
    for i in range(1, N_D + 1):
        base = rng.integers(0, 640, n).astype(float)
        df[f"D{i}"] = _mask(rng, base, min(0.02 + i * 0.055, 0.90))

    for i in range(1, N_M + 1):
        df[f"M{i}"] = _mask(rng, rng.choice(["T", "F"], n, p=[0.7, 0.3]), 0.42)

    # V columns: correlated blocks, a few of which carry the signal.
    v_block = rng.normal(0, 1, (n, N_V))
    for j in range(40):
        v_block[:, j] += risk * 2.4
    v_frame = pd.DataFrame(v_block.round(3), columns=[f"V{i}" for i in range(1, N_V + 1)])
    for j, col in enumerate(v_frame.columns):
        rate = 0.05 if j < 120 else (0.48 if j < 260 else 0.83)
        v_frame[col] = _mask(rng, v_frame[col].to_numpy(), rate)

    return pd.concat([df, v_frame], axis=1)


def build_identity(rng, transaction_ids, coverage, hyphenate):
    """Identity table covering only a subset of transactions, as in the real data."""
    chosen = np.sort(
        rng.choice(transaction_ids, size=int(len(transaction_ids) * coverage), replace=False)
    )
    n = len(chosen)

    out = pd.DataFrame({"TransactionID": chosen})
    for i in range(1, N_ID + 1):
        label = f"id_{i:02d}"
        if i <= 11:
            values = rng.normal(0, 40, n).round(1)
        elif i in (12, 15, 16, 27, 28, 29, 35, 36, 37, 38):
            values = rng.choice(["Found", "NotFound", "New"], n)
        else:
            values = rng.integers(0, 700, n).astype(float)
        out[label] = _mask(rng, values, 0.30 if i <= 11 else 0.55)

    out["DeviceType"] = _mask(rng, rng.choice(DEVICE_TYPES, n, p=[0.55, 0.45]), 0.05)
    out["DeviceInfo"] = _mask(rng, rng.choice(DEVICE_INFO, n), 0.20)

    if hyphenate:
        # Faithfully reproduce Kaggle's test_identity.csv naming.
        out = out.rename(columns=lambda c: c.replace("id_", "id-") if c.startswith("id_") else c)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=int, default=30000)
    parser.add_argument("--test-rows", type=int, default=10000)
    parser.add_argument("--fraud-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic IEEE-CIS data into {args.out} ...")

    train_txn = build_transactions(
        rng, args.train_rows, 2987000, 86400, with_label=True, fraud_rate=args.fraud_rate
    )
    train_id = build_identity(rng, train_txn["TransactionID"].to_numpy(), 0.24, hyphenate=False)

    last_dt = int(train_txn["TransactionDT"].max())
    test_txn = build_transactions(
        rng,
        args.test_rows,
        2987000 + args.train_rows,
        last_dt + 86400 * 30,
        with_label=False,
        fraud_rate=args.fraud_rate,
    )
    test_id = build_identity(rng, test_txn["TransactionID"].to_numpy(), 0.24, hyphenate=True)

    for name, frame in [
        ("train_transaction", train_txn),
        ("train_identity", train_id),
        ("test_transaction", test_txn),
        ("test_identity", test_id),
    ]:
        path = args.out / f"{name}.csv"
        frame.to_csv(path, index=False)
        size_mb = path.stat().st_size / 1e6
        print(f"  {name}.csv  {frame.shape[0]:,} x {frame.shape[1]}  ({size_mb:.1f} MB)")

    print(f"\nFraud rate: {train_txn['isFraud'].mean():.3%}")
    print("Note: test_identity.csv uses hyphenated `id-01` names, exactly like Kaggle's.")


if __name__ == "__main__":
    main()
