"""
train_model.py

Trains a baseline XGBoost classifier on the feature table produced by
process_trades.py. Target: whether price is higher 30 buckets (30s) ahead
than now (binary up/down). This is a placeholder trading signal purely to
give the VEC generator a real tree ensemble to work with -- swap in your
own target/features later without changing the VEC pipeline.

Split is chronological (not random) to avoid lookahead leakage:
    train       -> first 60% of time range
    calibration -> next 20% (used later for conformal calibration + VEC)
    test        -> final 20% (held out, untouched until final evaluation)

Usage:
    python train_model.py --features features.parquet --out model.json
    python train_model.py --features features.parquet --out model_big.json \
        --n_estimators 300 --max_depth 6 --no_calib_save
"""

import argparse

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURE_COLS = ["realized_vol", "trade_intensity", "avg_trade_size"]
HORIZON = 30  # buckets ahead (30 buckets x 1s = 30s lookahead for the label)


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    future_close = df["close"].shift(-HORIZON)
    df["label"] = (future_close > df["close"]).astype(int)
    df = df.iloc[:-HORIZON]  # drop rows with no future price to label
    return df


def chronological_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)
    train = df.iloc[:train_end]
    calib = df.iloc[train_end:calib_end]
    test = df.iloc[calib_end:]
    return train, calib, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features.parquet")
    parser.add_argument("--out", default="model.json", help="Output model file")
    parser.add_argument(
        "--calib_out",
        default="calibration.parquet",
        help="Output path for the calibration split (needed later by the VEC generator)",
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument(
        "--no_calib_save",
        action="store_true",
        help="Do not rewrite calibration.parquet (the split does not depend on the model)",
    )
    args = parser.parse_args()

    print(f"Loading {args.features} ...")
    df = pd.read_parquet(args.features)
    print(f"Loaded {len(df):,} rows")

    print("Building labels (30s-ahead up/down) ...")
    df = build_labels(df)

    print("Splitting chronologically into train / calibration / test ...")
    train, calib, test = chronological_split(df)
    print(f"  train: {len(train):,}  calib: {len(calib):,}  test: {len(test):,}")

    X_train, y_train = train[FEATURE_COLS], train["label"]
    X_calib, y_calib = calib[FEATURE_COLS], calib["label"]
    X_test, y_test = test[FEATURE_COLS], test["label"]

    print("Training XGBoost classifier ...")
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    print("Evaluating on held-out test set ...")
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    np.savez(
        "calibration_test_data.npz",
        y_true=y_test.to_numpy(),
        y_prob=test_proba
    )

    print("Saved calibration test data to calibration_test_data.npz")
    print(f"  Test accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"  Test AUC:      {roc_auc_score(y_test, test_proba):.4f}")
    print(f"  Label balance (test): {y_test.mean():.4f} positive")

    model.save_model(args.out)
    print(f"Saved model to {args.out}")

    # Save the calibration split (with regime labels) -- the VEC generator
    # needs this exact split to compute conformal intervals and branch-firing
    # rates without any leakage from train or test.
    if args.no_calib_save:
        print("Skipped saving calibration split (--no_calib_save)")
    else:
        calib.to_parquet(args.calib_out)
        print(f"Saved calibration split ({len(calib):,} rows) to {args.calib_out}")


if __name__ == "__main__":
    main()
