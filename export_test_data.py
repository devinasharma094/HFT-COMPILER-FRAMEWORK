"""Export the held-out TEST slice as raw float32 [N,3], mirroring train_model.py exactly:
rows are used in file order (no sorting), the last HORIZON rows are dropped (no label), then the final 20% is the test split.
Usage: python export_test_data.py --features features.parquet --out test_rows.bin
"""
import argparse
import numpy as np
import pandas as pd

FEATURES = ["realized_vol", "trade_intensity", "avg_trade_size"]
HORIZON = 30

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="features.parquet")
ap.add_argument("--out", default="test_rows.bin")
ap.add_argument("--split", choices=["test", "calib"], default="test",
                help="calib = calibration slice (use it for PGO profiling, never the test set)")
a = ap.parse_args()

df = pd.read_parquet(a.features).iloc[:-HORIZON]
n = len(df)
test = df.iloc[int(n * 0.8):] if a.split == "test" else df.iloc[int(n * 0.6):int(n * 0.8)]
X = test[FEATURES].to_numpy(dtype=np.float32)
X = X[np.isfinite(X).all(axis=1)]
X.tofile(a.out)
print(f"wrote {len(X):,} rows x {X.shape[1]} features -> {a.out}")
